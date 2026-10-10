// Copyright (c) Nex-AGI. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Full-attention output projection and TP reduction selected for the nex_n2_5_pro TP=8 backend.
//
// Two entry points serve the two decode graphs:
//   - NexN25ProOProjReduceLaunch (non-CP): ONE fused kernel computes the gated o_proj
//     partial AND the cross-rank sum. Protocol: per-CTA epochs (it_ctr, monotonic,
//     no host resets), multimem.st stripe replication into a symmetric
//     [8][2][kDim] fp32 buffer, one NVLS arrival red per CTA onto its own counter
//     slot, then a local fixed-tree combine -- deterministic and rank-identical
//     fp32 sums. Producer geometry: 512 threads (16 warps), one warp owns BOTH
//     halves of its stripe row pair, and each warp issues all 16 weight words +
//     16 scales for its two rows BEFORE the activation staging so the loads'
//     DRAM latency drains under the staging compute. The optional wait-shadow
//     prefetch L2-warms the next consumer's weights (the MoE gate GEMV) while
//     the publish drains and the peer wait spins.
//   - NexN25ProOProjWholeReduce(Resid)Launch (CP): the cross-rank reduce alone. The CP
//     tail kernel produces the o_proj partial into the parity slot of the shared
//     partial buffer and derives the generation from the arrival counter; this
//     kernel signals arrival and folds ranks 0 through 7 in fixed FP32 order.
//     The two graphs use DISJOINT
//     arrival-counter slots of one symmetric buffer, so interleaved non-CP/CP
//     replays never see each other's signals.
#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY bs1
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "../common/oproj_common.cuh"

namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY {
// Fused gated o_proj + push reduce (non-CP graph).
template <bool kWaitPf>
__global__ __launch_bounds__(512, 1) void OProjFusedPushKernel(
    const __nv_bfloat16* __restrict__ attn, const __nv_bfloat16* __restrict__ gate,
    const __nv_fp8_e4m3* __restrict__ W, const float* __restrict__ w_scales,
    unsigned long long rep_mc,            // replicated push buffer, multicast VA
    const float* __restrict__ rep_local,  // this rank's local copy of it
    unsigned long long bar_mc_base,       // NVLS arrival counters, multicast VA (one slot/CTA)
    int* __restrict__ bar_local_base,     // this rank's local counters
    float* __restrict__ out, int* __restrict__ status,
    unsigned* __restrict__ it_ctr,  // per-CTA epochs (monotonic, rank-private)
    int rank, int num_ranks,
    __nv_bfloat16* __restrict__ x_resid,  // fused residual: if non-null, x_resid += reduced (bf16, in place)
    const char* __restrict__ pf_base,  // optional: L2-warm this range under the peer wait
    long pf_bytes) {
    constexpr int kNumThreads = 512;
    constexpr int kWarpsPerCTA = kNumThreads / 32;
    constexpr int kRowsPerStripeHalf = 16;
    static_assert(kWarpsPerCTA == 16, "fused-push variant expects 16 warps/CTA");

    __shared__ __align__(16) float x_eff[kLocal];
    __shared__ __align__(16) float block_partials[32];
    __shared__ uint32_t cta_epoch;

    const int local_warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * kRowsPerStripeHalf;

    if (threadIdx.x == 0) {
        cta_epoch = it_ctr[blockIdx.x];
    }

    // Load order matters here: the staging inputs feed a CTA-wide barrier, so
    // their few small loads issue FIRST; the 32-word weight burst follows and
    // its DRAM latency drains under the staging compute. Issuing the burst
    // first parks the CTA-critical activation loads behind 32 KB/warp of
    // weight traffic in the LSU.
    static_assert(kLocal == 2 * kNumThreads, "staging pre-load expects 2 elements/thread");
    const int k0 = threadIdx.x;
    const int k1 = threadIdx.x + kNumThreads;
    const __nv_bfloat16 act_a0 = attn[k0];
    const __nv_bfloat16 act_a1 = attn[k1];
    const __nv_bfloat16 act_g0 = gate[k0];
    const __nv_bfloat16 act_g1 = gate[k1];

    // Eager phase: weights and scales depend only on the row indices, so their
    // loads issue here and complete while the staging below computes x_eff.
    const int row0 = row_base + local_warp;
    const int row1 = row0 + kDim / 2;
    const __nv_fp8_e4m3* row_w0 = W + static_cast<unsigned long long>(row0) * kLocal;
    const __nv_fp8_e4m3* row_w1 = W + static_cast<unsigned long long>(row1) * kLocal;
    const int row_scale0 = row0 >> 7;
    const int row_scale1 = row1 >> 7;
    uint32_t wreg0[kScaleCols];
    uint32_t wreg1[kScaleCols];
    float sreg0[kScaleCols];
    float sreg1[kScaleCols];
#pragma unroll
    for (int tile = 0; tile < kScaleCols; ++tile) {
        const int k = tile * 128 + lane * 4;
        wreg0[tile] = *reinterpret_cast<const uint32_t*>(row_w0 + k);
        wreg1[tile] = *reinterpret_cast<const uint32_t*>(row_w1 + k);
        sreg0[tile] = w_scales[row_scale0 * kScaleCols + tile];
        sreg1[tile] = w_scales[row_scale1 * kScaleCols + tile];
    }

    // Staging from the pre-loaded registers (fused gate: attn * sigmoid(gate)).
    {
        float v0 = __bfloat162float(act_a0);
        float v1 = __bfloat162float(act_a1);
        v0 *= sigmoid_f32(__bfloat162float(act_g0));
        v1 *= sigmoid_f32(__bfloat162float(act_g1));
        x_eff[k0] = v0;
        x_eff[k1] = v1;
    }
    __syncthreads();  // publishes x_eff and cta_epoch

    const uint32_t epoch = cta_epoch;
    const int gen = static_cast<int>(epoch + 1u);
    const long off = static_cast<long>(epoch & 1u) * kDim;

    float lane_acc0 = 0.0f;
    float lane_acc1 = 0.0f;
#pragma unroll
    for (int tile = 0; tile < kScaleCols; ++tile) {
        const int k = tile * 128 + lane * 4;
        const float4 x4 = *reinterpret_cast<const float4*>(x_eff + k);
        lane_acc0 += sreg0[tile] * dot_x4_wword(x4, wreg0[tile]);
        lane_acc1 += sreg1[tile] * dot_x4_wword(x4, wreg1[tile]);
    }

    const float acc0 = warp_sum(lane_acc0);
    const float acc1 = warp_sum(lane_acc1);
    if (lane == 0) {
        block_partials[local_warp] = acc0;                 // first-half slot (rows row_base..+15)
        block_partials[kWarpsPerCTA + local_warp] = acc1;  // second-half slot (+ kDim/2)
    }
    __syncthreads();

    // Wait-shadow prefetch: L2-warm the downstream consumer's weights (the MoE
    // gate GEMV) while this CTA's publish drains and the peer wait spins. The
    // o_proj CTAs have uniform 32-row work and finish together, so the wait
    // window's DRAM really is idle. Fire-and-forget; no functional effect.
    if constexpr (kWaitPf) {
        if (pf_base != nullptr) {
            const long lines = pf_bytes >> 7;
            for (long l = blockIdx.x * static_cast<long>(kNumThreads) + threadIdx.x; l < lines;
                 l += 128L * kNumThreads) {
                asm volatile("prefetch.global.L2 [%0];" ::"l"(pf_base + (l << 7)));
            }
        }
    }

    // Thread 0 performs the multicast stripe replication and the release signal,
    // so the release orders exactly the stores this CTA is publishing.
    if (threadIdx.x == 0) {
        const unsigned long long dst =
            rep_mc + (static_cast<unsigned long long>(rank) * 2ull * kDim +
                      static_cast<unsigned long long>(off + row_base)) *
                         sizeof(float);
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
            multimem_st_v4_f32(dst + static_cast<unsigned long long>(i) * sizeof(float),
                               *reinterpret_cast<float4*>(block_partials + i));
            multimem_st_v4_f32(dst + static_cast<unsigned long long>(i + kDim / 2) * sizeof(float),
                               *reinterpret_cast<float4*>(block_partials + i + 16));
        }
        multimem_red_add_rel_sys(bar_mc_base + static_cast<unsigned long long>(blockIdx.x) * 4ull,
                                 1);
        // Unsigned multiply: wraps (defined) at the 2^31 generation horizon; the
        // subtraction-based spin comparison below is already wrap-safe.
        const int target =
            static_cast<int>(static_cast<unsigned>(gen) * static_cast<unsigned>(num_ranks));
        int* local_ctr = bar_local_base + blockIdx.x;
        int spins = 0;
        while ((int)(ld_acq_sys_i32(local_ctr) - target) < 0) {
            if (++spins > kMaxSpins) {
                if (status) atomicCAS(status, 0, 1);
                break;
            }
            __nanosleep(64);
        }
    }
    __syncthreads();

    // Combine: each warp reduces BOTH its rows over the 8 ranks from LOCAL
    // memory in ONE branch -- lanes 0..7 own row0, lanes 16..23 own row1, both
    // groups issuing together (an if/else-if split serializes the two groups'
    // latency-bound loads). This rank's own values come straight from shared
    // memory -- bit-identical to its multicast-stored copy, and no location
    // written through the multicast alias is ever read back through unicast.
    // The fixed shuffle tree makes the sums deterministic and rank-identical.
    {
        const int r = lane & 15;
        if (r < 8) {
            const bool hi = lane >= 16;
            const int row = hi ? row1 : row0;
            float v = 0.0f;
            if (r < num_ranks) {
                v = (r == rank) ? block_partials[(hi ? kWarpsPerCTA : 0) + local_warp]
                                : rep_local[static_cast<long>(r) * 2 * kDim + off + row];
            }
#pragma unroll
            for (int delta = 4; delta > 0; delta >>= 1) {
                v += __shfl_down_sync(0x00ff00ffu, v, delta);
            }
            if (r == 0) {
                if (x_resid) {
                    x_resid[row] = __float2bfloat16(__bfloat162float(x_resid[row]) + v);
                } else {
                    out[row] = v;
                }
            }
        }
    }

    // No final barrier needed: no thread in this launch reads this location, and
    // the next same-stream kernel cannot start until all CTAs have completed.
    if (threadIdx.x == 0) {
        it_ctr[blockIdx.x] = epoch + 1u;
    }
}

// Cross-rank reduce alone (CP graph): the CP tail kernel wrote this rank's
// o_proj partial into the parity slot of the shared partial buffer and derived
// the generation from THIS kernel's arrival counter, so signal-at-start + wait
// + 8-rank fold completes the o_proj.
#ifndef NEXRT_MTP_RUNTIME_ONLY
__global__ __launch_bounds__(256, 1) void WholeVectorReduceKernel(
    const int64_t* __restrict__ peer_partial2, unsigned long long bar_mc,
    int* __restrict__ bar_local, float* __restrict__ out, int* __restrict__ status,
    const int* __restrict__ cur_gen, int num_ranks,
    __nv_bfloat16* __restrict__ x_resid) {  // fused residual: if non-null, x_resid += reduced (bf16, in place)
    const int gen = cur_gen[0];
    const long off = static_cast<long>((gen - 1) & 1) * kDim;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        multimem_red_add_rel_sys(bar_mc, 1);
    }

    if (threadIdx.x == 0) {
        const int target =
            static_cast<int>(static_cast<unsigned>(gen) * static_cast<unsigned>(num_ranks));
        int spins = 0;
        while ((int)(ld_acq_sys_i32(bar_local) - target) < 0) {
            if (++spins > kMaxSpins) {
                if (status) atomicCAS(status, 0, 1);
                break;
            }
            __nanosleep(64);
        }
    }
    __syncthreads();

    for (int vec = blockIdx.x * blockDim.x + threadIdx.x; vec < kDim / 4;
         vec += gridDim.x * blockDim.x) {
        const int elem = vec * 4;
        // The CP tail writes through unicast pointers. After all producers
        // arrive, every rank reads the same slots and adds in the same order.
        // Floating multimem reduction does not specify that arithmetic order.
        const float* first = reinterpret_cast<const float*>(peer_partial2[0]) + off;
        float4 sum = *reinterpret_cast<const float4*>(first + elem);
#pragma unroll
        for (int r = 1; r < kMaxRanks; ++r) {
            if (r < num_ranks) {
                const float* peer = reinterpret_cast<const float*>(peer_partial2[r]) + off;
                const float4 p = *reinterpret_cast<const float4*>(peer + elem);
                sum = add4(sum, p);
            }
        }

        if (x_resid) {  // fused residual add: x_bf16 = bf16(x_bf16 + reduced)
            __nv_bfloat162* d = reinterpret_cast<__nv_bfloat162*>(x_resid + elem);
            const float2 a = __bfloat1622float2(d[0]), b = __bfloat1622float2(d[1]);
            d[0] = __floats2bfloat162_rn(a.x + sum.x, a.y + sum.y);
            d[1] = __floats2bfloat162_rn(b.x + sum.z, b.y + sum.w);
        } else {
            *reinterpret_cast<float4*>(out + elem) = sum;
        }
    }
}
#endif

template <bool kWaitPf>
int launch_oproj_fused_push(const void* attn, const void* gate, const void* W,
                            const float* w_scales, unsigned long long rep_mc,
                            const float* rep_local, unsigned long long bar_mc, int* bar_local,
                            float* out, int* status, unsigned* it_ctr, int rank, int num_ranks,
                            __nv_bfloat16* x_resid, const char* pf_base, long pf_bytes,
                            void* stream) {
    OProjFusedPushKernel<kWaitPf><<<128, 512, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(attn), reinterpret_cast<const __nv_bfloat16*>(gate),
        reinterpret_cast<const __nv_fp8_e4m3*>(W), w_scales, rep_mc, rep_local, bar_mc, bar_local,
        out, status, it_ctr, rank, num_ranks, x_resid, pf_base, pf_bytes);
    return static_cast<int>(cudaGetLastError());
}

#ifndef NEXRT_MTP_RUNTIME_ONLY
int launch_whole_vector_reduce(const int64_t* peer_partial2, unsigned long long bar_mc,
                               int* bar_local, float* out, int* status, const int* cur_gen,
                               int num_ranks, __nv_bfloat16* x_resid, void* stream) {
    WholeVectorReduceKernel<<<4, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        peer_partial2, bar_mc, bar_local, out, status, cur_gen, num_ranks, x_resid);
    return static_cast<int>(cudaGetLastError());
}
#endif

}  // namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
using namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY;
// Fused gated o_proj + TP reduce in one launch (non-CP graph). x_resid (nullable)
// folds the residual add into the combine epilogue; pf_base/pf_bytes (nullable)
// L2-warm the downstream MoE gate weights during the peer wait.
extern "C" int NexN25ProOProjReduceLaunch(const void* attn, const void* gate, const void* W,
                                          const void* w_scales, unsigned long long rep_mc,
                                          const void* rep_local, unsigned long long bar_mc,
                                          void* bar_local, void* out, void* status, void* it_ctr,
                                          int rank, int num_ranks, void* x_resid,
                                          const void* pf_base, long long pf_bytes, void* stream) {
    if (num_ranks <= 0 || num_ranks > kMaxRanks || rep_mc == 0 || bar_mc == 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const float* sc = reinterpret_cast<const float*>(w_scales);
    const float* rl = reinterpret_cast<const float*>(rep_local);
    int* bl = reinterpret_cast<int*>(bar_local);
    float* o = reinterpret_cast<float*>(out);
    int* st = reinterpret_cast<int*>(status);
    unsigned* it = reinterpret_cast<unsigned*>(it_ctr);
    __nv_bfloat16* xr = reinterpret_cast<__nv_bfloat16*>(x_resid);
    const char* pf = reinterpret_cast<const char*>(pf_base);
    if (pf != nullptr && pf_bytes > 0) {
        return launch_oproj_fused_push<true>(attn, gate, W, sc, rep_mc, rl, bar_mc, bl, o, st, it,
                                             rank, num_ranks, xr, pf, static_cast<long>(pf_bytes),
                                             stream);
    }
    return launch_oproj_fused_push<false>(attn, gate, W, sc, rep_mc, rl, bar_mc, bl, o, st, it,
                                          rank, num_ranks, xr, nullptr, 0, stream);
}

#ifndef NEXRT_MTP_RUNTIME_ONLY
extern "C" int NexN25ProOProjWholeReduceLaunch(const void* peer_partial2, unsigned long long bar_mc,
                                               void* bar_local, void* out, void* status,
                                               void* cur_gen, int num_ranks, void* stream) {
    if (num_ranks <= 0 || num_ranks > kMaxRanks) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (bar_mc == 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const auto* pp = reinterpret_cast<const int64_t*>(peer_partial2);
    auto* bl = reinterpret_cast<int*>(bar_local);
    auto* output = reinterpret_cast<float*>(out);
    auto* st = reinterpret_cast<int*>(status);
    const auto* gen = reinterpret_cast<const int*>(cur_gen);
    return launch_whole_vector_reduce(pp, bar_mc, bl, output, st, gen, num_ranks, nullptr, stream);
}

extern "C" int NexN25ProOProjWholeReduceResidLaunch(const void* peer_partial2,
                                                    unsigned long long bar_mc, void* bar_local,
                                                    void* out, void* status, void* cur_gen,
                                                    int num_ranks, void* x_resid, void* stream) {
    if (num_ranks <= 0 || num_ranks > kMaxRanks) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (bar_mc == 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const auto* pp = reinterpret_cast<const int64_t*>(peer_partial2);
    auto* bl = reinterpret_cast<int*>(bar_local);
    auto* output = reinterpret_cast<float*>(out);
    auto* st = reinterpret_cast<int*>(status);
    const auto* gen = reinterpret_cast<const int*>(cur_gen);
    auto* resid = reinterpret_cast<__nv_bfloat16*>(x_resid);
    return launch_whole_vector_reduce(pp, bar_mc, bl, output, st, gen, num_ranks, resid, stream);
}

extern "C" const char* NexN25ProOProjReduceErrorString(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
#endif
