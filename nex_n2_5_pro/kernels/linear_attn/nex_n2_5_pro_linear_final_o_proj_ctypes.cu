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

// nex_n2_5_pro linear-attn final RMSNormGated + o_proj + TP reduce in ONE launch.
//
// Fixed one TP=8 rank decode shape:
//   recurrent out[8,128] bf16
//   z[8,128] bf16
//   norm_w[128] bf16
//   o_proj shard W[4096,1024] fp8 e4m3 + scales[32,8] fp32
//
// The kernel does not materialize core = rms_norm_gated(out,z,norm_w); each CTA
// stages the effective local input in shared memory and reuses it for its rows.
// Protocol (same as the full-attention twin): per-CTA epochs (it_ctr, monotonic,
// no host resets), multimem.st stripe replication into a symmetric [8][2][kDim]
// fp32 buffer, one NVLS arrival red per CTA onto its own counter slot, then a
// local fixed-tree combine -- deterministic and rank-identical fp32 sums.
// Producer geometry: 512 threads (16 warps), one warp owns BOTH halves of its
// stripe row pair, so __launch_bounds__(512,1) lifts the ptxas budget to 128
// regs/thread; each warp issues all 16 weight words + 16 scales for its two
// rows BEFORE the staging barrier, so the loads' DRAM latency drains under the
// rms/gate staging chain (the loads feeding the CTA-wide barrier issue first --
// bar.sync orders threads, not loads). The optional wait-shadow prefetch
// L2-warms the next consumer's weights (the MoE gate GEMV) during the peer wait.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY bs1
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "../common/oproj_common.cuh"

namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY {
template <bool kWaitPf>
__global__ __launch_bounds__(512, 1) void LinearFinalOProjFusedPushKernel(
    const __nv_bfloat16* __restrict__ recurrent_out, const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ norm_w, const __nv_fp8_e4m3* __restrict__ W,
    const float* __restrict__ w_scales,
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
    __shared__ __align__(16) float rms[kHeads];
    __shared__ __align__(16) float block_partials[32];
    __shared__ uint32_t cta_epoch;

    const int local_warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * kRowsPerStripeHalf;

    // Rank-private epoch: stream task completion makes the prior launch's plain
    // store visible before this launch starts, so no atomics are needed here.
    if (threadIdx.x == 0) {
        cta_epoch = it_ctr[blockIdx.x];
    }

    // Load order matters here: the staging inputs feed a CTA-wide barrier, so
    // their few small loads issue FIRST; the 32-word weight burst follows and
    // its DRAM latency drains under the rms/gate staging compute.
    static_assert(kLocal == 2 * kNumThreads, "staging pre-load expects 2 elements/thread");
    const int k0 = threadIdx.x;
    const int k1 = threadIdx.x + kNumThreads;
    const __nv_bfloat16 act_out0 = recurrent_out[k0];
    const __nv_bfloat16 act_out1 = recurrent_out[k1];
    const __nv_bfloat16 act_z0 = z[k0];
    const __nv_bfloat16 act_z1 = z[k1];
    const __nv_bfloat16 act_nw0 = norm_w[k0 & 127];
    const __nv_bfloat16 act_nw1 = norm_w[k1 & 127];
    float head_sumsq = 0.0f;
    if (local_warp < kHeads) {
        const __nv_bfloat16* out_h = recurrent_out + local_warp * kHeadDim;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const float v = __bfloat162float(out_h[lane + t * 32]);
            head_sumsq += v * v;
        }
    }

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

    // Staging from the pre-loaded registers: per-head rms reduction, then the
    // per-element bf16 round-trips of (norm_w * normed) * silu(z).
    if (local_warp < kHeads) {
        const float total = warp_sum(head_sumsq);
        if (lane == 0) {
            rms[local_warp] = rsqrtf(total * (1.0f / static_cast<float>(kHeadDim)) + kRmsEps);
        }
    }
    __syncthreads();  // publishes rms and cta_epoch

    {
        const float out_f0 = __bfloat162float(act_out0);
        const float normed_bf0 = __bfloat162float(__float2bfloat16(out_f0 * rms[k0 >> 7]));
        const float weighted_bf0 =
            __bfloat162float(__float2bfloat16(__bfloat162float(act_nw0) * normed_bf0));
        const float gated0 = weighted_bf0 * silu_f32(__bfloat162float(act_z0));
        x_eff[k0] = __bfloat162float(__float2bfloat16(gated0));

        const float out_f1 = __bfloat162float(act_out1);
        const float normed_bf1 = __bfloat162float(__float2bfloat16(out_f1 * rms[k1 >> 7]));
        const float weighted_bf1 =
            __bfloat162float(__float2bfloat16(__bfloat162float(act_nw1) * normed_bf1));
        const float gated1 = weighted_bf1 * silu_f32(__bfloat162float(act_z1));
        x_eff[k1] = __bfloat162float(__float2bfloat16(gated1));
    }
    __syncthreads();  // publishes x_eff

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
    // o_proj GEMV leaves DRAM headroom and its CTAs finish together, so the
    // wait window really absorbs the fills. Fire-and-forget; no functional effect.
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

template <bool kWaitPf>
int launch_fused_push(const void* recurrent_out, const void* z, const void* norm_w, const void* W,
                      const float* w_scales, unsigned long long rep_mc, const float* rep_local,
                      unsigned long long bar_mc, int* bar_local, float* out, int* status,
                      unsigned* it_ctr, int rank, int num_ranks, __nv_bfloat16* x_resid,
                      const char* pf_base, long pf_bytes, void* stream) {
    LinearFinalOProjFusedPushKernel<kWaitPf>
        <<<128, 512, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<const __nv_bfloat16*>(recurrent_out),
            reinterpret_cast<const __nv_bfloat16*>(z),
            reinterpret_cast<const __nv_bfloat16*>(norm_w),
            reinterpret_cast<const __nv_fp8_e4m3*>(W), w_scales, rep_mc, rep_local, bar_mc,
            bar_local, out, status, it_ctr, rank, num_ranks, x_resid, pf_base, pf_bytes);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
using namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY;
// Fused RMSNormGated + o_proj + TP reduce in one launch. x_resid (nullable) folds
// the residual add into the combine epilogue; pf_base/pf_bytes (nullable) L2-warm
// the downstream MoE gate weights during the peer wait.
extern "C" int NexN25ProLinearFinalOProjLaunch(
    const void* recurrent_out, const void* z, const void* norm_w, const void* W,
    const void* w_scales, unsigned long long rep_mc, const void* rep_local,
    unsigned long long bar_mc, void* bar_local, void* out, void* status, void* it_ctr, int rank,
    int num_ranks, void* x_resid, const void* pf_base, long long pf_bytes, void* stream) {
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
        return launch_fused_push<true>(recurrent_out, z, norm_w, W, sc, rep_mc, rl, bar_mc, bl, o,
                                       st, it, rank, num_ranks, xr, pf, static_cast<long>(pf_bytes),
                                       stream);
    }
    return launch_fused_push<false>(recurrent_out, z, norm_w, W, sc, rep_mc, rl, bar_mc, bl, o, st,
                                    it, rank, num_ranks, xr, nullptr, 0, stream);
}

extern "C" const char* NexN25ProLinearFinalOProjErrorString(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
