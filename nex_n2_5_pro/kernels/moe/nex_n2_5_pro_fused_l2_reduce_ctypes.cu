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

// Fused MoE L2 GEMV and TP reduction selected for the nex_n2_5_pro TP=8 backend.
//
// One kernel per MoE block does the L2 down-projection AND the cross-rank sum:
//   1. 128 CTAs x 16 warps: each warp computes one stripe row pair (row, row+2048)
//      of the routed + shared down-projection (fp8 W2, fp32 accumulation).
//   2. Push publication: warp 0 packs each row pair to bf16 and multicast-stores
//      one 4-byte wire slot per rank into every GPU's replicated buffer
//      (u32[parity][kDim/2 row pairs][8 ranks]); a row pair's 8 rank values are
//      32 contiguous bytes, and the bf16 wire halves the bytes the release-red
//      must flush versus an fp32 wire.
//   3. Per-CTA NVLS arrival: one multicast red.release per CTA onto its own
//      counter slot; each CTA spins only on its LOCAL slot. Per-CTA epochs
//      (it_ctr, monotonic) replace any grid-wide generation broadcast, so the
//      kernel needs no grid barrier and no host resets (CUDA-graph replay safe).
//   4. Combine from LOCAL memory: 16 threads fetch one row pair's 32-byte slot
//      group each, PATCH their own rank's value from shared memory (the vector
//      load does read back the own multicast-written word through unicast, but
//      that word is discarded before use -- the alias-safety mechanism; no
//      alias-proxy fence needed), and sum ranks in fixed r0..r7 order --
//      byte-identical inputs on every rank make the reduced output
//      DETERMINISTIC and rank-identical.
//
// Numerics: partials cross the wire in bf16 (round-to-nearest) before the fp32
// sum. The optional wait-shadow prefetch L2-warms the next kernel's weights
// while the publish drains and the peer wait spins (DRAM is otherwise idle in
// that window); it is fire-and-forget and has no functional effect.
#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY bs1
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY {
constexpr int kDim = 4096;
constexpr int kTopK = 10;
constexpr int kRankInter = 128;
constexpr int kScaleBlocks = kDim / 128;
constexpr int kMaxRanks = 8;
constexpr int kMaxSpins = 1 << 30;

// ---- L2 GEMV helpers ----
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float4 load_h4(const float* __restrict__ h, int lane) {
    return *reinterpret_cast<const float4*>(h + lane * 4);
}

__device__ __forceinline__ float dot_h4_wfp8x4(float4 h4, const __nv_fp8_e4m3* __restrict__ w,
                                               int lane) {
    const __nv_fp8x4_e4m3 w4 = *reinterpret_cast<const __nv_fp8x4_e4m3*>(w + lane * 4);
    const float4 wf = float4(w4);
    return h4.x * wf.x + h4.y * wf.y + h4.z * wf.z + h4.w * wf.w;
}

// ---- cross-rank push infra ----
__device__ __forceinline__ int ld_acq_sys_i32(const int* ptr) {
    int ret;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr) : "memory");
    return ret;
}

// Multicast arrival counter: one instruction atomically adds `val` to EVERY rank's slot
// (in the NVSwitch). Lets each rank poll its OWN local counter instead of reading all peers'.
__device__ __forceinline__ void multimem_red_add_rel_sys(unsigned long long mc, int val) {
    asm volatile("multimem.red.release.sys.global.add.s32 [%0], %1;" ::"l"(mc), "r"(val)
                 : "memory");
}

// Broadcast-store one 4-byte word to every rank's copy of the replicated buffer via
// the NVSwitch (weak; the arrival red's release orders it for peer observers).
__device__ __forceinline__ void multimem_st_u32_as_f32(unsigned long long mc, unsigned v) {
    asm volatile("multimem.st.weak.global.f32 [%0], %1;" ::"l"(mc), "f"(__uint_as_float(v))
                 : "memory");
}

template <bool kWaitPf>
__global__ __launch_bounds__(512, 1) void FusedL2ReducePushKernel(
    const int32_t* __restrict__ topk_ids, const float* __restrict__ topk_weights,
    const float* __restrict__ h, const __nv_fp8_e4m3* __restrict__ W2,
    const float* __restrict__ w2_scales, const __nv_fp8_e4m3* __restrict__ shared_W2,
    const float* __restrict__ shared_w2_scales,
    unsigned long long rep_mc,               // replicated push buffer, multicast VA
    const unsigned* __restrict__ rep_local,  // this rank's local copy of it
    unsigned long long bar_mc_base,          // NVLS arrival counters, multicast VA (one slot/CTA)
    int* __restrict__ bar_local_base,        // this rank's local counters
    float* __restrict__ out, int* __restrict__ status,
    unsigned* __restrict__ it_ctr,  // per-CTA epochs (monotonic, rank-private)
    int rank, int num_ranks,
    __nv_bfloat16* __restrict__ x_resid,  // fused residual: if non-null, x_resid += reduced (bf16, in place)
    const char* __restrict__ pf_base,  // optional: L2-warm this range under the peer wait
    long pf_bytes) {
    constexpr int kNumCTAs = 128;
    constexpr int kNumThreads = 512;
    constexpr int kWarpsPerCTA = kNumThreads / 32;
    static_assert(kWarpsPerCTA == 16, "selected kernel expects 16 warps/CTA");

    __shared__ __align__(16) float block_partials[32];
    __shared__ uint32_t cta_epoch;

    const int local_warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * kWarpsPerCTA;

    // Rank-private epoch: stream task completion makes the prior launch's plain
    // store visible before this launch starts, so no atomics are needed here.
    if (threadIdx.x == 0) {
        cta_epoch = it_ctr[blockIdx.x];
    }

    // Slot-outer / half-inner: per-(lane,half) accumulation order is slot 0..9
    // then shared, and h4 is loaded once per slot instead of twice.
    {
        int rows[2];
        int row_scales[2];
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            rows[half] = row_base + local_warp + half * (kDim / 2);
            row_scales[half] = rows[half] >> 7;
        }
        float lane_acc[2] = {0.0f, 0.0f};
#pragma unroll
        for (int logical = 0; logical < kTopK; ++logical) {
            const int expert = topk_ids[logical];
            const float* row_h = h + logical * kRankInter;
            const float4 h4 = load_h4(row_h, lane);
            const float tw = topk_weights[logical];
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const __nv_fp8_e4m3* row_w =
                    W2 + (static_cast<unsigned long long>(expert) * kDim + rows[half]) * kRankInter;
                const float scale = w2_scales[expert * kScaleBlocks + row_scales[half]];
                lane_acc[half] += tw * scale * dot_h4_wfp8x4(h4, row_w, lane);
            }
        }
        const float* shared_h = h + kTopK * kRankInter;
        const float4 shared_h4 = load_h4(shared_h, lane);
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const __nv_fp8_e4m3* shared_row_w =
                shared_W2 + static_cast<unsigned long long>(rows[half]) * kRankInter;
            lane_acc[half] +=
                shared_w2_scales[row_scales[half]] * dot_h4_wfp8x4(shared_h4, shared_row_w, lane);
        }
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const float acc = warp_sum(lane_acc[half]);
            if (lane == 0) block_partials[half * kWarpsPerCTA + local_warp] = acc;
        }
    }
    __syncthreads();  // publishes block_partials and cta_epoch

    const uint32_t epoch = cta_epoch;
    const int gen = static_cast<int>(epoch + 1u);
    const int parity = static_cast<int>(epoch & 1u);

    if constexpr (kWaitPf) {
        if (pf_base != nullptr) {
            // Wait-shadow prefetch: L2-warm the NEXT kernel's weight range while
            // this CTA's publish drains and the peer wait runs. DRAM is otherwise
            // idle during the wait window (every rank is spinning), so the fills
            // ride free bandwidth. Fire-and-forget; no functional effect.
            const long lines = pf_bytes >> 7;
            for (long l = blockIdx.x * static_cast<long>(kNumThreads) + threadIdx.x; l < lines;
                 l += static_cast<long>(kNumCTAs) * kNumThreads) {
                asm volatile("prefetch.global.L2 [%0];" ::"l"(pf_base + (l << 7)));
            }
        }
    }

    // bf16 wire publication: warp 0's lanes 0..15 each pack TWO of this CTA's
    // rows to bf16 and multicast-store their rank's 4-byte slot. bar.warp.sync
    // makes thread 0's release cumulative over the lanes' stores.
    if (threadIdx.x < 32) {
        if (threadIdx.x < 16) {
            const int k2 = (threadIdx.x & 7) * 2;
            const int sm_base = (threadIdx.x < 8) ? k2 : 16 + k2;
            const int prow = row_base + ((threadIdx.x < 8) ? 0 : (kDim / 2)) + k2;
            const __nv_bfloat162 pk =
                __floats2bfloat162_rn(block_partials[sm_base], block_partials[sm_base + 1]);
            const unsigned long long slot =
                rep_mc + ((static_cast<unsigned long long>(parity) * (kDim / 2) +
                           static_cast<unsigned long long>(prow >> 1)) *
                              8ull +
                          static_cast<unsigned long long>(rank)) *
                             4ull;
            multimem_st_u32_as_f32(slot, *reinterpret_cast<const unsigned*>(&pk));
        }
        __syncwarp();
    }

    // Thread 0 signals arrival and spins on the LOCAL counter until all ranks'
    // CTA blockIdx.x arrived for this generation.
    if (threadIdx.x == 0) {
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

    // Combine: 16 threads each fetch ONE row pair's 32-byte slot group (8 ranks
    // x 2 rows bf16) branch-free, patch their own rank from shared memory (the
    // discarded unicast readback of the multicast-written slot stays alias-safe),
    // and sum in fixed r0..r7 order -- deterministic on every rank.
    if (threadIdx.x < 16) {
        const int k2 = (threadIdx.x & 7) * 2;
        const int sm_base = (threadIdx.x < 8) ? k2 : 16 + k2;
        const int prow = row_base + ((threadIdx.x < 8) ? 0 : (kDim / 2)) + k2;
        const unsigned* slot = rep_local + (static_cast<unsigned long long>(parity) * (kDim / 2) +
                                            static_cast<unsigned long long>(prow >> 1)) *
                                               8ull;
        const uint4 lo4 = *reinterpret_cast<const uint4*>(slot);
        const uint4 hi4 = *reinterpret_cast<const uint4*>(slot + 4);
        unsigned bits[8] = {lo4.x, lo4.y, lo4.z, lo4.w, hi4.x, hi4.y, hi4.z, hi4.w};
        {
            const __nv_bfloat162 own =
                __floats2bfloat162_rn(block_partials[sm_base], block_partials[sm_base + 1]);
            bits[rank] = *reinterpret_cast<const unsigned*>(&own);
        }
        float acc0 = 0.0f;
        float acc1 = 0.0f;
#pragma unroll
        for (int rr = 0; rr < 8; ++rr) {
            if (rr < num_ranks) {
                const float2 v =
                    __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bits[rr]));
                acc0 += v.x;
                acc1 += v.y;
            }
        }
        if (x_resid) {  // fused residual add: r_bf16 += reduced (in place)
            x_resid[prow] = __float2bfloat16(__bfloat162float(x_resid[prow]) + acc0);
            x_resid[prow + 1] = __float2bfloat16(__bfloat162float(x_resid[prow + 1]) + acc1);
        } else {
            out[prow] = acc0;
            out[prow + 1] = acc1;
        }
    }

    // No final barrier needed: no thread in this launch reads this location, and
    // the next same-stream kernel cannot start until all CTAs have completed.
    if (threadIdx.x == 0) {
        it_ctr[blockIdx.x] = epoch + 1u;
    }
}

template <bool kWaitPf>
int launch_fused_push(const int32_t* topk_ids, const float* topk_weights, const float* h,
                      const void* W2, const float* w2_scales, const void* shared_W2,
                      const float* shared_w2_scales, unsigned long long rep_mc,
                      const unsigned* rep_local, unsigned long long bar_mc, int* bar_local,
                      float* out, int* status, unsigned* it_ctr, int rank, int num_ranks,
                      void* stream, __nv_bfloat16* x_resid = nullptr, const char* pf_base = nullptr,
                      long pf_bytes = 0) {
    FusedL2ReducePushKernel<kWaitPf><<<128, 512, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        topk_ids, topk_weights, h, reinterpret_cast<const __nv_fp8_e4m3*>(W2), w2_scales,
        reinterpret_cast<const __nv_fp8_e4m3*>(shared_W2), shared_w2_scales, rep_mc, rep_local,
        bar_mc, bar_local, out, status, it_ctr, rank, num_ranks, x_resid, pf_base, pf_bytes);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
using namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY;
#ifndef NEXRT_MTP_RUNTIME_ONLY
extern "C" int NexN25ProFusedL2ReduceLaunch(const void* topk_ids, const void* topk_weights,
                                            const void* h, const void* W2, const void* w2_scales,
                                            const void* shared_W2, const void* shared_w2_scales,
                                            unsigned long long rep_mc, const void* rep_local,
                                            unsigned long long bar_mc, void* bar_local, void* out,
                                            void* status, void* it_ctr, int rank, int num_ranks,
                                            void* stream) {
    if (num_ranks <= 0 || num_ranks > kMaxRanks || rep_mc == 0 || bar_mc == 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    return launch_fused_push<false>(
        reinterpret_cast<const int32_t*>(topk_ids), reinterpret_cast<const float*>(topk_weights),
        reinterpret_cast<const float*>(h), W2, reinterpret_cast<const float*>(w2_scales), shared_W2,
        reinterpret_cast<const float*>(shared_w2_scales), rep_mc,
        reinterpret_cast<const unsigned*>(rep_local), bar_mc, reinterpret_cast<int*>(bar_local),
        reinterpret_cast<float*>(out), reinterpret_cast<int*>(status),
        reinterpret_cast<unsigned*>(it_ctr), rank, num_ranks, stream);
}
#endif

// Fused-residual form: writes x_resid[bf16] += reduced (in place) instead of out[fp32],
// folding the decoder-layer residual add into the reduce epilogue. pf_base/pf_bytes
// (nullable) L2-warm the next kernel's weights -- normally the NEXT layer's front GEMV
// weights, clamped by the caller to what the peer-wait window absorbs -- during the wait.
extern "C" int NexN25ProFusedL2ReduceResidLaunch(
    const void* topk_ids, const void* topk_weights, const void* h, const void* W2,
    const void* w2_scales, const void* shared_W2, const void* shared_w2_scales,
    unsigned long long rep_mc, const void* rep_local, unsigned long long bar_mc, void* bar_local,
    void* out, void* status, void* it_ctr, int rank, int num_ranks, void* x_resid,
    const void* pf_base, long long pf_bytes, void* stream) {
    if (num_ranks <= 0 || num_ranks > kMaxRanks || rep_mc == 0 || bar_mc == 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int32_t* ids = reinterpret_cast<const int32_t*>(topk_ids);
    const float* tw = reinterpret_cast<const float*>(topk_weights);
    const float* hh = reinterpret_cast<const float*>(h);
    const float* w2s = reinterpret_cast<const float*>(w2_scales);
    const float* sw2s = reinterpret_cast<const float*>(shared_w2_scales);
    const unsigned* rl = reinterpret_cast<const unsigned*>(rep_local);
    int* bl = reinterpret_cast<int*>(bar_local);
    float* o = reinterpret_cast<float*>(out);
    int* st = reinterpret_cast<int*>(status);
    unsigned* it = reinterpret_cast<unsigned*>(it_ctr);
    __nv_bfloat16* xr = reinterpret_cast<__nv_bfloat16*>(x_resid);
    const char* pf = reinterpret_cast<const char*>(pf_base);
    if (pf != nullptr && pf_bytes > 0) {
        return launch_fused_push<true>(ids, tw, hh, W2, w2s, shared_W2, sw2s, rep_mc, rl, bar_mc,
                                       bl, o, st, it, rank, num_ranks, stream, xr, pf,
                                       static_cast<long>(pf_bytes));
    }
    return launch_fused_push<false>(ids, tw, hh, W2, w2s, shared_W2, sw2s, rep_mc, rl, bar_mc, bl,
                                    o, st, it, rank, num_ranks, stream, xr);
}

#ifndef NEXRT_MTP_RUNTIME_ONLY
extern "C" const char* NexN25ProFusedL2ReduceErrorString(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
#endif
