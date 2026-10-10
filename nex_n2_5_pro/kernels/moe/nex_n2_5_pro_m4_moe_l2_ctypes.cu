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

// Production M=4 MoE down projection and TP=8 LL-inbox reduction.
//
// The default LL-inbox implementation uses folded input, push reduction, unroll-2
// slot pairs, per-CTA arrivals, v4 packers, and a fused reducer. The launcher keeps
// the NexN25ProM4MoeL2Launch ABI and hard-wires this instance
// (no multicast accumulator, no grid arrival counter, no fused-quant shadow).
//
// The folded-input L1 has already applied each route weight. One 128x1024 grid
// computes all 4096 output rows (one warp per row, both routed slots of a pair
// loaded before either dot), publishes naturally aligned 8-byte
// [BF16x2 payload | epoch] packets directly into every rank's symmetric inbox,
// and uses its first eight CTAs to poll and fold one packet per thread.
// Readiness is carried in each packet: this path has no data multicast
// reduction, system fence, or grid-wide arrival counter. The epoch comes from a
// per-CTA private counter (ll_counters), so nothing is shared across ranks.
//
// DECLARED NUMERICS CLASS: the cross-rank sum runs in bf16 on the wire and the
// expert-major record order sets the fp32 accumulation order per (token, row).
#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M4
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

// Profiler hooks are compiled out; the two trailing pointer arguments remain in
// the kernel signature.
#define SMP_DECL(recs, anch, nctas) \
    do {                            \
        (void)(recs);               \
        (void)(anch);               \
    } while (0)
#define SMP_REC(id) \
    do {            \
    } while (0)
#define SMP_FLUSH() \
    do {            \
    } while (0)

namespace nexrt::sm90::nex_n2_5_pro::M4 {
constexpr int kDim = 4096;
constexpr int kTopK = 10;
constexpr int kRankInter = 128;
constexpr int kScaleBlocks = kDim / 128;
constexpr int kMaxRanks = 8;
constexpr int kMaxSpins = 1 << 30;
constexpr int kM4 = 4;
constexpr int kPackedWords = 64;                  // ids [0:10], weight bits [10:20], shared [63]
constexpr int kHRows = (kTopK + 1) * kRankInter;  // 1408: routed slots 0..9 then shared

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

__device__ __forceinline__ int ld_acq_sys_i32(const int* ptr) {
    int ret;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr) : "memory");
    return ret;
}

__device__ __forceinline__ void multimem_red_add_rel_sys(unsigned long long mc, int val) {
    asm volatile("multimem.red.release.sys.global.add.s32 [%0], %1;" ::"l"(mc), "r"(val)
                 : "memory");
}

__device__ __forceinline__ void multimem_st_f32(unsigned long long mc, float v) {
    asm volatile("multimem.st.weak.global.f32 [%0], %1;" ::"l"(mc), "f"(v) : "memory");
}
}  // namespace nexrt::sm90::nex_n2_5_pro::M4

namespace nexrt::sm90::nex_n2_5_pro::M4 {
template <int kCTAs, bool kLocal>
__device__ __forceinline__ void ue_publish_tail_v2(
    const float* __restrict__ block_partials4, int rows_per_cta, int row_of_slot_base,
    int high_half_offset, unsigned long long rep_mc, unsigned* __restrict__ rep_local, int parity,
    int rank, const char* __restrict__ pf_base, long pf_bytes, unsigned int* __restrict__ grid_ctr,
    unsigned long long bar_mc) {
    const int packers = kM4 * (rows_per_cta / 2);
    if (threadIdx.x < packers) {
        const int pairs_per_cta = rows_per_cta / 2;
        const int j = threadIdx.x / pairs_per_cta;
        const int pair_in_cta = threadIdx.x - j * pairs_per_cta;
        int output_row;
        int shared_base;
        if (high_half_offset > 0) {
            const int half_pairs = pairs_per_cta / 2;
            const int row_in_half = (pair_in_cta & (half_pairs - 1)) * 2;
            const bool high_half = pair_in_cta >= half_pairs;
            shared_base = (high_half ? rows_per_cta / 2 : 0) + row_in_half;
            output_row = row_of_slot_base + (high_half ? high_half_offset : 0) + row_in_half;
        } else {
            shared_base = pair_in_cta * 2;
            output_row = row_of_slot_base + pair_in_cta * 2;
        }
        const float* bp = block_partials4 + j * rows_per_cta;
        const __half2 packed = __floats2half2_rn(bp[shared_base], bp[shared_base + 1]);
        const unsigned bits = *reinterpret_cast<const unsigned*>(&packed);
        const unsigned long long idx =
            ((((static_cast<unsigned long long>(parity) * kM4 + j) * (kDim / 2)) +
              static_cast<unsigned long long>(output_row >> 1)) *
                 kMaxRanks +
             static_cast<unsigned long long>(rank));
        if (kLocal) {
            rep_local[idx] = bits;
        } else {
            multimem_st_f32(rep_mc + idx * 4ull, __uint_as_float(bits));
        }
    }
    __syncthreads();                           // packer stores happen-before thread 0
    if (pf_base != nullptr && pf_bytes > 0) {  // read-only hints: no ordering required
        const long lines = pf_bytes >> 7;
        const long per_cta = (lines + kCTAs - 1) / kCTAs;
        const long line = static_cast<long>(blockIdx.x) * per_cta + threadIdx.x;
        if (threadIdx.x < per_cta && line < lines) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(pf_base + (line << 7)));
        }
    }
    if (threadIdx.x == 0) {
        __threadfence_system();  // release side: this CTA's stores
        const unsigned int done = atomicAdd(grid_ctr, 1u);
        if ((done % kCTAs) == kCTAs - 1 && !kLocal) {
            // Acquire side: legacy atomicAdd is RELAXED, so the
            // winning CTA must fence again after observing the count to synchronize
            // with the other CTAs' release patterns before the cumulative release
            // arrival (PTX release/acquire pattern rules).
            __threadfence_system();
            multimem_red_add_rel_sys(bar_mc, 1);
        }
    }
}

// HALFSH body with a template tail selector, and the 128x1024 geometry (32 warps, one
// output row per warp, whole-h4 staged once per CTA, one CTA per SM at <=64 regs --
// 128 CTA-level protocol participants and the corresponding packer count).
constexpr int kHElems = kM4 * kHRows;  // 5632 floats = 22528 B
constexpr int kHs2DynBytes1024 = (kHElems + kM4 * 32) * 4;

}  // namespace nexrt::sm90::nex_n2_5_pro::M4

// ===== Folded-input L2 and push reduction ================================================
// FOLD: consumes rawh pre-multiplied by its route weight (nex_n2_5_pro_m4_l1_foldw_ctypes.cu),
// dropping the per-output-row tw multiply. PUSH producers send their fp32 pair sums
// straight into a parity accumulator with multimem.red.add.f32; the reducer then waits,
// folds the LOCAL accumulator into the residual, and re-zeroes the slot for gen+2 reuse.
// The bar chain orders every rank's gen g reducer before every rank's gen g+2 producer,
// and the arrival release publishes the zeroing. DECLARED NUMERICS CLASS for PUSH:
// NVLS delivers each rank's adds independently, so fp32 arrival order -- and therefore
// the final bf16 fold -- may differ per rank at ulp level.
namespace nexrt::sm90::nex_n2_5_pro::M4 {
__device__ __forceinline__ void multimem_red_add_f32(unsigned long long mc, float v) {
    asm volatile("multimem.red.relaxed.sys.global.add.f32 [%0], %1;" ::"l"(mc), "f"(v) : "memory");
}

// Vector reduction: PTX ISA 9.7.9.14 allows .vec {.v2,.v4} with .f32 (type x vec
// = 128 bits max), sm_90+. One 16 B transaction replaces four scalar reds.
__device__ __forceinline__ void multimem_red_add_v4_f32(unsigned long long mc, float4 v) {
    asm volatile("multimem.red.relaxed.sys.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(mc),
                 "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
                 : "memory");
}

// Default LL peer-inbox primitives. The packet is EXACTLY 8 bytes -- 4 B payload
// (2 bf16) + 4 B epoch tag -- because a
// naturally-aligned 8 B access is atomic, so a reader can never observe a fresh
// tag paired with stale payload. A 16 B v4 store is NOT guaranteed atomic and can
// tear into two 8 B halves.
//
// .relaxed.sys gives visibility to the peer with NO ordering obligation, which
// is the whole point -- there is no __threadfence_system and no arrival here.
// Readiness rides inside the packet, so no payload-before-flag ordering is
// required and the PTX release-pattern requirement never arises.
__device__ __forceinline__ void ll_store_pkt(unsigned long long addr, unsigned payload,
                                             unsigned epoch) {
    const unsigned long long v = (static_cast<unsigned long long>(epoch) << 32) | payload;
    asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(addr), "l"(v) : "memory");
}

__device__ __forceinline__ unsigned long long ll_load_pkt(unsigned long long addr) {
    unsigned long long v;
    asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(addr) : "memory");
    return v;
}

// With kPerCta, every CTA emits its OWN release arrival, whose fence covers only its
// own pushes. There is no grid counter, CTA election, or second system fence. The bar
// advances kCTAs per rank per round. Generation floor-division stays safe because a
// reading CTA has not yet arrived (k <= 1023).
// Launch geometry is explicit template state. For the default <128,1024,1> shape, one
// warp per output row (kCTAs * kThreads/32 == kDim), v4 packers own 4-row
// groups, and the fused reducer's first (kM4*kDim/2)/kThreads CTAs fold one
// float2 pair per thread.
template <bool kFold, bool kPush, bool kLocal = false, bool kUnroll2 = false,
          bool kHalfSplit = false, bool kUnroll4 = false, bool kPerCta = false,
          bool kWarpPush = false, bool kV4Pack = false, bool kFuseRed = false, bool kHeadOv = false,
          int kCTAs = 128, int kThreads = 1024, int kBlocksPerSM = 1, bool kLLInbox = false>
__global__ __launch_bounds__(kThreads, kBlocksPerSM) void M4MoeL2ProducerH32XKernel(
    const int32_t* __restrict__ packed4, const float* __restrict__ h4,
    const __nv_fp8_e4m3* __restrict__ W2, const float* __restrict__ w2_scales,
    const __nv_fp8_e4m3* __restrict__ shared_W2, const float* __restrict__ shared_w2_scales,
    const int* __restrict__ bar_local, int* __restrict__ cur_gen,
    unsigned int* __restrict__ grid_ctr, unsigned long long rep_mc, unsigned long long acc_mc,
    unsigned long long bar_mc, int rank, int num_ranks, const char* __restrict__ pf_base,
    long pf_bytes,
    float* __restrict__ fuse_acc = nullptr,          // rank-LOCAL accumulator VA
    int* __restrict__ fuse_status = nullptr,         // sticky status
    __nv_bfloat16* __restrict__ fuse_x4 = nullptr,   // bf16 residual
    __nv_fp8_e4m3* __restrict__ fuse_x4q = nullptr,  // optional quantized shadow
    float* __restrict__ fuse_xs4 = nullptr,          // optional quantization scales
    // With kLLInbox, inbox_ptrs[s] is peer s's directed inbox base VA; ll_ctr
    // is a per-CTA private call counter used
    // to derive the epoch WITHOUT any barrier (every CTA of every rank runs
    // exactly once per L2 call, so the counters agree by construction).
    unsigned long long* __restrict__ inbox_ptrs = nullptr,
    unsigned int* __restrict__ ll_ctr = nullptr, unsigned int* __restrict__ smp_records = nullptr,
    unsigned long long* __restrict__ smp_anchor = nullptr) {
    static_assert(!kFuseRed || kPerCta, "fused reduce requires per-CTA arrivals");
    constexpr int kWarpsPerCTA = kThreads / 32;
    constexpr int kRedCtas = (kM4 * kDim / 2) / kThreads;
    static_assert(kCTAs * kWarpsPerCTA == kDim, "one warp per output row");
    static_assert(kWarpsPerCTA % 4 == 0, "v4 packers need 4-row groups");
    static_assert(kRedCtas * kThreads == kM4 * kDim / 2, "one float2 pair per reducer thread");
    static_assert(kRedCtas <= kCTAs, "reducers are a grid prefix");
    extern __shared__ __align__(16) float dynsm[];
    float* s_h = dynsm;
    float* block_partials4 = dynsm + kHElems;
    __shared__ int cta_gen;
    SMP_DECL(smp_records, smp_anchor, kCTAs);
    SMP_REC(SMP_EV_STAGE_B);
    const int lane = threadIdx.x & 31;
    const int local_warp = threadIdx.x >> 5;
    const int row = blockIdx.x * kWarpsPerCTA + local_warp;
    const int row_scale = row >> 7;

    if (threadIdx.x == 0 && kLLInbox) {
        // Barrier-free epoch: ll_ctr[blockIdx.x] is private to this CTA
        // (no other CTA or rank touches it), so a plain RMW is sufficient. Every
        // CTA runs exactly once per L2 call on every rank, so all 128 CTAs on
        // all 8 ranks derive the SAME epoch without reading any shared counter.
        const unsigned g = ll_ctr[blockIdx.x] + 1u;
        ll_ctr[blockIdx.x] = g;
        cta_gen = static_cast<int>(g);
        if (blockIdx.x == 0) {
            cur_gen[0] = static_cast<int>(g);
        }
    } else if (threadIdx.x == 0) {
        // Use unsigned arithmetic because the bar is a free-running counter; at
        // x128 units signed overflow would otherwise be reachable in long sessions.
        const unsigned completed = kLocal ? 0u : static_cast<unsigned>(ld_acq_sys_i32(bar_local));
        const unsigned per_round =
            kPerCta ? static_cast<unsigned>(num_ranks) * kCTAs : static_cast<unsigned>(num_ranks);
        const unsigned genu = (per_round == 8u      ? (completed >> 3)
                               : per_round == 1024u ? (completed >> 10)
                                                    : completed / per_round) +
                              1u;
        const int gen = static_cast<int>(genu);
        cta_gen = gen;
        if (blockIdx.x == 0) {
            cur_gen[0] = gen;
        }
    }
    // With kHeadOv, the first u2 W pair has no s_h dependency -- stage it BEFORE
    // the h-copy so the 22.5 KB stage hides behind these 8 loads' flight time.
    __nv_fp8x4_e4m3 hov_wq[2][kM4];
    float hov_tw[2][kM4];
    float hov_sc[2][kM4];
    if (kHeadOv) {
#pragma unroll
        for (int u = 0; u < 2; ++u) {
#pragma unroll
            for (int j = 0; j < kM4; ++j) {
                const int32_t* pk = packed4 + j * kPackedWords;
                const int expert = pk[u];
                hov_wq[u][j] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(
                    W2 + (static_cast<unsigned long long>(expert) * kDim + row) * kRankInter +
                    lane * 4);
                hov_tw[u][j] = kFold ? 1.0f : reinterpret_cast<const float*>(pk)[kTopK + u];
                hov_sc[u][j] = w2_scales[expert * kScaleBlocks + row_scale];
            }
        }
    }
    {
        const float4* src = reinterpret_cast<const float4*>(h4);
        float4* dst = reinterpret_cast<float4*>(s_h);
        for (int i = threadIdx.x; i < kHElems / 4; i += kThreads) {
            dst[i] = src[i];
        }
    }
    __syncthreads();
    SMP_REC(SMP_EV_STAGE_E);
    SMP_REC(SMP_EV_GEMV_B);  // h4 staged into SMEM; expert GEMV begins
    const int gen = cta_gen;
    const int parity = (gen - 1) & 1;

    float lane_acc[kM4] = {0.f, 0.f, 0.f, 0.f};
    if (kUnroll4) {
// Stage FOUR slots' 16 W fp8x4 words before any dot (16 loads in
// flight per warp); tw/scale are loaded lazily at dot time to stay under the
// 1024-thread 64-register cap. Slots 8..9 run as a trailing U2 pair.
#pragma unroll 1
        for (int lg = 0; lg < 8; lg += 4) {
            __nv_fp8x4_e4m3 wq[4][kM4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
#pragma unroll
                for (int j = 0; j < kM4; ++j) {
                    const int expert = packed4[j * kPackedWords + lg + u];
                    wq[u][j] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(
                        W2 + (static_cast<unsigned long long>(expert) * kDim + row) * kRankInter +
                        lane * 4);
                }
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
#pragma unroll
                for (int j = 0; j < kM4; ++j) {
                    const int32_t* pk = packed4 + j * kPackedWords;
                    const int expert = pk[lg + u];
                    const float tw =
                        kFold ? 1.0f : reinterpret_cast<const float*>(pk)[kTopK + lg + u];
                    const float scale = w2_scales[expert * kScaleBlocks + row_scale];
                    const float4 hv = load_h4(s_h + j * kHRows + (lg + u) * kRankInter, lane);
                    const float4 wf = float4(wq[u][j]);
                    const float d = hv.x * wf.x + hv.y * wf.y + hv.z * wf.z + hv.w * wf.w;
                    lane_acc[j] += kFold ? (scale * d) : (tw * scale * d);
                }
            }
        }
        {  // trailing slots 8,9 as one U2 pair
            const int lg = 8;
            __nv_fp8x4_e4m3 wq[2][kM4];
#pragma unroll
            for (int u = 0; u < 2; ++u) {
#pragma unroll
                for (int j = 0; j < kM4; ++j) {
                    const int expert = packed4[j * kPackedWords + lg + u];
                    wq[u][j] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(
                        W2 + (static_cast<unsigned long long>(expert) * kDim + row) * kRankInter +
                        lane * 4);
                }
            }
#pragma unroll
            for (int u = 0; u < 2; ++u) {
#pragma unroll
                for (int j = 0; j < kM4; ++j) {
                    const int32_t* pk = packed4 + j * kPackedWords;
                    const int expert = pk[lg + u];
                    const float tw =
                        kFold ? 1.0f : reinterpret_cast<const float*>(pk)[kTopK + lg + u];
                    const float scale = w2_scales[expert * kScaleBlocks + row_scale];
                    const float4 hv = load_h4(s_h + j * kHRows + (lg + u) * kRankInter, lane);
                    const float4 wf = float4(wq[u][j]);
                    const float d = hv.x * wf.x + hv.y * wf.y + hv.z * wf.z + hv.w * wf.w;
                    lane_acc[j] += kFold ? (scale * d) : (tw * scale * d);
                }
            }
        }
    } else if (kUnroll2 && kHalfSplit) {
        // Half-warp slot split: lanes 0..15 own slot lg, lanes 16..31 own slot
        // lg+1; each half loads its 128 B W row as 16 x uint2 (8 fp8/lane) -- HALF the
        // LDG instructions for the same bytes, two assignments per warp step. Each
        // half accumulates its slot's 8-wide partial dots into the shared per-token
        // accumulators; the final 32-lane warp_sum merges both halves exactly.
        const int hl = lane & 15;
        const int side = lane >> 4;
#pragma unroll 1
        for (int lg = 0; lg < kTopK; lg += 2) {
            const int myslot = lg + side;
// Per (slot, token): one 8-fp8 uint2 load per lane (16 lanes cover the
// row); the unrolled j-loop keeps 4 independent loads in flight per half.
#pragma unroll
            for (int j = 0; j < kM4; ++j) {
                const int32_t* pk = packed4 + j * kPackedWords;
                const int expert = pk[myslot];
                const __nv_fp8_e4m3* row_w =
                    W2 + (static_cast<unsigned long long>(expert) * kDim + row) * kRankInter;
                const uint2 wq2 = *reinterpret_cast<const uint2*>(row_w + hl * 8);
                const float twj = kFold ? 1.0f : reinterpret_cast<const float*>(pk)[kTopK + myslot];
                const float scs = w2_scales[expert * kScaleBlocks + row_scale];
                const float* hp = s_h + j * kHRows + myslot * kRankInter + hl * 8;
                const __nv_fp8x4_e4m3 wlo = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&wq2.x);
                const __nv_fp8x4_e4m3 whi = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&wq2.y);
                const float4 wl = float4(wlo);
                const float4 wh = float4(whi);
                const float4 h0 = *reinterpret_cast<const float4*>(hp);
                const float4 h1 = *reinterpret_cast<const float4*>(hp + 4);
                const float d = (h0.x * wl.x + h0.y * wl.y + h0.z * wl.z + h0.w * wl.w) +
                                (h1.x * wh.x + h1.y * wh.y + h1.z * wh.z + h1.w * wh.w);
                lane_acc[j] += kFold ? (scs * d) : (twj * scs * d);
            }
        }
    } else if (kUnroll2 && kHeadOv) {
// Dots for the pre-staged pair 0 (loads landed during the h-copy),
// then the verbatim u2 loop over pairs 1..4. Identical FP order to u2.
#pragma unroll
        for (int u = 0; u < 2; ++u) {
#pragma unroll
            for (int j = 0; j < kM4; ++j) {
                const float4 hv = load_h4(s_h + j * kHRows + u * kRankInter, lane);
                const float4 wf = float4(hov_wq[u][j]);
                const float d = hv.x * wf.x + hv.y * wf.y + hv.z * wf.z + hv.w * wf.w;
                lane_acc[j] += kFold ? (hov_sc[u][j] * d) : (hov_tw[u][j] * hov_sc[u][j] * d);
            }
        }
#pragma unroll 1
        for (int lg = 2; lg < kTopK; lg += 2) {
            __nv_fp8x4_e4m3 wq[2][kM4];
            float twv[2][kM4];
            float scv[2][kM4];
#pragma unroll
            for (int u = 0; u < 2; ++u) {
#pragma unroll
                for (int j = 0; j < kM4; ++j) {
                    const int32_t* pk = packed4 + j * kPackedWords;
                    const int expert = pk[lg + u];
                    const __nv_fp8_e4m3* row_w =
                        W2 + (static_cast<unsigned long long>(expert) * kDim + row) * kRankInter;
                    wq[u][j] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(row_w + lane * 4);
                    twv[u][j] = kFold ? 1.0f : reinterpret_cast<const float*>(pk)[kTopK + lg + u];
                    scv[u][j] = w2_scales[expert * kScaleBlocks + row_scale];
                }
            }
#pragma unroll
            for (int u = 0; u < 2; ++u) {
#pragma unroll
                for (int j = 0; j < kM4; ++j) {
                    const float4 hv = load_h4(s_h + j * kHRows + (lg + u) * kRankInter, lane);
                    const float4 wf = float4(wq[u][j]);
                    const float d = hv.x * wf.x + hv.y * wf.y + hv.z * wf.z + hv.w * wf.w;
                    lane_acc[j] += kFold ? (scv[u][j] * d) : (twv[u][j] * scv[u][j] * d);
                }
            }
        }
    } else if (kUnroll2) {
// Issue BOTH slots' 8 W fp8x4 loads before any dot so the
// scoreboard holds 8 independent global loads per warp (vs ~4), then compute.
#pragma unroll 1
        for (int lg = 0; lg < kTopK; lg += 2) {
            __nv_fp8x4_e4m3 wq[2][kM4];
            float twv[2][kM4];
            float scv[2][kM4];
#pragma unroll
            for (int u = 0; u < 2; ++u) {
#pragma unroll
                for (int j = 0; j < kM4; ++j) {
                    const int32_t* pk = packed4 + j * kPackedWords;
                    const int expert = pk[lg + u];
                    const __nv_fp8_e4m3* row_w =
                        W2 + (static_cast<unsigned long long>(expert) * kDim + row) * kRankInter;
                    wq[u][j] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(row_w + lane * 4);
                    twv[u][j] = kFold ? 1.0f : reinterpret_cast<const float*>(pk)[kTopK + lg + u];
                    scv[u][j] = w2_scales[expert * kScaleBlocks + row_scale];
                }
            }
#pragma unroll
            for (int u = 0; u < 2; ++u) {
#pragma unroll
                for (int j = 0; j < kM4; ++j) {
                    const float4 hv = load_h4(s_h + j * kHRows + (lg + u) * kRankInter, lane);
                    const float4 wf = float4(wq[u][j]);
                    const float d = hv.x * wf.x + hv.y * wf.y + hv.z * wf.z + hv.w * wf.w;
                    lane_acc[j] += kFold ? (scv[u][j] * d) : (twv[u][j] * scv[u][j] * d);
                }
            }
        }
    } else {
#pragma unroll 1
        for (int logical = 0; logical < kTopK; ++logical) {
#pragma unroll
            for (int j = 0; j < kM4; ++j) {
                const int32_t* pk = packed4 + j * kPackedWords;
                const int expert = pk[logical];
                const float tw = kFold ? 1.0f : reinterpret_cast<const float*>(pk)[kTopK + logical];
                const float4 hv = load_h4(s_h + j * kHRows + logical * kRankInter, lane);
                const __nv_fp8_e4m3* row_w =
                    W2 + (static_cast<unsigned long long>(expert) * kDim + row) * kRankInter;
                const float scale = w2_scales[expert * kScaleBlocks + row_scale];
                const float d = dot_h4_wfp8x4(hv, row_w, lane);
                lane_acc[j] += kFold ? (scale * d) : (tw * scale * d);
            }
        }
    }
    {
        const __nv_fp8_e4m3* shared_row_w =
            shared_W2 + static_cast<unsigned long long>(row) * kRankInter;
        const __nv_fp8x4_e4m3 w4 =
            *reinterpret_cast<const __nv_fp8x4_e4m3*>(shared_row_w + lane * 4);
        const float4 wf = float4(w4);
        const float ss = shared_w2_scales[row_scale];
#pragma unroll
        for (int j = 0; j < kM4; ++j) {
            const float4 sh = load_h4(s_h + j * kHRows + kTopK * kRankInter, lane);
            lane_acc[j] += ss * (sh.x * wf.x + sh.y * wf.y + sh.z * wf.z + sh.w * wf.w);
        }
    }
#pragma unroll
    for (int j = 0; j < kM4; ++j) {
        const float acc = warp_sum(lane_acc[j]);
        if (lane == 0) {
            if (kWarpPush && kPush) {
                // Push THIS warp's row immediately so fabric latency overlaps
                // the slower warps' compute; no partials roundtrip, no packer phase.
                const unsigned long long idxb =
                    (((static_cast<unsigned long long>(parity) * kM4 + j) * kDim) +
                     static_cast<unsigned long long>(row)) *
                    4ull;
                if (kLocal) {
                    atomicAdd(reinterpret_cast<float*>(rep_mc + idxb), acc);
                } else {
                    multimem_red_add_f32(acc_mc + idxb, acc);
                }
            } else {
                block_partials4[j * kWarpsPerCTA + local_warp] = acc;
            }
        }
    }
    __syncthreads();
    SMP_REC(SMP_EV_FOLD_E);
    if (!kPush) {
        ue_publish_tail_v2<kCTAs, false>(block_partials4, kWarpsPerCTA, blockIdx.x * kWarpsPerCTA,
                                         0, rep_mc, nullptr, parity, rank, pf_base, pf_bytes,
                                         grid_ctr, bar_mc);
        return;
    }
    if (!kWarpPush) {
        SMP_REC(SMP_EV_PUB_B);
        if constexpr (kLLInbox) {
            // This CTA owns 4 tokens x 32 rows = 128 fp32
            // partials = 64 LL packets (2 bf16 each). It pushes all 64 to EVERY
            // rank's inbox slot [parity][my_rank], including its own -- 512
            // directed 8 B stores per CTA, 65536 per rank.
            //
            // NO multimem, NO __threadfence_system, NO arrival. Correctness
            // rests only on 8 B store atomicity + the epoch tag.
            constexpr int kPktPerCta = kM4 * (kWarpsPerCTA / 2);  // 64
            constexpr int kPktPerSrc = kM4 * (kDim / 2);          // 8192
            if (threadIdx.x < static_cast<unsigned>(num_ranks * kPktPerCta)) {
                const int d = threadIdx.x / kPktPerCta;      // destination rank
                const int q = threadIdx.x - d * kPktPerCta;  // packet in CTA
                const int j = q / (kWarpsPerCTA / 2);        // token
                const int pr = q - j * (kWarpsPerCTA / 2);   // row-pair in CTA
                const float* bp = block_partials4 + j * kWarpsPerCTA + pr * 2;
                const __nv_bfloat162 pv = __float22bfloat162_rn(make_float2(bp[0], bp[1]));
                const int pkt = j * (kDim / 2) + blockIdx.x * (kWarpsPerCTA / 2) + pr;
                const unsigned long long off =
                    ((static_cast<unsigned long long>(parity) * num_ranks + rank) * kPktPerSrc +
                     pkt) *
                    8ull;
                ll_store_pkt(inbox_ptrs[d] + off, *reinterpret_cast<const unsigned*>(&pv),
                             static_cast<unsigned>(gen));
            }
        } else if (kV4Pack) {
            // 32 packers each own one (token, 4-row group) -> ONE v4.f32 red
            // (16 B, 16 B-aligned since row0 % 4 == 0): 4x fewer multicast
            // transactions in the same dense burst.
            if (threadIdx.x < kM4 * (kWarpsPerCTA / 4)) {
                const int groups = kWarpsPerCTA / 4;  // 8
                const int j = threadIdx.x / groups;
                const int g = threadIdx.x - j * groups;
                const int row0 = blockIdx.x * kWarpsPerCTA + g * 4;
                const float* bp = block_partials4 + j * kWarpsPerCTA + g * 4;
                const unsigned long long idxb =
                    (((static_cast<unsigned long long>(parity) * kM4 + j) * kDim) +
                     static_cast<unsigned long long>(row0)) *
                    4ull;
                if (kLocal) {
                    float* lb = reinterpret_cast<float*>(rep_mc + idxb);
                    atomicAdd(lb, bp[0]);
                    atomicAdd(lb + 1, bp[1]);
                    atomicAdd(lb + 2, bp[2]);
                    atomicAdd(lb + 3, bp[3]);
                } else {
                    multimem_red_add_v4_f32(acc_mc + idxb, make_float4(bp[0], bp[1], bp[2], bp[3]));
                }
            }
        } else {
            // PUSH publication: 64 packers each own one row pair -> two red.add.f32
            // into the [parity][4][4096] accumulator; owner-fence ordering (tail v2).
            if (threadIdx.x < kM4 * (kWarpsPerCTA / 2)) {
                const int pairs_per_cta = kWarpsPerCTA / 2;  // 16
                const int j = threadIdx.x / pairs_per_cta;
                const int pair_in_cta = threadIdx.x - j * pairs_per_cta;
                const int output_row = blockIdx.x * kWarpsPerCTA + pair_in_cta * 2;
                const float* bp = block_partials4 + j * kWarpsPerCTA;
                const unsigned long long idxb =
                    (((static_cast<unsigned long long>(parity) * kM4 + j) * kDim) +
                     static_cast<unsigned long long>(output_row)) *
                    4ull;
                if (kLocal) {
                    float* lb = reinterpret_cast<float*>(rep_mc + idxb);
                    atomicAdd(lb, bp[pair_in_cta * 2]);
                    atomicAdd(lb + 1, bp[pair_in_cta * 2 + 1]);
                } else {
                    multimem_red_add_f32(acc_mc + idxb, bp[pair_in_cta * 2]);
                    multimem_red_add_f32(acc_mc + idxb + 4ull, bp[pair_in_cta * 2 + 1]);
                }
            }
        }
        __syncthreads();
    }
    SMP_REC(SMP_EV_PUB_E);
    if (pf_base != nullptr && pf_bytes > 0) {
        const long lines = pf_bytes >> 7;
        const long per_cta = (lines + kCTAs - 1) / kCTAs;
        const long line = static_cast<long>(blockIdx.x) * per_cta + threadIdx.x;
        if (threadIdx.x < per_cta && line < lines) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(pf_base + (line << 7)));
        }
    }
    if (threadIdx.x == 0 && !kLLInbox) {
        if (kPerCta) {
            // This CTA's release arrival covers exactly its own pushes; the
            // reducer's acquire read of bar==gen*num_ranks*kCTAs closes the chain.
            __threadfence_system();  // release for THIS CTA's pushes
            if (!kLocal) {
                multimem_red_add_rel_sys(bar_mc, 1);
            }
        } else {
            __threadfence_system();  // release side
            const unsigned int done = atomicAdd(grid_ctr, 1u);
            if ((done % kCTAs) == kCTAs - 1 && !kLocal) {
                __threadfence_system();  // acquire side after the relaxed atomic
                multimem_red_add_rel_sys(bar_mc, 1);
            }
        }
    }
    if (kFuseRed) {
        // CTAs 0..7 stay resident as the reducer (the spin+fold+zero
        // body; one pair/thread) -- the separate launch and its seam disappear; the
        // fold overlaps the remaining CTAs' publication/PF/exit. Deadlock-free:
        // arrivals never depend on folds, and this CTA's own arrival already happened.
        if (blockIdx.x >= kRedCtas) {
            SMP_FLUSH();  // this CTA's task ends here (publish-only)
            return;
        }
        if constexpr (kLLInbox) {
            // Eight reducer CTAs x kThreads = kM4*kDim/2 threads, so each thread
            // owns EXACTLY ONE packet index and polls it once per source: eight
            // polled locations per thread. No in-flight NVLS writes contend with polling.
            constexpr int kPktPerSrc = kM4 * (kDim / 2);
            const int gidl = blockIdx.x * kThreads + threadIdx.x;
            const unsigned long long mybox = inbox_ptrs[rank];
            const unsigned epoch = static_cast<unsigned>(gen);
            SMP_REC(SMP_EV_WAIT_B);
            float a0 = 0.f, a1 = 0.f;
            for (int s = 0; s < num_ranks; ++s) {
                const unsigned long long addr =
                    mybox +
                    ((static_cast<unsigned long long>(parity) * num_ranks + s) * kPktPerSrc +
                     gidl) *
                        8ull;
                unsigned long long pk = ll_load_pkt(addr);
                int spins = 0;
                while (static_cast<unsigned>(pk >> 32) != epoch) {
                    if (++spins > kMaxSpins) {
                        if (fuse_status != nullptr) {
                            atomicCAS(fuse_status, 0, 1);
                        }
                        break;
                    }
                    __nanosleep(64);
                    pk = ll_load_pkt(addr);
                }
                const unsigned pay = static_cast<unsigned>(pk & 0xffffffffull);
                const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&pay);
                a0 += __bfloat162float(v.x);
                a1 += __bfloat162float(v.y);
            }
            SMP_REC(SMP_EV_WAIT_E);
            SMP_REC(SMP_EV_COMB_B);
            __nv_bfloat162* d2l = reinterpret_cast<__nv_bfloat162*>(fuse_x4) + gidl;
            const __nv_bfloat162 rvl = *d2l;
            *d2l = __nv_bfloat162(__float2bfloat16(__bfloat162float(rvl.x) + a0),
                                  __float2bfloat16(__bfloat162float(rvl.y) + a1));
            SMP_REC(SMP_EV_COMB_E);
        } else if (threadIdx.x == 0) {
            const unsigned target =
                static_cast<unsigned>(gen) * static_cast<unsigned>(num_ranks) * kCTAs;
            int spins = 0;
            while ((int)(static_cast<unsigned>(ld_acq_sys_i32(bar_local)) - target) < 0) {
                if (++spins > kMaxSpins) {
                    if (fuse_status != nullptr) {
                        atomicCAS(fuse_status, 0, 1);
                    }
                    break;
                }
                __nanosleep(64);
            }
        }
        __syncthreads();
        const int gid = blockIdx.x * kThreads + threadIdx.x;
        const int e0 = gid * 2;
        if constexpr (kLLInbox) {
            // The residual fold already happened in the poll loop above.
            // (there is no separate accumulator to drain or re-zero -- the epoch
            // tag makes every packet self-describing, so slots need no clearing).
            // Fall through only for the optional fq epilogue.
        } else {
            float* slot = fuse_acc + static_cast<size_t>(parity) * kM4 * kDim;
            // Vectorized fold: float2 slot load/zero + one bf16x2 residual RMW.
            // (identical per-element adds/rounds; half the fold transactions).
            const float2 v2f = reinterpret_cast<const float2*>(slot)[gid];
            __nv_bfloat162* d2 = reinterpret_cast<__nv_bfloat162*>(fuse_x4) + gid;
            const __nv_bfloat162 rv = *d2;
            const float r0 = __bfloat162float(rv.x) + v2f.x;
            const float r1 = __bfloat162float(rv.y) + v2f.y;
            *d2 = __nv_bfloat162(__float2bfloat16(r0), __float2bfloat16(r1));
            reinterpret_cast<float2*>(slot)[gid] = make_float2(0.0f, 0.0f);
        }
        __nv_bfloat16* d = fuse_x4 + e0;  // fq epilogue reads the stored bf16s
        if (fuse_x4q != nullptr) {
            // The quantization epilogue reuses block_partials4 for warp_amax scratch.
            // (free after the publication barrier).
            float* warp_amax = block_partials4;
            const float q0 = __bfloat162float(d[0]);
            const float q1 = __bfloat162float(d[1]);
            float amax = fmaxf(fabsf(q0), fabsf(q1));
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
            }
            const int warp = threadIdx.x >> 5;
            if ((threadIdx.x & 31) == 0) {
                warp_amax[warp] = amax;
            }
            __syncthreads();
            amax = fmaxf(fmaxf(warp_amax[warp & ~1], warp_amax[warp | 1]), 1e-8f);
            const float inv = 448.0f / amax;
            const __nv_fp8x2_e4m3 q2(make_float2(q0 * inv, q1 * inv));
            *reinterpret_cast<uint16_t*>(fuse_x4q + e0) = *reinterpret_cast<const uint16_t*>(&q2);
            if ((threadIdx.x & 63) == 0) {
                fuse_xs4[(e0 >> 12) * (kDim / 128) + ((e0 & 4095) >> 7)] = amax / 448.0f;
            }
        }
        SMP_FLUSH();  // reducer CTAs: task ends after the fold
    }
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M4

using namespace nexrt::sm90::nex_n2_5_pro::M4;
extern "C" {

int NexN25ProM4MoeL2Launch(const void* packed4, const void* hidden4, const void* weights,
                           const void* weight_scales, const void* shared_weights,
                           const void* shared_weight_scales, void* current_generation,
                           const void* inbox_ptrs, void* ll_counters, int rank, int num_ranks,
                           const void* prefetch_base, long long prefetch_bytes, void* status,
                           void* residual4, void* stream) {
    if (num_ranks <= 0 || num_ranks > kMaxRanks || rank < 0 || rank >= num_ranks ||
        current_generation == nullptr || inbox_ptrs == nullptr || ll_counters == nullptr ||
        status == nullptr || residual4 == nullptr || prefetch_bytes < 0 ||
        (prefetch_bytes & 127ll) != 0ll) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const long prefetch_lines = static_cast<long>(prefetch_bytes >> 7);
    if ((prefetch_bytes > 0 && prefetch_base == nullptr) || prefetch_lines > 128L * 1024L) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    // The default LL-inbox instance: fold, push, unroll-2 slot pairs, per-CTA
    // arrivals, v4 packers, fused reducer, 128 CTAs x 1024 threads, LL inbox.
    static bool attributes_ready = false;
    if (!attributes_ready) {
        const cudaError_t error = cudaFuncSetAttribute(
            M4MoeL2ProducerH32XKernel<true, true, false, true, false, false, true, false, true,
                                      true, false, 128, 1024, 1, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kHs2DynBytes1024);
        if (error != cudaSuccess) return static_cast<int>(error);
        attributes_ready = true;
    }
    M4MoeL2ProducerH32XKernel<true, true, false, true, false, false, true, false, true, true, false,
                              128, 1024, 1, true>
        <<<dim3(128), dim3(1024), kHs2DynBytes1024, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const int32_t*>(packed4), static_cast<const float*>(hidden4),
            static_cast<const __nv_fp8_e4m3*>(weights), static_cast<const float*>(weight_scales),
            static_cast<const __nv_fp8_e4m3*>(shared_weights),
            static_cast<const float*>(shared_weight_scales),
            /*bar_local=*/nullptr, static_cast<int*>(current_generation),
            /*grid_ctr=*/nullptr,
            /*rep_mc=*/0ull, /*acc_mc=*/0ull, /*bar_mc=*/0ull, rank, num_ranks,
            static_cast<const char*>(prefetch_base), static_cast<long>(prefetch_bytes),
            /*fuse_acc=*/nullptr, static_cast<int*>(status), static_cast<__nv_bfloat16*>(residual4),
            /*fuse_x4q=*/nullptr, /*fuse_xs4=*/nullptr,
            reinterpret_cast<unsigned long long*>(const_cast<void*>(inbox_ptrs)),
            static_cast<unsigned int*>(ll_counters),
            /*smp_records=*/nullptr, /*smp_anchor=*/nullptr);
    return static_cast<int>(cudaGetLastError());
}

long long NexN25ProM4MoeL2InboxBytes() { return 2ll * kMaxRanks * (kM4 * (kDim / 2)) * 8ll; }
int NexN25ProM4MoeL2CounterInts() { return 128; }

}  // extern "C"
