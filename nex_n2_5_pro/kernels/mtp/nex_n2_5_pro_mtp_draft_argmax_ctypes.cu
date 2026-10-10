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

// CUSTOM argmax head tail for the T=1 MTP DRAFT passes (d2/d3 selects).
//
// REWRITE-DON'T-REUSE: routing the draft head through the bs=1 lm_head top-k chain
// (warp top-20 heaps -> 1280 cands -> peer merge -> 480 for the bs=1 SamplerKernel ->
// DraftSelect max-scan) costs 20.1 us/pass at 0.4% MBU for what is an argmax. This TU
// is a per-site rewrite: 31040 -> P -> C, then C x 8 ranks -> 1 computed REDUNDANTLY
// on every rank (all ranks identical afterward; zero further communication). The bs=1
// machinery is untouched (its sampling is real). The M=4 extend select has its OWN
// pair in nex_n2_5_pro_mtp_ext_argmax_ctypes.cu -- no shared code.
//
// Total order: identical DEFINITION to DraftKey in the glue TU -- finite
// values only, orderable float bits high, ~id low => ties pick the LOWEST global id.
// inv_rms is intentionally NOT applied: it is a positive per-token scalar, so the
// argmax and every tie relation over the raw logits are unchanged (the glue TU
// applies it only to feed TRUE logit VALUES to the sampler, which draft never uses).
//
// Cross-rank exchange readiness = SENTINEL (LL-family):
// a real key always has low word ~id != 0 (id < 2^18), and non-finite keys are
// guarded to 1ull, so 0ull is a forbidden pattern. Each rank multimem.st's its C
// candidates (8 B atomic, PTX multimem.st.u64) into symmetric slots; every rank
// polls its LOCAL replica until all 8xC slots are non-zero -- the payload proves
// readiness, no arrival counter round-trip. fence.proxy.alias separates this CTA's
// multicast store from its unicast polling of the same allocation (the same aliasing
// contract as the verify commit).
// Slots are zeroed-on-read after full consumption; the next write to the same bank
// is a full round away, behind the verify rendezvous (cross-rank happens-before).
// Degenerate all-non-finite logits produce key 1ull (garbage token, same class as
// the glue TU select's -1) instead of a stall.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M4
#endif

#include <cuda_runtime.h>

#include <cstdint>

#include "../common/m4_control_block.cuh"

namespace nexrt::sm90::nex_n2_5_pro::M4 {
constexpr int kShardRows = 31040;  // vocab 248320 / TP8, contiguous per rank
constexpr int kCand = 8;           // candidates pushed per rank
constexpr int kXchThreads = 64;    // one thread per slot in the poll phase

__device__ __forceinline__ uint64_t DraftArgmaxKey(float value, int id) {
    if (!isfinite(value)) return 0ull;  // non-finite loses
    uint32_t bits = __float_as_uint(value);
    bits ^= (bits >> 31) ? 0xffffffffu : 0x80000000u;  // total order on floats
    return (static_cast<uint64_t>(bits) << 32) | static_cast<uint32_t>(~static_cast<uint32_t>(id));
}

template <int kThreads>
__device__ __forceinline__ uint64_t BlockMaxU64(uint64_t v) {
    __shared__ uint64_t sm[kThreads];
    sm[threadIdx.x] = v;
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && sm[threadIdx.x + s] > sm[threadIdx.x]) {
            sm[threadIdx.x] = sm[threadIdx.x + s];
        }
        __syncthreads();
    }
    const uint64_t r = sm[0];
    __syncthreads();  // sm reuse safety
    return r;
}

// ---- producer: 31040 raw fp32 logits -> P = kCtas partial max keys ---------------
constexpr int kProducerCtas = 132;     // one CTA/SM; ~1 element/thread
constexpr int kProducerThreads = 256;  // selected producer geometry

__global__ __launch_bounds__(kProducerThreads) void MtpDraftArgmaxProducerKernel(
    const float* __restrict__ logits, unsigned long long* __restrict__ partials, int rank) {
    const int base = rank * kShardRows;
    uint64_t best = 0ull;
    for (int i = blockIdx.x * kProducerThreads + threadIdx.x; i < kShardRows;
         i += kProducerCtas * kProducerThreads) {
        const uint64_t k = DraftArgmaxKey(logits[i], base + i);
        if (k > best) {
            best = k;
        }
    }
    best = BlockMaxU64<kProducerThreads>(best);
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = best;
    }
}

// ---- exchange: P -> C local candidates -> multicast push -> Sentinel poll -> 1 ----
// Grid = kCand CTAs x kXchThreads. Every CTA reduces a grid-strided share of the
// partials and pushes ONE candidate; CTA 0 then polls all 8xC slots, reduces the
// global max redundantly (every rank computes the same value), and performs the
// same ctl staging as MtpDraftSelectKernel<mode> in the glue TU.
__global__ __launch_bounds__(kXchThreads) void MtpDraftArgmaxExchangeKernel(
    const unsigned long long* __restrict__ partials, unsigned long long mc_slots,
    volatile unsigned long long* __restrict__ uc_slots, int* __restrict__ ctl, int rank, int mode) {
    uint64_t cand = 0ull;
    for (int i = blockIdx.x * kXchThreads + threadIdx.x; i < kProducerCtas;
         i += kCand * kXchThreads) {
        const uint64_t v = partials[i];
        if (v > cand) {
            cand = v;
        }
    }
    cand = BlockMaxU64<kXchThreads>(cand);
    if (threadIdx.x == 0) {
        if (cand == 0ull) {
            cand = 1ull;
        }  // all-non-finite stall guard
        asm volatile(
            "multimem.st.relaxed.sys.global.u64 [%0], %1;" ::"l"(
                mc_slots + (static_cast<unsigned long long>(rank) * kCand + blockIdx.x) * 8ull),
            "l"(cand)
            : "memory");
    }
    if (blockIdx.x != 0) {
        return;
    }

    asm volatile("fence.proxy.alias;" ::: "memory");  // mc store -> uc poll path
    uint64_t got = 0ull;
    while ((got = uc_slots[threadIdx.x]) == 0ull) {
    }  // payload proves readiness
    const uint64_t final_key = BlockMaxU64<kXchThreads>(got);
    uc_slots[threadIdx.x] = 0ull;  // zero-on-read: bank re-armed
    if (threadIdx.x != 0) {
        return;
    }
    const int d = static_cast<int>(~static_cast<uint32_t>(final_key & 0xffffffffull));
    if (mode == 1) {  // draft #1 -> d2
        ctl[m4ctl::kChainSlotB] = d;
        ctl[m4ctl::kChainSlotB + 1] = ctl[m4ctl::kChainSlotA + 1] + 1;
        ctl[m4ctl::kVerifyInputIds + 2] = d;
    } else {  // draft #2 -> d3
        ctl[m4ctl::kVerifyInputIds + 3] = d;
    }
}

extern "C" {
// Geometry fixed at 132 CTAs x 256 thr by sweep: 264x128 and 264x256 tie
// within jitter, 132x128 loses +0.7 us (2 serial loads per thread past the
// ~1-element/thread knee).
int NexN25ProMtpDraftArgmaxProducerLaunch(const void* logits, void* partials, int rank,
                                          void* stream) {
    MtpDraftArgmaxProducerKernel<<<132, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(logits), static_cast<unsigned long long*>(partials), rank);
    return static_cast<int>(cudaGetLastError());
}

int NexN25ProMtpDraftArgmaxExchangeLaunch(const void* partials, unsigned long long mc_slots,
                                          void* uc_slots, void* ctl, int rank, int mode,
                                          void* stream) {
    if (mc_slots == 0ull || (mode != 1 && mode != 2)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    MtpDraftArgmaxExchangeKernel<<<kCand, kXchThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const unsigned long long*>(partials), mc_slots,
        static_cast<volatile unsigned long long*>(uc_slots), static_cast<int*>(ctl), rank, mode);
    return static_cast<int>(cudaGetLastError());
}
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M4