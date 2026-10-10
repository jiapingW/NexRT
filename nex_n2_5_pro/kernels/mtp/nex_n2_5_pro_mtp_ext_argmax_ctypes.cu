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

// CUSTOM argmax head tail for the M=4 EXTEND select (next round's d1).
//
// Own pair per the rewrite-don't-reuse rule: structurally similar
// to the T=1 draft pair but a SEPARATE site with its own geometry tuning and its own
// ctl contract (matching MtpDraftSelectKernel<0> in the glue TU). No code is shared
// with the draft TU, the bs=1 top-k chain, or the verify head. The extend select input
// is ONE row today (MtpRowSelectKernel picks the accepted row before the GEMV), so the
// problem is 31040 raw fp32 logits -> P -> C -> global argmax, computed REDUNDANTLY
// on every rank (all ranks identical afterward; zero further communication).
//
// Total order = the glue TU's DraftKey definition (finite only; orderable float bits
// high, ~id low => ties pick the LOWEST global id). inv_rms is a positive scalar and
// argmax-invariant, so it is not applied. Cross-rank readiness = SENTINEL (0ull is
// forbidden: real keys have low word ~id != 0, non-finite keys are guarded to 1ull);
// multimem.st.u64 pushes are 8 B atomic; fence.proxy.alias separates this CTA's
// multicast store from its unicast polling (the same aliasing contract as the verify
// commit); slots zero-on-read.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M4
#endif

#include <cuda_runtime.h>

#include <cstdint>

#include "../common/m4_control_block.cuh"

namespace nexrt::sm90::nex_n2_5_pro::M4 {
constexpr int kShardRows = 31040;  // vocab 248320 / TP8, contiguous per rank
constexpr int kCand = 8;           // candidates pushed per rank
constexpr int kXchThreads = 64;    // one thread per 8x8 slot in the poll

__device__ __forceinline__ uint64_t ExtArgmaxKey(float value, int id) {
    if (!isfinite(value)) return 0ull;  // non-finite loses
    uint32_t bits = __float_as_uint(value);
    bits ^= (bits >> 31) ? 0xffffffffu : 0x80000000u;  // total order on floats
    return (static_cast<uint64_t>(bits) << 32) | static_cast<uint32_t>(~static_cast<uint32_t>(id));
}

template <int kThreads>
__device__ __forceinline__ uint64_t ExtBlockMaxU64(uint64_t v) {
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
    __syncthreads();
    return r;
}

constexpr int kProducerCtas = 132;     // one CTA/SM
constexpr int kProducerThreads = 256;  // approximately one element/thread

__global__ __launch_bounds__(kProducerThreads) void MtpExtArgmaxProducerKernel(
    const float* __restrict__ logits, unsigned long long* __restrict__ partials, int rank) {
    const int base = rank * kShardRows;
    uint64_t best = 0ull;
    for (int i = blockIdx.x * kProducerThreads + threadIdx.x; i < kShardRows;
         i += kProducerCtas * kProducerThreads) {
        const uint64_t k = ExtArgmaxKey(logits[i], base + i);
        if (k > best) {
            best = k;
        }
    }
    best = ExtBlockMaxU64<kProducerThreads>(best);
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = best;
    }
}

// ctl contract matching MtpDraftSelectKernel<0> (glue TU): chain slot A = [d1, C],
// verify_input_ids[1] = d1.
__global__ __launch_bounds__(kXchThreads) void MtpExtArgmaxExchangeKernel(
    const unsigned long long* __restrict__ partials, unsigned long long mc_slots,
    volatile unsigned long long* __restrict__ uc_slots, int* __restrict__ ctl, int rank) {
    uint64_t cand = 0ull;
    for (int i = blockIdx.x * kXchThreads + threadIdx.x; i < kProducerCtas;
         i += kCand * kXchThreads) {
        const uint64_t v = partials[i];
        if (v > cand) {
            cand = v;
        }
    }
    cand = ExtBlockMaxU64<kXchThreads>(cand);
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
    const uint64_t final_key = ExtBlockMaxU64<kXchThreads>(got);
    uc_slots[threadIdx.x] = 0ull;  // zero-on-read: bank re-armed
    if (threadIdx.x != 0) {
        return;
    }
    const int d = static_cast<int>(~static_cast<uint32_t>(final_key & 0xffffffffull));
    ctl[m4ctl::kChainSlotA] = d;
    ctl[m4ctl::kChainSlotA + 1] = ctl[m4ctl::kCommittedLen];
    ctl[m4ctl::kVerifyInputIds + 1] = d;
}

extern "C" {
// Geometry fixed at 132 CTAs x 256 thr by sweep: 264x128 and 264x256 tie
// within jitter, 132x128 loses +0.35 us (2 serial loads per thread past the
// ~1-element/thread knee).
int NexN25ProMtpExtArgmaxProducerLaunch(const void* logits, void* partials, int rank,
                                        void* stream) {
    MtpExtArgmaxProducerKernel<<<132, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(logits), static_cast<unsigned long long*>(partials), rank);
    return static_cast<int>(cudaGetLastError());
}

int NexN25ProMtpExtArgmaxExchangeLaunch(const void* partials, unsigned long long mc_slots,
                                        void* uc_slots, void* ctl, int rank, void* stream) {
    if (mc_slots == 0ull) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    MtpExtArgmaxExchangeKernel<<<kCand, kXchThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const unsigned long long*>(partials), mc_slots,
        static_cast<volatile unsigned long long*>(uc_slots), static_cast<int*>(ctl), rank);
    return static_cast<int>(cudaGetLastError());
}
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M4
