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

// M=4 attention and GDN O-projections share this BF16 LL reduction protocol.
// Each publisher sends its rounded partials and polls peer epochs; every rank
// accumulates in rank order 0..7 in FP32 before the BF16 residual store.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M4
#endif

#pragma once

#include "oproj_common.cuh"

namespace nexrt::sm90::nex_n2_5_pro::M4 {
constexpr int kM4Fused = 4;
constexpr int kFusedCTAs = 128;
constexpr int kFusedSlots = 32;

// Wire: [parity][rank][CTA][slot][draft_pair], 8 bytes per BF16x2/epoch packet.
// The producer's CTA barrier makes part[][] ready before this helper runs.
// Self stays in registers and receives the same BF16 rounding as peer payloads.
constexpr long kM4LlPkts = kFusedSlots * (kM4Fused / 2);
constexpr long kM4LlRankStride = kFusedCTAs * kM4LlPkts;

__device__ __forceinline__ void m4_oproj_publish_and_reduce_ll(
    const float part[kM4Fused][kFusedSlots], unsigned long long rep_mc,
    const float* __restrict__ rep_local, int* __restrict__ status,
    __nv_bfloat16* __restrict__ x4_resid, int rank, int num_ranks, int gen, int row_base) {
    if (threadIdx.x >= kM4LlPkts) {
        return;
    }
    const int q = threadIdx.x;
    const int slot = q >> 1;
    const int jp = q & 1;
    const long parity_off = static_cast<long>((gen - 1) & 1) * kMaxRanks * kM4LlRankStride;
    const long elem = parity_off + static_cast<long>(blockIdx.x) * kM4LlPkts + q;
    const __nv_bfloat162 self =
        __float22bfloat162_rn(make_float2(part[2 * jp][slot], part[2 * jp + 1][slot]));
    const unsigned payload = *reinterpret_cast<const unsigned*>(&self);
    const unsigned long long packet =
        (static_cast<unsigned long long>(static_cast<unsigned>(gen)) << 32) | payload;
    const unsigned long long dst =
        rep_mc + static_cast<unsigned long long>(elem + rank * kM4LlRankStride) * 8ull;
    asm volatile("multimem.st.relaxed.sys.global.u64 [%0], %1;" ::"l"(dst), "l"(packet) : "memory");

    const auto* base = reinterpret_cast<const unsigned long long*>(rep_local) + elem;
    unsigned long long packets[kMaxRanks];
    bool ready = false;
    int spins = 0;
    while (!ready) {
        ready = true;
#pragma unroll
        for (int r = 0; r < kMaxRanks; ++r) {
            if (r < num_ranks && r != rank) {
                const auto* src = base + static_cast<long>(r) * kM4LlRankStride;
                asm volatile("ld.relaxed.sys.global.u64 %0, [%1];"
                             : "=l"(packets[r])
                             : "l"(src)
                             : "memory");
                ready &= static_cast<unsigned>(packets[r] >> 32) == static_cast<unsigned>(gen);
            }
        }
        if (!ready) {
            if (++spins > kMaxSpins) {
                if (status) {
                    atomicCAS(status, 0, 1);
                }
                return;
            }
            __nanosleep(64);
        }
    }

    // Self uses the same BF16 rounding as every peer, at its fixed rank position.
    // Initializing from rank 0 also preserves its signed zero before subsequent adds.
    float s0 = 0.0f;
    float s1 = 0.0f;
#pragma unroll
    for (int r = 0; r < kMaxRanks; ++r) {
        if (r < num_ranks) {
            const unsigned pay = r == rank ? payload : static_cast<unsigned>(packets[r]);
            const __nv_bfloat162 value = *reinterpret_cast<const __nv_bfloat162*>(&pay);
            if (r == 0) {
                s0 = __bfloat162float(value.x);
                s1 = __bfloat162float(value.y);
            } else {
                s0 += __bfloat162float(value.x);
                s1 += __bfloat162float(value.y);
            }
        }
    }
    const int row = row_base + (slot & 15) + ((slot >> 4) * (kDim / 2));
    auto* d0 = x4_resid + static_cast<long>(2 * jp) * kDim + row;
    auto* d1 = x4_resid + static_cast<long>(2 * jp + 1) * kDim + row;
    *d0 = __float2bfloat16(__bfloat162float(*d0) + s0);
    *d1 = __float2bfloat16(__bfloat162float(*d1) + s1);
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M4
