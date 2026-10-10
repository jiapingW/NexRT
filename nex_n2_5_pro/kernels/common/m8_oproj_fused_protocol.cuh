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

// M=8 fused output-projection LL peer-inbox protocol, shared by the GDN and
// attention twins. Mutable workspace and epochs are M=8-family state only.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M8
#endif

#pragma once

namespace nexrt::sm90::nex_n2_5_pro::M8 {
constexpr int kM8Fused = 8;
constexpr int kM8FusedCTAs = 128;  // must match the producer stripe geometry (16 rows/half)
constexpr int kM8FusedSlots = 32;  // warps per CTA == output rows per CTA

constexpr long kM8CtaValues = static_cast<long>(kM8FusedSlots) * kM8Fused;
constexpr long kM8RankValues = static_cast<long>(kM8FusedCTAs) * kM8CtaValues;

// Readiness rides INSIDE an 8 B packet -- 4 B payload + 4 B epoch tag --
// so the mc-red arrival signal, the ld.acquire counter spin and the
// post-wait __syncthreads all disappear; a combiner starts the moment its
// packets land. Packet publication and polling use the following constraints:
//   * the packet is EXACTLY 8 B: a naturally-aligned 8 B access is single-
//     copy atomic, so a fresh tag can never pair with stale payload. The
//     protocol must not be widened to a 16 B packet: H100 does not guarantee
//     a single-copy-atomic 16 B global transaction.
//   * store and poll are BOTH .relaxed.sys, which is what makes the pair
//     morally strong (PTX 9.1 SS8.10.3). A .weak store does not pair.
//   * publish is one multimem.st (the fabric replicates), avoiding eight
//     directed unicast copies per CTA.
//     Note this is multimem.st = REPLICATION; the standing advice against
//     multimem concerns multimem.red/ld_reduce = fabric ARITHMETIC.
//
// bf16 wire: the cross-rank sum runs on bf16-rounded partials, the same
// declared numerics class used by the shipped M4 MoE-L2 LL-inbox protocol.
// A bitwise-exact fp32-payload form costs more
// (the epoch tag doubles the wire), so it is not part of production.
//
// PROXY-ALIASING POSTURE: packets are written through the multicast alias
// and polled through the local UNICAST alias, which under the CUDA guide's
// object-granular reading voids the morally-strong argument. Mitigating
// fact, satisfied here: no single location is ever accessed through both
// aliases by one GPU -- a rank writes only its own slots and reads only
// its peers', and its own value never leaves registers.
//
// The publishing and polling warps are the same, so no writer-side
// fence.proxy.alias is inserted between their store and poll sequence.
constexpr long kM8LlPkts = kM8CtaValues / 2;  // 128 per (rank, CTA)
constexpr long kM8LlRankStride = static_cast<long>(kM8FusedCTAs) * kM8LlPkts;
// Workspace floats = 2 parities * kMaxRanks * kM8LlRankStride * 2 payload words.

__device__ __forceinline__ void m8_ll_st_pkt(unsigned long long mc, unsigned long long v) {
    asm volatile("multimem.st.relaxed.sys.global.u64 [%0], %1;" ::"l"(mc), "l"(v) : "memory");
}

__device__ __forceinline__ unsigned long long m8_ll_ld_pkt(const void* p) {
    unsigned long long v;
    asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

// One folding thread owns exactly one wire packet and the two (draft, row)
// outputs that packet carries -- slot = warp*8 + gid, jp = tig -- so the
// partials never reach shared memory, the post-fold barrier disappears,
// the separate combine pass disappears, and `self` stays in registers.
// v0/v1 are this thread's two folded fp32 partials.
__device__ __forceinline__ void m8_oproj_fold_publish_combine_ll(
    float v0, float v1, int slot, int jp, unsigned long long rep_mc,
    const float* __restrict__ rep_local, int* __restrict__ status,
    __nv_bfloat16* __restrict__ x8_resid, int rank, int num_ranks, int gen, int row_base) {
    const long parity_off = static_cast<long>((gen - 1) & 1) * kMaxRanks * kM8FusedCTAs * kM8LlPkts;
    const long my_src =
        parity_off + (static_cast<long>(rank) * kM8FusedCTAs + blockIdx.x) * kM8LlPkts;
    const int q = slot * 4 + jp;

    const __nv_bfloat162 pv = __float22bfloat162_rn(make_float2(v0, v1));
    const unsigned payload = *reinterpret_cast<const unsigned*>(&pv);
    m8_ll_st_pkt(rep_mc + static_cast<unsigned long long>(my_src + q) * 8ull,
                 (static_cast<unsigned long long>(static_cast<unsigned>(gen)) << 32) | payload);

    // Joint spin: one poll round issues all peer loads independently, so
    // the wait is one memory latency, not seven chained ones.
    const unsigned long long* base = reinterpret_cast<const unsigned long long*>(rep_local) +
                                     parity_off + static_cast<long>(blockIdx.x) * kM8LlPkts + q;
    unsigned long long pk[kMaxRanks];
    bool ready = false;
    int spins = 0;
    while (!ready) {
        ready = true;
#pragma unroll
        for (int r = 0; r < kMaxRanks; ++r) {
            if (r < num_ranks && r != rank) {
                pk[r] = m8_ll_ld_pkt(base + static_cast<long>(r) * kM8FusedCTAs * kM8LlPkts);
                ready &= (static_cast<unsigned>(pk[r] >> 32) == static_cast<unsigned>(gen));
            }
        }
        if (!ready) {
            if (++spins > kMaxSpins) {
                if (status) {
                    atomicCAS(status, 0, 1);
                }
                break;  // caller contract: fail-closed
            }
            __nanosleep(64);
        }
    }

    float s0 = __bfloat162float(pv.x);  // self first ...
    float s1 = __bfloat162float(pv.y);
#pragma unroll
    for (int r = 0; r < kMaxRanks; ++r) {  // ... then ascending
        if (r < num_ranks && r != rank) {
            const unsigned pay = static_cast<unsigned>(pk[r]);
            const __nv_bfloat162 pr = *reinterpret_cast<const __nv_bfloat162*>(&pay);
            s0 += __bfloat162float(pr.x);
            s1 += __bfloat162float(pr.y);
        }
    }
    const int row = row_base + (slot & 15) + ((slot >> 4) * (kDim / 2));
    __nv_bfloat16* d0 = x8_resid + static_cast<long>(2 * jp) * kDim + row;
    __nv_bfloat16* d1 = x8_resid + static_cast<long>(2 * jp + 1) * kDim + row;
    *d0 = __float2bfloat16(__bfloat162float(*d0) + s0);
    *d1 = __float2bfloat16(__bfloat162float(*d1) + s1);
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M8