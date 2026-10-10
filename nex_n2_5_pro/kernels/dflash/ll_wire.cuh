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

// DF-owned LL peer-inbox wire for the draft's TP reduce sites (o_proj, MLP
// down, fc): 8 B self-tagged packets (4 B fp32 payload | 4 B epoch), single-copy
// atomic; store AND poll are .relaxed.sys (the morally-strong pair); publish
// is ONE multimem.st (fabric replication, not arithmetic); combine sums SELF
// FIRST then ascending rank (order-stable, C-independent). Workspace and
// per-CTA epoch counters are DF-OWNED allocations — never share counters
// across kernel families.
//
// Geometry (all three DF sites): grid 128 CTAs x 32 warp-slots, kM=8 rows,
// output dim 4096 = 128 x 32.
#pragma once

#include <cstdint>

namespace nexrt::sm90::nex_n2_5_pro::DFlash {
constexpr int kLlCTAs = 128;
constexpr int kLlSlots = 32;
constexpr int kLlRows = 8;  // draft rows (fc uses rows < p only)
constexpr int kLlMaxRanks = 8;
constexpr int kLlMaxSpins = 1 << 22;

__device__ __forceinline__ void df_mm_st_rel_sys_b64(unsigned long long mc, uint64_t v) {
    asm volatile("multimem.st.relaxed.sys.global.b64 [%0], %1;\n" ::"l"(mc), "l"(v) : "memory");
}
__device__ __forceinline__ uint64_t df_ld_rel_sys_b64(const uint64_t* p) {
    uint64_t v;
    asm volatile("ld.relaxed.sys.global.b64 %0, [%1];\n" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint64_t df_ll_pack(uint32_t pay, uint32_t gen) {
    return static_cast<uint64_t>(pay) | (static_cast<uint64_t>(gen) << 32);
}

// Packet index (parity, cta, slot, row, rank), rank-MAJOR (one contiguous
// 1 KiB publish per (rank, CTA); combine polls stride).
__device__ __forceinline__ long df_ll_idx(int par, int cta, int slot, int j, int r) {
    const long per_rank = static_cast<long>(kLlCTAs) * kLlSlots * kLlRows;
    const long per_par = per_rank * kLlMaxRanks;
    return static_cast<long>(par) * per_par + static_cast<long>(r) * per_rank +
           (static_cast<long>(cta) * kLlSlots + slot) * kLlRows + j;
}

constexpr long kLlWireFloats = 2L * kLlMaxRanks * kLlCTAs * kLlSlots * kLlRows;

// Publish this CTA's [slot][row] fp32 partials (smem part[row][slot]) and
// combine with peers; adds the combined value into out_bf16 (+= residual).
// gen/parity from the caller's per-CTA epoch. 1024-thread CTAs assumed.
template <bool kAddResidual>
__device__ __forceinline__ void df_ll_reduce_tail(float (&part)[kLlRows][kLlSlots], int rows,
                                                  unsigned long long ll_mc,
                                                  const uint64_t* ll_local, int cta_gen, int rank,
                                                  int num_ranks, int* status, __nv_bfloat16* out,
                                                  int out_stride_row, int row_base) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const uint32_t gen_u = static_cast<uint32_t>(cta_gen);
    const int par = (cta_gen - 1) & 1;
    // publish: kLlSlots*kLlRows = 256 packets by the first 256 threads
    if (threadIdx.x < kLlSlots * kLlRows) {
        const int slot = threadIdx.x >> 3;  // /kLlRows
        const int j = threadIdx.x & 7;
        const long idx = df_ll_idx(par, blockIdx.x, slot, j, rank);
        df_mm_st_rel_sys_b64(ll_mc + static_cast<unsigned long long>(idx) * 8ull,
                             df_ll_pack(__float_as_uint(part[j][slot]), gen_u));
    }
    // combine: warp w = slot w; lane = row*8/rank split (kLlRows x 8 = 256
    // lanes needed per slot -> 8 lanes per row group like the M4 form with
    // kM=8: lane = j*4? Use j = lane >> 3 (0..3)... kLlRows=8 rows x 8 ranks
    // = 64 > 32 lanes: split rows over TWO passes (rows 0-3, 4-7).
    const int slot = warp;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int j = (lane >> 3) + half * 4;
        const int r = lane & 7;
        float v = 0.0f;
        if (j < kLlRows) {
            if (r == rank) {
                v = part[j][slot];
            } else if (r < num_ranks) {
                const uint64_t* p = ll_local + df_ll_idx(par, blockIdx.x, slot, j, r);
                uint64_t pv;
                int spins = 0;
                for (;;) {
                    pv = df_ld_rel_sys_b64(p);
                    if (static_cast<uint32_t>(pv >> 32) == gen_u) break;
                    if (++spins > kLlMaxSpins) {
                        if (status) {
                            atomicCAS(status, 0, 1);
                        }
                        break;
                    }
                }
                v = __uint_as_float(static_cast<uint32_t>(pv));
            }
        }
        const int gbase = lane & 24;
        float s = __shfl_sync(0xffffffffu, v, gbase | rank);  // self first
#pragma unroll
        for (int rr = 0; rr < kLlMaxRanks; ++rr) {
            if (rr < num_ranks && rr != rank) {
                s += __shfl_sync(0xffffffffu, v, gbase | rr);
            }
        }
        if ((lane & 7) == 0 && j < rows) {
            __nv_bfloat16* d = out + static_cast<long>(j) * out_stride_row + row_base + slot;
            if constexpr (kAddResidual) {
                *d = __float2bfloat16(__bfloat162float(*d) + s);
            } else {
                *d = __float2bfloat16(s);
            }
        }
    }
}
}  // namespace nexrt::sm90::nex_n2_5_pro::DFlash
