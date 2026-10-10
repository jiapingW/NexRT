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

// DF head argmax combine over TP ranks via the LL peer-inbox wire.
// Replaces two small collective gathers and their glue with one single-CTA
// kernel: each rank publishes
// 7 x (val fp32, gid u32) as 2x8 B self-tagged packets (multimem.st
// replication), polls all ranks' packets, and picks per row the max val with
// ties to the SMALLEST gid — exactly the torch argsort(gid)->argmax(first)
// semantics (per-rank gid ranges are disjoint and ascending in rank).
// Deterministic and row-invariant: fixed ascending-rank scan, strict >.
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_bf16.h>

#include "../common/pdl.cuh"
#include "ll_wire.cuh"

namespace nexrt::sm90::nex_n2_5_pro::DFlash {

constexpr int kRows7 = 7;        // proposal rows (block-8 minus anchor)
constexpr int kPk = kRows7 * 2;  // packets per rank: (val, gid) per row

__global__ __launch_bounds__(128) void DfArgmaxLLKernel(
    const float* __restrict__ val7,  // [7] fp32
    const long* __restrict__ idx7,   // [7] int64 local
    long rank_base,                  // rank * rows_local
    unsigned long long ll_mc, const uint64_t* __restrict__ ll_local,
    unsigned int* __restrict__ ll_ctr,  // [>=1]
    int* __restrict__ status,
    int* __restrict__ ids7_out,  // [7] int32 global ids
    int rank, int num_ranks) {
    NEXRT_PDL_ENTRY_SYNC();
    __shared__ unsigned cta_gen;
    __shared__ float sval[kLlMaxRanks][kRows7];
    __shared__ uint32_t sgid[kLlMaxRanks][kRows7];
    if (threadIdx.x == 0) {
        unsigned g = ll_ctr[0] + 1u;
        if (g == 0u) g = 2u;
        ll_ctr[0] = g;
        cta_gen = g;
    }
    __syncthreads();
    const uint32_t gen_u = cta_gen;
    const int par = (static_cast<int>(gen_u) - 1) & 1;
    // publish this rank's 14 packets: slot = row*2 + kind
    if (threadIdx.x < kPk) {
        const int row = threadIdx.x >> 1, kind = threadIdx.x & 1;
        const uint32_t pay =
            kind ? static_cast<uint32_t>(idx7[row] + rank_base) : __float_as_uint(val7[row]);
        const long idx = df_ll_idx(par, 0, threadIdx.x, 0, rank);
        df_mm_st_rel_sys_b64(ll_mc + static_cast<unsigned long long>(idx) * 8ull,
                             df_ll_pack(pay, gen_u));
    }
    // poll all ranks' packets: thread t <- (r, slot)
    if (threadIdx.x < num_ranks * kPk) {
        const int r = threadIdx.x / kPk, slot = threadIdx.x % kPk;
        const int row = slot >> 1, kind = slot & 1;
        const long idx = df_ll_idx(par, 0, slot, 0, r);
        uint64_t pkt;
        int spins = 0;
        do {
            pkt = df_ld_rel_sys_b64(ll_local + idx);
            if (++spins > kLlMaxSpins) {
                atomicOr(status, 4);
                break;
            }
        } while (static_cast<uint32_t>(pkt >> 32) != gen_u);
        const uint32_t pay = static_cast<uint32_t>(pkt);
        if (kind) {
            sgid[r][row] = pay;
        } else {
            sval[r][row] = __uint_as_float(pay);
        }
    }
    __syncthreads();
    if (threadIdx.x < kRows7) {
        const int row = threadIdx.x;
        float best = sval[0][row];
        uint32_t bg = sgid[0][row];
        for (int r = 1; r < num_ranks; ++r) {  // ascending rank == ascending gid
            const float v = sval[r][row];
            if (v > best) {
                best = v;
                bg = sgid[r][row];
            }
        }
        ids7_out[row] = static_cast<int>(bg);
    }
}

}  // namespace nexrt::sm90::nex_n2_5_pro::DFlash
using namespace nexrt::sm90::nex_n2_5_pro::DFlash;
extern "C" {

int NexN25ProDFlashArgmaxLLLaunch(const void* val7, const void* idx7, long rank_base,
                                  unsigned long long ll_mc, const void* ll_local, void* ll_ctr,
                                  void* status, void* ids7_out, int rank, int num_ranks,
                                  void* stream) {
    if (num_ranks < 1 || num_ranks > kLlMaxRanks) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    DfArgmaxLLKernel<<<dim3(1), dim3(128), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(val7), static_cast<const long*>(idx7), rank_base, ll_mc,
        static_cast<const uint64_t*>(ll_local), static_cast<unsigned int*>(ll_ctr),
        static_cast<int*>(status), static_cast<int*>(ids7_out), rank, num_ranks);
    return static_cast<int>(cudaGetLastError());
}
}
