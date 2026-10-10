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

// Bounded per-request top-k across TP=8. Grid and buffer strides are fixed;
// producer/reducer selection lengths are read from the shared sampling buffer.
#include "sampling.cuh"
namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY {
constexpr int ROWS = 31040;
constexpr int K = kMaxTopK;
constexpr int kNumRanks = 8;
constexpr int kThreads = 128;
constexpr int kProducerCtas = 24;
constexpr int kReducerSplits = 1;

__device__ __forceinline__ int ld_acq_sys_i32(const int* ptr) {
    int r;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(r) : "l"(ptr) : "memory");
    return r;
}
__device__ __forceinline__ void multimem_red_add_rel_sys(unsigned long long mc, int val) {
    asm volatile("multimem.red.release.sys.global.add.s32 [%0], %1;" ::"l"(mc), "r"(val)
                 : "memory");
}
__device__ __forceinline__ int atom_add_acqrel_gpu(int* ptr,
                                                   int val) {  // gpu-scope count (cumulative)
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;"
                 : "=r"(old)
                 : "l"(ptr), "r"(val)
                 : "memory");
    return old;
}

// ---- Kernel B: producer (grid = NC). CHUNK & ITEMS are compile-time from <NT,NC>. -----------
template <int NT, int NC>
__global__ __launch_bounds__(NT) void ProducerKernel(
    const float* __restrict__ logits, float* __restrict__ cand_val, int* __restrict__ cand_id,
    const int* __restrict__ bar_local, int* __restrict__ cur_gen, unsigned long long bar_mc,
    int* __restrict__ done_ctr, int rank, const uint64_t* __restrict__ sampling_buffer) {
    constexpr int CHUNK = (ROWS + NC - 1) / NC;  // ceil: no divisibility requirement, tail padded
    constexpr int ITEMS = (CHUNK + NT - 1) / NT;
    constexpr int NW = NT / 32;
    const int cta = blockIdx.x;
    const int k = SamplingTopK(sampling_buffer);
    const int row0 = cta * CHUNK;
    const int gen = ld_acq_sys_i32(bar_local) / kNumRanks + 1;
    if (cta == 0 && threadIdx.x == 0) cur_gen[0] = gen;
    const int off = ((gen - 1) & 1) * (NC * K);

    uint32_t keys[ITEMS];
    int ids[ITEMS];
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const int loc = threadIdx.x + i * NT;
        const int row = row0 + loc;
        if (loc < CHUNK && row < ROWS) {
            keys[i] = OrderedFloat(logits[row]);
            ids[i] = rank * ROWS + row;
        } else {
            keys[i] = 0u;
            ids[i] = -1;
        }
    }
    __shared__ uint32_t s_key[K * NW];
    __shared__ int s_val[K * NW];
    SelectTopK<NT, ITEMS>(keys, ids, s_key, s_val, cand_val + off + cta * K,
                          cand_id + off + cta * K, k);  // raw (scale=1): reducer applies inv_rms

    {
        // The parallel merge writes from multiple threads. Publish the whole
        // CTA's candidate stores before its representative enters the release chain.
        __syncthreads();
        __shared__ int is_last;
        if (threadIdx.x == 0) is_last = (atom_add_acqrel_gpu(done_ctr, 1) == NC - 1);
        __syncthreads();
        if (is_last && threadIdx.x == 0) {
            // Acquire each CTA publication before the system-scope arrival.
            multimem_red_add_rel_sys(bar_mc, 1);  // ARRIVE (cumulative system-scope release)
            *done_ctr = 0;
        }
    }
}

// ---- Kernel C: reducer (grid = 8*M). CTA idx=(peer*M+sub) reduces one peer sub-chunk. ------
template <int NT, int NC, int M>
__global__ __launch_bounds__(NT) void ReducerKernel(
    const long* __restrict__ peer_cand_val, const long* __restrict__ peer_cand_id,
    int* __restrict__ bar_local, float* __restrict__ out_val, int* __restrict__ out_id,
    int* __restrict__ status, const int* __restrict__ cur_gen, const float* __restrict__ inv_rms,
    const uint64_t* __restrict__ sampling_buffer) {
    constexpr int PERPEER = NC * K;
    constexpr int SUBLEN = (PERPEER + M - 1) / M;  // candidates one reducer CTA scans
    constexpr int ITEMS = (SUBLEN + NT - 1) / NT;
    constexpr int NW = NT / 32;
    const int k = SamplingTopK(sampling_buffer);
    const int idx = blockIdx.x;  // 0 .. 8*M-1
    const int peer = idx / M, sub = idx % M;
    const int gen = cur_gen[0];
    const long base = static_cast<long>((gen - 1) & 1) * PERPEER + static_cast<long>(sub) * SUBLEN;

    if (threadIdx.x == 0) {  // wait for all 8 ranks' producer arrivals (signed spin, replay-safe)
        const int target = gen * kNumRanks;
        int spins = 0;
        while (static_cast<int>(ld_acq_sys_i32(bar_local) - target) < 0) {
            if (++spins > 2000000) {
                if (status) atomicCAS(status, 0, 1);
                break;
            }
            __nanosleep(64);
        }
    }
    __syncthreads();

    const int valid = min(SUBLEN, PERPEER - sub * SUBLEN);  // last sub-chunk may be partial
    const float* pv =
        reinterpret_cast<const float*>(peer_cand_val[peer]) + base;  // one peer -> coalesced
    const int* pi = reinterpret_cast<const int*>(peer_cand_id[peer]) + base;
    uint32_t keys[ITEMS];
    int ids[ITEMS];
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const int loc = threadIdx.x + i * NT;
        if (loc < valid && (sub * SUBLEN + loc) % K < k) {
            keys[i] = OrderedFloat(pv[loc]);
            ids[i] = pi[loc];
        } else {
            keys[i] = 0u;
            ids[i] = -1;
        }
    }
    __shared__ uint32_t s_key[K * NW];
    __shared__ int s_val[K * NW];
    // Selection uses raw logits; normalize the emitted values for sampling.
    SelectTopK<NT, ITEMS>(keys, ids, s_key, s_val, out_val + idx * K, out_id + idx * K, k,
                          inv_rms[0]);
}
}  // namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
using namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY;
extern "C" {
// Producer: 24 CTAs, 128 threads; fixed 24*28 candidate slots per rank.
int NexN25ProLmHeadTopkProducerLaunch(const void* lg, void* cv, void* ci, const void* bl, void* cg,
                                      unsigned long long bmc, void* dc, int rank,
                                      const void* sampling_buffer, void* s) {
    ProducerKernel<kThreads, kProducerCtas>
        <<<kProducerCtas, kThreads, 0, static_cast<cudaStream_t>(s)>>>(
            static_cast<const float*>(lg), static_cast<float*>(cv), static_cast<int*>(ci),
            static_cast<const int*>(bl), static_cast<int*>(cg), bmc, static_cast<int*>(dc), rank,
            static_cast<const uint64_t*>(sampling_buffer));
    return static_cast<int>(cudaGetLastError());
}
// Reducer: eight CTAs, one per peer; 8*28 candidates scaled by inv_rms.
int NexN25ProLmHeadTopkReducerLaunch(const void* pv, const void* pi, void* bl, void* ov, void* oi,
                                     void* st, const void* cg, const void* irms,
                                     const void* sampling_buffer, void* s) {
    ReducerKernel<kThreads, kProducerCtas, kReducerSplits>
        <<<8 * kReducerSplits, kThreads, 0, static_cast<cudaStream_t>(s)>>>(
            static_cast<const long*>(pv), static_cast<const long*>(pi), static_cast<int*>(bl),
            static_cast<float*>(ov), static_cast<int*>(oi), static_cast<int*>(st),
            static_cast<const int*>(cg), static_cast<const float*>(irms),
            static_cast<const uint64_t*>(sampling_buffer));
    return static_cast<int>(cudaGetLastError());
}
int NexN25ProLmHeadTopkNC() { return kProducerCtas; }
int NexN25ProLmHeadTopkM() { return kReducerSplits; }
int NexN25ProLmHeadTopkK() { return K; }
const char* NexN25ProLmHeadTopkCudaErrorString(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
}
