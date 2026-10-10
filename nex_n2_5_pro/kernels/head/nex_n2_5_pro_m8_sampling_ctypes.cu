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

#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M8
#include "sampling.cuh"
#include "../common/m8_control_block.cuh"
#include <cub/warp/warp_scan.cuh>
namespace nexrt::sm90::nex_n2_5_pro::M8 {
constexpr int R = 8, NT = 128, NC = 24, M = 1;
constexpr int ROWS = 31040, K = kMaxTopK, kNumRanks = 8;
constexpr int kM8Tail = R;
// Two status words follow the two candidate-ID banks, one per replay parity.
constexpr int kStatusOffset = 2 * R * NC * K;
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

// Producer grid = R * NC. Each CTA handles one vocabulary chunk of one row.
__global__ __launch_bounds__(NT) void M8ProducerKernel(
    const float* __restrict__ logits, float* __restrict__ cand_val, int* __restrict__ cand_id,
    const int* __restrict__ bar_local, int* __restrict__ cur_gen, unsigned long long bar_mc,
    int* __restrict__ done_ctr, int rank, const uint64_t* __restrict__ sampling_buffer,
    const long* upstream) {
    constexpr int CHUNK = (ROWS + NC - 1) / NC;  // ceil: no divisibility requirement, tail padded
    constexpr int ITEMS = (CHUNK + NT - 1) / NT;
    constexpr int NW = NT / 32;
    const int row = blockIdx.x / NC;
    const int cta = blockIdx.x % NC;
    const int k = SamplingTopK(sampling_buffer);
    const int row0 = cta * CHUNK;
    const int gen = ld_acq_sys_i32(bar_local) / kNumRanks + 1;
    if (blockIdx.x == 0 && threadIdx.x == 0) cur_gen[0] = gen;
    const int off = ((gen - 1) & 1) * (R * NC * K) + row * NC * K;

    uint32_t keys[ITEMS];
    int ids[ITEMS];
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const int loc = threadIdx.x + i * NT;
        const int row = row0 + loc;
        if (loc < CHUNK && row < ROWS) {
            keys[i] = OrderedFloat(logits[(blockIdx.x / NC) * ROWS + row]);
            ids[i] = rank * ROWS + row;
        } else {
            keys[i] = 0u;
            ids[i] = -1;
        }
    }
    __shared__ uint32_t s_key[K * NW];
    __shared__ int s_val[K * NW];
    SelectTopK<NT, ITEMS>(keys, ids, s_key, s_val, cand_val + off + cta * K,
                          cand_id + off + cta * K,
                          k);  // These verification logits are already normalized.

    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int failed = 0;
#pragma unroll
        for (int i = 0; i < 3; ++i) failed |= *reinterpret_cast<const int*>(upstream[i]);
        // Slot 4 aliases sampling status unless the CP arm supplies its own status.
        if (upstream[4] != upstream[3]) failed |= *reinterpret_cast<const int*>(upstream[4]);
        const int sampling_status = *reinterpret_cast<const int*>(upstream[3]);
        cand_id[kStatusOffset + ((gen - 1) & 1)] =
            sampling_status | (failed ? m8ctl::kStatusUpstreamFailed : 0);
    }
    {
        // The parallel merge writes from multiple threads. Publish the whole
        // CTA's candidate stores before its representative enters the release chain.
        __syncthreads();
        __shared__ int is_last;
        if (threadIdx.x == 0) is_last = (atom_add_acqrel_gpu(done_ctr, 1) == R * NC - 1);
        __syncthreads();
        if (is_last && threadIdx.x == 0) {
            // Acquire each CTA publication before the system-scope arrival.
            multimem_red_add_rel_sys(bar_mc, 1);  // ARRIVE (cumulative system-scope release)
            *done_ctr = 0;
        }
    }
}

// 64 reducer CTAs: one peer per token row.
__global__ __launch_bounds__(NT) void M8ReducerKernel(
    const long* __restrict__ peer_cand_val, const long* __restrict__ peer_cand_id,
    int* __restrict__ bar_local, float* __restrict__ out_val, int* __restrict__ out_id,
    int* __restrict__ status, const int* __restrict__ cur_gen,

    const uint64_t* __restrict__ sampling_buffer) {
    constexpr int PERPEER = NC * K;
    constexpr int SUBLEN = (PERPEER + M - 1) / M;  // candidates one reducer CTA scans
    constexpr int ITEMS = (SUBLEN + NT - 1) / NT;
    constexpr int NW = NT / 32;
    const int k = SamplingTopK(sampling_buffer);
    const int idx = blockIdx.x;  // 0 .. R*8*M-1
    const int row = idx / (8 * M);
    const int peer = (idx / M) % 8, sub = idx % M;
    const int gen = cur_gen[0];
    const long base = static_cast<long>((gen - 1) & 1) * R * PERPEER + row * PERPEER +
                      static_cast<long>(sub) * SUBLEN;

    if (threadIdx.x == 0) {  // wait for all 8 ranks' producer arrivals (signed spin, replay-safe)
        const int target = gen * kNumRanks;
        int spins = 0;
        while (static_cast<int>(ld_acq_sys_i32(bar_local) - target) < 0) {
            if (++spins > 2000000) {
                if (status) atomicOr(status, m8ctl::kStatusUpstreamFailed);
                break;
            }
            __nanosleep(64);
        }
    }
    __syncthreads();

    // The producer's existing release/acquire barrier also publishes this status.
    if (row == 0 && threadIdx.x == 0) {
        const int peer_status =
            reinterpret_cast<const int*>(peer_cand_id[peer])[kStatusOffset + ((gen - 1) & 1)];
        if (peer_status) atomicOr(status, peer_status);
    }

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
    {
        SelectTopK<NT, ITEMS>(keys, ids, s_key, s_val, out_val + idx * K, out_id + idx * K, k,
                              1.0f);
    }
}
__device__ __forceinline__ void M8PublishFailure(int* control, int status) {
    control[m8ctl::kCommitCount] = 0;
    control[m8ctl::kOutputCount] = 0;
    control[m8ctl::kAcceptedDraftCount] = 0;
    control[m8ctl::kFlags] = 0;
    control[m8ctl::kStatus] = status;
    control[m8ctl::kStepEpoch] = control[m8ctl::kStepEpoch] + 1;
}

__device__ __forceinline__ void M8PublishCommit(int* control, const int* emitted,
                                                int accepted_drafts, int final_token) {
    const int commit_count = accepted_drafts + 1;
#pragma unroll
    for (int row = 0; row < kM8Tail; ++row) {
        control[m8ctl::kEmittedIds + row] = row < commit_count ? emitted[row] : 0;
    }
    control[m8ctl::kAcceptedDraftCount] = accepted_drafts;
    control[m8ctl::kCommitCount] = commit_count;
    control[m8ctl::kOutputCount] = commit_count;
    control[m8ctl::kPendingTargetId] = final_token;
    control[m8ctl::kFlags] =
        m8ctl::kFlagVerified | (commit_count == kM8Tail ? m8ctl::kFlagFullAccept : 0);
    control[m8ctl::kStatus] = m8ctl::kStatusOk;
    control[m8ctl::kStepEpoch] = control[m8ctl::kStepEpoch] + 1;

    __threadfence();
    const int committed_len = control[m8ctl::kCommittedLen] + commit_count;
    control[m8ctl::kCommittedLen] = committed_len;  // published LAST
}

// Resolve eight sampled ids against [t0, t1..t7] and publish. Single thread.
// Sampled row j is compared with input t(j+1) for j = 0..6 until the first
// mismatch; accepted_draft_count in 0..7, commit_count = accepted + 1 in 1..8.
__device__ __forceinline__ void M8ResolvePrefixAndPublish(int* control,
                                                          const int (&target_ids)[kM8Tail],
                                                          int status) {
    if (status != m8ctl::kStatusOk) {
        M8PublishFailure(control, status);
        return;
    }
    int accepted = 0;
#pragma unroll
    for (int row = 0; row < kM8Tail - 1; ++row) {
        const int draft = control[m8ctl::kVerifyInputIds + row + 1];
        if (accepted == row && target_ids[row] == draft) {
            accepted = row + 1;
        }
    }
    int emitted[kM8Tail];
#pragma unroll
    for (int row = 0; row < kM8Tail; ++row) {
        emitted[row] =
            row < accepted ? control[m8ctl::kVerifyInputIds + row + 1] : target_ids[accepted];
    }
    M8PublishCommit(control, emitted, accepted, target_ids[accepted]);
}

__device__ __forceinline__ uint64_t MixSeed(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

__global__ __launch_bounds__(256) void M8SamplerKernel(const float* cv, const int* ci,
                                                       const uint64_t* sampling_buffer,
                                                       int* top_ids, float* top_probs,
                                                       int* target_ids, int* control, int* slot,
                                                       int* status, const int* forced_commit) {
    constexpr int NT = 256 / R;
    constexpr int GROUPS = R;
    constexpr int NW = NT / 32;
    constexpr int ITEMS = (8 * kMaxTopK + NT - 1) / NT;
    const int row = threadIdx.x / NT;
    const int local_row = row;
    const int tid = threadIdx.x % NT;
    const int k = SamplingTopK(sampling_buffer);
    const float t = SamplingTemperature(sampling_buffer);
    const float p = SamplingTopP(sampling_buffer);
    const int pos = control[m8ctl::kCommittedLen] + row;
    __shared__ uint32_t keys_s[GROUPS][kMaxTopK * NW];
    __shared__ int ids_s[GROUPS][kMaxTopK * NW];
    __shared__ float top_val[GROUPS][kMaxTopK];
    __shared__ int top_id[GROUPS][kMaxTopK];
    __shared__ int decisions[R];
    __shared__ int failure;
    __shared__ typename cub::WarpScan<float>::TempStorage scan[GROUPS];
    if (threadIdx.x == 0) {
        failure = *status;
        if (!isfinite(t) || t < 0 || !isfinite(p) || p <= 0 || p > 1 || sampling_buffer[2] < 1 ||
            sampling_buffer[2] > kMaxTopK)
            failure |= m8ctl::kStatusBadSamplingConfig;
    }
    if (tid == 0) decisions[row] = -1;
    if (tid < kMaxTopK) {
        top_ids[row * kMaxTopK + tid] = -1;
        top_probs[row * kMaxTopK + tid] = 0.0f;
    }
    __syncthreads();
    uint32_t keys[ITEMS];
    int ids[ITEMS];
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const int loc = tid + i * NT;
        const int index = row * 8 * kMaxTopK + loc;
        const bool valid = loc < 8 * kMaxTopK && loc % kMaxTopK < k && ci[index] >= 0;
        keys[i] = valid ? OrderedFloat(cv[index]) : 0;
        ids[i] = valid ? ci[index] : -1;
    }
    SelectTopK<NT, ITEMS>(keys, ids, keys_s[local_row], ids_s[local_row], top_val[local_row],
                          top_id[local_row], k);
    __syncthreads();
    if (tid == 0 && (!isfinite(top_val[local_row][0]) || top_id[local_row][0] < 0))
        atomicOr(&failure, m8ctl::kStatusBadResolution);
    __syncthreads();
    if (failure == 0) {
        if (tid < k) top_ids[row * kMaxTopK + tid] = top_id[local_row][tid];
        if (k == 1) {
            if (tid == 0) {
                decisions[row] = top_id[local_row][0];
                top_probs[row * kMaxTopK] = 1.0f;
            }
        } else if (tid < 32) {
            const int lane = tid;
            const float e =
                lane < k ? expf((top_val[local_row][lane] - top_val[local_row][0]) / t) : 0.0f;
            float prefix, total;
            cub::WarpScan<float>(scan[local_row]).InclusiveSum(e, prefix, total);
            const float cdf = prefix / total;
            const float probability = e / total;
            const int reverse = lane < k ? k - 1 - lane : lane;
            const float reversed = __shfl_sync(0xffffffffu, probability, reverse);
            float rev_prefix, unused;
            cub::WarpScan<float>(scan[local_row]).InclusiveSum(reversed, rev_prefix, unused);
            const float tail = __shfl_sync(0xffffffffu, rev_prefix, reverse);
            const bool keep = lane < k && (lane == 0 || tail > 1.0f - p);
            const unsigned keep_mask = __ballot_sync(0xffffffffu, keep);
            const int last = 31 - __clz(keep_mask);
            const float mass = __shfl_sync(0xffffffffu, cdf, last);
            if (lane < k) top_probs[row * kMaxTopK + lane] = keep ? probability / mass : 0.0f;
            const uint64_t seed = sampling_buffer[0];
            const uint64_t hash = MixSeed(seed ^ MixSeed(static_cast<uint32_t>(pos)));
            const float u = static_cast<float>(hash >> 40) * (1.0f / 16777216.0f);
            const unsigned hit = __ballot_sync(0xffffffffu, lane < k && cdf >= u * mass);
            if (lane == 0) {
                const int selected = __ffs(hit) - 1;
                if (selected < 0)
                    atomicOr(&failure, m8ctl::kStatusEmptySamplingMass);
                else
                    decisions[row] = top_id[local_row][selected];
            }
        }
    }
    __syncthreads();
    if (tid == 0) target_ids[row] = failure ? -1 : decisions[row];
    if (threadIdx.x == 0) {
        if (failure) atomicOr(status, failure);
        if (forced_commit && !failure) {
            const int count = forced_commit[0];
            if (count < 1 || count > R) {
                M8PublishFailure(control, m8ctl::kStatusBadCommitCount);
                return;
            }
            int emitted[R];
#pragma unroll
            for (int r = 0; r < R; ++r)
                emitted[r] =
                    r < count - 1 ? control[m8ctl::kVerifyInputIds + r + 1] : decisions[count - 1];
            M8PublishCommit(control, emitted, count - 1, decisions[count - 1]);
            return;
        }
        M8ResolvePrefixAndPublish(control, decisions, failure);
    }
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M8
using namespace nexrt::sm90::nex_n2_5_pro::M8;
extern "C" int NexN25ProM8SamplingProducerLaunch(const float* logits, float* cv, int* ci,
                                                 const int* bar, int* gen, uint64_t multicast,
                                                 int* done, int rank, const uint64_t* sampling,
                                                 const long* upstream, cudaStream_t stream) {
    M8ProducerKernel<<<R * NC, NT, 0, stream>>>(logits, cv, ci, bar, gen, multicast, done, rank,
                                                sampling, upstream);
    return static_cast<int>(cudaGetLastError());
}
extern "C" int NexN25ProM8SamplingReducerLaunch(const long* pv, const long* pi, int* bar, float* ov,
                                                int* oi, int* status, const int* gen,
                                                const uint64_t* sampling, cudaStream_t stream) {
    M8ReducerKernel<<<R * 8, NT, 0, stream>>>(pv, pi, bar, ov, oi, status, gen, sampling);
    return static_cast<int>(cudaGetLastError());
}
extern "C" int NexN25ProM8SamplerLaunch(const float* ov, const int* oi, const uint64_t* sampling,
                                        int* top_ids, float* top_probs, int* target_ids,
                                        int* control, int* slot, int* status, const int* forced,
                                        cudaStream_t stream) {
    M8SamplerKernel<<<1, 256, 0, stream>>>(ov, oi, sampling, top_ids, top_probs, target_ids,
                                           control, slot, status, forced);
    return static_cast<int>(cudaGetLastError());
}
