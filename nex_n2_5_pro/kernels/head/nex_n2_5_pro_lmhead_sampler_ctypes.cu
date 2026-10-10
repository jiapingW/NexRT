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

// Graph-stable greedy / top-k nucleus sampling. The same kernel and buffer
// addresses serve all requests; only the fixed sampling fields change.
#include "sampling.cuh"
#include <cub/warp/warp_scan.cuh>
namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY {
__device__ __forceinline__ unsigned long long splitmix64(unsigned long long z) {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
// Draw seed: per-request `sampling_seed` hashed with the token `position`. Fixed per request,
// distinct + decorrelated per position => reproducible across graph replays, one draw per generated token.
__device__ __forceinline__ float seeded_uniform(unsigned long long sampling_seed, int position) {
    const unsigned long long h = splitmix64(
        sampling_seed ^
        splitmix64(static_cast<unsigned long long>(static_cast<unsigned int>(position))));
    return static_cast<float>(h >> 40) * (1.0f / 16777216.0f);  // [0,1), 24-bit
}

template <int NT, int NCAND>
__global__ __launch_bounds__(NT) void SamplerKernel(const float* __restrict__ cand_val,
                                                    const int* __restrict__ cand_id,
                                                    const uint64_t* __restrict__ sampling_buffer,
                                                    const int* position,
                                                    int* __restrict__ out_token, int* seq_out,
                                                    int* __restrict__ status) {
    constexpr int NW = NT / 32;
    constexpr int ITEMS = (NCAND + NT - 1) / NT;
    const float temperature = SamplingTemperature(sampling_buffer);
    const float top_p = SamplingTopP(sampling_buffer);
    const int k = SamplingTopK(sampling_buffer);
    const int pos = position[blockIdx.x];
    if (!isfinite(temperature) || temperature < 0 || !isfinite(top_p) || top_p <= 0 || top_p > 1 ||
        sampling_buffer[2] < 1 || sampling_buffer[2] > kMaxTopK) {
        if (threadIdx.x == 0) {
            atomicCAS(status, 0, 3);
            out_token[blockIdx.x] = -1;
            if (seq_out) seq_out[0] = pos + 1;
        }
        return;
    }
    uint32_t keys[ITEMS];
    int ids[ITEMS];
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const int loc = threadIdx.x + i * NT;
        if (loc < NCAND && loc % kMaxTopK < k && cand_id[loc] >= 0) {
            keys[i] = OrderedFloat(cand_val[loc]);
            ids[i] = cand_id[loc];
        } else {
            keys[i] = 0;
            ids[i] = -1;
        }
    }
    __shared__ uint32_t s_key[kMaxTopK * NW];
    __shared__ int s_id[kMaxTopK * NW];
    __shared__ float top_val[kMaxTopK];
    __shared__ int top_id[kMaxTopK];
    __shared__ typename cub::WarpScan<float>::TempStorage scan_tmp;
    SelectTopK<NT, ITEMS>(keys, ids, s_key, s_id, top_val, top_id, k);
    __syncthreads();
    if (*status || !isfinite(top_val[0]) || top_id[0] < 0) {
        if (threadIdx.x == 0) {
            atomicCAS(status, 0, 2);
            out_token[blockIdx.x] = -1;
            if (seq_out) seq_out[0] = pos + 1;
        }
        return;
    }
    if (k == 1) {
        if (threadIdx.x == 0) {
            out_token[blockIdx.x] = top_id[0];
            if (seq_out) seq_out[0] = pos + 1;
        }
        return;
    }
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        // Subtract before scaling: even a small positive temperature cannot
        // overflow the maximum logit's softmax numerator.
        const float e = lane < k ? expf((top_val[lane] - top_val[0]) / temperature) : 0.0f;
        float prefix, total;
        cub::WarpScan<float>(scan_tmp).InclusiveSum(e, prefix, total);
        const float cdf = prefix / total;
        const float probability = e / total;
        const int reverse = lane < k ? k - 1 - lane : lane;
        const float reversed_p = __shfl_sync(0xffffffffu, probability, reverse);
        float reversed_prefix, unused;
        cub::WarpScan<float>(scan_tmp).InclusiveSum(reversed_p, reversed_prefix, unused);
        const float tail = __shfl_sync(0xffffffffu, reversed_prefix, reverse);
        const bool keep = lane < k && (lane == 0 || tail > 1.0f - top_p);
        const unsigned keep_mask = __ballot_sync(0xffffffffu, keep);
        const int last = 31 - __clz(keep_mask);
        const float mass = __shfl_sync(0xffffffffu, cdf, last);
        const uint64_t seed = sampling_buffer[0];
        const float threshold = seeded_uniform(seed, pos) * mass;
        const unsigned hit = __ballot_sync(0xffffffffu, lane < k && cdf >= threshold);
        if (lane == 0) {
            const int winner = __ffs(hit) - 1;
            if (winner < 0) atomicCAS(status, 0, 2);
            out_token[blockIdx.x] = winner < 0 ? -1 : top_id[winner];
            if (seq_out) seq_out[0] = pos + 1;
        }
    }
}
}  // namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
using namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY;
extern "C" {
int NexN25ProLmHeadSamplerLaunch(const void* cv, const void* ci, const void* sampling_buffer,
                                 const void* position, void* out_token, int batch, void* status,
                                 void* stream) {
    SamplerKernel<256, 8 * kMaxTopK><<<batch, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(cv), static_cast<const int*>(ci),
        static_cast<const uint64_t*>(sampling_buffer), static_cast<const int*>(position),
        static_cast<int*>(out_token), nullptr, static_cast<int*>(status));
    return static_cast<int>(cudaGetLastError());
}
int NexN25ProLmHeadSamplerSlotLaunch(const void* cv, const void* ci, const void* sampling_buffer,
                                     void* slot, void* status, void* stream) {
    auto* sl = static_cast<int*>(slot);
    SamplerKernel<256, 8 * kMaxTopK><<<1, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(cv), static_cast<const int*>(ci),
        static_cast<const uint64_t*>(sampling_buffer), sl + 1, sl, sl + 1,
        static_cast<int*>(status));
    return static_cast<int>(cudaGetLastError());
}
const char* NexN25ProLmHeadSamplerCudaErrorString(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
}
