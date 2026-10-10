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

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY bs1
#endif

namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY {
constexpr int kMaxTopK = 28;
constexpr int kSamplingWords = 8;
// Fixed uint64_t words: [0] seed; [1] float32 temperature/top_p; [2] top_k;
// [3..7] reserved. Values change between requests; field offsets never change.
__device__ __forceinline__ float SamplingTemperature(const uint64_t* sampling_buffer) {
    return __uint_as_float(static_cast<uint32_t>(sampling_buffer[1]));
}
__device__ __forceinline__ float SamplingTopP(const uint64_t* sampling_buffer) {
    return __uint_as_float(static_cast<uint32_t>(sampling_buffer[1] >> 32));
}
__device__ __forceinline__ int SamplingTopK(const uint64_t* sampling_buffer) {
    if (SamplingTemperature(sampling_buffer) == 0.0f) return 1;
    // Host validation enforces the range; clamp here to keep malformed buffers bounded.
    return static_cast<int>(min(max(sampling_buffer[2], uint64_t{1}), uint64_t{kMaxTopK}));
}
__device__ __forceinline__ uint32_t OrderedFloat(float x) {
    const uint32_t bits = __float_as_uint(x);
    const uint32_t magnitude = bits & 0x7fffffffu;
    const uint32_t ordered = bits ^ ((bits >> 31) ? 0xffffffffu : 0x80000000u);
    // Both NaN signs must reach the sampler's invalid-logit check. Canonical
    // signed zeros keep the lower-token-ID tie break independent of sign.
    return magnitude > 0x7f800000u ? 0xffffffffu : (magnitude == 0 ? 0x80000000u : ordered);
}
__device__ __forceinline__ float OrderedFloatToFloat(uint32_t value) {
    return __uint_as_float(value ^ ((value >> 31) ? 0x80000000u : 0xffffffffu));
}
__device__ __forceinline__ uint32_t ReduxMax32(uint32_t value) {
    uint32_t out;
    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(out) : "r"(value));
    return out;
}
// Same bounded selection in all three stages. Equal logits select the lower
// token ID. Inactive output slots are cleared so changing top_k cannot expose
// candidates retained from an earlier request.
template <int NT, int ITEMS>
__device__ __forceinline__ void SelectTopK(uint32_t (&keys)[ITEMS], int (&ids)[ITEMS],
                                           uint32_t* s_key, int* s_id, float* out_val, int* out_id,
                                           int k, float scale = 1.0f) {
    constexpr int NW = NT / 32;
    const int tid = threadIdx.x % NT;
    const int lane = tid & 31, warp = tid >> 5;
    if (tid < kMaxTopK) {
        out_val[tid] = -__int_as_float(0x7f800000);
        out_id[tid] = -1;
    }
    for (int r = 0; r < k; ++r) {
        uint32_t local = 0;
#pragma unroll
        for (int i = 0; i < ITEMS; ++i) local = max(local, keys[i]);
        const uint32_t best = ReduxMax32(local);
        int local_id = 2147483647;
#pragma unroll
        for (int i = 0; i < ITEMS; ++i)
            if (best != 0 && keys[i] == best) local_id = min(local_id, ids[i]);
        const int winner = __reduce_min_sync(0xffffffffu, local_id);
        if (lane == 0) {
            s_key[r * NW + warp] = best;
            s_id[r * NW + warp] = best ? winner : -1;
        }
#pragma unroll
        for (int i = 0; i < ITEMS; ++i)
            if (keys[i] == best && ids[i] == winner) keys[i] = 0;
    }
    __syncthreads();
    // Rank each warp winner against the other sorted warp lists in parallel.
    // At most 8*28 items fit in one block; no single warp serially merges k winners.
    const int item = tid;
    if (item < NW * k && s_key[item] != 0) {
        const uint32_t key = s_key[item];
        const int id = s_id[item];
        const int source_warp = item % NW;
        int order = item / NW;
        for (int other = 0; other < NW && order < k; ++other) {
            if (other == source_warp) continue;
            int lo = 0, hi = k;
            while (lo < hi) {
                const int mid = (lo + hi) / 2;
                const uint32_t other_key = s_key[mid * NW + other];
                const int other_id = s_id[mid * NW + other];
                if (other_key > key || (other_key == key && other_id < id))
                    lo = mid + 1;
                else
                    hi = mid;
            }
            order += lo;
        }
        if (order < k) {
            out_val[order] = scale * OrderedFloatToFloat(key);
            out_id[order] = id;
        }
    }
}
}  // namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
