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

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY bs1
#endif

#pragma once

#include <cuda_bf16.h>

#include <cstddef>

namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY {
constexpr int kHid = 4096;
constexpr int kQkvRows = 1536;
constexpr int kZRows = 1024;
constexpr int kFp8Rows = kQkvRows + kZRows;
constexpr int kBRows = 8;
constexpr int kARows = 8;
constexpr int kScaleBlocksK = kHid / 128;
constexpr int kTilesK = kHid / 512;
constexpr float kSoftplusThreshold = 20.0f;

static_assert(kFp8Rows % 2 == 0, "FP8 rows must pair into whole groups");

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float sigmoid_f32(float value) { return 1.0f / (1.0f + __expf(-value)); }

__device__ __forceinline__ float silu_f32(float value) { return value * sigmoid_f32(value); }

__device__ __forceinline__ float softplus_f32(float value) {
    return value > kSoftplusThreshold ? value : log1pf(expf(value));
}

__device__ __forceinline__ void unpack_bf16x16(int4 raw0, int4 raw1, float (&values)[16]) {
    const __nv_bfloat162* pairs0 = reinterpret_cast<const __nv_bfloat162*>(&raw0);
    const __nv_bfloat162* pairs1 = reinterpret_cast<const __nv_bfloat162*>(&raw1);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const float2 value = __bfloat1622float2(pairs0[pair]);
        values[2 * pair] = value.x;
        values[2 * pair + 1] = value.y;
    }
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const float2 value = __bfloat1622float2(pairs1[pair]);
        values[8 + 2 * pair] = value.x;
        values[8 + 2 * pair + 1] = value.y;
    }
}

__device__ __forceinline__ void z_epilogue(int row, float accumulator,
                                           __nv_bfloat16* __restrict__ output) {
    output[row - kQkvRows] = __float2bfloat16(accumulator);
}

}  // namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY