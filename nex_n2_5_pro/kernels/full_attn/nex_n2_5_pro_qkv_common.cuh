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
constexpr int kHeadDim = 256;
constexpr int kNHeadsLocal = 4;
constexpr int kQGate = 2;
constexpr int kQRows = kNHeadsLocal * kHeadDim * kQGate;
constexpr int kKRows = kHeadDim;
constexpr int kVRows = kHeadDim;
constexpr int kTotalRows = kQRows + kKRows + kVRows;
constexpr int kQEnd = kQRows;
constexpr int kKEnd = kQRows + kKRows;
constexpr int kScaleBlocksK = kHid / 128;
constexpr int kTilesK = kHid / 512;

__device__ __forceinline__ void scatter_row(int row, float accumulator,
                                            __nv_bfloat16* __restrict__ query_output,
                                            __nv_bfloat16* __restrict__ gate_output,
                                            __nv_bfloat16* __restrict__ key_cache,
                                            __nv_bfloat16* __restrict__ value_cache, int position) {
    const __nv_bfloat16 value = __float2bfloat16(accumulator);
    if (row < kQEnd) {
        const int head = row >> 9;
        const int within_head = row & 511;
        if (within_head < kHeadDim) {
            query_output[head * kHeadDim + within_head] = value;
        } else {
            gate_output[head * kHeadDim + within_head - kHeadDim] = value;
        }
    } else if (row < kKEnd) {
        key_cache[static_cast<size_t>(position) * kHeadDim + row - kQEnd] = value;
    } else {
        value_cache[static_cast<size_t>(position) * kHeadDim + row - kKEnd] = value;
    }
}

}  // namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY