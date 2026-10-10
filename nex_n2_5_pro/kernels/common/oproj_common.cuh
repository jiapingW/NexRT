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
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY {
constexpr int kDim = 4096;
constexpr int kLocal = 1024;
constexpr int kHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kScaleCols = kLocal / 128;
constexpr int kMaxRanks = 8;
constexpr int kMaxSpins = 1 << 30;
constexpr float kRmsEps = 1.0e-6f;

static_assert(kLocal == kHeads * kHeadDim, "linear-attention local geometry must remain 8 x 128");

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float sigmoid_f32(float value) { return 1.0f / (1.0f + __expf(-value)); }

__device__ __forceinline__ float silu_f32(float value) { return value * sigmoid_f32(value); }

__device__ __forceinline__ float4 add4(float4 left, float4 right) {
    left.x += right.x;
    left.y += right.y;
    left.z += right.z;
    left.w += right.w;
    return left;
}

__device__ __forceinline__ float dot_x4_wword(float4 input, uint32_t word) {
    __nv_fp8x4_e4m3 packed;
    packed.__x = word;
    const float4 weight = float4(packed);
    return input.x * weight.x + input.y * weight.y + input.z * weight.z + input.w * weight.w;
}

__device__ __forceinline__ int ld_acq_sys_i32(const int* pointer) {
    int value;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(value) : "l"(pointer) : "memory");
    return value;
}

__device__ __forceinline__ void multimem_red_add_rel_sys(unsigned long long address, int value) {
    asm volatile("multimem.red.release.sys.global.add.s32 [%0], %1;" ::"l"(address), "r"(value)
                 : "memory");
}

__device__ __forceinline__ void multimem_st_v4_f32(unsigned long long address, float4 value) {
    asm volatile("multimem.st.weak.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(address),
                 "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w)
                 : "memory");
}

}  // namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
