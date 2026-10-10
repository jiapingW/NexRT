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

// M=4 MoE L1 with the ROUTE WEIGHT FOLDED into rawh (fold path).
// Standard L1 W13 GEMV body except the epilogue: routed rows multiply by their softmax
// route weight (packed[10+slot]) instead of staying unweighted, so the fold-aware L2
// drops its per-output-row tw multiply (40*4096 -> only 40*128 multiplies, applied
// here once per rawh element). Shared rows keep packed[63] as their weight.
// NUMERICS: the weight is applied before the L2 K-dot instead of after -- bounded
// rounding difference, gated against an f64 reference.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M4
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace nexrt::sm90::nex_n2_5_pro::M4 {
constexpr int kDim = 4096;
constexpr int kTopK = 10;
constexpr int kRowsPerExpert = 128;
constexpr int kRoutedRows = kTopK * kRowsPerExpert;
constexpr int kAllRows = kRoutedRows + kRowsPerExpert;
constexpr int kScaleBlocks = kDim / 128;
constexpr int kAllScales = kScaleBlocks * 2;
constexpr int kPackedWords = 64;
constexpr int kRows = 4;
constexpr int kNumWarps = 4;

__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, o);
    }
    return v;
}

__global__ __launch_bounds__(kNumWarps * 32) void M4L1FoldWKernel(
    const int32_t* __restrict__ packed4, const __nv_bfloat16* __restrict__ xn4,
    const __nv_fp8_e4m3* __restrict__ W13, const float* __restrict__ w13_scales,
    const __nv_fp8_e4m3* __restrict__ shared_gate_up, const float* __restrict__ shared_scales,
    float* __restrict__ rawh4) {
    constexpr int kChunk = kDim / kNumWarps;
    constexpr size_t kExpertStride = static_cast<size_t>(2 * kRowsPerExpert) * kDim;
    constexpr size_t kUpOffset = static_cast<size_t>(kRowsPerExpert) * kDim;
    __shared__ float gate_dots[kNumWarps];
    __shared__ float up_dots[kNumWarps];

    const int role = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int b = blockIdx.x;
    const int row = b / kAllRows;
    const int item = b - row * kAllRows;

    const int32_t* packed_topk64 = packed4 + static_cast<size_t>(row) * kPackedWords;
    const __nv_bfloat16* xn = xn4 + static_cast<size_t>(row) * kDim;
    float* raw_h = rawh4 + static_cast<size_t>(row) * kAllRows;

    const __nv_fp8_e4m3* row_w1;
    const __nv_fp8_e4m3* row_w3;
    const float* row_s1;
    const float* row_s3;
    float rw;
    if (item < kRoutedRows) {
        const int logical_expert = item / kRowsPerExpert;
        const int r = item - logical_expert * kRowsPerExpert;
        const int expert = packed_topk64[logical_expert];
        const __nv_fp8_e4m3* base =
            W13 + static_cast<size_t>(expert) * kExpertStride + static_cast<size_t>(r) * kDim;
        row_w1 = base;
        row_w3 = base + kUpOffset;
        row_s1 = w13_scales + expert * kAllScales;
        row_s3 = row_s1 + kScaleBlocks;
        rw = __int_as_float(packed_topk64[kTopK + logical_expert]);  // THE FOLD
    } else {
        const int r = item - kRoutedRows;
        row_w1 = shared_gate_up + static_cast<size_t>(r) * kDim;
        row_w3 = shared_gate_up + kUpOffset + static_cast<size_t>(r) * kDim;
        row_s1 = shared_scales;
        row_s3 = shared_scales + kScaleBlocks;
        rw = __int_as_float(packed_topk64[63]);
    }

    float gate = 0.0f;
    float up = 0.0f;
    const int kb0 = (role * kChunk) >> 7;
    const int kb1 = kb0 + (kChunk >> 7);
#pragma unroll
    for (int kb = kb0; kb < kb1; ++kb) {
        const int k = (kb << 7) + lane * 4;
        const float scale1 = row_s1[kb];
        const float scale3 = row_s3[kb];
        const __nv_fp8x4_e4m3 w1_4 = *reinterpret_cast<const __nv_fp8x4_e4m3*>(row_w1 + k);
        const __nv_fp8x4_e4m3 w3_4 = *reinterpret_cast<const __nv_fp8x4_e4m3*>(row_w3 + k);
        const __nv_bfloat162 x01 = *reinterpret_cast<const __nv_bfloat162*>(xn + k);
        const __nv_bfloat162 x23 = *reinterpret_cast<const __nv_bfloat162*>(xn + k + 2);
        const float4 w1f = float4(w1_4);
        const float4 w3f = float4(w3_4);
        const float2 x01f = __bfloat1622float2(x01);
        const float2 x23f = __bfloat1622float2(x23);
        gate += scale1 * (x01f.x * w1f.x + x01f.y * w1f.y + x23f.x * w1f.z + x23f.y * w1f.w);
        up += scale3 * (x01f.x * w3f.x + x01f.y * w3f.y + x23f.x * w3f.z + x23f.y * w3f.w);
    }
    gate = warp_sum(gate);
    up = warp_sum(up);
    if (lane == 0) {
        gate_dots[role] = gate;
        up_dots[role] = up;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const float gate01 = gate_dots[0] + gate_dots[1];
        const float up01 = up_dots[0] + up_dots[1];
        const float gate23 = gate_dots[2] + gate_dots[3];
        const float up23 = up_dots[2] + up_dots[3];
        raw_h[item] = silu(gate01 + gate23) * (up01 + up23) * rw;
    }
}

extern "C" int NexN25ProM4L1FoldWLaunch(const void* packed4, const void* xn4, const void* W13,
                                        const void* w13_scales, const void* shared_gate_up,
                                        const void* shared_scales, void* rawh4, void* stream) {
    M4L1FoldWKernel<<<kRows * kAllRows, kNumWarps * 32, 0,
                      reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const int32_t*>(packed4), reinterpret_cast<const __nv_bfloat16*>(xn4),
        reinterpret_cast<const __nv_fp8_e4m3*>(W13), reinterpret_cast<const float*>(w13_scales),
        reinterpret_cast<const __nv_fp8_e4m3*>(shared_gate_up),
        reinterpret_cast<const float*>(shared_scales), reinterpret_cast<float*>(rawh4));
    return static_cast<int>(cudaGetLastError());
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M4