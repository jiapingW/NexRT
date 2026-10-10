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

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY {
constexpr int kDim = 4096;
constexpr int kTopK = 10;
constexpr int kRowsPerExpert = 128;
constexpr int kRoutedRows = kTopK * kRowsPerExpert;
constexpr int kAllRows = kRoutedRows + kRowsPerExpert;
constexpr int kScaleBlocks = kDim / 128;
constexpr int kAllScales = kScaleBlocks * 2;

__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, o);
    }
    return v;
}

__device__ __forceinline__ void resolve_row(
    int item, const int32_t* __restrict__ topk_ids, const __nv_fp8_e4m3* __restrict__ W1,
    const __nv_fp8_e4m3* __restrict__ W3, const float* __restrict__ w1_w3_scales,
    const __nv_fp8_e4m3* __restrict__ shared_W1, const __nv_fp8_e4m3* __restrict__ shared_W3,
    const float* __restrict__ shared_w1_w3_scales, const __nv_fp8_e4m3** row_w1,
    const __nv_fp8_e4m3** row_w3, const float** row_s1, const float** row_s3, float* route_weight) {
    if (item < kRoutedRows) {
        const int logical_expert = item / kRowsPerExpert;
        const int row = item - logical_expert * kRowsPerExpert;
        const int expert = topk_ids[logical_expert];
        const size_t expert_base = static_cast<size_t>(expert) * kRowsPerExpert * kDim;
        const int scale_base = expert * kAllScales;
        *row_w1 = W1 + expert_base + static_cast<size_t>(row) * kDim;
        *row_w3 = W3 + expert_base + static_cast<size_t>(row) * kDim;
        *row_s1 = w1_w3_scales + scale_base;
        *row_s3 = w1_w3_scales + scale_base + kScaleBlocks;
        *route_weight = -static_cast<float>(logical_expert + 1);
    } else {
        const int row = item - kRoutedRows;
        *row_w1 = shared_W1 + static_cast<size_t>(row) * kDim;
        *row_w3 = shared_W3 + static_cast<size_t>(row) * kDim;
        *row_s1 = shared_w1_w3_scales;
        *row_s3 = shared_w1_w3_scales + kScaleBlocks;
        *route_weight = 1.0f;
    }
}

__device__ __forceinline__ void dot_pair_xbf16_fp8_v4(const __nv_bfloat16* __restrict__ x,
                                                      const __nv_fp8_e4m3* __restrict__ w1,
                                                      const __nv_fp8_e4m3* __restrict__ w3,
                                                      const float* __restrict__ s1,
                                                      const float* __restrict__ s3, int k0, int k1,
                                                      int lane, float* gate_out, float* up_out) {
    float gate = 0.0f;
    float up = 0.0f;
    const int kb0 = k0 >> 7;
    const int kb1 = k1 >> 7;
#pragma unroll
    for (int kb = kb0; kb < kb1; ++kb) {
        const int k = (kb << 7) + lane * 4;
        const float scale1 = s1[kb];
        const float scale3 = s3[kb];
        const __nv_fp8x4_e4m3 w1_4 = *reinterpret_cast<const __nv_fp8x4_e4m3*>(w1 + k);
        const __nv_fp8x4_e4m3 w3_4 = *reinterpret_cast<const __nv_fp8x4_e4m3*>(w3 + k);
        const __nv_bfloat162 x01 = *reinterpret_cast<const __nv_bfloat162*>(x + k);
        const __nv_bfloat162 x23 = *reinterpret_cast<const __nv_bfloat162*>(x + k + 2);
        const float4 w1f = float4(w1_4);
        const float4 w3f = float4(w3_4);
        const float2 x01f = __bfloat1622float2(x01);
        const float2 x23f = __bfloat1622float2(x23);
        gate += scale1 * (x01f.x * w1f.x + x01f.y * w1f.y + x23f.x * w1f.z + x23f.y * w1f.w);
        up += scale3 * (x01f.x * w3f.x + x01f.y * w3f.y + x23f.x * w3f.z + x23f.y * w3f.w);
    }
    *gate_out = warp_sum(gate);
    *up_out = warp_sum(up);
}
// Split-K=4 is the measured H100 production path.
#ifndef NEXRT_MTP_RUNTIME_ONLY
template <int kNumWarps>
__global__
__launch_bounds__(kNumWarps * 32) void L1K2PairPacked64UnweightedXbf16V4SplitKRmsScaleKernel(
    const int32_t* __restrict__ packed_topk64, const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ rms_scale, const __nv_fp8_e4m3* __restrict__ W1_prefused,
    const __nv_fp8_e4m3* __restrict__ W3_prefused, const float* __restrict__ w1_w3_scales,
    const __nv_fp8_e4m3* __restrict__ shared_W1_prefused,
    const __nv_fp8_e4m3* __restrict__ shared_W3_prefused,
    const float* __restrict__ shared_w1_w3_scales, float* __restrict__ raw_h) {
    static_assert(kNumWarps == 4, "the production L1 kernel uses four warps");
    static_assert(kDim % kNumWarps == 0, "K must divide evenly across warps");
    constexpr int kChunk = kDim / kNumWarps;
    static_assert(kChunk % 128 == 0, "warp K ranges must align to FP8 scale blocks");

    __shared__ float gate_dots[kNumWarps];
    __shared__ float up_dots[kNumWarps];

    const int role = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int item = blockIdx.x;
    if (item >= kAllRows) {
        return;
    }

    const __nv_fp8_e4m3* row_w1;
    const __nv_fp8_e4m3* row_w3;
    const float* row_s1;
    const float* row_s3;
    float route_marker;
    resolve_row(item, packed_topk64, W1_prefused, W3_prefused, w1_w3_scales, shared_W1_prefused,
                shared_W3_prefused, shared_w1_w3_scales, &row_w1, &row_w3, &row_s1, &row_s3,
                &route_marker);

    float gate = 0.0f;
    float up = 0.0f;
    const int k0 = role * kChunk;
    const int k1 = k0 + kChunk;
    dot_pair_xbf16_fp8_v4(x, row_w1, row_w3, row_s1, row_s3, k0, k1, lane, &gate, &up);

    const float inv_rms = rms_scale[0];
    if (lane == 0) {
        gate_dots[role] = gate * inv_rms;
        up_dots[role] = up * inv_rms;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // Pairwise tree gives the retained split-4 path deterministic grouping.
        const float gate01 = gate_dots[0] + gate_dots[1];
        const float up01 = up_dots[0] + up_dots[1];
        const float gate23 = gate_dots[2] + gate_dots[3];
        const float up23 = up_dots[2] + up_dots[3];
        const float gate_sum = gate01 + gate23;
        const float up_sum = up01 + up23;
        float rw = 1.0f;
        if (route_marker > 0.0f) {
            rw = __int_as_float(packed_topk64[63]);
        }
        raw_h[item] = silu(gate_sum) * up_sum * rw;
    }
}
#endif
// RAW-weight variant (weights SHARED with sglang-prefill): reads the checkpoint/sglang layout
// directly -- stacked w13 [E, 2*kRowsPerExpert, kDim] with rows 0..127 = gate_proj and
// 128..255 = up_proj per expert (sglang FusedMoE w13_weight), ORIGINAL checkpoint block scales
// in the same fused [E, 64] view ([E,2,32] flat: w1 blocks then w3 blocks), and the
// PRE-NORMALIZED activation xn = x * rms * (1 + post_ln) written by the top-k Xn kernel. No
// gamma fold, no requant, no per-output rms multiply. Same split-K=4 geometry as the default.
__device__ __forceinline__ void resolve_row_w13(
    int item, const int32_t* __restrict__ topk_ids, const __nv_fp8_e4m3* __restrict__ W13,
    const float* __restrict__ w13_scales, const __nv_fp8_e4m3* __restrict__ shared_gate_up,
    const float* __restrict__ shared_scales, const __nv_fp8_e4m3** row_w1,
    const __nv_fp8_e4m3** row_w3, const float** row_s1, const float** row_s3, float* route_weight) {
    constexpr size_t kExpertStride = static_cast<size_t>(2 * kRowsPerExpert) * kDim;
    constexpr size_t kUpOffset = static_cast<size_t>(kRowsPerExpert) * kDim;
    if (item < kRoutedRows) {
        const int logical_expert = item / kRowsPerExpert;
        const int row = item - logical_expert * kRowsPerExpert;
        const int expert = topk_ids[logical_expert];
        const __nv_fp8_e4m3* base =
            W13 + static_cast<size_t>(expert) * kExpertStride + static_cast<size_t>(row) * kDim;
        const int scale_base = expert * kAllScales;
        *row_w1 = base;
        *row_w3 = base + kUpOffset;
        *row_s1 = w13_scales + scale_base;
        *row_s3 = w13_scales + scale_base + kScaleBlocks;
        *route_weight = -static_cast<float>(logical_expert + 1);
    } else {
        const int row = item - kRoutedRows;
        *row_w1 = shared_gate_up + static_cast<size_t>(row) * kDim;
        *row_w3 = shared_gate_up + kUpOffset + static_cast<size_t>(row) * kDim;
        *row_s1 = shared_scales;
        *row_s3 = shared_scales + kScaleBlocks;
        *route_weight = 1.0f;
    }
}

template <int kNumWarps>
__global__
__launch_bounds__(kNumWarps * 32) void L1K2PairPacked64UnweightedXbf16V4SplitKW13XnKernel(
    const int32_t* __restrict__ packed_topk64, const __nv_bfloat16* __restrict__ xn,
    const __nv_fp8_e4m3* __restrict__ W13, const float* __restrict__ w13_scales,
    const __nv_fp8_e4m3* __restrict__ shared_gate_up, const float* __restrict__ shared_scales,
    float* __restrict__ raw_h) {
    static_assert(kNumWarps == 4, "the raw-weight L1 kernel keeps the measured split-4 geometry");
    static_assert(kDim % kNumWarps == 0, "K must divide evenly across warps");
    constexpr int kChunk = kDim / kNumWarps;
    static_assert(kChunk % 128 == 0, "warp K ranges must align to FP8 scale blocks");

    __shared__ float gate_dots[kNumWarps];
    __shared__ float up_dots[kNumWarps];

    const int role = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int item = blockIdx.x;
    if (item >= kAllRows) {
        return;
    }

    const __nv_fp8_e4m3* row_w1;
    const __nv_fp8_e4m3* row_w3;
    const float* row_s1;
    const float* row_s3;
    float route_marker;
    resolve_row_w13(item, packed_topk64, W13, w13_scales, shared_gate_up, shared_scales, &row_w1,
                    &row_w3, &row_s1, &row_s3, &route_marker);

    float gate = 0.0f;
    float up = 0.0f;
    const int k0 = role * kChunk;
    const int k1 = k0 + kChunk;
    dot_pair_xbf16_fp8_v4(xn, row_w1, row_w3, row_s1, row_s3, k0, k1, lane, &gate, &up);

    if (lane == 0) {
        gate_dots[role] = gate;  // xn already carries rms * gamma
        up_dots[role] = up;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // Same pairwise tree as the folded split-4 path (deterministic grouping).
        const float gate01 = gate_dots[0] + gate_dots[1];
        const float up01 = up_dots[0] + up_dots[1];
        const float gate23 = gate_dots[2] + gate_dots[3];
        const float up23 = up_dots[2] + up_dots[3];
        const float gate_sum = gate01 + gate23;
        const float up_sum = up01 + up23;
        float rw = 1.0f;
        if (route_marker > 0.0f) {
            rw = __int_as_float(packed_topk64[63]);
        }
        raw_h[item] = silu(gate_sum) * up_sum * rw;
    }
}
}  // namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
using namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY;
#ifndef NEXRT_MTP_RUNTIME_ONLY
extern "C" int NexN25ProL1Launch(const int32_t* packed_topk64, const void* x,
                                 const float* rms_scale, const void* W1_prefused,
                                 const void* W3_prefused, const float* w1_w3_scales,
                                 const void* shared_W1_prefused, const void* shared_W3_prefused,
                                 const float* shared_w1_w3_scales, float* raw_h, void* stream) {
    L1K2PairPacked64UnweightedXbf16V4SplitKRmsScaleKernel<4>
        <<<kAllRows, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            packed_topk64, reinterpret_cast<const __nv_bfloat16*>(x), rms_scale,
            reinterpret_cast<const __nv_fp8_e4m3*>(W1_prefused),
            reinterpret_cast<const __nv_fp8_e4m3*>(W3_prefused), w1_w3_scales,
            reinterpret_cast<const __nv_fp8_e4m3*>(shared_W1_prefused),
            reinterpret_cast<const __nv_fp8_e4m3*>(shared_W3_prefused), shared_w1_w3_scales, raw_h);
    return static_cast<int>(cudaGetLastError());
}
#endif

// Bridge / raw-weight L1: stacked w13 + original checkpoint scales + pre-normalized xn.
extern "C" int NexN25ProL1W13XnLaunch(const int32_t* packed_topk64, const void* xn, const void* W13,
                                      const float* w13_scales, const void* shared_gate_up,
                                      const float* shared_scales, float* raw_h, void* stream) {
    L1K2PairPacked64UnweightedXbf16V4SplitKW13XnKernel<4>
        <<<kAllRows, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            packed_topk64, reinterpret_cast<const __nv_bfloat16*>(xn),
            reinterpret_cast<const __nv_fp8_e4m3*>(W13), w13_scales,
            reinterpret_cast<const __nv_fp8_e4m3*>(shared_gate_up), shared_scales, raw_h);
    return static_cast<int>(cudaGetLastError());
}

#ifndef NEXRT_MTP_RUNTIME_ONLY
extern "C" const char* nex_n2_5_pro_l1_cuda_error_string(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
#endif
