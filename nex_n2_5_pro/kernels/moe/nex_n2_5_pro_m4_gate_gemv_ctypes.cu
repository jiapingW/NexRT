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

// C002: M=4 MoE gate GEMV -- FOUR raw residual rows -> FOUR raw gate-logit rows + FOUR rms scales
// in ONE pass over the gamma-folded gate weights.
//
//   logits4[j][row] = bf16( x4[j] . gate_w[row] )        j = 0..3, row = 0..512 (512 = shared gate)
//   rms_scale4[j]   = rsqrtf( mean(x4[j]^2) + eps )
//
// Exact M1 semantics (reference gate GEMV implementation): RAW prefused dots
// are stored -- the unchanged 1-CTA top-k kernel applies the positive rms scale for softmax
// weights and the shared sigmoid, and selected expert ids are scale-invariant. Per output row the
// per-thread pair striding and block reduction match the M1 kernel exactly, so each logit row is
// the M1 result bit-for-bit; the M=4 point is that each loaded gate-weight pair is reused across
// FOUR accumulators (one weight pass for four rows) instead of four separate GEMV passes over the
// ~4.2 MB gate matrix. CTAs 0..3 additionally accumulate the M1 rms sum for x row j = blockIdx.x
// with the same striding/reduction order as M1's CTA0.
//
// Layout: x4 [4, 4096] bf16 (raw residual rows); gate_w [528, 4096] bf16 gamma-folded (513 active
// rows, padded physical rows never touched); logits4 [4, 528] bf16 row-major (active 513 written);
// rms_scale4 float[4].

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace nexrt::sm90::nex_n2_5_pro::M4 {
constexpr int kDim = 4096;
constexpr int kDimPairs = kDim / 2;
constexpr int kGateRowsActive = 513;
constexpr int kGateRowsPhysical = 528;
constexpr int kM = 4;

__device__ __forceinline__ float WarpSum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

template <int kNumThreads>
__device__ __forceinline__ float BlockSum(float v) {
    constexpr int kNumWarps = kNumThreads / 32;
    __shared__ float warp_sums[kNumWarps];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = WarpSum(v);
    if (lane == 0) {
        warp_sums[warp] = v;
    }
    __syncthreads();
    v = (threadIdx.x < kNumWarps) ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        v = WarpSum(v);
    }
    return v;
}

template <int kNumThreads>
__global__ __launch_bounds__(kNumThreads) void M4GateGemvRawAndRmsKernel(
    const __nv_bfloat16* __restrict__ x4, const __nv_bfloat16* __restrict__ gate_w,
    __nv_bfloat16* __restrict__ logits4, float* __restrict__ rms_scale4, float eps) {
    const int row = blockIdx.x;
    if (row >= kGateRowsActive) {
        return;
    }
    const auto* __restrict__ w2 =
        reinterpret_cast<const __nv_bfloat162*>(gate_w + static_cast<size_t>(row) * kDim);
    const auto* __restrict__ h2 = reinterpret_cast<const __nv_bfloat162*>(x4);
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f, rms_acc = 0.0f;
    const bool rms_cta = (row < kM);  // CTA j owns rms of x row j
    for (int pair = threadIdx.x; pair < kDimPairs; pair += kNumThreads) {
        const float2 wf = __bfloat1622float2(w2[pair]);  // ONE weight load ...
        const float2 f0 = __bfloat1622float2(h2[pair]);  // ... reused across four rows
        const float2 f1 = __bfloat1622float2(h2[kDimPairs + pair]);
        const float2 f2 = __bfloat1622float2(h2[2 * kDimPairs + pair]);
        const float2 f3 = __bfloat1622float2(h2[3 * kDimPairs + pair]);
        acc0 += f0.x * wf.x + f0.y * wf.y;
        acc1 += f1.x * wf.x + f1.y * wf.y;
        acc2 += f2.x * wf.x + f2.y * wf.y;
        acc3 += f3.x * wf.x + f3.y * wf.y;
        if (rms_cta) {
            const float2 fr = __bfloat1622float2(h2[row * kDimPairs + pair]);
            rms_acc += fr.x * fr.x + fr.y * fr.y;  // M1 CTA0 striding/order
        }
    }
    float accs[kM] = {acc0, acc1, acc2, acc3};
#pragma unroll
    for (int j = 0; j < kM; ++j) {
        const float sum = BlockSum<kNumThreads>(accs[j]);
        if (threadIdx.x == 0) {
            logits4[static_cast<size_t>(j) * kGateRowsPhysical + row] = __float2bfloat16(sum);
        }
        __syncthreads();  // shared warp_sums reuse across rows
    }
    if (rms_cta) {
        const float sq_sum = BlockSum<kNumThreads>(rms_acc);
        if (threadIdx.x == 0) {
            rms_scale4[row] = rsqrtf(sq_sum * (1.0f / static_cast<float>(kDim)) + eps);
        }
    }
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M4

using namespace nexrt::sm90::nex_n2_5_pro::M4;
extern "C" {
// x4 [4,4096] bf16 raw rows; gate_w [528,4096] bf16 gamma-folded; logits4 [4,528] bf16 out;
// rms_scale4 float[4] out; eps = rms epsilon (1e-6 in production).
int NexN25ProM4GateGemvRawAndRmsLaunch(const void* x4, const void* gate_w, void* logits4,
                                       float* rms_scale4, float eps, void* stream) {
    M4GateGemvRawAndRmsKernel<256>
        <<<dim3(kGateRowsActive), dim3(256), 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const __nv_bfloat16*>(x4), static_cast<const __nv_bfloat16*>(gate_w),
            static_cast<__nv_bfloat16*>(logits4), rms_scale4, eps);
    return static_cast<int>(cudaGetLastError());
}
int NexN25ProM4GateRowsPhysical() { return kGateRowsPhysical; }
const char* NexN25ProM4GateCudaErrorString(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
}
