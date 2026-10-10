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

// QKV projection selected for the nex_n2_5_pro TP=8 backend.
#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY bs1
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "nex_n2_5_pro_qkv_common.cuh"

namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY {
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

// Three rows per CTA, split-K8, streaming weights, L2 prefetch, FP32 lane-dot.
// Each 256-thread CTA owns three adjacent rows inside one 128-row scale block;
// its eight warps split K. Weights are one-use: loaded with ld.global.cs after a
// register-light L2 prefetch of all twelve 128B lines. Products within one
// 8-element lane span and scale-block sums accumulate in FP32.

// The residual input is BF16 and is normalized downstream. Keep both the
// activation conversion and lane accumulation in FP32 to avoid FP16 overflow.
__device__ __forceinline__ float dot8_fp8_f32acc(const uint2 wraw, const float2* __restrict__ xf) {
    const __nv_fp8x2_e4m3* wp = reinterpret_cast<const __nv_fp8x2_e4m3*>(&wraw);
    float2 acc = make_float2(0.0f, 0.0f);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const float2 w = __half22float2(static_cast<__half2>(wp[pair]));
        acc.x = fmaf(w.x, xf[pair].x, acc.x);
        acc.y = fmaf(w.y, xf[pair].y, acc.y);
    }
    return acc.x + acc.y;
}

__device__ __forceinline__ uint2 load_weight_cs(const void* ptr) {
    uint2 v;
    asm volatile("ld.global.cs.v2.u32 {%0, %1}, [%2];\n"
                 : "=r"(v.x), "=r"(v.y)
                 : "l"(ptr)
                 : "memory");
    return v;
}

constexpr int kRowsPerCTA = 3;
constexpr int kSplitK = 8;
constexpr int kGroupsPerScaleBlock = (128 + kRowsPerCTA - 1) / kRowsPerCTA;
constexpr int kNumScaleBlocks = kTotalRows / 128;

__global__ __launch_bounds__(256) void QkvGroupSplitKKernel(
    const __nv_bfloat16* __restrict__ x, const __nv_fp8_e4m3* __restrict__ W,
    const float* __restrict__ scales, __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ gate_out, __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache, int pos, const unsigned int* __restrict__ seqlen) {
    if (seqlen) pos = static_cast<int>(seqlen[1]);  // device-side KV write slot (decode graph)
    constexpr int kLinesPerRow = 4;
    __shared__ float partials[kRowsPerCTA][kSplitK];

    const int split = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int scale_group = blockIdx.x / kGroupsPerScaleBlock;
    const int group_in_scale = blockIdx.x - scale_group * kGroupsPerScaleBlock;
    const int row0 = scale_group * 128 + group_in_scale * kRowsPerCTA;
    const int valid_rows = min(kRowsPerCTA, scale_group * 128 + 128 - row0);
    const int k0 = split * 512;
    const float* s = scales + static_cast<size_t>(row0 >> 7) * kScaleBlocksK;

    if (lane < valid_rows * kLinesPerRow) {
        const int row_offset = lane >> 2;
        const int line = lane & 3;
        const void* ptr = W + static_cast<size_t>(row0 + row_offset) * kHid + k0 + line * 128;
        asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr));
    }

    float acc[kRowsPerCTA] = {};
    uint2 wbuf[2][kRowsPerCTA];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int k = k0 + half * 256 + lane * 8;
#pragma unroll
        for (int r = 0; r < kRowsPerCTA; ++r) {
            wbuf[half][r] = (r < valid_rows)
                                ? load_weight_cs(W + static_cast<size_t>(row0 + r) * kHid + k)
                                : make_uint2(0, 0);
        }
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int k = k0 + half * 256 + lane * 8;
        const float sc = s[split * 4 + half * 2 + (lane >> 4)];
        const int4 xraw = *reinterpret_cast<const int4*>(x + k);
        const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xraw);
        float2 xf[4];
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            xf[pair] = __bfloat1622float2(xp[pair]);
        }
#pragma unroll
        for (int r = 0; r < kRowsPerCTA; ++r) {
            acc[r] += sc * dot8_fp8_f32acc(wbuf[half][r], xf);
        }
    }

#pragma unroll
    for (int r = 0; r < kRowsPerCTA; ++r) {
        const float v = warp_sum(acc[r]);
        if (lane == 0) partials[r][split] = v;
    }
    __syncthreads();

    for (int r = threadIdx.x; r < valid_rows; r += blockDim.x) {
        float v = 0.0f;
#pragma unroll
        for (int si = 0; si < kSplitK; ++si) {
            v += partials[r][si];
        }
        scatter_row(row0 + r, v, q_out, gate_out, k_cache, v_cache, pos);
    }
}

int launch_qkv_group_splitk(const void* x, const void* W, const float* scales, void* q_out,
                            void* gate_out, void* k_cache, void* v_cache, int pos,
                            const void* seqlen, void* stream) {
    QkvGroupSplitKKernel<<<kNumScaleBlocks * kGroupsPerScaleBlock, 256, 0,
                           reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const __nv_fp8_e4m3*>(W),
        scales, reinterpret_cast<__nv_bfloat16*>(q_out), reinterpret_cast<__nv_bfloat16*>(gate_out),
        reinterpret_cast<__nv_bfloat16*>(k_cache), reinterpret_cast<__nv_bfloat16*>(v_cache), pos,
        reinterpret_cast<const unsigned int*>(seqlen));
    return static_cast<int>(cudaGetLastError());
}

}  // namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
using namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY;
#ifndef NEXRT_MTP_RUNTIME_ONLY
extern "C" int NexN25ProQkvProjLaunch(const void* x, const void* W, const float* scales,
                                      void* q_out, void* gate_out, void* k_cache, void* v_cache,
                                      int pos, void* stream) {
    return launch_qkv_group_splitk(x, W, scales, q_out, gate_out, k_cache, v_cache, pos, nullptr,
                                   stream);
}
#endif

extern "C" int NexN25ProQkvProjDynLaunch(const void* x, const void* W, const float* scales,
                                         void* q_out, void* gate_out, void* k_cache, void* v_cache,
                                         const void* seqlen, void* stream) {
    return launch_qkv_group_splitk(x, W, scales, q_out, gate_out, k_cache, v_cache, 0, seqlen,
                                   stream);
}

#ifndef NEXRT_MTP_RUNTIME_ONLY
extern "C" const char* nex_n2_5_pro_qkv_cuda_error_string(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
#endif
