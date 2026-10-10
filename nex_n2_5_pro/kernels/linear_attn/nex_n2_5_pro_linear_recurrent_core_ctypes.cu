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

// Gated DeltaNet recurrent core selected for the nex_n2_5_pro TP=8 backend.
//
// Inputs:
//   qkv_out[1536] bf16 = query[256] | key[256] | value[1024], after qkv conv+SiLU
//   beta[8] fp32
//   g[8] fp32
//   recurrent_state[8,128,128] fp32, updated in place
//
// Output:
//   out[8,128] bf16
//
// No repeat_interleave is materialized. Local v-head h maps to q/k head h >> 2.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY bs1
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY {
constexpr int kHeads = 8;
constexpr int kQKHeads = 2;
constexpr int kGqaLocal = 4;
constexpr int kDim = 128;
constexpr int kQOff = 0;
constexpr int kKOff = kQKHeads * kDim;      // 256
constexpr int kVOff = 2 * kQKHeads * kDim;  // 512
constexpr float kEps = 1.0e-6f;
constexpr float kScale = 0.08838834764831845f;  // 1/sqrt(128)

static_assert(kHeads == kQKHeads * kGqaLocal, "local GQA geometry mismatch");

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_sum_broadcast(float v) {
    return __shfl_sync(0xffffffffu, warp_sum(v), 0);
}

template <int kTileV>
__global__ __launch_bounds__(kTileV * 32) void RecurrentCoreTransposedSharedNormKernel(
    const __nv_bfloat16* __restrict__ qkv_out, const float* __restrict__ beta,
    const float* __restrict__ g, float* __restrict__ state_t, __nv_bfloat16* __restrict__ out) {
    static_assert(kDim % kTileV == 0, "value tile must divide 128");

    __shared__ float sq_norm;
    __shared__ float sk_norm;
    __shared__ float sqk_dot;

    const int head = blockIdx.x;
    const int value_base = blockIdx.y * kTileV;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int value_col = value_base + warp;
    const int qk_head = head >> 2;
    const __nv_bfloat16* q_ptr = qkv_out + kQOff + qk_head * kDim;
    const __nv_bfloat16* k_ptr = qkv_out + kKOff + qk_head * kDim;

    if (warp == 0) {
        float q_sq = 0.0f;
        float k_sq = 0.0f;
        float qk_raw = 0.0f;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int i = lane + t * 32;
            const float qv = __bfloat162float(q_ptr[i]);
            const float kv = __bfloat162float(k_ptr[i]);
            q_sq += qv * qv;
            k_sq += kv * kv;
            qk_raw += qv * kv;
        }
        q_sq = warp_sum(q_sq);
        k_sq = warp_sum(k_sq);
        qk_raw = warp_sum(qk_raw);
        if (lane == 0) {
            const float qn = rsqrtf(q_sq + kEps) * kScale;
            const float kn = rsqrtf(k_sq + kEps);
            sq_norm = qn;
            sk_norm = kn;
            sqk_dot = qk_raw * qn * kn;
        }
    }
    __syncthreads();

    if (head >= kHeads || value_col >= kDim) {
        return;
    }

    const __nv_bfloat16* v_ptr = qkv_out + kVOff + head * kDim;
    float* s_vec =
        state_t + (static_cast<size_t>(head) * kDim * kDim) + static_cast<size_t>(value_col) * kDim;
    const float q_norm = sq_norm;
    const float k_norm = sk_norm;
    const float qk_dot = sqk_dot;
    const float decay = expf(g[head]);

    float kv_mem = 0.0f;
    float q_mem = 0.0f;
    float k_normed[4];
    float s_old[4];

#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int i = lane + t * 32;
        const float qv = __bfloat162float(q_ptr[i]) * q_norm;
        const float kv = __bfloat162float(k_ptr[i]) * k_norm;
        const float sv = s_vec[i];
        kv_mem += kv * sv;
        q_mem += qv * sv;
        k_normed[t] = kv;
        s_old[t] = sv;
    }

    kv_mem = warp_sum_broadcast(kv_mem) * decay;
    q_mem = warp_sum_broadcast(q_mem) * decay;

    const float value = __bfloat162float(v_ptr[value_col]);
    const float delta = (value - kv_mem) * beta[head];
    const float out_f = q_mem + qk_dot * delta;

#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int i = lane + t * 32;
        s_vec[i] = decay * s_old[t] + k_normed[t] * delta;
    }

    if (lane == 0) {
        out[head * kDim + value_col] = __float2bfloat16(out_f);
    }
}

template <int kTileV>
int launch_recurrent_core_transposed_sharednorm(const void* qkv_out, const float* beta,
                                                const float* g, float* state_t, void* out,
                                                void* stream) {
    dim3 grid(kHeads, kDim / kTileV);
    RecurrentCoreTransposedSharedNormKernel<kTileV>
        <<<grid, kTileV * 32, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<const __nv_bfloat16*>(qkv_out), beta, g, state_t,
            reinterpret_cast<__nv_bfloat16*>(out));
    return static_cast<int>(cudaGetLastError());
}

}  // namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
using namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY;
extern "C" int NexN25ProLinearRecurrentLaunch(const void* qkv_out, const float* beta,
                                              const float* g, float* state_t, void* out,
                                              void* stream) {
    return launch_recurrent_core_transposed_sharednorm<4>(qkv_out, beta, g, state_t, out, stream);
}

extern "C" const char* nex_n2_5_pro_linear_recurrent_cuda_error_string(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
