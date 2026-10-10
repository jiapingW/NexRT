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

// M=4 GDN recurrent core, hoisted-norms form.
//
// NexN25ProM4GdnCoreLaunch: a BIT-EXACT restructuring of the baseline core, which
// computes per-token q/k norms + qk_dot inside the sequential token loop (warp 0
// only, bracketed by TWO __syncthreads per token = 8 block syncs on the serial
// path). Those scalars are input-only: this kernel computes ALL FOUR tokens'
// norms in one parallel entry phase -- warp w reduces token w with the IDENTICAL
// warp_sum order the baseline used (so every produced bit matches) -- then runs
// the token loop with ZERO block syncs (each warp touches only its private state
// column, per-token smem slots are read-only after the single entry barrier).

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M4
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "../common/m4_control_block.cuh"

namespace nexrt::sm90::nex_n2_5_pro::M4 {
constexpr int kTokens = m4ctl::kM;
constexpr int kHeads = 8;
constexpr int kQKHeads = 2;
constexpr int kDim = 128;
constexpr int kQOff = 0;
constexpr int kKOff = kQKHeads * kDim;                           // 256
constexpr int kVOff = 2 * kQKHeads * kDim;                       // 512
constexpr int kTokStride = 2 * kQKHeads * kDim + kHeads * kDim;  // 1536
constexpr int kOutStride = kHeads * kDim;                        // 1024
constexpr float kEps = 1.0e-6f;
constexpr float kScale = 0.08838834764831845f;  // 1/sqrt(128)
constexpr int kTileV = 4;
constexpr int kStateElems = kHeads * kDim * kDim;

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

__global__ __launch_bounds__(kTileV * 32) void M4GdnCoreKernel(
    const __nv_bfloat16* __restrict__ qkv4, const float* __restrict__ beta4,
    const float* __restrict__ g4, float* __restrict__ state_ring, const int* __restrict__ ctl,
    __nv_bfloat16* __restrict__ out4) {
    __shared__ float sq_norm[kTokens];
    __shared__ float sk_norm[kTokens];
    __shared__ float sqk_dot[kTokens];

    const int head = blockIdx.x;
    const int value_base = blockIdx.y * kTileV;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int value_col = value_base + warp;
    const int qk_head = head >> 2;

    const int committed_len = ctl[m4ctl::kCommittedLen];
    const size_t vec_offset =
        static_cast<size_t>(head) * kDim * kDim + static_cast<size_t>(value_col) * kDim;

    // Hoisted entry phase: warp w computes token w's norms with the baseline's exact
    // reduction order (bit-identical scalars), all four tokens in parallel.
    {
        const int tok = warp;  // kTokens == kTileV == 4
        const __nv_bfloat16* qkv = qkv4 + static_cast<size_t>(tok) * kTokStride;
        const __nv_bfloat16* q_ptr = qkv + kQOff + qk_head * kDim;
        const __nv_bfloat16* k_ptr = qkv + kKOff + qk_head * kDim;
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
            sq_norm[tok] = qn;
            sk_norm[tok] = kn;
            sqk_dot[tok] = qk_raw * qn * kn;
        }
    }

    const float* base_vec = state_ring +
                            static_cast<size_t>(m4ctl::StateSlot(committed_len - 1)) * kStateElems +
                            vec_offset;
    float s_reg[4];
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        s_reg[t] = base_vec[lane + t * 32];
    }
    __syncthreads();  // the ONLY block sync in the kernel

#pragma unroll 1
    for (int tok = 0; tok < kTokens; ++tok) {
        const __nv_bfloat16* qkv = qkv4 + static_cast<size_t>(tok) * kTokStride;
        const __nv_bfloat16* q_ptr = qkv + kQOff + qk_head * kDim;
        const __nv_bfloat16* k_ptr = qkv + kKOff + qk_head * kDim;
        const __nv_bfloat16* v_ptr = qkv + kVOff + head * kDim;
        const float q_norm = sq_norm[tok];
        const float k_norm = sk_norm[tok];
        const float qk_dot = sqk_dot[tok];
        const float decay = expf(g4[tok * kHeads + head]);

        float kv_mem = 0.0f;
        float q_mem = 0.0f;
        float k_normed[4];
        float s_old[4];

#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int i = lane + t * 32;
            const float qv = __bfloat162float(q_ptr[i]) * q_norm;
            const float kv = __bfloat162float(k_ptr[i]) * k_norm;
            const float sv = s_reg[t];
            kv_mem += kv * sv;
            q_mem += qv * sv;
            k_normed[t] = kv;
            s_old[t] = sv;
        }

        kv_mem = warp_sum_broadcast(kv_mem) * decay;
        q_mem = warp_sum_broadcast(q_mem) * decay;

        const float value = __bfloat162float(v_ptr[value_col]);
        const float delta = (value - kv_mem) * beta4[tok * kHeads + head];
        const float out_f = q_mem + qk_dot * delta;

#pragma unroll
        for (int t = 0; t < 4; ++t) {
            s_reg[t] = decay * s_old[t] + k_normed[t] * delta;
        }

        float* snap = state_ring +
                      static_cast<size_t>(m4ctl::StateSlot(committed_len + tok)) * kStateElems +
                      vec_offset;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            snap[lane + t * 32] = s_reg[t];
        }

        if (lane == 0) {
            out4[static_cast<size_t>(tok) * kOutStride + head * kDim + value_col] =
                __float2bfloat16(out_f);
        }
        // NO loop sync: per-token smem slots are read-only after the entry barrier
        // and every warp writes only its private column.
    }
}

extern "C" {
int NexN25ProM4GdnCoreLaunch(const void* qkv4, const void* beta4, const void* g4, void* state_ring,
                             const void* ctl, void* out4, void* stream) {
    dim3 grid(kHeads, kDim / kTileV);
    M4GdnCoreKernel<<<grid, kTileV * 32, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv4), reinterpret_cast<const float*>(beta4),
        reinterpret_cast<const float*>(g4), reinterpret_cast<float*>(state_ring),
        reinterpret_cast<const int*>(ctl), reinterpret_cast<__nv_bfloat16*>(out4));
    return static_cast<int>(cudaGetLastError());
}
}  // extern "C"

}  // namespace nexrt::sm90::nex_n2_5_pro::M4