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

// M=8 GDN recurrent core: the ordered eight-token gated-delta recurrence in ONE
// launch, on the M=8-ONLY recurrent state ring.
//
// M=8-FAMILY TU: native eight-row implementation, no BS=1/M=4 production source.
// Mechanism class: the proven "hoist" restructuring (parallel input-only entry phase,
// zero block syncs on the serial path), re-derived for eight tokens:
//
//   * ENTRY PHASE (parallel): the per-token q/k norms and qk_dot are input-only.
//     Warp w computes tokens w and w+4 with the exact warp_sum reduction order of the
//     sequential form, so every produced scalar bit matches a one-token-at-a-time
//     evaluation. One __syncthreads total.
//   * TOKEN LOOP (ordered, 8 steps): each warp owns one private state column
//     (value_col); the recurrence
//         S <- decay * S + k_normed (x) delta,   delta = (v - k.S*decay) * beta
//         out = q.S*decay + qk_dot * delta
//     is carried entirely in registers; per-token SMEM slots are read-only after the
//     entry barrier, so NO loop synchronization is needed. The recurrence ORDER is
//     unchanged -- only input-only work was hoisted.
//
// STATE RING. slot(p) = p % 9 holds the complete state after consuming absolute
// position p. Base (C-1) and rows C..C+7 are nine consecutive positions, so they
// take nine DISTINCT slots: this kernel never overwrites the base it read.
// Consequences, all relied on by the family contract:
//   * a FAILED unit (status != 0, commit_count 0) leaves committed_len at C and
//     the base snapshot intact, so the unit is retryable without reloading the
//     request (the caller clears the sticky protocol latches first);
//   * partial acceptance needs no copy -- advancing committed_len by
//     commit_count makes slot((C + commit_count - 1) % 9) the next base, which
//     row commit_count-1 wrote during this unit.
// The base is still read ONCE at entry into registers before the token loop;
// that is now a pure locality/latency choice, not a correctness requirement.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M8
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "../common/m8_control_block.cuh"

namespace nexrt::sm90::nex_n2_5_pro::M8 {
constexpr int kTokens = m8ctl::kM;  // 8
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

__device__ __forceinline__ float m8_warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float m8_warp_sum_broadcast(float v) {
    return __shfl_sync(0xffffffffu, m8_warp_sum(v), 0);
}

// Every input-only value -- normalized q/k vectors, v, beta and exp(g) -- is
// staged before the entry barrier, leaving the ordered recurrence on registers
// and shared memory.
__global__ __launch_bounds__(kTileV * 32) void M8GdnCoreKernel(
    const __nv_bfloat16* __restrict__ qkv8,  // [8, 1536]
    const float* __restrict__ beta8,         // [8, 8]
    const float* __restrict__ g8,            // [8, 8]
    float* __restrict__ state_ring,          // [9, 8, 128, 128] M=8-only ring
    const int* __restrict__ ctl,             // M=8 control block (committed_len)
    __nv_bfloat16* __restrict__ out8) {      // [8, 1024]

    __shared__ float sqk_dot[kTokens];
    __shared__ float sqv[kTokens][kDim];
    __shared__ float skv[kTokens][kDim];
    __shared__ float sbeta[kTokens];
    __shared__ float sdecay[kTokens];

    const int head = blockIdx.x;
    const int value_base = blockIdx.y * kTileV;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int value_col = value_base + warp;
    const int qk_head = head >> 2;

    const int committed_len = ctl[m8ctl::kCommittedLen];
    const size_t vec_offset =
        static_cast<size_t>(head) * kDim * kDim + static_cast<size_t>(value_col) * kDim;

// Entry phase: warp w computes tokens w and w+4 with the exact sequential-form
// reduction order (bit-identical scalars), all eight tokens across four warps.
#pragma unroll
    for (int half = 0; half < kTokens / kTileV; ++half) {
        const int tok = warp + half * kTileV;
        const __nv_bfloat16* qkv = qkv8 + static_cast<size_t>(tok) * kTokStride;
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
        q_sq = m8_warp_sum(q_sq);
        k_sq = m8_warp_sum(k_sq);
        qk_raw = m8_warp_sum(qk_raw);
        const float q_sq_b = __shfl_sync(0xffffffffu, q_sq, 0);
        const float k_sq_b = __shfl_sync(0xffffffffu, k_sq, 0);
        const float qk_b = __shfl_sync(0xffffffffu, qk_raw, 0);
        const float qn = rsqrtf(q_sq_b + kEps) * kScale;
        const float kn = rsqrtf(k_sq_b + kEps);
        if (lane == 0) {
            sqk_dot[tok] = qk_b * qn * kn;
        }
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int i = lane + t * 32;
            sqv[tok][i] = __bfloat162float(q_ptr[i]) * qn;
            skv[tok][i] = __bfloat162float(k_ptr[i]) * kn;
        }
    }
    if (warp == 0 && lane < kTokens) {
        sbeta[lane] = beta8[lane * kHeads + head];
        sdecay[lane] = expf(g8[lane * kHeads + head]);
    }
    float v_reg[kTokens];
#pragma unroll
    for (int tk = 0; tk < kTokens; ++tk)
        v_reg[tk] = __bfloat162float(
            qkv8[static_cast<size_t>(tk) * kTokStride + kVOff + head * kDim + value_col]);

    // Base state (after C-1): its slot is distinct from all eight snapshot slots
    // (9-slot ring), so it survives this unit and any failed retry of it.
    const float* base_vec = state_ring +
                            static_cast<size_t>(m8ctl::StateSlot(committed_len - 1)) * kStateElems +
                            vec_offset;
    float s_reg[4];
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        s_reg[t] = base_vec[lane + t * 32];
    }
    __syncthreads();  // the ONLY block sync in the kernel

#pragma unroll 1
    for (int tok = 0; tok < kTokens; ++tok) {
        const float qk_dot = sqk_dot[tok];
        const float decay = sdecay[tok];

        float kv_mem = 0.0f;
        float q_mem = 0.0f;
        float k_normed[4];
        float s_old[4];

#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int i = lane + t * 32;
            const float qv = sqv[tok][i];
            const float kv = skv[tok][i];
            const float sv = s_reg[t];
            kv_mem += kv * sv;
            q_mem += qv * sv;
            k_normed[t] = kv;
            s_old[t] = sv;
        }

        kv_mem = m8_warp_sum_broadcast(kv_mem) * decay;
        q_mem = m8_warp_sum_broadcast(q_mem) * decay;

        const float value = v_reg[tok];
        const float beta = sbeta[tok];
        const float delta = (value - kv_mem) * beta;
        const float out_f = q_mem + qk_dot * delta;

#pragma unroll
        for (int t = 0; t < 4; ++t) {
            s_reg[t] = decay * s_old[t] + k_normed[t] * delta;
        }

        float* snap = state_ring +
                      static_cast<size_t>(m8ctl::StateSlot(committed_len + tok)) * kStateElems +
                      vec_offset;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            snap[lane + t * 32] = s_reg[t];
        }

        if (lane == 0) {
            out8[static_cast<size_t>(tok) * kOutStride + head * kDim + value_col] =
                __float2bfloat16(out_f);
        }
        // NO loop sync: per-token smem slots are read-only after the entry barrier
        // and every warp writes only its private state column.
    }
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M8
using namespace nexrt::sm90::nex_n2_5_pro::M8;
extern "C" {
// qkv8 [8,1536] bf16; beta8/g8 [8,8] fp32; state_ring [9,8,128,128] fp32 (M=8-only);
// ctl int32[32] M=8 control block; out8 [8,1024] bf16.
int NexN25ProM8GdnCoreLaunch(const void* qkv8, const void* beta8, const void* g8, void* state_ring,
                             const void* ctl, void* out8, void* stream) {
    dim3 grid(kHeads, kDim / kTileV);
    M8GdnCoreKernel<<<grid, kTileV * 32, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv8), reinterpret_cast<const float*>(beta8),
        reinterpret_cast<const float*>(g8), reinterpret_cast<float*>(state_ring),
        reinterpret_cast<const int*>(ctl), reinterpret_cast<__nv_bfloat16*>(out8));
    return static_cast<int>(cudaGetLastError());
}
int NexN25ProM8StateRingSlots() { return m8ctl::kStateRingSlots; }
int NexN25ProM8GdnCoreTokens() { return kTokens; }
}
