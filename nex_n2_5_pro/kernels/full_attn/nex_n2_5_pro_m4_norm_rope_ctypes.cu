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

// ONE-LAUNCH M=4 fused RMSNorm + partial RoPE for the nex_n2_5_pro full-attn M=4 verification
// path. Replaces FOUR per-position the selected BS=1 T=32 kernel
// launches with ONE launch of a (6 x 4)-CTA grid: blockIdx.y = draft row j (position
// pos_j = seqlen[1] + j, consecutive S..S+3), blockIdx.x = the M1 kernel's CTA role
// (0..3 = Q heads, 4 = K head in place at k_cache[pos_j], 5 = input-RMS scalar s that
// scales v_cache[pos_j] and the row's gate in place).
//
// The per-CTA body is a VERBATIM transcription of NormRopeFusedKernel<T> in
// nex_n2_5_pro_norm_rope_ctypes.cu (same load_chunk vectorization, butterfly reduction, shared
// staging, rotation-partner math, and fp operation order). The four rows are mutually
// independent here: row j reads only its own q_raw/x/gate rows and its own cache row
// pos_j (already written by the M=4 QKV scatter), so batching the launch changes no
// operand and no arithmetic -- outputs and cache mutations are BIT-EXACT vs the
// four-launch path on identical inputs.
// Ordering safety: the causal core for row j reads cache rows only up to
// pos_j, and cache rows are finalized by their OWN row's norm/scale in both schedules,
// so finalizing all four rows before any core runs is value-invisible.
//
// Row strides: q_raw4/q_out4/gate4 stride 1024 (4 heads x 256), x4 stride `hid`;
// pos is ALWAYS device-driven (seqlen[1] + row).

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M4
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace nexrt::sm90::nex_n2_5_pro::M4 {
constexpr int kHeadDim = 256;
constexpr int kRot = 64;
constexpr int kNqHeads = 4;
constexpr int kNheads = 5;  // 4 q + 1 k
constexpr int kRows = 4;
constexpr int kQRow = kNqHeads * kHeadDim;  // 1024
constexpr float kEps = 1e-6f;

__device__ __forceinline__ float butterfly(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ void m4_norm_load_chunk(const __nv_bfloat16* __restrict__ global,
                                                   float* __restrict__ registers,
                                                   __nv_bfloat16* __restrict__ shared) {
    const int4 packed = *reinterpret_cast<const int4*>(global);
    *reinterpret_cast<int4*>(shared) = packed;
    const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const float2 values = __bfloat1622float2(pairs[pair]);
        registers[2 * pair] = values.x;
        registers[2 * pair + 1] = values.y;
    }
}

__global__ __launch_bounds__(32) void M4NormRopeKernel(
    const __nv_bfloat16* __restrict__ q_raw4, __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ qn, const __nv_bfloat16* __restrict__ kn,
    const float* __restrict__ cos_tab, const float* __restrict__ sin_tab,
    __nv_bfloat16* __restrict__ q_out4, const __nv_bfloat16* __restrict__ x4,
    __nv_bfloat16* __restrict__ v_cache, __nv_bfloat16* __restrict__ gate4, int hid, float eps_in,
    const unsigned int* __restrict__ seqlen) {
    constexpr int VPT = 8;
    constexpr int T = 32;
    const int cta = blockIdx.x, tid = threadIdx.x;
    const int row = blockIdx.y;  // draft position j: pos = S + j
    const int pos = static_cast<int>(seqlen[1]) + row;
    const __nv_bfloat16* q_raw = q_raw4 + static_cast<size_t>(row) * kQRow;
    __nv_bfloat16* q_out = q_out4 + static_cast<size_t>(row) * kQRow;
    const __nv_bfloat16* x_in = x4 + static_cast<size_t>(row) * hid;
    __nv_bfloat16* gate = gate4 + static_cast<size_t>(row) * kQRow;

    if (cta < kNheads) {  // ---- q/k head = cta: RMSNorm + partial RoPE ----
        __shared__ __align__(16) __nv_bfloat16 x_sh[kHeadDim], w_sh[kHeadDim];
        const __nv_bfloat16* src;
        const __nv_bfloat16* wt;
        __nv_bfloat16* dst;
        if (cta < kNqHeads) {
            src = q_raw + cta * kHeadDim;
            wt = qn;
            dst = q_out + cta * kHeadDim;
        } else {
            __nv_bfloat16* kp = k_cache + static_cast<size_t>(pos) * kHeadDim;
            src = kp;
            wt = kn;
            dst = kp;
        }
        const int d0 = tid * VPT;
        float xr[VPT], wr[VPT];
        m4_norm_load_chunk(src + d0, xr, &x_sh[d0]);
        m4_norm_load_chunk(wt + d0, wr, &w_sh[d0]);
        float ss = 0.f;
#pragma unroll
        for (int e = 0; e < VPT; ++e) ss += xr[e] * xr[e];
        ss = butterfly(ss);
        __syncthreads();
        const float rms = rsqrtf(ss * (1.0f / kHeadDim) + kEps);
        const float* cosv = cos_tab + static_cast<size_t>(pos) * (kRot / 2);
        const float* sinv = sin_tab + static_cast<size_t>(pos) * (kRot / 2);
#pragma unroll
        for (int e = 0; e < VPT; ++e) {
            const int d = d0 + e;
            const float nd = xr[e] * rms * (1.0f + wr[e]);
            if (d < kRot) {
                const int p = d ^ 32;
                const float np =
                    __bfloat162float(x_sh[p]) * rms * (1.0f + __bfloat162float(w_sh[p]));
                const int ang = d & 31;
                const float c = cosv[ang], s = sinv[ang];
                dst[d] = __float2bfloat16((d < 32) ? (nd * c - np * s) : (nd * c + np * s));
            } else
                dst[d] = __float2bfloat16(nd);
        }
    } else {  // ---- CTA 5: s = rsqrt(mean(x^2)+eps_in); scale v & gate ----
        float ss = 0.f;
        for (int i = tid; i < hid; i += T) {
            const float xi = __bfloat162float(x_in[i]);
            ss += xi * xi;
        }
        ss = butterfly(ss);

        const float s = rsqrtf(ss * (1.0f / hid) + eps_in);
        __nv_bfloat16* vp = v_cache + static_cast<size_t>(pos) * kHeadDim;
        for (int i = tid; i < kHeadDim; i += T)
            vp[i] = __float2bfloat16(__bfloat162float(vp[i]) * s);
        for (int i = tid; i < kQRow; i += T)
            gate[i] = __float2bfloat16(__bfloat162float(gate[i]) * s);
    }
}

extern "C" int NexN25ProM4NormRopeLaunch(const void* q_raw4, void* k_cache, const void* qn,
                                         const void* kn, const void* cos_table,
                                         const void* sin_table, void* q_out4, const void* seqlen,
                                         const void* x4, void* v_cache, void* gate4,
                                         int hidden_size, float epsilon, void* stream) {
    const dim3 grid(kNheads + 1, kRows);
    M4NormRopeKernel<<<grid, 32, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_raw4), reinterpret_cast<__nv_bfloat16*>(k_cache),
        reinterpret_cast<const __nv_bfloat16*>(qn), reinterpret_cast<const __nv_bfloat16*>(kn),
        reinterpret_cast<const float*>(cos_table), reinterpret_cast<const float*>(sin_table),
        reinterpret_cast<__nv_bfloat16*>(q_out4), reinterpret_cast<const __nv_bfloat16*>(x4),
        reinterpret_cast<__nv_bfloat16*>(v_cache), reinterpret_cast<__nv_bfloat16*>(gate4),
        hidden_size, epsilon, reinterpret_cast<const unsigned int*>(seqlen));
    return static_cast<int>(cudaGetLastError());
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M4