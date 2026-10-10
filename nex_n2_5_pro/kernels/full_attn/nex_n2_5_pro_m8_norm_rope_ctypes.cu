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

// ONE-LAUNCH M=8 fused RMSNorm + partial RoPE for the full-attn M=8 verification
// path. M=8-FAMILY TU: native implementation, no BS=1/M=4 production source.
//
// One (6 x 8)-CTA grid: blockIdx.y = target row j (position pos_j = committed_len + j,
// consecutive C..C+7), blockIdx.x = CTA role (0..3 = Q heads: gemma-style RMSNorm
// (1 + weight) + partial RoPE; 4 = K head in place at k_cache[pos_j]; 5 = input-RMS
// scalar s that scales v_cache[pos_j] and the row's gate in place).
//
// The eight rows are mutually independent: row j reads only its own q_raw/x/gate rows
// and its own cache row pos_j (already written by the M=8 QKV scatter), and cache rows
// are finalized by their OWN row's norm/scale, so finalizing all eight rows before any
// attention core runs is value-invisible to the causal core (row j attends <= pos_j).

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M8
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "../common/m8_control_block.cuh"

namespace nexrt::sm90::nex_n2_5_pro::M8 {
constexpr int kHeadDim = 256;
constexpr int kRot = 64;
constexpr int kNqHeads = 4;
constexpr int kNheads = 5;                  // 4 q + 1 k
constexpr int kRows = m8ctl::kM;            // 8
constexpr int kQRow = kNqHeads * kHeadDim;  // 1024
constexpr int kThreads = 128;
constexpr int kValuesPerThread = 2;
constexpr int kWarps = kThreads / 32;
constexpr float kEps = 1e-6f;

__device__ __forceinline__ float m8_butterfly(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ uint32_t m8_nr_pack_bf16(float a, float b) {
    uint32_t r;
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(b), "f"(a));
    return r;
}

__device__ __forceinline__ void m8_load_pair(const __nv_bfloat16* __restrict__ g,
                                             float* __restrict__ r,
                                             __nv_bfloat16* __restrict__ sh) {
    const int v = *reinterpret_cast<const int*>(g);
    *reinterpret_cast<int*>(sh) = v;
    const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&v);
    const float2 f = __bfloat1622float2(b);
    r[0] = f.x;
    r[1] = f.y;
}

__global__ __launch_bounds__(kThreads) void M8NormRopeFusedKernel(
    const __nv_bfloat16* __restrict__ q_raw8, __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ qn, const __nv_bfloat16* __restrict__ kn,
    const float* __restrict__ cos_tab, const float* __restrict__ sin_tab,
    __nv_bfloat16* __restrict__ q_out8, const __nv_bfloat16* __restrict__ x8,
    __nv_bfloat16* __restrict__ v_cache, __nv_bfloat16* __restrict__ gate8, int hid, float eps_in,
    const int* __restrict__ ctl) {  // M=8 control block: base = word 1
    const int cta = blockIdx.x, tid = threadIdx.x;
    const int row = blockIdx.y;  // target row j: pos = C + j
    const int pos = m8ctl::CommittedLen(ctl) + row;
    const __nv_bfloat16* q_raw = q_raw8 + static_cast<size_t>(row) * kQRow;
    __nv_bfloat16* q_out = q_out8 + static_cast<size_t>(row) * kQRow;
    const __nv_bfloat16* x_in = x8 + static_cast<size_t>(row) * hid;
    __nv_bfloat16* gate = gate8 + static_cast<size_t>(row) * kQRow;

    if (cta < kNheads) {  // ---- q/k head = cta: RMSNorm + partial RoPE ----
        __shared__ __align__(16) __nv_bfloat16 x_sh[kHeadDim], w_sh[kHeadDim];
        __shared__ float red[kWarps];
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
        const int d0 = tid * kValuesPerThread;
        float xr[kValuesPerThread], wr[kValuesPerThread];
        m8_load_pair(src + d0, xr, &x_sh[d0]);
        m8_load_pair(wt + d0, wr, &w_sh[d0]);
        float ss = 0.f;
#pragma unroll
        for (int e = 0; e < kValuesPerThread; ++e) ss += xr[e] * xr[e];
        ss = m8_butterfly(ss);
        const int warp = tid >> 5, lane = tid & 31;
        if (lane == 0) red[warp] = ss;
        __syncthreads();
        ss = 0.f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) ss += red[w];
        const float rms = rsqrtf(ss * (1.0f / kHeadDim) + kEps);
        const float* cosv = cos_tab + static_cast<size_t>(pos) * (kRot / 2);
        const float* sinv = sin_tab + static_cast<size_t>(pos) * (kRot / 2);
#pragma unroll
        for (int e = 0; e < kValuesPerThread; e += 2) {
            float out[2];
#pragma unroll
            for (int pidx = 0; pidx < 2; ++pidx) {
                const int ee = e + pidx;
                const int d = d0 + ee;
                const float nd = xr[ee] * rms * (1.0f + wr[ee]);
                if (d < kRot) {
                    const int p = d ^ 32;
                    const float np =
                        __bfloat162float(x_sh[p]) * rms * (1.0f + __bfloat162float(w_sh[p]));
                    const int ang = d & 31;
                    const float c = cosv[ang], s = sinv[ang];
                    out[pidx] = (d < 32) ? (nd * c - np * s) : (nd * c + np * s);
                } else {
                    out[pidx] = nd;
                }
            }
            *reinterpret_cast<uint32_t*>(dst + d0 + e) = m8_nr_pack_bf16(out[0], out[1]);
        }
    } else {  // ---- CTA 5: s = rsqrt(mean(x^2)+eps_in); scale v & gate ----
        __shared__ float sred[kWarps];
        float ss = 0.f;
        for (int i = tid; i < hid; i += kThreads) {
            const float xi = __bfloat162float(x_in[i]);
            ss += xi * xi;
        }
        ss = m8_butterfly(ss);
        const int warp = tid >> 5, lane = tid & 31;
        if (lane == 0) sred[warp] = ss;
        __syncthreads();
        ss = 0.f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) ss += sred[w];
        const float s = rsqrtf(ss * (1.0f / hid) + eps_in);
        __nv_bfloat16* vp = v_cache + static_cast<size_t>(pos) * kHeadDim;
        for (int i = tid * 2; i < kHeadDim; i += kThreads * 2) {
            const __nv_bfloat162 x = *reinterpret_cast<const __nv_bfloat162*>(vp + i);
            const float2 f = __bfloat1622float2(x);
            *reinterpret_cast<uint32_t*>(vp + i) = m8_nr_pack_bf16(f.x * s, f.y * s);
        }
        for (int i = tid * 2; i < kQRow; i += kThreads * 2) {
            const __nv_bfloat162 x = *reinterpret_cast<const __nv_bfloat162*>(gate + i);
            const float2 f = __bfloat1622float2(x);
            *reinterpret_cast<uint32_t*>(gate + i) = m8_nr_pack_bf16(f.x * s, f.y * s);
        }
    }
}

int launch_m8_norm_rope(const void* q_raw8, void* k_cache, const void* qn, const void* kn,
                        const void* cos_tab, const void* sin_tab, void* q_out8, const void* ctl,
                        const void* x8, void* v_cache, void* gate8, int hid, float eps_in,
                        void* stream) {
    dim3 grid(kNheads + 1, kRows);
    M8NormRopeFusedKernel<<<grid, kThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_raw8), reinterpret_cast<__nv_bfloat16*>(k_cache),
        reinterpret_cast<const __nv_bfloat16*>(qn), reinterpret_cast<const __nv_bfloat16*>(kn),
        reinterpret_cast<const float*>(cos_tab), reinterpret_cast<const float*>(sin_tab),
        reinterpret_cast<__nv_bfloat16*>(q_out8), reinterpret_cast<const __nv_bfloat16*>(x8),
        reinterpret_cast<__nv_bfloat16*>(v_cache), reinterpret_cast<__nv_bfloat16*>(gate8), hid,
        eps_in, reinterpret_cast<const int*>(ctl));
    return static_cast<int>(cudaGetLastError());
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M8
using namespace nexrt::sm90::nex_n2_5_pro::M8;
extern "C" int NexN25ProM8NormRopeLaunch(const void* q_raw8, void* k_cache, const void* qn,
                                         const void* kn, const void* cos_tab, const void* sin_tab,
                                         void* q_out8, const void* ctl, const void* x8,
                                         void* v_cache, void* gate8, int hid, float eps_in,
                                         void* stream) {
    return launch_m8_norm_rope(q_raw8, k_cache, qn, kn, cos_tab, sin_tab, q_out8, ctl, x8, v_cache,
                               gate8, hid, eps_in, stream);
}
