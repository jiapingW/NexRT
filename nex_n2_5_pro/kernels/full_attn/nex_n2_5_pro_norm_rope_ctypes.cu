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

// RMSNorm + partial RoPE for q (4 heads) and k (1 head), nex_n2_5_pro full-attn decode (TP=8).
// rms_norm(x,w) = x*rsqrt(mean(x^2)+eps)*(1+w), then partial RoPE on the first 64 dims.
//
// Templated on <kThreadsPerHead, kHeadsPerCTA> so warp count / block-reduce tree / shared sizing
// / loop bounds are all compile-time. Each head:
//  - ONE wide vectorized coalesced load/thread (int4/int2/bfloat2 for T=32/64/128) of x and w into
//    a contiguous-chunk layout, staged in shared (so the RoPE partner dim d^32 is a shared read).
//  - templated block reduce (warp butterfly; cross-warp via shared when T>32) for the rms scalar.
//  - per-dim: normed = x*rms*(1+w); rotate dims<64 (partner from shared), pass-through 64:256.
// k written in place to k_cache[pos]. cos/sin precomputed tables [seq,32] indexed by pos.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY bs1
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY {
constexpr int kHeadDim = 256;
constexpr int kRot = 64;
constexpr int kNqHeads = 4;
constexpr int kNheads = 5;  // 4 q + 1 k
constexpr float kEps = 1e-6f;

__device__ __forceinline__ float butterfly(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Load VPT contiguous bf16 from g -> float regs r[], and copy (vectorized) to shared sh[].
template <int VPT>
__device__ __forceinline__ void load_chunk(const __nv_bfloat16* __restrict__ g,
                                           float* __restrict__ r, __nv_bfloat16* __restrict__ sh) {
    static_assert(VPT == 2 || VPT == 4 || VPT == 8, "VPT in {2,4,8}");
    if constexpr (VPT == 8) {
        const int4 v = *reinterpret_cast<const int4*>(g);
        *reinterpret_cast<int4*>(sh) = v;
        const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 f = __bfloat1622float2(b[i]);
            r[2 * i] = f.x;
            r[2 * i + 1] = f.y;
        }
    } else if constexpr (VPT == 4) {
        const int2 v = *reinterpret_cast<const int2*>(g);
        *reinterpret_cast<int2*>(sh) = v;
        const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            float2 f = __bfloat1622float2(b[i]);
            r[2 * i] = f.x;
            r[2 * i + 1] = f.y;
        }
    } else {
        const int v = *reinterpret_cast<const int*>(g);
        *reinterpret_cast<int*>(sh) = v;
        const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&v);
        float2 f = __bfloat1622float2(b);
        r[0] = f.x;
        r[1] = f.y;
    }
}

template <int T>
__global__ __launch_bounds__(T) void NormRopeFusedKernel(
    const __nv_bfloat16* __restrict__ q_raw, __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ qn, const __nv_bfloat16* __restrict__ kn,
    const float* __restrict__ cos_tab, const float* __restrict__ sin_tab,
    __nv_bfloat16* __restrict__ q_out, int pos, const __nv_bfloat16* __restrict__ x_in,
    const __nv_bfloat16* __restrict__ v_raw, __nv_bfloat16* __restrict__ v_out,
    const __nv_bfloat16* __restrict__ gate_raw, __nv_bfloat16* __restrict__ gate_out, int hid,
    float eps_in, const unsigned int* __restrict__ seqlen, __nv_bfloat16* __restrict__ v_cache) {
    constexpr int VPT = kHeadDim / T, W = T / 32;
    const int cta = blockIdx.x, tid = threadIdx.x;
    if (seqlen) pos = static_cast<int>(seqlen[1]);  // device-side seq position (decode graph)

    if (cta < kNheads) {  // ---- q/k head = cta: RMSNorm + partial RoPE (s cancels) ----
        // explicit 16B alignment: load_chunk stores int4 into &x_sh[d0]; without it the
        // alignment is layout luck that any extra __shared__ declared above would break
        __shared__ __align__(16) __nv_bfloat16 x_sh[kHeadDim], w_sh[kHeadDim];
        __shared__ float red[W > 1 ? W : 1];
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
        load_chunk<VPT>(src + d0, xr, &x_sh[d0]);
        load_chunk<VPT>(wt + d0, wr, &w_sh[d0]);
        float ss = 0.f;
#pragma unroll
        for (int e = 0; e < VPT; ++e) ss += xr[e] * xr[e];
        ss = butterfly(ss);
        if constexpr (W > 1) {
            const int warp = tid >> 5, lane = tid & 31;
            if (lane == 0) red[warp] = ss;
            __syncthreads();
            ss = 0.f;
#pragma unroll
            for (int w = 0; w < W; ++w) ss += red[w];
        } else
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
        __shared__ float sred[W > 1 ? W : 1];
        float ss = 0.f;
        for (int i = tid; i < hid; i += T) {
            const float xi = __bfloat162float(x_in[i]);
            ss += xi * xi;
        }
        ss = butterfly(ss);
        if constexpr (W > 1) {
            const int warp = tid >> 5, lane = tid & 31;
            if (lane == 0) sred[warp] = ss;
            __syncthreads();
            ss = 0.f;
#pragma unroll
            for (int w = 0; w < W; ++w) ss += sred[w];
        }
        const float s = rsqrtf(ss * (1.0f / hid) + eps_in);
        const __nv_bfloat16* vsrc =
            v_cache ? (v_cache + static_cast<size_t>(pos) * kHeadDim) : v_raw;
        __nv_bfloat16* vdst = v_cache ? (v_cache + static_cast<size_t>(pos) * kHeadDim) : v_out;
        for (int i = tid; i < kHeadDim; i += T)
            vdst[i] = __float2bfloat16(__bfloat162float(vsrc[i]) * s);
        for (int i = tid; i < kNqHeads * kHeadDim; i += T)
            gate_out[i] = __float2bfloat16(__bfloat162float(gate_raw[i]) * s);
    }
}

#ifndef NEXRT_MTP_RUNTIME_ONLY
template <int T>
int launch_norm_rope_fused(const void* q_raw, void* k_cache, const void* qn, const void* kn,
                           const float* cos_tab, const float* sin_tab, void* q_out, int pos,
                           const void* x_in, const void* v_raw, void* v_out, const void* gate_raw,
                           void* gate_out, int hid, float eps_in, void* stream) {
    NormRopeFusedKernel<T><<<kNheads + 1, T, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_raw), reinterpret_cast<__nv_bfloat16*>(k_cache),
        reinterpret_cast<const __nv_bfloat16*>(qn), reinterpret_cast<const __nv_bfloat16*>(kn),
        cos_tab, sin_tab, reinterpret_cast<__nv_bfloat16*>(q_out), pos,
        reinterpret_cast<const __nv_bfloat16*>(x_in), reinterpret_cast<const __nv_bfloat16*>(v_raw),
        reinterpret_cast<__nv_bfloat16*>(v_out), reinterpret_cast<const __nv_bfloat16*>(gate_raw),
        reinterpret_cast<__nv_bfloat16*>(gate_out), hid, eps_in, nullptr, nullptr);
    return static_cast<int>(cudaGetLastError());
}
#endif

// Device-driven-S: pos read from seqlen[1]; v read+written at v_cache base + pos*head_dim (on-device).
// k is still normalized in place at k_cache[pos]; gate scaled in place. By-value path passes seqlen=nullptr.
template <int T>
int launch_norm_rope_fused_dyn(const void* q_raw, void* k_cache, const void* qn, const void* kn,
                               const float* cos_tab, const float* sin_tab, void* q_out,
                               const void* seqlen, const void* x_in, void* v_cache,
                               const void* gate_raw, void* gate_out, int hid, float eps_in,
                               void* stream) {
    NormRopeFusedKernel<T><<<kNheads + 1, T, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_raw), reinterpret_cast<__nv_bfloat16*>(k_cache),
        reinterpret_cast<const __nv_bfloat16*>(qn), reinterpret_cast<const __nv_bfloat16*>(kn),
        cos_tab, sin_tab, reinterpret_cast<__nv_bfloat16*>(q_out), 0,
        reinterpret_cast<const __nv_bfloat16*>(x_in), nullptr, nullptr,
        reinterpret_cast<const __nv_bfloat16*>(gate_raw),
        reinterpret_cast<__nv_bfloat16*>(gate_out), hid, eps_in,
        reinterpret_cast<const unsigned int*>(seqlen), reinterpret_cast<__nv_bfloat16*>(v_cache));
    return static_cast<int>(cudaGetLastError());
}

#ifndef NEXRT_MTP_RUNTIME_ONLY
// ---- CP variant: 16 duplicate-Q heads (all q-heads of the group's single KV head), 4 gate heads (this
// rank's own), 1 K head. Same per-head math as NormRopeFusedKernel; only the Q count (16) and gate count
// (4) differ. Launched <<<18,T>>>: CTAs 0..15 = Q heads, 16 = K head, 17 = s + v-scale + 4-gate-scale.
// v is always at v_cache[pos] (qgate wrote it there); seqlen!=null -> device-S pos.

template <int T>
__global__ __launch_bounds__(T) void NormRopeFusedCpKernel(
    const __nv_bfloat16* __restrict__ q_raw, __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ qn, const __nv_bfloat16* __restrict__ kn,
    const float* __restrict__ cos_tab, const float* __restrict__ sin_tab,
    __nv_bfloat16* __restrict__ q_out, int pos, const __nv_bfloat16* __restrict__ x_in,
    const __nv_bfloat16* __restrict__ gate_raw, __nv_bfloat16* __restrict__ gate_out, int hid,
    float eps_in, const unsigned int* __restrict__ seqlen, __nv_bfloat16* __restrict__ v_cache) {
    constexpr int NQ = 16, NG = 4, NH = NQ + 1;  // 16 Q + 1 K q/k CTAs; CTA NH = s/v/gate
    constexpr int VPT = kHeadDim / T, W = T / 32;
    const int cta = blockIdx.x, tid = threadIdx.x;
    if (seqlen) pos = static_cast<int>(seqlen[1]);
    if (cta < NH) {  // q/k head = cta: RMSNorm + partial RoPE (s cancels)
        // explicit 16B alignment: load_chunk stores int4 into &x_sh[d0]; without it the
        // alignment is layout luck that any extra __shared__ declared above would break
        __shared__ __align__(16) __nv_bfloat16 x_sh[kHeadDim], w_sh[kHeadDim];
        __shared__ float red[W > 1 ? W : 1];
        const __nv_bfloat16* src;
        const __nv_bfloat16* wt;
        __nv_bfloat16* dst;
        if (cta < NQ) {
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
        load_chunk<VPT>(src + d0, xr, &x_sh[d0]);
        load_chunk<VPT>(wt + d0, wr, &w_sh[d0]);
        float ss = 0.f;
#pragma unroll
        for (int e = 0; e < VPT; ++e) ss += xr[e] * xr[e];
        ss = butterfly(ss);
        if constexpr (W > 1) {
            const int warp = tid >> 5, lane = tid & 31;
            if (lane == 0) red[warp] = ss;
            __syncthreads();
            ss = 0.f;
#pragma unroll
            for (int w = 0; w < W; ++w) ss += red[w];
        } else
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
    } else {  // CTA NH: s = rsqrt(mean(x^2)+eps_in); scale v & 4 gate heads
        __shared__ float sred[W > 1 ? W : 1];
        float ss = 0.f;
        for (int i = tid; i < hid; i += T) {
            const float xi = __bfloat162float(x_in[i]);
            ss += xi * xi;
        }
        ss = butterfly(ss);
        if constexpr (W > 1) {
            const int warp = tid >> 5, lane = tid & 31;
            if (lane == 0) sred[warp] = ss;
            __syncthreads();
            ss = 0.f;
#pragma unroll
            for (int w = 0; w < W; ++w) ss += sred[w];
        }
        const float s = rsqrtf(ss * (1.0f / hid) + eps_in);
        __nv_bfloat16* vp = v_cache + static_cast<size_t>(pos) * kHeadDim;
        for (int i = tid; i < kHeadDim; i += T)
            vp[i] = __float2bfloat16(__bfloat162float(vp[i]) * s);
        for (int i = tid; i < NG * kHeadDim; i += T)
            gate_out[i] = __float2bfloat16(__bfloat162float(gate_raw[i]) * s);
    }
}

template <int T>
int launch_norm_rope_fused_cp(const void* q_raw, void* k_cache, const void* qn, const void* kn,
                              const float* cos_tab, const float* sin_tab, void* q_out,
                              const void* seqlen, int pos, const void* x_in, void* v_cache,
                              const void* gate_raw, void* gate_out, int hid, float eps_in,
                              void* stream) {
    NormRopeFusedCpKernel<T><<<18, T, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_raw), reinterpret_cast<__nv_bfloat16*>(k_cache),
        reinterpret_cast<const __nv_bfloat16*>(qn), reinterpret_cast<const __nv_bfloat16*>(kn),
        cos_tab, sin_tab, reinterpret_cast<__nv_bfloat16*>(q_out), pos,
        reinterpret_cast<const __nv_bfloat16*>(x_in),
        reinterpret_cast<const __nv_bfloat16*>(gate_raw),
        reinterpret_cast<__nv_bfloat16*>(gate_out), hid, eps_in,
        reinterpret_cast<const unsigned int*>(seqlen), reinterpret_cast<__nv_bfloat16*>(v_cache));
    return static_cast<int>(cudaGetLastError());
}
#endif

// Near-empty control kernel: isolates pure CUDA-graph launch/replay overhead.

}  // namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
using namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY;
#ifndef NEXRT_MTP_RUNTIME_ONLY
extern "C" int NexN25ProNormRopeLaunch(const void* q_raw, void* k_cache, const void* qn,
                                       const void* kn, const float* cos_tab, const float* sin_tab,
                                       void* q_out, int pos, const void* x_in, const void* v_raw,
                                       void* v_out, const void* gate_raw, void* gate_out, int hid,
                                       float eps_in, void* stream) {
    return launch_norm_rope_fused<32>(q_raw, k_cache, qn, kn, cos_tab, sin_tab, q_out, pos, x_in,
                                      v_raw, v_out, gate_raw, gate_out, hid, eps_in, stream);
}
#endif

extern "C" int NexN25ProNormRopeDynLaunch(const void* q_raw, void* k_cache, const void* qn,
                                          const void* kn, const float* cos_tab,
                                          const float* sin_tab, void* q_out, const void* seqlen,
                                          const void* x_in, void* v_cache, const void* gate_raw,
                                          void* gate_out, int hid, float eps_in, void* stream) {
    return launch_norm_rope_fused_dyn<32>(q_raw, k_cache, qn, kn, cos_tab, sin_tab, q_out, seqlen,
                                          x_in, v_cache, gate_raw, gate_out, hid, eps_in, stream);
}

#ifndef NEXRT_MTP_RUNTIME_ONLY
extern "C" int NexN25ProCpNormRopeLaunch(const void* q_raw, void* k_cache, const void* qn,
                                         const void* kn, const float* cos_tab, const float* sin_tab,
                                         void* q_out, int pos, const void* x_in, void* v_cache,
                                         const void* gate_raw, void* gate_out, int hid,
                                         float eps_in, void* stream) {
    return launch_norm_rope_fused_cp<32>(q_raw, k_cache, qn, kn, cos_tab, sin_tab, q_out, nullptr,
                                         pos, x_in, v_cache, gate_raw, gate_out, hid, eps_in,
                                         stream);
}
extern "C" int NexN25ProCpNormRopeDynLaunch(const void* q_raw, void* k_cache, const void* qn,
                                            const void* kn, const float* cos_tab,
                                            const float* sin_tab, void* q_out, const void* seqlen,
                                            const void* x_in, void* v_cache, const void* gate_raw,
                                            void* gate_out, int hid, float eps_in, void* stream) {
    return launch_norm_rope_fused_cp<32>(q_raw, k_cache, qn, kn, cos_tab, sin_tab, q_out, seqlen, 0,
                                         x_in, v_cache, gate_raw, gate_out, hid, eps_in, stream);
}

extern "C" const char* nex_n2_5_pro_norm_rope_cuda_error_string(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
#endif
