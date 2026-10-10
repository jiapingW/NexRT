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

// M=4 lm_head GEMM (one TP rank) -- FOUR final hidden rows -> FOUR TRUE-logit rows in one pass
// over the weight shard.
//
//   logits4[j][row] = inv_rms[j] * ( h4[j] . W_fused[row] )      j = 0..3, row = 0..31040
//
// Same column/vocab-parallel layout as the shipped M1 head (nex_n2_5_pro_lmhead_gemv_ctypes.cu): h rows are
// REPLICATED raw residual-stream vectors [4096] bf16; W_fused = (1+norm_w) (x) lm_head_w is this
// rank's vocab shard [31040,4096] bf16, so every logit is a complete dot with NO cross-rank reduce.
//
// The M=4 point: the M1 kernel is HBM-bound on the 254 MB weight shard. Here each CTA loads its
// W_fused row ONCE (bf16x2 register loads, fp32 accumulate -- identical arithmetic to M1) and reuses
// every loaded weight pair against ALL FOUR hidden rows (4 accumulators), so four logit rows cost one
// weight pass instead of four. This is a genuine fused M=4 kernel, not four M1 invocations.
//
// Two-kernel launch inside one C call, ordered on the caller's stream:
//   1. M4LmHeadRmsKernel  (1 CTA): inv_rms[j] = rsqrt(mean(h4[j]^2) + eps) for j=0..3 -- the exact
//      M1 semantics (LmHeadGemvRowRmsCta0Kernel's CTA0 reduce), for four rows.
//   2. M4LmHeadGemmKernel (grid 1940 x 256): the fused 4-row GEMM; applies inv_rms in the epilogue so the
//      output rows are TRUE logits (fp32), directly comparable and top-k-ready.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M4
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

namespace nexrt::sm90::nex_n2_5_pro::M4 {
constexpr int kDim = 4096;
constexpr int kDimPairs = kDim / 2;  // 2048 __nv_bfloat162 pairs
constexpr int kRows = 31040;         // vocab / TP = 248320 / 8
constexpr int kM = 4;
constexpr float kInvDim = 1.0f / static_cast<float>(kDim);
static_assert(kDim % 2 == 0, "kDim must be even for bf16x2");

// --- 1. per-row inv_rms for the four raw hidden rows (exact M1 RMS semantics) ----------------
template <int NT>
__global__ __launch_bounds__(NT) void M4LmHeadRmsKernel(const __nv_bfloat16* __restrict__ h4,
                                                        float* __restrict__ inv_rms, float eps) {
    __shared__ typename cub::BlockReduce<float, NT>::TempStorage br_tmp;
    for (int j = 0; j < kM; ++j) {
        const auto* __restrict__ h2 =
            reinterpret_cast<const __nv_bfloat162*>(h4 + static_cast<size_t>(j) * kDim);
        float rms_acc = 0.0f;
        for (int p = threadIdx.x; p < kDimPairs; p += NT) {
            const float2 hf = __bfloat1622float2(h2[p]);
            rms_acc += hf.x * hf.x + hf.y * hf.y;
        }
        const float sq = cub::BlockReduce<float, NT>(br_tmp).Sum(rms_acc);
        if (threadIdx.x == 0) {
            inv_rms[j] = rsqrtf(sq * kInvDim + eps);
        }
        __syncthreads();  // br_tmp reuse across rows
    }
}

// The selected W-rows-as-A tensor-core mapping uses this BF16 MMA primitive.
__device__ __forceinline__ void mma_m16n8k16_bf16(float (&c)[4], const uint32_t (&a)[4],
                                                  const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Production tensor-core projection. The lane-major prepack supplies one
// coalesced uint4 fragment per k16 step; all sixteen vocabulary rows are real.
__global__ __launch_bounds__(256) void M4LmHeadGemmKernel(const __nv_bfloat16* __restrict__ h4,
                                                          const __nv_bfloat16* __restrict__ Wpack,
                                                          float* __restrict__ logits4,
                                                          const float* __restrict__ inv_rms) {
    constexpr int kWarpK = kDim / 8;          // 512 = 32 k16 steps
    constexpr int kXPitch = kWarpK * 2 + 16;  // 1040 B
    __shared__ __align__(16) char sh[8][kM][kXPitch];
    __shared__ float red[8][32][4];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int gid = lane >> 2;
    const int tig = lane & 3;
    const int tile = blockIdx.x;  // 16 vocab rows
    const int k0 = warp * kWarpK;
    char* const shw = &sh[warp][0][0];

#pragma unroll
    for (int m = 0; m < kM; ++m) {  // h slice: 4 rows x 1 KB, one shot
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t dst = static_cast<uint32_t>(
                __cvta_generic_to_shared(shw + m * kXPitch + h * 512 + lane * 16));
            const void* src = h4 + static_cast<size_t>(m) * kDim + k0 + (h * 512 + lane * 16) / 2;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
        }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncwarp();

    const uint4* wp_base = reinterpret_cast<const uint4*>(Wpack) +
                           (static_cast<size_t>(tile) * 256 + (k0 >> 4)) * 32 + lane;
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint4 wbuf[4];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(wbuf[i].x), "=r"(wbuf[i].y), "=r"(wbuf[i].z), "=r"(wbuf[i].w)
                     : "l"(wp_base + static_cast<size_t>(i) * 32));
    }
#pragma unroll
    for (int s_ = 0; s_ < 32; ++s_) {  // 32 k16 steps, 1 uint4 + 1 mma each
        if (s_ + 3 < 32) {
            const int nx = s_ + 3;
            asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(wbuf[nx & 3].x), "=r"(wbuf[nx & 3].y), "=r"(wbuf[nx & 3].z),
                           "=r"(wbuf[nx & 3].w)
                         : "l"(wp_base + static_cast<size_t>(nx) * 32));
        }
        const uint4 w = wbuf[s_ & 3];
        unsigned b[2] = {0u, 0u};
        if (gid < kM) {
            const char* xp = shw + gid * kXPitch + s_ * 32 + tig * 4;
            b[0] = *reinterpret_cast<const unsigned*>(xp);
            b[1] = *reinterpret_cast<const unsigned*>(xp + 16);
        }
        const uint32_t a[4] = {w.x, w.y, w.z, w.w};
        mma_m16n8k16_bf16(c, a, b);
    }

#pragma unroll
    for (int e = 0; e < 4; ++e) {
        red[warp][lane][e] = c[e];
    }
    __syncthreads();
    if (threadIdx.x < 32 && tig * 2 < kM) {  // 16 real rows; tokens tig*2, tig*2+1
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int sp = 0; sp < 8; ++sp) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                v[e] += red[sp][lane][e];
            }
        }
        const float ir0 = inv_rms[tig * 2];
        const float ir1 = inv_rms[tig * 2 + 1];
        const size_t r_lo = static_cast<size_t>(tile) * 16 + gid;
        logits4[static_cast<size_t>(tig * 2) * kRows + r_lo] = v[0] * ir0;
        logits4[static_cast<size_t>(tig * 2 + 1) * kRows + r_lo] = v[1] * ir1;
        logits4[static_cast<size_t>(tig * 2) * kRows + r_lo + 8] = v[2] * ir0;
        logits4[static_cast<size_t>(tig * 2 + 1) * kRows + r_lo + 8] = v[3] * ir1;
    }
}

extern "C" {
// bf16tc: Wf must be the lane-major prepack (pack_lmhead_bf16tc), not row-major W_fused.
int NexN25ProM4LmHeadGemmLaunch(const void* h4, const void* Wpack, void* logits4, float* inv_rms4,
                                float eps, void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    M4LmHeadRmsKernel<256><<<1, 256, 0, s>>>(static_cast<const __nv_bfloat16*>(h4), inv_rms4, eps);
    const int rc = static_cast<int>(cudaGetLastError());
    if (rc != 0) {
        return rc;
    }
    M4LmHeadGemmKernel<<<dim3(kRows / 16), dim3(256), 0, s>>>(
        static_cast<const __nv_bfloat16*>(h4), static_cast<const __nv_bfloat16*>(Wpack),
        static_cast<float*>(logits4), inv_rms4);
    return static_cast<int>(cudaGetLastError());
}

}  // extern "C"

}  // namespace nexrt::sm90::nex_n2_5_pro::M4
