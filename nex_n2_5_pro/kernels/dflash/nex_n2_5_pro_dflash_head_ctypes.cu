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

// DFlash draft lm_head: bf16 logits GEMM + local greedy argmax, feeding the
// DfArgmaxLL cross-rank combine (nex_n2_5_pro_dflash_argmax_ctypes.cu).
//
//   K1 DfHeadGemm    logits8 [8, 31040] bf16 = xn . lm_raw_r^T. The M8 target
//                    head form (M8LmHeadBf16TcKernel) without the inv_rms
//                    epilogue: the draft's final RMSNorm + gamma is
//                    a SEPARATE kernel (DfRmsNorm, user design), so xn arrives
//                    pre-normed and the pack is a PURE PERMUTATION of the RAW
//                    lm_head shard (shared with sglang-prefill; no gamma fold,
//                    no third value-copy). Grid 1940 x 256 (8 warps = K-split-8,
//                    one 16-vocab-row tile per CTA, ~15 waves); weights stream
//                    register-direct via ld.global.nc.L1::no_allocate.v4 with a
//                    4-deep prefetch ring (no smem staging for weights); per-warp
//                    xn K-slice staged with cp.async.ca + __syncwarp only.
//                    Logits stored bf16: the reference path (torch bf16 GEMM)
//                    and sglang's draft sampler both compare bf16 logit values.
//                    All 8 n columns are real draft rows; row 0 (anchor) is
//                    computed uniformly and never read (row-invariance rule).
//   K2 DfHeadArgmax  rows 1..7: (val fp32, idx i32) = first-max over the 31040
//                    bf16 logits, i64 local ids (torch tie rule: strict >, lowest index on
//                    equality), 7 CTAs x 1024 thr, uint4 scan in fixed strided
//                    order + fixed reduction tree -> deterministic. Output feeds
//                    NexN25ProDFlashArgmaxLLLaunch unchanged.
//
// The head uses the serial launch form.
// DF-family TU: draft-only; no BS=1/M4/M8 production symbol here.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY DF
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "../common/pdl.cuh"

namespace nexrt::sm90::nex_n2_5_pro::DFlash {
constexpr int kTok = 8;
constexpr int kHid = 4096;
constexpr int kRowsLocal = 31040;            // vocab / TP = 248320 / 8
constexpr int kHeadTiles = kRowsLocal / 16;  // 1940

__device__ __forceinline__ void df_ldg_nc16(uint4& v, const uint4* p) {
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
}
__device__ __forceinline__ void df_mma16816h(float (&c)[4], const uint4& a,
                                             const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b[0]), "r"(b[1]));
}

// ---- K1: eight-row head GEMM, 16 real vocab rows per tile (M8 head form) ----
__global__ __launch_bounds__(256) void DfHeadGemmKernel(
    const __nv_bfloat16* __restrict__ xn,     // [8, 4096] final-normed rows
    const __nv_bfloat16* __restrict__ wpack,  // pack_lmhead law of the RAW shard
    __nv_bfloat16* __restrict__ logits8) {    // [8, 31040] bf16
    constexpr int kWarpK = kHid / 8;          // 512 = 32 k16 steps per warp
    constexpr int kXPitch = kWarpK * 2 + 16;  // 1040 B
    // 70,656 B total -> dynamic smem (static allocations cap at 48 KB)
    extern __shared__ __align__(16) char dsm[];
    char (*sh)[kTok][kXPitch] = reinterpret_cast<char (*)[kTok][kXPitch]>(dsm);
    float (*red)[32][4] = reinterpret_cast<float (*)[32][4]>(dsm + 8 * kTok * kXPitch);
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int gid = lane >> 2;
    const int tig = lane & 3;
    const int tile = blockIdx.x;
    const int k0 = warp * kWarpK;
    char* const shw = &sh[warp][0][0];  // dynamic-smem carve

    // Weight prefetch ring: graph-static bytes, issued BEFORE the grid sync.
    const uint4* wp_base = reinterpret_cast<const uint4*>(wpack) +
                           (static_cast<size_t>(tile) * 256 + (k0 >> 4)) * 32 + lane;
    uint4 wbuf[4];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        df_ldg_nc16(wbuf[i], wp_base + static_cast<size_t>(i) * 32);
    }
    NEXRT_PDL_ENTRY_SYNC();  // xn is the predecessor's output

#pragma unroll
    for (int m = 0; m < kTok; ++m) {  // this warp's xn K-slice: 8 x 1 KB
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t dst = static_cast<uint32_t>(
                __cvta_generic_to_shared(shw + m * kXPitch + h * 512 + lane * 16));
            const void* src = xn + static_cast<size_t>(m) * kHid + k0 + (h * 512 + lane * 16) / 2;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
        }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncwarp();

    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int s_ = 0; s_ < 32; ++s_) {  // 32 k16 steps, 1 uint4 + 1 mma each
        if (s_ + 3 < 32) {
            df_ldg_nc16(wbuf[(s_ + 3) & 3], wp_base + static_cast<size_t>(s_ + 3) * 32);
        }
        const uint4 w = wbuf[s_ & 3];
        const char* xp = shw + gid * kXPitch + s_ * 32 + tig * 4;
        unsigned b[2];
        b[0] = *reinterpret_cast<const unsigned*>(xp);
        b[1] = *reinterpret_cast<const unsigned*>(xp + 16);
        df_mma16816h(c, w, b);
    }

#pragma unroll
    for (int e = 0; e < 4; ++e) {
        red[warp][lane][e] = c[e];
    }
    __syncthreads();
    if (threadIdx.x < 32) {  // 16 real rows; tokens tig*2, tig*2+1
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int sp = 0; sp < 8; ++sp) {  // fixed K order
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                v[e] += red[sp][lane][e];
            }
        }
        const size_t r_lo = static_cast<size_t>(tile) * 16 + gid;
        logits8[static_cast<size_t>(tig * 2) * kRowsLocal + r_lo] = __float2bfloat16(v[0]);
        logits8[static_cast<size_t>(tig * 2 + 1) * kRowsLocal + r_lo] = __float2bfloat16(v[1]);
        logits8[static_cast<size_t>(tig * 2) * kRowsLocal + r_lo + 8] = __float2bfloat16(v[2]);
        logits8[static_cast<size_t>(tig * 2 + 1) * kRowsLocal + r_lo + 8] = __float2bfloat16(v[3]);
    }
}

// ---- K2: per-row first-max over the bf16 logits (rows 1..7) -----------------
constexpr int kAmThreads = 1024;
constexpr int kRowU4 = kRowsLocal / 8;  // 3880 uint4 per row (31040 % 8 == 0)
static_assert(kRowsLocal % 8 == 0, "uint4 scan");

__global__ __launch_bounds__(kAmThreads) void DfHeadArgmaxKernel(
    const __nv_bfloat16* __restrict__ logits8,  // [8, 31040] bf16
    float* __restrict__ val7,                   // [7] fp32
    long* __restrict__ idx7) {                  // [7] i64 local ids (DfArgmaxLL contract)
    NEXRT_PDL_ENTRY_SYNC();
    __shared__ float sv[kAmThreads / 32];
    __shared__ int si[kAmThreads / 32];
    const int row = blockIdx.x + 1;  // draft rows 1..7
    const uint4* lr =
        reinterpret_cast<const uint4*>(logits8 + static_cast<size_t>(row) * kRowsLocal);
    float bv = -__int_as_float(0x7f800000);  // -inf
    int bi = 0;
    for (int u = threadIdx.x; u < kRowU4; u += kAmThreads) {  // fixed strided order
        const uint4 q = lr[u];
        const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&q);
#pragma unroll
        for (int j = 0; j < 8; ++j) {  // ascending index: strict > keeps the first max
            const float v = __bfloat162float(e[j]);
            if (v > bv) {
                bv = v;
                bi = u * 8 + j;
            }
        }
    }
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {  // fixed tree; lower index wins ties
        const float ov = __shfl_down_sync(0xffffffffu, bv, o);
        const int oi = __shfl_down_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0) {
        sv[threadIdx.x >> 5] = bv;
        si[threadIdx.x >> 5] = bi;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
        for (int w = 1; w < kAmThreads / 32; ++w) {  // fixed ascending warp order
            if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) {
                bv = sv[w];
                bi = si[w];
            }
        }
        val7[row - 1] = bv;
        idx7[row - 1] = static_cast<long>(bi);
    }
}
}  // namespace nexrt::sm90::nex_n2_5_pro::DFlash
using namespace nexrt::sm90::nex_n2_5_pro::DFlash;
extern "C" {

constexpr int kHeadDyn =
    8 * kTok * (kHid / 8 * 2 + 16) + 8 * 32 * 4 * static_cast<int>(sizeof(float));

int NexN25ProDFlashHeadGemmLaunch(const void* xn, const void* wpack, void* logits8, int pdl,
                                  void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    static bool ready = false;
    if (!ready) {
        int rc = static_cast<int>(
            cudaFuncSetAttribute(reinterpret_cast<const void*>(&DfHeadGemmKernel),
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, kHeadDyn));
        if (rc) return rc;
        ready = true;
    }
    if (pdl) return static_cast<int>(cudaErrorInvalidValue);
    DfHeadGemmKernel<<<dim3(kHeadTiles), dim3(256), kHeadDyn, s>>>(
        static_cast<const __nv_bfloat16*>(xn), static_cast<const __nv_bfloat16*>(wpack),
        static_cast<__nv_bfloat16*>(logits8));
    return static_cast<int>(cudaGetLastError());
}

int NexN25ProDFlashHeadArgmaxLaunch(const void* logits8, void* val7, void* idx7, int pdl,
                                    void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    if (pdl) return static_cast<int>(cudaErrorInvalidValue);
    DfHeadArgmaxKernel<<<dim3(kTok - 1), dim3(kAmThreads), 0, s>>>(
        static_cast<const __nv_bfloat16*>(logits8), static_cast<float*>(val7),
        static_cast<long*>(idx7));
    return static_cast<int>(cudaGetLastError());
}

int NexN25ProDFlashHeadRowsLocal() { return kRowsLocal; }
}
