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

// M=8 attention o-projection: fp8 o_proj GEMM on the pre-gated x_eff + TP8
// push/reduce + residual fold for eight ordered target rows in ONE launch.
//
// M=8-FAMILY TU: native eight-row implementation, no BS=1/M=4 production source.
// Derived from the M=4 transposed-mma geometry. The gating
// (ctx * sigmoid(gate)) is hoisted into the M=8 combine-xeff epilogue
// (nex_n2_5_pro_m8_attn_core_ctypes.cu), so this producer consumes precomputed bf16 xeff8,
// runs the 512-thread 16-warp (4 row-tiles x 4 K-slices) transposed-mma GEMM on the
// lane-major fp8 prepack, and publishes through the M=8 LL peer inbox
// (kernels/common/m8_oproj_fused_protocol.cuh).
//
// THE M=8 STRUCTURAL WIN: all eight mma n-slots are real tokens (the M=4 form
// zero-padded half).

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M8
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace nexrt::sm90::nex_n2_5_pro::M8 {
constexpr int kDim = 4096;
constexpr int kLocal = 1024;
constexpr int kScaleCols = kLocal / 128;  // 8
constexpr int kMaxRanks = 8;
constexpr int kMaxSpins = 1 << 30;

__device__ __forceinline__ int ld_acq_sys_i32(const int* ptr) {
    int ret;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr) : "memory");
    return ret;
}
__device__ __forceinline__ void multimem_red_add_rel_sys(unsigned long long mc, int val) {
    asm volatile("multimem.red.release.sys.global.add.s32 [%0], %1;" ::"l"(mc), "r"(val)
                 : "memory");
}
__device__ __forceinline__ void multimem_st_v4_f32(unsigned long long mc, float4 v) {
    asm volatile("multimem.st.weak.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(mc), "f"(v.x),
                 "f"(v.y), "f"(v.z), "f"(v.w)
                 : "memory");
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M8
#include "../common/m8_oproj_fused_protocol.cuh"

namespace nexrt::sm90::nex_n2_5_pro::M8 {
constexpr int kM8 = kM8Fused;            // 8
constexpr int kProdCTAs = kM8FusedCTAs;  // 128
constexpr int kRowsStripe = 16;

__device__ __forceinline__ void m8_mma_m16n8k16_bf16_op(float (&c)[4], const unsigned (&a)[4],
                                                        const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Wpack contract: identical index math to the M=8 GDN o_proj pack --
//   Wpack[((row >> 3) * (kLocal/32) + k32) * 256 + lane * 8 .. +7].
__global__ __launch_bounds__(512, 1) void M8AttnOProjKernel(
    const __nv_bfloat16* __restrict__ xeff8,  // [8, 1024] precomputed bf16
    const __nv_fp8_e4m3* __restrict__ Wpack,  // M8 lane-major prepack of o_w
    const float* __restrict__ w_scales,       // [32, 8]
    unsigned long long rep_mc, const float* __restrict__ rep_local,
    unsigned int* __restrict__ it_ctr,  // [128] per-CTA epoch (M=8-owned)
    int* __restrict__ status,
    __nv_bfloat16* __restrict__ x8_resid,  // [8, 4096] folded in place
    int rank, int num_ranks) {
    constexpr int kXPitchB = kLocal + 8;
    __shared__ __align__(16) __nv_bfloat16 x_eff_b[kM8][kXPitchB];
    __shared__ float red[16][32][4];
    __shared__ int cta_gen;
    const int lane = threadIdx.x & 31;
    const int local_warp = threadIdx.x >> 5;
    const int gid = lane >> 2;
    const int tig = lane & 3;
    const int row_base = blockIdx.x * kRowsStripe;

    if (threadIdx.x == 0) {
        cta_gen = static_cast<int>(atomicAdd(&it_ctr[blockIdx.x], 1u)) + 1;
    }
    {
// 16 B/thread copy of xeff8: 8 rows x 1024 elems at 512 threads = 2 passes.
#pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            const int idx = pass * 512 + threadIdx.x;
            const int j = idx >> 7;
            const int k8 = (idx & 127) * 8;
            *reinterpret_cast<uint4*>(&x_eff_b[j][k8]) =
                *reinterpret_cast<const uint4*>(xeff8 + j * kLocal + k8);
        }
    }

    const int tile = local_warp >> 2;  // 4 row-tiles of 8
    const int kw = local_warp & 3;     // 4 K-slices of 256
    const int row0 = (tile < 2) ? row_base + tile * 8 : row_base + (kDim / 2) + (tile - 2) * 8;
    const float s0 = w_scales[(row0 >> 7) * kScaleCols + kw * 2];
    const float s1 = w_scales[(row0 >> 7) * kScaleCols + kw * 2 + 1];
    const __nv_fp8_e4m3* wp_base =
        Wpack + ((static_cast<size_t>(row0) >> 3) * (kLocal / 32) + kw * 8) * 256 + lane * 8;
    uint2 wreg[8];
#pragma unroll
    for (int t32 = 0; t32 < 8; ++t32) {
        asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];"
                     : "=r"(wreg[t32].x), "=r"(wreg[t32].y)
                     : "l"(wp_base + t32 * 256));
    }
    __syncthreads();

    float c0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float c1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int t4 = 0; t4 < 4; ++t4) {
            const int t32 = half * 4 + t4;
            const uint2 raw = wreg[t32];
            const __nv_fp8x2_e4m3* wp = reinterpret_cast<const __nv_fp8x2_e4m3*>(&raw);
            unsigned awf[4];
#pragma unroll
            for (int p_ = 0; p_ < 4; ++p_) {
                const __nv_bfloat162 bb = __float22bfloat162_rn(static_cast<float2>(wp[p_]));
                awf[p_] = *reinterpret_cast<const unsigned*>(&bb);
            }
#pragma unroll
            for (int s_ = 0; s_ < 2; ++s_) {
                // Every n column is a real token at M=8: no zero-pad guard.
                const __nv_bfloat16* xp = &x_eff_b[gid][kw * 256 + t32 * 32 + s_ * 16 + tig * 2];
                unsigned b[2];
                b[0] = *reinterpret_cast<const unsigned*>(xp);
                b[1] = *reinterpret_cast<const unsigned*>(xp + 8);
                const unsigned a[4] = {awf[s_ * 2], 0u, awf[s_ * 2 + 1], 0u};
                if (half == 0) {
                    m8_mma_m16n8k16_bf16_op(c0, a, b);
                } else {
                    m8_mma_m16n8k16_bf16_op(c1, a, b);
                }
            }
        }
    }
    red[local_warp][lane][0] = s0 * c0[0];
    red[local_warp][lane][1] = s0 * c0[1];
    red[local_warp][lane][2] = s1 * c1[0];
    red[local_warp][lane][3] = s1 * c1[1];
    __syncthreads();

    if (local_warp < 4) {  // K-slice fold, kw ascending
        float v0 = 0.0f, v1 = 0.0f;
#pragma unroll
        for (int kw4 = 0; kw4 < 4; ++kw4) {
            v0 += red[local_warp * 4 + kw4][lane][0];
            v0 += red[local_warp * 4 + kw4][lane][2];
            v1 += red[local_warp * 4 + kw4][lane][1];
            v1 += red[local_warp * 4 + kw4][lane][3];
        }
        m8_oproj_fold_publish_combine_ll(v0, v1, local_warp * 8 + gid, tig, rep_mc, rep_local,
                                         status, x8_resid, rank, num_ranks, cta_gen, row_base);
    }
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M8
using namespace nexrt::sm90::nex_n2_5_pro::M8;
extern "C" {
// xeff8 [8,1024] bf16 (from the M8 combine-xeff); Wpack = M8 lane-major prepack of
// attn o_w [4096,1024] fp8; w_scales [32,8]; rep/it_ctr = M=8-owned protocol state;
// x8_resid [8,4096] bf16 folded in place.
int NexN25ProM8AttnOProjLaunch(const void* xeff8, const void* Wpack, const void* w_scales,
                               unsigned long long rep_mc, const void* rep_local, void* it_ctr,
                               void* status, void* x8_resid, int rank, int num_ranks,
                               void* stream) {
    M8AttnOProjKernel<<<dim3(kProdCTAs), dim3(512), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(xeff8), static_cast<const __nv_fp8_e4m3*>(Wpack),
        static_cast<const float*>(w_scales), rep_mc, static_cast<const float*>(rep_local),
        static_cast<unsigned int*>(it_ctr), static_cast<int*>(status),
        static_cast<__nv_bfloat16*>(x8_resid), rank, num_ranks);
    return static_cast<int>(cudaGetLastError());
}
int NexN25ProM8AttnOProjRows() { return kM8; }
}
