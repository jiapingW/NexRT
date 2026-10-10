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

// M=4-native FUSED attention o-projection push/reduce -- ONE launch replacing a
// split [producer + reduce] pair.
//
// The quad-attention combine has already produced BF16 gated activations. The
// tensor-core projection reuses one lane-major weight pass across all four rows;
// the shared M=4 LL protocol sums BF16 payloads in fixed rank order in FP32.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M4
#endif

#include "../common/m4_oproj_fused_protocol.cuh"

namespace nexrt::sm90::nex_n2_5_pro::M4 {
constexpr int kM4 = kM4Fused;
constexpr int kProdCTAs = kFusedCTAs;
constexpr int kRowsStripe = 16;

// BF16 tensor-core fragment helper.
// The producer below keeps the baseline slot layout (part[4][32], 16-row stripes at
// +0/+2048) and run the GEMM as a transposed-mma core on the lane-major W prepack
// (pack_gemm_bf16tc(W, 1024)): one coalesced 8 B/lane load per k32, no W smem, e4m3
// decoded once via the value-exact f16 transit; x_eff rounded once to bf16 for the mma
// B operand (the ONLY class change vs the fp32-SIMT base), products fp32-accumulated.
__device__ __forceinline__ void mma_m16n8k16_bf16_op_(float (&c)[4], const unsigned (&a)[4],
                                                      const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Selected 512-thread split-K geometry.
// 512 threads form 16 warps: four row tiles by four K slices of 256.
// blocks per warp). BIT-EXACT vs the 1024-thread form: c0 accumulates k32 tiles 0..3 (==
// baseline K-warp 2kw chain), c1 tiles 4..7 (== warp 2kw+1); the fold consumes
// (kw,c0),(kw,c1) for kw = 0..3 == the baseline w8 = 0..7 fp32 addition order, and the
// part[] slot mapping (tile*8 + gid) is unchanged.
__global__ __launch_bounds__(512, 1) void M4AttnOProjKernel(
    const __nv_bfloat16* __restrict__ xeff4,  // [4, 1024] precomputed bf16
    const __nv_fp8_e4m3* __restrict__ Wpack, const float* __restrict__ w_scales,
    unsigned long long rep_mc, const float* __restrict__ rep_local,
    unsigned int* __restrict__ it_ctr, int* __restrict__ status,
    __nv_bfloat16* __restrict__ x4_resid, int rank, int num_ranks) {
    constexpr int kXPitchB = kLocal + 8;
    __shared__ __align__(16) __nv_bfloat16 x_eff_b[kM4][kXPitchB];
    __shared__ __align__(16) float part[kM4][kFusedSlots];
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
        const int j = threadIdx.x >> 7;  // 16 B/thread copy
        const int k8 = (threadIdx.x & 127) * 8;
        *reinterpret_cast<uint4*>(&x_eff_b[j][k8]) =
            *reinterpret_cast<const uint4*>(xeff4 + j * kLocal + k8);
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
                unsigned b[2] = {0u, 0u};
                if (gid < kM4) {
                    const __nv_bfloat16* xp =
                        &x_eff_b[gid][kw * 256 + t32 * 32 + s_ * 16 + tig * 2];
                    b[0] = *reinterpret_cast<const unsigned*>(xp);
                    b[1] = *reinterpret_cast<const unsigned*>(xp + 8);
                }
                const unsigned a[4] = {awf[s_ * 2], 0u, awf[s_ * 2 + 1], 0u};
                if (half == 0) {
                    mma_m16n8k16_bf16_op_(c0, a, b);
                } else {
                    mma_m16n8k16_bf16_op_(c1, a, b);
                }
            }
        }
    }
    red[local_warp][lane][0] = s0 * c0[0];
    red[local_warp][lane][1] = s0 * c0[1];
    red[local_warp][lane][2] = s1 * c1[0];
    red[local_warp][lane][3] = s1 * c1[1];
    __syncthreads();

    if (local_warp < 4 && tig * 2 < kM4) {  // baseline w8 = 0..7 addition order
        float v0 = 0.0f, v1 = 0.0f;
#pragma unroll
        for (int kw4 = 0; kw4 < 4; ++kw4) {
            v0 += red[local_warp * 4 + kw4][lane][0];
            v0 += red[local_warp * 4 + kw4][lane][2];
            v1 += red[local_warp * 4 + kw4][lane][1];
            v1 += red[local_warp * 4 + kw4][lane][3];
        }
        part[tig * 2][local_warp * 8 + gid] = v0;
        part[tig * 2 + 1][local_warp * 8 + gid] = v1;
    }
    __syncthreads();

    m4_oproj_publish_and_reduce_ll(part, rep_mc, rep_local, status, x4_resid, rank, num_ranks,
                                   cta_gen, row_base);
}

extern "C" int NexN25ProM4AttnOProjLaunch(const void* xeff4, const void* Wpack,
                                          const void* w_scales, unsigned long long rep_mc,
                                          const void* rep_local, void* it_ctr, void* status,
                                          void* x4_resid, int rank, int num_ranks, void* stream) {
    M4AttnOProjKernel<<<dim3(kProdCTAs), dim3(512), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(xeff4), static_cast<const __nv_fp8_e4m3*>(Wpack),
        static_cast<const float*>(w_scales), rep_mc, static_cast<const float*>(rep_local),
        static_cast<unsigned int*>(it_ctr), static_cast<int*>(status),
        static_cast<__nv_bfloat16*>(x4_resid), rank, num_ranks);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M4
