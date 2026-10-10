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

// M=8 GDN o-projection: gated head-RMS staging + fp8 o_proj GEMM + TP8 push/reduce +
// residual fold for eight ordered target rows in ONE launch.
//
// M=8-FAMILY TU: native eight-row implementation, no BS=1/M=4 production source.
// Derived from the M=4 transposed-mma geometry and reworked for eight rows:
// overlap producer (activation loads FIRST to feed the barrier, then the eager W
// burst register-resident, head-RMS re-reads under the outstanding W latency),
// bf16 transposed-mma GEMM on the lane-major fp8 prepack, and the M=8 LL peer
// inbox protocol (kernels/common/m8_oproj_fused_protocol.cuh).
//
// THE M=8 STRUCTURAL WIN: the mma B operand's n=8 token columns are ALL real (the
// M=4 form zero-padded half), so each weight fragment retires twice the useful work
// at unchanged weight traffic.
//
// Staging chain per element (arithmetic class of the family, preserved exactly):
//   rec -> float -> * head_rms -> bf16 -> * (norm_w -> float) -> bf16 ->
//   * silu(z) -> bf16 smem mirror (value-exact: the fp32 form also rounded there).

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
constexpr int kHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kScaleCols = kLocal / 128;  // 8
constexpr int kMaxRanks = 8;
constexpr int kMaxSpins = 1 << 30;
constexpr float kRmsEps = 1.0e-6f;

__device__ __forceinline__ float m8_warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float m8_silu_f32(float x) {
    // Reciprocal-then-multiply, exactly the family's rounding chain (NOT x / (1+e)).
    const float sigmoid = 1.0f / (1.0f + __expf(-x));
    return x * sigmoid;
}

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
constexpr int kProdThreads = 1024;
constexpr int kRowsStripe = 16;

__device__ __forceinline__ void m8_mma_m16n8k16_bf16_op(float (&c)[4], const unsigned (&a)[4],
                                                        const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Wpack contract (built once by the M=8 runtime from the fp8 row-major [4096, 1024]
// o_proj weight): lane-major fragment pack with 8-row tiles and k32 blocks --
//   Wpack[((row_tile) * (kLocal/32) + k32) * 256 + lane * 8 .. +7],  row_tile = row >> 3.

__global__ __launch_bounds__(kProdThreads, 1) void M8GdnOProjKernel(
    const __nv_bfloat16* __restrict__ rec8,    // [8, 1024]
    const __nv_bfloat16* __restrict__ z8,      // [8, 1024]
    const __nv_bfloat16* __restrict__ norm_w,  // [128]
    const __nv_fp8_e4m3* __restrict__ Wpack,   // M8 lane-major prepack of o_w
    const float* __restrict__ w_scales,        // [32, 8]
    unsigned long long rep_mc, const float* __restrict__ rep_local,
    unsigned int* __restrict__ it_ctr,  // [128] per-CTA epoch (M=8-owned)
    int* __restrict__ status,
    __nv_bfloat16* __restrict__ x8_resid,  // [8, 4096] residual, folded in place
    int rank, int num_ranks) {
    constexpr int kXPitchB = kLocal + 8;
    __shared__ __align__(16) __nv_bfloat16 x_eff_b[kM8][kXPitchB];
    __shared__ __align__(16) float rms8[kM8][kHeads];
    __shared__ float red[32][32][2];
    __shared__ int cta_gen;
    const int lane = threadIdx.x & 31;
    const int local_warp = threadIdx.x >> 5;
    const int gid = lane >> 2;
    const int tig = lane & 3;
    const int row_base = blockIdx.x * kRowsStripe;

    if (threadIdx.x == 0) {
        cta_gen = static_cast<int>(atomicAdd(&it_ctr[blockIdx.x], 1u)) + 1;
    }
    // ---- activation loads FIRST (feed the barrier) ----
    static_assert(kLocal == kProdThreads, "one element per thread per target row");
    __nv_bfloat16 act_r[kM8], act_z[kM8];
#pragma unroll
    for (int j = 0; j < kM8; ++j) {
        act_r[j] = rec8[j * kLocal + threadIdx.x];
        act_z[j] = z8[j * kLocal + threadIdx.x];
    }
    const __nv_bfloat16 act_nw = norm_w[threadIdx.x & 127];

    // ---- eager W burst: the warp's whole 1 KB slice, register-resident ----
    const int tile = local_warp >> 3;
    const int kw = local_warp & 7;
    const int row0 = (tile < 2) ? row_base + tile * 8 : row_base + (kDim / 2) + (tile - 2) * 8;
    const float sblk = w_scales[(row0 >> 7) * kScaleCols + kw];
    const __nv_fp8_e4m3* wp_base =
        Wpack + ((static_cast<size_t>(row0) >> 3) * (kLocal / 32) + kw * 4) * 256 + lane * 8;
    uint2 wreg[4];
#pragma unroll
    for (int t32 = 0; t32 < 4; ++t32) {
        asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];"
                     : "=r"(wreg[t32].x), "=r"(wreg[t32].y)
                     : "l"(wp_base + t32 * 256));
    }

// 64 (token, head) RMS reductions spread over all 32 warps, two each.
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int flat = local_warp * 2 + u;
        const int j = flat >> 3;
        const int head = flat & 7;
        const __nv_bfloat16* out_h = rec8 + j * kLocal + head * kHeadDim;
        float sumsq = 0.0f;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const float v = __bfloat162float(out_h[lane + t * 32]);
            sumsq += v * v;
        }
        const float total = m8_warp_sum(sumsq);
        if (lane == 0) {
            rms8[j][head] = rsqrtf(total * (1.0f / static_cast<float>(kHeadDim)) + kRmsEps);
        }
    }
    __syncthreads();

// ---- staging from the pre-loaded registers ----
#pragma unroll
    for (int j = 0; j < kM8; ++j) {
        const float out_f = __bfloat162float(act_r[j]);
        const float normed_bf =
            __bfloat162float(__float2bfloat16(out_f * rms8[j][threadIdx.x >> 7]));
        const float weighted_bf =
            __bfloat162float(__float2bfloat16(__bfloat162float(act_nw) * normed_bf));
        const float gated = weighted_bf * m8_silu_f32(__bfloat162float(act_z[j]));
        x_eff_b[j][threadIdx.x] = __float2bfloat16(gated);
    }
    __syncthreads();

    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int t32 = 0; t32 < 4; ++t32) {
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
            // Every n column is a real token row at M=8: no zero-pad guard.
            const __nv_bfloat16* xp = &x_eff_b[gid][kw * 128 + t32 * 32 + s_ * 16 + tig * 2];
            unsigned b[2];
            b[0] = *reinterpret_cast<const unsigned*>(xp);
            b[1] = *reinterpret_cast<const unsigned*>(xp + 8);
            const unsigned a[4] = {awf[s_ * 2], 0u, awf[s_ * 2 + 1], 0u};
            m8_mma_m16n8k16_bf16_op(c, a, b);
        }
    }
    red[local_warp][lane][0] = sblk * c[0];
    red[local_warp][lane][1] = sblk * c[1];
    __syncthreads();

    // The folding thread owns one wire packet and its two output rows.
    if (local_warp < 4) {
        float v0 = 0.0f, v1 = 0.0f;
#pragma unroll
        for (int w8 = 0; w8 < 8; ++w8) {
            v0 += red[local_warp * 8 + w8][lane][0];
            v1 += red[local_warp * 8 + w8][lane][1];
        }
        m8_oproj_fold_publish_combine_ll(v0, v1, local_warp * 8 + gid, tig, rep_mc, rep_local,
                                         status, x8_resid, rank, num_ranks, cta_gen, row_base);
    }
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M8
using namespace nexrt::sm90::nex_n2_5_pro::M8;
extern "C" {
// rec8/z8 [8,1024] bf16; norm_w [128] bf16; Wpack = M8 lane-major prepack of o_w
// [4096,1024] fp8; w_scales [32,8] fp32; rep is the M=8-owned symmetric inbox;
// it_ctr [128] is the per-CTA epoch; status int32[1];
// x8_resid [8,4096] bf16 folded in place.
int NexN25ProM8GdnOProjLaunch(const void* rec8, const void* z8, const void* norm_w,
                              const void* Wpack, const void* w_scales, unsigned long long rep_mc,
                              const void* rep_local, void* it_ctr, void* status, void* x8_resid,
                              int rank, int num_ranks, void* stream) {
    M8GdnOProjKernel<<<dim3(kProdCTAs), dim3(kProdThreads), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(rec8), static_cast<const __nv_bfloat16*>(z8),
        static_cast<const __nv_bfloat16*>(norm_w), static_cast<const __nv_fp8_e4m3*>(Wpack),
        static_cast<const float*>(w_scales), rep_mc, static_cast<const float*>(rep_local),
        static_cast<unsigned int*>(it_ctr), static_cast<int*>(status),
        static_cast<__nv_bfloat16*>(x8_resid), rank, num_ranks);
    return static_cast<int>(cudaGetLastError());
}
// ABI probes for the runtime's workspace sizing.
int NexN25ProM8OProjWorkspaceFloats() { return 2 * kMaxRanks * static_cast<int>(kM8RankValues); }
int NexN25ProM8OProjFusedCTAs() { return kM8FusedCTAs; }
}
