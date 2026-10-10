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

// M=4-native FUSED GDN o-projection push/reduce -- ONE launch replacing a split
// [producer + reduce] pair. GDN twin of
// nex_n2_5_pro_m4_attn_oproj_fused_ctypes.cu.
//
// Staging and GEMM: the
// per-head RMSNorm x norm_w x silu(z) staging with its bf16 round-trips, the 16-row
// half-stripe row mapping, and a single weight-tile load unpacked once and reused
// across the four draft rows.
// The publish/reduce protocol is shared -- see kernels/common/m4_oproj_fused_protocol.cuh.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M4
#endif

#include "../common/m4_oproj_fused_protocol.cuh"

namespace nexrt::sm90::nex_n2_5_pro::M4 {
constexpr int kM4 = kM4Fused;
constexpr int kProdCTAs = kFusedCTAs;
constexpr int kProdThreads = 1024;
constexpr int kRowsStripe = 16;

// ===== bf16tc family shared mma helper ===================================================
// The reference staging ALREADY rounds x_eff through bf16 (float(bf16(gated))), so a bf16
// smem mirror is VALUE-EXACT -- only the products change class (fp32-SIMT -> bf16 mma,
// fp32-accumulated). GEMM core identical to the attn twin: 32 warps = 4 row-tiles of 8 x
// 8 K-slices of 128, W from the lane-major prepack pack_gemm_bf16tc(W, 1024).
__device__ __forceinline__ void mma_m16n8k16_bf16_op_(float (&c)[4], const unsigned (&a)[4],
                                                      const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ===== OVERLAP PRODUCER (tc16, W burst overlapped with staging) ==========================
// Activation loads issue first, then the entire W burst. All 32 warps compute
// the 32 (draft, head) RMS reductions from the already-loaded activation values
// staged through shared memory; no second global rec4 read is needed. The
// reduction expression and per-element staging order are unchanged.
__global__ __launch_bounds__(kProdThreads, 1) void M4GdnOProjKernel(
    const __nv_bfloat16* __restrict__ rec4,  // [4, 1024]
    const __nv_bfloat16* __restrict__ z4,    // [4, 1024]
    const __nv_bfloat16* __restrict__ norm_w,
    const __nv_fp8_e4m3* __restrict__ Wpack,  // pack_gemm_bf16tc(W, 1024)
    const float* __restrict__ w_scales, unsigned long long rep_mc,
    const float* __restrict__ rep_local, unsigned int* __restrict__ it_ctr,
    int* __restrict__ status, __nv_bfloat16* __restrict__ x4_resid, int rank, int num_ranks) {
    constexpr int kXPitchB = kLocal + 8;
    __shared__ __align__(16) __nv_bfloat16 x_eff_b[kM4][kXPitchB];
    __shared__ __align__(16) float rms4[kM4][kHeads];
    __shared__ __align__(16) float part[kM4][kFusedSlots];
    __shared__ float red[32][32][2];
    __shared__ __align__(16) __nv_bfloat16 staged_rec[kM4][kLocal];
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
    static_assert(kLocal == kProdThreads, "one element per thread per draft row");
    __nv_bfloat16 act_r[kM4], act_z[kM4];
#pragma unroll
    for (int j = 0; j < kM4; ++j) {
        act_r[j] = rec4[j * kLocal + threadIdx.x];
        act_z[j] = z4[j * kLocal + threadIdx.x];
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

    // ---- head-RMS from the activation registers (all 32 warps active) ----
    static_assert(kProdThreads / 32 == kM4 * kHeads, "one warp per (draft, head) reduction");
#pragma unroll
    for (int j = 0; j < kM4; ++j) {
        staged_rec[j][threadIdx.x] = act_r[j];
    }
    __syncthreads();
    const int rms_row = local_warp / kHeads;
    const int rms_head = local_warp - rms_row * kHeads;
    const __nv_bfloat16* rms_source = &staged_rec[rms_row][rms_head * kHeadDim];
    float sumsq = 0.0f;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const float value = __bfloat162float(rms_source[lane + t * 32]);
        sumsq += value * value;
    }
    const float total = warp_sum(sumsq);
    if (lane == 0) {
        rms4[rms_row][rms_head] = rsqrtf(total * (1.0f / static_cast<float>(kHeadDim)) + kRmsEps);
    }
    __syncthreads();

// ---- staging from the pre-loaded registers (identical per-element chain) ----
#pragma unroll
    for (int j = 0; j < kM4; ++j) {
        const float out_f = __bfloat162float(act_r[j]);
        const float normed_bf =
            __bfloat162float(__float2bfloat16(out_f * rms4[j][threadIdx.x >> 7]));
        const float weighted_bf =
            __bfloat162float(__float2bfloat16(__bfloat162float(act_nw) * normed_bf));
        const float gated = weighted_bf * silu_f32(__bfloat162float(act_z[j]));
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
            unsigned b[2] = {0u, 0u};
            if (gid < kM4) {
                const __nv_bfloat16* xp = &x_eff_b[gid][kw * 128 + t32 * 32 + s_ * 16 + tig * 2];
                b[0] = *reinterpret_cast<const unsigned*>(xp);
                b[1] = *reinterpret_cast<const unsigned*>(xp + 8);
            }
            const unsigned a[4] = {awf[s_ * 2], 0u, awf[s_ * 2 + 1], 0u};
            mma_m16n8k16_bf16_op_(c, a, b);
        }
    }
    red[local_warp][lane][0] = sblk * c[0];
    red[local_warp][lane][1] = sblk * c[1];
    __syncthreads();

    if (local_warp < 4 && tig * 2 < kM4) {
        float v0 = 0.0f, v1 = 0.0f;
#pragma unroll
        for (int w8 = 0; w8 < 8; ++w8) {
            v0 += red[local_warp * 8 + w8][lane][0];
            v1 += red[local_warp * 8 + w8][lane][1];
        }
        part[tig * 2][local_warp * 8 + gid] = v0;
        part[tig * 2 + 1][local_warp * 8 + gid] = v1;
    }
    __syncthreads();

    m4_oproj_publish_and_reduce_ll(part, rep_mc, rep_local, status, x4_resid, rank, num_ranks,
                                   cta_gen, row_base);
}

extern "C" int NexN25ProM4GdnOProjLaunch(const void* rec4, const void* z4, const void* norm_w,
                                         const void* Wpack, const void* w_scales,
                                         unsigned long long rep_mc, const void* rep_local,
                                         void* it_ctr, void* status, void* x4_resid, int rank,
                                         int num_ranks, void* stream) {
    M4GdnOProjKernel<<<dim3(kProdCTAs), dim3(kProdThreads), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(rec4), static_cast<const __nv_bfloat16*>(z4),
        static_cast<const __nv_bfloat16*>(norm_w), static_cast<const __nv_fp8_e4m3*>(Wpack),
        static_cast<const float*>(w_scales), rep_mc, static_cast<const float*>(rep_local),
        static_cast<unsigned int*>(it_ctr), static_cast<int*>(status),
        static_cast<__nv_bfloat16*>(x4_resid), rank, num_ranks);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M4
