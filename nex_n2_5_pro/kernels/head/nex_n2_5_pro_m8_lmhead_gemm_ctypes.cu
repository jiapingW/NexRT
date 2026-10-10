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

// M=8 verify lm_head: explicit final RMSNorm followed by an eight-row GEMM over
// the raw BF16 head shard, producing TP-sharded TRUE fp32 logits.
//
// M=8-FAMILY TU: native eight-row implementation, no BS=1/M=4 production source.
// Derived from the M=4 W-rows-as-A transposed-mma geometry on the
// lane-major bf16 prepack, with sixteen real vocab rows per tile through the
// a1/a3 rows+8 fragments. B = h8^T fills all
// eight n columns (the M=4 form zero-padded half), so the ~254 MB/rank weight
// stream is shared across eight rows at unchanged traffic -- per-row lm-head cost
// halves relative to M=4.
//
// Two launches, ordered on the caller's stream:
//   1. M8FinalNormKernel (8 CTAs, one row each):
//      xn8[j] = bf16(h8[j] * rsqrt(mean(h8[j]^2) + eps) * (1 + norm_w)).
//      This matches the reference Gemma RMSNorm chain and its single BF16 round.
//   2. M8LmHeadBf16TcKernel (grid 1940 x 256): the eight-row GEMM on xn8.
//
// Wpack contract (pure permutation of the raw BF16 [31040, 4096] shard, built
// once by M=8 and shared with DFlash; identical to DFlash's tc16 pack law):
//   Wpack[tile 16r][k16][lane][frag f in {a0,a1,a2,a3}][2 bf16]
//   row = tile*16 + (f&1)*8 + gid;  k = k16*16 + (f>>1)*8 + tig*2 + e

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M8
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

namespace nexrt::sm90::nex_n2_5_pro::M8 {
constexpr int kDim = 4096;
constexpr int kDimPairs = kDim / 2;
constexpr int kRows = 31040;  // vocab / TP = 248320 / 8
constexpr int kM8 = 8;
constexpr float kInvDim = 1.0f / static_cast<float>(kDim);
constexpr int kHeadWarps = 8;
constexpr int kHeadWarpK = kDim / kHeadWarps;     // 512 = 32 k16 steps
constexpr int kHeadXPitch = kHeadWarpK * 2 + 16;  // 1040 B
constexpr int kHeadInputBytes = kHeadWarps * kM8 * kHeadXPitch;
constexpr int kHeadReduceBytes = kHeadWarps * 32 * 4 * sizeof(float);
constexpr int kHeadDynamicBytes = kHeadInputBytes + kHeadReduceBytes;
static_assert((kHeadInputBytes & 15) == 0, "reduction buffer must stay 16-byte aligned");

// --- 1. explicit final norm: one CTA per row, FP32 math and one BF16 round ----
template <int NT>
__global__ __launch_bounds__(NT) void M8FinalNormKernel(const __nv_bfloat16* __restrict__ h8,
                                                        const __nv_bfloat16* __restrict__ norm_w,
                                                        __nv_bfloat16* __restrict__ xn8,
                                                        float eps) {
    __shared__ typename cub::BlockReduce<float, NT>::TempStorage br_tmp;
    __shared__ float inverse_rms;
    const int row = blockIdx.x;
    const auto* __restrict__ input =
        reinterpret_cast<const __nv_bfloat162*>(h8 + static_cast<size_t>(row) * kDim);
    const auto* __restrict__ weight = reinterpret_cast<const __nv_bfloat162*>(norm_w);
    auto* __restrict__ output =
        reinterpret_cast<__nv_bfloat162*>(xn8 + static_cast<size_t>(row) * kDim);

    float square_sum = 0.0f;
    for (int pair = threadIdx.x; pair < kDimPairs; pair += NT) {
        const float2 value = __bfloat1622float2(input[pair]);
        square_sum += value.x * value.x + value.y * value.y;
    }
    const float reduced = cub::BlockReduce<float, NT>(br_tmp).Sum(square_sum);
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(reduced * kInvDim + eps);
    }
    __syncthreads();

    const float scale = inverse_rms;
    for (int pair = threadIdx.x; pair < kDimPairs; pair += NT) {
        const float2 value = __bfloat1622float2(input[pair]);
        const float2 gamma = __bfloat1622float2(weight[pair]);
        output[pair] = __floats2bfloat162_rn((value.x * scale) * (1.0f + gamma.x),
                                             (value.y * scale) * (1.0f + gamma.y));
    }
}

__device__ __forceinline__ void m8_mma_m16n8k16_bf16_head(float (&c)[4], const uint32_t (&a)[4],
                                                          const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// --- 2. eight-row transposed-mma GEMM, 16 real vocab rows per tile ---------------
__global__ __launch_bounds__(256) void M8LmHeadBf16TcKernel(const __nv_bfloat16* __restrict__ xn8,
                                                            const __nv_bfloat16* __restrict__ Wpack,
                                                            float* __restrict__ logits8) {
    extern __shared__ __align__(16) unsigned char dynamic_shared[];
    auto* sh = reinterpret_cast<char (*)[kM8][kHeadXPitch]>(dynamic_shared);
    auto* red = reinterpret_cast<float (*)[32][4]>(dynamic_shared + kHeadInputBytes);

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int gid = lane >> 2;
    const int tig = lane & 3;
    const int tile = blockIdx.x;  // 16 vocab rows
    const int k0 = warp * kHeadWarpK;
    char* const shw = &sh[warp][0][0];

#pragma unroll
    for (int m = 0; m < kM8; ++m) {  // h slice: 8 rows x 1 KB, one shot
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t dst = static_cast<uint32_t>(
                __cvta_generic_to_shared(shw + m * kHeadXPitch + h * 512 + lane * 16));
            const void* src = xn8 + static_cast<size_t>(m) * kDim + k0 + (h * 512 + lane * 16) / 2;
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
        // Every n column is a real token at M=8: no zero-pad guard.
        const char* xp = shw + gid * kHeadXPitch + s_ * 32 + tig * 4;
        unsigned b[2];
        b[0] = *reinterpret_cast<const unsigned*>(xp);
        b[1] = *reinterpret_cast<const unsigned*>(xp + 16);
        const uint32_t a[4] = {w.x, w.y, w.z, w.w};
        m8_mma_m16n8k16_bf16_head(c, a, b);
    }

#pragma unroll
    for (int e = 0; e < 4; ++e) {
        red[warp][lane][e] = c[e];
    }
    __syncthreads();
    if (threadIdx.x < 32) {  // 16 real rows; tokens tig*2, tig*2+1
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int sp = 0; sp < 8; ++sp) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                v[e] += red[sp][lane][e];
            }
        }
        const size_t r_lo = static_cast<size_t>(tile) * 16 + gid;
        logits8[static_cast<size_t>(tig * 2) * kRows + r_lo] = v[0];
        logits8[static_cast<size_t>(tig * 2 + 1) * kRows + r_lo] = v[1];
        logits8[static_cast<size_t>(tig * 2) * kRows + r_lo + 8] = v[2];
        logits8[static_cast<size_t>(tig * 2 + 1) * kRows + r_lo + 8] = v[3];
    }
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M8
using namespace nexrt::sm90::nex_n2_5_pro::M8;
extern "C" {
// h8 [8,4096] raw BF16 residual rows; norm_w [4096] BF16 final-norm weight;
// xn8 [8,4096] BF16 explicit-normalization output.
int NexN25ProM8FinalNormLaunch(const void* h8, const void* norm_w, void* xn8, float eps,
                               void* stream) {
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    M8FinalNormKernel<256><<<dim3(kM8), dim3(256), 0, cuda_stream>>>(
        static_cast<const __nv_bfloat16*>(h8), static_cast<const __nv_bfloat16*>(norm_w),
        static_cast<__nv_bfloat16*>(xn8), eps);
    return static_cast<int>(cudaGetLastError());
}

// xn8 [8,4096] normalized BF16 rows; Wpack is the tc16 permutation of the raw
// [31040,4096] BF16 shard; logits8 [8,31040] FP32 true logits.
int NexN25ProM8LmHeadGemmTc16Launch(const void* xn8, const void* Wpack, void* logits8,
                                    void* stream) {
    static const cudaError_t smem_status =
        cudaFuncSetAttribute(reinterpret_cast<const void*>(&M8LmHeadBf16TcKernel),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kHeadDynamicBytes);
    if (smem_status != cudaSuccess) {
        return static_cast<int>(smem_status);
    }
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    M8LmHeadBf16TcKernel<<<dim3(kRows / 16), dim3(256), kHeadDynamicBytes, s>>>(
        static_cast<const __nv_bfloat16*>(xn8), static_cast<const __nv_bfloat16*>(Wpack),
        static_cast<float*>(logits8));
    return static_cast<int>(cudaGetLastError());
}
int NexN25ProM8LmHeadRows() { return kM8; }
}
