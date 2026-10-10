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

// M=4 full-attention qkvgate projection -- FOUR residual rows in ONE pass over the fp8
// qkv weights, writing provisional K/V rows at positions S..S+3 and per-row q/gate outputs.
//
// The lane-major weight pack and transposed BF16 MMA mapping match the selected
// BS=1 projection's numerical class while sharing one weight pass across four rows.
//
// Position: S is read on-device from slot[1] (the unit base); row j scatters K/V at S+j (the
// device draft offsets), so a single launch produces all four provisional KV rows; the sequential
// causal core then runs per row at slot = S+j and sees the prefix plus earlier draft rows.
// q_out4 / gate_out4 are [4, 1024] row-major copies of the M1 q_out / gate buffers.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M4
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "nex_n2_5_pro_qkv_common.cuh"

namespace nexrt::sm90::nex_n2_5_pro::M4 {
__device__ __forceinline__ uint2 m4_qkv_load_weight(const void* pointer) {
    uint2 value;
    asm volatile("ld.global.cs.v2.u32 {%0,%1}, [%2];"
                 : "=r"(value.x), "=r"(value.y)
                 : "l"(pointer));
    return value;
}

constexpr int kM4Rows = 4;

// Production tensor-core projection. The transposed mapping collapses the
// four-row SIMT loop into one MMA stream while staying in the BF16 class:
// collapse the 4-row SIMT loop into ONE mma stream while staying in the BF16 class.
//   mma.m16n8k16.row.col.f32.bf16.bf16.f32 with a TRANSPOSED
//   mapping: A[0:8,:] = EIGHT decoded W rows (A[8:16] = zero), B[:,0:4] = x4^T (cols 4:8
//   zero), useful D[0:8, 0:4] = 8 output rows x 4 tokens per mma. Products are exact in
//   the fp32 accumulator (bf16 8-bit significands => 16-bit products < fp32's 24), so
//   this class is TIGHTER than both f16acc and the SIMT-bf16 chain; zero fp16 math (the
//   e4m3->f16 unpack transit is value-exact; decode via float2 keeps zero/subnormal/NaN
//   semantics -- the integer f16->bf16 repack idea breaks on +-0 without fixups).
//   Geometry: 320 CTAs x 8 real rows, 8 warps x K=512,
//   per-k128 fp32 block accumulate x w_scale (no x quant, no x-side scales), in-CTA reduce, same
//   scatter. Warp-level mma.sync, NOT wgmma (m64 would pad 8x or need 40 CTAs).
// W feed needs NO smem and NO ldmatrix: the weights are PREPACKED at model load into a
// lane-major layout Wpack[tile n8][k32][lane][8 B] (nexrt is a decode engine; weight
// reshaping happens at load, with dual layouts where bs=1 draft + M4 extend share
// a block). Each lane's 8 bytes are exactly its A-fragment pairs for the two k16 steps
// of one k32 tile -> one fully coalesced 256 B warp load per 2 mmas, streamed .cs with
// a register double buffer. x is staged once per warp-slice in smem (1040 B pitch ->
// conflict-free 4 B B-fragment LDS at banks (4*gid+tig)%32).
__device__ __forceinline__ void mma_m16n8k16_bf16_(float (&c)[4], const unsigned (&a)[4],
                                                   const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ __launch_bounds__(256) void M4AttnQkvKernel(
    const __nv_bfloat16* __restrict__ x4,     // [4, 4096]
    const __nv_fp8_e4m3* __restrict__ Wpack,  // [320][128][32][8] lane-major prepack
    const float* __restrict__ scales,
    __nv_bfloat16* __restrict__ q_out4,  // [4, 1024]
    __nv_bfloat16* __restrict__ gate_out4, __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache, const unsigned int* __restrict__ seqlen) {
    const int pos0 = static_cast<int>(seqlen[1]);
    constexpr int kWarpK = 512;                             // 16 k32 tiles per warp
    constexpr int kXPitch = kWarpK * 2 + 16;                // 1040 B row pitch: bank spread
    __shared__ __align__(16) char sx[8][kM4Rows][kXPitch];  // 33.3 KB warp-private slices
    __shared__ float red[8][32][2];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int gid = lane >> 2;  // A row within tile / B token
    const int tig = lane & 3;
    const int row0 = blockIdx.x * 8;
    const int k0 = warp * kWarpK;
    char* const sxw = &sx[warp][0][0];

#pragma unroll
    for (int m = 0; m < kM4Rows; ++m) {  // x slice: 4 rows x 1 KB, one shot
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t dst = static_cast<uint32_t>(
                __cvta_generic_to_shared(sxw + m * kXPitch + h * 512 + lane * 16));
            const void* src = x4 + static_cast<size_t>(m) * kHid + k0 + (h * 512 + lane * 16) / 2;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
        }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncwarp();

    const float* ws = scales + static_cast<size_t>(row0 >> 7) * kScaleBlocksK + warp * 4;
    float sblk[4];
#pragma unroll
    for (int b_ = 0; b_ < 4; ++b_) {
        sblk[b_] = ws[b_];
    }

    const __nv_fp8_e4m3* wp_base =
        Wpack + (static_cast<size_t>(blockIdx.x) * 128 + (k0 >> 5)) * 256 + lane * 8;
    float sacc[2] = {0.0f, 0.0f};
    uint2 wbuf[4];  // depth-4: cover .cs latency at 2.4 CTAs/SM
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        wbuf[i] = m4_qkv_load_weight(wp_base + i * 256);
    }
#pragma unroll
    for (int blk = 0; blk < 4; ++blk) {  // four k128 scale blocks
        float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int t32 = 0; t32 < 4; ++t32) {  // k32 tiles: 1 LDG.64 feeds 2 mmas
            const int ci = blk * 4 + t32;
            if (ci + 3 < 16) {
                wbuf[(ci + 3) & 3] = m4_qkv_load_weight(wp_base + (ci + 3) * 256);
            }
            const uint2 raw = wbuf[ci & 3];
            const __nv_fp8x2_e4m3* wp = reinterpret_cast<const __nv_fp8x2_e4m3*>(&raw);
            unsigned awf[4];  // decoded A pairs as bf16x2 bits
#pragma unroll
            for (int p_ = 0; p_ < 4; ++p_) {
                const __nv_bfloat162 bb = __float22bfloat162_rn(static_cast<float2>(wp[p_]));
                awf[p_] = *reinterpret_cast<const unsigned*>(&bb);
            }
#pragma unroll
            for (int s_ = 0; s_ < 2; ++s_) {  // two k16 steps
                unsigned b[2] = {0u, 0u};
                if (gid < kM4Rows) {  // B token rows; cols 4:8 stay zero
                    const char* xp =
                        sxw + gid * kXPitch + (blk * 128 + t32 * 32 + s_ * 16) * 2 + tig * 4;
                    b[0] = *reinterpret_cast<const unsigned*>(xp);
                    b[1] = *reinterpret_cast<const unsigned*>(xp + 16);
                }
                // a0 = row gid k0..7, a2 = row gid k8..15; a1/a3 = rows 8..15 (zero pad)
                const unsigned a[4] = {awf[s_ * 2], 0u, awf[s_ * 2 + 1], 0u};
                mma_m16n8k16_bf16_(c, a, b);
            }
        }
        sacc[0] += sblk[blk] * c[0];  // c[2]/c[3] = A rows 8..15 == 0
        sacc[1] += sblk[blk] * c[1];
    }

    red[warp][lane][0] = sacc[0];
    red[warp][lane][1] = sacc[1];
    __syncthreads();
    if (threadIdx.x < 32 && tig * 2 < kM4Rows) {  // all 8 rows real; tokens tig*2,+1
        float v0 = 0.0f, v1 = 0.0f;
#pragma unroll
        for (int sp = 0; sp < 8; ++sp) {
            v0 += red[sp][lane][0];
            v1 += red[sp][lane][1];
        }
        scatter_row(row0 + gid, v0, q_out4 + static_cast<size_t>(tig * 2) * (kQRows / 2),
                    gate_out4 + static_cast<size_t>(tig * 2) * (kQRows / 2), k_cache, v_cache,
                    pos0 + tig * 2);
        scatter_row(row0 + gid, v1, q_out4 + static_cast<size_t>(tig * 2 + 1) * (kQRows / 2),
                    gate_out4 + static_cast<size_t>(tig * 2 + 1) * (kQRows / 2), k_cache, v_cache,
                    pos0 + tig * 2 + 1);
    }
}

extern "C" {
// x4 [4,4096] bf16; W/scales identical to the M1 qkv projection; q_out4/gate_out4 [4,1024];
// K/V written into the caches at seqlen[1]+j for row j.

// Wpack is the lane-major production prepack, not the row-major BS=1 tensor.
int NexN25ProM4AttnQkvLaunch(const void* x4, const void* Wpack, const float* scales, void* q_out4,
                             void* gate_out4, void* k_cache, void* v_cache, const void* seqlen,
                             void* stream) {
    M4AttnQkvKernel<<<dim3(kTotalRows / 8), dim3(256), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(x4), static_cast<const __nv_fp8_e4m3*>(Wpack), scales,
        static_cast<__nv_bfloat16*>(q_out4), static_cast<__nv_bfloat16*>(gate_out4),
        static_cast<__nv_bfloat16*>(k_cache), static_cast<__nv_bfloat16*>(v_cache),
        static_cast<const unsigned int*>(seqlen));
    return static_cast<int>(cudaGetLastError());
}
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M4
