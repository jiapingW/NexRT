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

// M=8 full-attention qkvgate projection: EIGHT residual rows in ONE pass over the
// fp8 qkv weights, writing provisional K/V rows at positions C..C+7 and per-row
// q/gate outputs.
//
// M=8-FAMILY TU: native eight-row implementation, no BS=1/M=4 production source.
// Derived from the M=4 transposed-mma + lane-major-prepack geometry with
// fp32 accumulation; at M=8 the B operand's n=8 token
// columns are ALL real (the M=4 form zero-padded half), so each weight fragment
// retires twice the useful mma work at unchanged weight traffic.
//
// Position: the unit base C is read on-device from the M=8 control block's word 1
// (committed_len); row j scatters K/V at C+j, so a single launch produces all eight
// provisional KV rows. Rejected rows are gated off by committed_len and overwritten
// by later units. q_out8/gate_out8 are [8, 1024] row-major.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M8
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "../common/m8_control_block.cuh"

namespace nexrt::sm90::nex_n2_5_pro::M8 {
constexpr int kHid = 4096;
constexpr int kHeadDim = 256;
constexpr int kQRows = 2048;  // 4 local heads x 256 x (q | gate)
constexpr int kKRows = 256;
constexpr int kVRows = 256;
constexpr int kTotalRows = kQRows + kKRows + kVRows;  // 2560
constexpr int kQEnd = kQRows;
constexpr int kKEnd = kQRows + kKRows;
constexpr int kScaleBlocksK = kHid / 128;  // 32
constexpr int kM8Rows = m8ctl::kM;         // 8

__device__ __forceinline__ void m8_scatter_row(int row, float acc,
                                               __nv_bfloat16* __restrict__ q_out,
                                               __nv_bfloat16* __restrict__ gate_out,
                                               __nv_bfloat16* __restrict__ k_cache,
                                               __nv_bfloat16* __restrict__ v_cache, int pos) {
    const __nv_bfloat16 val = __float2bfloat16(acc);
    if (row < kQEnd) {
        const int head = row >> 9;
        const int within = row & 511;
        if (within < kHeadDim) {
            q_out[head * kHeadDim + within] = val;
        } else {
            gate_out[head * kHeadDim + (within - kHeadDim)] = val;
        }
    } else if (row < kKEnd) {
        k_cache[static_cast<size_t>(pos) * kHeadDim + (row - kQEnd)] = val;
    } else {
        v_cache[static_cast<size_t>(pos) * kHeadDim + (row - kKEnd)] = val;
    }
}

__device__ __forceinline__ uint2 m8_load_weight_uint2(const void* ptr) {
    uint2 value;
    asm volatile("ld.global.cs.v2.u32 {%0,%1}, [%2];" : "=r"(value.x), "=r"(value.y) : "l"(ptr));
    return value;
}

__device__ __forceinline__ void m8_mma_m16n8k16_bf16(float (&c)[4], const unsigned (&a)[4],
                                                     const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Wpack contract: the same lane-major fragment pack as the M=8 GDN front (8-row
// tiles of the [2560, 4096] fp8 weight, k32 blocks, 256 B per (tile, k32), 8 B/lane).
__global__ __launch_bounds__(256) void M8QkvBf16TcKernel(
    const __nv_bfloat16* __restrict__ x8,     // [8, 4096]
    const __nv_fp8_e4m3* __restrict__ Wpack,  // [320][128][32][8] lane-major
    const float* __restrict__ scales,
    __nv_bfloat16* __restrict__ q_out8,     // [8, 1024]
    __nv_bfloat16* __restrict__ gate_out8,  // [8, 1024]
    __nv_bfloat16* __restrict__ k_cache, __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ ctl) {  // M=8 control block: base = word 1
    const int pos0 = m8ctl::CommittedLen(ctl);
    constexpr int kWarpK = 512;                             // 16 k32 tiles per warp
    constexpr int kXPitch = kWarpK * 2 + 16;                // 1040 B row pitch: bank spread
    __shared__ __align__(16) char sx[8][kM8Rows][kXPitch];  // 66.6 KB warp-private slices
    __shared__ float red[8][32][2];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int gid = lane >> 2;  // A row within tile / B token
    const int tig = lane & 3;
    const int row0 = blockIdx.x * 8;
    const int k0 = warp * kWarpK;
    char* const sxw = &sx[warp][0][0];

#pragma unroll
    for (int m = 0; m < kM8Rows; ++m) {  // x slice: 8 rows x 1 KB, one shot
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t dst = static_cast<uint32_t>(
                __cvta_generic_to_shared(sxw + m * kXPitch + h * 512 + lane * 16));
            const void* src = x8 + static_cast<size_t>(m) * kHid + k0 + (h * 512 + lane * 16) / 2;
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
    uint2 wbuf[4];  // depth-4 register double buffer
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        wbuf[i] = m8_load_weight_uint2(wp_base + i * 256);
    }
#pragma unroll
    for (int blk = 0; blk < 4; ++blk) {  // four k128 scale blocks
        float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int t32 = 0; t32 < 4; ++t32) {  // k32 tiles: 1 LDG.64 feeds 2 mmas
            const int ci = blk * 4 + t32;
            if (ci + 3 < 16) {
                wbuf[(ci + 3) & 3] = m8_load_weight_uint2(wp_base + (ci + 3) * 256);
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
                // Every n column is a real token at M=8: no zero-pad guard.
                const char* xp =
                    sxw + gid * kXPitch + (blk * 128 + t32 * 32 + s_ * 16) * 2 + tig * 4;
                unsigned b[2];
                b[0] = *reinterpret_cast<const unsigned*>(xp);
                b[1] = *reinterpret_cast<const unsigned*>(xp + 16);
                // a0 = row gid k0..7, a2 = row gid k8..15; a1/a3 = rows 8..15 (zero pad)
                const unsigned a[4] = {awf[s_ * 2], 0u, awf[s_ * 2 + 1], 0u};
                m8_mma_m16n8k16_bf16(c, a, b);
            }
        }
        sacc[0] += sblk[blk] * c[0];  // c[2]/c[3] = A rows 8..15 == 0
        sacc[1] += sblk[blk] * c[1];
    }

    red[warp][lane][0] = sacc[0];
    red[warp][lane][1] = sacc[1];
    __syncthreads();
    if (threadIdx.x < 32) {  // all 8 rows real; tokens tig*2,+1
        float v0 = 0.0f, v1 = 0.0f;
#pragma unroll
        for (int sp = 0; sp < 8; ++sp) {
            v0 += red[sp][lane][0];
            v1 += red[sp][lane][1];
        }
        m8_scatter_row(row0 + gid, v0, q_out8 + static_cast<size_t>(tig * 2) * (kQRows / 2),
                       gate_out8 + static_cast<size_t>(tig * 2) * (kQRows / 2), k_cache, v_cache,
                       pos0 + tig * 2);
        m8_scatter_row(row0 + gid, v1, q_out8 + static_cast<size_t>(tig * 2 + 1) * (kQRows / 2),
                       gate_out8 + static_cast<size_t>(tig * 2 + 1) * (kQRows / 2), k_cache,
                       v_cache, pos0 + tig * 2 + 1);
    }
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M8
using namespace nexrt::sm90::nex_n2_5_pro::M8;
extern "C" {
// x8 [8,4096] bf16; Wpack = M8 lane-major prepack of qkv_w [2560,4096] fp8 (NOT
// row-major); scales [20,32] fp32; q_out8/gate_out8 [8,1024] bf16; K/V written into
// the shared caches at committed_len + j for row j; ctl = M=8 control block.
int NexN25ProM8AttnQkvBf16TcLaunch(const void* x8, const void* Wpack, const float* scales,
                                   void* q_out8, void* gate_out8, void* k_cache, void* v_cache,
                                   const void* ctl, void* stream) {
    M8QkvBf16TcKernel<<<dim3(kTotalRows / 8), dim3(256), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(x8), static_cast<const __nv_fp8_e4m3*>(Wpack), scales,
        static_cast<__nv_bfloat16*>(q_out8), static_cast<__nv_bfloat16*>(gate_out8),
        static_cast<__nv_bfloat16*>(k_cache), static_cast<__nv_bfloat16*>(v_cache),
        static_cast<const int*>(ctl));
    return static_cast<int>(cudaGetLastError());
}
int NexN25ProM8AttnQkvRows() { return kM8Rows; }
}
