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

// Eight ordered K512 partial sums; existing gate pack and 66*256 prefetch stride.
// Selected prefetch depth: 16.

#include <cuda_bf16.h>
#include "../common/pdl.cuh"
#include <cuda_runtime.h>

#include <cstdint>

namespace nexrt::sm90::nex_n2_5_pro::M8 {
constexpr int kDim = 4096;
constexpr int kDimPairs = kDim / 2;
constexpr int kGateRowsActive = 513;
constexpr int kGateRowsPhysical = 528;
constexpr int kM8 = 8;

__device__ __forceinline__ float m8_warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

template <int kNumThreads>
__device__ __forceinline__ float m8_block_sum(float v) {
    constexpr int kNumWarps = kNumThreads / 32;
    __shared__ float warp_sums[kNumWarps];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = m8_warp_sum(v);
    if (lane == 0) {
        warp_sums[warp] = v;
    }
    __syncthreads();
    v = (threadIdx.x < kNumWarps) ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        v = m8_warp_sum(v);
    }
    return v;
}

__device__ __forceinline__ void m8_mma_m16n8k16_bf16_gate(float (&c)[4], const unsigned int (&a)[4],
                                                          const unsigned int (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

constexpr int kTcWarpK = kDim / 8;            // 512 k per warp = 32 k16 steps
constexpr int kTcXPitch = kTcWarpK * 2 + 16;  // 1040 B: pad kills B-LDS conflicts

// Per-warp x8 staging: 8 tokens x 1 KB k-slice into this warp's smem region.
__device__ __forceinline__ void m8_tc_stage_x_issue(const __nv_bfloat16* __restrict__ x8,
                                                    char* __restrict__ shw, int lane, int k0) {
#pragma unroll
    for (int m = 0; m < kM8; ++m) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned int dst = static_cast<unsigned int>(
                __cvta_generic_to_shared(shw + m * kTcXPitch + h * 512 + lane * 16));
            const void* src = x8 + static_cast<size_t>(m) * kDim + k0 + (h * 512 + lane * 16) / 2;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
        }
    }
    asm volatile("cp.async.commit_group;");
}

// Only the original 66 CTAs participate in prefetch.
__device__ __forceinline__ void M8TcPrefetchTail(const char* __restrict__ pf_base,
                                                 long long pf_bytes, long long grid_threads) {
    if (pf_base != nullptr && pf_bytes > 0) {
        const long long lines = pf_bytes >> 7;
        const long long gtid = static_cast<long long>(blockIdx.x) * 256 + threadIdx.x;
        for (long long line = gtid; line < lines; line += grid_threads) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(pf_base + (line << 7)));
        }
    }
}

// wpack contract (built once by the M=8 runtime; identical index math to the M=4
// family gate pack): uint2 index ((tile*256 + k16)*32 + lane), row = tile*8 +
// (lane>>2), bf16 pure permutation zero-padded to kGateRowsPhysical rows.
__global__ __launch_bounds__(256) void M8GateGemmKernel(const __nv_bfloat16* __restrict__ x8,
                                                        const __nv_bfloat16* __restrict__ wpack,
                                                        __nv_bfloat16* __restrict__ logits8,
                                                        float* __restrict__ rms_scale8, float eps,
                                                        const char* __restrict__ pf_base,
                                                        long long pf_bytes) {
    NEXRT_PDL_ENTRY_SYNC();
    if (blockIdx.x >= kGateRowsPhysical / 8) {
        const auto* __restrict__ h2 = reinterpret_cast<const __nv_bfloat162*>(x8);
        float rms_acc = 0.0f;
        for (int pair = threadIdx.x; pair < kDimPairs; pair += 256) {
            const float2 fr =
                __bfloat1622float2(h2[(blockIdx.x - kGateRowsPhysical / 8) * kDimPairs + pair]);
            rms_acc += fr.x * fr.x + fr.y * fr.y;
        }
        const float sq_sum = m8_block_sum<256>(rms_acc);
        if (threadIdx.x == 0) {
            rms_scale8[blockIdx.x - kGateRowsPhysical / 8] =
                rsqrtf(sq_sum * (1.0f / static_cast<float>(kDim)) + eps);
        }
        return;
    }

    constexpr int kDepth = 16;  // uint2 ring slots; ptxas chooses physical registers.
    __shared__ __align__(16) char sh[8][kM8][kTcXPitch];
    __shared__ float red[8][32][2];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int gid = lane >> 2;
    const int tig = lane & 3;
    const int tile = blockIdx.x;  // 8 gate rows (528/8 = 66 CTAs)
    const int k0 = warp * kTcWarpK;
    char* const shw = &sh[warp][0][0];

    m8_tc_stage_x_issue(x8, shw, lane, k0);
    const uint2* wp = reinterpret_cast<const uint2*>(wpack) +
                      (static_cast<size_t>(tile) * 256 + (k0 >> 4)) * 32 + lane;
    uint2 wbuf[kDepth];
#pragma unroll
    for (int i = 0; i < kDepth - 1; ++i) {
        asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];"
                     : "=r"(wbuf[i].x), "=r"(wbuf[i].y)
                     : "l"(wp + static_cast<size_t>(i) * 32));
    }
    asm volatile("cp.async.wait_group 0;");
    __syncwarp();

    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // c[2],c[3] stay 0 (A rows +8 zeroed)
#pragma unroll
    for (int s_ = 0; s_ < 32; ++s_) {
        if (s_ + kDepth - 1 < 32) {
            const int nx = s_ + kDepth - 1;
            asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];"
                         : "=r"(wbuf[nx & (kDepth - 1)].x), "=r"(wbuf[nx & (kDepth - 1)].y)
                         : "l"(wp + static_cast<size_t>(nx) * 32));
        }
        const uint2 w = wbuf[s_ & (kDepth - 1)];
        // Every n column is a real token at M=8: no zero-pad guard.
        const char* xp = shw + gid * kTcXPitch + s_ * 32 + tig * 4;
        unsigned b[2];
        b[0] = *reinterpret_cast<const unsigned*>(xp);
        b[1] = *reinterpret_cast<const unsigned*>(xp + 16);
        const unsigned int a[4] = {w.x, 0u, w.y, 0u};
        m8_mma_m16n8k16_bf16_gate(c, a, b);
    }

    red[warp][lane][0] = c[0];
    red[warp][lane][1] = c[1];
    __syncthreads();
    if (threadIdx.x < 32) {  // tokens tig*2, tig*2+1; row gid
        float v0 = 0.0f, v1 = 0.0f;
#pragma unroll
        for (int sp = 0; sp < 8; ++sp) {
            v0 += red[sp][lane][0];
            v1 += red[sp][lane][1];
        }
        const int r = tile * 8 + gid;
        if (r < kGateRowsActive) {
            logits8[static_cast<size_t>(tig * 2) * kGateRowsPhysical + r] = __float2bfloat16(v0);
            logits8[static_cast<size_t>(tig * 2 + 1) * kGateRowsPhysical + r] =
                __float2bfloat16(v1);
        }
    }
    M8TcPrefetchTail(pf_base, pf_bytes, static_cast<long long>(kGateRowsPhysical / 8) * 256);
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M8
using namespace nexrt::sm90::nex_n2_5_pro::M8;
extern "C" {
// x8 [8,4096] bf16; wpack = M8 lane-major gate prepack (NOT row-major gate_w);
// logits8 [8,528] bf16; rms_scale8 [8] fp32. pf_bytes <= 0 (or null pf_base)
// disables the hint tail.
int NexN25ProM8GateGemmLaunch(const void* x8, const void* wpack, void* logits8, float* rms_scale8,
                              float eps, const void* pf_base, long long pf_bytes, void* stream) {
    M8GateGemmKernel<<<dim3(kGateRowsPhysical / 8 + kM8), dim3(256), 0,
                       static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(x8), static_cast<const __nv_bfloat16*>(wpack),
        static_cast<__nv_bfloat16*>(logits8), rms_scale8, eps, static_cast<const char*>(pf_base),
        pf_bytes);
    return static_cast<int>(cudaGetLastError());
}

int NexN25ProM8GateGemmPdlLaunch(const void* x8, const void* wpack, void* logits8,
                                 float* rms_scale8, float eps, const void* pf_base,
                                 long long pf_bytes, void* stream) {
    return static_cast<int>(NexrtPdlLaunchKernel(
        dim3(kGateRowsPhysical / 8 + kM8), dim3(256), 0, static_cast<cudaStream_t>(stream),
        M8GateGemmKernel, static_cast<const __nv_bfloat16*>(x8),
        static_cast<const __nv_bfloat16*>(wpack), static_cast<__nv_bfloat16*>(logits8), rms_scale8,
        eps, static_cast<const char*>(pf_base), pf_bytes));
}
int NexN25ProM8GateRowsPhysical() { return kGateRowsPhysical; }
}

extern "C" int GateRolesAttributes(int* out) {
    cudaFuncAttributes a;
    auto e = cudaFuncGetAttributes(&a, M8GateGemmKernel);
    if (e != cudaSuccess) return int(e);
    out[0] = a.numRegs;
    out[1] = a.sharedSizeBytes;
    out[2] = a.localSizeBytes;
    return 0;
}
