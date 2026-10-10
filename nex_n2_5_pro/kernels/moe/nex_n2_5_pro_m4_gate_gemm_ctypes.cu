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
// Selected prefetch depth: 12.

#include "nex_n2_5_pro_m4_gate_gemv_ctypes.cu"
#include "../common/pdl.cuh"

namespace nexrt::sm90::nex_n2_5_pro::M4 {
__device__ __forceinline__ void mma_m16n8k16_bf16_gate_(float (&c)[4], const unsigned int (&a)[4],
                                                        const unsigned int (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

constexpr int kTcWarpK = kDim / 8;            // 512 k per warp = 32 k16 steps
constexpr int kTcXPitch = kTcWarpK * 2 + 16;  // 1040 B: pad kills B-LDS conflicts

// Per-warp x4 staging: 4 tokens x 1 KB k-slice into this warp's smem region.
// Issue and wait are SPLIT so the W-stream prologue can enter the DRAM queue
// between them (sub-wave: the x wait would otherwise serialize ~launch-critical).
__device__ __forceinline__ void tc_stage_x_issue_(const __nv_bfloat16* __restrict__ x4,
                                                  char* __restrict__ shw, int lane, int k0) {
#pragma unroll
    for (int m = 0; m < kM; ++m) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned int dst = static_cast<unsigned int>(
                __cvta_generic_to_shared(shw + m * kTcXPitch + h * 512 + lane * 16));
            const void* src = x4 + static_cast<size_t>(m) * kDim + k0 + (h * 512 + lane * 16) / 2;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
        }
    }
    asm volatile("cp.async.commit_group;");
}

// Only the original 66 CTAs participate in prefetch.
__device__ __forceinline__ void TcPrefetchTail(const char* __restrict__ pf_base, long long pf_bytes,
                                               long long grid_threads) {
    if (pf_base != nullptr && pf_bytes > 0) {
        const long long lines = pf_bytes >> 7;
        const long long gtid = static_cast<long long>(blockIdx.x) * 256 + threadIdx.x;
        for (long long line = gtid; line < lines; line += grid_threads) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(pf_base + (line << 7)));
        }
    }
}

__global__ __launch_bounds__(256) void M4GateGemmKernel(const __nv_bfloat16* __restrict__ x4,
                                                        const __nv_bfloat16* __restrict__ wpack,
                                                        __nv_bfloat16* __restrict__ logits4,
                                                        float* __restrict__ rms_scale4, float eps,
                                                        const char* __restrict__ pf_base,
                                                        long long pf_bytes) {
    NEXRT_PDL_ENTRY_SYNC();
    if (blockIdx.x >= kGateRowsPhysical / 8) {
        const auto* __restrict__ h2 = reinterpret_cast<const __nv_bfloat162*>(x4);
        float rms_acc = 0.0f;
        for (int pair = threadIdx.x; pair < kDimPairs; pair += 256) {
            const float2 fr =
                __bfloat1622float2(h2[(blockIdx.x - kGateRowsPhysical / 8) * kDimPairs + pair]);
            rms_acc += fr.x * fr.x + fr.y * fr.y;
        }
        const float sq_sum = BlockSum<256>(rms_acc);
        if (threadIdx.x == 0) {
            rms_scale4[blockIdx.x - kGateRowsPhysical / 8] =
                rsqrtf(sq_sum * (1.0f / static_cast<float>(kDim)) + eps);
        }
        return;
    }

    constexpr int kDepth = 12;  // uint2 ring slots; ptxas chooses physical registers.
    __shared__ __align__(16) char sh[8][kM][kTcXPitch];
    __shared__ float red[8][32][2];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int gid = lane >> 2;
    const int tig = lane & 3;
    const int tile = blockIdx.x;  // 8 gate rows (528/8 = 66 CTAs)
    const int k0 = warp * kTcWarpK;
    char* const shw = &sh[warp][0][0];

    tc_stage_x_issue_(x4, shw, lane, k0);
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
                         : "=r"(wbuf[nx % kDepth].x), "=r"(wbuf[nx % kDepth].y)
                         : "l"(wp + static_cast<size_t>(nx) * 32));
        }
        const uint2 w = wbuf[s_ % kDepth];
        unsigned b[2] = {0u, 0u};
        if (gid < kM) {
            const char* xp = shw + gid * kTcXPitch + s_ * 32 + tig * 4;
            b[0] = *reinterpret_cast<const unsigned*>(xp);
            b[1] = *reinterpret_cast<const unsigned*>(xp + 16);
        }
        const unsigned int a[4] = {w.x, 0u, w.y, 0u};
        mma_m16n8k16_bf16_gate_(c, a, b);
    }

    red[warp][lane][0] = c[0];
    red[warp][lane][1] = c[1];
    __syncthreads();
    if (threadIdx.x < 32 && tig * 2 < kM) {  // tokens tig*2, tig*2+1; row gid only
        float v0 = 0.0f, v1 = 0.0f;
#pragma unroll
        for (int sp = 0; sp < 8; ++sp) {
            v0 += red[sp][lane][0];
            v1 += red[sp][lane][1];
        }
        const int r = tile * 8 + gid;
        if (r < kGateRowsActive) {
            logits4[static_cast<size_t>(tig * 2) * kGateRowsPhysical + r] = __float2bfloat16(v0);
            logits4[static_cast<size_t>(tig * 2 + 1) * kGateRowsPhysical + r] =
                __float2bfloat16(v1);
        }
    }
    TcPrefetchTail(pf_base, pf_bytes, static_cast<long long>(kGateRowsPhysical / 8) * 256);
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M4

using namespace nexrt::sm90::nex_n2_5_pro::M4;
extern "C" {
// wpack is the lane-major prepack (rig _pack / production packer), NOT row-major
// gate_w. pf_bytes <= 0 (or null pf_base) disables the hint tail.
int NexN25ProM4GateGemmLaunch(const void* x4, const void* wpack, void* logits4, float* rms_scale4,
                              float eps, const void* pf_base, long long pf_bytes, void* stream) {
    M4GateGemmKernel<<<dim3(kGateRowsPhysical / 8 + kM), dim3(256), 0,
                       static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(x4), static_cast<const __nv_bfloat16*>(wpack),
        static_cast<__nv_bfloat16*>(logits4), rms_scale4, eps, static_cast<const char*>(pf_base),
        pf_bytes);
    return static_cast<int>(cudaGetLastError());
}
}

// PDL arm: programmatic-edge launch of the tc8 gate GEMV.
extern "C" int NexN25ProM4GateGemmPdlLaunch(const void* x4, const void* wpack, void* logits4,
                                            float* rms_scale4, float eps, const void* pf_base,
                                            long long pf_bytes, void* stream) {
    return static_cast<int>(NexrtPdlLaunchKernel(
        dim3(kGateRowsPhysical / 8 + kM), dim3(256), 0, static_cast<cudaStream_t>(stream),
        M4GateGemmKernel, static_cast<const __nv_bfloat16*>(x4),
        static_cast<const __nv_bfloat16*>(wpack), static_cast<__nv_bfloat16*>(logits4), rms_scale4,
        eps, static_cast<const char*>(pf_base), pf_bytes));
}

extern "C" int GateRolesAttributes(int* out) {
    cudaFuncAttributes a;
    auto e = cudaFuncGetAttributes(&a, M4GateGemmKernel);
    if (e != cudaSuccess) return int(e);
    out[0] = a.numRegs;
    out[1] = a.sharedSizeBytes;
    out[2] = a.localSizeBytes;
    return 0;
}
