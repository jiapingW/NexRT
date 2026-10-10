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

#define L1_VALID 64
#define L1_DEPTH 5
#define L1_MODE 2
#define L1_BATCH 1
#define L1_PIPE 2
#define L1_XOR 1
#define L1_COMPACT 1
#define L1_EARLY 0
#define L1_ENDSYNC 0
#define L1_LEADER 1
#define L1_TRIGGER 16
#include <cuda.h>
#include <new>
#define L1_GROUPS (256 / L1_VALID)
// M=8 MoE L1: row-major FP8 W13, two warps and a depth-5 K128 copy ring.
// K16 fragments retain the existing accumulation order and PDL handoff.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace nexrt::sm90::nex_n2_5_pro::M8 {
constexpr int kDim = 4096;
constexpr int kTopK = 10;
constexpr int kM8 = 8;
constexpr int kNExp = 512;
constexpr int kRinter = 128;
constexpr int kPackedWords = 64;
// Eight token lanes each read four adjacent IDs: 4-word row padding
// gives each lane a distinct four-bank group. Keep global rows at 64 words.
constexpr int kPackedSharedWords = kPackedWords + 4;
constexpr int kMaxDesc = kM8 * kTopK + 1;
constexpr int kHRows = kTopK * kRinter + kRinter;
constexpr int kTileBytes = 2 * 16 * kDim;
constexpr int kStageBytes = 2 * 16 * 128;
constexpr int kWmDepth = 5;

struct alignas(128) M8RoutePlan {
    int32_t n_desc;
    int32_t reserved[31];
    alignas(128) int32_t expert_id[128];
};

__device__ __forceinline__ float wm_silu(float x) { return x / (1.0f + __expf(-x)); }

__device__ __forceinline__ void wm_cp_async16_cg(void* smem_dst, const void* gmem_src) {
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;" ::"r"(d), "l"(gmem_src));
}
__device__ __forceinline__ void wm_cp_async16_ca(void* smem_dst, const void* gmem_src) {
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(d), "l"(gmem_src));
}
__device__ __forceinline__ void wm_cp_async8_ca(void* smem_dst, const void* gmem_src) {
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(d), "l"(gmem_src));
}

__device__ __forceinline__ void wm_mma_m16n8k16_bf16(float (&c)[4], const unsigned (&a)[4],
                                                     const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void native_fp8_mma(float (&c)[4], const unsigned (&a)[4],
                                               const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ------------------------------- route plan ----------------------------------
struct WmPlanSmem {
    int32_t pk[kM8 * kPackedWords];  // 2 KiB staged packed8
    char ptbl[kNExp][kM8];           // 4 KiB (expert, token) marks
    int wsum[16];
};

// ONE CTA, 256 threads. packed8 word layout (shipped top-k): [j*64 + 0..9]
// expert ids, [+10..19] route weights (float bits), [+63] shared weight. Only
// the ids matter here: membership comes from ids, so zero-weight routes stay.
__global__ __launch_bounds__(256) void M8WmPlanKernel(const int32_t* __restrict__ packed8,
                                                      M8RoutePlan* __restrict__ plan) {
    __shared__ WmPlanSmem sm;
    const int tid = threadIdx.x;
    // Whole 2 KiB staged once: 256 coalesced 8 B vector loads.
    reinterpret_cast<int64_t*>(sm.pk)[tid] = reinterpret_cast<const int64_t*>(packed8)[tid];
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {  // 4 KiB init: 8 B per thread
        *reinterpret_cast<int64_t*>(sm.ptbl[pass * 256 + tid]) = -1;
    }
    __syncthreads();
    if (tid < kM8 * kTopK) {        // 80 parallel marks: (e, j) pairs
        const int j = tid / kTopK;  // are distinct (top-k)
        const int s = tid - j * kTopK;
        sm.ptbl[sm.pk[j * kPackedWords + s]][j] = static_cast<char>(s);
    }
    __syncthreads();
    const int lane = tid & 31;
    const int warp = tid >> 5;
    unsigned mm[2];
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {  // each thread tests two experts;
        const int e = pass * 256 + tid;     // pass-major keeps ids ascending
        const bool act = *reinterpret_cast<const int64_t*>(sm.ptbl[e]) != -1;
        mm[pass] = __ballot_sync(0xffffffffu, act);
        if (lane == 0) sm.wsum[pass * 8 + warp] = __popc(mm[pass]);
    }
    __syncthreads();
    if (tid == 0) {
        int acc = 0;
#pragma unroll
        for (int w_ = 0; w_ < 16; ++w_) {  // 16-segment shared prefix
            const int n = sm.wsum[w_];
            sm.wsum[w_] = acc;
            acc += n;
        }
        plan->expert_id[acc] = -1;  // shared expert, LAST
        plan->n_desc = acc + 1;
    }
    __syncthreads();
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        if ((mm[pass] >> lane) & 1u) {
            const int pos = sm.wsum[pass * 8 + warp] + __popc(mm[pass] & ((1u << lane) - 1u));
            plan->expert_id[pos] = pass * 256 + tid;
        }
    }
    // Publication ordering: consumers are later launches on the SAME stream;
    // kernel completion is the fence. No atomics, no cross-grid protocol.
}

constexpr int kW4Threads = 64;
constexpr int kW4Grid = kMaxDesc * 4;
constexpr int kW4XPitch = (kDim / 2) * 2 + 16;

constexpr int kW4RingOff = kM8 * kW4XPitch;
constexpr int kW4RingWarp = kWmDepth * kStageBytes;
constexpr int kW4SclOff = kW4RingOff + 2 * kW4RingWarp;
constexpr int kW4DescOff = kW4SclOff + 256;
constexpr int kW4SoutOff = kW4DescOff + 128;
constexpr int kW4XsOff = kW4SoutOff + 2 * kM8 * 16 * 4;
constexpr int kW4Dyn = kW4XsOff + 8 * 33 * 4;
constexpr int kW4GroupBytes = 4 * kTileBytes;
constexpr int kW4PdlTrigger = L1_TRIGGER;

// MMA scalar stores and float4 transpose loads both use distinct banks.
__device__ __forceinline__ int wm_sout_offset(int row, int col) {
    return (row * 16 + col) ^ ((row >> 1) * 8);
}

__device__ __forceinline__ void wm4_l1_body(
    const M8RoutePlan* __restrict__ plan, const int32_t* __restrict__ packed8,
    const __nv_bfloat16* __restrict__ xn8, const __nv_fp8_e4m3* __restrict__ w13_raw,
    const float* __restrict__ w13s, const __nv_fp8_e4m3* __restrict__ shared_w13_raw,
    const float* __restrict__ sscales, float* __restrict__ rawh8,
    const __nv_bfloat16* __restrict__ xq8, const float* __restrict__ xs8) {
    const int desc = blockIdx.x / L1_GROUPS;
    const int group = blockIdx.x % L1_GROUPS;
    const int n_desc = plan->n_desc;
    if (desc >= n_desc) return;
    const int e = -1;  // Shared-expert fallback only.
    const bool routed = e >= 0;

    extern __shared__ __align__(1024) char sm[];
    char* const sxh = sm;
    float* const sscl = reinterpret_cast<float*>(sm + kW4SclOff);
    int* const s_tok = reinterpret_cast<int*>(sm + kW4DescOff);
    float* const s_w = reinterpret_cast<float*>(sm + kW4DescOff + 32);
    int* const s_hoff = reinterpret_cast<int*>(sm + kW4DescOff + 64);
    int* const s_c = reinterpret_cast<int*>(sm + kW4DescOff + 96);

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tile = group * 2 + warp;
    const int gid = lane >> 2;
    const int tig = lane & 3;
    char* const ring = sm + kW4RingOff + warp * kW4RingWarp;
    float* const sout = reinterpret_cast<float*>(sm + kW4SoutOff) + warp * (kM8 * 16);

    const __nv_fp8_e4m3* const gbase =
        (routed ? w13_raw + static_cast<size_t>(e) * (2 * kW4GroupBytes) : shared_w13_raw);
    const float* const scg = routed ? w13s + static_cast<size_t>(e) * 64 : sscales;
    const auto* raw_lane_base =
        gbase + static_cast<size_t>(tile * 16 + (lane >> 3)) * kDim + (lane & 7) * 16;
    const int raw_lane_smem = (lane * 16) ^ ((lane >> 3) * 16);

    if (warp == 0) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            wm_cp_async16_ca(
                sxh + (j * 2 + (lane >> 4)) * (kPackedSharedWords * 4) + (lane & 15) * 16,
                reinterpret_cast<const char*>(packed8) + j * 512 + lane * 16);
        }
        wm_cp_async8_ca(reinterpret_cast<char*>(sscl) + lane * 8,
                        reinterpret_cast<const char*>(scg) + lane * 8);
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncwarp();
        const int32_t* const pk = reinterpret_cast<const int32_t*>(sxh);
        int slot = routed ? -1 : kTopK;
        int weight_bits = 0;
        if (lane < kM8) {
            if (routed) {
// Read weights from the already-staged packet at fixed,
// unrolled slot offsets. Extra tiny metadata reads are fine:
// each token's padded row has distinct banks, and selecting
// the value here needs neither a global gather nor a transpose.
#pragma unroll
                for (int s = 0; s < kTopK; ++s) {
                    const int bits = pk[lane * kPackedSharedWords + kTopK + s];
                    if (pk[lane * kPackedSharedWords + s] == e) {
                        slot = s;
                        weight_bits = bits;
                    }
                }
            } else {
                weight_bits = pk[lane * kPackedSharedWords + 63];
            }
        }
        const bool act = (lane < kM8) && (slot >= 0);
        const unsigned amask = __ballot_sync(0xffffffffu, act);
        if (act) {
            const int pos = __popc(amask & ((1u << lane) - 1u));
            s_tok[pos] = lane;
            s_w[pos] = __int_as_float(weight_bits);
            s_hoff[pos] = lane * kHRows + slot * kRinter;
        }
        if (lane == 0) s_c[0] = __popc(amask);
    } else {
#pragma unroll
        for (int st = 0; st < kWmDepth - 1; ++st) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                wm_cp_async16_cg(
                    ring + st * kStageBytes + j * 512 + (raw_lane_smem ^ ((j & 1) << 6)),
                    raw_lane_base + st * 128 + (j & 3) * (4 * kDim) + (j >> 2) * (128 * kDim));
            }
            asm volatile("cp.async.commit_group;");
        }
    }
    asm volatile("bar.sync 1, 64;" ::: "memory");
    const int c = s_c[0];
    const __nv_bfloat16* xsource = routed ? xq8 : xn8;
    const int chunks = 256;  // Entire FP8 input or half BF16 input fits this tile.
    const int xstride = routed ? kDim : 2 * kDim;
    const int halfbytes = xstride / 2;
    float* s_xs = reinterpret_cast<float*>(sm + kW4XsOff);
    for (int i = threadIdx.x; i < 256; i += kW4Threads)
        s_xs[(i / 32) * 33 + i % 32] =
            (routed && (i / 32) < c) ? xs8[s_tok[i / 32] * 32 + i % 32] : 1.f;

    if (warp == 0) {
#pragma unroll
        for (int st = 0; st < kWmDepth - 1; ++st) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                wm_cp_async16_cg(
                    ring + st * kStageBytes + j * 512 + (raw_lane_smem ^ ((j & 1) << 6)),
                    raw_lane_base + st * 128 + (j & 3) * (4 * kDim) + (j >> 2) * (128 * kDim));
            }
            asm volatile("cp.async.commit_group;");
        }
    }

    const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
    for (int i = threadIdx.x; i < (kM8 - c) * chunks; i += kW4Threads) {
        *reinterpret_cast<uint4*>(sxh + (c + (i / chunks)) * kW4XPitch + (i % chunks) * 16) = z4;
    }

    for (int i = threadIdx.x; i < c * chunks; i += kW4Threads) {
        const int n = i / chunks;
        const int off = (i % chunks) * 16;
        wm_cp_async16_ca(
            sxh + n * kW4XPitch + off,
            reinterpret_cast<const char*>(xsource) + static_cast<size_t>(s_tok[n]) * xstride + off);
    }
    asm volatile("cp.async.commit_group;");

    asm volatile("cp.async.wait_group 0;");
    asm volatile("bar.sync 1, 64;" ::: "memory");

    float accg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float accu[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float cg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float cu[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int rslot = 0;
    int islot = kWmDepth - 1;
    const char* const sxg = sxh + gid * kW4XPitch;
#pragma unroll 1
    for (int cl = 0; cl < 32; ++cl) {
        if (cl == 16 && !routed) {
            asm volatile("bar.sync 1, 64;" ::: "memory");
            for (int i = threadIdx.x; i < c * chunks; i += kW4Threads) {
                const int n = i / chunks;
                const int off = (i % chunks) * 16;
                wm_cp_async16_ca(sxh + n * kW4XPitch + off,
                                 reinterpret_cast<const char*>(xsource) +
                                     static_cast<size_t>(s_tok[n]) * xstride + halfbytes + off);
            }
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 0;");
            asm volatile("bar.sync 1, 64;" ::: "memory");
        }
        if (cl == kW4PdlTrigger) cudaTriggerProgrammaticLaunchCompletion();
        const char* weight_stage = ring + rslot * kStageBytes;
        if (cl + kWmDepth - 1 < 32) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                wm_cp_async16_cg(
                    ring + islot * kStageBytes + j * 512 + (raw_lane_smem ^ ((j & 1) << 6)),
                    raw_lane_base + (cl + kWmDepth - 1) * 128 + (j & 3) * (4 * kDim) +
                        (j >> 2) * (128 * kDim));
            }
        }
        asm volatile("cp.async.commit_group;");
        // Previous iteration finished reading this free slot. Issue its
        // next copy before waiting for the distinct current read slot.
        asm volatile("cp.async.wait_group %0;" ::"n"(kWmDepth - 1));
        __syncwarp();
        if (routed) {
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                unsigned ag[4], au[4];
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const int row = gid + ((p & 1) ? 8 : 0);
                    const int col = kk * 32 + ((p & 2) ? 16 : 0) + tig * 4;
                    const int off = (row * 128 + col) ^ ((row & 7) << 4);
                    ag[p] = *reinterpret_cast<const unsigned*>(weight_stage + off);
                    au[p] = *reinterpret_cast<const unsigned*>(weight_stage + 2048 + off);
                }
                const char* xp = sxg + cl * 128 + kk * 32 + tig * 4;
                unsigned b[2] = {*reinterpret_cast<const unsigned*>(xp),
                                 *reinterpret_cast<const unsigned*>(xp + 16)};
                native_fp8_mma(cg, ag, b);
                native_fp8_mma(cu, au, b);
            }
        } else {
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                unsigned ag[4], au[4];
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const int row = gid + ((p & 1) ? 8 : 0);
                    const int col = kk * 16 + ((p & 2) ? 8 : 0) + tig * 2;
                    const int offset = (row * 128 + col) ^ ((row & 7) << 4);
                    const auto g8 =
                        *reinterpret_cast<const __nv_fp8x2_e4m3*>(weight_stage + offset);
                    const auto u8 =
                        *reinterpret_cast<const __nv_fp8x2_e4m3*>(weight_stage + 16 * 128 + offset);
                    const __nv_bfloat162 g = __float22bfloat162_rn(static_cast<float2>(g8));
                    const __nv_bfloat162 u = __float22bfloat162_rn(static_cast<float2>(u8));
                    ag[p] = *reinterpret_cast<const unsigned*>(&g);
                    au[p] = *reinterpret_cast<const unsigned*>(&u);
                }
                const char* xp = sxg + (cl & 15) * 256 + kk * 32 + tig * 4;
                unsigned b[2] = {*reinterpret_cast<const unsigned*>(xp),
                                 *reinterpret_cast<const unsigned*>(xp + 16)};
                wm_mma_m16n8k16_bf16(cg, ag, b);
                wm_mma_m16n8k16_bf16(cu, au, b);
            }
        }
        const float sg = sscl[cl];
        const float su = sscl[32 + cl];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float as = s_xs[(tig * 2 + (i & 1)) * 33 + cl];
            accg[i] += (sg * as) * cg[i];
            accu[i] += (su * as) * cu[i];
            cg[i] = 0.0f;
            cu[i] = 0.0f;
        }
        __syncwarp();  // Finish all per-warp shared-weight reads before next asynchronous slot reuse.
        rslot = (rslot + 1 == kWmDepth) ? 0 : rslot + 1;
        islot = (islot + 1 == kWmDepth) ? 0 : islot + 1;
    }

    float h[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (routed) {
            const float g = __bfloat162float(__float2bfloat16(accg[i]));
            const float u = __bfloat162float(__float2bfloat16(accu[i]));
            h[i] = __bfloat162float(__float2bfloat16(__fdividef(g, 1.f + __expf(-g)) * u));
        } else
            h[i] = wm_silu(accg[i]) * accu[i] * s_w[tig * 2 + (i & 1)];
    }
    sout[wm_sout_offset(tig * 2, gid)] = h[0];
    sout[wm_sout_offset(tig * 2 + 1, gid)] = h[1];
    sout[wm_sout_offset(tig * 2, gid + 8)] = h[2];
    sout[wm_sout_offset(tig * 2 + 1, gid + 8)] = h[3];
    __syncwarp();
    const int n = lane >> 2;
    const int q = lane & 3;
    if (n < c) {
        const float4 v = *reinterpret_cast<const float4*>(sout + wm_sout_offset(n, q * 4));
        *reinterpret_cast<float4*>(rawh8 + s_hoff[n] + tile * 16 + q * 4) = v;
    }
}

#include "m8_wgmma_k32_l1.cuh"
constexpr int TuneDyn = WG1Dyn > kW4Dyn ? WG1Dyn : kW4Dyn;
constexpr int TuneGrid = kMaxDesc * L1_GROUPS;

__global__ __launch_bounds__(128) void M8MoeWm4L1Kernel(
    const M8RoutePlan* __restrict__ plan, const int32_t* __restrict__ packed8,
    const __nv_bfloat16* __restrict__ xn8, const __nv_fp8_e4m3* __restrict__ w13_raw,
    const float* __restrict__ w13s, const __nv_fp8_e4m3* __restrict__ shared_w13_raw,
    const float* __restrict__ sscales, float* __restrict__ rawh8,
    const __nv_bfloat16* __restrict__ xq8, const float* __restrict__ xs8,
    const __grid_constant__ CUtensorMap w_map) {
    if ((blockIdx.x / L1_GROUPS) >= plan->n_desc) return;
    if (plan->expert_id[blockIdx.x / L1_GROUPS] >= 0)
        wg1_body(plan, packed8, w13_raw, w13s, rawh8, xq8, xs8, &w_map);
    else if (threadIdx.x < 64 && blockIdx.x % L1_GROUPS < 4)
        wm4_l1_body(plan, packed8, xn8, w13_raw, w13s, shared_w13_raw, sscales, rawh8, xq8, xs8);
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M8
using namespace nexrt::sm90::nex_n2_5_pro::M8;

extern "C" int NexN25ProM8WmPlanLaunch(const void* packed8, void* plan, void* stream) {
    M8WmPlanKernel<<<1, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const int32_t*>(packed8), static_cast<M8RoutePlan*>(plan));
    return static_cast<int>(cudaGetLastError());
}
extern "C" int NexN25ProM8WmPlanInts() { return static_cast<int>(sizeof(M8RoutePlan) / 4); }
extern "C" const char* NexN25ProM8WmCudaErrorString(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}

extern "C" int NexN25ProM8WmPrepare() {
    return static_cast<int>(cudaFuncSetAttribute(
        M8MoeWm4L1Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, TuneDyn));
}
extern "C" int NexN25ProM8Wm4L1Attributes(int* out) {
    cudaFuncAttributes a;
    auto rc = cudaFuncGetAttributes(&a, M8MoeWm4L1Kernel);
    if (rc != cudaSuccess) return static_cast<int>(rc);
    int blocks = 0;
    rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, M8MoeWm4L1Kernel, 128, TuneDyn);
    if (rc != cudaSuccess) return static_cast<int>(rc);
    out[0] = a.numRegs;
    out[1] = a.sharedSizeBytes;
    out[2] = a.localSizeBytes;
    out[3] = TuneDyn;
    out[4] = blocks;
    out[5] = TuneGrid;
    out[6] = 128;
    return 0;
}

// The runtime owns one descriptor per model weight, not a process-global cache.
struct WeightMap {
    CUtensorMap tensor;
    CUcontext context;
    const void* weight;
};
extern "C" int NexN25ProM8WmMapCreate(const void* weight, void** result) {
    if (!weight || !result) return static_cast<int>(cudaErrorInvalidValue);
    *result = nullptr;
    auto* owned = new (std::nothrow) WeightMap{};
    if (!owned) return static_cast<int>(cudaErrorMemoryAllocation);
    CUresult rc = cuCtxGetCurrent(&owned->context);
    if (rc == CUDA_SUCCESS && !owned->context) rc = CUDA_ERROR_INVALID_CONTEXT;
    const cuuint64_t dims[2] = {4096, 512 * 256}, strides[1] = {4096};
    const cuuint32_t box[2] = {128, 32}, elements[2] = {1, 1};
    if (rc == CUDA_SUCCESS)
        rc = cuTensorMapEncodeTiled(
            &owned->tensor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(weight), dims,
            strides, box, elements, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (rc != CUDA_SUCCESS) {
        delete owned;
        return 10000 + static_cast<int>(rc);
    }
    owned->weight = weight;
    *result = owned;
    return 0;
}
extern "C" void NexN25ProM8WmMapDestroy(void* handle) { delete static_cast<WeightMap*>(handle); }

extern "C" int NexN25ProM8Wm4L1Launch(const void* plan, const void* packed, const void* xn,
                                      const void* w13, const void* ws, const void* sw,
                                      const void* ss, void* hidden, const void* xq, const void* xs,
                                      const void* descriptor, void* stream) {
    auto* owned = static_cast<const WeightMap*>(descriptor);
    CUcontext context = nullptr;
    CUresult rc = cuCtxGetCurrent(&context);
    if (rc != CUDA_SUCCESS) return 10000 + static_cast<int>(rc);
    if (!owned || owned->weight != w13 || owned->context != context)
        return static_cast<int>(cudaErrorInvalidValue);
    M8MoeWm4L1Kernel<<<TuneGrid, 128, TuneDyn, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const M8RoutePlan*>(plan), static_cast<const int32_t*>(packed),
        static_cast<const __nv_bfloat16*>(xn), static_cast<const __nv_fp8_e4m3*>(w13),
        static_cast<const float*>(ws), static_cast<const __nv_fp8_e4m3*>(sw),
        static_cast<const float*>(ss), static_cast<float*>(hidden),
        static_cast<const __nv_bfloat16*>(xq), static_cast<const float*>(xs), owned->tensor);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int NexN25ProM8Wm4L1Ctas() { return TuneGrid; }
extern "C" int NexN25ProM8Wm4L1Smem() { return TuneDyn; }
extern "C" int NexN25ProM8Wm4L1BlocksPerSm() {
    int blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, M8MoeWm4L1Kernel, 128, TuneDyn);
    return blocks;
}
