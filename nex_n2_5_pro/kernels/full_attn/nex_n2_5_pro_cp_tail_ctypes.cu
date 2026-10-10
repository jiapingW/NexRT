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

// Production CP tail: peer-summary merge, gate, and TP o_proj partial.
#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY bs1
#endif

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY {
namespace cg = cooperative_groups;

constexpr int kCpRanks = 4;
constexpr int kLocalHeads = 4;
constexpr int kHeadDim = 256;
constexpr int kLocal = kLocalHeads * kHeadDim;
constexpr int kDim = 4096;
constexpr int kScaleCols = kLocal / 128;
constexpr int kRowsPerStripeHalf = 16;
constexpr int kMaxSpins = 1 << 30;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float sigmoid_f32(float x) { return 1.0f / (1.0f + __expf(-x)); }

__device__ __forceinline__ float dot_x4_wfp8x4(float4 x4, const __nv_fp8_e4m3* __restrict__ w,
                                               int offset) {
    const __nv_fp8x4_e4m3 w4 = *reinterpret_cast<const __nv_fp8x4_e4m3*>(w + offset);
    const float4 wf = float4(w4);
    return x4.x * wf.x + x4.y * wf.y + x4.z * wf.z + x4.w * wf.w;
}

__device__ __forceinline__ int ld_acq_sys_i32(const int* ptr) {
    int ret;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr) : "memory");
    return ret;
}

__device__ __forceinline__ void wait_for_cp_producers(const int* __restrict__ bar_local,
                                                      int* __restrict__ status,
                                                      const int* __restrict__ cur_gen,
                                                      int num_ranks) {
    cg::grid_group grid = cg::this_grid();
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
        const int gen = ld_acq_sys_i32(cur_gen);
        if (gen <= 0) {
            if (status) atomicCAS(status, 0, 2);
        }
        const int target = gen * num_ranks;
        int spins = 0;
        while ((int)(ld_acq_sys_i32(bar_local) - target) < 0) {
            if (++spins > kMaxSpins) {
                if (status) atomicCAS(status, 0, 1);
                break;
            }
            __nanosleep(64);
        }
    }
    grid.sync();
}

__device__ __forceinline__ void stage_attn_bf16(const __nv_bfloat16* __restrict__ attn,
                                                float* __restrict__ x_eff) {
    for (int k = threadIdx.x; k < kLocal; k += blockDim.x) {
        x_eff[k] = __bfloat162float(attn[k]);
    }
}

__device__ __forceinline__ void run_oproj_from_staged(const float* __restrict__ x_eff,
                                                      const __nv_fp8_e4m3* __restrict__ W,
                                                      const float* __restrict__ w_scales,
                                                      float* __restrict__ partial) {
    const int local_warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * kRowsPerStripeHalf;
    const int row = row_base + (local_warp & 15) + ((local_warp >> 4) * (kDim / 2));
    const int row_scale = row >> 7;
    const __nv_fp8_e4m3* row_w = W + static_cast<unsigned long long>(row) * kLocal;

    float lane_acc = 0.0f;
#pragma unroll
    for (int tile = 0; tile < kScaleCols; ++tile) {
        const int k = tile * 128 + lane * 4;
        const float4 x4 = *reinterpret_cast<const float4*>(x_eff + k);
        const float scale = w_scales[row_scale * kScaleCols + tile];
        lane_acc += scale * dot_x4_wfp8x4(x4, row_w, k);
    }

    const float acc = warp_sum(lane_acc);
    if (lane == 0) {
        partial[row] = acc;
    }
}
__global__ __launch_bounds__(1024, 2) void SummaryThenPartialSlotProducerWaitKernel(
    const int64_t* __restrict__ peer_ctx, const int64_t* __restrict__ peer_lse,
    const __nv_bfloat16* __restrict__ gate, __nv_bfloat16* __restrict__ summary,
    const __nv_fp8_e4m3* __restrict__ W, const float* __restrict__ w_scales,
    const int64_t* __restrict__ peer_partial2, const int* __restrict__ cp_bar_local,
    int* __restrict__ cp_status, const int* __restrict__ cp_cur_gen,
    const int* __restrict__ tp_bar_local, int* __restrict__ tp_cur_gen, int cp_rank,
    int cp_num_ranks, int tp_rank, int tp_num_ranks) {
    wait_for_cp_producers(cp_bar_local, cp_status, cp_cur_gen, cp_num_ranks);
    cg::grid_group grid = cg::this_grid();

    if (blockIdx.x < kLocalHeads * 8 && threadIdx.x < 32) {
        const int hlocal = blockIdx.x >> 3;
        const int dblock = blockIdx.x & 7;
        const int d = dblock * 32 + threadIdx.x;
        const int h = cp_rank * kLocalHeads + hlocal;

        float m = -3.402823466e38f;
#pragma unroll
        for (int r = 0; r < kCpRanks; ++r) {
            if (r < cp_num_ranks) {
                const float* lse = reinterpret_cast<const float*>(peer_lse[r]);
                m = fmaxf(m, lse[h]);
            }
        }

        float denom = 0.0f;
        float acc = 0.0f;
#pragma unroll
        for (int r = 0; r < kCpRanks; ++r) {
            if (r < cp_num_ranks) {
                const float* lse = reinterpret_cast<const float*>(peer_lse[r]);
                const __nv_bfloat16* ctx = reinterpret_cast<const __nv_bfloat16*>(peer_ctx[r]);
                const float w = __expf(lse[h] - m);
                denom += w;
                acc += w * __bfloat162float(ctx[h * kHeadDim + d]);
            }
        }

        const int out_idx = hlocal * kHeadDim + d;
        const float y = (acc / denom) * sigmoid_f32(__bfloat162float(gate[out_idx]));
        summary[out_idx] = __float2bfloat16(y);
    }
    grid.sync();

    __shared__ __align__(16) float x_eff[kLocal];
    __shared__ int cta_gen;

    if (threadIdx.x == 0) {
        const int completed = ld_acq_sys_i32(tp_bar_local);
        const int gen = (tp_num_ranks == 8 ? (completed >> 3) : completed / tp_num_ranks) + 1;
        cta_gen = gen;
        if (blockIdx.x == 0) {
            tp_cur_gen[0] = gen;
        }
    }
    stage_attn_bf16(summary, x_eff);
    __syncthreads();

    const long off = static_cast<long>((cta_gen - 1) & 1) * kDim;
    float* my_partial = reinterpret_cast<float*>(peer_partial2[tp_rank]) + off;
    run_oproj_from_staged(x_eff, W, w_scales, my_partial);
}
template <typename Kernel, typename... Args>
int launch_coop(Kernel kernel, dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                Args... args) {
    void* params[] = {static_cast<void*>(&args)...};
    return static_cast<int>(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel), grid,
                                                        block, params, smem, stream));
}

}  // namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
using namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY;
extern "C" int NexN25ProCpTailLaunch(const void* peer_ctx, const void* peer_lse, const void* gate,
                                     void* summary, const void* W, const void* w_scales,
                                     const void* peer_partial2, const void* cp_bar_local,
                                     void* cp_status, const void* cp_cur_gen,
                                     const void* tp_bar_local, void* tp_cur_gen, int cp_rank,
                                     int cp_num_ranks, int tp_rank, int tp_num_ranks,
                                     void* stream) {
    if (cp_rank < 0 || cp_rank >= cp_num_ranks || cp_num_ranks != kCpRanks) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (tp_rank < 0 || tp_rank >= tp_num_ranks || tp_num_ranks <= 0 || tp_num_ranks > 8) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    return launch_coop(
        SummaryThenPartialSlotProducerWaitKernel, dim3(128, 1, 1), dim3(1024, 1, 1), 0,
        reinterpret_cast<cudaStream_t>(stream), reinterpret_cast<const int64_t*>(peer_ctx),
        reinterpret_cast<const int64_t*>(peer_lse), reinterpret_cast<const __nv_bfloat16*>(gate),
        reinterpret_cast<__nv_bfloat16*>(summary), reinterpret_cast<const __nv_fp8_e4m3*>(W),
        reinterpret_cast<const float*>(w_scales), reinterpret_cast<const int64_t*>(peer_partial2),
        reinterpret_cast<const int*>(cp_bar_local), reinterpret_cast<int*>(cp_status),
        reinterpret_cast<const int*>(cp_cur_gen), reinterpret_cast<const int*>(tp_bar_local),
        reinterpret_cast<int*>(tp_cur_gen), cp_rank, cp_num_ranks, tp_rank, tp_num_ranks);
}
extern "C" const char* NexN25ProCpTailErrorString(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
