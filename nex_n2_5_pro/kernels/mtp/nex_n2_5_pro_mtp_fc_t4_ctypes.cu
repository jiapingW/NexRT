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

// Four-row MTP FC projection used by the extend phase.
//
// This is a separate MTP-family operation, not an attention o-projection
// variant.  It consumes the row-major FP8 FC weights, reuses each weight tile
// across the four rows, and uses a dedicated FP32 reduction wire and epochs.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M4
#endif

#include "../common/oproj_common.cuh"

namespace nexrt::sm90::nex_n2_5_pro::M4 {
constexpr int kMtpFcRows = 4;
constexpr int kMtpFcCTAs = 128;
constexpr int kMtpFcSlots = 32;
constexpr int kMtpFcThreads = 1024;
constexpr int kMtpFcRowsPerStripe = 16;

// part[j][slot]: this CTA's fp32 partial for draft row j at output row
//   row(slot) = row_base + (slot & 15) + ((slot >> 4) * (kDim / 2))
// which is exactly the producer's `row` mapping for warp index `slot`.
__device__ __forceinline__ void mtp_fc_publish_and_reduce(
    const float part[kMtpFcRows][kMtpFcSlots],
    unsigned long long rep_mc,            // multicast VA of the replicated workspace
    const float* __restrict__ rep_local,  // local VA of the same workspace
    unsigned long long bar_mc_base,       // multicast VA of the per-CTA arrival counters
    int* __restrict__ bar_local_base,     // local VA of the same counters
    int* __restrict__ status, __nv_bfloat16* __restrict__ x4_resid, int rank, int num_ranks,
    int gen, int row_base) {
    const long rank_stride = static_cast<long>(kMtpFcRows) * kDim;
    const long parity_stride = static_cast<long>(kMaxRanks) * rank_stride;
    const long parity_off = static_cast<long>((gen - 1) & 1) * parity_stride;
    const long my_base = parity_off + static_cast<long>(rank) * rank_stride;

    // ---- publish: 32 stripe replications (4 draft rows x 2 halves x 4 quads) ----
    if (threadIdx.x < 32) {
        const int j = threadIdx.x >> 3;  // draft row 0..3
        const int q = threadIdx.x & 7;   // 8 v4 stores per draft row
        const int half = q >> 2;         // 0 = rows [0,16), 1 = rows [kDim/2, +16)
        const int quad = q & 3;
        const int slot0 = half * kMtpFcSlots / 2 + quad * 4;
        const int row0 = row_base + quad * 4 + half * (kDim / 2);
        const float4 v =
            make_float4(part[j][slot0], part[j][slot0 + 1], part[j][slot0 + 2], part[j][slot0 + 3]);
        const long elem = my_base + static_cast<long>(j) * kDim + row0;
        multimem_st_v4_f32(rep_mc + static_cast<unsigned long long>(elem) * sizeof(float), v);
    }
    __threadfence_system();
    __syncthreads();

    // ---- per-CTA arrival: signal once, then wait only for this CTA's peers ----
    if (threadIdx.x == 0) {
        multimem_red_add_rel_sys(bar_mc_base + static_cast<unsigned long long>(blockIdx.x) * 4ull,
                                 1);
        const int target =
            static_cast<int>(static_cast<unsigned>(gen) * static_cast<unsigned>(num_ranks));
        int* ctr = bar_local_base + blockIdx.x;
        int spins = 0;
        while ((int)(ld_acq_sys_i32(ctr) - target) < 0) {
            if (++spins > kMaxSpins) {
                if (status) {
                    atomicCAS(status, 0, 1);
                }
                break;
            }
            __nanosleep(64);
        }
    }
    __syncthreads();

    // Every TP replica must sum in the same order before the BF16 residual store.
    if (threadIdx.x < kMtpFcRows * kMtpFcSlots) {
        const int j = threadIdx.x >> 5;
        const int slot = threadIdx.x & 31;
        const int row = row_base + (slot & 15) + ((slot >> 4) * (kDim / 2));
        const long elem = static_cast<long>(j) * kDim + row;
        float sum = rep_local[parity_off + elem];
#pragma unroll
        for (int r = 1; r < kMaxRanks; ++r) {
            if (r < num_ranks) {
                sum += rep_local[parity_off + static_cast<long>(r) * rank_stride + elem];
            }
        }
        __nv_bfloat16* d = x4_resid + elem;
        *d = __float2bfloat16(__bfloat162float(*d) + sum);
    }
}

__global__ __launch_bounds__(kMtpFcThreads, 1) void MtpFcT4Kernel(
    const __nv_bfloat16* __restrict__ input4, const __nv_bfloat16* __restrict__ gate4,
    const __nv_fp8_e4m3* __restrict__ weights, const float* __restrict__ weight_scales,
    unsigned long long replica_multicast, const float* __restrict__ replica_local,
    unsigned long long barrier_multicast, int* __restrict__ barrier_local,
    unsigned int* __restrict__ iteration, int* __restrict__ status,
    __nv_bfloat16* __restrict__ residual4, int rank, int num_ranks) {
    __shared__ __align__(16) float input_effective[kMtpFcRows][kLocal];
    __shared__ __align__(16) float partial[kMtpFcRows][kMtpFcSlots];
    __shared__ int generation;

    const int lane = threadIdx.x & 31;
    const int local_warp = threadIdx.x >> 5;
    const int row_base = blockIdx.x * kMtpFcRowsPerStripe;

    if (threadIdx.x == 0) {
        generation = static_cast<int>(atomicAdd(&iteration[blockIdx.x], 1u)) + 1;
    }
#pragma unroll
    for (int row = 0; row < kMtpFcRows; ++row) {
        for (int column = threadIdx.x; column < kLocal; column += blockDim.x) {
            float value = __bfloat162float(input4[row * kLocal + column]);
            value *= sigmoid_f32(__bfloat162float(gate4[row * kLocal + column]));
            input_effective[row][column] = value;
        }
    }
    __syncthreads();

    const int output_row = row_base + (local_warp & 15) + ((local_warp >> 4) * (kDim / 2));
    const int output_scale = output_row >> 7;
    const __nv_fp8_e4m3* weight_row =
        weights + static_cast<unsigned long long>(output_row) * kLocal;
    float accumulators[kMtpFcRows] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
    for (int tile = 0; tile < kScaleCols; ++tile) {
        const int column = tile * 128 + lane * 4;
        const float scale = weight_scales[output_scale * kScaleCols + tile];
        const int4 packed = *reinterpret_cast<const int4*>(weight_row + column);
        const __nv_fp8x4_e4m3* words = reinterpret_cast<const __nv_fp8x4_e4m3*>(&packed);
        const float4 weight = float4(words[0]);
#pragma unroll
        for (int row = 0; row < kMtpFcRows; ++row) {
            const float4 input = *reinterpret_cast<const float4*>(&input_effective[row][column]);
            accumulators[row] += scale * (input.x * weight.x + input.y * weight.y +
                                          input.z * weight.z + input.w * weight.w);
        }
    }
#pragma unroll
    for (int row = 0; row < kMtpFcRows; ++row) {
        const float value = warp_sum(accumulators[row]);
        if (lane == 0) {
            partial[row][local_warp] = value;
        }
    }
    __syncthreads();

    mtp_fc_publish_and_reduce(partial, replica_multicast, replica_local, barrier_multicast,
                              barrier_local, status, residual4, rank, num_ranks, generation,
                              row_base);
}

extern "C" int NexN25ProMtpFcT4Launch(const void* input4, const void* gate4, const void* weights,
                                      const void* weight_scales,
                                      unsigned long long replica_multicast,
                                      const void* replica_local,
                                      unsigned long long barrier_multicast, void* barrier_local,
                                      void* iteration, void* status, void* residual4, int rank,
                                      int num_ranks, void* stream) {
    MtpFcT4Kernel<<<dim3(kMtpFcCTAs), dim3(kMtpFcThreads), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(input4), static_cast<const __nv_bfloat16*>(gate4),
        static_cast<const __nv_fp8_e4m3*>(weights), static_cast<const float*>(weight_scales),
        replica_multicast, static_cast<const float*>(replica_local), barrier_multicast,
        static_cast<int*>(barrier_local), static_cast<unsigned int*>(iteration),
        static_cast<int*>(status), static_cast<__nv_bfloat16*>(residual4), rank, num_ranks);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M4