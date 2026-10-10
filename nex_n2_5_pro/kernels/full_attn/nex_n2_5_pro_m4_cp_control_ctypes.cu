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

// Control helpers for M=4 context-parallel attention.
//
// The M=4 verify control block (frozen int32[16] v1 region) remains the sole
// authoritative state.  The
// first kernel snapshots its current committed length and four verify-input
// tokens into four ordinary [token_id, position] slot views.  The unchanged
// FullAttnCP device-S kernels consume one view per provisional row.
//
// The second kernel folds every local CP/TP status into the verify commit's existing
// ws_status input before the unchanged verify-commit row producer publishes that
// rank's status.  All producers, this fold, and that producer launch on the same
// CUDA stream.  CUDA stream semantics therefore complete prior kernels before
// this fold and complete this fold before the producer; all accesses use the
// ordinary generic/global proxy and the same virtual addresses.  No multicast
// alias is crossed here, so this helper neither extends nor makes a new proof
// claim about the verify commit's multicast-alias fence protocol.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M4
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "../common/m4_control_block.cuh"

namespace nexrt::sm90::nex_n2_5_pro::M4 {
constexpr int kControlThreads = 32;
constexpr int kHeadDim = 256;
constexpr int kHid = 4096;
constexpr int kCpHeads = 16;
constexpr int kLocalHeads = 4;
constexpr int kCpQRows = kCpHeads * kHeadDim;
constexpr int kLocalRows = kLocalHeads * kHeadDim;
constexpr int kProjectedRows = kCpQRows + kLocalRows + 2 * kHeadDim;
constexpr int kScaleBlocksK = kHid / 128;
constexpr int kTilesK = kHid / 512;
constexpr int kRowsPerWarp = 2;
constexpr int kSplitK = 2;
constexpr int kGroupsPerCta = 2;
constexpr int kProjectionThreads = kGroupsPerCta * kSplitK * 32;

__global__ __launch_bounds__(kControlThreads) void StageCpSlotsKernel(
    const int* __restrict__ control, int* __restrict__ slots) {
    const int row = threadIdx.x;
    if (row < m4ctl::kM) {
        const int committed_len = m4ctl::CommittedLen(control);
        slots[2 * row] = control[m4ctl::kVerifyInputIds + row];
        slots[2 * row + 1] = m4ctl::RowPosition(committed_len, row);
    }
}

__global__ __launch_bounds__(kControlThreads) void FoldCpStatusKernel(
    const unsigned long long* __restrict__ status_pointers, int count,
    int* __restrict__ destination, int rank, int injected_rank) {
    if (threadIdx.x != 0) {
        return;
    }
    int failed = destination[0] != 0;
    for (int index = 0; index < count; ++index) {
        const auto address = static_cast<uintptr_t>(status_pointers[index]);
        const int* status = reinterpret_cast<const int*>(address);
        failed |= status != nullptr && status[0] != 0;
    }
    failed |= rank == injected_rank;
    if (failed) {
        destination[0] |= m4ctl::kStatusUpstreamFailed;
    }
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float butterfly_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ void scatter_projected_row(
    int projected_row, int draft_row, float accumulator, __nv_bfloat16* __restrict__ q_group4,
    __nv_bfloat16* __restrict__ gate4, __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache, const int* __restrict__ slots4) {
    const __nv_bfloat16 value = __float2bfloat16(accumulator);
    if (projected_row < kCpQRows) {
        q_group4[static_cast<size_t>(draft_row) * kCpQRows + projected_row] = value;
        return;
    }
    if (projected_row < kCpQRows + kLocalRows) {
        gate4[static_cast<size_t>(draft_row) * kLocalRows + projected_row - kCpQRows] = value;
        return;
    }
    const int position = slots4[2 * draft_row + 1];
    if (projected_row < kCpQRows + kLocalRows + kHeadDim) {
        k_cache[static_cast<size_t>(position) * kHeadDim + projected_row - kCpQRows - kLocalRows] =
            value;
        return;
    }
    v_cache[static_cast<size_t>(position) * kHeadDim + projected_row - kCpQRows - kLocalRows -
            kHeadDim] = value;
}

// Four-row transcription of the unchanged CP R2S2G2 projection.  Every row
// keeps the original tile, split-reduction, and BF16 scatter order; only the
// FP8 weight/scaling loads are shared across the four independent x vectors.
__global__ __launch_bounds__(kProjectionThreads) void M4CpQGate4Kernel(
    const __nv_bfloat16* __restrict__ x4, const __nv_fp8_e4m3* __restrict__ weights,
    const float* __restrict__ scales, __nv_bfloat16* __restrict__ q_group4,
    __nv_bfloat16* __restrict__ gate4, __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache, const int* __restrict__ slots4) {
    constexpr int kTilesPerSplit = kTilesK / kSplitK;
    constexpr int kNumGroups = kProjectedRows / kRowsPerWarp;
    __shared__ float partial[kGroupsPerCta][kRowsPerWarp][kSplitK][m4ctl::kM];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int local_group = warp / kSplitK;
    const int split = warp - local_group * kSplitK;
    const int group = blockIdx.x * kGroupsPerCta + local_group;

    if (group < kNumGroups) {
        const int row0 = group * kRowsPerWarp;
        const float* row_scales = scales + static_cast<size_t>(row0 >> 7) * kScaleBlocksK;
        float accumulators[kRowsPerWarp][m4ctl::kM] = {};

#pragma unroll
        for (int tile_in_split = 0; tile_in_split < kTilesPerSplit; ++tile_in_split) {
            const int tile = split * kTilesPerSplit + tile_in_split;
            const int k = tile * 512 + lane * 16;
            const float scale = row_scales[tile * 4 + (lane >> 3)];

            float x_values[m4ctl::kM][16];
#pragma unroll
            for (int draft_row = 0; draft_row < m4ctl::kM; ++draft_row) {
                const __nv_bfloat16* x = x4 + static_cast<size_t>(draft_row) * kHid + k;
                const int4 x0 = *reinterpret_cast<const int4*>(x);
                const int4 x1 = *reinterpret_cast<const int4*>(x + 8);
                const __nv_bfloat162* pair0 = reinterpret_cast<const __nv_bfloat162*>(&x0);
                const __nv_bfloat162* pair1 = reinterpret_cast<const __nv_bfloat162*>(&x1);
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    const float2 value = __bfloat1622float2(pair0[pair]);
                    x_values[draft_row][2 * pair] = value.x;
                    x_values[draft_row][2 * pair + 1] = value.y;
                }
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    const float2 value = __bfloat1622float2(pair1[pair]);
                    x_values[draft_row][8 + 2 * pair] = value.x;
                    x_values[draft_row][8 + 2 * pair + 1] = value.y;
                }
            }

#pragma unroll
            for (int row_in_group = 0; row_in_group < kRowsPerWarp; ++row_in_group) {
                const int4 packed_weight = *reinterpret_cast<const int4*>(
                    weights + static_cast<size_t>(row0 + row_in_group) * kHid + k);
                const __nv_fp8x4_e4m3* weight4 =
                    reinterpret_cast<const __nv_fp8x4_e4m3*>(&packed_weight);
                const float4 w0 = float4(weight4[0]);
                const float4 w1 = float4(weight4[1]);
                const float4 w2 = float4(weight4[2]);
                const float4 w3 = float4(weight4[3]);
#pragma unroll
                for (int draft_row = 0; draft_row < m4ctl::kM; ++draft_row) {
                    const float* x = x_values[draft_row];
                    const float part = w0.x * x[0] + w0.y * x[1] + w0.z * x[2] + w0.w * x[3] +
                                       w1.x * x[4] + w1.y * x[5] + w1.z * x[6] + w1.w * x[7] +
                                       w2.x * x[8] + w2.y * x[9] + w2.z * x[10] + w2.w * x[11] +
                                       w3.x * x[12] + w3.y * x[13] + w3.z * x[14] + w3.w * x[15];
                    accumulators[row_in_group][draft_row] += scale * part;
                }
            }
        }

#pragma unroll
        for (int row_in_group = 0; row_in_group < kRowsPerWarp; ++row_in_group) {
#pragma unroll
            for (int draft_row = 0; draft_row < m4ctl::kM; ++draft_row) {
                const float value = warp_sum(accumulators[row_in_group][draft_row]);
                if (lane == 0) {
                    partial[local_group][row_in_group][split][draft_row] = value;
                }
            }
        }
    }
    __syncthreads();

    if (group < kNumGroups && split == 0 && lane == 0) {
        const int row0 = group * kRowsPerWarp;
#pragma unroll
        for (int row_in_group = 0; row_in_group < kRowsPerWarp; ++row_in_group) {
#pragma unroll
            for (int draft_row = 0; draft_row < m4ctl::kM; ++draft_row) {
                float value = 0.0f;
#pragma unroll
                for (int reduction_split = 0; reduction_split < kSplitK; ++reduction_split) {
                    value += partial[local_group][row_in_group][reduction_split][draft_row];
                }
                scatter_projected_row(row0 + row_in_group, draft_row, value, q_group4, gate4,
                                      k_cache, v_cache, slots4);
            }
        }
    }
}

// Four-row T=32 transcription of NormRopeFusedCpKernel.  A single 72-CTA wave
// replaces four sequential 18-CTA waves while retaining each row/head's
// original reduction tree and rounding points.
__global__ __launch_bounds__(32) void M4CpNormRope4Kernel(
    const __nv_bfloat16* __restrict__ q_raw4, __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ qn, const __nv_bfloat16* __restrict__ kn,
    const float* __restrict__ cos_table, const float* __restrict__ sin_table,
    __nv_bfloat16* __restrict__ q_out4, const __nv_bfloat16* __restrict__ x4,
    __nv_bfloat16* __restrict__ v_cache, __nv_bfloat16* __restrict__ gate4,
    const int* __restrict__ slots4, int hid, float epsilon) {
    constexpr int kStagesPerRow = kCpHeads + 2;
    constexpr int kValuesPerThread = kHeadDim / 32;
    const int draft_row = blockIdx.x / kStagesPerRow;
    const int stage = blockIdx.x - draft_row * kStagesPerRow;
    const int lane = threadIdx.x;
    const int position = slots4[2 * draft_row + 1];

    if (stage < kCpHeads + 1) {
        __shared__ __align__(16) __nv_bfloat16 x_shared[kHeadDim];
        __shared__ __align__(16) __nv_bfloat16 w_shared[kHeadDim];
        const __nv_bfloat16* source;
        const __nv_bfloat16* norm_weight;
        __nv_bfloat16* destination;
        if (stage < kCpHeads) {
            source = q_raw4 + static_cast<size_t>(draft_row) * kCpQRows + stage * kHeadDim;
            norm_weight = qn;
            destination = q_out4 + static_cast<size_t>(draft_row) * kCpQRows + stage * kHeadDim;
        } else {
            __nv_bfloat16* key = k_cache + static_cast<size_t>(position) * kHeadDim;
            source = key;
            norm_weight = kn;
            destination = key;
        }

        const int dimension0 = lane * kValuesPerThread;
        const int4 packed_x = *reinterpret_cast<const int4*>(source + dimension0);
        const int4 packed_w = *reinterpret_cast<const int4*>(norm_weight + dimension0);
        *reinterpret_cast<int4*>(x_shared + dimension0) = packed_x;
        *reinterpret_cast<int4*>(w_shared + dimension0) = packed_w;
        const __nv_bfloat162* x_pairs = reinterpret_cast<const __nv_bfloat162*>(&packed_x);
        const __nv_bfloat162* w_pairs = reinterpret_cast<const __nv_bfloat162*>(&packed_w);
        float x_values[kValuesPerThread];
        float w_values[kValuesPerThread];
#pragma unroll
        for (int pair = 0; pair < kValuesPerThread / 2; ++pair) {
            const float2 x_value = __bfloat1622float2(x_pairs[pair]);
            const float2 w_value = __bfloat1622float2(w_pairs[pair]);
            x_values[2 * pair] = x_value.x;
            x_values[2 * pair + 1] = x_value.y;
            w_values[2 * pair] = w_value.x;
            w_values[2 * pair + 1] = w_value.y;
        }

        float sum_squares = 0.0f;
#pragma unroll
        for (int element = 0; element < kValuesPerThread; ++element) {
            sum_squares += x_values[element] * x_values[element];
        }
        sum_squares = butterfly_sum(sum_squares);
        __syncthreads();
        const float rms = rsqrtf(sum_squares * (1.0f / kHeadDim) + 1.0e-6f);
        const float* cosine = cos_table + static_cast<size_t>(position) * 32;
        const float* sine = sin_table + static_cast<size_t>(position) * 32;
#pragma unroll
        for (int element = 0; element < kValuesPerThread; ++element) {
            const int dimension = dimension0 + element;
            const float normalized = x_values[element] * rms * (1.0f + w_values[element]);
            if (dimension < 64) {
                const int partner = dimension ^ 32;
                const float normalized_partner = __bfloat162float(x_shared[partner]) * rms *
                                                 (1.0f + __bfloat162float(w_shared[partner]));
                const int angle = dimension & 31;
                const float c = cosine[angle];
                const float s = sine[angle];
                destination[dimension] =
                    __float2bfloat16(dimension < 32 ? normalized * c - normalized_partner * s
                                                    : normalized * c + normalized_partner * s);
            } else {
                destination[dimension] = __float2bfloat16(normalized);
            }
        }
        return;
    }

    float sum_squares = 0.0f;
    const __nv_bfloat16* x = x4 + static_cast<size_t>(draft_row) * kHid;
    for (int index = lane; index < hid; index += 32) {
        const float value = __bfloat162float(x[index]);
        sum_squares += value * value;
    }
    sum_squares = butterfly_sum(sum_squares);
    const float input_rms = rsqrtf(sum_squares * (1.0f / hid) + epsilon);
    __nv_bfloat16* value = v_cache + static_cast<size_t>(position) * kHeadDim;
    for (int index = lane; index < kHeadDim; index += 32) {
        value[index] = __float2bfloat16(__bfloat162float(value[index]) * input_rms);
    }
    __nv_bfloat16* gate = gate4 + static_cast<size_t>(draft_row) * kLocalRows;
    for (int index = lane; index < kLocalRows; index += 32) {
        gate[index] = __float2bfloat16(__bfloat162float(gate[index]) * input_rms);
    }
}

extern "C" {

int NexN25ProM4CpStageSlotsLaunch(const void* control, void* slots, void* stream) {
    StageCpSlotsKernel<<<1, kControlThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const int*>(control), static_cast<int*>(slots));
    return static_cast<int>(cudaGetLastError());
}

int NexN25ProM4CpQGate4Launch(const void* x4, const void* weights, const void* scales,
                              void* q_group4, void* gate4, void* k_cache, void* v_cache,
                              const void* slots4, void* stream) {
    constexpr int kNumGroups = kProjectedRows / kRowsPerWarp;
    constexpr int kNumCtas = (kNumGroups + kGroupsPerCta - 1) / kGroupsPerCta;
    M4CpQGate4Kernel<<<kNumCtas, kProjectionThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(x4), static_cast<const __nv_fp8_e4m3*>(weights),
        static_cast<const float*>(scales), static_cast<__nv_bfloat16*>(q_group4),
        static_cast<__nv_bfloat16*>(gate4), static_cast<__nv_bfloat16*>(k_cache),
        static_cast<__nv_bfloat16*>(v_cache), static_cast<const int*>(slots4));
    return static_cast<int>(cudaGetLastError());
}

int NexN25ProM4CpNormRope4Launch(const void* q_raw4, void* k_cache, const void* qn, const void* kn,
                                 const void* cos_table, const void* sin_table, void* q_out4,
                                 const void* x4, void* v_cache, void* gate4, const void* slots4,
                                 int hid, float epsilon, void* stream) {
    M4CpNormRope4Kernel<<<m4ctl::kM*(kCpHeads + 2), 32, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(q_raw4), static_cast<__nv_bfloat16*>(k_cache),
        static_cast<const __nv_bfloat16*>(qn), static_cast<const __nv_bfloat16*>(kn),
        static_cast<const float*>(cos_table), static_cast<const float*>(sin_table),
        static_cast<__nv_bfloat16*>(q_out4), static_cast<const __nv_bfloat16*>(x4),
        static_cast<__nv_bfloat16*>(v_cache), static_cast<__nv_bfloat16*>(gate4),
        static_cast<const int*>(slots4), hid, epsilon);
    return static_cast<int>(cudaGetLastError());
}

int NexN25ProM4CpFoldStatusLaunch(const void* status_pointers, int count, void* destination,
                                  int rank, int injected_rank, void* stream) {
    FoldCpStatusKernel<<<1, kControlThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const unsigned long long*>(status_pointers), count,
        static_cast<int*>(destination), rank, injected_rank);
    return static_cast<int>(cudaGetLastError());
}

}  // extern "C"

}  // namespace nexrt::sm90::nex_n2_5_pro::M4