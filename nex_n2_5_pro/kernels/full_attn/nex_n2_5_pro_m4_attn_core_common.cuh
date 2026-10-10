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

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M4
#endif

#pragma once

// Shared primitives for the production M=4 quad attention core.  These are
// direct counterparts of the BS=1 WGMMA helpers and combine reduction; this
// header intentionally contains no launchable alternative kernel family.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#ifndef WGMMA_STAGES
#define WGMMA_STAGES 2
#endif
#ifndef WGMMA_KV_EVICT_FIRST
#define WGMMA_KV_EVICT_FIRST 1
#endif
#ifndef WGMMA_COMBINE_THREADS
#define WGMMA_COMBINE_THREADS 256
#endif

namespace nexrt::sm90::nex_n2_5_pro::M4 {
using bf16 = __nv_bfloat16;
constexpr int HD = 256;
constexpr int NH = 4;
constexpr int TILE = 64;
constexpr int MROWS = 4;
constexpr int KV_CAPACITY = 256 * 1024 + 1024;
constexpr uint64_t TMA_EVICT_FIRST = 0x12f0000000000000ULL;

__device__ __forceinline__ uint64_t cvta(const void* pointer) {
    return static_cast<uint64_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ uint64_t make_desc(const void* pointer, uint32_t leading_byte_offset,
                                              uint32_t stride_byte_offset, uint32_t swizzle) {
    const uint64_t address = (cvta(pointer) >> 4) & 0x3fffull;
    return address | (static_cast<uint64_t>(leading_byte_offset & 0x3fff) << 16) |
           (static_cast<uint64_t>(stride_byte_offset & 0x3fff) << 32) |
           (static_cast<uint64_t>(swizzle & 3) << 62);
}

__device__ __forceinline__ void wg_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int kPendingGroups>
__device__ __forceinline__ void wg_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(kPendingGroups) : "memory");
}

__device__ __forceinline__ void mbar_init(uint64_t* barrier, int count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(static_cast<uint32_t>(cvta(barrier))),
                 "r"(count));
}

__device__ __forceinline__ void mbar_expect(uint64_t* barrier, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(
                     static_cast<uint32_t>(cvta(barrier))),
                 "r"(bytes));
}

__device__ __forceinline__ void mbar_wait(uint64_t* barrier, int phase) {
    asm volatile(
        "{\n.reg .pred p;\nL%=:\n"
        "mbarrier.try_wait.parity.shared.b64 p,[%0],%1;\n"
        "@!p bra L%=;\n}\n" ::"r"(static_cast<uint32_t>(cvta(barrier))),
        "r"(phase));
}

__device__ __forceinline__ void tma_load(const CUtensorMap* map, uint64_t* barrier,
                                         void* destination, int coordinate0, int coordinate1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(
            static_cast<uint32_t>(cvta(destination))),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(coordinate0), "r"(coordinate1),
        "r"(static_cast<uint32_t>(cvta(barrier)))
        : "memory");
}

__device__ __forceinline__ void tma_load_kv(const CUtensorMap* map, uint64_t* barrier,
                                            void* destination, int coordinate0, int coordinate1) {
#if WGMMA_KV_EVICT_FIRST
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1, {%2, %3}], [%4], %5;\n" ::"r"(static_cast<uint32_t>(cvta(destination))),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(coordinate0), "r"(coordinate1),
        "r"(static_cast<uint32_t>(cvta(barrier))), "l"(TMA_EVICT_FIRST)
        : "memory");
#else
    tma_load(map, barrier, destination, coordinate0, coordinate1);
#endif
}

__device__ __forceinline__ void prefetch_map(const CUtensorMap* map) {
    asm volatile("prefetch.tensormap [%0];\n" ::"l"(reinterpret_cast<uint64_t>(map)));
}

__device__ __forceinline__ void partial_sync_128() {
    asm volatile("bar.sync 1, 128;\n" ::: "memory");
}

template <int kElementsPerBlock, int kThreads>
__device__ __forceinline__ void m4_combine_body(const float* __restrict__ workspace_max,
                                                const float* __restrict__ workspace_sum,
                                                const bf16* __restrict__ workspace_accumulator,
                                                bf16* __restrict__ output, int num_splits,
                                                float scaling) {
    static_assert(kElementsPerBlock == 8, "16-byte split gather requires eight elements");
    const int output_start = blockIdx.x * kElementsPerBlock;
    const int head = output_start / HD;
    const int dimension_start = output_start % HD;
    const int thread = threadIdx.x;
    const float scale2 = scaling * 1.4426950408889634f;
    extern __shared__ __align__(16) char combine_shared[];
    bf16* gathered = reinterpret_cast<bf16*>(combine_shared);
    float* split_max = reinterpret_cast<float*>(gathered + num_splits * kElementsPerBlock);
    float* split_sum = split_max + num_splits;
    __shared__ float numerator_reduction[kThreads];
    __shared__ float denominator_reduction[kThreads];

    for (int split = thread; split < num_splits; split += kThreads) {
        *reinterpret_cast<uint4*>(gathered + split * kElementsPerBlock) =
            *reinterpret_cast<const uint4*>(workspace_accumulator +
                                            (static_cast<size_t>(split) * NH + head) * HD +
                                            dimension_start);
    }
    for (int split = thread; split < num_splits; split += kThreads) {
        split_max[split] = workspace_max[split * NH + head];
        split_sum[split] = workspace_sum[split * NH + head];
    }
    __syncthreads();

    float local_max = -1.0e30f;
    for (int split = thread; split < num_splits; split += kThreads) {
        local_max = fmaxf(local_max, split_max[split]);
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, offset));
    }
    if ((thread & 31) == 0) {
        numerator_reduction[thread >> 5] = local_max;
    }
    __syncthreads();
    float global_max = -1.0e30f;
#pragma unroll
    for (int warp = 0; warp < kThreads / 32; ++warp) {
        global_max = fmaxf(global_max, numerator_reduction[warp]);
    }

    constexpr int kGroups = kThreads / kElementsPerBlock;
    const int dimension = thread % kElementsPerBlock;
    const int group = thread / kElementsPerBlock;
    float numerator = 0.0f;
    float denominator = 0.0f;
    for (int split = group; split < num_splits; split += kGroups) {
        const float weight = exp2f((split_max[split] - global_max) * scale2) * split_sum[split];
        numerator += weight * static_cast<float>(gathered[split * kElementsPerBlock + dimension]);
        denominator += weight;
    }
    __syncthreads();
    numerator_reduction[thread] = numerator;
    denominator_reduction[thread] = denominator;
    __syncthreads();
    if (thread < kElementsPerBlock) {
        float total_numerator = 0.0f;
        float total_denominator = 0.0f;
#pragma unroll
        for (int item = 0; item < kGroups; ++item) {
            total_numerator += numerator_reduction[item * kElementsPerBlock + thread];
            total_denominator += denominator_reduction[item * kElementsPerBlock + thread];
        }
        output[output_start + thread] =
            static_cast<bf16>(total_numerator / (total_denominator + 1.0e-20f));
    }
}

template <int kElementsPerBlock, int kThreads>
__device__ __forceinline__ void m4_combine_body_xeff(const float* __restrict__ workspace_max,
                                                     const float* __restrict__ workspace_sum,
                                                     const bf16* __restrict__ workspace_accumulator,
                                                     bf16* __restrict__ output,
                                                     const bf16* __restrict__ gate,
                                                     bf16* __restrict__ effective_output,
                                                     int num_splits, float scaling) {
    static_assert(kElementsPerBlock == 8, "16-byte split gather requires eight elements");
    const int output_start = blockIdx.x * kElementsPerBlock;
    const int head = output_start / HD;
    const int dimension_start = output_start % HD;
    const int thread = threadIdx.x;
    const float scale2 = scaling * 1.4426950408889634f;
    extern __shared__ __align__(16) char combine_shared[];
    bf16* gathered = reinterpret_cast<bf16*>(combine_shared);
    float* split_max = reinterpret_cast<float*>(gathered + num_splits * kElementsPerBlock);
    float* split_sum = split_max + num_splits;
    __shared__ float numerator_reduction[kThreads];
    __shared__ float denominator_reduction[kThreads];

    for (int split = thread; split < num_splits; split += kThreads) {
        *reinterpret_cast<uint4*>(gathered + split * kElementsPerBlock) =
            *reinterpret_cast<const uint4*>(workspace_accumulator +
                                            (static_cast<size_t>(split) * NH + head) * HD +
                                            dimension_start);
    }
    for (int split = thread; split < num_splits; split += kThreads) {
        split_max[split] = workspace_max[split * NH + head];
        split_sum[split] = workspace_sum[split * NH + head];
    }
    __syncthreads();

    float local_max = -1.0e30f;
    for (int split = thread; split < num_splits; split += kThreads) {
        local_max = fmaxf(local_max, split_max[split]);
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, offset));
    }
    if ((thread & 31) == 0) {
        numerator_reduction[thread >> 5] = local_max;
    }
    __syncthreads();
    float global_max = -1.0e30f;
#pragma unroll
    for (int warp = 0; warp < kThreads / 32; ++warp) {
        global_max = fmaxf(global_max, numerator_reduction[warp]);
    }

    constexpr int kGroups = kThreads / kElementsPerBlock;
    const int dimension = thread % kElementsPerBlock;
    const int group = thread / kElementsPerBlock;
    float numerator = 0.0f;
    float denominator = 0.0f;
    for (int split = group; split < num_splits; split += kGroups) {
        const float weight = exp2f((split_max[split] - global_max) * scale2) * split_sum[split];
        numerator += weight * static_cast<float>(gathered[split * kElementsPerBlock + dimension]);
        denominator += weight;
    }
    __syncthreads();
    numerator_reduction[thread] = numerator;
    denominator_reduction[thread] = denominator;
    __syncthreads();
    if (thread < kElementsPerBlock) {
        float total_numerator = 0.0f;
        float total_denominator = 0.0f;
#pragma unroll
        for (int item = 0; item < kGroups; ++item) {
            total_numerator += numerator_reduction[item * kElementsPerBlock + thread];
            total_denominator += denominator_reduction[item * kElementsPerBlock + thread];
        }
        const bf16 context = static_cast<bf16>(total_numerator / (total_denominator + 1.0e-20f));
        output[output_start + thread] = context;
        const float gate_value = __bfloat162float(gate[output_start + thread]);
        effective_output[output_start + thread] =
            __float2bfloat16(__bfloat162float(context) * (1.0f / (1.0f + __expf(-gate_value))));
    }
}

template <int kElementsPerBlock, int kThreads>
__global__ __launch_bounds__(kThreads) void M4AttnCombineKernel(
    const float* __restrict__ workspace_max4, const float* __restrict__ workspace_sum4,
    const bf16* __restrict__ workspace_accumulator4, bf16* __restrict__ context4, int num_splits,
    float scaling) {
    const int row = blockIdx.y;
    m4_combine_body<kElementsPerBlock, kThreads>(
        workspace_max4 + static_cast<size_t>(row) * num_splits * NH,
        workspace_sum4 + static_cast<size_t>(row) * num_splits * NH,
        workspace_accumulator4 + static_cast<size_t>(row) * num_splits * NH * HD,
        context4 + static_cast<size_t>(row) * NH * HD, num_splits, scaling);
}

template <int kElementsPerBlock, int kThreads>
__global__ __launch_bounds__(kThreads) void M4AttnCombineXeffKernel(
    const float* __restrict__ workspace_max4, const float* __restrict__ workspace_sum4,
    const bf16* __restrict__ workspace_accumulator4, bf16* __restrict__ context4,
    const bf16* __restrict__ gate4, bf16* __restrict__ effective_output4, int num_splits,
    float scaling) {
    const int row = blockIdx.y;
    m4_combine_body_xeff<kElementsPerBlock, kThreads>(
        workspace_max4 + static_cast<size_t>(row) * num_splits * NH,
        workspace_sum4 + static_cast<size_t>(row) * num_splits * NH,
        workspace_accumulator4 + static_cast<size_t>(row) * num_splits * NH * HD,
        context4 + static_cast<size_t>(row) * NH * HD, gate4 + static_cast<size_t>(row) * NH * HD,
        effective_output4 + static_cast<size_t>(row) * NH * HD, num_splits, scaling);
}

static inline CUtensorMap encode_map(void* pointer, int outer, int inner, int box_outer,
                                     int box_inner) {
    CUtensorMap map{};
    uint64_t dimensions[2] = {static_cast<uint64_t>(inner), static_cast<uint64_t>(outer)};
    uint64_t strides[1] = {static_cast<uint64_t>(inner) * sizeof(bf16)};
    uint32_t box[2] = {static_cast<uint32_t>(box_inner), static_cast<uint32_t>(box_outer)};
    uint32_t element_strides[2] = {1, 1};
    cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, pointer, dimensions, strides,
                           box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                           CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return map;
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M4