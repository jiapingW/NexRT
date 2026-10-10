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

// Shared SM90 primitives for the production CP WGMMA megakernel.
#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY bs1
#endif

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY {
using bf16 = __nv_bfloat16;

constexpr int kHeadDim = 256;
constexpr int kHeads = 16;
constexpr int kBlockN = 64;
constexpr int kThreads = 128;
constexpr int kWarps = 4;
constexpr int kCp = 4;
constexpr int kLocalHeads = 4;
constexpr int kSummaryDChunks = 4;
constexpr int kSummaryDimsPerWorker = kHeadDim / kSummaryDChunks;
constexpr float kScaleLog2 = 0.0625f * 1.4426950408889634f;
constexpr int kSegmentedPublishComputeOnly = 0;
constexpr int kSegmentedPublishDataOnly = 1;
constexpr int kSegmentedPublishDataReady = 2;

__device__ __forceinline__ uint64_t cvta(const void* ptr) {
    return static_cast<uint64_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint32_t cvta32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint64_t make_desc(const void* smem, uint32_t leading_byte_offset,
                                              uint32_t stride_byte_offset, uint32_t layout_type) {
    const uint64_t addr = (cvta(smem) >> 4) & 0x3fffull;
    return addr | (static_cast<uint64_t>(leading_byte_offset & 0x3fff) << 16) |
           (static_cast<uint64_t>(stride_byte_offset & 0x3fff) << 32) |
           (static_cast<uint64_t>(layout_type & 0x3) << 62);
}

__device__ __forceinline__ void wg_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wg_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void wg_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(cvta32(bar)), "r"(count));
}

__device__ __forceinline__ void mbar_expect(uint64_t* bar, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(cvta32(bar)),
                 "r"(bytes));
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, int phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT%=;\n"
        "}\n" ::"r"(cvta32(bar)),
        "r"(phase)
        : "memory");
}

__device__ __forceinline__ void prefetch_tma(const CUtensorMap* map) {
    asm volatile("prefetch.tensormap [%0];\n" ::"l"(reinterpret_cast<uint64_t>(map)));
}

__device__ __forceinline__ void tma_load_3d(const CUtensorMap* map, uint64_t* bar, void* dst,
                                            int c0, int c1, int c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(cvta32(dst)),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(cvta32(bar))
        : "memory");
}

__device__ __forceinline__ void tma_load_4d(const CUtensorMap* map, uint64_t* bar, void* dst,
                                            int c0, int c1, int c2, int c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3, %4, %5}], [%6];\n" ::"r"(cvta32(dst)),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(cvta32(bar))
        : "memory");
}

__device__ __forceinline__ void stmatrix_x4(void* smem_ptr, uint32_t v0, uint32_t v1, uint32_t v2,
                                            uint32_t v3) {
    asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                     cvta32(smem_ptr)),
                 "r"(v0), "r"(v1), "r"(v2), "r"(v3));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ void st_release_sys_i32(int* ptr, int val) {
    asm volatile("st.release.sys.global.s32 [%0], %1;\n" ::"l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ void multimem_red_add_rel_sys(unsigned long long mc, int val) {
    asm volatile("multimem.red.release.sys.global.add.s32 [%0], %1;\n" ::"l"(mc), "r"(val)
                 : "memory");
}

__device__ __forceinline__ void st_release_gpu_i32(int* ptr, int val) {
    asm volatile("st.release.gpu.global.s32 [%0], %1;\n" ::"l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ unsigned long long read_globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
    return t;
}

__device__ __forceinline__ int ld_acquire_sys_i32(const int* ptr) {
    int val;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];\n" : "=r"(val) : "l"(ptr) : "memory");
    return val;
}

__device__ __forceinline__ int ld_acquire_gpu_i32(const int* ptr) {
    int val;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(val) : "l"(ptr) : "memory");
    return val;
}

__device__ __forceinline__ float warp_all_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    }
    return v;
}

__device__ __forceinline__ float warp_all_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

__device__ __forceinline__ float warpgroup_tg_max(float v) {
#pragma unroll
    for (int off = 4; off <= 16; off <<= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    }
    return v;
}

__device__ __forceinline__ float warpgroup_tg_sum(float v) {
#pragma unroll
    for (int off = 4; off <= 16; off <<= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

__device__ __forceinline__ void reduce4_max(const float in[4], float out[4], float* scratch) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int gid = lane >> 2;
    const int tg = lane & 3;
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        const float w = warpgroup_tg_max(in[h]);
        if (gid == 0) {
            scratch[h * kWarps * 4 + warp * 4 + tg] = w;
        }
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        float v = -3.402823466e38f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            v = fmaxf(v, scratch[h * kWarps * 4 + w * 4 + tg]);
        }
        out[h] = v;
    }
    __syncthreads();
}

__device__ __forceinline__ void reduce4_sum(const float in[4], float out[4], float* scratch) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int gid = lane >> 2;
    const int tg = lane & 3;
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        const float w = warpgroup_tg_sum(in[h]);
        if (gid == 0) {
            scratch[h * kWarps * 4 + warp * 4 + tg] = w;
        }
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            v += scratch[h * kWarps * 4 + w * 4 + tg];
        }
        out[h] = v;
    }
    __syncthreads();
}

__device__ __forceinline__ void wgmma_m64n16_nn(float (&d)[8], uint64_t desc_a, uint64_t desc_b) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, p, 1, 1, %11, %12;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
          "+f"(d[7])
        : "l"(desc_a), "l"(desc_b), "r"(1), "n"(0), "n"(0));
}

__device__ __forceinline__ void wgmma_m64n16_tn(float* d, uint64_t desc_a, uint64_t desc_b) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, p, 1, 1, 1, 1;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
          "+f"(d[7])
        : "l"(desc_a), "l"(desc_b), "r"(1));
}

static CUresult make_q_map(void* q, CUtensorMap* map) {
    uint64_t global_dim[3] = {256, 16, 1};
    uint64_t global_stride[2] = {512, 8192};
    uint32_t box_dim[3] = {64, 16, 1};
    uint32_t elem_stride[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, q, global_dim, global_stride, box_dim,
        elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

static CUresult make_kv_map(void* kv, int seqlen_kv, CUtensorMap* map) {
    uint64_t global_dim[4] = {256, 1, static_cast<uint64_t>(seqlen_kv), 1};
    uint64_t global_stride[3] = {512, 512, static_cast<uint64_t>(seqlen_kv) * 512};
    uint32_t box_dim[4] = {64, 1, 64, 1};
    uint32_t elem_stride[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, kv, global_dim, global_stride, box_dim,
        elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

}  // namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
using namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY;