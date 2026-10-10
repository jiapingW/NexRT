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

// DFlash M=8 output projection, tensor-core producer plus custom-LL tail.
//
//   ctx8 [8,512] BF16 x packed wo [4096,512] BF16
//       -> rank partial [8,4096] FP32
//       -> BF16x2 self-tagged LL packets, fixed ascending-rank fold,
//          residual update h = bf16(h + reduced).
//
// The producer uses the DFlash MLP-down transposed-MMA structure
// specialized from K=1536 to K=512: 128 CTAs, two m16 output tiles and two
// K splits per CTA (four warps). Static weights are staged before the
// kernel's entry synchronization; input and output-dependent work stay after it.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY DF
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "../common/pdl.cuh"
#include "ll_wire.cuh"

namespace nexrt::sm90::nex_n2_5_pro::DFlash {
constexpr int kOpTok = 8;
constexpr int kOpHid = 4096;
constexpr int kOpK = 512;
constexpr int kOpTileRows = 16;
constexpr int kOpStepsPerChunk = 4;
constexpr int kOpChunkElems = kOpStepsPerChunk * 32 * 8;
constexpr int kOpChunkBytes = kOpChunkElems * static_cast<int>(sizeof(__nv_bfloat16));

__device__ __forceinline__ uint32_t op_smem_u32(const void* pointer) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void op_cp16(uint32_t destination, const void* source) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(destination), "l"(source));
}

__device__ __forceinline__ void op_mma16816(float (&accumulator)[4], const uint4& weight,
                                            const uint32_t (&input)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]), "+f"(accumulator[3])
        : "r"(weight.x), "r"(weight.y), "r"(weight.z), "r"(weight.w), "r"(input[0]), "r"(input[1]));
}

template <int kDepth, int kWarps>
__device__ __forceinline__ void op_ring_prologue(const __nv_bfloat16* __restrict__ source,
                                                 unsigned char* __restrict__ ring, int warp,
                                                 int lane) {
#pragma unroll
    for (int chunk = 0; chunk < kDepth - 1; ++chunk) {
        const __nv_bfloat16* chunk_source =
            source + (static_cast<size_t>(chunk) * kWarps + warp) * kOpChunkElems;
        unsigned char* destination = ring + chunk * kOpChunkBytes;
#pragma unroll
        for (int step = 0; step < kOpStepsPerChunk; ++step) {
            op_cp16(op_smem_u32(destination + (step * 32 + lane) * 16),
                    chunk_source + (step * 32 + lane) * 8);
        }
        asm volatile("cp.async.commit_group;");
    }
}

template <int kThreads, int kPitch>
__device__ __forceinline__ void op_stage_input(const __nv_bfloat16* __restrict__ source,
                                               __nv_bfloat16* __restrict__ staged) {
    for (int item = threadIdx.x; item < kOpTok * (kOpK / 8); item += kThreads) {
        const int row = item / (kOpK / 8);
        const int k8 = item % (kOpK / 8);
        op_cp16(op_smem_u32(staged + row * kPitch + k8 * 8), source + row * kOpK + k8 * 8);
    }
    asm volatile("cp.async.commit_group;");
}

template <int kDepth, int kWarps, int kChunks, int kPitch>
__device__ __forceinline__ void op_mma_stream(const __nv_bfloat16* __restrict__ source,
                                              unsigned char* __restrict__ ring,
                                              const __nv_bfloat16* __restrict__ staged, int warp,
                                              int lane, int step_base, float (&accumulator)[4]) {
    const int group = lane >> 2;
    const int col4 = lane & 3;
    auto issue = [&](int chunk, int slot) {
        const __nv_bfloat16* chunk_source =
            source + (static_cast<size_t>(chunk) * kWarps + warp) * kOpChunkElems;
        unsigned char* destination = ring + slot * kOpChunkBytes;
#pragma unroll
        for (int step = 0; step < kOpStepsPerChunk; ++step) {
            op_cp16(op_smem_u32(destination + (step * 32 + lane) * 16),
                    chunk_source + (step * 32 + lane) * 8);
        }
    };
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    int read_slot = 0;
    int issue_slot = kDepth - 1;
#pragma unroll
    for (int chunk = 0; chunk < kChunks; ++chunk) {
        asm volatile("cp.async.wait_group %0;" ::"n"(kDepth - 2));
        const unsigned char* buffer = ring + read_slot * kOpChunkBytes;
        uint4 weight[kOpStepsPerChunk];
#pragma unroll
        for (int step = 0; step < kOpStepsPerChunk; ++step) {
            weight[step] = *reinterpret_cast<const uint4*>(buffer + (step * 32 + lane) * 16);
        }
        if (chunk + kDepth - 1 < kChunks) issue(chunk + kDepth - 1, issue_slot);
        asm volatile("cp.async.commit_group;");
        read_slot = (read_slot + 1 == kDepth) ? 0 : read_slot + 1;
        issue_slot = (issue_slot + 1 == kDepth) ? 0 : issue_slot + 1;
        const int first_step = step_base + chunk * kOpStepsPerChunk;
#pragma unroll
        for (int step = 0; step < kOpStepsPerChunk; ++step) {
            const __nv_bfloat16* row =
                staged + group * kPitch + (first_step + step) * 16 + col4 * 2;
            uint32_t input[2];
            input[0] = *reinterpret_cast<const uint32_t*>(row);
            input[1] = *reinterpret_cast<const uint32_t*>(row + 8);
            op_mma16816(accumulator, weight[step], input);
        }
    }
}

__device__ __forceinline__ void op_put_accumulator(float* destination, int lane,
                                                   const float (&accumulator)[4]) {
    const int group = lane >> 2;
    const int col4 = lane & 3;
    destination[group * kOpTok + col4 * 2] = accumulator[0];
    destination[group * kOpTok + col4 * 2 + 1] = accumulator[1];
    destination[(group + 8) * kOpTok + col4 * 2] = accumulator[2];
    destination[(group + 8) * kOpTok + col4 * 2 + 1] = accumulator[3];
}

template <int kDepth>
struct OProjCfg {
    static constexpr int kSplit = 2;
    static constexpr int kTiles = 2;
    static constexpr int kWarps = kSplit * kTiles;
    static constexpr int kThreads = kWarps * 32;
    static constexpr int kKSteps = kOpK / 16;
    static constexpr int kSplitSteps = kKSteps / kSplit;
    static constexpr int kChunks = kSplitSteps / kOpStepsPerChunk;
    static constexpr int kCtas = kOpHid / kOpTileRows / kTiles;
    static constexpr int kPitch = kOpK + 8;
    static constexpr int kInputBytes = kOpTok * kPitch * sizeof(__nv_bfloat16);
    static constexpr int kRingBytes = kWarps * kDepth * kOpChunkBytes;
    static constexpr int kReductionBytes = kWarps * kOpTileRows * kOpTok * sizeof(float);
    static constexpr int kSmem = kInputBytes + kRingBytes + kReductionBytes;
    static_assert(kChunks == 4, "K=512, split-2 producer must have four chunks");
};

template <int kDepth>
__global__ __launch_bounds__(OProjCfg<kDepth>::kThreads) void DfOProjTcGemmKernel(
    const __nv_bfloat16* __restrict__ ctx8, const __nv_bfloat16* __restrict__ weight_pack,
    float* __restrict__ partial) {
    using C = OProjCfg<kDepth>;
    extern __shared__ __align__(16) unsigned char shared[];
    __nv_bfloat16* staged = reinterpret_cast<__nv_bfloat16*>(shared);
    unsigned char* ring = shared + C::kInputBytes;
    float* reduction = reinterpret_cast<float*>(shared + C::kInputBytes + C::kRingBytes);
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int ksplit = warp % C::kSplit;
    const __nv_bfloat16* source =
        weight_pack + static_cast<size_t>(blockIdx.x) * C::kChunks * C::kWarps * kOpChunkElems;
    unsigned char* warp_ring = ring + warp * kDepth * kOpChunkBytes;

    op_ring_prologue<kDepth, C::kWarps>(source, warp_ring, warp, lane);
    NEXRT_PDL_ENTRY_SYNC();
    op_stage_input<C::kThreads, C::kPitch>(ctx8, staged);
    float accumulator[4] = {0.f, 0.f, 0.f, 0.f};
    op_mma_stream<kDepth, C::kWarps, C::kChunks, C::kPitch>(source, warp_ring, staged, warp, lane,
                                                            ksplit * C::kSplitSteps, accumulator);
    op_put_accumulator(reduction + warp * (kOpTileRows * kOpTok), lane, accumulator);
    __syncthreads();
    if (threadIdx.x < kOpTileRows * kOpTok) {
        const int output_row = threadIdx.x >> 3;
        const int token = threadIdx.x & 7;
#pragma unroll
        for (int tile = 0; tile < C::kTiles; ++tile) {
            float sum = 0.f;
#pragma unroll
            for (int split = 0; split < C::kSplit; ++split) {
                sum += reduction[(tile * C::kSplit + split) * (kOpTileRows * kOpTok) +
                                 output_row * kOpTok + token];
            }
            const int column = (blockIdx.x * C::kTiles + tile) * kOpTileRows + output_row;
            partial[token * kOpHid + column] = sum;
        }
    }
}

constexpr int kOpReduceCtas = 128;
constexpr int kOpReduceThreads = 128;
constexpr int kOpPacketsPerSource = kOpTok * kOpHid / 2;
static_assert(kOpReduceCtas * kOpReduceThreads == kOpPacketsPerSource,
              "one reducer thread must own one BF16x2 packet");
static_assert(2L * kLlMaxRanks * kOpPacketsPerSource <= kLlWireFloats,
              "DFlash LL allocation must hold the O-projection packet layout");

__global__ __launch_bounds__(kOpReduceThreads, 1) void DfOProjTcReduceKernel(
    const float* __restrict__ partial, unsigned long long inbox_mc,
    const uint64_t* __restrict__ inbox_local, unsigned int* __restrict__ ll_counter,
    int* __restrict__ status, __nv_bfloat16* __restrict__ hidden, int rank, int num_ranks) {
    __shared__ unsigned generation_shared;
    if (threadIdx.x == 0) {
        unsigned generation = ll_counter[blockIdx.x] + 1u;
        if (generation == 0u) generation = 2u;
        ll_counter[blockIdx.x] = generation;
        generation_shared = generation;
    }
    const int packet = blockIdx.x * kOpReduceThreads + threadIdx.x;
    __nv_bfloat162* destination = reinterpret_cast<__nv_bfloat162*>(hidden) + packet;
    const __nv_bfloat162 residual = *destination;
    NEXRT_PDL_ENTRY_SYNC();
    __syncthreads();

    const unsigned generation = generation_shared;
    const unsigned parity = (generation - 1u) & 1u;
    const float2 local_fp32 = *reinterpret_cast<const float2*>(partial + 2 * packet);
    const __nv_bfloat162 local = __float22bfloat162_rn(local_fp32);
    const unsigned payload = *reinterpret_cast<const unsigned*>(&local);
    const uint64_t self_packet = (static_cast<uint64_t>(generation) << 32) | payload;
    if (num_ranks > 1) {
        df_mm_st_rel_sys_b64(
            inbox_mc + ((static_cast<uint64_t>(parity) * num_ranks + rank) * kOpPacketsPerSource +
                        packet) *
                           sizeof(uint64_t),
            self_packet);
    }
    const uint64_t* my_inbox =
        inbox_local + static_cast<uint64_t>(parity) * num_ranks * kOpPacketsPerSource;
    const unsigned peer_mask = ((1u << num_ranks) - 1u) & ~(1u << rank);
    uint64_t peer_packets[kLlMaxRanks];
#pragma unroll
    for (int source = 0; source < kLlMaxRanks; ++source) peer_packets[source] = 0ull;
    unsigned arrived = 0u;
    int spins = 0;
    while (true) {
        arrived = 0u;
#pragma unroll
        for (int source = 0; source < kLlMaxRanks; ++source) {
            if (source < num_ranks && source != rank) {
                peer_packets[source] = df_ld_rel_sys_b64(
                    my_inbox + static_cast<uint64_t>(source) * kOpPacketsPerSource + packet);
                arrived |= (static_cast<unsigned>(peer_packets[source] >> 32) == generation)
                               ? (1u << source)
                               : 0u;
            }
        }
        if (arrived == peer_mask) break;
        if (++spins > kLlMaxSpins) {
            atomicOr(status, 16);
            break;
        }
        __nanosleep(64);
    }
    float sum0 = 0.f;
    float sum1 = 0.f;
#pragma unroll
    for (int source = 0; source < kLlMaxRanks; ++source) {
        if (source >= num_ranks) break;
        if (source == rank) {
            sum0 += __bfloat162float(local.x);
            sum1 += __bfloat162float(local.y);
        } else if ((arrived >> source) & 1u) {
            const unsigned word = static_cast<unsigned>(peer_packets[source]);
            const __nv_bfloat162 value = *reinterpret_cast<const __nv_bfloat162*>(&word);
            sum0 += __bfloat162float(value.x);
            sum1 += __bfloat162float(value.y);
        }
    }
    *destination = __nv_bfloat162(__float2bfloat16(__bfloat162float(residual.x) + sum0),
                                  __float2bfloat16(__bfloat162float(residual.y) + sum1));
}

template <int kDepth>
int launch_oproj_gemm(const void* context, const void* weight_pack, void* partial, void* stream) {
    using C = OProjCfg<kDepth>;
    auto* kernel = &DfOProjTcGemmKernel<kDepth>;
    static bool initialized = false;
    if (!initialized) {
        const cudaError_t result =
            cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmem);
        if (result != cudaSuccess) return static_cast<int>(result);
        initialized = true;
    }
    const auto* input = static_cast<const __nv_bfloat16*>(context);
    const auto* weights = static_cast<const __nv_bfloat16*>(weight_pack);
    auto* output = static_cast<float*>(partial);
    kernel<<<dim3(C::kCtas), dim3(C::kThreads), C::kSmem, static_cast<cudaStream_t>(stream)>>>(
        input, weights, output);
    return static_cast<int>(cudaGetLastError());
}
}  // namespace nexrt::sm90::nex_n2_5_pro::DFlash

using namespace nexrt::sm90::nex_n2_5_pro::DFlash;

extern "C" {

int NexN25ProDFlashOProjTcGemmLaunch(const void* context, const void* weight_pack, void* partial,
                                     int depth, int pdl, void* stream) {
    if (depth == 5 && !pdl) {
        return launch_oproj_gemm<5>(context, weight_pack, partial, stream);
    }
    return static_cast<int>(cudaErrorInvalidValue);
}

int NexN25ProDFlashOProjTcReduceLaunch(const void* partial, unsigned long long inbox_mc,
                                       const void* inbox_local, void* ll_counter, void* status,
                                       void* hidden, int rank, int num_ranks, int pdl,
                                       void* stream) {
    if (num_ranks < 1 || num_ranks > kLlMaxRanks || rank < 0 || rank >= num_ranks ||
        status == nullptr || (num_ranks > 1 && inbox_mc == 0ull)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const auto* input = static_cast<const float*>(partial);
    const auto* local = static_cast<const uint64_t*>(inbox_local);
    auto* counter = static_cast<unsigned int*>(ll_counter);
    auto* status_word = static_cast<int*>(status);
    auto* output = static_cast<__nv_bfloat16*>(hidden);
    if (pdl) return static_cast<int>(cudaErrorInvalidValue);
    DfOProjTcReduceKernel<<<dim3(kOpReduceCtas), dim3(kOpReduceThreads), 0,
                            static_cast<cudaStream_t>(stream)>>>(
        input, inbox_mc, local, counter, status_word, output, rank, num_ranks);
    return static_cast<int>(cudaGetLastError());
}

int NexN25ProDFlashOProjTcK() { return kOpK; }

}  // extern "C"
