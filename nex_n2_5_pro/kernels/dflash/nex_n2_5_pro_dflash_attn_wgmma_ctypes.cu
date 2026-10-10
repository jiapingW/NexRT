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

// DFlash M=8 non-CP attention, re-derived from the main M=8 TMA/WGMMA core.
//
// Shape per TP rank:
//   Q       [8 draft rows, 4 local heads, 128]
//   K/V     [physical cache rows, 128] (one local KV head)
//   context [8, 4, 128]
//
// Four consumer warpgroups jointly cover all 32 query rows.  Each CTA owns 64
// key rows.  K and V use independent TMA completion rings, QK/PV use Hopper
// WGMMA, and partial online-softmax states are reduced by a separate
// deterministic combine kernel.
// The production kernel exposes K in 64-wide readiness chunks (the main-M8
// production form). One-box histories publish directly and make combine a no-op.
//
// Cache modes:
//   linear   final layer, unmasked over [lo, start + 8)
//   ring8192 first five layers, causal window 4096
//
// Ring safety: logical work starts are rounded down to an absolute multiple of
// 64.  Because 8192 is also a multiple of 64, (absolute & 8191) always names a
// contiguous physical 64-row box.  Masking uses absolute positions after QK;
// RoPE/mask semantics never use the physical index.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY DF
#endif

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "../common/pdl.cuh"

namespace nexrt::sm90::nex_n2_5_pro::DFlash {
using bf16 = __nv_bfloat16;

constexpr int kDfWgDraftRows = 8;
constexpr int kDfWgHeads = 4;
constexpr int kDfWgQueryRows = kDfWgDraftRows * kDfWgHeads;
constexpr int kDfWgHeadDim = 128;
constexpr int kDfWgTile = 64;
constexpr int kDfWgKStages = 2;
constexpr int kDfWgVStages = 2;
constexpr int kDfWgWarpgroups = 4;
constexpr int kDfWgQkChunk = 64;
constexpr int kDfWgCombineDim = 16;
constexpr int kDfWgCombineThreads = 256;
constexpr float kDfWgScale = 0.08838834764831845f;
constexpr uint64_t kDfWgTmaEvictFirst = 0x12f0000000000000ULL;

enum DfWgCacheMode : int {
    kDfWgLinear = 0,
    kDfWgRing8192 = 1,
};

__device__ __forceinline__ uint32_t dfwg_pack_bf16(float first, float second) {
    uint32_t packed;
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(packed) : "f"(second), "f"(first));
    return packed;
}

__device__ __forceinline__ uint64_t dfwg_cvta(const void* pointer) {
    return static_cast<uint64_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ uint64_t dfwg_make_desc(const void* pointer, uint32_t leading,
                                                   uint32_t stride, uint32_t swizzle) {
    const uint64_t address = (dfwg_cvta(pointer) >> 4) & 0x3fffull;
    return address | (static_cast<uint64_t>(leading & 0x3fff) << 16) |
           (static_cast<uint64_t>(stride & 0x3fff) << 32) |
           (static_cast<uint64_t>(swizzle & 3) << 62);
}

__device__ __forceinline__ void dfwg_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int kPending>
__device__ __forceinline__ void dfwg_wait_group() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(kPending) : "memory");
}

__device__ __forceinline__ void dfwg_mbar_init(uint64_t* barrier, int count) {
    asm volatile(
        "mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(static_cast<uint32_t>(dfwg_cvta(barrier))),
        "r"(count));
}

__device__ __forceinline__ void dfwg_mbar_expect(uint64_t* barrier, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(
                     static_cast<uint32_t>(dfwg_cvta(barrier))),
                 "r"(bytes));
}

__device__ __forceinline__ void dfwg_mbar_arrive(uint64_t* barrier) {
    asm volatile(
        "mbarrier.arrive.shared.b64 _, [%0];\n" ::"r"(static_cast<uint32_t>(dfwg_cvta(barrier)))
        : "memory");
}

__device__ __forceinline__ void dfwg_mbar_wait(uint64_t* barrier, int phase) {
    asm volatile(
        "{\n.reg .pred ready;\nL%=:\n"
        "mbarrier.try_wait.parity.shared.b64 ready,[%0],%1;\n"
        "@!ready bra L%=;\n}\n" ::"r"(static_cast<uint32_t>(dfwg_cvta(barrier))),
        "r"(phase));
}

__device__ __forceinline__ void dfwg_tma_load(const CUtensorMap* map, uint64_t* barrier,
                                              void* destination, int coordinate0, int coordinate1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(
            static_cast<uint32_t>(dfwg_cvta(destination))),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(coordinate0), "r"(coordinate1),
        "r"(static_cast<uint32_t>(dfwg_cvta(barrier)))
        : "memory");
}

__device__ __forceinline__ void dfwg_tma_load_kv(const CUtensorMap* map, uint64_t* barrier,
                                                 void* destination, int coordinate0,
                                                 int coordinate1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1, {%2, %3}], [%4], %5;\n" ::"r"(static_cast<uint32_t>(dfwg_cvta(destination))),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(coordinate0), "r"(coordinate1),
        "r"(static_cast<uint32_t>(dfwg_cvta(barrier))), "l"(kDfWgTmaEvictFirst)
        : "memory");
}

__device__ __forceinline__ void dfwg_prefetch_map(const CUtensorMap* map) {
    asm volatile("prefetch.tensormap [%0];\n" ::"l"(reinterpret_cast<uint64_t>(map)));
}

__device__ __forceinline__ void dfwg_wg_sync(int warpgroup) {
    asm volatile("bar.sync %0, 128;\n" ::"r"(warpgroup + 1) : "memory");
}

template <int kCacheMode>
__device__ __forceinline__ int dfwg_physical_tile(int absolute_tile) {
    if constexpr (kCacheMode == kDfWgRing8192) {
        return absolute_tile & 8191;
    }
    return absolute_tile;
}

template <int kCacheMode>
__device__ __forceinline__ bool dfwg_visible(int key_position, int draft_row, int start, int lo,
                                             int window, int union_end) {
    if constexpr (kCacheMode == kDfWgLinear) {
        return key_position >= lo && key_position < union_end;
    }
    const int query_position = start + draft_row;
    const int row_begin = max(lo, query_position - window + 1);
    return key_position >= row_begin && key_position <= query_position;
}

template <int kCacheMode>
__global__ __launch_bounds__(128 * kDfWgWarpgroups) void DfAttnWgmmaPartialKernel(
    const __grid_constant__ CUtensorMap key_map, const __grid_constant__ CUtensorMap value_map,
    const __grid_constant__ CUtensorMap query_map, float* __restrict__ max_workspace,
    float* __restrict__ sum_workspace, bf16* __restrict__ acc_workspace, bf16* __restrict__ context,
    const int* __restrict__ df_meta, const int* __restrict__ status, int lo, int window,
    int workers) {
    NEXRT_PDL_ENTRY_SYNC();
    constexpr int kGroupsPerWarpgroup = 1;
    constexpr int kQkChunks = kDfWgHeadDim / kDfWgQkChunk;
    constexpr int kQkAtoms = kDfWgQkChunk / 16;
    constexpr int kPvChunks = kDfWgHeadDim / 64;
    constexpr int kKvElements = kDfWgTile * kDfWgHeadDim;
    constexpr int kKeyChunkElements = kDfWgTile * kDfWgQkChunk;
    constexpr int kKeyChunkBytes = kKeyChunkElements * static_cast<int>(sizeof(bf16));
    constexpr int kQueryChunkElements = kDfWgQueryRows * kDfWgQkChunk;
    constexpr int kValueChunkElements = kDfWgTile * 64;
    constexpr int kValueChunkBytes = kValueChunkElements * static_cast<int>(sizeof(bf16));

    if (status != nullptr && *status != 0) return;

    const int start = df_meta[1];
    int union_begin = lo;
    if constexpr (kCacheMode == kDfWgRing8192) {
        union_begin = max(union_begin, start - window + 1);
    }
    const int union_end = start + kDfWgDraftRows;
    const int first_tile = (union_begin / kDfWgTile) * kDfWgTile;
    const int tile_count = max(0, (union_end - first_tile + kDfWgTile - 1) / kDfWgTile);

    const int split = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warpgroup = tid >> 7;
    const int local_tid = tid & 127;
    const int warp = local_tid >> 5;
    const int group_lane = lane >> 2;
    const int thread_group = lane & 3;
    const float scale_log2 = kDfWgScale * 1.4426950408889634f;
    const int local_tiles = split < tile_count ? 1 + (tile_count - 1 - split) / workers : 0;

    if (local_tiles == 0) {
        // Combine reads only the contiguous live prefix [0, tile_count), so
        // graph-static overprovisioned CTAs never need identity workspaces.
        return;
    }

    extern __shared__ __align__(1024) char dynamic_smem[];
    bf16* query_smem = reinterpret_cast<bf16*>(dynamic_smem);
    bf16* key_smem = query_smem + kDfWgQueryRows * kDfWgHeadDim;
    bf16* value_smem = key_smem + kDfWgKStages * kKvElements;
    bf16* probability_smem = value_smem + kDfWgVStages * kKvElements;

    __shared__ __align__(8) uint64_t query_full;
    __shared__ __align__(8) uint64_t key_full[kDfWgKStages][kQkChunks];
    __shared__ __align__(8) uint64_t key_empty[kDfWgKStages][kQkChunks];
    __shared__ __align__(8) uint64_t value_full[kDfWgVStages][kPvChunks];
    __shared__ __align__(8) uint64_t value_empty[kDfWgVStages][kPvChunks];
    __shared__ float max_part[4][kDfWgQueryRows];
    __shared__ float sum_part[4][kDfWgQueryRows];

    if (tid == 0) {
        dfwg_mbar_init(&query_full, 1);
#pragma unroll
        for (int stage = 0; stage < kDfWgKStages; ++stage) {
#pragma unroll
            for (int chunk = 0; chunk < kQkChunks; ++chunk) {
                dfwg_mbar_init(&key_full[stage][chunk], 1);
                dfwg_mbar_init(&key_empty[stage][chunk], kDfWgWarpgroups);
            }
        }
#pragma unroll
        for (int stage = 0; stage < kDfWgVStages; ++stage) {
#pragma unroll
            for (int chunk = 0; chunk < kPvChunks; ++chunk) {
                dfwg_mbar_init(&value_full[stage][chunk], 1);
                dfwg_mbar_init(&value_empty[stage][chunk], kDfWgWarpgroups);
            }
        }
        dfwg_prefetch_map(&key_map);
        dfwg_prefetch_map(&value_map);
        dfwg_prefetch_map(&query_map);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    __syncthreads();

    if (tid == 0) {
        dfwg_mbar_expect(&query_full,
                         kDfWgQueryRows * kDfWgHeadDim * static_cast<int>(sizeof(bf16)));
#pragma unroll
        for (int chunk = 0; chunk < kQkChunks; ++chunk) {
            dfwg_tma_load(&query_map, &query_full, query_smem + chunk * kQueryChunkElements,
                          chunk * kDfWgQkChunk, 0);
        }
        for (int tile = 0; tile < min(kDfWgKStages, local_tiles); ++tile) {
            const int stage = tile % kDfWgKStages;
            const int absolute_tile = first_tile + (split + tile * workers) * kDfWgTile;
            const int physical_tile = dfwg_physical_tile<kCacheMode>(absolute_tile);
#pragma unroll
            for (int chunk = 0; chunk < kQkChunks; ++chunk) {
                dfwg_mbar_expect(&key_full[stage][chunk], kKeyChunkBytes);
                dfwg_tma_load_kv(&key_map, &key_full[stage][chunk],
                                 key_smem + stage * kKvElements + chunk * kKeyChunkElements,
                                 chunk * kDfWgQkChunk, physical_tile);
            }
        }
        for (int tile = 0; tile < min(kDfWgVStages, local_tiles); ++tile) {
            const int stage = tile % kDfWgVStages;
            const int absolute_tile = first_tile + (split + tile * workers) * kDfWgTile;
            const int physical_tile = dfwg_physical_tile<kCacheMode>(absolute_tile);
#pragma unroll
            for (int chunk = 0; chunk < kPvChunks; ++chunk) {
                dfwg_mbar_expect(&value_full[stage][chunk], kValueChunkBytes);
                dfwg_tma_load_kv(&value_map, &value_full[stage][chunk],
                                 value_smem + stage * kKvElements + chunk * kValueChunkElements,
                                 chunk * 64, physical_tile);
            }
        }
    }
    dfwg_mbar_wait(&query_full, 0);

    float output_fragment[kGroupsPerWarpgroup][kPvChunks * 4];
#pragma unroll
    for (int group = 0; group < kGroupsPerWarpgroup; ++group) {
#pragma unroll
        for (int item = 0; item < kPvChunks * 4; ++item) {
            output_fragment[group][item] = 0.0f;
        }
    }
    float running_max[kGroupsPerWarpgroup][2];
    float running_sum[kGroupsPerWarpgroup][2];
#pragma unroll
    for (int group = 0; group < kGroupsPerWarpgroup; ++group) {
        running_max[group][0] = running_max[group][1] = -1.0e30f;
        running_sum[group][0] = running_sum[group][1] = 0.0f;
    }

    for (int tile = 0; tile < local_tiles; ++tile) {
        const int key_stage = tile % kDfWgKStages;
        const int value_stage = tile % kDfWgVStages;
        const int key_phase = (tile / kDfWgKStages) & 1;
        const int value_phase = (tile / kDfWgVStages) & 1;
        bf16* key_box = key_smem + key_stage * kKvElements;
        bf16* value_box = value_smem + value_stage * kKvElements;

        float score[kGroupsPerWarpgroup][4];
#pragma unroll
        for (int group = 0; group < kGroupsPerWarpgroup; ++group) {
#pragma unroll
            for (int item = 0; item < 4; ++item) score[group][item] = 0.0f;
        }

        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        constexpr uint32_t kQkSwizzle = 1;
        const uint64_t key_descriptor = dfwg_make_desc(key_box, 1, kDfWgQkChunk, kQkSwizzle);
        const uint64_t query_descriptor = dfwg_make_desc(query_smem, 1, kDfWgQkChunk, kQkSwizzle);
#pragma unroll
        for (int chunk = 0; chunk < kQkChunks; ++chunk) {
            dfwg_mbar_wait(&key_full[key_stage][chunk], key_phase);
#pragma unroll
            for (int atom = 0; atom < kQkAtoms; ++atom) {
                const uint64_t key_offset =
                    (static_cast<uint64_t>(chunk) * kKeyChunkBytes + atom * 32u) >> 4;
                const uint64_t query_offset =
                    (static_cast<uint64_t>(chunk) * kQueryChunkElements * sizeof(bf16) +
                     atom * 32u) >>
                    4;
#pragma unroll
                for (int local_group = 0; local_group < kGroupsPerWarpgroup; ++local_group) {
                    const uint64_t group_offset =
                        static_cast<uint64_t>(warpgroup * kGroupsPerWarpgroup + local_group) *
                        kDfWgQkChunk;
                    asm volatile(
                        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
                        "{%0,%1,%2,%3}, %4, %5, 1,1,1,0,0;\n"
                        : "+f"(score[local_group][0]), "+f"(score[local_group][1]),
                          "+f"(score[local_group][2]), "+f"(score[local_group][3])
                        : "l"(key_descriptor + key_offset),
                          "l"(query_descriptor + query_offset + group_offset));
                }
            }
            dfwg_commit();
        }
        dfwg_wait_group<0>();
        if (local_tid == 0) {
#pragma unroll
            for (int chunk = 0; chunk < kQkChunks; ++chunk) {
                dfwg_mbar_arrive(&key_empty[key_stage][chunk]);
            }
        }

        const int absolute_tile = first_tile + (split + tile * workers) * kDfWgTile;
        const int key_offset0 = warp * 16 + group_lane;
        const int key_offset1 = key_offset0 + 8;
        const int key_position0 = absolute_tile + key_offset0;
        const int key_position1 = absolute_tile + key_offset1;
        bool linear_full_tile = false;
        bool linear_visible0 = false;
        bool linear_visible1 = false;
        if constexpr (kCacheMode == kDfWgLinear) {
            linear_full_tile = absolute_tile >= lo && absolute_tile + kDfWgTile <= union_end;
            if (!linear_full_tile) {
                linear_visible0 = key_position0 >= lo && key_position0 < union_end;
                linear_visible1 = key_position1 >= lo && key_position1 < union_end;
            }
        }

#pragma unroll
        for (int local_group = 0; local_group < kGroupsPerWarpgroup; ++local_group) {
            const int group = warpgroup * kGroupsPerWarpgroup + local_group;
            const int query0 = group * 8 + thread_group * 2;
            const int query1 = query0 + 1;
            const int row0 = query0 >> 2;
            const int row1 = query1 >> 2;
            if constexpr (kCacheMode == kDfWgLinear) {
                if (!linear_full_tile) {
                    if (!linear_visible0) {
                        score[local_group][0] = -1.0e30f;
                        score[local_group][1] = -1.0e30f;
                    }
                    if (!linear_visible1) {
                        score[local_group][2] = -1.0e30f;
                        score[local_group][3] = -1.0e30f;
                    }
                }
            } else {
                const bool visible00 =
                    dfwg_visible<kCacheMode>(key_position0, row0, start, lo, window, union_end);
                const bool visible01 =
                    dfwg_visible<kCacheMode>(key_position0, row1, start, lo, window, union_end);
                const bool visible10 =
                    dfwg_visible<kCacheMode>(key_position1, row0, start, lo, window, union_end);
                const bool visible11 =
                    dfwg_visible<kCacheMode>(key_position1, row1, start, lo, window, union_end);
                if (!visible00) score[local_group][0] = -1.0e30f;
                if (!visible01) score[local_group][1] = -1.0e30f;
                if (!visible10) score[local_group][2] = -1.0e30f;
                if (!visible11) score[local_group][3] = -1.0e30f;
            }
            float local_max0 = fmaxf(score[local_group][0], score[local_group][2]);
            float local_max1 = fmaxf(score[local_group][1], score[local_group][3]);
#pragma unroll
            for (int offset = 4; offset <= 16; offset <<= 1) {
                local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffffu, local_max0, offset));
                local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffffu, local_max1, offset));
            }
            if (group_lane == 0) {
                max_part[warp][query0] = local_max0;
                max_part[warp][query1] = local_max1;
            }
        }
        dfwg_wg_sync(warpgroup);

#pragma unroll
        for (int local_group = 0; local_group < kGroupsPerWarpgroup; ++local_group) {
            const int group = warpgroup * kGroupsPerWarpgroup + local_group;
            const int query0 = group * 8 + thread_group * 2;
            const int query1 = query0 + 1;
            const int row0 = query0 >> 2;
            const int row1 = query1 >> 2;
            float tile_max0 = -1.0e30f;
            float tile_max1 = -1.0e30f;
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                tile_max0 = fmaxf(tile_max0, max_part[part][query0]);
                tile_max1 = fmaxf(tile_max1, max_part[part][query1]);
            }
            const float next_max0 = fmaxf(running_max[local_group][0], tile_max0);
            const float next_max1 = fmaxf(running_max[local_group][1], tile_max1);
            if constexpr (kCacheMode == kDfWgLinear) {
                if (linear_full_tile) {
                    score[local_group][0] = exp2f((score[local_group][0] - next_max0) * scale_log2);
                    score[local_group][1] = exp2f((score[local_group][1] - next_max1) * scale_log2);
                    score[local_group][2] = exp2f((score[local_group][2] - next_max0) * scale_log2);
                    score[local_group][3] = exp2f((score[local_group][3] - next_max1) * scale_log2);
                } else {
                    score[local_group][0] =
                        linear_visible0 ? exp2f((score[local_group][0] - next_max0) * scale_log2)
                                        : 0.0f;
                    score[local_group][1] =
                        linear_visible0 ? exp2f((score[local_group][1] - next_max1) * scale_log2)
                                        : 0.0f;
                    score[local_group][2] =
                        linear_visible1 ? exp2f((score[local_group][2] - next_max0) * scale_log2)
                                        : 0.0f;
                    score[local_group][3] =
                        linear_visible1 ? exp2f((score[local_group][3] - next_max1) * scale_log2)
                                        : 0.0f;
                }
            } else {
                const bool visible00 =
                    dfwg_visible<kCacheMode>(key_position0, row0, start, lo, window, union_end);
                const bool visible01 =
                    dfwg_visible<kCacheMode>(key_position0, row1, start, lo, window, union_end);
                const bool visible10 =
                    dfwg_visible<kCacheMode>(key_position1, row0, start, lo, window, union_end);
                const bool visible11 =
                    dfwg_visible<kCacheMode>(key_position1, row1, start, lo, window, union_end);
                score[local_group][0] =
                    visible00 ? exp2f((score[local_group][0] - next_max0) * scale_log2) : 0.0f;
                score[local_group][1] =
                    visible01 ? exp2f((score[local_group][1] - next_max1) * scale_log2) : 0.0f;
                score[local_group][2] =
                    visible10 ? exp2f((score[local_group][2] - next_max0) * scale_log2) : 0.0f;
                score[local_group][3] =
                    visible11 ? exp2f((score[local_group][3] - next_max1) * scale_log2) : 0.0f;
            }
            float local_sum0 = score[local_group][0] + score[local_group][2];
            float local_sum1 = score[local_group][1] + score[local_group][3];
#pragma unroll
            for (int offset = 4; offset <= 16; offset <<= 1) {
                local_sum0 += __shfl_xor_sync(0xffffffffu, local_sum0, offset);
                local_sum1 += __shfl_xor_sync(0xffffffffu, local_sum1, offset);
            }
            if (group_lane == 0) {
                sum_part[warp][query0] = local_sum0;
                sum_part[warp][query1] = local_sum1;
            }
        }
        dfwg_wg_sync(warpgroup);

#pragma unroll
        for (int local_group = 0; local_group < kGroupsPerWarpgroup; ++local_group) {
            const int group = warpgroup * kGroupsPerWarpgroup + local_group;
            const int query0 = group * 8 + thread_group * 2;
            const int query1 = query0 + 1;
            float tile_sum0 = 0.0f;
            float tile_sum1 = 0.0f;
            float tile_max0 = -1.0e30f;
            float tile_max1 = -1.0e30f;
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                tile_sum0 += sum_part[part][query0];
                tile_sum1 += sum_part[part][query1];
                tile_max0 = fmaxf(tile_max0, max_part[part][query0]);
                tile_max1 = fmaxf(tile_max1, max_part[part][query1]);
            }
            const float next_max0 = fmaxf(running_max[local_group][0], tile_max0);
            const float next_max1 = fmaxf(running_max[local_group][1], tile_max1);
            const float correction0 = exp2f((running_max[local_group][0] - next_max0) * scale_log2);
            const float correction1 = exp2f((running_max[local_group][1] - next_max1) * scale_log2);
            running_sum[local_group][0] = running_sum[local_group][0] * correction0 + tile_sum0;
            running_sum[local_group][1] = running_sum[local_group][1] * correction1 + tile_sum1;
            running_max[local_group][0] = next_max0;
            running_max[local_group][1] = next_max1;
#pragma unroll
            for (int item = 0; item < kPvChunks * 4; ++item) {
                output_fragment[local_group][item] *= (item & 1) ? correction1 : correction0;
            }

            bf16* probability = probability_smem + group * kDfWgTile * 8;
            probability[(key_offset0 >> 3) * 64 + (thread_group * 2) * 8 + (key_offset0 & 7)] =
                static_cast<bf16>(score[local_group][0]);
            probability[(key_offset0 >> 3) * 64 + (thread_group * 2 + 1) * 8 + (key_offset0 & 7)] =
                static_cast<bf16>(score[local_group][1]);
            probability[(key_offset1 >> 3) * 64 + (thread_group * 2) * 8 + (key_offset1 & 7)] =
                static_cast<bf16>(score[local_group][2]);
            probability[(key_offset1 >> 3) * 64 + (thread_group * 2 + 1) * 8 + (key_offset1 & 7)] =
                static_cast<bf16>(score[local_group][3]);
        }
        dfwg_wg_sync(warpgroup);

        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        const uint64_t value_descriptor = dfwg_make_desc(value_box, 256, 64, 1);
#pragma unroll
        for (int output_chunk = 0; output_chunk < kPvChunks; ++output_chunk) {
            dfwg_mbar_wait(&value_full[value_stage][output_chunk], value_phase);
#pragma unroll
            for (int local_group = 0; local_group < kGroupsPerWarpgroup; ++local_group) {
                const int group = warpgroup * kGroupsPerWarpgroup + local_group;
                const uint64_t probability_descriptor =
                    dfwg_make_desc(probability_smem + group * kDfWgTile * 8, 8, 256, 0);
#pragma unroll
                for (int atom = 0; atom < 4; ++atom) {
                    const uint64_t value_offset = ((output_chunk * 8192 + atom * 2048) >> 4);
                    const uint64_t probability_offset = ((atom * 2) * 64 * 2 >> 4);
                    asm volatile(
                        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
                        "{%0,%1,%2,%3}, %4,%5, 1,1,1,1,0;\n"
                        : "+f"(output_fragment[local_group][output_chunk * 4]),
                          "+f"(output_fragment[local_group][output_chunk * 4 + 1]),
                          "+f"(output_fragment[local_group][output_chunk * 4 + 2]),
                          "+f"(output_fragment[local_group][output_chunk * 4 + 3])
                        : "l"(value_descriptor + value_offset),
                          "l"(probability_descriptor + probability_offset));
                }
            }
            dfwg_commit();
        }
        dfwg_wait_group<0>();
        if (local_tid == 0) {
#pragma unroll
            for (int chunk = 0; chunk < kPvChunks; ++chunk) {
                dfwg_mbar_arrive(&value_empty[value_stage][chunk]);
            }
        }

        if (tid == 0) {
            if (tile + kDfWgKStages < local_tiles) {
                const int next_tile = tile + kDfWgKStages;
                const int next_absolute = first_tile + (split + next_tile * workers) * kDfWgTile;
                const int next_physical = dfwg_physical_tile<kCacheMode>(next_absolute);
#pragma unroll
                for (int chunk = 0; chunk < kQkChunks; ++chunk) {
                    dfwg_mbar_wait(&key_empty[key_stage][chunk], key_phase);
                }
#pragma unroll
                for (int chunk = 0; chunk < kQkChunks; ++chunk) {
                    dfwg_mbar_expect(&key_full[key_stage][chunk], kKeyChunkBytes);
                    dfwg_tma_load_kv(&key_map, &key_full[key_stage][chunk],
                                     key_smem + key_stage * kKvElements + chunk * kKeyChunkElements,
                                     chunk * kDfWgQkChunk, next_physical);
                }
            }
            if (tile + kDfWgVStages < local_tiles) {
                const int next_tile = tile + kDfWgVStages;
                const int next_absolute = first_tile + (split + next_tile * workers) * kDfWgTile;
                const int next_physical = dfwg_physical_tile<kCacheMode>(next_absolute);
#pragma unroll
                for (int chunk = 0; chunk < kPvChunks; ++chunk) {
                    dfwg_mbar_wait(&value_empty[value_stage][chunk], value_phase);
                }
#pragma unroll
                for (int chunk = 0; chunk < kPvChunks; ++chunk) {
                    dfwg_mbar_expect(&value_full[value_stage][chunk], kValueChunkBytes);
                    dfwg_tma_load_kv(
                        &value_map, &value_full[value_stage][chunk],
                        value_smem + value_stage * kKvElements + chunk * kValueChunkElements,
                        chunk * 64, next_physical);
                }
            }
        }
        dfwg_wg_sync(warpgroup);
    }

    // The coalesced epilogue reuses K storage.  A warpgroup-local barrier is
    // sufficient between tiles, but not for that CTA-wide lifetime handoff:
    // a faster WG must not overwrite K while another WG still reads it.
    if (tile_count > 1) __syncthreads();

#pragma unroll
    for (int local_group = 0; local_group < kGroupsPerWarpgroup; ++local_group) {
        const int group = warpgroup * kGroupsPerWarpgroup + local_group;
        const int query0 = group * 8 + thread_group * 2;
        const int query1 = query0 + 1;
        const float inverse_sum0 =
            running_sum[local_group][0] > 0.0f ? 1.0f / running_sum[local_group][0] : 0.0f;
        const float inverse_sum1 =
            running_sum[local_group][1] > 0.0f ? 1.0f / running_sum[local_group][1] : 0.0f;
#pragma unroll
        for (int item = 0; item < kPvChunks * 4; ++item) {
            output_fragment[local_group][item] *= (item & 1) ? inverse_sum1 : inverse_sum0;
        }

        const int row0 = query0 >> 2;
        const int head0 = query0 & 3;
        const int row1 = query1 >> 2;
        const int head1 = query1 & 3;
        if (tile_count == 1) {
            bf16* output0 =
                context + (static_cast<size_t>(row0) * kDfWgHeads + head0) * kDfWgHeadDim;
            bf16* output1 =
                context + (static_cast<size_t>(row1) * kDfWgHeads + head1) * kDfWgHeadDim;
#pragma unroll
            for (int output_chunk = 0; output_chunk < kPvChunks; ++output_chunk) {
                const int dimension0 = output_chunk * 64 + warp * 16 + group_lane;
                const int dimension1 = dimension0 + 8;
                output0[dimension0] =
                    static_cast<bf16>(output_fragment[local_group][output_chunk * 4]);
                output0[dimension1] =
                    static_cast<bf16>(output_fragment[local_group][output_chunk * 4 + 2]);
                output1[dimension0] =
                    static_cast<bf16>(output_fragment[local_group][output_chunk * 4 + 1]);
                output1[dimension1] =
                    static_cast<bf16>(output_fragment[local_group][output_chunk * 4 + 3]);
            }
            continue;
        }
        if (group_lane == 0) {
            max_workspace[(static_cast<size_t>(row0) * workers + split) * kDfWgHeads + head0] =
                running_max[local_group][0];
            sum_workspace[(static_cast<size_t>(row0) * workers + split) * kDfWgHeads + head0] =
                running_sum[local_group][0];
            max_workspace[(static_cast<size_t>(row1) * workers + split) * kDfWgHeads + head1] =
                running_max[local_group][1];
            sum_workspace[(static_cast<size_t>(row1) * workers + split) * kDfWgHeads + head1] =
                running_sum[local_group][1];
        }

        // Each warpgroup owns eight complete (row,head) vectors.  K is dead
        // after the final QK, so transpose the exact BF16 words into disjoint
        // padded slices of that storage, then drain one aligned 16-byte vector
        // per active lane.  HD128 uses 16 active lanes per output vector.
        constexpr int kEpilogueVectors = 8;
        constexpr int kEpilogueStrideWords = kDfWgHeadDim / 2 + 4;  // 256 B data + 16 B bank pad
        constexpr int kEpilogueWgWords = kEpilogueVectors * kEpilogueStrideWords;
        static_assert(kDfWgWarpgroups * kEpilogueWgWords * sizeof(uint32_t) <=
                          kDfWgKStages * kKvElements * sizeof(bf16),
                      "epilogue transpose must fit in dead K storage");
        uint32_t* epilogue = reinterpret_cast<uint32_t*>(key_smem) + warpgroup * kEpilogueWgWords;
        const int local_vector0 = query0 - warpgroup * 8;
        const int local_vector1 = local_vector0 + 1;

#pragma unroll
        for (int output_chunk = 0; output_chunk < kPvChunks; ++output_chunk) {
            const int dimension0 = output_chunk * 64 + warp * 16 + group_lane;
            const int dimension1 = dimension0 + 8;
            const float value00 = output_fragment[local_group][output_chunk * 4];
            const float value01 = output_fragment[local_group][output_chunk * 4 + 2];
            const float value10 = output_fragment[local_group][output_chunk * 4 + 1];
            const float value11 = output_fragment[local_group][output_chunk * 4 + 3];
            const float peer00 = __shfl_down_sync(0xffffffffu, value00, 4);
            const float peer01 = __shfl_down_sync(0xffffffffu, value01, 4);
            const float peer10 = __shfl_down_sync(0xffffffffu, value10, 4);
            const float peer11 = __shfl_down_sync(0xffffffffu, value11, 4);
            if ((group_lane & 1) == 0) {
                epilogue[local_vector0 * kEpilogueStrideWords + dimension0 / 2] =
                    dfwg_pack_bf16(value00, peer00);
                epilogue[local_vector0 * kEpilogueStrideWords + dimension1 / 2] =
                    dfwg_pack_bf16(value01, peer01);
                epilogue[local_vector1 * kEpilogueStrideWords + dimension0 / 2] =
                    dfwg_pack_bf16(value10, peer10);
                epilogue[local_vector1 * kEpilogueStrideWords + dimension1 / 2] =
                    dfwg_pack_bf16(value11, peer11);
            }
        }

        dfwg_wg_sync(warpgroup);
#pragma unroll
        for (int vector = warp; vector < kEpilogueVectors; vector += 4) {
            if (lane < kDfWgHeadDim / 8) {
                const uint4 packed =
                    reinterpret_cast<const uint4*>(epilogue + vector * kEpilogueStrideWords)[lane];
                const int query = warpgroup * kEpilogueVectors + vector;
                bf16* destination =
                    acc_workspace +
                    ((static_cast<size_t>(query >> 2) * workers + split) * kDfWgHeads +
                     (query & 3)) *
                        kDfWgHeadDim;
                reinterpret_cast<uint4*>(destination)[lane] = packed;
            }
        }
    }
}

__device__ __forceinline__ void dfwg_combine_body(const float* __restrict__ max_workspace,
                                                  const float* __restrict__ sum_workspace,
                                                  const bf16* __restrict__ acc_workspace,
                                                  bf16* __restrict__ output, int workers) {
    constexpr int kUnits = kDfWgCombineDim / 8;
    const int output_begin = blockIdx.x * kDfWgCombineDim;
    const int head = output_begin / kDfWgHeadDim;
    const int dimension_begin = output_begin % kDfWgHeadDim;
    const int tid = threadIdx.x;
    const float scale_log2 = kDfWgScale * 1.4426950408889634f;

    extern __shared__ __align__(16) char combine_smem[];
    bf16* acc_smem = reinterpret_cast<bf16*>(combine_smem);
    float* max_smem = reinterpret_cast<float*>(acc_smem + workers * kDfWgCombineDim);
    float* sum_smem = max_smem + workers;
    __shared__ float numerator_reduction[kDfWgCombineThreads];
    __shared__ float denominator_reduction[kDfWgCombineThreads];

    for (int index = tid; index < workers * kUnits; index += kDfWgCombineThreads) {
        const int split = index / kUnits;
        const int unit = index - split * kUnits;
        *reinterpret_cast<uint4*>(acc_smem + split * kDfWgCombineDim + unit * 8) =
            *reinterpret_cast<const uint4*>(
                acc_workspace + (static_cast<size_t>(split) * kDfWgHeads + head) * kDfWgHeadDim +
                dimension_begin + unit * 8);
    }
    for (int split = tid; split < workers; split += kDfWgCombineThreads) {
        max_smem[split] = max_workspace[split * kDfWgHeads + head];
        sum_smem[split] = sum_workspace[split * kDfWgHeads + head];
    }
    __syncthreads();

    float local_max = -1.0e30f;
    for (int split = tid; split < workers; split += kDfWgCombineThreads) {
        local_max = fmaxf(local_max, max_smem[split]);
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, offset));
    }
    if ((tid & 31) == 0) numerator_reduction[tid >> 5] = local_max;
    __syncthreads();

    float global_max = -1.0e30f;
#pragma unroll
    for (int warp = 0; warp < kDfWgCombineThreads / 32; ++warp) {
        global_max = fmaxf(global_max, numerator_reduction[warp]);
    }

    constexpr int kGroups = kDfWgCombineThreads / kDfWgCombineDim;
    const int dimension = tid % kDfWgCombineDim;
    const int group = tid / kDfWgCombineDim;
    float numerator = 0.0f;
    float denominator = 0.0f;
    for (int split = group; split < workers; split += kGroups) {
        const float weight = exp2f((max_smem[split] - global_max) * scale_log2) * sum_smem[split];
        numerator += weight * static_cast<float>(acc_smem[split * kDfWgCombineDim + dimension]);
        denominator += weight;
    }
    __syncthreads();
    numerator_reduction[tid] = numerator;
    denominator_reduction[tid] = denominator;
    __syncthreads();

    if (tid < kDfWgCombineDim) {
        float total_numerator = 0.0f;
        float total_denominator = 0.0f;
#pragma unroll
        for (int source_group = 0; source_group < kGroups; ++source_group) {
            total_numerator += numerator_reduction[source_group * kDfWgCombineDim + tid];
            total_denominator += denominator_reduction[source_group * kDfWgCombineDim + tid];
        }
        output[output_begin + tid] =
            __float2bfloat16(total_numerator / (total_denominator + 1.0e-20f));
    }
}

__global__ __launch_bounds__(kDfWgCombineThreads) void DfAttnWgmmaCombineKernel(
    const float* __restrict__ max_workspace, const float* __restrict__ sum_workspace,
    const bf16* __restrict__ acc_workspace, bf16* __restrict__ context,
    const int* __restrict__ df_meta, const int* __restrict__ status, int lo, int window,
    int workers) {
    NEXRT_PDL_ENTRY_SYNC();
    if (status != nullptr && *status != 0) return;
    const int start = df_meta[1];
    int union_begin = lo;
    if (window > 0) union_begin = max(union_begin, start - window + 1);
    const int first_tile = (union_begin / kDfWgTile) * kDfWgTile;
    const int tile_count =
        max(0, (start + kDfWgDraftRows - first_tile + kDfWgTile - 1) / kDfWgTile);
    if (tile_count <= 1) return;
    const int active_workers = min(workers, tile_count);
    const int row = blockIdx.y;
    dfwg_combine_body(
        max_workspace + static_cast<size_t>(row) * workers * kDfWgHeads,
        sum_workspace + static_cast<size_t>(row) * workers * kDfWgHeads,
        acc_workspace + static_cast<size_t>(row) * workers * kDfWgHeads * kDfWgHeadDim,
        context + static_cast<size_t>(row) * kDfWgHeads * kDfWgHeadDim, active_workers);
}

CUtensorMap dfwg_encode_map(void* pointer, int outer, int inner, int box_outer, int box_inner,
                            CUtensorMapSwizzle swizzle) {
    CUtensorMap map{};
    uint64_t dimensions[2] = {static_cast<uint64_t>(inner), static_cast<uint64_t>(outer)};
    uint64_t strides[1] = {static_cast<uint64_t>(inner) * sizeof(bf16)};
    uint32_t box[2] = {static_cast<uint32_t>(box_inner), static_cast<uint32_t>(box_outer)};
    uint32_t element_strides[2] = {1, 1};
    cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, pointer, dimensions, strides,
                           box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                           CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return map;
}

int dfwg_smem_bytes() {
    return (kDfWgQueryRows * kDfWgHeadDim +
            (kDfWgKStages + kDfWgVStages) * kDfWgTile * kDfWgHeadDim + kDfWgTile * kDfWgQueryRows) *
               static_cast<int>(sizeof(bf16)) +
           1024;
}

template <int kCacheMode>
int launch_df_attn_wgmma_partial(const void* query, const void* key_cache, const void* value_cache,
                                 void* max_workspace, void* sum_workspace, void* acc_workspace,
                                 void* context, const void* df_meta, const void* status, int lo,
                                 int window, int physical_capacity, int workers, void* stream) {
    constexpr CUtensorMapSwizzle kQkSwizzle = CU_TENSOR_MAP_SWIZZLE_128B;
    CUtensorMap key_map = dfwg_encode_map(const_cast<void*>(key_cache), physical_capacity,
                                          kDfWgHeadDim, kDfWgTile, kDfWgQkChunk, kQkSwizzle);
    CUtensorMap value_map =
        dfwg_encode_map(const_cast<void*>(value_cache), physical_capacity, kDfWgHeadDim, kDfWgTile,
                        64, CU_TENSOR_MAP_SWIZZLE_128B);
    CUtensorMap query_map = dfwg_encode_map(const_cast<void*>(query), kDfWgQueryRows, kDfWgHeadDim,
                                            kDfWgQueryRows, kDfWgQkChunk, kQkSwizzle);
    const int smem = dfwg_smem_bytes();
    static bool initialized = false;
    if (!initialized) {
        const cudaError_t attribute =
            cudaFuncSetAttribute(DfAttnWgmmaPartialKernel<kCacheMode>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (attribute != cudaSuccess) return static_cast<int>(attribute);
        initialized = true;
    }
    DfAttnWgmmaPartialKernel<kCacheMode>
        <<<dim3(workers), dim3(128 * kDfWgWarpgroups), smem, static_cast<cudaStream_t>(stream)>>>(
            key_map, value_map, query_map, static_cast<float*>(max_workspace),
            static_cast<float*>(sum_workspace), static_cast<bf16*>(acc_workspace),
            static_cast<bf16*>(context), static_cast<const int*>(df_meta),
            static_cast<const int*>(status), lo, window, workers);
    return static_cast<int>(cudaGetLastError());
}

int dispatch_df_attn_wgmma_partial(const void* query, const void* key_cache,
                                   const void* value_cache, void* max_workspace,
                                   void* sum_workspace, void* acc_workspace, void* context,
                                   const void* df_meta, const void* status, int lo, int window,
                                   int physical_capacity, int workers, int cache_mode,
                                   void* stream) {
    if (workers < 1 || physical_capacity < 1 || lo < 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (cache_mode == kDfWgLinear && window == 0) {
        return launch_df_attn_wgmma_partial<kDfWgLinear>(
            query, key_cache, value_cache, max_workspace, sum_workspace, acc_workspace, context,
            df_meta, status, lo, window, physical_capacity, workers, stream);
    }
    if (cache_mode == kDfWgRing8192 && physical_capacity == 8192 && window > 0) {
        return launch_df_attn_wgmma_partial<kDfWgRing8192>(
            query, key_cache, value_cache, max_workspace, sum_workspace, acc_workspace, context,
            df_meta, status, lo, window, physical_capacity, workers, stream);
    }
    return static_cast<int>(cudaErrorInvalidValue);
}

int launch_df_attn_wgmma_combine(const void* max_workspace, const void* sum_workspace,
                                 const void* acc_workspace, void* context, const void* df_meta,
                                 const void* status, int lo, int window, int workers,
                                 void* stream) {
    if (workers < 1) return static_cast<int>(cudaErrorInvalidValue);
    const int smem = workers * kDfWgCombineDim * static_cast<int>(sizeof(bf16)) +
                     2 * workers * static_cast<int>(sizeof(float));
    DfAttnWgmmaCombineKernel<<<dim3(kDfWgHeads * kDfWgHeadDim / kDfWgCombineDim, kDfWgDraftRows),
                               dim3(kDfWgCombineThreads), smem,
                               static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(max_workspace), static_cast<const float*>(sum_workspace),
        static_cast<const bf16*>(acc_workspace), static_cast<bf16*>(context),
        static_cast<const int*>(df_meta), static_cast<const int*>(status), lo, window, workers);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace nexrt::sm90::nex_n2_5_pro::DFlash

using namespace nexrt::sm90::nex_n2_5_pro::DFlash;

extern "C" {

int NexN25ProDFlashAttnWgmmaPartialCtlLaunch(const void* query, const void* key_cache,
                                             const void* value_cache, void* max_workspace,
                                             void* sum_workspace, void* acc_workspace,
                                             void* context, const void* df_meta, const void* status,
                                             int lo, int window, int physical_capacity, int workers,
                                             int cache_mode, void* stream) {
    return dispatch_df_attn_wgmma_partial(
        query, key_cache, value_cache, max_workspace, sum_workspace, acc_workspace, context,
        df_meta, status, lo, window, physical_capacity, workers, cache_mode, stream);
}

int NexN25ProDFlashAttnWgmmaCombineLaunch(const void* max_workspace, const void* sum_workspace,
                                          const void* acc_workspace, void* context,
                                          const void* df_meta, const void* status, int lo,
                                          int window, int workers, void* stream) {
    return launch_df_attn_wgmma_combine(max_workspace, sum_workspace, acc_workspace, context,
                                        df_meta, status, lo, window, workers, stream);
}

int NexN25ProDFlashAttnWgmmaTile() { return kDfWgTile; }

}  // extern "C"
