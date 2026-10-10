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

// Production context-parallel attention megakernel for H100. A fixed 128-CTA
// grid computes the local WGMMA partial, combines it after a reset-free
// cooperative barrier, and publishes CP readiness. Sequence length remains
// device-driven, with a masked path only for the final 64-token tile.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY bs1
#endif

#include "nex_n2_5_pro_cp_hopper_common.cuh"

#include <cooperative_groups.h>

namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY {
constexpr int kWsThreads = 256;
constexpr int kMathThreads = 128;
constexpr int kNumSplit128 = 128;
constexpr int kPvDChunk = 64;
constexpr int kPvChunks = kHeadDim / kPvDChunk;
constexpr int kKvStages = 1;
constexpr int kMathNamedBarrier = 1;
constexpr int kPStoreNamedBarrier = 2;
// Apply the SM90 evict-first cache policy to the one-pass K/V stream; each
// split touches disjoint K/V rows, while the Q tile is loaded repeatedly.
constexpr uint64_t kTmaEvictFirst = 0x12F0000000000000ULL;

template <int N>
__device__ __forceinline__ void setmaxnreg_inc() {
    static_assert(N >= 24 && N <= 256 && (N % 8) == 0, "invalid setmaxnreg value");
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(N));
}

template <int N>
__device__ __forceinline__ void setmaxnreg_dec() {
    static_assert(N >= 24 && N <= 256 && (N % 8) == 0, "invalid setmaxnreg value");
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(N));
}

// Reusable sense-reversing device-wide barrier.  Cooperative launch is a
// correctness requirement: all 128 CTAs must be resident while they spin.
// The high bit is the sense and the low 31 bits form the arrival counter.
__device__ __forceinline__ void grid_bar_sync_modern(uint32_t* counter, int num_blocks) {
    constexpr uint32_t kTag = 0x80000000u;
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t increment =
            (blockIdx.x == 0) ? (kTag - static_cast<uint32_t>(num_blocks - 1)) : 1u;
        uint32_t old;
        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;\n"
                     : "=r"(old)
                     : "l"(counter), "r"(increment)
                     : "memory");
        uint32_t current;
        do {
            asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n"
                         : "=r"(current)
                         : "l"(counter)
                         : "memory");
        } while (((current ^ old) & kTag) == 0u);
    }
    __syncthreads();
}

template <int BarrierId>
__device__ __forceinline__ void named_barrier_sync_128() {
    asm volatile("bar.sync %0, %1;\n" ::"r"(BarrierId), "r"(kMathThreads) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" ::"r"(cvta32(bar)) : "memory");
}

__device__ __forceinline__ void release_empty_from_math(uint64_t* bar, int lane) {
    if (lane == 0) {
        mbar_arrive(bar);
    }
}

__device__ __forceinline__ int physical_tile_start(int logical_start, int seqlen_kv) {
    return min(max(logical_start, 0), seqlen_kv - kBlockN);
}

__device__ __forceinline__ void reduce4_max_math(const float in[4], float out[4], float* scratch) {
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
    named_barrier_sync_128<kMathNamedBarrier>();
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        float v = -3.402823466e38f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            v = fmaxf(v, scratch[h * kWarps * 4 + w * 4 + tg]);
        }
        out[h] = v;
    }
    named_barrier_sync_128<kMathNamedBarrier>();
}

__device__ __forceinline__ void reduce4_sum_math(const float in[4], float out[4], float* scratch) {
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
    named_barrier_sync_128<kMathNamedBarrier>();
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            v += scratch[h * kWarps * 4 + w * 4 + tg];
        }
        out[h] = v;
    }
    named_barrier_sync_128<kMathNamedBarrier>();
}

template <int QkHeadChunk>
__device__ __forceinline__ uint64_t make_k_qk_desc(bf16* k_sh) {
    static_assert(QkHeadChunk == 32 || QkHeadChunk == 64, "unsupported QK chunk");
    if constexpr (QkHeadChunk == 32) {
        return make_desc(k_sh, 1, 32, 2);
    } else {
        return make_desc(k_sh, 1, 64, 1);
    }
}

__device__ __forceinline__ void tma_load_4d_kv(const CUtensorMap* map, uint64_t* bar, void* dst,
                                               int c0, int c1, int c2, int c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1, {%2, %3, %4, %5}], [%6], %7;\n" ::"r"(cvta32(dst)),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(cvta32(bar)),
        "l"(kTmaEvictFirst)
        : "memory");
}

template <int QkHeadChunk>
__device__ __forceinline__ void issue_k_tile(const CUtensorMap* k_desc, uint64_t* k_full,
                                             bf16* k_sh, int kv_start) {
    constexpr int kQkChunks = kHeadDim / QkHeadChunk;
    constexpr int kQkChunkElems = kBlockN * QkHeadChunk;
#pragma unroll
    for (int j = 0; j < kQkChunks; ++j) {
        mbar_expect(k_full + j, kQkChunkElems * static_cast<int>(sizeof(bf16)));
        tma_load_4d_kv(k_desc, k_full + j, k_sh + j * kQkChunkElems, j * QkHeadChunk, 0, kv_start,
                       0);
    }
}

__device__ __forceinline__ void issue_v_tile(const CUtensorMap* v_desc, uint64_t* v_full,
                                             bf16* v_sh, int kv_start) {
    constexpr int kPvChunkElems = kBlockN * kPvDChunk;
#pragma unroll
    for (int j = 0; j < kPvChunks; ++j) {
        mbar_expect(v_full + j, kPvChunkElems * static_cast<int>(sizeof(bf16)));
        tma_load_4d_kv(v_desc, v_full + j, v_sh + j * kPvChunkElems, j * kPvDChunk, 0, kv_start, 0);
    }
}

template <int NumSplit, int QkHeadChunk, bool EqualTileRanges>
__device__ __forceinline__ bool Raw16HopperWsPartialBody(
    const CUtensorMap* k_desc, const CUtensorMap* q_desc, const CUtensorMap* v_desc,
    bf16* __restrict__ op, bf16* __restrict__ glse, int seqlen_arg, const int* __restrict__ slot,
    int cp_rank, int cp) {
    static_assert(NumSplit == kNumSplit128, "production path uses 128 CTAs");
    static_assert(QkHeadChunk == 32 || QkHeadChunk == 64, "unsupported QK chunk");

    constexpr int kQkChunks = kHeadDim / QkHeadChunk;
    constexpr int kQkAtomsPerChunk = QkHeadChunk / 16;
    constexpr int kQkChunkElems = kBlockN * QkHeadChunk;
    constexpr int kPvChunkElems = kBlockN * kPvDChunk;
    constexpr int kKvTileElems = kBlockN * kHeadDim;

    const int split = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int math_warp = warp;
    const int gid = lane >> 2;
    const int tg = lane & 3;
    const bool is_math = tid < kMathThreads;
    const bool is_producer = !is_math;
    const bool is_tma_thread = tid == kMathThreads;

    extern __shared__ __align__(1024) char smem[];
    bf16* q_sh = reinterpret_cast<bf16*>(smem);
    bf16* k_sh = q_sh + kHeads * kHeadDim;
    bf16* v_sh = k_sh + kKvStages * kKvTileElems;
    bf16* p_sh = v_sh + kKvStages * kKvTileElems;

    __shared__ __align__(8) uint64_t q_full[1];
    __shared__ __align__(8) uint64_t k_full[kKvStages * kQkChunks], k_empty[kKvStages * kQkChunks];
    __shared__ __align__(8) uint64_t v_full[kKvStages * kPvChunks], v_empty[kKvStages * kPvChunks];
    __shared__ float red_scratch[4 * kWarps * 4];

    if (tid == 0) {
        mbar_init(q_full, 1);
#pragma unroll
        for (int stage = 0; stage < kKvStages; ++stage) {
#pragma unroll
            for (int j = 0; j < kQkChunks; ++j) {
                mbar_init(k_full + stage * kQkChunks + j, 1);
                mbar_init(k_empty + stage * kQkChunks + j, kWarps);
            }
#pragma unroll
            for (int j = 0; j < kPvChunks; ++j) {
                mbar_init(v_full + stage * kPvChunks + j, 1);
                mbar_init(v_empty + stage * kPvChunks + j, kWarps);
            }
        }
        prefetch_tma(k_desc);
        prefetch_tma(q_desc);
        prefetch_tma(v_desc);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    __syncthreads();

    // Fixed-S launches pass slot=nullptr and seqlen_arg is the local length.
    // Device-S launches encode TMA maps against the full cache capacity, then
    // derive this rank's contiguous CP shard from slot[1] inside the graph.
    int seqlen_kv = seqlen_arg;
    int shard_off = 0;
    if (slot != nullptr) {
        const int total_seqlen = slot[1] + 1;
        const int base = total_seqlen / cp;
        shard_off = cp_rank * base;
        seqlen_kv = (cp_rank < cp - 1) ? base : (total_seqlen - (cp - 1) * base);
    }

    int range_begin;
    int range_end;
    int ntiles;
    if constexpr (EqualTileRanges) {
        const int kv_per = seqlen_kv / NumSplit;
        range_begin = split * kv_per;
        range_end = range_begin + kv_per;
        ntiles = kv_per / kBlockN;
    } else {
        const int total_tiles = (seqlen_kv + kBlockN - 1) / kBlockN;
        const int tile_begin =
            static_cast<int>((static_cast<long long>(split) * total_tiles) / NumSplit);
        const int tile_end =
            static_cast<int>((static_cast<long long>(split + 1) * total_tiles) / NumSplit);
        range_begin = tile_begin * kBlockN;
        range_end = min(seqlen_kv, tile_end * kBlockN);
        ntiles = tile_end - tile_begin;
    }

    if (ntiles <= 0) {
        if (is_math) {
#pragma unroll
            for (int dchunk = 0; dchunk < 4; ++dchunk) {
#pragma unroll
                for (int hg = 0; hg < 2; ++hg) {
#pragma unroll
                    for (int hb = 0; hb < 2; ++hb) {
                        const int head = hg * 8 + tg * 2 + hb;
                        const int d0 = dchunk * 64 + math_warp * 16 + gid;
                        const int d1 = d0 + 8;
                        op[(static_cast<size_t>(head) * NumSplit + split) * kHeadDim + d0] =
                            __float2bfloat16(0.0f);
                        op[(static_cast<size_t>(head) * NumSplit + split) * kHeadDim + d1] =
                            __float2bfloat16(0.0f);
                    }
                }
            }
            if (math_warp == 0 && gid == 0) {
#pragma unroll
                for (int hslot = 0; hslot < 4; ++hslot) {
                    const int head = (hslot >> 1) * 8 + tg * 2 + (hslot & 1);
                    glse[head * NumSplit + split] = __float2bfloat16(-3.402823466e38f);
                }
            }
        }
        return false;
    }

    if (is_producer) {
        setmaxnreg_dec<40>();
        if (is_tma_thread) {
            mbar_expect(q_full, kHeads * kHeadDim * static_cast<int>(sizeof(bf16)));
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                tma_load_3d(q_desc, q_full, q_sh + j * (kHeads * 64), j * 64, 0, 0);
            }

            for (int tile = 0; tile < min(kKvStages, ntiles); ++tile) {
                const int stage = tile;
                const int logical_start = range_begin + tile * kBlockN;
                const int kv_start = shard_off + physical_tile_start(logical_start, seqlen_kv);
                issue_k_tile<QkHeadChunk>(k_desc, k_full + stage * kQkChunks,
                                          k_sh + stage * kKvTileElems, kv_start);
                issue_v_tile(v_desc, v_full + stage * kPvChunks, v_sh + stage * kKvTileElems,
                             kv_start);
            }

            for (int tile = kKvStages; tile < ntiles; ++tile) {
                const int stage = tile % kKvStages;
                const int prev_phase = ((tile - kKvStages) / kKvStages) & 1;
                const int logical_start = range_begin + tile * kBlockN;
                const int kv_start = shard_off + physical_tile_start(logical_start, seqlen_kv);
#pragma unroll
                for (int j = 0; j < kQkChunks; ++j) {
                    mbar_wait(k_empty + stage * kQkChunks + j, prev_phase);
                }
                issue_k_tile<QkHeadChunk>(k_desc, k_full + stage * kQkChunks,
                                          k_sh + stage * kKvTileElems, kv_start);

#pragma unroll
                for (int j = 0; j < kPvChunks; ++j) {
                    mbar_wait(v_empty + stage * kPvChunks + j, prev_phase);
                }
                issue_v_tile(v_desc, v_full + stage * kPvChunks, v_sh + stage * kKvTileElems,
                             kv_start);
            }
        }
        return true;
    }

    setmaxnreg_inc<232>();
    mbar_wait(q_full, 0);

    float acc_o[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        acc_o[i] = 0.0f;
    }
    float m_i[4] = {-3.402823466e38f, -3.402823466e38f, -3.402823466e38f, -3.402823466e38f};
    float l_i[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int tile = 0; tile < ntiles; ++tile) {
        const int stage = tile % kKvStages;
        const int phase = (tile / kKvStages) & 1;
        bf16* k_tile = k_sh + stage * kKvTileElems;
        bf16* v_tile = v_sh + stage * kKvTileElems;
        const int logical_start = range_begin + tile * kBlockN;
        const int logical_end = min(range_end, logical_start + kBlockN);
        const int physical_start = physical_tile_start(logical_start, seqlen_kv);
        const bool full_tile =
            (physical_start == logical_start) && (logical_end == logical_start + kBlockN);
        float acc_s[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        wg_fence();
        const uint64_t k_gmma = make_k_qk_desc<QkHeadChunk>(k_tile);
        const uint64_t q_gmma = make_desc(q_sh, 1, 64, 1);
#pragma unroll
        for (int kc = 0; kc < kQkChunks; ++kc) {
            mbar_wait(k_full + stage * kQkChunks + kc, phase);
#pragma unroll
            for (int kk = 0; kk < kQkAtomsPerChunk; ++kk) {
                const int ki = kc * kQkAtomsPerChunk + kk;
                const uint64_t ko =
                    ((kc * kQkChunkElems * static_cast<int>(sizeof(bf16)) + kk * 32) >> 4);
                const uint64_t qo = (((ki >> 2) * 2048) + ((ki & 3) * 32)) >> 4;
                wgmma_m64n16_nn(acc_s, k_gmma + ko, q_gmma + qo);
            }
        }
        wg_commit();
        wg_wait<0>();

#pragma unroll
        for (int j = 0; j < kQkChunks; ++j) {
            release_empty_from_math(k_empty + stage * kQkChunks + j, lane);
        }

        float local_max[4];
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            local_max[h] = -3.402823466e38f;
            if constexpr (EqualTileRanges) {
#pragma unroll
                for (int rv = 0; rv < 2; ++rv) {
                    const int idx = ((h >> 1) * 4) + rv * 2 + (h & 1);
                    local_max[h] = fmaxf(local_max[h], acc_s[idx]);
                }
            } else {
                if (full_tile) {
#pragma unroll
                    for (int rv = 0; rv < 2; ++rv) {
                        const int idx = ((h >> 1) * 4) + rv * 2 + (h & 1);
                        local_max[h] = fmaxf(local_max[h], acc_s[idx]);
                    }
                } else {
#pragma unroll
                    for (int rv = 0; rv < 2; ++rv) {
                        const int row = math_warp * 16 + gid + rv * 8;
                        const int key = physical_start + row;
                        const int idx = ((h >> 1) * 4) + rv * 2 + (h & 1);
                        if (key >= logical_start && key < logical_end) {
                            local_max[h] = fmaxf(local_max[h], acc_s[idx]);
                        }
                    }
                }
            }
        }
        float tile_max[4];
        reduce4_max_math(local_max, tile_max, red_scratch);

        float corr[4];
        float local_sum[4];
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const float new_m = fmaxf(m_i[h], tile_max[h]);
            corr[h] = exp2f((m_i[h] - new_m) * kScaleLog2);
            local_sum[h] = 0.0f;
            if constexpr (EqualTileRanges) {
#pragma unroll
                for (int rv = 0; rv < 2; ++rv) {
                    const int idx = ((h >> 1) * 4) + rv * 2 + (h & 1);
                    const float e = exp2f((acc_s[idx] - new_m) * kScaleLog2);
                    acc_s[idx] = e;
                    local_sum[h] += e;
                }
            } else {
                if (full_tile) {
#pragma unroll
                    for (int rv = 0; rv < 2; ++rv) {
                        const int idx = ((h >> 1) * 4) + rv * 2 + (h & 1);
                        const float e = exp2f((acc_s[idx] - new_m) * kScaleLog2);
                        acc_s[idx] = e;
                        local_sum[h] += e;
                    }
                } else {
#pragma unroll
                    for (int rv = 0; rv < 2; ++rv) {
                        const int row = math_warp * 16 + gid + rv * 8;
                        const int key = physical_start + row;
                        const int idx = ((h >> 1) * 4) + rv * 2 + (h & 1);
                        const float e = (key >= logical_start && key < logical_end)
                                            ? exp2f((acc_s[idx] - new_m) * kScaleLog2)
                                            : 0.0f;
                        acc_s[idx] = e;
                        local_sum[h] += e;
                    }
                }
            }
            m_i[h] = new_m;
        }
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int h = (((i & 7) >> 2) * 2) + (i & 1);
            acc_o[i] *= corr[h];
        }
        float tile_sum[4];
        reduce4_sum_math(local_sum, tile_sum, red_scratch);
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            l_i[h] = l_i[h] * corr[h] + tile_sum[h];
        }

        named_barrier_sync_128<kPStoreNamedBarrier>();
        stmatrix_x4(p_sh + math_warp * 256 + (lane & 15) * 16 +
                        (((((lane & 31) >> 4) + ((lane & 7) >> 2)) & 1) * 8),
                    pack_bf16(acc_s[0], acc_s[1]), pack_bf16(acc_s[2], acc_s[3]),
                    pack_bf16(acc_s[4], acc_s[5]), pack_bf16(acc_s[6], acc_s[7]));
        named_barrier_sync_128<kPStoreNamedBarrier>();

        wg_fence();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        const uint64_t v_gmma = make_desc(v_tile, 256, 64, 1);
        const uint64_t p_gmma = make_desc(p_sh, 0, 16, 3);
#pragma unroll
        for (int i8 = 0; i8 < kPvChunks; ++i8) {
            mbar_wait(v_full + stage * kPvChunks + i8, phase);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const uint64_t vo =
                    ((i8 * kPvChunkElems * static_cast<int>(sizeof(bf16)) + kk * 2048) >> 4);
                const uint64_t po = ((kk * 512) >> 4);
                wgmma_m64n16_tn(acc_o + i8 * 8, v_gmma + vo, p_gmma + po);
            }
        }
        wg_commit();
        wg_wait<0>();

#pragma unroll
        for (int j = 0; j < kPvChunks; ++j) {
            release_empty_from_math(v_empty + stage * kPvChunks + j, lane);
        }
    }

    float inv_l_i[4];
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        inv_l_i[h] = 1.0f / l_i[h];
    }
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int h = (((i & 7) >> 2) * 2) + (i & 1);
        acc_o[i] *= inv_l_i[h];
    }

    if (math_warp == 0 && gid == 0) {
#pragma unroll
        for (int hslot = 0; hslot < 4; ++hslot) {
            const int head = (hslot >> 1) * 8 + tg * 2 + (hslot & 1);
            glse[head * NumSplit + split] =
                __float2bfloat16(log2f(l_i[hslot]) + m_i[hslot] * kScaleLog2);
        }
    }

#pragma unroll
    for (int dchunk = 0; dchunk < 4; ++dchunk) {
#pragma unroll
        for (int hg = 0; hg < 2; ++hg) {
#pragma unroll
            for (int hb = 0; hb < 2; ++hb) {
                const int head = hg * 8 + tg * 2 + hb;
                const int d0 = dchunk * 64 + math_warp * 16 + gid;
                const int d1 = d0 + 8;
                const int base_idx = dchunk * 8 + hg * 4 + hb;
                op[(static_cast<size_t>(head) * NumSplit + split) * kHeadDim + d0] =
                    __float2bfloat16(acc_o[base_idx]);
                op[(static_cast<size_t>(head) * NumSplit + split) * kHeadDim + d1] =
                    __float2bfloat16(acc_o[base_idx + 2]);
            }
        }
    }

    return true;
}

// Device form of the local combine stage. Its arithmetic, synchronization,
// and BF16 rounding points match the standalone formulation.
template <int W, bool NaturalLogLse = false>
__device__ __forceinline__ void Raw16LocalCombineBody(const bf16* __restrict__ op,
                                                      const bf16* __restrict__ glse_log2,
                                                      int num_split, bf16* __restrict__ out,
                                                      float* __restrict__ out_lse, int dblock,
                                                      int head, float* corr_sh) {
    constexpr int NT = W * 32;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int d = dblock * 32 + lane;
    __shared__ float red[W], part[W][32], lmx_sh, denom_sh;

    float pm = -3.402823466e38f;
    for (int s = tid; s < num_split; s += NT) {
        pm = fmaxf(pm, __bfloat162float(glse_log2[head * num_split + s]));
    }
    pm = warp_all_max(pm);
    if (lane == 0) {
        red[warp] = pm;
    }
    __syncthreads();
    if (tid == 0) {
        float m = -3.402823466e38f;
#pragma unroll
        for (int w = 0; w < W; ++w) {
            m = fmaxf(m, red[w]);
        }
        lmx_sh = m;
    }
    __syncthreads();
    const float lmx = lmx_sh;

    float ps = 0.0f;
    for (int s = tid; s < num_split; s += NT) {
        const float c = exp2f(__bfloat162float(glse_log2[head * num_split + s]) - lmx);
        corr_sh[s] = c;
        ps += c;
    }
    ps = warp_all_sum(ps);
    if (lane == 0) {
        red[warp] = ps;
    }
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < W; ++w) {
            sum += red[w];
        }
        denom_sh = sum;
        if (dblock == 0) {
            const float lse_log2 = log2f(sum) + lmx;
            if constexpr (NaturalLogLse) {
                out_lse[head] = lse_log2 * 0.6931471805599453f;
            } else {
                out_lse[head] = lse_log2;
            }
        }
    }
    __syncthreads();
    const float inv = 1.0f / denom_sh;

    float acc = 0.0f;
    for (int s = warp; s < num_split; s += W) {
        const size_t idx = (static_cast<size_t>(head) * num_split + s) * kHeadDim + d;
        acc += corr_sh[s] * __bfloat162float(op[idx]);
    }
    part[warp][lane] = acc;
    __syncthreads();
    if (warp == 0) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < W; ++w) {
            sum += part[w][lane];
        }
        out[head * kHeadDim + d] = __float2bfloat16(sum * inv);
    }
}

// NexRT device-S CP path: fuse the partial-to-combine stage switch with
// the existing four-rank publish contract.  The first reset-free grid barrier
// replaces the partial/combine kernel boundary; the second replaces the
// combine kernel's cooperative grid.sync before CTA 0 publishes readiness.
template <int NumSplit, int QkHeadChunk>
__global__ __launch_bounds__(kWsThreads, 1) void Raw16HopperWsStageSwitchPublishDynKernel(
    __grid_constant__ const CUtensorMap k_desc, __grid_constant__ const CUtensorMap q_desc,
    __grid_constant__ const CUtensorMap v_desc, bf16* __restrict__ op, bf16* __restrict__ glse,
    bf16* __restrict__ cp_ctx_partial, float* __restrict__ cp_lse_partial,
    uint32_t* __restrict__ grid_counter, const int* __restrict__ slot, int cp_rank, int cp,
    unsigned long long ready_mc, int* __restrict__ ready_status, int* __restrict__ it_ctr,
    int* __restrict__ cur_gen) {
    static_assert(NumSplit == kNumSplit128, "stage-switch path requires 128 CTAs");
    static_assert(QkHeadChunk == 64, "stage-switch path requires 64-head QK chunks");
    static_assert(kHeadDim / 32 == 8, "combine CTA remap assumes eight D blocks");

    const bool redistributed = Raw16HopperWsPartialBody<NumSplit, QkHeadChunk, false>(
        &k_desc, &q_desc, &v_desc, op, glse, 0, slot, cp_rank, cp);

    if (redistributed && threadIdx.x >= kMathThreads) {
        setmaxnreg_inc<232>();
    }

    grid_bar_sync_modern(grid_counter, NumSplit);

    constexpr int kCombineWarps = 8;
    constexpr int kCombineDBlocks = kHeadDim / 32;
    const int dblock = static_cast<int>(blockIdx.x) & (kCombineDBlocks - 1);
    const int head = static_cast<int>(blockIdx.x) / kCombineDBlocks;
    extern __shared__ __align__(1024) char smem[];
    Raw16LocalCombineBody<kCombineWarps, true>(op, glse, NumSplit, cp_ctx_partial, cp_lse_partial,
                                               dblock, head, reinterpret_cast<float*>(smem));

    grid_bar_sync_modern(grid_counter, NumSplit);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int gen = atomicAdd(it_ctr, 1) + 1;
        cur_gen[0] = gen;
        if (ready_mc == 0) {
            if (ready_status != nullptr) {
                atomicCAS(ready_status, 0, 3);
            }
        } else {
            multimem_red_add_rel_sys(ready_mc, 1);
        }
    }
}

template <int QkHeadChunk>
static CUresult make_k_qk_map(void* kv, int seqlen_kv, CUtensorMap* map) {
    static_assert(QkHeadChunk == 32 || QkHeadChunk == 64, "unsupported QK chunk");
    uint64_t global_dim[4] = {256, 1, static_cast<uint64_t>(seqlen_kv), 1};
    uint64_t global_stride[3] = {512, 512, static_cast<uint64_t>(seqlen_kv) * 512};
    uint32_t box_dim[4] = {static_cast<uint32_t>(QkHeadChunk), 1, 64, 1};
    uint32_t elem_stride[4] = {1, 1, 1, 1};
    constexpr CUtensorMapSwizzle swizzle =
        QkHeadChunk == 32 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B;
    return cuTensorMapEncodeTiled(
        map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, kv, global_dim, global_stride, box_dim,
        elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

static int launch_hopper_ws_stage_switch_publish_dyn(void* q, void* k, void* v, void* op,
                                                     void* glse, void* cp_ctx_partial,
                                                     void* cp_lse_partial, void* grid_counter,
                                                     const void* slot, int max_kv, int cp_rank,
                                                     int cp, unsigned long long ready_mc,
                                                     void* ready_status, void* it_ctr,
                                                     void* cur_gen, cudaStream_t stream) {
    CUtensorMap k_map{}, q_map{}, v_map{};
    CUresult q_res = make_q_map(q, &q_map);
    CUresult k_res = make_k_qk_map<64>(k, max_kv, &k_map);
    CUresult v_res = make_kv_map(v, max_kv, &v_map);
    if (q_res != CUDA_SUCCESS || k_res != CUDA_SUCCESS || v_res != CUDA_SUCCESS) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const int smem = static_cast<int>(
        (kHeads * kHeadDim + 2 * kKvStages * kBlockN * kHeadDim + kBlockN * kHeads) * sizeof(bf16));
    auto kernel = Raw16HopperWsStageSwitchPublishDynKernel<kNumSplit128, 64>;
    cudaError_t status =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }

    int active_blocks_per_sm = 0;
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks_per_sm, kernel,
                                                           kWsThreads, smem);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    int device = 0;
    int sm_count = 0;
    status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    status = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    if (active_blocks_per_sm * sm_count < kNumSplit128) {
        return static_cast<int>(cudaErrorCooperativeLaunchTooLarge);
    }

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(kNumSplit128, 1, 1);
    config.blockDim = dim3(kWsThreads, 1, 1);
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    cudaLaunchAttribute launch_attr[1]{};
    launch_attr[0].id = cudaLaunchAttributeCooperative;
    launch_attr[0].val.cooperative = 1;
    config.attrs = launch_attr;
    config.numAttrs = 1;
    status = cudaLaunchKernelEx(
        &config, kernel, k_map, q_map, v_map, reinterpret_cast<bf16*>(op),
        reinterpret_cast<bf16*>(glse), reinterpret_cast<bf16*>(cp_ctx_partial),
        reinterpret_cast<float*>(cp_lse_partial), reinterpret_cast<uint32_t*>(grid_counter),
        reinterpret_cast<const int*>(slot), cp_rank, cp, ready_mc,
        reinterpret_cast<int*>(ready_status), reinterpret_cast<int*>(it_ctr),
        reinterpret_cast<int*>(cur_gen));
    return static_cast<int>(status);
}

}  // namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
using namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY;
extern "C" int NexN25ProCpAttnMegaDynLaunch(void* q, void* k, void* v, void* op, void* glse,
                                            void* cp_ctx_partial, void* cp_lse_partial,
                                            void* grid_counter, const void* slot, int max_kv,
                                            int cp_rank, unsigned long long ready_mc,
                                            void* ready_status, void* it_ctr, void* cur_gen,
                                            void* stream) {
    if (q == nullptr || k == nullptr || v == nullptr || op == nullptr || glse == nullptr ||
        cp_ctx_partial == nullptr || cp_lse_partial == nullptr || grid_counter == nullptr ||
        slot == nullptr || ready_mc == 0 || ready_status == nullptr || it_ctr == nullptr ||
        cur_gen == nullptr || max_kv < kBlockN || cp_rank < 0 || cp_rank >= 4) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    return launch_hopper_ws_stage_switch_publish_dyn(
        q, k, v, op, glse, cp_ctx_partial, cp_lse_partial, grid_counter, slot, max_kv, cp_rank, 4,
        ready_mc, ready_status, it_ctr, cur_gen, reinterpret_cast<cudaStream_t>(stream));
}

extern "C" const char* NexN25ProCpAttnMegaErrorString(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
