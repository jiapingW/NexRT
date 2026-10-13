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

// True-four-head TMA + WGMMA attention partial and combine for Hopper.
// WGMMA BF16 starts at N=8. The Q tensor map describes four real rows but
// loads an eight-row box; TMA's specified OOB behavior zero-fills heads 4..7.
#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY bs1
#endif

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY {
using bf16 = __nv_bfloat16;
constexpr int HD = 256;
constexpr int NH = 4;
constexpr int TNH = 8;
constexpr int TILE = 64;
constexpr int WGMMA_STAGES = 2;
// This descriptor extent must exactly match FullAttn.k_cache/v_cache.  Live S
// remains device-driven and may be much smaller; TMA handles the last partial
// tile against this fixed allocation extent.
constexpr int KV_CAPACITY = 256 * 1024 + 1024;
constexpr uint64_t TMA_EVICT_FIRST = 0x12f0000000000000ULL;

__device__ __forceinline__ uint64_t cvta(const void* p) {
    return static_cast<uint64_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ uint64_t make_desc(const void* p, uint32_t lbo, uint32_t sbo,
                                              uint32_t swizzle) {
    const uint64_t a = (cvta(p) >> 4) & 0x3fffull;
    return a | (static_cast<uint64_t>(lbo & 0x3fff) << 16) |
           (static_cast<uint64_t>(sbo & 0x3fff) << 32) | (static_cast<uint64_t>(swizzle & 3) << 62);
}
__device__ __forceinline__ void wg_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void wg_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(static_cast<uint32_t>(cvta(b))),
                 "r"(count));
}
__device__ __forceinline__ void mbar_expect(uint64_t* b, int bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(static_cast<uint32_t>(cvta(b))),
        "r"(bytes));
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, int phase) {
    asm volatile(
        "{\n.reg .pred p;\nL%=:\n"
        "mbarrier.try_wait.parity.shared.b64 p,[%0],%1;\n"
        "@!p bra L%=;\n}\n" ::"r"(static_cast<uint32_t>(cvta(b))),
        "r"(phase));
}
__device__ __forceinline__ void tma_load(const CUtensorMap* map, uint64_t* bar, void* dst, int c0,
                                         int c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(
            static_cast<uint32_t>(cvta(dst))),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1),
        "r"(static_cast<uint32_t>(cvta(bar)))
        : "memory");
}
__device__ __forceinline__ void tma_load_kv(const CUtensorMap* map, uint64_t* bar, void* dst,
                                            int c0, int c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1, {%2, %3}], [%4], %5;\n" ::"r"(static_cast<uint32_t>(cvta(dst))),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1),
        "r"(static_cast<uint32_t>(cvta(bar))), "l"(TMA_EVICT_FIRST)
        : "memory");
}
__device__ __forceinline__ void prefetch_map(const CUtensorMap* map) {
    asm volatile("prefetch.tensormap [%0];\n" ::"l"(reinterpret_cast<uint64_t>(map)));
}
__device__ __forceinline__ void partial_sync_128() {
    asm volatile("bar.sync 1, 128;\n" ::: "memory");
}
__device__ __forceinline__ void grid_bar_sync_modern(uint32_t* counter, int num_blocks) {
    constexpr uint32_t TAG = 0x80000000u;
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t increment =
            blockIdx.x == 0 ? TAG - static_cast<uint32_t>(num_blocks - 1) : 1u;
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
        } while (((current ^ old) & TAG) == 0u);
    }
    __syncthreads();
}

template <int STAGES>
__device__ __forceinline__ void wgmma_fd4_partial_body(
    const CUtensorMap& Kmap, const CUtensorMap& Vmap, const CUtensorMap& Qmap,
    float* __restrict__ ws_m, float* __restrict__ ws_l, bf16* __restrict__ ws_acc, int S, int ns,
    float scaling) {
    const int split = blockIdx.x, tid = threadIdx.x, lane = tid & 31;
    const int warp = tid >> 5, gid = lane >> 2, tg = lane & 3;
    const float scale2 = scaling * 1.4426950408889634f;
    const int base0 = static_cast<int>((static_cast<long long>(S) * split) / ns);
    const int end = static_cast<int>((static_cast<long long>(S) * (split + 1)) / ns);
    const int kv_per = end - base0;
    const int ntiles = (kv_per + TILE - 1) / TILE;

    // A fixed Graph may keep the split count larger than very short live S.
    // Empty split CTAs publish the exact flash-combine identity element.
    if (kv_per == 0) {
        if (tid < NH) {
            ws_m[split * NH + tid] = -1.0e30f;
            ws_l[split * NH + tid] = 0.0f;
        }
        for (int i = tid; i < NH * HD; i += 128)
            ws_acc[(static_cast<size_t>(split) * NH * HD) + i] = __float2bfloat16(0.0f);
        return;
    }

    extern __shared__ __align__(1024) char smem[];
    constexpr int KVB = TILE * HD;
    bf16* Qsh = reinterpret_cast<bf16*>(smem);  // [TNH][HD], swizzle128
    bf16* Ksh = Qsh + TNH * HD;                 // [STAGES][TILE][HD]
    bf16* Vsh = Ksh + STAGES * KVB;
    bf16* Psh = Vsh + STAGES * KVB;  // [TILE][TNH]
    __shared__ __align__(8) uint64_t bq, bkv[STAGES];
    __shared__ float max_part[4][TNH], sum_part[4][TNH];

    if (tid == 0) {
        mbar_init(&bq, 1);
#pragma unroll
        for (int s = 0; s < STAGES; ++s) mbar_init(&bkv[s], 1);
        prefetch_map(&Kmap);
        prefetch_map(&Vmap);
        prefetch_map(&Qmap);
    }
    partial_sync_128();
    if (tid == 0) {
        mbar_expect(&bq, TNH * HD * static_cast<int>(sizeof(bf16)));
#pragma unroll
        for (int j = 0; j < 4; ++j) tma_load(&Qmap, &bq, Qsh + j * TNH * 64, j * 64, 0);
        for (int t = 0; t < min(STAGES, ntiles); ++t) {
            mbar_expect(&bkv[t], 2 * KVB * static_cast<int>(sizeof(bf16)));
            const int key = base0 + t * TILE;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                tma_load_kv(&Kmap, &bkv[t], Ksh + t * KVB + j * 4096, j * 64, key);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                tma_load_kv(&Vmap, &bkv[t], Vsh + t * KVB + j * 4096, j * 64, key);
        }
    }
    mbar_wait(&bq, 0);

    float out_frag[16];
#pragma unroll
    for (float& x : out_frag) x = 0.0f;
    float m_i[2] = {-1.0e30f, -1.0e30f};
    float l_i[2] = {0.0f, 0.0f};

    for (int tile = 0; tile < ntiles; ++tile) {
        const int stage = tile % STAGES;
        mbar_wait(&bkv[stage], (tile / STAGES) & 1);
        bf16* Kb = Ksh + stage * KVB;
        bf16* Vb = Vsh + stage * KVB;

        float score[4] = {0, 0, 0, 0};
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        const uint64_t kd = make_desc(Kb, 1, 64, 1);
        const uint64_t qd = make_desc(Qsh, 1, 64, 1);
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const uint64_t ko = (((k >> 2) * 8192 + (k & 3) * 32) >> 4);
            const uint64_t qo = (((k >> 2) * 1024 + (k & 3) * 32) >> 4);
            asm volatile(
                "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
                "{%0,%1,%2,%3}, %4, %5, 1,1,1,0,0;\n"
                : "+f"(score[0]), "+f"(score[1]), "+f"(score[2]), "+f"(score[3])
                : "l"(kd + ko), "l"(qd + qo));
        }
        wg_commit();
        wg_wait<0>();

        const int valid = min(TILE, kv_per - tile * TILE);
        const int k0 = warp * 16 + gid, k1 = k0 + 8;
        if (k0 >= valid) {
            score[0] = -1.0e30f;
            score[1] = -1.0e30f;
        }
        if (k1 >= valid) {
            score[2] = -1.0e30f;
            score[3] = -1.0e30f;
        }
        float lm0 = fmaxf(score[0], score[2]);
        float lm1 = fmaxf(score[1], score[3]);
#pragma unroll
        for (int o = 4; o <= 16; o <<= 1) {
            lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffffu, lm0, o));
            lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffffu, lm1, o));
        }
        if (gid == 0) {
            max_part[warp][tg * 2] = lm0;
            max_part[warp][tg * 2 + 1] = lm1;
        }
        partial_sync_128();
        float tm0 = -1.0e30f, tm1 = -1.0e30f;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            tm0 = fmaxf(tm0, max_part[w][tg * 2]);
            tm1 = fmaxf(tm1, max_part[w][tg * 2 + 1]);
        }
        const float mn0 = fmaxf(m_i[0], tm0), mn1 = fmaxf(m_i[1], tm1);
        const float c0 = exp2f((m_i[0] - mn0) * scale2);
        const float c1 = exp2f((m_i[1] - mn1) * scale2);
        score[0] = k0 < valid ? exp2f((score[0] - mn0) * scale2) : 0.0f;
        score[1] = k0 < valid ? exp2f((score[1] - mn1) * scale2) : 0.0f;
        score[2] = k1 < valid ? exp2f((score[2] - mn0) * scale2) : 0.0f;
        score[3] = k1 < valid ? exp2f((score[3] - mn1) * scale2) : 0.0f;
        float s0 = score[0] + score[2], s1 = score[1] + score[3];
#pragma unroll
        for (int o = 4; o <= 16; o <<= 1) {
            s0 += __shfl_xor_sync(0xffffffffu, s0, o);
            s1 += __shfl_xor_sync(0xffffffffu, s1, o);
        }
        if (gid == 0) {
            sum_part[warp][tg * 2] = s0;
            sum_part[warp][tg * 2 + 1] = s1;
        }
        partial_sync_128();
        float ts0 = 0.0f, ts1 = 0.0f;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            ts0 += sum_part[w][tg * 2];
            ts1 += sum_part[w][tg * 2 + 1];
        }
        l_i[0] = l_i[0] * c0 + ts0;
        l_i[1] = l_i[1] * c1 + ts1;
        m_i[0] = mn0;
        m_i[1] = mn1;

        Psh[(k0 >> 3) * 64 + (tg * 2) * 8 + (k0 & 7)] = static_cast<bf16>(score[0]);
        Psh[(k0 >> 3) * 64 + (tg * 2 + 1) * 8 + (k0 & 7)] = static_cast<bf16>(score[1]);
        Psh[(k1 >> 3) * 64 + (tg * 2) * 8 + (k1 & 7)] = static_cast<bf16>(score[2]);
        Psh[(k1 >> 3) * 64 + (tg * 2 + 1) * 8 + (k1 & 7)] = static_cast<bf16>(score[3]);
#pragma unroll
        for (int i = 0; i < 16; ++i) out_frag[i] *= (i & 1) ? c1 : c0;
        partial_sync_128();

        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        const uint64_t vd = make_desc(Vb, 256, 64, 1);
        const uint64_t pd = make_desc(Psh, 8, 256, 0);
#pragma unroll
        for (int i8 = 0; i8 < 4; ++i8) {
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const uint64_t vo = ((i8 * 8192 + kk * 2048) >> 4);
                const uint64_t po = ((kk * 2) * 64 * 2 >> 4);
                asm volatile(
                    "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
                    "{%0,%1,%2,%3}, %4,%5, 1,1,1,1,0;\n"
                    : "+f"(out_frag[i8 * 4]), "+f"(out_frag[i8 * 4 + 1]),
                      "+f"(out_frag[i8 * 4 + 2]), "+f"(out_frag[i8 * 4 + 3])
                    : "l"(vd + vo), "l"(pd + po));
            }
        }
        wg_commit();
        wg_wait<0>();
        partial_sync_128();

        if (tid == 0 && tile + STAGES < ntiles) {
            const int next = tile + STAGES, next_stage = next % STAGES;
            const int key = base0 + next * TILE;
            mbar_expect(&bkv[next_stage], 2 * KVB * static_cast<int>(sizeof(bf16)));
#pragma unroll
            for (int j = 0; j < 4; ++j)
                tma_load_kv(&Kmap, &bkv[next_stage], Ksh + next_stage * KVB + j * 4096, j * 64,
                            key);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                tma_load_kv(&Vmap, &bkv[next_stage], Vsh + next_stage * KVB + j * 4096, j * 64,
                            key);
        }
    }

    const int h0 = tg * 2, h1 = h0 + 1;
    const float inv_l0 = 1.0f / l_i[0], inv_l1 = 1.0f / l_i[1];
#pragma unroll
    for (int i = 0; i < 16; ++i) out_frag[i] *= (i & 1) ? inv_l1 : inv_l0;
    if (gid == 0) {
        if (h0 < NH) {
            ws_m[split * NH + h0] = m_i[0];
            ws_l[split * NH + h0] = l_i[0];
        }
        if (h1 < NH) {
            ws_m[split * NH + h1] = m_i[1];
            ws_l[split * NH + h1] = l_i[1];
        }
    }
#pragma unroll
    for (int i8 = 0; i8 < 4; ++i8) {
        const int d0 = i8 * 64 + warp * 16 + gid, d1 = d0 + 8;
        if (h0 < NH) {
            ws_acc[(static_cast<size_t>(split) * NH + h0) * HD + d0] =
                static_cast<bf16>(out_frag[i8 * 4]);
            ws_acc[(static_cast<size_t>(split) * NH + h0) * HD + d1] =
                static_cast<bf16>(out_frag[i8 * 4 + 2]);
        }
        if (h1 < NH) {
            ws_acc[(static_cast<size_t>(split) * NH + h1) * HD + d0] =
                static_cast<bf16>(out_frag[i8 * 4 + 1]);
            ws_acc[(static_cast<size_t>(split) * NH + h1) * HD + d1] =
                static_cast<bf16>(out_frag[i8 * 4 + 3]);
        }
    }
}

template <int DT, int BLK>
__device__ __forceinline__ void wgmma_fd4_combine_body(const float* __restrict__ ws_m,
                                                       const float* __restrict__ ws_l,
                                                       const bf16* __restrict__ ws_acc,
                                                       bf16* __restrict__ out, int ns,
                                                       float scaling) {
    const int o0 = blockIdx.x * DT, h = o0 / HD, d0 = o0 % HD, tid = threadIdx.x;
    const float scale2 = scaling * 1.4426950408889634f;
    extern __shared__ bf16 sm[];
    __shared__ float red[BLK], den_red[BLK];
    for (int k = tid; k < ns * DT; k += BLK) {
        const int s = k / DT;
        sm[k] = ws_acc[(static_cast<size_t>(s) * NH + h) * HD + d0 + k % DT];
    }
    float lm = -1.0e30f;
    for (int s = tid; s < ns; s += BLK) lm = fmaxf(lm, ws_m[s * NH + h]);
#pragma unroll
    for (int o = 16; o; o >>= 1) lm = fmaxf(lm, __shfl_xor_sync(0xffffffffu, lm, o));
    if ((tid & 31) == 0) red[tid >> 5] = lm;
    __syncthreads();
    float gm = -1.0e30f;
#pragma unroll
    for (int i = 0; i < BLK / 32; ++i) gm = fmaxf(gm, red[i]);
    constexpr int GROUPS = BLK / DT;
    const int d = tid % DT, group = tid / DT;
    float num = 0.0f, den = 0.0f;
    for (int s = group; s < ns; s += GROUPS) {
        const float w = exp2f((ws_m[s * NH + h] - gm) * scale2) * ws_l[s * NH + h];
        num += w * static_cast<float>(sm[s * DT + d]);
        den += w;
    }
    red[tid] = num;
    den_red[tid] = den;
    __syncthreads();
    if (tid < DT) {
        float n = 0.0f, z = 0.0f;
#pragma unroll
        for (int g = 0; g < GROUPS; ++g) {
            n += red[g * DT + tid];
            z += den_red[g * DT + tid];
        }
        out[o0 + tid] = static_cast<bf16>(n / (z + 1.0e-20f));
    }
}

template <int STAGES>
__global__ __launch_bounds__(256) void wgmma_fd4_mega(
    const __grid_constant__ CUtensorMap Kmap, const __grid_constant__ CUtensorMap Vmap,
    const __grid_constant__ CUtensorMap Qmap, float* __restrict__ ws_m, float* __restrict__ ws_l,
    bf16* __restrict__ ws_acc, bf16* __restrict__ out, int S, const uint32_t* __restrict__ seqlen,
    int ns, float scaling, uint32_t* barrier_counter) {
    if (seqlen) S = static_cast<int>(seqlen[1]) + 1;
    if (threadIdx.x < 128)
        wgmma_fd4_partial_body<STAGES>(Kmap, Vmap, Qmap, ws_m, ws_l, ws_acc, S, ns, scaling);
    __syncthreads();
    grid_bar_sync_modern(barrier_counter, ns);
    if (blockIdx.x < NH * HD / 8)
        wgmma_fd4_combine_body<8, 256>(ws_m, ws_l, ws_acc, out, ns, scaling);
}

static CUtensorMap encode_map(void* ptr, int outer, int inner, int box_outer, int box_inner) {
    CUtensorMap map{};
    uint64_t dims[2] = {static_cast<uint64_t>(inner), static_cast<uint64_t>(outer)};
    uint64_t strides[1] = {static_cast<uint64_t>(inner) * sizeof(bf16)};
    uint32_t box[2] = {static_cast<uint32_t>(box_inner), static_cast<uint32_t>(box_outer)};
    uint32_t elem[2] = {1, 1};
    cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, dims, strides, box, elem,
                           CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                           CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return map;
}

// Smaller Hopper devices cannot keep all 128 partial CTAs resident for the
// cooperative barrier. Keep the same splits and reduction order, but use the
// stream dependency between two launches as the partial->combine barrier.
template <int STAGES>
__global__ __launch_bounds__(128) void wgmma_fd4_partial(const __grid_constant__ CUtensorMap Kmap,
                                                         const __grid_constant__ CUtensorMap Vmap,
                                                         const __grid_constant__ CUtensorMap Qmap,
                                                         float* ws_m, float* ws_l, bf16* ws_acc,
                                                         int S, const uint32_t* seqlen, int ns,
                                                         float scaling) {
    if (seqlen) S = static_cast<int>(seqlen[1]) + 1;
    wgmma_fd4_partial_body<STAGES>(Kmap, Vmap, Qmap, ws_m, ws_l, ws_acc, S, ns, scaling);
}

__global__ __launch_bounds__(256) void wgmma_fd4_combine(const float* ws_m, const float* ws_l,
                                                         const bf16* ws_acc, bf16* out, int ns,
                                                         float scaling) {
    wgmma_fd4_combine_body<8, 256>(ws_m, ws_l, ws_acc, out, ns, scaling);
}

// Cooperative partial->combine stage switch.  The caller owns the reset-free
// counter so separately executable blocks/requests never alias barrier state.
static int launch_wgmma_fd4_mega(const void* Q, const void* K, const void* V, int S,
                                 const void* seqlen, int ns, float scaling, void* wm, void* wl,
                                 void* wa, void* ctx, void* barrier_counter, void* stream) {
    CUtensorMap Kmap = encode_map(const_cast<void*>(K), KV_CAPACITY, HD, TILE, 64);
    CUtensorMap Vmap = encode_map(const_cast<void*>(V), KV_CAPACITY, HD, TILE, 64);
    CUtensorMap Qmap = encode_map(const_cast<void*>(Q), NH, HD, TNH, 64);
    constexpr int STAGES = WGMMA_STAGES;
    constexpr int smem =
        (TNH * HD + 2 * STAGES * TILE * HD + TILE * TNH) * static_cast<int>(sizeof(bf16)) + 1024;
    static bool initialized = false;
    static int cooperative_capacity = 0;
    if (!initialized) {
        cudaError_t error = cudaFuncSetAttribute(wgmma_fd4_mega<STAGES>,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (error != cudaSuccess) return static_cast<int>(error);
        int active_per_sm = 0, sm_count = 0, device = 0;
        error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_per_sm,
                                                              wgmma_fd4_mega<STAGES>, 256, smem);
        if (error != cudaSuccess) return static_cast<int>(error);
        error = cudaGetDevice(&device);
        if (error != cudaSuccess) return static_cast<int>(error);
        error = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
        if (error != cudaSuccess) return static_cast<int>(error);
        cooperative_capacity = active_per_sm * sm_count;
        if (ns > cooperative_capacity) {
            error = cudaFuncSetAttribute(wgmma_fd4_partial<STAGES>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            if (error != cudaSuccess) return static_cast<int>(error);
        }
        initialized = true;
    }
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    if (!ctx) return static_cast<int>(cudaErrorInvalidDevicePointer);
    if (ns != NH * HD / 8) return static_cast<int>(cudaErrorInvalidConfiguration);
    float* wm_p = static_cast<float*>(wm);
    float* wl_p = static_cast<float*>(wl);
    bf16* wa_p = static_cast<bf16*>(wa);
    bf16* out_p = static_cast<bf16*>(ctx);
    const uint32_t* seqlen_p = static_cast<const uint32_t*>(seqlen);
    uint32_t* counter_p = static_cast<uint32_t*>(barrier_counter);
    if (!counter_p) return static_cast<int>(cudaErrorInvalidDevicePointer);
    if (ns > cooperative_capacity) {
        wgmma_fd4_partial<STAGES>
            <<<ns, 128, smem, st>>>(Kmap, Vmap, Qmap, wm_p, wl_p, wa_p, S, seqlen_p, ns, scaling);
        const cudaError_t partial_error = cudaGetLastError();
        if (partial_error != cudaSuccess) return static_cast<int>(partial_error);
        wgmma_fd4_combine<<<NH * HD / 8, 256, ns * 8 * sizeof(bf16), st>>>(wm_p, wl_p, wa_p, out_p,
                                                                           ns, scaling);
        return static_cast<int>(cudaGetLastError());
    }
    void* args[] = {
        &Kmap, &Vmap, &Qmap, &wm_p, &wl_p, &wa_p, &out_p, &S, &seqlen_p, &ns, &scaling, &counter_p,
    };
    const cudaError_t error = cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(wgmma_fd4_mega<STAGES>), dim3(ns), dim3(256), args, smem, st);
    return static_cast<int>(error);
}

#ifndef NEXRT_MTP_RUNTIME_ONLY
extern "C" int NexN25ProAttnWgmmaLaunch(const void* Q, const void* K, const void* V, int S,
                                        void* wm, void* wl, void* wa, void* ctx,
                                        void* barrier_counter, void* stream) {
    return launch_wgmma_fd4_mega(Q, K, V, S, nullptr, 128, 0.0625f, wm, wl, wa, ctx,
                                 barrier_counter, stream);
}
#endif

extern "C" int NexN25ProAttnWgmmaDynLaunch(const void* Q, const void* K, const void* V,
                                           const void* seqlen, void* wm, void* wl, void* wa,
                                           void* ctx, void* barrier_counter, void* stream) {
    return launch_wgmma_fd4_mega(Q, K, V, 0, seqlen, 128, 0.0625f, wm, wl, wa, ctx, barrier_counter,
                                 stream);
}

}  // namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
