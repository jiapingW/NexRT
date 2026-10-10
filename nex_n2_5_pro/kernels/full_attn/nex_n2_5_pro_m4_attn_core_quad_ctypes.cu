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

// "Quad" M=4 causal attention partial: ONE KV pass for all 16 q-rows.
//
// The earlier per-row core launched the
// per-row WGMMA body on grid (ns, 4): every draft row re-streams the whole KV
// through TMA (~4x KV bytes) into an m64n8k16 whose B operand carries only 4
// real q-rows. This TU keeps that body's descriptor/softmax/epilogue math
// VERBATIM but views q_out4 [4 rows x 4 heads, 256] as ONE [16, 256] B operand:
// grid (ns, 1), K/V staged once per tile, and the 16 columns computed as TWO
// n8 column-groups over the same smem K/V (group g covers columns g*8..g*8+7 =
// draft rows 2g, 2g+1). KV traffic drops 4x; the QK/PV wgmma issue count per KV
// byte is unchanged (2 groups x n8 == the useful half of the old 4 launches).
//
// CAUSALITY (frozen staircase semantics): column c belongs to draft row j=c>>2
// with extent E_j = seqlen[1] + 1 + j. All rows walk row-3's split ranges
// (S3 = seqlen[1] + 4 keys); keys beyond a column's extent are masked AFTER the
// QK wgmma exactly like the existing tile-tail predicate: -1e30 into the
// running max, 0.0 into the P tile and row-sum. NUMERICS:
// per-(row,head) results are a split/tile-boundary REORDER of the shipped
// kernel's sums (rows 0..2 see different split boundaries than their per-row
// launches used), same arithmetic per element; gated against f64, against the
// shipped kernel, and against the E2E
// golden. The combine stage retains the BS=1 reduction order and workspace
// layout, including the Xeff epilogue used by the non-CP graph arm.
//
// All-masked-split guard: with shared row-3 ranges, a whole split can land past
// a lower row's extent (e.g. the last split at small S3/ns). Columns whose
// row-sum is zero write m=-1e30, l=0, acc=0 -- the exact kv_per==0 contract the
// combine already handles -- instead of 0 * (1/0) = NaN.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M4
#endif

#include "nex_n2_5_pro_m4_attn_core_common.cuh"

namespace nexrt::sm90::nex_n2_5_pro::M4 {
constexpr int QROWS = 16;           // 4 draft rows x 4 heads as one B operand
constexpr int QCHUNK = QROWS * 64;  // elements per 64-col TMA chunk

// Quad partial: one pass over row-3's split range, two n8 column-groups.
template <int STAGES>
__global__ __launch_bounds__(128) void m4_attn_partial_quad_dyn(
    const __grid_constant__ CUtensorMap Kmap, const __grid_constant__ CUtensorMap Vmap,
    const __grid_constant__ CUtensorMap Qmap, float* __restrict__ ws_m4, float* __restrict__ ws_l4,
    bf16* __restrict__ ws_acc4, const uint32_t* __restrict__ seqlen, int ns, float scaling) {
    const int Sbase = static_cast<int>(seqlen[1]);  // committed length; draft j at Sbase+j
    const int S3 = Sbase + 4;                       // row-3 extent = shared walk range
    const int split = blockIdx.x, tid = threadIdx.x, lane = tid & 31;
    const int warp = tid >> 5, gid = lane >> 2, tg = lane & 3;
    const float scale2 = scaling * 1.4426950408889634f;
    const int base0 = static_cast<int>((static_cast<long long>(S3) * split) / ns);
    const int end = static_cast<int>((static_cast<long long>(S3) * (split + 1)) / ns);
    const int kv_per = end - base0;
    const int ntiles = (kv_per + TILE - 1) / TILE;

    if (kv_per == 0) {
        // Empty split: publish the neutral contract for ALL 16 (row, head) columns.
        if (tid < MROWS * NH) {
            const int r = tid >> 2, h = tid & 3;
            ws_m4[(static_cast<size_t>(r) * ns + split) * NH + h] = -1.0e30f;
            ws_l4[(static_cast<size_t>(r) * ns + split) * NH + h] = 0.0f;
        }
        for (int r = 0; r < MROWS; ++r)
            for (int i = tid; i < NH * HD; i += 128)
                ws_acc4[(static_cast<size_t>(r) * ns + split) * NH * HD + i] =
                    __float2bfloat16(0.0f);
        return;
    }

    extern __shared__ __align__(1024) char smem[];
    constexpr int KVB = TILE * HD;
    bf16* Qsh = reinterpret_cast<bf16*>(smem);  // [4 chunks][16 rows][64]
    bf16* Ksh = Qsh + QROWS * HD;
    bf16* Vsh = Ksh + STAGES * KVB;
    bf16* Psh = Vsh + STAGES * KVB;  // two [TILE][8] blocks
    __shared__ __align__(8) uint64_t bq, bkv[STAGES];
    __shared__ float max_part[4][QROWS], sum_part[4][QROWS];

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
        mbar_expect(&bq, QROWS * HD * static_cast<int>(sizeof(bf16)));
#pragma unroll
        for (int j = 0; j < 4; ++j) tma_load(&Qmap, &bq, Qsh + j * QCHUNK, j * 64, 0);
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

    float out_frag[2][16];
#pragma unroll
    for (int g = 0; g < 2; ++g)
#pragma unroll
        for (int i = 0; i < 16; ++i) out_frag[g][i] = 0.0f;
    float m_i[2][2] = {{-1.0e30f, -1.0e30f}, {-1.0e30f, -1.0e30f}};
    float l_i[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    for (int tile = 0; tile < ntiles; ++tile) {
        const int stage = tile % STAGES;
        mbar_wait(&bkv[stage], (tile / STAGES) & 1);
        bf16* Kb = Ksh + stage * KVB;
        bf16* Vb = Vsh + stage * KVB;

        // QK^T for both column-groups over the SAME staged K tile.
        float score[2][4];
#pragma unroll
        for (int g = 0; g < 2; ++g)
#pragma unroll
            for (int i = 0; i < 4; ++i) score[g][i] = 0.0f;
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        const uint64_t kd = make_desc(Kb, 1, 64, 1);
        const uint64_t qd = make_desc(Qsh, 1, 64, 1);
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const uint64_t ko = (((k >> 2) * 8192 + (k & 3) * 32) >> 4);
            // 16-row chunks: chunk stride 2048 B; group 1 rows at +1024 B.
            const uint64_t qo = (((k >> 2) * 2048 + (k & 3) * 32) >> 4);
#pragma unroll
            for (int g = 0; g < 2; ++g) {
                asm volatile(
                    "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
                    "{%0,%1,%2,%3}, %4, %5, 1,1,1,0,0;\n"
                    : "+f"(score[g][0]), "+f"(score[g][1]), "+f"(score[g][2]), "+f"(score[g][3])
                    : "l"(kd + ko), "l"(qd + qo + (g ? 64u : 0u)));
            }
        }
        wg_commit();
        wg_wait<0>();

        // Per-column causal limits: column c = g*8 + (tg*2 | parity), draft row
        // j = c>>2, extent E_j = Sbase+1+j; key (base0 + tile*TILE + k) valid
        // iff k < lim_j. lim clamps identically to the old tile-tail predicate.
        const int tbase = base0 + tile * TILE;
        const int valid = min(TILE, kv_per - tile * TILE);
        const int k0 = warp * 16 + gid, k1 = k0 + 8;
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const int c0 = g * 8 + tg * 2, c1 = c0 + 1;
            const int j0 = c0 >> 2, j1 = c1 >> 2;
            const int lim0 = min(valid, Sbase + 1 + j0 - tbase);
            const int lim1 = min(valid, Sbase + 1 + j1 - tbase);
            if (k0 >= lim0) score[g][0] = -1.0e30f;
            if (k0 >= lim1) score[g][1] = -1.0e30f;
            if (k1 >= lim0) score[g][2] = -1.0e30f;
            if (k1 >= lim1) score[g][3] = -1.0e30f;
            float lm0 = fmaxf(score[g][0], score[g][2]);
            float lm1 = fmaxf(score[g][1], score[g][3]);
#pragma unroll
            for (int o = 4; o <= 16; o <<= 1) {
                lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffffu, lm0, o));
                lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffffu, lm1, o));
            }
            if (gid == 0) {
                max_part[warp][c0] = lm0;
                max_part[warp][c1] = lm1;
            }
        }
        partial_sync_128();
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const int c0 = g * 8 + tg * 2, c1 = c0 + 1;
            const int j0 = c0 >> 2, j1 = c1 >> 2;
            const int lim0 = min(valid, Sbase + 1 + j0 - tbase);
            const int lim1 = min(valid, Sbase + 1 + j1 - tbase);
            float tm0 = -1.0e30f, tm1 = -1.0e30f;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                tm0 = fmaxf(tm0, max_part[w][c0]);
                tm1 = fmaxf(tm1, max_part[w][c1]);
            }
            const float mn0 = fmaxf(m_i[g][0], tm0), mn1 = fmaxf(m_i[g][1], tm1);
            const float c0f = exp2f((m_i[g][0] - mn0) * scale2);
            const float c1f = exp2f((m_i[g][1] - mn1) * scale2);
            score[g][0] = k0 < lim0 ? exp2f((score[g][0] - mn0) * scale2) : 0.0f;
            score[g][1] = k0 < lim1 ? exp2f((score[g][1] - mn1) * scale2) : 0.0f;
            score[g][2] = k1 < lim0 ? exp2f((score[g][2] - mn0) * scale2) : 0.0f;
            score[g][3] = k1 < lim1 ? exp2f((score[g][3] - mn1) * scale2) : 0.0f;
            float s0 = score[g][0] + score[g][2], s1 = score[g][1] + score[g][3];
#pragma unroll
            for (int o = 4; o <= 16; o <<= 1) {
                s0 += __shfl_xor_sync(0xffffffffu, s0, o);
                s1 += __shfl_xor_sync(0xffffffffu, s1, o);
            }
            if (gid == 0) {
                sum_part[warp][c0] = s0;
                sum_part[warp][c1] = s1;
            }
        }
        partial_sync_128();
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const int c0 = g * 8 + tg * 2, c1 = c0 + 1;
            float ts0 = 0.0f, ts1 = 0.0f;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                ts0 += sum_part[w][c0];
                ts1 += sum_part[w][c1];
            }
            // Recompute the SAME mn as the exp pass (max_part is untouched
            // since that sync), so l/m updates and out_frag rescale agree.
            float mm0 = -1.0e30f, mm1 = -1.0e30f;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                mm0 = fmaxf(mm0, max_part[w][c0]);
                mm1 = fmaxf(mm1, max_part[w][c1]);
            }
            const float mn0 = fmaxf(m_i[g][0], mm0), mn1 = fmaxf(m_i[g][1], mm1);
            const float c0f = exp2f((m_i[g][0] - mn0) * scale2);
            const float c1f = exp2f((m_i[g][1] - mn1) * scale2);
            l_i[g][0] = l_i[g][0] * c0f + ts0;
            l_i[g][1] = l_i[g][1] * c1f + ts1;
            m_i[g][0] = mn0;
            m_i[g][1] = mn1;
#pragma unroll
            for (int i = 0; i < 16; ++i) out_frag[g][i] *= (i & 1) ? c1f : c0f;
            // Stage this group's probabilities into its P block.
            bf16* Pg = Psh + g * TILE * 8;
            Pg[(k0 >> 3) * 64 + (tg * 2) * 8 + (k0 & 7)] = static_cast<bf16>(score[g][0]);
            Pg[(k0 >> 3) * 64 + (tg * 2 + 1) * 8 + (k0 & 7)] = static_cast<bf16>(score[g][1]);
            Pg[(k1 >> 3) * 64 + (tg * 2) * 8 + (k1 & 7)] = static_cast<bf16>(score[g][2]);
            Pg[(k1 >> 3) * 64 + (tg * 2 + 1) * 8 + (k1 & 7)] = static_cast<bf16>(score[g][3]);
        }
        partial_sync_128();

        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        const uint64_t vd = make_desc(Vb, 256, 64, 1);
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const uint64_t pd = make_desc(Psh + g * TILE * 8, 8, 256, 0);
#pragma unroll
            for (int i8 = 0; i8 < 4; ++i8) {
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const uint64_t vo = ((i8 * 8192 + kk * 2048) >> 4);
                    const uint64_t po = ((kk * 2) * 64 * 2 >> 4);
                    asm volatile(
                        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
                        "{%0,%1,%2,%3}, %4,%5, 1,1,1,1,0;\n"
                        : "+f"(out_frag[g][i8 * 4]), "+f"(out_frag[g][i8 * 4 + 1]),
                          "+f"(out_frag[g][i8 * 4 + 2]), "+f"(out_frag[g][i8 * 4 + 3])
                        : "l"(vd + vo), "l"(pd + po));
                }
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

// Epilogue: column c -> draft row r=c>>2, head h=c&3; l==0 columns publish
// the empty-split contract (guard against 0 * inf).
#pragma unroll
    for (int g = 0; g < 2; ++g) {
        const int c0 = g * 8 + tg * 2, c1 = c0 + 1;
        const float inv_l0 = l_i[g][0] > 0.0f ? 1.0f / l_i[g][0] : 0.0f;
        const float inv_l1 = l_i[g][1] > 0.0f ? 1.0f / l_i[g][1] : 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) out_frag[g][i] *= (i & 1) ? inv_l1 : inv_l0;
        const int r0 = c0 >> 2, h0 = c0 & 3, r1 = c1 >> 2, h1 = c1 & 3;
        float* wm0 = ws_m4 + (static_cast<size_t>(r0) * ns + split) * NH + h0;
        float* wl0 = ws_l4 + (static_cast<size_t>(r0) * ns + split) * NH + h0;
        float* wm1 = ws_m4 + (static_cast<size_t>(r1) * ns + split) * NH + h1;
        float* wl1 = ws_l4 + (static_cast<size_t>(r1) * ns + split) * NH + h1;
        if (gid == 0) {
            *wm0 = m_i[g][0];
            *wl0 = l_i[g][0];
            *wm1 = m_i[g][1];
            *wl1 = l_i[g][1];
        }
        bf16* wa0 = ws_acc4 + ((static_cast<size_t>(r0) * ns + split) * NH + h0) * HD;
        bf16* wa1 = ws_acc4 + ((static_cast<size_t>(r1) * ns + split) * NH + h1) * HD;
#pragma unroll
        for (int i8 = 0; i8 < 4; ++i8) {
            const int d0 = i8 * 64 + warp * 16 + gid, d1 = d0 + 8;
            wa0[d0] = static_cast<bf16>(out_frag[g][i8 * 4]);
            wa0[d1] = static_cast<bf16>(out_frag[g][i8 * 4 + 2]);
            wa1[d0] = static_cast<bf16>(out_frag[g][i8 * 4 + 1]);
            wa1[d1] = static_cast<bf16>(out_frag[g][i8 * 4 + 3]);
        }
    }
}

static int quad_smem_bytes() {
    constexpr int STAGES = WGMMA_STAGES;
    return (QROWS * HD + 2 * STAGES * TILE * HD + TILE * QROWS) * static_cast<int>(sizeof(bf16)) +
           1024;
}

extern "C" int NexN25ProM4AttnCoreQuadDynLaunch(const void* q_out4, const void* K, const void* V,
                                                const void* seqlen, int ns, float scaling,
                                                void* wm4, void* wl4, void* wa4, void* ctx4,
                                                void* stream) {
    CUtensorMap Kmap = encode_map(const_cast<void*>(K), KV_CAPACITY, HD, TILE, 64);
    CUtensorMap Vmap = encode_map(const_cast<void*>(V), KV_CAPACITY, HD, TILE, 64);
    CUtensorMap Qmap = encode_map(const_cast<void*>(q_out4), QROWS, HD, QROWS, 64);
    constexpr int STAGES = WGMMA_STAGES;
    const int smem = quad_smem_bytes();
    static bool initialized = false;
    if (!initialized) {
        cudaFuncSetAttribute(m4_attn_partial_quad_dyn<STAGES>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        initialized = true;
    }
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    m4_attn_partial_quad_dyn<STAGES><<<dim3(ns), 128, smem, st>>>(
        Kmap, Vmap, Qmap, static_cast<float*>(wm4), static_cast<float*>(wl4),
        static_cast<bf16*>(wa4), static_cast<const uint32_t*>(seqlen), ns, scaling);
    constexpr int DT = 8, BLK = WGMMA_COMBINE_THREADS;
    M4AttnCombineKernel<DT, BLK>
        <<<dim3(NH * HD / DT, MROWS), BLK, ns * DT * sizeof(bf16) + 2 * ns * sizeof(float), st>>>(
            static_cast<const float*>(wm4), static_cast<const float*>(wl4),
            static_cast<const bf16*>(wa4), static_cast<bf16*>(ctx4), ns, scaling);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int NexN25ProM4AttnCoreQuadDynXeffLaunch(const void* q_out4, const void* K,
                                                    const void* V, const void* seqlen, int ns,
                                                    float scaling, void* wm4, void* wl4, void* wa4,
                                                    void* ctx4, const void* gate4, void* xeff4,
                                                    void* stream) {
    CUtensorMap Kmap = encode_map(const_cast<void*>(K), KV_CAPACITY, HD, TILE, 64);
    CUtensorMap Vmap = encode_map(const_cast<void*>(V), KV_CAPACITY, HD, TILE, 64);
    CUtensorMap Qmap = encode_map(const_cast<void*>(q_out4), QROWS, HD, QROWS, 64);
    constexpr int STAGES = WGMMA_STAGES;
    const int smem = quad_smem_bytes();
    static bool initialized = false;
    if (!initialized) {
        cudaFuncSetAttribute(m4_attn_partial_quad_dyn<STAGES>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        initialized = true;
    }
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    m4_attn_partial_quad_dyn<STAGES><<<dim3(ns), 128, smem, st>>>(
        Kmap, Vmap, Qmap, static_cast<float*>(wm4), static_cast<float*>(wl4),
        static_cast<bf16*>(wa4), static_cast<const uint32_t*>(seqlen), ns, scaling);
    constexpr int DT = 8, BLK = WGMMA_COMBINE_THREADS;
    M4AttnCombineXeffKernel<DT, BLK>
        <<<dim3(NH * HD / DT, MROWS), BLK, ns * DT * sizeof(bf16) + 2 * ns * sizeof(float), st>>>(
            static_cast<const float*>(wm4), static_cast<const float*>(wl4),
            static_cast<const bf16*>(wa4), static_cast<bf16*>(ctx4),
            static_cast<const bf16*>(gate4), static_cast<bf16*>(xeff4), ns, scaling);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M4