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

// CP M=4 Q-as-A quad attention PARTIAL (shard-local).
//
// CP GQA is 16 q-heads : 1 kv-head per rank (attn_cp.py NH_CP), so all
// 4 drafts x 16 heads = 64 q-rows share one K tile and fill WGMMA's M=64
// EXACTLY: A = q4 viewed as [64, 256] (row index r = draft*16 + head),
// B = the staged K tile, D = scores[64 q, 64 keys] via m64n64k16 -- the
// "no transpose" orientation. The online-softmax key-reduction then lives
// along each thread-quad's own fragment columns: row max/sum = 16 in-thread
// regs + two shfl_xor hops, NO cross-warp smem/barrier dance.
//
// PV keeps the PROVEN descriptor orientation (vd-formula A over V d-blocks,
// pd-formula B over Psh col-groups): the out accumulator [64 q x 256 d] fp32
// is 128 regs/thread at 128 threads REGARDLESS of orientation -- that
// register arithmetic rules out a keys-as-A m64n64 four-row form -- so this
// kernel runs 256 threads: warpgroup 0 owns QK + softmax + P/c publication +
// d-blocks {0,1}; warpgroup 1 consumes Psh and owns d-blocks {2,3}. 64 out
// regs per thread each.
//
// CAUSAL WINDOWS (frozen staircase, shard-local): row r (draft j = r>>4) has
// total_j = slots4[j*2+1] + 1, base_j = total_j/cp, and this rank's window
//   [off_j, off_j + loc_j), off_j = cp_rank*base_j,
//   loc_j = (cp_rank < cp-1) ? base_j : total_j - (cp-1)*base_j
// -- the same shard-window arithmetic as the bs=1 CP path.
// The split walk covers the UNION [u0, u1) of the four windows (they differ
// by <= a few keys); every score is masked by its row's window on BOTH the
// max input and the exp/P-store predicate. cp=1 degenerates to the full
// staircase; all-masked columns publish m=-1e30, l=0, acc=0.
//
// NUMERICS: reorder vs the per-row/pair CP cores
// (quad-shuffle max/sum tree + union tiling differ); gated vs f64 + class
// parity + perturbation tests.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M4
#endif

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#ifndef CPQ_STAGES
#define CPQ_STAGES 2
#endif

namespace nexrt::sm90::nex_n2_5_pro::M4 {
using bf16 = __nv_bfloat16;
constexpr int HD = 256;
constexpr int ROWS = 64;  // 4 drafts x 16 heads
constexpr int TILE = 64;
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
        "mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1, {%2, %3}], [%4], %5;\n" ::"r"(static_cast<uint32_t>(cvta(dst))),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1),
        "r"(static_cast<uint32_t>(cvta(bar))), "l"(TMA_EVICT_FIRST)
        : "memory");
}
__device__ __forceinline__ void prefetch_map(const CUtensorMap* map) {
    asm volatile("prefetch.tensormap [%0];\n" ::"l"(reinterpret_cast<uint64_t>(map)));
}
__device__ __forceinline__ void bar_sync(int bar, int threads) {
    asm volatile("bar.sync %0, %1;\n" ::"r"(bar), "r"(threads) : "memory");
}

// m64n64k16 bf16 -> fp32, D in 32 regs: groups i=0..7 of 4 regs covering
// rows {r0, r0+8} x cols {c0+8i, c0+8i+1}, r0 = warp*16 + (lane>>2),
// c0 = (lane&3)*2 (PTX wgmma D fragment layout).
__device__ __forceinline__ void wgmma_n64(float (&d)[32], uint64_t a, uint64_t b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, 1,1,1,0,0;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
          "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
          "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
          "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(a), "l"(b));
}
__device__ __forceinline__ void wgmma_n8_acc(float (&d)[4], uint64_t a, uint64_t b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3}, %4, %5, 1,1,1,1,0;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "l"(a), "l"(b));
}

// One partial: grid (ns), 256 threads (wg0 = softmax owner + d-blocks 0/1,
// wg1 = d-blocks 2/3). ws_m/ws_l: [ROWS, ns] fp32. ws_acc: [ROWS, ns, HD] bf16.
template <int STAGES>
__global__ __launch_bounds__(256) void m4_cp_quad_partial(
    const __grid_constant__ CUtensorMap Kmap, const __grid_constant__ CUtensorMap Vmap,
    const __grid_constant__ CUtensorMap Qmap, const int* __restrict__ slots4, int cp_rank, int cp,
    float* __restrict__ ws_m, float* __restrict__ ws_l, bf16* __restrict__ ws_acc, int ns,
    float scaling) {
    const int split = blockIdx.x, tid = threadIdx.x;
    const int wg = tid >> 7, wtid = tid & 127;
    const int lane = tid & 31, warp = wtid >> 5, gid = lane >> 2, tg = lane & 3;
    const float scale2 = scaling * 1.4426950408889634f;

    // Per-draft windows (shard-window arithmetic).
    int off[4], loc[4];
    int u0 = INT_MAX, u1 = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int total = slots4[j * 2 + 1] + 1;
        const int base = total / cp;
        off[j] = cp_rank * base;
        loc[j] = (cp_rank < cp - 1) ? base : total - (cp - 1) * base;
        u0 = min(u0, off[j]);
        u1 = max(u1, off[j] + loc[j]);
    }
    const int span = u1 - u0;
    const int base0 = static_cast<int>((static_cast<long long>(span) * split) / ns);
    const int end = static_cast<int>((static_cast<long long>(span) * (split + 1)) / ns);
    const int kv_per = end - base0;
    const int ntiles = (kv_per + TILE - 1) / TILE;

    if (kv_per == 0) {
        for (int r = tid; r < ROWS; r += 256) {
            ws_m[static_cast<size_t>(r) * ns + split] = -1.0e30f;
            ws_l[static_cast<size_t>(r) * ns + split] = 0.0f;
        }
        for (int i = tid; i < ROWS * HD; i += 256)
            ws_acc[(static_cast<size_t>(i / HD) * ns + split) * HD + (i % HD)] =
                __float2bfloat16(0.0f);
        return;
    }

    extern __shared__ __align__(1024) char smem[];
    constexpr int KVB = TILE * HD;
    bf16* Qsh = reinterpret_cast<bf16*>(smem);  // [4 chunks][64][64]
    bf16* Ksh = Qsh + ROWS * HD;
    bf16* Vsh = Ksh + STAGES * KVB;
    bf16* Psh = Vsh + STAGES * KVB;                              // 8 blocks [64k][8q]
    float* c_row = reinterpret_cast<float*>(Psh + TILE * ROWS);  // [64]
    float* l_row = c_row + ROWS;                                 // [64]
    float* m_fin = l_row + ROWS;                                 // [64]
    __shared__ __align__(8) uint64_t bq, bkv[STAGES];

    if (tid == 0) {
        mbar_init(&bq, 1);
#pragma unroll
        for (int s = 0; s < STAGES; ++s) mbar_init(&bkv[s], 1);
        prefetch_map(&Kmap);
        prefetch_map(&Vmap);
        prefetch_map(&Qmap);
    }
    bar_sync(3, 256);
    if (tid == 0) {
        mbar_expect(&bq, ROWS * HD * static_cast<int>(sizeof(bf16)));
#pragma unroll
        for (int j = 0; j < 4; ++j) tma_load(&Qmap, &bq, Qsh + j * ROWS * 64, j * 64, 0);
        for (int t = 0; t < min(STAGES, ntiles); ++t) {
            mbar_expect(&bkv[t], 2 * KVB * static_cast<int>(sizeof(bf16)));
            const int key = u0 + base0 + t * TILE;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                tma_load(&Kmap, &bkv[t], Ksh + t * KVB + j * 4096, j * 64, key);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                tma_load(&Vmap, &bkv[t], Vsh + t * KVB + j * 4096, j * 64, key);
        }
    }
    mbar_wait(&bq, 0);

    // Softmax state (wg0; rows r0 = warp*16+gid, r1 = r0+8, replicated per quad).
    float m_i[2] = {-1.0e30f, -1.0e30f}, l_i[2] = {0.0f, 0.0f};
    // PV accumulators: this wg's two d-blocks, 8 col-groups x 4 regs each.
    float acc[2][8][4];
#pragma unroll
    for (int b = 0; b < 2; ++b)
#pragma unroll
        for (int g = 0; g < 8; ++g)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[b][g][i] = 0.0f;

    for (int tile = 0; tile < ntiles; ++tile) {
        const int stage = tile % STAGES;
        const int tbase = u0 + base0 + tile * TILE;
        mbar_wait(&bkv[stage], (tile / STAGES) & 1);
        bf16* Kb = Ksh + stage * KVB;
        bf16* Vb = Vsh + stage * KVB;

        if (wg == 0) {
            // QK: A = Qsh [64 q, k16 chunks], B = Kb [k16, 64 keys].
            float sc[32];
#pragma unroll
            for (float& x : sc) x = 0.0f;
            asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
            const uint64_t ad = make_desc(Qsh, 1, 64, 1);
            const uint64_t bd = make_desc(Kb, 1, 64, 1);
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                const uint64_t o = (((k >> 2) * 8192 + (k & 3) * 32) >> 4);
                wgmma_n64(sc, ad + o, bd + o);
            }
            wg_commit();
            wg_wait<0>();

            // Mask + per-row online softmax. Thread rows: r0, r1; cols
            // c = tg*2 + 8i (+1). Window masks use the row's draft j = r>>4.
            const int r0 = warp * 16 + gid, r1 = r0 + 8;
            const int j0 = r0 >> 4, j1 = r1 >> 4;
            const int lo0 = off[j0] - tbase, hi0 = off[j0] + loc[j0] - tbase;
            const int lo1 = off[j1] - tbase, hi1 = off[j1] + loc[j1] - tbase;
            const int vend = kv_per - tile * TILE;  // union-range tail clamp
            float rm0 = -1.0e30f, rm1 = -1.0e30f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int c0 = tg * 2 + i * 8, c1 = c0 + 1;
                if (c0 < lo0 || c0 >= hi0 || c0 >= vend) sc[i * 4 + 0] = -1.0e30f;
                if (c1 < lo0 || c1 >= hi0 || c1 >= vend) sc[i * 4 + 1] = -1.0e30f;
                if (c0 < lo1 || c0 >= hi1 || c0 >= vend) sc[i * 4 + 2] = -1.0e30f;
                if (c1 < lo1 || c1 >= hi1 || c1 >= vend) sc[i * 4 + 3] = -1.0e30f;
                rm0 = fmaxf(rm0, fmaxf(sc[i * 4 + 0], sc[i * 4 + 1]));
                rm1 = fmaxf(rm1, fmaxf(sc[i * 4 + 2], sc[i * 4 + 3]));
            }
#pragma unroll
            for (int o = 1; o <= 2; o <<= 1) {
                rm0 = fmaxf(rm0, __shfl_xor_sync(0xffffffffu, rm0, o));
                rm1 = fmaxf(rm1, __shfl_xor_sync(0xffffffffu, rm1, o));
            }
            const float mn0 = fmaxf(m_i[0], rm0), mn1 = fmaxf(m_i[1], rm1);
            const float cf0 = exp2f((m_i[0] - mn0) * scale2);
            const float cf1 = exp2f((m_i[1] - mn1) * scale2);
            float rs0 = 0.0f, rs1 = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const bool up = (q < 2);
                    const float mn = up ? mn0 : mn1;
                    float e = sc[i * 4 + q];
                    e = (e > -1.0e29f) ? exp2f((e - mn) * scale2) : 0.0f;
                    sc[i * 4 + q] = e;
                    if (up)
                        rs0 += e;
                    else
                        rs1 += e;
                }
            }
#pragma unroll
            for (int o = 1; o <= 2; o <<= 1) {
                rs0 += __shfl_xor_sync(0xffffffffu, rs0, o);
                rs1 += __shfl_xor_sync(0xffffffffu, rs1, o);
            }
            l_i[0] = l_i[0] * cf0 + rs0;
            l_i[1] = l_i[1] * cf1 + rs1;
            m_i[0] = mn0;
            m_i[1] = mn1;
            if (tg == 0) {
                c_row[r0] = cf0;
                c_row[r1] = cf1;
            }
// P store: col-group g = c>>3; Psh block layout matches pd usage:
// Pg[(key>>3)*64 + qcol*8 + (key&7)], qcol = c&7, key = row of D?
// Here D rows are q and cols are KEYS: key = c, qrow = r.
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int k0 = tg * 2 + i * 8, k1 = k0 + 1;
                bf16* P0 = Psh + (r0 >> 3) * TILE * 8;
                bf16* P1 = Psh + (r1 >> 3) * TILE * 8;
                P0[(k0 >> 3) * 64 + (r0 & 7) * 8 + (k0 & 7)] = static_cast<bf16>(sc[i * 4 + 0]);
                P0[(k1 >> 3) * 64 + (r0 & 7) * 8 + (k1 & 7)] = static_cast<bf16>(sc[i * 4 + 1]);
                P1[(k0 >> 3) * 64 + (r1 & 7) * 8 + (k0 & 7)] = static_cast<bf16>(sc[i * 4 + 2]);
                P1[(k1 >> 3) * 64 + (r1 & 7) * 8 + (k1 & 7)] = static_cast<bf16>(sc[i * 4 + 3]);
            }
        }
        bar_sync(1, 256);  // P + c_row visible to both warpgroups

        // PV: this wg's d-blocks; A = V^T d-block (vd), B = Psh col-group g
        // (pd). Rescale accumulators by this tile's c_row per q-col first.
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        const uint64_t vd = make_desc(Vb, 256, 64, 1);
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            const int ib = wg * 2 + b;
#pragma unroll
            for (int g = 0; g < 8; ++g) {
                const int q0 = g * 8 + tg * 2;
                const float cq0 = c_row[q0], cq1 = c_row[q0 + 1];
                acc[b][g][0] *= cq0;
                acc[b][g][1] *= cq1;
                acc[b][g][2] *= cq0;
                acc[b][g][3] *= cq1;
                const uint64_t pd = make_desc(Psh + g * TILE * 8, 8, 256, 0);
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const uint64_t vo = ((ib * 8192 + kk * 2048) >> 4);
                    const uint64_t po = ((kk * 2) * 64 * 2 >> 4);
                    wgmma_n8_acc(acc[b][g], vd + vo, pd + po);
                }
            }
        }
        wg_commit();
        wg_wait<0>();
        bar_sync(2, 256);  // PV consumed Psh; wg0 may overwrite next tile

        if (tid == 0 && tile + STAGES < ntiles) {
            const int next = tile + STAGES, next_stage = next % STAGES;
            const int key = u0 + base0 + next * TILE;
            mbar_expect(&bkv[next_stage], 2 * KVB * static_cast<int>(sizeof(bf16)));
#pragma unroll
            for (int j = 0; j < 4; ++j)
                tma_load(&Kmap, &bkv[next_stage], Ksh + next_stage * KVB + j * 4096, j * 64, key);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                tma_load(&Vmap, &bkv[next_stage], Vsh + next_stage * KVB + j * 4096, j * 64, key);
        }
    }

    // Publish m/l (wg0) then normalize + store both wgs' d-blocks.
    if (wg == 0) {
        const int r0 = warp * 16 + gid, r1 = r0 + 8;
        if (tg == 0) {
            m_fin[r0] = m_i[0];
            m_fin[r1] = m_i[1];
            l_row[r0] = l_i[0];
            l_row[r1] = l_i[1];
        }
    }
    bar_sync(1, 256);
    if (wg == 0 && tg == 0) {
        const int r0 = warp * 16 + gid, r1 = r0 + 8;
        ws_m[static_cast<size_t>(r0) * ns + split] = m_i[0];
        ws_l[static_cast<size_t>(r0) * ns + split] = l_i[0];
        ws_m[static_cast<size_t>(r1) * ns + split] = m_i[1];
        ws_l[static_cast<size_t>(r1) * ns + split] = l_i[1];
    }
#pragma unroll
    for (int b = 0; b < 2; ++b) {
        const int ib = wg * 2 + b;
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int q0 = g * 8 + tg * 2;
            const float il0 = l_row[q0] > 0.0f ? 1.0f / l_row[q0] : 0.0f;
            const float il1 = l_row[q0 + 1] > 0.0f ? 1.0f / l_row[q0 + 1] : 0.0f;
            const int d0 = ib * 64 + warp * 16 + gid, d1 = d0 + 8;
            bf16* a0 = ws_acc + (static_cast<size_t>(q0) * ns + split) * HD;
            bf16* a1 = ws_acc + (static_cast<size_t>(q0 + 1) * ns + split) * HD;
            a0[d0] = static_cast<bf16>(acc[b][g][0] * il0);
            a1[d0] = static_cast<bf16>(acc[b][g][1] * il1);
            a0[d1] = static_cast<bf16>(acc[b][g][2] * il0);
            a1[d1] = static_cast<bf16>(acc[b][g][3] * il1);
        }
    }
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

// q64: [64, 256] bf16 (4 drafts x 16 heads); slots4: int32[4][2] device
// ([token_id, S+j] per draft); ws_m/ws_l: [64, ns] fp32; ws_acc: [64, ns, 256]
// bf16. Partial only -- combine/publish are separate stages.
extern "C" int NexN25ProM4CpQuadPartialLaunch(const void* q64, const void* K, const void* V,
                                              const void* slots4, int cp_rank, int cp, int ns,
                                              float scaling, void* wm, void* wl, void* wa,
                                              void* stream) {
    CUtensorMap Kmap = encode_map(const_cast<void*>(K), KV_CAPACITY, HD, TILE, 64);
    CUtensorMap Vmap = encode_map(const_cast<void*>(V), KV_CAPACITY, HD, TILE, 64);
    CUtensorMap Qmap = encode_map(const_cast<void*>(q64), ROWS, HD, ROWS, 64);
    constexpr int STAGES = CPQ_STAGES;
    const int smem =
        (ROWS * HD + 2 * STAGES * TILE * HD + TILE * ROWS) * static_cast<int>(sizeof(bf16)) +
        3 * ROWS * static_cast<int>(sizeof(float)) + 1024;
    static bool initialized = false;
    if (!initialized) {
        cudaFuncSetAttribute(m4_cp_quad_partial<STAGES>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        initialized = true;
    }
    m4_cp_quad_partial<STAGES><<<dim3(ns), 256, smem, reinterpret_cast<cudaStream_t>(stream)>>>(
        Kmap, Vmap, Qmap, static_cast<const int*>(slots4), cp_rank, cp, static_cast<float*>(wm),
        static_cast<float*>(wl), static_cast<bf16*>(wa), ns, scaling);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M4