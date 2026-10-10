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

// M=8 non-CP causal attention core: ONE KV pass for all 32 q-rows
// (8 target rows x 4 local heads), WGMMA/TMA class, plus the per-row combine
// (and its xeff variant that hoists the o_proj gating).
//
// M=8-FAMILY TU: native eight-row implementation, no BS=1/M=4 production source.
// The one-KV-pass multi-column partial is derived from the M=4 quad geometry,
// re-derived for eight rows. q_out8 [8 rows x 4 heads, 256] is viewed as ONE
// [32, 256] B operand: grid (ns, 1), K/V staged once per tile through TMA, and the
// 32 columns computed as FOUR n8 column-groups over the same smem K/V. The shared
// K/V history is read ONCE for all eight causal queries -- extra query arithmetic,
// not extra cache traffic.
//
// CAUSALITY (staircase): column c belongs to target row j = c>>2 with extent
// E_j = C + 1 + j. All rows walk row-7's split ranges (S7 = C + 8 keys); keys
// beyond a column's extent are masked AFTER the QK wgmma: -1e30 into the running
// max, 0.0 into the P tile and row-sum. DECLARED NUMERICS CLASS: per-(row, head)
// results are a split/tile-boundary reorder of a per-row walk (the accepted quad
// class); the all-masked-split guard publishes m=-1e30, l=0, acc=0 -- the exact
// kv_per==0 contract the combine handles.
//
// Build contract: hopper=True, link_cuda=True (WGMMA + cuTensorMapEncodeTiled).

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M8
#endif

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "../common/m8_control_block.cuh"

namespace nexrt::sm90::nex_n2_5_pro::M8 {
using bf16 = __nv_bfloat16;

__device__ __forceinline__ uint32_t m8_pack_bf16(float a, float b) {
    uint32_t r;
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(b), "f"(a));
    return r;
}

#define M8_COMBINE_THREADS 256

constexpr int HD = 256;
constexpr int NH = 4;  // local q heads per rank
constexpr int TILE = 64;
constexpr int MROWS8 = m8ctl::kM;     // 8 target rows
constexpr int QROWS32 = MROWS8 * NH;  // 32 q-rows as one B operand
constexpr int QCHUNK = QROWS32 * 64;  // elements per 64-col TMA chunk
constexpr int KV_CAPACITY = 256 * 1024 + 1024;
constexpr uint64_t TMA_EVICT_FIRST = 0x12f0000000000000ULL;

__device__ __forceinline__ uint64_t m8_cvta(const void* p) {
    return static_cast<uint64_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ uint64_t m8_make_desc(const void* p, uint32_t lbo, uint32_t sbo,
                                                 uint32_t swizzle) {
    const uint64_t a = (m8_cvta(p) >> 4) & 0x3fffull;
    return a | (static_cast<uint64_t>(lbo & 0x3fff) << 16) |
           (static_cast<uint64_t>(sbo & 0x3fff) << 32) | (static_cast<uint64_t>(swizzle & 3) << 62);
}
__device__ __forceinline__ void m8_wg_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void m8_wg_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ void m8_mbar_init(uint64_t* b, int count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(static_cast<uint32_t>(m8_cvta(b))),
                 "r"(count));
}
__device__ __forceinline__ void m8_mbar_expect(uint64_t* b, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(
                     static_cast<uint32_t>(m8_cvta(b))),
                 "r"(bytes));
}
// Plain arrival (no expect_tx): used for the consumer-side EMPTY barriers,
// whose completion count is the number of consumer warpgroups, not bytes.
__device__ __forceinline__ void m8_mbar_arrive(uint64_t* b) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" ::"r"(static_cast<uint32_t>(m8_cvta(b)))
                 : "memory");
}
__device__ __forceinline__ void m8_mbar_wait(uint64_t* b, int phase) {
    asm volatile(
        "{\n.reg .pred p;\nL%=:\n"
        "mbarrier.try_wait.parity.shared.b64 p,[%0],%1;\n"
        "@!p bra L%=;\n}\n" ::"r"(static_cast<uint32_t>(m8_cvta(b))),
        "r"(phase));
}
__device__ __forceinline__ void m8_tma_load(const CUtensorMap* map, uint64_t* bar, void* dst,
                                            int c0, int c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(
            static_cast<uint32_t>(m8_cvta(dst))),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1),
        "r"(static_cast<uint32_t>(m8_cvta(bar)))
        : "memory");
}
__device__ __forceinline__ void m8_tma_load_kv(const CUtensorMap* map, uint64_t* bar, void* dst,
                                               int c0, int c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1, {%2, %3}], [%4], %5;\n" ::"r"(static_cast<uint32_t>(m8_cvta(dst))),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1),
        "r"(static_cast<uint32_t>(m8_cvta(bar))), "l"(TMA_EVICT_FIRST)
        : "memory");
}
__device__ __forceinline__ void m8_prefetch_map(const CUtensorMap* map) {
    asm volatile("prefetch.tensormap [%0];\n" ::"l"(reinterpret_cast<uint64_t>(map)));
}
// Per-warpgroup barrier: warpgroup w owns named barrier 1+w over its 128
// threads (barrier 0 stays reserved for __syncthreads()).
__device__ __forceinline__ void m8_wg_sync(int wg) {
    asm volatile("bar.sync %0, 128;\n" ::"r"(wg + 1) : "memory");
}

// Four consumer warpgroups share one K/V pass. Each warpgroup owns one n8
// column group and uses a private named barrier for its softmax reductions.
template <int DT, int BLK>
__device__ __forceinline__ void m8_combine_body(
    const float* __restrict__ ws_m, const float* __restrict__ ws_l, const bf16* __restrict__ ws_acc,
    bf16* __restrict__ out,
    const bf16* __restrict__ gate,  // nullptr = plain combine
    bf16* __restrict__ xeff, int ns, float scaling) {
    // DT bf16 channels per CTA. Production ships DT=64 for BOTH arms; the
    // comment below describes the general gather, and at DT=8 each split's
    // a single 16 B uint4, but consecutive threads then step `s`, whose stride
    // is NH*HD = 2048 B -- so one warp touches 32 DIFFERENT cache lines to use
    // 512 B. NCU on the shipped combine: L1/TEX 47%, DRAM 8.5%, achieved
    // occupancy 69.6% of 100% theoretical = transaction-bound, not
    // bandwidth-bound. Wider DT loads DT/8 uint4s per split with consecutive
    // threads covering ONE split's contiguous DT*2 bytes, cutting lines per
    // warp by DT/8 -- at the cost of a DT-times-smaller grid.
    static_assert(DT >= 8 && DT % 8 == 0, "split gather works in 16B units");
    static_assert(BLK >= DT, "the reduction needs at least one group");
    constexpr int U = DT / 8;  // 16 B units per split
    const int o0 = blockIdx.x * DT, h = o0 / HD, d0 = o0 % HD, tid = threadIdx.x;
    const float scale2 = scaling * 1.4426950408889634f;
    extern __shared__ __align__(16) char csmem[];
    bf16* sm = reinterpret_cast<bf16*>(csmem);
    float* m_sh = reinterpret_cast<float*>(sm + ns * DT);
    float* l_sh = m_sh + ns;
    __shared__ float red[BLK], den_red[BLK];
    // idx = s*U + u, so consecutive threads walk u first: threads 0..U-1 read
    // one split's DT*2 contiguous bytes. At U=1 this is exactly the shipped
    // loop, instruction for instruction.
    for (int idx = tid; idx < ns * U; idx += BLK) {
        const int s = idx / U, u = idx - s * U;
        *reinterpret_cast<uint4*>(sm + s * DT + u * 8) = *reinterpret_cast<const uint4*>(
            ws_acc + (static_cast<size_t>(s) * NH + h) * HD + d0 + u * 8);
    }
    for (int s = tid; s < ns; s += BLK) {
        m_sh[s] = ws_m[s * NH + h];
        l_sh[s] = ws_l[s * NH + h];
    }
    __syncthreads();
    float lm = -1.0e30f;
    for (int s = tid; s < ns; s += BLK) lm = fmaxf(lm, m_sh[s]);
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
        const float w = exp2f((m_sh[s] - gm) * scale2) * l_sh[s];
        num += w * static_cast<float>(sm[s * DT + d]);
        den += w;
    }
    __syncthreads();
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
        const bf16 ctx_b = static_cast<bf16>(n / (z + 1.0e-20f));
        out[o0 + tid] = ctx_b;
        if (xeff != nullptr) {
            const float gv = __bfloat162float(gate[o0 + tid]);
            xeff[o0 + tid] =
                __float2bfloat16(__bfloat162float(ctx_b) * (1.0f / (1.0f + __expf(-gv))));
        }
    }
}

template <int DT, int BLK>
__global__ __launch_bounds__(BLK) void m8_attn_combine_xeff(
    const float* __restrict__ ws_m8, const float* __restrict__ ws_l8,
    const bf16* __restrict__ ws_acc8, bf16* __restrict__ ctx8, const bf16* __restrict__ gate8,
    bf16* __restrict__ xeff8, int ns, float scaling) {
    const int row = blockIdx.y;
    m8_combine_body<DT, BLK>(
        ws_m8 + static_cast<size_t>(row) * ns * NH, ws_l8 + static_cast<size_t>(row) * ns * NH,
        ws_acc8 + static_cast<size_t>(row) * ns * NH * HD,
        ctx8 + static_cast<size_t>(row) * NH * HD, gate8 + static_cast<size_t>(row) * NH * HD,
        xeff8 + static_cast<size_t>(row) * NH * HD, ns, scaling);
}

CUtensorMap m8_encode_map(void* ptr, int outer, int inner, int box_outer, int box_inner) {
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

// ---- four-warpgroup split-K/V partial ----
// Production split-K/V path with independent K and V completion.
// ===========================================================================
// Four symmetric compute warpgroups. K and V use independent completion.
//
// A combined K/V barrier would span
// (2*KVB*sizeof(bf16) = 64 KiB), so a consumer may not touch K until V has
// also landed and QK cannot start until the last of eight TMA transactions
// retires, preventing the intended load/compute overlap.
//
// Here K and V carry independent per-chunk state:
//     k_full[STAGES][QKC]   k_empty[STAGES][QKC]
//     v_full[STAGES][PVC]   v_empty[STAGES][PVC]
// so that
//   * QK starts as soon as K chunk 0 lands, chunks 1..3 still in flight;
//   * V is issued with K but awaited only immediately before the PV atoms that
//     read it, so it streams under QK, the softmax and P materialization;
//   * a K stage is released for reuse right after QK rather than at end of
//     tile, because nothing past QK reads Ksh.
//
// NUMERICAL ORDER. The QK k-loop is already chunk-major:
// ko = (k>>2)*8192 + (k&3)*32 bytes, so k == kc*4 + kk reproduces the identical
// atom sequence. K smem is already four contiguous 4096-element chunks filled
// by four separate TMA ops at column offset j*64, so per-chunk readiness needs
// no layout change at all. On the PV side vo = i8*8192 + kk*2048 makes the PV
// block index i8 identical to the V TMA chunk index, and (gl, i8) address
// DISJOINT accumulators out_frag[gl][i8*4+..], so hoisting the v_full wait into
// the i8 loop cannot reorder any reduction. Scalar arithmetic, masking, the
// online m/l/acc update order and the output layout are untouched.
//
// BARRIER ARITHMETIC, proved from the transactions rather than assumed:
//   k_full[s][c]  arrival count 1 (the TMA itself); expect_tx =
//                 TILE*QK_CHUNK*sizeof(bf16) = 64*64*2 = 8192 B. Four chunks
//                 = 32 KiB, the K half of the 64 KiB K+V tile.
//   v_full[s][c]  arrival count 1; expect_tx = TILE*64*sizeof(bf16) = 8192 B.
//   k_empty/v_empty  arrival count kWgT, one elected lane per consumer
//                 warpgroup (lt == 0), arriving only after that warpgroup's
//                 wg_wait<0> retired every atom that reads the buffer.
//   PHASE: stage s serves tiles s, s+STAGES, s+2*STAGES, ...; generation
//                 g = tile / STAGES. An mbarrier parity starts at 0 and flips
//                 on each completion, so the g-th fill completes phase (g & 1),
//                 and a producer about to write generation g+1 waits for the
//                 g-th empty, parity (g & 1).
//
// DEADLOCK NOTE. tid 0 is also a compute thread of warpgroup 0 and wgmma is a
// 128-thread collective, so thread 0 may never block anywhere its warpgroup
// has a pending collective. The re-arm therefore sits after PV's wg_wait<0>
// and after the empty releases, and warpgroup 0 rejoins through m8_wg_sync
// before the next tile's QK, so no member diverges around a WGMMA sequence.
// SK / SV are the K and V ring depths. They are INDEPENDENT only because this
// kernel splits the barriers; a single K+V barrier would force SK == SV.
// Per-stage cost is TILE*HD*2 = 32 KiB each, so SK+SV is the smem budget and
// (4,2) costs exactly what (3,3) costs while prefetching K one tile deeper --
// K is what gates QK, V is consumed later.
template <int SK, int SV, int kWgT, int QK_CHUNK>
__global__ __launch_bounds__(128 * kWgT) void m8_attn_partial_splitkv_dyn(
    const __grid_constant__ CUtensorMap Kmap, const __grid_constant__ CUtensorMap Vmap,
    const __grid_constant__ CUtensorMap Qmap, float* __restrict__ ws_m8, float* __restrict__ ws_l8,
    bf16* __restrict__ ws_acc8, const int* __restrict__ ctl, int ns, float scaling) {
    static_assert(4 % kWgT == 0, "the four n8 column-groups must divide evenly");
    static_assert(HD % QK_CHUNK == 0, "QK chunk must divide the head dim");
    static_assert(QK_CHUNK == 64, "only the 128B-swizzle 64-wide K chunk is wired");
    static_assert(SK >= 1 && SV >= 1, "ring depths must be positive");
    constexpr int GPW = 4 / kWgT;
    constexpr int NTHREADS = 128 * kWgT;
    constexpr int QKC = HD / QK_CHUNK;                           // 4 K chunks
    constexpr int PVC = HD / 64;                                 // 4, fixed by wgmma PV M=64
    constexpr int KCH = TILE * QK_CHUNK;                         // 4096 elements per K chunk
    constexpr int VCH = TILE * 64;                               // 4096 elements per V chunk
    constexpr int KCH_B = KCH * static_cast<int>(sizeof(bf16));  // 8192
    constexpr int VCH_B = VCH * static_cast<int>(sizeof(bf16));  // 8192

    const int Sbase = m8ctl::CommittedLen(ctl);
    const int S7 = Sbase + MROWS8;
    const int split = blockIdx.x, tid = threadIdx.x, lane = tid & 31;
    const int wg = tid >> 7, lt = tid & 127;
    const int warp = lt >> 5, gid = lane >> 2, tg = lane & 3;
    const float scale2 = scaling * 1.4426950408889634f;
    const int base0 = static_cast<int>((static_cast<long long>(S7) * split) / ns);
    const int end = static_cast<int>((static_cast<long long>(S7) * (split + 1)) / ns);
    const int kv_per = end - base0;
    const int ntiles = (kv_per + TILE - 1) / TILE;

    // Empty split: publish the exact flash-combine identity element, unchanged.
    if (kv_per == 0) {
        if (tid < MROWS8 * NH) {
            const int r = tid >> 2, h = tid & 3;
            ws_m8[(static_cast<size_t>(r) * ns + split) * NH + h] = -1.0e30f;
            ws_l8[(static_cast<size_t>(r) * ns + split) * NH + h] = 0.0f;
        }
        // The full identity workspace is 16 KiB. Each thread issues aligned
        // 16-byte stores rather than scalar BF16 stores.
        constexpr int ZV_ROW = NH * HD * sizeof(bf16) / sizeof(uint4);
        const uint4 zero = make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int r = 0; r < MROWS8; ++r) {
            uint4* const z =
                reinterpret_cast<uint4*>(ws_acc8 + (static_cast<size_t>(r) * ns + split) * NH * HD);
            for (int i = tid; i < ZV_ROW; i += NTHREADS) z[i] = zero;
        }
        return;
    }

    extern __shared__ __align__(1024) char smem[];
    constexpr int KVB = TILE * HD;
    bf16* Qsh = reinterpret_cast<bf16*>(smem);
    bf16* Ksh = Qsh + QROWS32 * HD;
    bf16* Vsh = Ksh + SK * KVB;
    bf16* Psh = Vsh + SV * KVB;
    __shared__ __align__(8) uint64_t bq;
    __shared__ __align__(8) uint64_t k_full[SK][QKC], k_empty[SK][QKC];
    __shared__ __align__(8) uint64_t v_full[SV][PVC], v_empty[SV][PVC];
    __shared__ float max_part[4][QROWS32], sum_part[4][QROWS32];

    if (tid == 0) {
        m8_mbar_init(&bq, 1);
#pragma unroll
        for (int s = 0; s < SK; ++s)
#pragma unroll
            for (int c = 0; c < QKC; ++c) {
                m8_mbar_init(&k_full[s][c], 1);
                m8_mbar_init(&k_empty[s][c], kWgT);
            }
#pragma unroll
        for (int s = 0; s < SV; ++s)
#pragma unroll
            for (int c = 0; c < PVC; ++c) {
                m8_mbar_init(&v_full[s][c], 1);
                m8_mbar_init(&v_empty[s][c], kWgT);
            }
        m8_prefetch_map(&Kmap);
        m8_prefetch_map(&Vmap);
        m8_prefetch_map(&Qmap);
    }
    // mbarrier init must be visible to the async proxy before any expect_tx.
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    __syncthreads();

    if (tid == 0) {
        m8_mbar_expect(&bq, QROWS32 * HD * static_cast<int>(sizeof(bf16)));
#pragma unroll
        for (int j = 0; j < 4; ++j) m8_tma_load(&Qmap, &bq, Qsh + j * QCHUNK, j * 64, 0);
        for (int t = 0; t < min(SK, ntiles); ++t) {
            const int s = t % SK, key = base0 + t * TILE;
#pragma unroll
            for (int c = 0; c < QKC; ++c) {
                m8_mbar_expect(&k_full[s][c], KCH_B);
                m8_tma_load_kv(&Kmap, &k_full[s][c], Ksh + s * KVB + c * KCH, c * QK_CHUNK, key);
            }
        }
        for (int t = 0; t < min(SV, ntiles); ++t) {
            const int s = t % SV, key = base0 + t * TILE;
#pragma unroll
            for (int c = 0; c < PVC; ++c) {
                m8_mbar_expect(&v_full[s][c], VCH_B);
                m8_tma_load_kv(&Vmap, &v_full[s][c], Vsh + s * KVB + c * VCH, c * 64, key);
            }
        }
    }
    m8_mbar_wait(&bq, 0);

    float out_frag[GPW][16];
#pragma unroll
    for (int g = 0; g < GPW; ++g)
#pragma unroll
        for (int i = 0; i < 16; ++i) out_frag[g][i] = 0.0f;
    float m_i[GPW][2], l_i[GPW][2];
#pragma unroll
    for (int g = 0; g < GPW; ++g) {
        m_i[g][0] = m_i[g][1] = -1.0e30f;
        l_i[g][0] = l_i[g][1] = 0.0f;
    }

    for (int tile = 0; tile < ntiles; ++tile) {
        const int kstage = tile % SK, vstage = tile % SV;
        const int kph = (tile / SK) & 1, vph = (tile / SV) & 1;
        bf16* Kb = Ksh + kstage * KVB;
        bf16* Vb = Vsh + vstage * KVB;

        float score[GPW][4];
#pragma unroll
        for (int g = 0; g < GPW; ++g)
#pragma unroll
            for (int i = 0; i < 4; ++i) score[g][i] = 0.0f;
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        const uint64_t kd = m8_make_desc(Kb, 1, 64, 1);
        const uint64_t qd = m8_make_desc(Qsh, 1, 64, 1);
// INNER QK PIPELINE: one 64-D chunk at a time, so atoms for chunk kc
// issue while chunks kc+1.. are still in flight. k == kc*4+kk keeps the
// production atom order.
#pragma unroll
        for (int kc = 0; kc < QKC; ++kc) {
            m8_mbar_wait(&k_full[kstage][kc], kph);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const int k = kc * 4 + kk;
                const uint64_t ko = (((k >> 2) * 8192 + (k & 3) * 32) >> 4);
                const uint64_t qo = (((k >> 2) * 4096 + (k & 3) * 32) >> 4);
#pragma unroll
                for (int gl = 0; gl < GPW; ++gl) {
                    const uint64_t gq = static_cast<uint64_t>(wg * GPW + gl) * 64u;
                    asm volatile(
                        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
                        "{%0,%1,%2,%3}, %4, %5, 1,1,1,0,0;\n"
                        : "+f"(score[gl][0]), "+f"(score[gl][1]), "+f"(score[gl][2]),
                          "+f"(score[gl][3])
                        : "l"(kd + ko), "l"(qd + qo + gq));
                }
            }
            m8_wg_commit();  // one group per 64-D chunk
        }
        m8_wg_wait<0>();
        // Nothing past QK reads Ksh, so this stage's K is reusable NOW.
        if (lt == 0) {
#pragma unroll
            for (int c = 0; c < QKC; ++c) m8_mbar_arrive(&k_empty[kstage][c]);
        }

        const int tbase = base0 + tile * TILE;
        const int valid = min(TILE, kv_per - tile * TILE);
        const int k0 = warp * 16 + gid, k1 = k0 + 8;
#pragma unroll
        for (int gl = 0; gl < GPW; ++gl) {
            const int g = wg * GPW + gl;
            const int c0 = g * 8 + tg * 2, c1 = c0 + 1;
            const int j0 = c0 >> 2, j1 = c1 >> 2;
            const int lim0 = min(valid, Sbase + 1 + j0 - tbase);
            const int lim1 = min(valid, Sbase + 1 + j1 - tbase);
            if (k0 >= lim0) score[gl][0] = -1.0e30f;
            if (k0 >= lim1) score[gl][1] = -1.0e30f;
            if (k1 >= lim0) score[gl][2] = -1.0e30f;
            if (k1 >= lim1) score[gl][3] = -1.0e30f;
            float lm0 = fmaxf(score[gl][0], score[gl][2]);
            float lm1 = fmaxf(score[gl][1], score[gl][3]);
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
        m8_wg_sync(wg);
#pragma unroll
        for (int gl = 0; gl < GPW; ++gl) {
            const int g = wg * GPW + gl;
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
            const float mn0 = fmaxf(m_i[gl][0], tm0), mn1 = fmaxf(m_i[gl][1], tm1);
            score[gl][0] = k0 < lim0 ? exp2f((score[gl][0] - mn0) * scale2) : 0.0f;
            score[gl][1] = k0 < lim1 ? exp2f((score[gl][1] - mn1) * scale2) : 0.0f;
            score[gl][2] = k1 < lim0 ? exp2f((score[gl][2] - mn0) * scale2) : 0.0f;
            score[gl][3] = k1 < lim1 ? exp2f((score[gl][3] - mn1) * scale2) : 0.0f;
            float s0 = score[gl][0] + score[gl][2], s1 = score[gl][1] + score[gl][3];
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
        m8_wg_sync(wg);
#pragma unroll
        for (int gl = 0; gl < GPW; ++gl) {
            const int g = wg * GPW + gl;
            const int c0 = g * 8 + tg * 2, c1 = c0 + 1;
            float ts0 = 0.0f, ts1 = 0.0f;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                ts0 += sum_part[w][c0];
                ts1 += sum_part[w][c1];
            }
            float mm0 = -1.0e30f, mm1 = -1.0e30f;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                mm0 = fmaxf(mm0, max_part[w][c0]);
                mm1 = fmaxf(mm1, max_part[w][c1]);
            }
            const float mn0 = fmaxf(m_i[gl][0], mm0), mn1 = fmaxf(m_i[gl][1], mm1);
            const float c0f = exp2f((m_i[gl][0] - mn0) * scale2);
            const float c1f = exp2f((m_i[gl][1] - mn1) * scale2);
            l_i[gl][0] = l_i[gl][0] * c0f + ts0;
            l_i[gl][1] = l_i[gl][1] * c1f + ts1;
            m_i[gl][0] = mn0;
            m_i[gl][1] = mn1;
#pragma unroll
            for (int i = 0; i < 16; ++i) out_frag[gl][i] *= (i & 1) ? c1f : c0f;
            bf16* Pg = Psh + g * TILE * 8;
            Pg[(k0 >> 3) * 64 + (tg * 2) * 8 + (k0 & 7)] = static_cast<bf16>(score[gl][0]);
            Pg[(k0 >> 3) * 64 + (tg * 2 + 1) * 8 + (k0 & 7)] = static_cast<bf16>(score[gl][1]);
            Pg[(k1 >> 3) * 64 + (tg * 2) * 8 + (k1 & 7)] = static_cast<bf16>(score[gl][2]);
            Pg[(k1 >> 3) * 64 + (tg * 2 + 1) * 8 + (k1 & 7)] = static_cast<bf16>(score[gl][3]);
        }
        m8_wg_sync(wg);

        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        const uint64_t vd = m8_make_desc(Vb, 256, 64, 1);
// V was in flight through QK, the softmax and the P store; wait for each
// 64-D chunk only immediately before the atoms that read it. (gl, i8)
// address disjoint accumulators, so the i8-outer order is bitwise.
#pragma unroll
        for (int i8 = 0; i8 < PVC; ++i8) {
            m8_mbar_wait(&v_full[vstage][i8], vph);
#pragma unroll
            for (int gl = 0; gl < GPW; ++gl) {
                const int g = wg * GPW + gl;
                const uint64_t pd = m8_make_desc(Psh + g * TILE * 8, 8, 256, 0);
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const uint64_t vo = ((i8 * 8192 + kk * 2048) >> 4);
                    const uint64_t po = ((kk * 2) * 64 * 2 >> 4);
                    asm volatile(
                        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
                        "{%0,%1,%2,%3}, %4,%5, 1,1,1,1,0;\n"
                        : "+f"(out_frag[gl][i8 * 4]), "+f"(out_frag[gl][i8 * 4 + 1]),
                          "+f"(out_frag[gl][i8 * 4 + 2]), "+f"(out_frag[gl][i8 * 4 + 3])
                        : "l"(vd + vo), "l"(pd + po));
                }
            }
            m8_wg_commit();  // one group per PV chunk
        }
        m8_wg_wait<0>();
        if (lt == 0) {
#pragma unroll
            for (int c = 0; c < PVC; ++c) m8_mbar_arrive(&v_empty[vstage][c]);
        }

        // Re-arm this stage for tile + STAGES. The empty barriers, not a
        // CTA-wide __syncthreads, establish that every warpgroup is done.
        if (tid == 0) {
            if (tile + SK < ntiles) {
                const int key = base0 + (tile + SK) * TILE;
#pragma unroll
                for (int c = 0; c < QKC; ++c) m8_mbar_wait(&k_empty[kstage][c], kph);
#pragma unroll
                for (int c = 0; c < QKC; ++c) {
                    m8_mbar_expect(&k_full[kstage][c], KCH_B);
                    m8_tma_load_kv(&Kmap, &k_full[kstage][c], Ksh + kstage * KVB + c * KCH,
                                   c * QK_CHUNK, key);
                }
            }
            if (tile + SV < ntiles) {
                const int key = base0 + (tile + SV) * TILE;
#pragma unroll
                for (int c = 0; c < PVC; ++c) m8_mbar_wait(&v_empty[vstage][c], vph);
#pragma unroll
                for (int c = 0; c < PVC; ++c) {
                    m8_mbar_expect(&v_full[vstage][c], VCH_B);
                    m8_tma_load_kv(&Vmap, &v_full[vstage][c], Vsh + vstage * KVB + c * VCH, c * 64,
                                   key);
                }
            }
        }
        // Warpgroup 0 rejoins its elected thread before the next QK so that no
        // member diverges around a WGMMA sequence.
        m8_wg_sync(wg);
    }

    // Retire every warpgroup's final QK/PV before the epilogue reuses Ksh.
    __syncthreads();

#pragma unroll
    for (int gl = 0; gl < GPW; ++gl) {
        const int g = wg * GPW + gl;
        const int c0 = g * 8 + tg * 2, c1 = c0 + 1;
        const float inv_l0 = l_i[gl][0] > 0.0f ? 1.0f / l_i[gl][0] : 0.0f;
        const float inv_l1 = l_i[gl][1] > 0.0f ? 1.0f / l_i[gl][1] : 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) out_frag[gl][i] *= (i & 1) ? inv_l1 : inv_l0;
        const int r0 = c0 >> 2, h0 = c0 & 3, r1 = c1 >> 2, h1 = c1 & 3;
        float* wm0 = ws_m8 + (static_cast<size_t>(r0) * ns + split) * NH + h0;
        float* wl0 = ws_l8 + (static_cast<size_t>(r0) * ns + split) * NH + h0;
        float* wm1 = ws_m8 + (static_cast<size_t>(r1) * ns + split) * NH + h1;
        float* wl1 = ws_l8 + (static_cast<size_t>(r1) * ns + split) * NH + h1;
        if (gid == 0) {
            *wm0 = m_i[gl][0];
            *wl0 = l_i[gl][0];
            *wm1 = m_i[gl][1];
            *wl1 = l_i[gl][1];
        }
        // A warpgroup owns eight complete (row,head) vectors. K/V storage is
        // dead after final PV, so reuse it as a padded transpose buffer and
        // drain complete vectors with 16-byte-per-lane stores.
        static_assert(kWgT == 4 && GPW == 1,
                      "coalesced epilogue requires the production WG4 geometry");
        constexpr int EPI_VECS = 8;
        constexpr int EPI_STRIDE_WORDS = 132;  // 512 B data + 16 B pad
        constexpr int EPI_WG_WORDS = EPI_VECS * EPI_STRIDE_WORDS;
        static_assert(kWgT * EPI_WG_WORDS * sizeof(uint32_t) <= SK * KVB * sizeof(bf16),
                      "epilogue transpose must fit in dead K storage");
        uint32_t* const epi = reinterpret_cast<uint32_t*>(Ksh) + wg * EPI_WG_WORDS;
        const int lv0 = c0 - wg * 8;
        const int lv1 = lv0 + 1;

#pragma unroll
        for (int i8 = 0; i8 < 4; ++i8) {
            const int d0 = i8 * 64 + warp * 16 + gid;
            const int d1 = d0 + 8;
            const float a00 = out_frag[gl][i8 * 4];
            const float a01 = out_frag[gl][i8 * 4 + 2];
            const float a10 = out_frag[gl][i8 * 4 + 1];
            const float a11 = out_frag[gl][i8 * 4 + 3];
            // Lanes with adjacent gid are four lanes apart. Even gid packs the
            // exact two scalar conversions; odd gid emits no shared store.
            const float b00 = __shfl_down_sync(0xffffffffu, a00, 4);
            const float b01 = __shfl_down_sync(0xffffffffu, a01, 4);
            const float b10 = __shfl_down_sync(0xffffffffu, a10, 4);
            const float b11 = __shfl_down_sync(0xffffffffu, a11, 4);
            if ((gid & 1) == 0) {
                epi[lv0 * EPI_STRIDE_WORDS + d0 / 2] = m8_pack_bf16(a00, b00);
                epi[lv0 * EPI_STRIDE_WORDS + d1 / 2] = m8_pack_bf16(a01, b01);
                epi[lv1 * EPI_STRIDE_WORDS + d0 / 2] = m8_pack_bf16(a10, b10);
                epi[lv1 * EPI_STRIDE_WORDS + d1 / 2] = m8_pack_bf16(a11, b11);
            }
        }

        m8_wg_sync(wg);
#pragma unroll
        for (int vec = warp; vec < EPI_VECS; vec += 4) {
            const uint4 v = reinterpret_cast<const uint4*>(epi + vec * EPI_STRIDE_WORDS)[lane];
            const int c = wg * EPI_VECS + vec;
            bf16* const dst =
                ws_acc8 + ((static_cast<size_t>(c >> 2) * ns + split) * NH + (c & 3)) * HD;
            reinterpret_cast<uint4*>(dst)[lane] = v;
        }
    }
}

// The split barriers preserve the shared-memory budget and launch geometry.
// Asymmetric-ring smem: SK+SV stages of TILE*HD bf16 each, plus Q and P.
template <int SK, int SV>
static int m8_splitkv_smem_bytes() {
    return (QROWS32 * HD + (SK + SV) * TILE * HD + TILE * QROWS32) *
               static_cast<int>(sizeof(bf16)) +
           1024;
}

template <int kWgT, int SK, int SV, int QK_CHUNK>
static int m8_launch_splitkv(const void* q_out8, const void* K, const void* V, const void* ctl,
                             int ns, float scaling, void* wm8, void* wl8, void* wa8, void* ctx8,
                             const void* gate8, void* xeff8, void* stream) {
    CUtensorMap Kmap = m8_encode_map(const_cast<void*>(K), KV_CAPACITY, HD, TILE, 64);
    CUtensorMap Vmap = m8_encode_map(const_cast<void*>(V), KV_CAPACITY, HD, TILE, 64);
    CUtensorMap Qmap = m8_encode_map(const_cast<void*>(q_out8), QROWS32, HD, QROWS32, 64);
    const int smem = m8_splitkv_smem_bytes<SK, SV>();
    static bool init_sk = false;
    if (!init_sk) {
        const cudaError_t attr =
            cudaFuncSetAttribute(m8_attn_partial_splitkv_dyn<SK, SV, kWgT, QK_CHUNK>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (attr != cudaSuccess) return static_cast<int>(attr);  // smem over cap
        init_sk = true;
    }
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    m8_attn_partial_splitkv_dyn<SK, SV, kWgT, QK_CHUNK><<<dim3(ns), 128 * kWgT, smem, st>>>(
        Kmap, Vmap, Qmap, static_cast<float*>(wm8), static_cast<float*>(wl8),
        static_cast<bf16*>(wa8), static_cast<const int*>(ctl), ns, scaling);
    constexpr int DT = 64, BLK = M8_COMBINE_THREADS;
    // The combine's dynamic smem is ns*(DT*2 + 8) bytes, so it crosses the 48 KiB
    // default opt-in at ns ~= 361. Production never exceeds 132 so this never
    // fires in the shipped configuration; it is retained so that any future ns
    // exploration fails loudly rather than silently refusing to launch.
    const int csmem =
        ns * DT * static_cast<int>(sizeof(bf16)) + 2 * ns * static_cast<int>(sizeof(float));
    static int csmem_set = 0;
    if (csmem > 48 * 1024 && csmem > csmem_set) {
        const cudaError_t cattr = cudaFuncSetAttribute(
            m8_attn_combine_xeff<DT, BLK>, cudaFuncAttributeMaxDynamicSharedMemorySize, csmem);
        if (cattr != cudaSuccess) return static_cast<int>(cattr);
        csmem_set = csmem;
    }
    m8_attn_combine_xeff<DT, BLK>
        <<<dim3(NH * HD / DT, MROWS8), BLK, ns * DT * sizeof(bf16) + 2 * ns * sizeof(float), st>>>(
            static_cast<const float*>(wm8), static_cast<const float*>(wl8),
            static_cast<const bf16*>(wa8), static_cast<bf16*>(ctx8),
            static_cast<const bf16*>(gate8), static_cast<bf16*>(xeff8), ns, scaling);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M8
using namespace nexrt::sm90::nex_n2_5_pro::M8;
extern "C" {

// Production M=8 non-CP attention core: split-K/V staging with independent
// completion barriers, 132 CTA decomposition, and coalesced BF16 epilogue.
// K and V carry independent per-64D-chunk completion
// barriers, so QK begins when K chunk 0 lands rather than when the whole K+V
// stage does, and V streams under QK / softmax / P-store, awaited only
// immediately before the PV atoms that read it. Ring depths are independent
// (SK+SV is the smem budget); (2,2) was the best single config across dynamic S
// by lowest max-regret against the per-context optimum, and costs 152576 B --
// 65 KiB LESS than the (3,3) form.
// The accumulator epilogue stages exact BF16 words in dead K storage and drains
// complete (row,head) vectors with 16-byte-per-lane stores.
int NexN25ProM8AttnCoreLaunch(const void* q_out8, const void* K, const void* V, const void* ctl,
                              int ns, float scaling, void* wm8, void* wl8, void* wa8, void* ctx8,
                              const void* gate8, void* xeff8, void* stream) {
    return m8_launch_splitkv<4, 2, 2, 64>(q_out8, K, V, ctl, ns, scaling, wm8, wl8, wa8, ctx8,
                                          gate8, xeff8, stream);
}

int NexN25ProM8AttnCoreRows() { return MROWS8; }
}
