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

// M=8 CONTEXT-PARALLEL full attention: front, partial, local combine/publish and
// four-rank LSE merge. ADDITIVE -- nothing here is reachable from the shipped
// non-CP M=8 graph, and the non-CP TU is not touched.
//
// M=8-FAMILY TU: native eight-row implementation. NO BS=1 or M=4 kernel is called
// from, or linked into, the M=8 CP graph. (The BS=1 CP arm in
// nex_n2_5_pro_cp_hopper_mega_ctypes.cu / nex_n2_5_pro_cp_tail_ctypes.cu and the M=4 CP arm in
// nex_n2_5_pro_m4_cp_control_ctypes.cu were read as PROTOCOL references only.)
//
// TOPOLOGY. TP=8 is two independent CP groups, ranks [0-3] and [4-7]. cp_rank =
// rank % 4, cp_gid = rank / 4. Each group owns ONE KV head and every rank in the
// group holds a full replica of that head's KV cache: read windows are sharded,
// storage is replicated. Under CP a rank computes ALL 16 group q-heads over its
// 1/4 key shard instead of its own 4 heads over the whole cache; after the merge
// it keeps only its own four heads, [8, 4, 256].
//
// THE ROW PLAN (per target row j, j in [0,8), for unit base C = committed_len):
//
//     total_j = C + j + 1                 // identical to the non-CP causal extent
//     base_j  = total_j / CP              // CP == 4
//     off_j   = cp_rank * base_j
//     loc_j   = cp_rank < CP-1 ? base_j : total_j - (CP-1) * base_j
//
// Rank CP-1 absorbs the remainder, so the four windows exactly tile [0, total_j)
// with no gap and no overlap. An empty shard (loc_j == 0) publishes the neutral
// flash element m = -1e30, l = 0, acc = 0 -- the same contract the non-CP core's
// empty-split guard already publishes, and the identity the merge needs.
//
// BOTH bounds are enforced, per row, on every masked quantity: the running-max
// input, the exp/P store, and therefore the value accumulation. Using one row's
// bounds for all eight rows is the bug this comment exists to prevent.
//
// WHY THE PARTIAL IS DERIVED FROM THE NON-CP q-as-B CORE, NOT FROM M4's QUAD.
// M4's packed quad kernel fills WGMMA's M=64 exactly with 4 drafts x 16 heads.
// M=8 x 16 heads is 128 query rows -- twice that tile -- and that kernel is
// already register-bound carrying a live [64 q x 256 d] fp32 accumulator. The
// non-CP M=8 core solves eight rows the other way: q is the B operand, columns
// are computed as n8 column-groups over ONE staged K/V, and widening the head
// count scales the GROUP LOOP, not the live accumulator.
//
// HEAD-GROUP SPLIT (kHS). 16 heads x 8 rows = 128 columns = 16 n8 groups. Giving
// all 16 to one CTA of four warpgroups would put GPW = 4 groups (64 fp32) in
// every thread's accumulator -- four times the shipped core's live state, which
// spills at 512 threads. Instead the head axis is split across the grid's y
// dimension: CTA (split, hg) owns heads [hg*16/kHS, (hg+1)*16/kHS). At kHS == 4
// each CTA carries 8 rows x 4 heads = 32 columns = 4 groups = GPW 1, which is
// EXACTLY the shipped non-CP core's live accumulator -- the register profile is
// unchanged by construction. The kHS CTAs of one split read the same K/V tiles
// concurrently at 1 CTA/SM, so the repeat reads are L2-served; that claim is
// selected from measured whole-graph behavior.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M8
#endif

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "../common/m8_control_block.cuh"

namespace nexrt::sm90::nex_n2_5_pro::M8 {
using bf16 = __nv_bfloat16;

constexpr int HD = 256;
constexpr int NHCP = 16;  // CP q heads computed per rank
constexpr int NHL = 4;    // local heads the rank keeps after the merge
constexpr int TILE = 64;
constexpr int MROWS8 = m8ctl::kM;     // 8 target rows
constexpr int CPR = 4;                // ranks per CP group
constexpr int QCOLS = MROWS8 * NHCP;  // 128 query columns
constexpr int KV_CAPACITY = 256 * 1024 + 1024;
constexpr int kHid = 4096;
// FAIL-CLOSED SPIN BUDGET. The M=4/BS=1 CP arms use 1<<30, which MEASURES at
// 274.9 s on this part (~256 ns per iteration: __nanosleep(64) plus the
// ld.acquire.sys round trip). A four-and-a-half-minute stall is not an
// acceptable fail-closed latency for a serving path, so the M=8 arm bounds it.
// Budget derivation: the longest LEGITIMATE wait is one peer's whole partial,
// about 200 us at 256K context; sanitizers inflate that by 10-100x. 1<<21
// iterations is about 536 ms -- roughly 2700x the legitimate wait, still safe
// under compute-sanitizer, and 500x shorter than the inherited value.
#ifndef M8_CP_MAX_SPINS
#define M8_CP_MAX_SPINS (1 << 21)
#endif
constexpr int kMaxSpins = M8_CP_MAX_SPINS;
constexpr uint64_t TMA_EVICT_FIRST = 0x12f0000000000000ULL;

// CP qgate weight row map, matching attn.convert_weights_cp's
// cat([q16, gate4, k1, v1]) -> [5632, 4096]:
constexpr int kCpQRows = NHCP * HD;                 // 4096
constexpr int kGateRows = NHL * HD;                 // 1024
constexpr int kCpKEnd = kCpQRows + kGateRows + HD;  // 5376
constexpr int kCpTotalRows = kCpKEnd + HD;          // 5632
constexpr int kScaleBlocksK = kHid / 128;           // 32

// ---------------------------------------------------------------------------
// shared low-level helpers (same PTX the non-CP M=8 core uses)
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint64_t cp_cvta(const void* p) {
    return static_cast<uint64_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ uint64_t cp_make_desc(const void* p, uint32_t lbo, uint32_t sbo,
                                                 uint32_t swizzle) {
    const uint64_t a = (cp_cvta(p) >> 4) & 0x3fffull;
    return a | (static_cast<uint64_t>(lbo & 0x3fff) << 16) |
           (static_cast<uint64_t>(sbo & 0x3fff) << 32) | (static_cast<uint64_t>(swizzle & 3) << 62);
}
__device__ __forceinline__ void cp_wg_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void cp_wg_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ void cp_mbar_init(uint64_t* b, int count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(static_cast<uint32_t>(cp_cvta(b))),
                 "r"(count));
}
__device__ __forceinline__ void cp_mbar_expect(uint64_t* b, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(
                     static_cast<uint32_t>(cp_cvta(b))),
                 "r"(bytes));
}
__device__ __forceinline__ void cp_mbar_arrive(uint64_t* b) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" ::"r"(static_cast<uint32_t>(cp_cvta(b)))
                 : "memory");
}
__device__ __forceinline__ void cp_mbar_wait(uint64_t* b, int phase) {
    asm volatile(
        "{\n.reg .pred p;\nL%=:\n"
        "mbarrier.try_wait.parity.shared.b64 p,[%0],%1;\n"
        "@!p bra L%=;\n}\n" ::"r"(static_cast<uint32_t>(cp_cvta(b))),
        "r"(phase));
}
__device__ __forceinline__ void cp_tma_load(const CUtensorMap* map, uint64_t* bar, void* dst,
                                            int c0, int c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(
            static_cast<uint32_t>(cp_cvta(dst))),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1),
        "r"(static_cast<uint32_t>(cp_cvta(bar)))
        : "memory");
}
__device__ __forceinline__ void cp_tma_load_kv(const CUtensorMap* map, uint64_t* bar, void* dst,
                                               int c0, int c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1, {%2, %3}], [%4], %5;\n" ::"r"(static_cast<uint32_t>(cp_cvta(dst))),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1),
        "r"(static_cast<uint32_t>(cp_cvta(bar))), "l"(TMA_EVICT_FIRST)
        : "memory");
}
__device__ __forceinline__ void cp_prefetch_map(const CUtensorMap* map) {
    asm volatile("prefetch.tensormap [%0];\n" ::"l"(reinterpret_cast<uint64_t>(map)));
}

// ---------------------------------------------------------------------------
template <int kWgT>
__device__ __forceinline__ void cp_wg_sync(int wg) {
    if (kWgT == 1) {
        asm volatile("bar.sync 1, 128;\n" ::: "memory");
    } else {
        asm volatile("bar.sync %0, 128;\n" ::"r"(wg + 1) : "memory");
    }
}
__device__ __forceinline__ float cp_butterfly(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float cp_warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float cp_sigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ int cp_ld_acquire_sys(const int* p) {
    int v;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
// multimem.red.release.sys.global.add.s32: ONE release-ordered add landing in
// every CP peer's copy of the barrier line. Verified against PTX ISA 9.1
// (multimem.red requires .sys/.gpu scope with an ordinary .add op on a
// multicast address; it is a REDUCTION with no return value, which is why the
// consumer polls its own local copy with ld.acquire.sys instead of reading a
// returned count -- multimem.st and multimem.red are not interchangeable here).
__device__ __forceinline__ void cp_multimem_red_add_release(unsigned long long mc, int v) {
    asm volatile("multimem.red.release.sys.global.add.s32 [%0], %1;" ::"l"(mc), "r"(v) : "memory");
}

// The frozen row plan. Returns [lo, hi) absolute key bounds of THIS rank's shard
// for target row j. Callers must apply both bounds.
__device__ __forceinline__ void cp_row_window(int committed_len, int j, int cp_rank, int* lo,
                                              int* hi) {
    const int total = committed_len + j + 1;
    const int base = total / CPR;
    *lo = cp_rank * base;
    *hi = (cp_rank < CPR - 1) ? (*lo + base) : total;
}

CUtensorMap cp_encode_map(void* ptr, int outer, int inner, int box_outer, int box_inner) {
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

// ===========================================================================
// FRONT 1/2 -- CP qkv+gate projection, eight rows in one pass.
//
// Mechanism: the M=8 non-CP QKV kernel (M8QkvBf16TcKernel) re-pointed at the CP
// [5632, 4096] fp8 qgate weight. Same lane-major prepack, same transposed
// mma.m16n8k16 with all eight n columns real, same fp32 accumulation and the
// same one-shot cp.async x slice. Only the output row map differs, because the
// CP weight is cat([q16, gate4, k1, v1]) while the non-CP one interleaves
// q|gate per head.
//
// K and V are written ONCE per row, at C+j, by the single row that owns them.
// ===========================================================================
__device__ __forceinline__ uint2 cp_load_weight_uint2(const void* ptr) {
    uint2 value;
    asm volatile("ld.global.cs.v2.u32 {%0,%1}, [%2];" : "=r"(value.x), "=r"(value.y) : "l"(ptr));
    return value;
}
__device__ __forceinline__ void cp_mma_m16n8k16_bf16(float (&c)[4], const unsigned (&a)[4],
                                                     const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void cp_scatter_projected(int row, float acc, int m,
                                                     bf16* __restrict__ q_group8,  // [8, 16, 256]
                                                     bf16* __restrict__ gate8,     // [8, 4, 256]
                                                     bf16* __restrict__ k_cache,
                                                     bf16* __restrict__ v_cache, int pos) {
    const bf16 val = __float2bfloat16(acc);
    if (row < kCpQRows) {
        q_group8[static_cast<size_t>(m) * kCpQRows + row] = val;
    } else if (row < kCpQRows + kGateRows) {
        gate8[static_cast<size_t>(m) * kGateRows + (row - kCpQRows)] = val;
    } else if (row < kCpKEnd) {
        k_cache[static_cast<size_t>(pos) * HD + (row - kCpQRows - kGateRows)] = val;
    } else {
        v_cache[static_cast<size_t>(pos) * HD + (row - kCpKEnd)] = val;
    }
}

__global__ __launch_bounds__(256) void M8CpQGateKernel(
    const bf16* __restrict__ x8,              // [8, 4096]
    const __nv_fp8_e4m3* __restrict__ Wpack,  // lane-major prepack of [5632,4096]
    const float* __restrict__ scales,         // [44, 32]
    bf16* __restrict__ q_group8,              // [8, 16, 256]
    bf16* __restrict__ gate8,                 // [8, 4, 256]
    bf16* __restrict__ k_cache, bf16* __restrict__ v_cache, const int* __restrict__ ctl) {
    const int pos0 = m8ctl::CommittedLen(ctl);
    constexpr int kWarpK = 512;
    constexpr int kXPitch = kWarpK * 2 + 16;
    __shared__ __align__(16) char sx[8][MROWS8][kXPitch];
    __shared__ float red[8][32][2];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int gid = lane >> 2;
    const int tig = lane & 3;
    const int row0 = blockIdx.x * 8;
    const int k0 = warp * kWarpK;
    char* const sxw = &sx[warp][0][0];

#pragma unroll
    for (int m = 0; m < MROWS8; ++m) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t dst = static_cast<uint32_t>(
                __cvta_generic_to_shared(sxw + m * kXPitch + h * 512 + lane * 16));
            const void* src = x8 + static_cast<size_t>(m) * kHid + k0 + (h * 512 + lane * 16) / 2;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
        }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncwarp();

    const float* ws = scales + static_cast<size_t>(row0 >> 7) * kScaleBlocksK + warp * 4;
    float sblk[4];
#pragma unroll
    for (int b_ = 0; b_ < 4; ++b_) {
        sblk[b_] = ws[b_];
    }

    const __nv_fp8_e4m3* wp_base =
        Wpack + (static_cast<size_t>(blockIdx.x) * 128 + (k0 >> 5)) * 256 + lane * 8;
    float sacc[2] = {0.0f, 0.0f};
    uint2 wbuf[4];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        wbuf[i] = cp_load_weight_uint2(wp_base + i * 256);
    }
#pragma unroll
    for (int blk = 0; blk < 4; ++blk) {
        float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int t32 = 0; t32 < 4; ++t32) {
            const int ci = blk * 4 + t32;
            if (ci + 3 < 16) {
                wbuf[(ci + 3) & 3] = cp_load_weight_uint2(wp_base + (ci + 3) * 256);
            }
            const uint2 raw = wbuf[ci & 3];
            const __nv_fp8x2_e4m3* wp = reinterpret_cast<const __nv_fp8x2_e4m3*>(&raw);
            unsigned awf[4];
#pragma unroll
            for (int p_ = 0; p_ < 4; ++p_) {
                const __nv_bfloat162 bb = __float22bfloat162_rn(static_cast<float2>(wp[p_]));
                awf[p_] = *reinterpret_cast<const unsigned*>(&bb);
            }
#pragma unroll
            for (int s_ = 0; s_ < 2; ++s_) {
                const char* xp =
                    sxw + gid * kXPitch + (blk * 128 + t32 * 32 + s_ * 16) * 2 + tig * 4;
                unsigned b[2];
                b[0] = *reinterpret_cast<const unsigned*>(xp);
                b[1] = *reinterpret_cast<const unsigned*>(xp + 16);
                const unsigned a[4] = {awf[s_ * 2], 0u, awf[s_ * 2 + 1], 0u};
                cp_mma_m16n8k16_bf16(c, a, b);
            }
        }
        sacc[0] += sblk[blk] * c[0];
        sacc[1] += sblk[blk] * c[1];
    }

    red[warp][lane][0] = sacc[0];
    red[warp][lane][1] = sacc[1];
    __syncthreads();
    if (threadIdx.x < 32) {
        float v0 = 0.0f, v1 = 0.0f;
#pragma unroll
        for (int sp = 0; sp < 8; ++sp) {
            v0 += red[sp][lane][0];
            v1 += red[sp][lane][1];
        }
        cp_scatter_projected(row0 + gid, v0, tig * 2, q_group8, gate8, k_cache, v_cache,
                             pos0 + tig * 2);
        cp_scatter_projected(row0 + gid, v1, tig * 2 + 1, q_group8, gate8, k_cache, v_cache,
                             pos0 + tig * 2 + 1);
    }
}

// ===========================================================================
// FRONT 2/2 -- CP norm + partial RoPE for Q [8, 16, 256].
//
// The M=8 non-CP norm/rope kernel with kNqHeads 4 -> 16. Grid (18, 8):
// x in [0,16) = a CP q head, 16 = the K head in place at k_cache[pos_j], 17 =
// the input-RMS scalar that scales v_cache[pos_j] and this row's four gate
// heads in place. Rows stay mutually independent, so finalizing all eight
// before the partial runs is value-invisible to a causal reader.
//
// Q OUTPUT LAYOUT IS HEAD-GROUP MAJOR: q_out8 is [kHS][8 rows][16/kHS heads][256]
// so that CTA (split, hg) can pull its 8 x (16/kHS) columns with ONE TMA box.
// hs_stride/hg_heads carry that mapping; at kHS == 1 it degenerates to the
// natural [8, 16, 256].
// ===========================================================================
template <int T>
__global__ __launch_bounds__(T) void M8CpNormRopeKernel(
    const bf16* __restrict__ q_raw8, bf16* __restrict__ k_cache, const bf16* __restrict__ qn,
    const bf16* __restrict__ kn, const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab, bf16* __restrict__ q_out8, const bf16* __restrict__ x8,
    bf16* __restrict__ v_cache, bf16* __restrict__ gate8, int hid, float eps_in, int hg_heads,
    const int* __restrict__ ctl) {
    constexpr int VPT = HD / T, W = T / 32;
    constexpr int kRot = 64;
    constexpr float kEps = 1e-6f;
    const int cta = blockIdx.x, tid = threadIdx.x;
    const int row = blockIdx.y;
    const int pos = m8ctl::CommittedLen(ctl) + row;
    const bf16* q_raw = q_raw8 + static_cast<size_t>(row) * kCpQRows;
    const bf16* x_in = x8 + static_cast<size_t>(row) * hid;
    bf16* gate = gate8 + static_cast<size_t>(row) * kGateRows;

    if (cta < NHCP + 1) {
        __shared__ __align__(16) bf16 x_sh[HD], w_sh[HD];
        __shared__ float red[W > 1 ? W : 1];
        const bf16* src;
        const bf16* wt;
        bf16* dst;
        if (cta < NHCP) {
            src = q_raw + cta * HD;
            wt = qn;
            // head-group-major destination: head cta -> group cta/hg_heads,
            // slot cta%hg_heads.
            const int hg = cta / hg_heads, within = cta - hg * hg_heads;
            dst = q_out8 + ((static_cast<size_t>(hg) * MROWS8 + row) * hg_heads + within) * HD;
        } else {
            bf16* kp = k_cache + static_cast<size_t>(pos) * HD;
            src = kp;
            wt = kn;
            dst = kp;
        }
        const int d0 = tid * VPT;
        float xr[VPT], wr[VPT];
#pragma unroll
        for (int e = 0; e < VPT; ++e) {
            xr[e] = __bfloat162float(src[d0 + e]);
            wr[e] = __bfloat162float(wt[d0 + e]);
            x_sh[d0 + e] = src[d0 + e];
            w_sh[d0 + e] = wt[d0 + e];
        }
        float ss = 0.f;
#pragma unroll
        for (int e = 0; e < VPT; ++e) ss += xr[e] * xr[e];
        ss = cp_butterfly(ss);
        if constexpr (W > 1) {
            const int warp = tid >> 5, lane = tid & 31;
            if (lane == 0) red[warp] = ss;
            __syncthreads();
            ss = 0.f;
#pragma unroll
            for (int w = 0; w < W; ++w) ss += red[w];
        } else {
            __syncthreads();
        }
        const float rms = rsqrtf(ss * (1.0f / HD) + kEps);
        const float* cosv = cos_tab + static_cast<size_t>(pos) * (kRot / 2);
        const float* sinv = sin_tab + static_cast<size_t>(pos) * (kRot / 2);
        float result[VPT];
#pragma unroll
        for (int e = 0; e < VPT; ++e) {
            const int d = d0 + e;
            const float nd = xr[e] * rms * (1.0f + wr[e]);
            if (d < kRot) {
                const int p = d ^ 32;
                const float np =
                    __bfloat162float(x_sh[p]) * rms * (1.0f + __bfloat162float(w_sh[p]));
                const int ang = d & 31;
                const float c = cosv[ang], s = sinv[ang];
                result[e] = (d < 32) ? (nd * c - np * s) : (nd * c + np * s);
            } else {
                result[e] = nd;
            }
        }
        static_assert(VPT % 2 == 0, "packed CP norm stores require BF16 pairs");
#pragma unroll
        for (int e = 0; e < VPT; e += 2) {
            const __nv_bfloat162 pair = __floats2bfloat162_rn(result[e], result[e + 1]);
            *reinterpret_cast<__nv_bfloat162*>(dst + d0 + e) = pair;
        }
    } else {
        __shared__ float sred[W > 1 ? W : 1];
        float ss = 0.f;
        for (int i = tid; i < hid; i += T) {
            const float xi = __bfloat162float(x_in[i]);
            ss += xi * xi;
        }
        ss = cp_butterfly(ss);
        if constexpr (W > 1) {
            const int warp = tid >> 5, lane = tid & 31;
            if (lane == 0) sred[warp] = ss;
            __syncthreads();
            ss = 0.f;
#pragma unroll
            for (int w = 0; w < W; ++w) ss += sred[w];
        }
        const float s = rsqrtf(ss * (1.0f / hid) + eps_in);
        bf16* vp = v_cache + static_cast<size_t>(pos) * HD;
        __nv_bfloat162* vp2 = reinterpret_cast<__nv_bfloat162*>(vp);
        for (int i = tid; i < HD / 2; i += T) {
            const float2 v = __bfloat1622float2(vp2[i]);
            vp2[i] = __floats2bfloat162_rn(v.x * s, v.y * s);
        }
        __nv_bfloat162* gate2 = reinterpret_cast<__nv_bfloat162*>(gate);
        for (int i = tid; i < kGateRows / 2; i += T) {
            const float2 v = __bfloat1622float2(gate2[i]);
            gate2[i] = __floats2bfloat162_rn(v.x * s, v.y * s);
        }
    }
}

// ===========================================================================
__device__ __forceinline__ uint32_t cp_pack_bf16(float a, float b) {
    uint32_t r;
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(b), "f"(a));
    return r;
}

// SS m64n64k16: D[8 n8-blocks][4], A and B from SMEM descriptors, no transpose
// (both operands K-major): 32 accumulator outputs plus two SMEM descriptors.
__device__ __forceinline__ void cp_wgmma_ss_n64(float d[8][4], uint64_t da, uint64_t db) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, 1,1,1,0,0;\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
          "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
          "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
          "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]),
          "+f"(d[7][2]), "+f"(d[7][3])
        : "l"(da), "l"(db));
}

// RS m64n64k16: A = P from registers {a0..a3}, B = V descriptor, trans-b = 1
// (V subtiles are [64 keys][64 dims], dims contiguous = N-major).
__device__ __forceinline__ void cp_wgmma_rs_n64(float d[8][4], const uint32_t a[4], uint64_t db) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "{%32,%33,%34,%35}, %36, 1,1,1,1;\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
          "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
          "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
          "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]),
          "+f"(d[7][2]), "+f"(d[7][3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
}

// Production M8 CP partial: one TMA producer warpgroup, two WGMMA consumer
// warpgroups, two-stage K/V rings, resident Q, and a coalesced epilogue.
template <int SK, int SV>
__global__ __launch_bounds__(384, 1) void m8_cp_partial_wsx(
    const __grid_constant__ CUtensorMap Kmap, const __grid_constant__ CUtensorMap Vmap,
    const __grid_constant__ CUtensorMap Qmap, float* __restrict__ ws_m, float* __restrict__ ws_l,
    bf16* __restrict__ ws_acc, const int* __restrict__ ctl, int ns, float scaling, int cp_rank) {
    constexpr int QKC = HD / 64, PVC = HD / 64;
    constexpr int KCH = TILE * 64, VCH = TILE * 64;
    constexpr int KCH_B = KCH * static_cast<int>(sizeof(bf16));
    constexpr int VCH_B = VCH * static_cast<int>(sizeof(bf16));
    constexpr int QSUB = 64 * 64;

    const int Sbase = m8ctl::CommittedLen(ctl);
    const int split = blockIdx.x;
    const int tid = threadIdx.x, wg = tid >> 7, lt = tid & 127;
    const int warp = lt >> 5, lane = tid & 31, gid = lane >> 2, tg = lane & 3;
    const float scale2 = scaling * 1.4426950408889634f;

    __shared__ int sh_lo[MROWS8], sh_hi[MROWS8], sh_w[2];
    if (tid < MROWS8) {
        int lo, hi;
        cp_row_window(Sbase, tid, cp_rank, &lo, &hi);
        sh_lo[tid] = lo;
        sh_hi[tid] = hi;
    }
    if (tid == 0) {
        int w0 = 0x7fffffff, w1 = 0;
#pragma unroll
        for (int j = 0; j < MROWS8; ++j) {
            int lo, hi;
            cp_row_window(Sbase, j, cp_rank, &lo, &hi);
            if (hi > lo) {
                w0 = min(w0, lo);
                w1 = max(w1, hi);
            }
        }
        if (w1 <= w0) {
            w0 = 0;
            w1 = 0;
        }
        sh_w[0] = w0;
        sh_w[1] = w1;
    }
    __syncthreads();
    const int W0 = sh_w[0], WS = sh_w[1] - sh_w[0];
    const int base0 = W0 + static_cast<int>((static_cast<long long>(WS) * split) / ns);
    const int end = W0 + static_cast<int>((static_cast<long long>(WS) * (split + 1)) / ns);
    const int kv_per = end - base0;
    const int ntiles = (kv_per + TILE - 1) / TILE;

    if (kv_per == 0) {  // uniform for all 384 threads
        for (int i = tid; i < MROWS8 * NHCP; i += 384) {
            const int r = i / NHCP, h = i - r * NHCP;
            ws_m[(static_cast<size_t>(r) * ns + split) * NHCP + h] = -1.0e30f;
            ws_l[(static_cast<size_t>(r) * ns + split) * NHCP + h] = 0.0f;
        }
        // Each row workspace is 8 KiB and 16-byte aligned.  Do not emit a
        // scalar BF16 zero stream on the active empty-split path.
        constexpr int kAccVecsPerRow = NHCP * HD * sizeof(bf16) / sizeof(uint4);
        const uint4 zero = make_uint4(0, 0, 0, 0);
        for (int r = 0; r < MROWS8; ++r) {
            uint4* const row_acc = reinterpret_cast<uint4*>(
                ws_acc + (static_cast<size_t>(r) * ns + split) * NHCP * HD);
            for (int i = tid; i < kAccVecsPerRow; i += 384) row_acc[i] = zero;
        }
        return;
    }

    extern __shared__ __align__(1024) char smem[];
    constexpr int KVB = TILE * HD;
    bf16* Qsh = reinterpret_cast<bf16*>(smem);
    bf16* Ksh = Qsh + QCOLS * HD;
    bf16* Vsh = Ksh + SK * KVB;
    __shared__ __align__(8) uint64_t bq[2];
    __shared__ __align__(8) uint64_t k_full[SK][QKC], k_empty[SK][QKC];
    __shared__ __align__(8) uint64_t v_full[SV][PVC], v_empty[SV][PVC];

    if (tid == 0) {
#pragma unroll
        for (int b = 0; b < 2; ++b) cp_mbar_init(&bq[b], 1);
#pragma unroll
        for (int st = 0; st < SK; ++st)
#pragma unroll
            for (int c = 0; c < QKC; ++c) {
                cp_mbar_init(&k_full[st][c], 1);
                cp_mbar_init(&k_empty[st][c], 2);  // the two consumer wgs
            }
#pragma unroll
        for (int st = 0; st < SV; ++st)
#pragma unroll
            for (int c = 0; c < PVC; ++c) {
                cp_mbar_init(&v_full[st][c], 1);
                cp_mbar_init(&v_empty[st][c], 2);
            }
        cp_prefetch_map(&Kmap);
        cp_prefetch_map(&Vmap);
        cp_prefetch_map(&Qmap);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    __syncthreads();

    if (wg == 2) {
        // ------------------------- PRODUCER -------------------------------
        // First instruction of the region: release registers to the pool.
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;\n");
        if (lt == 0) {
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                cp_mbar_expect(&bq[b], 4 * QSUB * static_cast<int>(sizeof(bf16)));
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    cp_tma_load(&Qmap, &bq[b], Qsh + b * (4 * QSUB) + j * QSUB, j * 64, b * 64);
            }
            for (int t = 0; t < min(SK, ntiles); ++t) {
                const int st = t % SK, key = base0 + t * TILE;
#pragma unroll
                for (int c = 0; c < QKC; ++c) {
                    cp_mbar_expect(&k_full[st][c], KCH_B);
                    cp_tma_load_kv(&Kmap, &k_full[st][c], Ksh + st * KVB + c * KCH, c * 64, key);
                }
            }
            for (int t = 0; t < min(SV, ntiles); ++t) {
                const int st = t % SV, key = base0 + t * TILE;
#pragma unroll
                for (int c = 0; c < PVC; ++c) {
                    cp_mbar_expect(&v_full[st][c], VCH_B);
                    cp_tma_load_kv(&Vmap, &v_full[st][c], Vsh + st * KVB + c * VCH, c * 64, key);
                }
            }
            // Refills, paced ONLY by the consumers' release barriers: K(t+SK)
            // the instant both consumers finished QK(t), V(t+SV) the instant
            // both finished PV(t).
            for (int t = 0; t < ntiles; ++t) {
                const int kst = t % SK, kph = (t / SK) & 1;
                const int vst = t % SV, vph = (t / SV) & 1;
                if (t + SK < ntiles) {
                    const int key = base0 + (t + SK) * TILE;
#pragma unroll
                    for (int c = 0; c < QKC; ++c) cp_mbar_wait(&k_empty[kst][c], kph);
#pragma unroll
                    for (int c = 0; c < QKC; ++c) {
                        cp_mbar_expect(&k_full[kst][c], KCH_B);
                        cp_tma_load_kv(&Kmap, &k_full[kst][c], Ksh + kst * KVB + c * KCH, c * 64,
                                       key);
                    }
                }
                if (t + SV < ntiles) {
                    const int key = base0 + (t + SV) * TILE;
#pragma unroll
                    for (int c = 0; c < PVC; ++c) cp_mbar_wait(&v_empty[vst][c], vph);
#pragma unroll
                    for (int c = 0; c < PVC; ++c) {
                        cp_mbar_expect(&v_full[vst][c], VCH_B);
                        cp_tma_load_kv(&Vmap, &v_full[vst][c], Vsh + vst * KVB + c * VCH, c * 64,
                                       key);
                    }
                }
            }
        }
        return;  // producer warpgroup exits
    }

    // --------------------------- CONSUMERS --------------------------------
    // First instruction of the region: acquire the released registers.
    // Pool: 384 x 168 = 64512 at launch -> 2 x 128 x 224 + 128 x 56 = 64512.
    asm volatile("setmaxnreg.inc.sync.aligned.u32 240;\n");
    const int j0 = 2 * warp, j1 = 2 * warp + 1;
    const int lo0 = sh_lo[j0], hi0 = sh_hi[j0];
    const int lo1 = sh_lo[j1], hi1 = sh_hi[j1];
    float o[PVC][8][4];
#pragma unroll
    for (int c = 0; c < PVC; ++c)
#pragma unroll
        for (int b = 0; b < 8; ++b)
#pragma unroll
            for (int e = 0; e < 4; ++e) o[c][b][e] = 0.0f;
    float m0 = -1.0e30f, m1 = -1.0e30f, l0 = 0.0f, l1 = 0.0f;
    const uint64_t qd = cp_make_desc(Qsh + wg * (4 * QSUB), 1, 64, 1);
    cp_mbar_wait(&bq[wg], 0);  // only THIS warpgroup's Q half

    // =====================================================================
    // FA-3 cross-tile overlap. Dependency schedule per tile t >= 1:
    //
    //   [top: NOTHING outstanding]
    //   zero s                       (generic write; legal: nothing in flight)
    //   wait k_full(t); fence; issue QK(t) x16; commit          -> group A
    //   wait v_full(t-1); fence; issue PV(t-1) from pf[cur]; commit -> group B
    //   wait_group<1>   : A (older) complete, B (newest) MAY PEND.
    //                     s is now readable; o and pf[cur] are UNTOUCHABLE.
    //   release k_empty(t)          (QK finished consuming K(t))
    //   mask + online softmax on s: READS s only; writes go to pf[nxt]
    //                     (the alternate P buffer, referenced by no in-flight
    //                     group) and to the scalar m/l state. No generic
    //                     instruction defines any register of A or B.
    //   wait_group<0>   : B drains. o and pf[cur] become free.
    //   release v_empty(t-1)
    //   rescale o by cf(t)          (o at scale m(t-1) after PV(t-1) ✓)
    //   swap cur/nxt
    //   [back edge: no accumulator group remains outstanding and no generic
    //    write lands inside an open pipeline region]
    //
    // Overlap achieved: QK(t) executes concurrently with PV(t-1)'s issue and
    // drain window; the ENTIRE softmax(t) (mask, max, exp, pack-into-alt)
    // runs while PV(t-1) is still pending. Deviation from the directive's
    // step order, documented: P(t) is packed into the ALTERNATE register
    // buffer before wait_group<0> rather than after -- storage that PV(t-1)
    // reads is never reused or overwritten, and the pack is hidden under the
    // overlap window too.
    float s[8][4];
    float e[8][4];      // exp-staging: softmax writes ONLY here + the
                        // scalar m/l state while a PV is in flight
    uint32_t pf[4][4];  // packed AFTER wait_group<0>, single buffer

    auto qk_issue = [&](int t) {
#pragma unroll
        for (int b = 0; b < 8; ++b)
#pragma unroll
            for (int e = 0; e < 4; ++e) s[b][e] = 0.0f;
        const int st = t % SK, ph = (t / SK) & 1;
#pragma unroll
        for (int c = 0; c < QKC; ++c) cp_mbar_wait(&k_full[st][c], ph);
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        const uint64_t kd = cp_make_desc(Ksh + st * KVB, 1, 64, 1);
#pragma unroll
        for (int c = 0; c < QKC; ++c)
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const uint64_t off = ((c * 8192 + kk * 32) >> 4);
                cp_wgmma_ss_n64(s, qd + off, kd + off);
            }
        cp_wg_commit();
    };
    auto pv_issue = [&](int t) {
        const int st = t % SV, ph = (t / SV) & 1;
#pragma unroll
        for (int c = 0; c < PVC; ++c) cp_mbar_wait(&v_full[st][c], ph);
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        const uint64_t vd = cp_make_desc(Vsh + st * KVB, 256, 64, 1);
#pragma unroll
        for (int c = 0; c < PVC; ++c)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                cp_wgmma_rs_n64(o[c], pf[j], vd + ((c * 8192 + j * 2048) >> 4));
        cp_wg_commit();
    };
    auto k_release = [&](int t) {
        if (lt == 0) {
            const int st = t % SK;
#pragma unroll
            for (int c = 0; c < QKC; ++c) cp_mbar_arrive(&k_empty[st][c]);
        }
    };
    auto v_release = [&](int t) {
        if (lt == 0) {
            const int st = t % SV;
#pragma unroll
            for (int c = 0; c < PVC; ++c) cp_mbar_arrive(&v_empty[st][c]);
        }
    };
    // Mask + online softmax on the COMPLETED s; writes pf[buf] and returns the
    // o-rescale factors via cf0/cf1. Reads s; never writes it.
    float cf0 = 1.0f, cf1 = 1.0f;
    auto softmax_stage = [&](int t) {
        const int tbase = base0 + t * TILE;
        const int valid = min(TILE, kv_per - t * TILE);
        const int a0 = max(0, lo0 - tbase), b0 = min(valid, hi0 - tbase);
        const int a1 = max(0, lo1 - tbase), b1 = min(valid, hi1 - tbase);
        float rm0 = -1.0e30f, rm1 = -1.0e30f;
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            const int k0 = b * 8 + tg * 2, k1 = k0 + 1;
            const float v00 = (k0 >= a0 && k0 < b0) ? s[b][0] : -1.0e30f;
            const float v01 = (k1 >= a0 && k1 < b0) ? s[b][1] : -1.0e30f;
            const float v10 = (k0 >= a1 && k0 < b1) ? s[b][2] : -1.0e30f;
            const float v11 = (k1 >= a1 && k1 < b1) ? s[b][3] : -1.0e30f;
            rm0 = fmaxf(rm0, fmaxf(v00, v01));
            rm1 = fmaxf(rm1, fmaxf(v10, v11));
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            rm0 = fmaxf(rm0, __shfl_xor_sync(0xffffffffu, rm0, off));
            rm1 = fmaxf(rm1, __shfl_xor_sync(0xffffffffu, rm1, off));
        }
        const float mn0 = fmaxf(m0, rm0), mn1 = fmaxf(m1, rm1);
        float rs0 = 0.0f, rs1 = 0.0f;
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            const int k0 = b * 8 + tg * 2, k1 = k0 + 1;
            const float p00 = (k0 >= a0 && k0 < b0) ? exp2f((s[b][0] - mn0) * scale2) : 0.0f;
            const float p01 = (k1 >= a0 && k1 < b0) ? exp2f((s[b][1] - mn0) * scale2) : 0.0f;
            const float p10 = (k0 >= a1 && k0 < b1) ? exp2f((s[b][2] - mn1) * scale2) : 0.0f;
            const float p11 = (k1 >= a1 && k1 < b1) ? exp2f((s[b][3] - mn1) * scale2) : 0.0f;
            rs0 += p00 + p01;
            rs1 += p10 + p11;
            e[b][0] = p00;
            e[b][1] = p01;
            e[b][2] = p10;
            e[b][3] = p11;
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            rs0 += __shfl_xor_sync(0xffffffffu, rs0, off);
            rs1 += __shfl_xor_sync(0xffffffffu, rs1, off);
        }
        cf0 = exp2f((m0 - mn0) * scale2);
        cf1 = exp2f((m1 - mn1) * scale2);
        l0 = l0 * cf0 + rs0;
        l1 = l1 * cf1 + rs1;
        m0 = mn0;
        m1 = mn1;
    };
    // pack ONLY when nothing is outstanding (the directive's step order).
    auto pack_pf = [&]() {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            pf[j][0] = cp_pack_bf16(e[2 * j][0], e[2 * j][1]);
            pf[j][1] = cp_pack_bf16(e[2 * j][2], e[2 * j][3]);
            pf[j][2] = cp_pack_bf16(e[2 * j + 1][0], e[2 * j + 1][1]);
            pf[j][3] = cp_pack_bf16(e[2 * j + 1][2], e[2 * j + 1][3]);
        }
    };
    auto rescale_o = [&]() {
#pragma unroll
        for (int c = 0; c < PVC; ++c)
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                o[c][b][0] *= cf0;
                o[c][b][1] *= cf0;
                o[c][b][2] *= cf1;
                o[c][b][3] *= cf1;
            }
    };

    // Bootstrap tile 0: serial (nothing to overlap with yet).
    qk_issue(0);
    cp_wg_wait<0>();
    k_release(0);
    softmax_stage(0);
    rescale_o();  // cf from softmax(0); o was zero -- harmless,
                  // kept so both store variants share math.
    pack_pf();
    for (int t = 1; t < ntiles; ++t) {
        qk_issue(t);      // group A (s zeroed inside; nothing in flight)
        pv_issue(t - 1);  // group B (newest), reads pf = P(t-1)
        cp_wg_wait<1>();  // A complete; B may pend
        k_release(t);
        softmax_stage(t);  // reads s; writes e + scalars ONLY
        cp_wg_wait<0>();   // B drains; o and pf now free
        v_release(t - 1);
        rescale_o();
        pack_pf();  // P(t) into pf, nothing outstanding
    }
    // Epilogue: last PV.
    pv_issue(ntiles - 1);
    cp_wg_wait<0>();
    v_release(ntiles - 1);

    const float inv0 = l0 > 0.0f ? 1.0f / l0 : 0.0f;
    const float inv1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    const int h = 8 * wg + gid;
    if (tg == 0) {
        ws_m[(static_cast<size_t>(j0) * ns + split) * NHCP + h] = m0;
        ws_l[(static_cast<size_t>(j0) * ns + split) * NHCP + h] = l0;
        ws_m[(static_cast<size_t>(j1) * ns + split) * NHCP + h] = m1;
        ws_l[(static_cast<size_t>(j1) * ns + split) * NHCP + h] = l1;
    }
    // K/V storage is dead after the final PV and producer exit. Stage
    // [8 rows][8 heads][128 BF16x2 words] per consumer warpgroup.
    constexpr int EPI_ROWS = 8, EPI_HEADS = 8;
    constexpr int EPI_STRIDE_WORDS = 132;
    constexpr int EPI_WG_WORDS = EPI_ROWS * EPI_HEADS * EPI_STRIDE_WORDS;
    static_assert(2 * EPI_WG_WORDS * static_cast<int>(sizeof(uint32_t)) <=
                      (SK + SV) * KVB * static_cast<int>(sizeof(bf16)),
                  "coalesced epilogue must fit in dead K/V shared storage");
    uint32_t* const epi = reinterpret_cast<uint32_t*>(Ksh) + wg * EPI_WG_WORDS;

#pragma unroll
    for (int c = 0; c < PVC; ++c)
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            const int p = c * 32 + b * 4 + tg;
            epi[(j0 * EPI_HEADS + gid) * EPI_STRIDE_WORDS + p] =
                cp_pack_bf16(o[c][b][0] * inv0, o[c][b][1] * inv0);
            epi[(j1 * EPI_HEADS + gid) * EPI_STRIDE_WORDS + p] =
                cp_pack_bf16(o[c][b][2] * inv1, o[c][b][3] * inv1);
        }

    // The producer warpgroup has returned. A CTA-wide barrier would
    // deadlock; ids 1 and 2 synchronize the two 128-thread consumers.
    cp_wg_sync<2>(wg);

// Four warps drain complete aligned 512-byte (row, head) vectors.
// Every lane moves one contiguous 16-byte word per vector.
#pragma unroll
    for (int vec = warp; vec < EPI_ROWS * EPI_HEADS; vec += 4) {
        const int row = vec >> 3;
        const int h_local = vec & 7;
        const uint4 v = reinterpret_cast<const uint4*>(epi + vec * EPI_STRIDE_WORDS)[lane];
        bf16* const dst =
            ws_acc + ((static_cast<size_t>(row) * ns + split) * NHCP + (8 * wg + h_local)) * HD;
        reinterpret_cast<uint4*>(dst)[lane] = v;
    }
}

// ===========================================================================
// LOCAL SPLIT REDUCTION -- level 1 of the two-level protocol.
//
// Merges THIS rank's ns flash partials into one normalized (ctx, lse) per
// (row, head), into the rank's OWN slice of the CP-group symmetric buffers.
// This is deliberately a separate kernel from the CP merge: the local split
// reduction is a private, launch-ordered operation with no cross-rank traffic,
// and the CP merge is a collective. Fusing them would hide a protocol cost in
// an arithmetic kernel.
//
// LSE CONVENTION. The partial's m is a RAW score max and l = sum exp((s-m)*
// scaling), so the natural-log LSE is scaling*m + ln(l). An all-masked
// (row, head) has l == 0 and publishes -FLT_MAX, which contributes exactly zero
// weight in the merge. At least one rank always has a non-empty shard, because
// rank CP-1 absorbs the remainder and total_j >= 1.
// ===========================================================================
template <int DT, int BLK>
__global__ __launch_bounds__(BLK) void M8CpLocalCombineKernel(
    const float* __restrict__ ws_m, const float* __restrict__ ws_l, const bf16* __restrict__ ws_acc,
    bf16* __restrict__ cp_ctx,   // [2][8, 16, 256]: gen-parity slices
    float* __restrict__ cp_lse,  // [2][8, 16]
    int ns, float scaling, int* __restrict__ it_ctr) {
    static_assert(DT >= 8 && DT % 8 == 0, "split gather works in 16B units");
    static_assert(BLK >= DT, "the reduction needs at least one group");
    constexpr int U = DT / 8;
    const int row = blockIdx.y;
    const int o0 = blockIdx.x * DT, h = o0 / HD, d0 = o0 % HD, tid = threadIdx.x;
    const float scale2 = scaling * 1.4426950408889634f;
    const float* wm = ws_m + static_cast<size_t>(row) * ns * NHCP;
    const float* wl = ws_l + static_cast<size_t>(row) * ns * NHCP;
    const bf16* wa = ws_acc + static_cast<size_t>(row) * ns * NHCP * HD;
    // WAR-race fix: publish generation g+1 into slice (g+1) & 1, so the
    // slice a straggling peer merge is still READING (generation g, the other
    // parity) is never the one being written. it_ctr is this rank's OWN
    // generation counter; during this kernel it is stable -- the bump happens
    // in the stream-ordered arrive kernel. Reading it per CTA keeps parity
    // correct under CUDA graph replay, where the host never re-intervenes.
    __shared__ int pub_par;
    if (threadIdx.x == 0) pub_par = (*it_ctr + 1) & 1;
    extern __shared__ __align__(16) char csmem[];
    bf16* sm = reinterpret_cast<bf16*>(csmem);
    float* m_sh = reinterpret_cast<float*>(sm + ns * DT);
    float* l_sh = m_sh + ns;
    __shared__ float red[BLK], den_red[BLK];

    for (int idx = tid; idx < ns * U; idx += BLK) {
        const int s = idx / U, u = idx - s * U;
        *reinterpret_cast<uint4*>(sm + s * DT + u * 8) = *reinterpret_cast<const uint4*>(
            wa + (static_cast<size_t>(s) * NHCP + h) * HD + d0 + u * 8);
    }
    for (int s = tid; s < ns; s += BLK) {
        m_sh[s] = wm[s * NHCP + h];
        l_sh[s] = wl[s * NHCP + h];
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
        // pub_par was written by thread 0 before the first __syncthreads().
        bf16* ctx_s = cp_ctx + static_cast<size_t>(pub_par) * (MROWS8 * NHCP * HD);
        float* lse_s = cp_lse + static_cast<size_t>(pub_par) * (MROWS8 * NHCP);
        ctx_s[(static_cast<size_t>(row) * NHCP + h) * HD + d0 + tid] =
            __float2bfloat16(n / (z + 1.0e-20f));
        // z is d-independent, so the CTA that owns dimension 0 publishes the LSE.
        if (tid == 0 && d0 == 0) {
            lse_s[static_cast<size_t>(row) * NHCP + h] =
                (z > 0.0f) ? (scaling * gm + logf(z)) : -3.402823466e38f;
        }
    }
}

// ===========================================================================
// PUBLICATION ARRIVAL.
//
// One thread. It runs as its OWN launch after the local combine, so the
// same-stream kernel boundary has already made every ctx/lse byte visible
// device-wide before this release-ordered multicast add is issued. That is
// what makes the protocol formally safe without a cooperative grid barrier or
// a hand-rolled software fence.
// ===========================================================================
__global__ __launch_bounds__(32) void M8CpArriveKernel(unsigned long long bar_mc,
                                                       int* __restrict__ it_ctr,
                                                       int* __restrict__ cur_gen,
                                                       int* __restrict__ status) {
    if (threadIdx.x != 0) return;
    const int gen = atomicAdd(it_ctr, 1) + 1;
    cur_gen[0] = gen;
    if (bar_mc == 0) {
        if (status) atomicCAS(status, 0, 3);  // 3 == no multicast channel
    } else {
        // SYS-scope fence before the release: the combine's ctx/lse writes are
        // in a PRIOR kernel (gpu-scope visible at this launch boundary); the
        // consumer is a remote rank, so promote them to sys scope here.
        __threadfence_system();
        cp_multimem_red_add_release(bar_mc, 1);
    }
}

// ===========================================================================
// CP-GROUP REDUCTION -- level 2. Four ranks, stable LSE algebra, reduce-scatter
// to the OWNING rank's four heads only.
//
// Grid (4 local heads x 8 dimension blocks, 8 rows) x 128 threads: warp r reads
// peer r. Every CTA's thread 0 acquires the barrier against generation*CP
// before any peer byte is read, so the wait is per-CTA and needs no grid-wide
// sync. Failure is level-triggered into cp_status, which the M=8 tail folds
// into its existing upstream-status decision: a failed CP round can never
// advance committed_len.
// ===========================================================================
__global__ __launch_bounds__(CPR * 32) void M8CpMergeKernel(
    const int64_t* __restrict__ peer_ctx, const int64_t* __restrict__ peer_lse,
    const bf16* __restrict__ gate8,  // [8, 4, 256] this rank's gate
    bf16* __restrict__ ctx_out,      // [8, 1024] merged, ungated (may be null)
    bf16* __restrict__ xeff_out,     // [8, 1024] merged * sigmoid(gate)
    const int* __restrict__ cp_bar_local, int* __restrict__ cp_status,
    const int* __restrict__ cp_gen, int cp_rank, const void* prefetch_base, long prefetch_bytes) {
    const int row = blockIdx.y;
    // Opportunistic o-projection L2 warm-up, sized to the wait window and issued
    // only from the last row's CTAs (the family's prefetch law: target the
    // immediately-next consumer, size the payload to the window).
    if (row == MROWS8 - 1 && prefetch_base != nullptr) {
        const char* p = static_cast<const char*>(prefetch_base);
        const long lines = prefetch_bytes >> 7;
        for (long line = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x; line < lines;
             line += static_cast<long>(gridDim.x) * blockDim.x) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(p + (line << 7)));
        }
    }

    __shared__ int s_gen;
    if (threadIdx.x == 0) {
        const int gen = cp_ld_acquire_sys(cp_gen);
        s_gen = gen;
        if (gen <= 0 && cp_status != nullptr) atomicCAS(cp_status, 0, 2);
        const int target = gen * CPR;
        int spins = 0;
        while (static_cast<int>(cp_ld_acquire_sys(cp_bar_local) - target) < 0) {
            if (++spins > kMaxSpins) {
                if (cp_status != nullptr) atomicCAS(cp_status, 0, 1);
                break;
            }
            __nanosleep(64);
        }
    }
    __syncthreads();
    // Generation g was published into parity slice g & 1.
    const int par = s_gen & 1;

    const int cta = blockIdx.x;
    const int local_head = cta >> 3;
    const int dblock = cta & 7;
    const int head = cp_rank * NHL + local_head;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int d = dblock * 32 + lane;

    __shared__ float lse_vals[CPR];
    __shared__ float weights[CPR];
    __shared__ float inv_denom;
    __shared__ float rank_part[CPR][32];

    if (threadIdx.x < CPR) {
        const float* lse = reinterpret_cast<const float*>(peer_lse[threadIdx.x]) +
                           static_cast<size_t>(par) * (MROWS8 * NHCP) +
                           static_cast<size_t>(row) * NHCP;
        lse_vals[threadIdx.x] = lse[head];
    }
    __syncthreads();
    const float m = fmaxf(fmaxf(lse_vals[0], lse_vals[1]), fmaxf(lse_vals[2], lse_vals[3]));
    if (threadIdx.x < CPR) weights[threadIdx.x] = __expf(lse_vals[threadIdx.x] - m);
    __syncthreads();
    if (threadIdx.x == 0)
        inv_denom = 1.0f / (weights[0] + weights[1] + weights[2] + weights[3] + 1.0e-30f);
    __syncthreads();
    if (threadIdx.x < CPR) weights[threadIdx.x] *= inv_denom;
    __syncthreads();

    const bf16* ctx = reinterpret_cast<const bf16*>(peer_ctx[warp]) +
                      static_cast<size_t>(par) * (MROWS8 * NHCP * HD) +
                      static_cast<size_t>(row) * NHCP * HD;
    rank_part[warp][lane] = weights[warp] * __bfloat162float(ctx[head * HD + d]);
    __syncthreads();

    if (warp == 0) {
        const float acc =
            rank_part[0][lane] + rank_part[1][lane] + rank_part[2][lane] + rank_part[3][lane];
        const size_t o = static_cast<size_t>(row) * (NHL * HD) + local_head * HD + d;
        if (ctx_out != nullptr) ctx_out[o] = __float2bfloat16(acc);
        const float g = __bfloat162float(gate8[o]);
        xeff_out[o] = __float2bfloat16(acc * cp_sigmoid(g));
    }
    __syncthreads();
    // Fail-closed tripwire: slice par is overwritten by publication
    // gen+2, and any rank that has ARRIVED for gen+2 pushed bar past a full
    // gen+1 round. If that already happened, this CTA's reads above may have
    // overlapped the overwrite -- status 4, host retries, same contract as
    // every other failure latch. HONEST LIMIT: a combine(gen+2) that has
    // started writing but not yet arrived is not heralded by bar, so a merge
    // stalled ~2 full generations mid-read has a microsecond-scale window this
    // check cannot see. Reaching it requires the straggler to ALSO have lagged
    // through two complete front+partial+combine rounds of every peer.
    if (threadIdx.x == 0 && cp_status != nullptr) {
        if (static_cast<int>(cp_ld_acquire_sys(cp_bar_local) - (s_gen + 1) * CPR) > 0)
            atomicCAS(cp_status, 0, 4);
    }
}

// ---------------------------------------------------------------------------
// launchers
// ---------------------------------------------------------------------------
template <int SK, int SV>
static constexpr int cp_partial_wsx_smem_bytes() {
    return (QCOLS * HD + (SK + SV) * TILE * HD) * static_cast<int>(sizeof(bf16)) + 1024;
}

static int cp_launch_local_combine(const void* wm, const void* wl, const void* wa, void* cp_ctx,
                                   void* cp_lse, int ns, float scaling, void* it_ctr,
                                   void* stream) {
    constexpr int DT = 64, BLK = 256;
    const int csmem =
        ns * DT * static_cast<int>(sizeof(bf16)) + 2 * ns * static_cast<int>(sizeof(float));
    static int csmem_set = 0;
    if (csmem > 48 * 1024 && csmem > csmem_set) {
        const cudaError_t a = cudaFuncSetAttribute(
            M8CpLocalCombineKernel<DT, BLK>, cudaFuncAttributeMaxDynamicSharedMemorySize, csmem);
        if (a != cudaSuccess) return static_cast<int>(a);
        csmem_set = csmem;
    }
    M8CpLocalCombineKernel<DT, BLK>
        <<<dim3(NHCP * HD / DT, MROWS8), BLK, csmem, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const float*>(wm), static_cast<const float*>(wl),
            static_cast<const bf16*>(wa), static_cast<bf16*>(cp_ctx), static_cast<float*>(cp_lse),
            ns, scaling, static_cast<int*>(it_ctr));
    return static_cast<int>(cudaGetLastError());
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M8
using namespace nexrt::sm90::nex_n2_5_pro::M8;
extern "C" {

int NexN25ProM8CpQGateLaunch(const void* x8, const void* Wpack, const void* scales, void* q_group8,
                             void* gate8, void* k_cache, void* v_cache, const void* ctl,
                             void* stream) {
    M8CpQGateKernel<<<dim3(kCpTotalRows / 8), dim3(256), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const bf16*>(x8), static_cast<const __nv_fp8_e4m3*>(Wpack),
        static_cast<const float*>(scales), static_cast<bf16*>(q_group8), static_cast<bf16*>(gate8),
        static_cast<bf16*>(k_cache), static_cast<bf16*>(v_cache), static_cast<const int*>(ctl));
    return static_cast<int>(cudaGetLastError());
}

int NexN25ProM8CpNormRopeLaunch(const void* q_raw8, void* k_cache, const void* qn, const void* kn,
                                const void* cos_tab, const void* sin_tab, void* q_out8,
                                const void* ctl, const void* x8, void* v_cache, void* gate8,
                                int hid, float eps_in, int hg_heads, void* stream) {
    M8CpNormRopeKernel<32><<<dim3(NHCP + 2, MROWS8), 32, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const bf16*>(q_raw8), static_cast<bf16*>(k_cache), static_cast<const bf16*>(qn),
        static_cast<const bf16*>(kn), static_cast<const float*>(cos_tab),
        static_cast<const float*>(sin_tab), static_cast<bf16*>(q_out8),
        static_cast<const bf16*>(x8), static_cast<bf16*>(v_cache), static_cast<bf16*>(gate8), hid,
        eps_in, hg_heads, static_cast<const int*>(ctl));
    return static_cast<int>(cudaGetLastError());
}

// Production coalesced epilogue. Fragment owners stage packed BF16x2 words in
// dead K/V storage; four warps per consumer warpgroup drain full 512-byte rows
// using 16-byte per-lane stores.
int NexN25ProM8CpPartialLaunch(const void* q_out8, const void* K, const void* V, const void* ctl,
                               int ns, float scaling, int cp_rank, void* wm, void* wl, void* wa,
                               void* stream) {
    CUtensorMap Kmap = cp_encode_map(const_cast<void*>(K), KV_CAPACITY, HD, TILE, 64);
    CUtensorMap Vmap = cp_encode_map(const_cast<void*>(V), KV_CAPACITY, HD, TILE, 64);
    CUtensorMap Qmap = cp_encode_map(const_cast<void*>(q_out8), QCOLS, HD, 64, 64);
    const int smem = cp_partial_wsx_smem_bytes<2, 2>();
    static bool init_done = false;
    if (!init_done) {
        const cudaError_t a = cudaFuncSetAttribute(
            m8_cp_partial_wsx<2, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (a != cudaSuccess) return static_cast<int>(a);
        init_done = true;
    }
    m8_cp_partial_wsx<2, 2><<<dim3(ns), 384, smem, static_cast<cudaStream_t>(stream)>>>(
        Kmap, Vmap, Qmap, static_cast<float*>(wm), static_cast<float*>(wl), static_cast<bf16*>(wa),
        static_cast<const int*>(ctl), ns, scaling, cp_rank);
    return static_cast<int>(cudaGetLastError());
}

// The split-tail combine needs the rank's own generation counter to
// pick the publication parity slice; the arrive that bumps it stays separate.
int NexN25ProM8CpLocalCombineLaunch(const void* wm, const void* wl, const void* wa, void* cp_ctx,
                                    void* cp_lse, int ns, float scaling, void* it_ctr,
                                    void* stream) {
    return cp_launch_local_combine(wm, wl, wa, cp_ctx, cp_lse, ns, scaling, it_ctr, stream);
}

int NexN25ProM8CpArriveLaunch(unsigned long long bar_mc, void* it_ctr, void* cur_gen, void* status,
                              void* stream) {
    M8CpArriveKernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
        bar_mc, static_cast<int*>(it_ctr), static_cast<int*>(cur_gen), static_cast<int*>(status));
    return static_cast<int>(cudaGetLastError());
}

int NexN25ProM8CpMergeLaunch(const void* peer_ctx, const void* peer_lse, const void* gate8,
                             void* ctx_out, void* xeff_out, const void* cp_bar_local,
                             void* cp_status, const void* cp_gen, int cp_rank,
                             const void* prefetch_base, long prefetch_bytes, void* stream) {
    M8CpMergeKernel<<<dim3(NHL * 8, MROWS8), CPR * 32, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const int64_t*>(peer_ctx), static_cast<const int64_t*>(peer_lse),
        static_cast<const bf16*>(gate8), static_cast<bf16*>(ctx_out), static_cast<bf16*>(xeff_out),
        static_cast<const int*>(cp_bar_local), static_cast<int*>(cp_status),
        static_cast<const int*>(cp_gen), cp_rank, prefetch_base, prefetch_bytes);
    return static_cast<int>(cudaGetLastError());
}

int NexN25ProM8CpQGateRows() { return kCpTotalRows; }
int NexN25ProM8CpHeads() { return NHCP; }
}
