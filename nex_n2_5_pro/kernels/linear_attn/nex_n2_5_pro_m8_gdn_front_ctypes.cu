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

// M=8 GDN front: qkvz/b/a projections for eight ordered target rows in ONE launch,
// on the M=8-ONLY convolution history ring.
//
// M=8-FAMILY TU. This file is a native eight-row implementation: it includes no BS=1
// or M=4 production source and launches nothing from those families. The mechanism
// is a transposed-mma + lane-major-prepack front derived from the M=4 geometry:
//
//   * fp8 qkvz rows [2560, 4096] ride 320 tiles x 8 weight rows; each of 8 warps owns
//     a K=512 slice; the e4m3 weights are decoded once per k32 block via the
//     value-exact f16 transit into bf16x2 and used as mma.m16n8k16 A fragments
//     (8 real W rows in m0..7; m8..15 zero).
//   * THE M=8 STRUCTURAL WIN: the B operand's n=8 token columns are ALL real. The
//     M=4 front zero-padded 4 of 8 columns; here every mma retires 8 rows x 8 tokens
//     of useful work, so the weight stream is amortized over twice the dot products
//     at unchanged traffic.
//   * the in-pass x RMS survives: each warp accumulates ss[token] from its staged x
//     slice; the fixed-order 8-warp combine and the bf16(sum * rms) rounding point
//     mirror the M=4 family class (fp32 block accumulate x w_scale).
//
// CONVOLUTION RING. Width-4 causal conv over the M=8-only ring
//     raw projected convolution input at absolute position p -> slot p & 15.
// Sixteen slots cover the eleven live positions C-3 .. C+7 of one unit, so no live
// value is aliased, and a unit's three history taps survive the unit intact. Nothing
// is erased on rejection: advancing committed_len re-selects the logical history and
// later units overwrite dead slots. Row j reads exactly the causal window
// C+j-3 .. C+j; in-unit positions come from registers/SMEM of this very call.
//
// EPILOGUE (parallel, tc16pe class): the ring history line (32 B = all 16 slots) is
// loaded at kernel entry by warp 0 and staged to SMEM just before the kernel's one
// existing __syncthreads, so its latency rides under the whole GEMM. The conv/z
// epilogue is a flat parallel map on warp 0: lane = (position j = lane>>3 (+4 on the
// second sweep), channel c = lane&7); four taps per lane in the fixed association
// ((t0+t2)+(t1+t3)) with __fmul_rn/__fadd_rn (no fma contraction), one silu output
// store and one ring publish per (j, c). Slots read (C-3..C-1) are disjoint from
// slots written (C..C+7) under the &15 rotation, so the entry hoist is safe.
//
// b/a rows (beta / decay gates) keep the replicated-row scalar path: 16 CTAs, each
// streaming one bf16 [4096] weight row once for all eight tokens.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M8
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "../common/m8_control_block.cuh"

namespace nexrt::sm90::nex_n2_5_pro::M8 {
constexpr int kHid = 4096;
constexpr int kQkvRows = 1536;
constexpr int kZRows = 1024;
constexpr int kFp8Rows = kQkvRows + kZRows;  // 2560
constexpr int kBRows = 8;
constexpr int kARows = 8;
constexpr int kScaleBlocksK = kHid / 128;          // 32
constexpr int kM8 = m8ctl::kM;                     // 8
constexpr int kConvSlots = m8ctl::kConvRingSlots;  // 16
constexpr float kSoftplusThreshold = 20.0f;
constexpr float kEps = 1e-6f;

__device__ __forceinline__ float m8_warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float m8_sigmoid_f32(float x) { return 1.0f / (1.0f + __expf(-x)); }

__device__ __forceinline__ float m8_silu_f32(float x) { return x * m8_sigmoid_f32(x); }

__device__ __forceinline__ float m8_softplus_f32(float x) {
    if (x > kSoftplusThreshold) {
        return x;
    }
    return log1pf(expf(x));
}

__device__ __forceinline__ void m8_unpack_bf16x16(const int4 raw0, const int4 raw1,
                                                  float (&xf)[16]) {
    const __nv_bfloat162* xp0 = reinterpret_cast<const __nv_bfloat162*>(&raw0);
    const __nv_bfloat162* xp1 = reinterpret_cast<const __nv_bfloat162*>(&raw1);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 v = __bfloat1622float2(xp0[j]);
        xf[2 * j] = v.x;
        xf[2 * j + 1] = v.y;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 v = __bfloat1622float2(xp1[j]);
        xf[8 + 2 * j] = v.x;
        xf[8 + 2 * j + 1] = v.y;
    }
}

__device__ __forceinline__ void m8_mma_m16n8k16_bf16(float (&c)[4], const unsigned (&a)[4],
                                                     const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// The qkvz weight prepack contract (built once by the M=8 runtime from the fp8
// row-major [2560, 4096] weight; identical index math to the M=4/attn family pack so
// a lane's uint2 per k32 step IS its two mma A-fragment bf16x2 pairs after decode):
//   pack[(tile * 128 + k32) * 256 + lane * 8 .. +7]  covers rows tile*8 .. tile*8+7,
//   k = k32*32 .. k32*32+31, in the (gid = lane>>2 -> row, tig = lane&3 -> k pair)
//   lane-major fragment order.

// Both retained arms use full 16-real-row MMA tiles and two K-half CTAs per
// descriptor. Pack contract (pack_qkv16_m8, m8_verify.py): for tile t and k32
// block ci,
//   pack[(t * 128 + ci) * 512 + lane * 16 + b], byte b (rowh=(b>>1)&1, e=b&1):
//     row = t*16 + rowh*8 + (lane>>2)
//     k   = ci*32 + (b>>3)*16 + ((b>>2)&1)*8 + (lane&3)*2 + e
// so one uint4 per lane per k32 IS its four A fragments for the two k16 steps.

constexpr int kTileRows16 = 16;
constexpr int kFt16Tiles = kFp8Rows / kTileRows16;  // 160 (qkv tiles 0..95, z 96..159)

__device__ __forceinline__ uint4 m8_load_weight_uint4(const __nv_fp8_e4m3* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

// Production FT16SK2 path. Each (tile, K-half) CTA owns K=2048 and publishes
// fp32 partials, and the second CTA of the pair (parity ticket, atomicAdd -- no
// reset needed, graph-replay-safe) folds and runs the epilogue.
//
// NUMERICS: family-class reorder (the K-fold association changes at the split boundary:
// (halfA)+(halfB) instead of the 8-warp sequential chain; scale folds group 2 blocks per
// warp instead of 4). NOT bit-exact vs the shipped kernel — gated like every family
// implementation: fp32 reference, completeness, BS=1-oracle, and smoke tests.
// The b/a path is verbatim (beta/g stay bit-exact).
//
// Workspace (M8-owned, one per rank): ws fp32 [160 tiles x 272]
//   (per tile: cpart[2 half][16 row][8 tok] at +0, sspart[2 half][8 tok] at +256)
// tickets int32 [160], never reset: each launch adds exactly 2 per tile, so exactly one
// CTA of each pair sees an odd ticket regardless of the starting value.

constexpr int kSk2TileF32 = 2 * kTileRows16 * kM8 + 2 * kM8;  // 272

__global__ __launch_bounds__(256) void M8GdnFrontFt16Sk2Kernel(
    const __nv_bfloat16* __restrict__ x8,           // [8, 4096]
    const __nv_fp8_e4m3* __restrict__ qkvz_pack16,  // 16-row lane-major prepack
    const float* __restrict__ qkvz_scales,          // [20, 32] fp32 block scales
    const __nv_bfloat16* __restrict__ b_w,          // [8, 4096]
    const __nv_bfloat16* __restrict__ a_w,          // [8, 4096]
    const float* __restrict__ a_log_dt,             // [16] (a_log | dt_bias)
    const __nv_bfloat16* __restrict__ conv_w,       // [1536, 4]
    __nv_bfloat16* __restrict__ conv_ring,          // [1536, 16] M=8-only ring
    const int* __restrict__ ctl,                    // M=8 control block (committed_len)
    __nv_bfloat16* __restrict__ qkv_out8,           // [8, 1536]
    __nv_bfloat16* __restrict__ z_out8,             // [8, 1024]
    float* __restrict__ beta_out8,                  // [8, 8]
    float* __restrict__ g_out8,                     // [8, 8]
    float* __restrict__ ws,                         // [160, 272] partials
    int* __restrict__ tickets) {                    // [160] parity tickets
    constexpr int kNumThreads = 256;
    constexpr int kWarpsPerCTA = 8;
    const int lane = threadIdx.x & 31;
    const int warp_in_cta = threadIdx.x >> 5;

    if (blockIdx.x < (kBRows + kARows)) {
        // ---- b/a rows: VERBATIM the shipped kernel's replicated-row scalar path ----
        const int actual = blockIdx.x;
        const bool is_b = actual < kBRows;
        const int row = is_b ? actual : (actual - kBRows);
        const __nv_bfloat16* w = (is_b ? b_w : a_w) + static_cast<size_t>(row) * kHid;
        constexpr int kElemsPerIter = kNumThreads * 16;
        static_assert(kHid % kElemsPerIter == 0, "b/a iteration count");
        float acc[kM8] = {};
        float ss[kM8] = {};
#pragma unroll
        for (int it = 0; it < kHid / kElemsPerIter; ++it) {
            const int k = (it * kNumThreads + threadIdx.x) * 16;
            const int4 wraw0 = *reinterpret_cast<const int4*>(w + k);
            const int4 wraw1 = *reinterpret_cast<const int4*>(w + k + 8);
            float wf[16];
            m8_unpack_bf16x16(wraw0, wraw1, wf);
#pragma unroll
            for (int j = 0; j < kM8; ++j) {
                const __nv_bfloat16* xj = x8 + static_cast<size_t>(j) * kHid;
                const int4 xraw0 = *reinterpret_cast<const int4*>(xj + k);
                const int4 xraw1 = *reinterpret_cast<const int4*>(xj + k + 8);
                float xf[16];
                m8_unpack_bf16x16(xraw0, xraw1, xf);
#pragma unroll
                for (int e = 0; e < 16; ++e) {
                    acc[j] += xf[e] * wf[e];
                    ss[j] += xf[e] * xf[e];
                }
            }
        }
        __shared__ float warp_partials[kWarpsPerCTA][kM8];
        __shared__ float warp_ss[kWarpsPerCTA][kM8];
#pragma unroll
        for (int j = 0; j < kM8; ++j) {
            const float a = m8_warp_sum(acc[j]);
            const float s = m8_warp_sum(ss[j]);
            if (lane == 0) {
                warp_partials[warp_in_cta][j] = a;
                warp_ss[warp_in_cta][j] = s;
            }
        }
        __syncthreads();
        if (warp_in_cta == 0) {
#pragma unroll
            for (int j = 0; j < kM8; ++j) {
                float sum = (lane < kWarpsPerCTA) ? warp_partials[lane][j] : 0.0f;
                float sumss = (lane < kWarpsPerCTA) ? warp_ss[lane][j] : 0.0f;
                sum = m8_warp_sum(sum);
                sumss = m8_warp_sum(sumss);
                if (lane == 0) {
                    const float rms = rsqrtf(sumss / kHid + kEps);
                    const float y = __bfloat162float(__float2bfloat16(sum * rms));
                    if (is_b) {
                        beta_out8[j * kBRows + row] = m8_sigmoid_f32(y);
                    } else {
                        const float a_log = a_log_dt[row];
                        const float dt_bias = a_log_dt[kARows + row];
                        g_out8[j * kARows + row] = -expf(a_log) * m8_softplus_f32(y + dt_bias);
                    }
                }
            }
        }
        return;
    }

    // ---- fp8 qkvz rows: 16-real-row mma over a K=2048 half per CTA ----
    const int idx = blockIdx.x - (kBRows + kARows);  // 0..319
    const int tile = idx >> 1;                       // 0..159
    const int half = idx & 1;                        // k half 0 | 1
    const int row0 = tile * kTileRows16;
    const bool qkv_tile = row0 < kQkvRows;
    const int gid = lane >> 2;
    const int tig = lane & 3;
    constexpr int kWarpK = 256;               // per-warp slice of the 2048 half
    constexpr int kXPitch = kWarpK * 2 + 16;  // 528 B
    __shared__ __align__(16) char sx[kWarpsPerCTA][kM8][kXPitch];
    __shared__ float red[kWarpsPerCTA][32][4];
    __shared__ float ssred[kWarpsPerCTA][kM8];
    __shared__ float out_sm[kTileRows16][kM8];
    __shared__ float rms_sm[kM8];
    __shared__ __align__(16) __nv_bfloat16 hist_sm[kTileRows16][kConvSlots];
    const int k0 = half * (kHid / 2) + warp_in_cta * kWarpK;
    char* const sxw = &sx[warp_in_cta][0][0];
    float* const tws = ws + static_cast<size_t>(tile) * kSk2TileF32;

// Stage this warp's 256-element slice of every token (one 16 B cp.async per token)
// WITHOUT waiting yet: the whole weight stream is issued underneath it.
#pragma unroll
    for (int m = 0; m < kM8; ++m) {
        const uint32_t dst =
            static_cast<uint32_t>(__cvta_generic_to_shared(sxw + m * kXPitch + lane * 16));
        const void* src = x8 + static_cast<size_t>(m) * kHid + k0 + lane * 8;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
    }
    asm volatile("cp.async.commit_group;");

    // FULL-DEPTH weight prefetch (r3, NCU-driven): the 8-step ring left warps stalled
    // on long_scoreboard (50% of issue-active) — issue ALL eight 16 B loads now, 4 KB
    // per warp in flight, and consume from registers. Consumption order is unchanged,
    // so the numerics class is identical.
    const __nv_fp8_e4m3* wp_base =
        qkvz_pack16 + (static_cast<size_t>(tile) * 128 + (k0 >> 5)) * 512 + lane * 16;
    uint4 wbuf[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        wbuf[i] = m8_load_weight_uint4(wp_base + i * 512);
    }

    asm volatile("cp.async.wait_group 0;");
    __syncwarp();

    float ss[kM8] = {};
#pragma unroll
    for (int m = 0; m < kM8; ++m) {
        const __nv_bfloat162* xr = reinterpret_cast<const __nv_bfloat162*>(sxw + m * kXPitch);
#pragma unroll
        for (int i = lane; i < kWarpK / 2; i += 32) {
            const float2 v = __bfloat1622float2(xr[i]);
            ss[m] += v.x * v.x + v.y * v.y;
        }
    }

    // This warp's 256 k span exactly 2 scale blocks.
    const float* wsc = qkvz_scales + static_cast<size_t>(row0 >> 7) * kScaleBlocksK + (k0 >> 7);
    float sblk[2];
    sblk[0] = wsc[0];
    sblk[1] = wsc[1];

    float sacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int blk = 0; blk < 2; ++blk) {
        float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int t32 = 0; t32 < 4; ++t32) {
            const int ci = blk * 4 + t32;
            const uint4 raw = wbuf[ci];
            const __nv_fp8x2_e4m3* wp = reinterpret_cast<const __nv_fp8x2_e4m3*>(&raw);
            unsigned awf[8];
#pragma unroll
            for (int p_ = 0; p_ < 8; ++p_) {
                const __nv_bfloat162 bb = __float22bfloat162_rn(static_cast<float2>(wp[p_]));
                awf[p_] = *reinterpret_cast<const unsigned*>(&bb);
            }
            const int kloc = ci * 32;  // k offset inside the staged slice
#pragma unroll
            for (int s_ = 0; s_ < 2; ++s_) {
                const char* xp = sxw + gid * kXPitch + (kloc + s_ * 16) * 2 + tig * 4;
                unsigned b[2];
                b[0] = *reinterpret_cast<const unsigned*>(xp);
                b[1] = *reinterpret_cast<const unsigned*>(xp + 16);
                const unsigned a[4] = {awf[s_ * 4], awf[s_ * 4 + 1], awf[s_ * 4 + 2],
                                       awf[s_ * 4 + 3]};
                m8_mma_m16n8k16_bf16(c, a, b);
            }
        }
        sacc[0] += sblk[blk] * c[0];
        sacc[1] += sblk[blk] * c[1];
        sacc[2] += sblk[blk] * c[2];
        sacc[3] += sblk[blk] * c[3];
    }

    red[warp_in_cta][lane][0] = sacc[0];
    red[warp_in_cta][lane][1] = sacc[1];
    red[warp_in_cta][lane][2] = sacc[2];
    red[warp_in_cta][lane][3] = sacc[3];
#pragma unroll
    for (int m = 0; m < kM8; ++m) {
        const float s = m8_warp_sum(ss[m]);
        if (lane == 0) {
            ssred[warp_in_cta][m] = s;
        }
    }
    __syncthreads();
    if (warp_in_cta != 0) {
        return;  // warps 1..7 are done
    }

    // ---- warp 0: fold this half's 8 warps, publish, parity-elect the folder ----
    {
        float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
#pragma unroll
        for (int sp = 0; sp < kWarpsPerCTA; ++sp) {
            v0 += red[sp][lane][0];
            v1 += red[sp][lane][1];
            v2 += red[sp][lane][2];
            v3 += red[sp][lane][3];
        }
        float* cp = tws + half * (kTileRows16 * kM8);
        cp[gid * kM8 + tig * 2] = v0;
        cp[gid * kM8 + tig * 2 + 1] = v1;
        cp[(gid + 8) * kM8 + tig * 2] = v2;
        cp[(gid + 8) * kM8 + tig * 2 + 1] = v3;
    }
    if (lane < kM8) {
        float sst = 0.0f;
#pragma unroll
        for (int wgt = 0; wgt < kWarpsPerCTA; ++wgt) {
            sst += ssred[wgt][lane];
        }
        tws[2 * kTileRows16 * kM8 + half * kM8 + lane] = sst;
    }
    __threadfence();
    int ticket = 0;
    if (lane == 0) {
        ticket = atomicAdd(tickets + tile, 1);
    }
    ticket = __shfl_sync(0xffffffffu, ticket, 0);
    if ((ticket & 1) == 0) {
        return;  // first arrival: partner folds
    }
    __threadfence();  // partner's publish is visible

    // ---- folder: hist prefetch, cross-half fold, rms, epilogue (ft16 arithmetic) ----
    const int cl = qkv_tile ? m8ctl::CommittedLen(ctl) : 0;
    if (qkv_tile) {
        const uint4 hraw = *reinterpret_cast<const uint4*>(
            conv_ring + static_cast<size_t>(row0 + (lane >> 1)) * kConvSlots + (lane & 1) * 8);
        *reinterpret_cast<uint4*>(&hist_sm[lane >> 1][(lane & 1) * 8]) = hraw;
    }
#pragma unroll
    for (int i = lane; i < kTileRows16 * kM8; i += 32) {
        out_sm[i >> 3][i & 7] = tws[i] + tws[kTileRows16 * kM8 + i];
    }
    if (lane < kM8) {
        const float sst =
            tws[2 * kTileRows16 * kM8 + lane] + tws[2 * kTileRows16 * kM8 + kM8 + lane];
        rms_sm[lane] = rsqrtf(sst / kHid + kEps);
    }
    __syncwarp();

    const int c = lane & 15;
    const int row = row0 + c;
    if (qkv_tile) {
        const __nv_bfloat16* cw = conv_w + static_cast<size_t>(row) * 4;
        const __nv_bfloat16 pb0 = __float2bfloat16(out_sm[c][0] * rms_sm[0]);
        const __nv_bfloat16 pb1 = __float2bfloat16(out_sm[c][1] * rms_sm[1]);
        const __nv_bfloat16 pb2 = __float2bfloat16(out_sm[c][2] * rms_sm[2]);
        const __nv_bfloat16 pb3 = __float2bfloat16(out_sm[c][3] * rms_sm[3]);
        const __nv_bfloat16 pb4 = __float2bfloat16(out_sm[c][4] * rms_sm[4]);
        const __nv_bfloat16 pb5 = __float2bfloat16(out_sm[c][5] * rms_sm[5]);
        const __nv_bfloat16 pb6 = __float2bfloat16(out_sm[c][6] * rms_sm[6]);
        const __nv_bfloat16 pb7 = __float2bfloat16(out_sm[c][7] * rms_sm[7]);
        const float cw0 = __bfloat162float(cw[0]);
        const float cw1 = __bfloat162float(cw[1]);
        const float cw2 = __bfloat162float(cw[2]);
        const float cw3 = __bfloat162float(cw[3]);
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            const int j = (lane >> 4) + jj * 2;
            float t[4];
#pragma unroll
            for (int tap = 0; tap < 4; ++tap) {
                const int off = j - 3 + tap;
                __nv_bfloat16 v_b;
                if (off < 0) {
                    v_b = hist_sm[c][m8ctl::ConvSlot(cl + off)];
                } else if (off == 0) {
                    v_b = pb0;
                } else if (off == 1) {
                    v_b = pb1;
                } else if (off == 2) {
                    v_b = pb2;
                } else if (off == 3) {
                    v_b = pb3;
                } else if (off == 4) {
                    v_b = pb4;
                } else if (off == 5) {
                    v_b = pb5;
                } else if (off == 6) {
                    v_b = pb6;
                } else {
                    v_b = pb7;
                }
                const float cw_f = (tap == 0) ? cw0 : (tap == 1) ? cw1 : (tap == 2) ? cw2 : cw3;
                t[tap] = __fmul_rn(__bfloat162float(v_b), cw_f);
            }
            const float s = __fadd_rn(__fadd_rn(t[0], t[2]), __fadd_rn(t[1], t[3]));
            qkv_out8[static_cast<size_t>(j) * kQkvRows + row] = __float2bfloat16(m8_silu_f32(s));
            const __nv_bfloat16 pj = (j == 0)   ? pb0
                                     : (j == 1) ? pb1
                                     : (j == 2) ? pb2
                                     : (j == 3) ? pb3
                                     : (j == 4) ? pb4
                                     : (j == 5) ? pb5
                                     : (j == 6) ? pb6
                                                : pb7;
            conv_ring[static_cast<size_t>(row) * kConvSlots + m8ctl::ConvSlot(cl + j)] = pj;
        }
    } else {
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            const int j = (lane >> 4) + jj * 2;
            z_out8[static_cast<size_t>(j) * kZRows + (row - kQkvRows)] =
                __float2bfloat16(out_sm[c][j] * rms_sm[j]);
        }
    }
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M8
using namespace nexrt::sm90::nex_n2_5_pro::M8;
extern "C" {
// x8 [8,4096] bf16; qkvz_pack = M8 lane-major prepack of qkvz_w [2560,4096] fp8;
// qkvz_scales [20,32] fp32; b_w/a_w [8,4096] bf16; a_log_dt [16] fp32; conv_w [1536,4]
// bf16; conv_ring [1536,16] bf16 (M=8-only); ctl int32[32] M=8 control block;
// qkv_out8 [8,1536] bf16; z_out8 [8,1024] bf16; beta_out8/g_out8 [8,8] fp32.
// qkvz_pack16 is the 16-row lane-major prepack. ws is the [160*272] fp32 partial workspace
// and tickets int32 [160] parity counters are M8-owned, zero-init ONCE, never reset
// (each launch adds exactly 2 per tile). Family-class numerics are described above.
int NexN25ProM8GdnFrontLaunch(const void* x8, const void* qkvz_pack16, const void* qkvz_scales,
                              const void* b_w, const void* a_w, const void* a_log_dt,
                              const void* conv_w, void* conv_ring, const void* ctl, void* qkv_out8,
                              void* z_out8, void* beta_out8, void* g_out8, void* ws, void* tickets,
                              void* stream) {
    M8GdnFrontFt16Sk2Kernel<<<dim3(kBRows + kARows + 2 * kFt16Tiles), dim3(256), 0,
                              static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(x8), static_cast<const __nv_fp8_e4m3*>(qkvz_pack16),
        static_cast<const float*>(qkvz_scales), static_cast<const __nv_bfloat16*>(b_w),
        static_cast<const __nv_bfloat16*>(a_w), static_cast<const float*>(a_log_dt),
        static_cast<const __nv_bfloat16*>(conv_w), static_cast<__nv_bfloat16*>(conv_ring),
        static_cast<const int*>(ctl), static_cast<__nv_bfloat16*>(qkv_out8),
        static_cast<__nv_bfloat16*>(z_out8), static_cast<float*>(beta_out8),
        static_cast<float*>(g_out8), static_cast<float*>(ws), static_cast<int*>(tickets));
    return static_cast<int>(cudaGetLastError());
}
// ABI probes for the runtime's workspace sizing.
int NexN25ProM8GdnFrontSk2WsFloats() { return kFt16Tiles * kSk2TileF32; }
int NexN25ProM8GdnFrontSk2Tiles() { return kFt16Tiles; }
}  // extern "C"
