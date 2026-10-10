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

// M=4 GDN front on the M=4-ONLY convolution HISTORY RING.
//
// Identical projection arithmetic to the M=4 split-K front form:
// the fp8 and b/a paths below are a verbatim transcription -- same one-CTA-per-2-row-group
// mapping, same K split across kWarpsPerCTA warps, same four-row weight reuse, same 16-term
// dot expansion, same fixed-order warp combine, same per-row rms. ONLY the convolution
// epilogue changes.
//
// WHY. The shipped epilogue (qkv_conv_epilogue_pair_warp) PHYSICALLY SHIFTS a three-deep
// history window: st[0]=st[1], st[1]=st[2], st[2]=new. Shifting is destructive, so after a
// four-row M=4 call the window unconditionally holds the last three of the four NEW rows --
// it has already committed all four. Speculative verification must be able to commit only
// commit_count of them, so the M=4 path replaces the shifted window with a RING:
//
//     raw projected convolution input at absolute position p  ->  slot p & 7
//
// Eight slots cover the seven distinct positions C-3 .. C+3 that one unit touches, so no
// live value is ever aliased. Nothing is erased on rejection: candidate positions simply
// stay in the ring, and advancing committed_len re-selects the logical history. A later M=4
// call overwrites the rejected slots.
//
// ROW j reads exactly the width-four causal window C+j-3, C+j-2, C+j-1, C+j.
//
// NUMERICS: BIT-EQUAL to the split-K front. The value stored at
// position p is the same __float2bfloat16(mixed) word the shifted window stored; the four
// tap operands are the same four bf16 values in the same tap order; the product, the two
// __shfl_down_sync reduction steps and the silu round-trip are unchanged. Positions produced
// inside this unit are carried in REGISTERS rather than re-read through global memory, which
// removes the intra-warp read-after-write dependence the shifted window relies on; a bf16
// store/reload is lossless, so this is a scheduling change, not an arithmetic one.
//
// BS1 IS UNTOUCHED. conv_state4, its shift epilogue and every BS1 kernel are unmodified;
// this file adds a parallel M=4-only path and never writes conv_state4.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M4
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "nex_n2_5_pro_linear_front_common.cuh"

#include "../common/m4_control_block.cuh"

namespace nexrt::sm90::nex_n2_5_pro::M4 {
constexpr int kM4Ring = m4ctl::kM;
constexpr int kConvSlots = m4ctl::kConvRingSlots;

// Production tensor-core front.
// The FP8 qkvz GEMM is shape-identical to the attention qkvgate (2560 x 4096, 128-block
// scales), so it takes the same transposed-mma + lane-major-prepack core: 320 tiles x 8
// rows x 8 warps (K=512 each), one coalesced 8 B/lane .cs load per k32 from the SAME
// pack layout (pack_qkv_bf16tc applied to qkvz_w), e4m3 decoded once via the value-exact
// f16 transit into bf16x2, one mma per k16, fp32 block accumulate x w_scale. Products
// move from this kernel's fp32-SIMT class to bf16 (fp32-accumulated) -- tolerance-gated.
// The in-pass x RMS survives: each warp accumulates ss[j] from its staged x slice, the
// fixed-order combine and the bf16(sum*rms) rounding point are preserved.
//
// EPILOGUE. Nothing in the conv/z unit is truly sequential: all four bf16(sum*rms)
// words exist together after the warp-partial fold, in-call taps of later positions use
// values from that same fold, and the ring slots read (cl-3..cl-1) are DISJOINT from the
// slots written (cl..cl+3) under the &7 rotation. So this kernel
//   - issues ONE 16 B ring-row load per channel at kernel entry (warp 0, lanes 0..7),
//     parks it in registers across the whole GEMM, and stages it to shared just before
//     the kernel's one existing __syncthreads -- the history-load latency rides under
//     the GEMM instead of surfacing at the CTA tail;
//   - runs the conv/z epilogue as ONE flat parallel step on warp 0: lane = (position
//     j = lane>>3, channel c = lane&7); four taps per lane in the fixed association
//     ((t0+t2)+(t1+t3)) via __fmul_rn/__fadd_rn (forbidding fma contraction), one
//     output store and one ring publish per lane. z tiles use the same flat map.
// The stored ring word, tap operands, product/sum association and every bf16 rounding
// point are unchanged from the segmented-octet epilogue this replaces: byte-identical
// outputs and ring bytes.
// b/a rows keep their verbatim path on blockIdx.x < 16. Tiles never straddle the
// qkv/z boundary (1536 = 192 tiles) or a conv group (8 rows = 4 whole groups).

__device__ __forceinline__ uint2 m4_linear_front_load_weight(const __nv_fp8_e4m3* p) {
    uint2 v;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];"
                 : "=r"(v.x), "=r"(v.y)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ void mma_m16n8k16_bf16_tc_(float (&c)[4], const unsigned (&a)[4],
                                                      const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ __launch_bounds__(256) void M4LinearFrontKernel(
    const __nv_bfloat16* __restrict__ x4,
    const __nv_fp8_e4m3* __restrict__ qkvz_pack,  // pack_qkv_bf16tc(qkvz_w)
    const float* __restrict__ qkvz_scales, const __nv_bfloat16* __restrict__ b_w,
    const __nv_bfloat16* __restrict__ a_w, const float* __restrict__ a_log_dt,
    const __nv_bfloat16* __restrict__ conv_w, __nv_bfloat16* __restrict__ conv_ring,
    const int* __restrict__ ctl, __nv_bfloat16* __restrict__ qkv_out4,
    __nv_bfloat16* __restrict__ z_out4, float* __restrict__ beta_out4, float* __restrict__ g_out4) {
    constexpr int kNumThreads = 256;
    constexpr int kWarpsPerCTA = 8;
    constexpr float kEps = 1e-6f;
    const int lane = threadIdx.x & 31;
    const int warp_in_cta = threadIdx.x >> 5;

    if (blockIdx.x < (kBRows + kARows)) {
        // ---- b/a rows: verbatim original path at 256 threads ----
        const int actual = blockIdx.x;
        const bool is_b = actual < kBRows;
        const int row = is_b ? actual : (actual - kBRows);
        const __nv_bfloat16* w = (is_b ? b_w : a_w) + static_cast<size_t>(row) * kHid;
        constexpr int kElemsPerIter = kNumThreads * 16;
        static_assert(kHid % kElemsPerIter == 0, "b/a iteration count");
        float acc[kM4Ring] = {0.f, 0.f, 0.f, 0.f};
        float ss[kM4Ring] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int it = 0; it < kHid / kElemsPerIter; ++it) {
            const int k = (it * kNumThreads + threadIdx.x) * 16;
            const int4 wraw0 = *reinterpret_cast<const int4*>(w + k);
            const int4 wraw1 = *reinterpret_cast<const int4*>(w + k + 8);
            float wf[16];
            unpack_bf16x16(wraw0, wraw1, wf);
#pragma unroll
            for (int j = 0; j < kM4Ring; ++j) {
                const __nv_bfloat16* xj = x4 + static_cast<size_t>(j) * kHid;
                const int4 xraw0 = *reinterpret_cast<const int4*>(xj + k);
                const int4 xraw1 = *reinterpret_cast<const int4*>(xj + k + 8);
                float xf[16];
                unpack_bf16x16(xraw0, xraw1, xf);
#pragma unroll
                for (int e = 0; e < 16; ++e) {
                    acc[j] += xf[e] * wf[e];
                    ss[j] += xf[e] * xf[e];
                }
            }
        }
        __shared__ float warp_partials[kWarpsPerCTA][kM4Ring];
        __shared__ float warp_ss[kWarpsPerCTA][kM4Ring];
#pragma unroll
        for (int j = 0; j < kM4Ring; ++j) {
            const float a = warp_sum(acc[j]);
            const float s = warp_sum(ss[j]);
            if (lane == 0) {
                warp_partials[warp_in_cta][j] = a;
                warp_ss[warp_in_cta][j] = s;
            }
        }
        __syncthreads();
        if (warp_in_cta == 0) {
#pragma unroll
            for (int j = 0; j < kM4Ring; ++j) {
                float sum = (lane < kWarpsPerCTA) ? warp_partials[lane][j] : 0.0f;
                float sumss = (lane < kWarpsPerCTA) ? warp_ss[lane][j] : 0.0f;
                sum = warp_sum(sum);
                sumss = warp_sum(sumss);
                if (lane == 0) {
                    const float rms = rsqrtf(sumss / kHid + kEps);
                    const float y = __bfloat162float(__float2bfloat16(sum * rms));
                    if (is_b) {
                        beta_out4[j * kBRows + row] = sigmoid_f32(y);
                    } else {
                        const float a_log = a_log_dt[row];
                        const float dt_bias = a_log_dt[kARows + row];
                        g_out4[j * kARows + row] = -expf(a_log) * softplus_f32(y + dt_bias);
                    }
                }
            }
        }
        return;
    }

    // ---- fp8 qkvz rows: transposed-mma core on the lane-major prepack ----
    const int tile = blockIdx.x - (kBRows + kARows);  // 0..319; rows tile*8..tile*8+7
    const int row0 = tile * 8;
    const int gid = lane >> 2;
    const int tig = lane & 3;
    constexpr int kWarpK = kHid / 8;          // 512
    constexpr int kXPitch = kWarpK * 2 + 16;  // 1040 B
    __shared__ __align__(16) char sx[8][kM4Ring][kXPitch];
    __shared__ float red[8][32][2];
    __shared__ float ssred[8][kM4Ring];
    __shared__ float out_sm[8][kM4Ring];
    __shared__ float rms_sm[kM4Ring];
    __shared__ __align__(16) __nv_bfloat16 hist_sm[8][kConvSlots];  // raw ring rows
    const int k0 = warp_in_cta * kWarpK;
    char* const sxw = &sx[warp_in_cta][0][0];

    // Entry-issued ring-history line per channel (16 B = all 8 slots; the row stride
    // IS 16 B). Consumed only after the __syncthreads below. committed_len is
    // round-stable (the commit kernel runs after every front in the round graph), and
    // the slots this kernel writes are disjoint from the ones it reads, so hoisting
    // is safe.
    const bool qkv_tile = row0 < kQkvRows;
    const int cl = (qkv_tile && warp_in_cta == 0) ? ctl[m4ctl::kCommittedLen] : 0;
    uint4 hist_raw = make_uint4(0u, 0u, 0u, 0u);
    if (qkv_tile && warp_in_cta == 0 && lane < 8) {
        hist_raw = *reinterpret_cast<const uint4*>(conv_ring +
                                                   static_cast<size_t>(row0 + lane) * kConvSlots);
    }

#pragma unroll
    for (int m = 0; m < kM4Ring; ++m) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t dst = static_cast<uint32_t>(
                __cvta_generic_to_shared(sxw + m * kXPitch + h * 512 + lane * 16));
            const void* src = x4 + static_cast<size_t>(m) * kHid + k0 + (h * 512 + lane * 16) / 2;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
        }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncwarp();

    float ss[kM4Ring] = {0.f, 0.f, 0.f, 0.f};  // in-pass x RMS from the staged slice
#pragma unroll
    for (int m = 0; m < kM4Ring; ++m) {
        const __nv_bfloat162* xr = reinterpret_cast<const __nv_bfloat162*>(sxw + m * kXPitch);
#pragma unroll
        for (int i = lane; i < kWarpK / 2; i += 32) {
            const float2 v = __bfloat1622float2(xr[i]);
            ss[m] += v.x * v.x + v.y * v.y;
        }
    }

    const float* ws =
        qkvz_scales + static_cast<size_t>(row0 >> 7) * kScaleBlocksK + warp_in_cta * 4;
    float sblk[4];
#pragma unroll
    for (int b_ = 0; b_ < 4; ++b_) {
        sblk[b_] = ws[b_];
    }

    const __nv_fp8_e4m3* wp_base =
        qkvz_pack + (static_cast<size_t>(tile) * 128 + (k0 >> 5)) * 256 + lane * 8;
    float sacc[2] = {0.0f, 0.0f};
    uint2 wbuf[4];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        wbuf[i] = m4_linear_front_load_weight(wp_base + i * 256);
    }
#pragma unroll
    for (int blk = 0; blk < 4; ++blk) {
        float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int t32 = 0; t32 < 4; ++t32) {
            const int ci = blk * 4 + t32;
            if (ci + 3 < 16) {
                wbuf[(ci + 3) & 3] = m4_linear_front_load_weight(wp_base + (ci + 3) * 256);
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
                unsigned b[2] = {0u, 0u};
                if (gid < kM4Ring) {
                    const char* xp =
                        sxw + gid * kXPitch + (blk * 128 + t32 * 32 + s_ * 16) * 2 + tig * 4;
                    b[0] = *reinterpret_cast<const unsigned*>(xp);
                    b[1] = *reinterpret_cast<const unsigned*>(xp + 16);
                }
                const unsigned a[4] = {awf[s_ * 2], 0u, awf[s_ * 2 + 1], 0u};
                mma_m16n8k16_bf16_tc_(c, a, b);
            }
        }
        sacc[0] += sblk[blk] * c[0];
        sacc[1] += sblk[blk] * c[1];
    }

    red[warp_in_cta][lane][0] = sacc[0];
    red[warp_in_cta][lane][1] = sacc[1];
#pragma unroll
    for (int m = 0; m < kM4Ring; ++m) {
        const float s = warp_sum(ss[m]);
        if (lane == 0) {
            ssred[warp_in_cta][m] = s;
        }
    }
    if (qkv_tile && warp_in_cta == 0 && lane < 8) {  // stage the prefetched history
        *reinterpret_cast<uint4*>(&hist_sm[lane][0]) = hist_raw;
    }
    __syncthreads();

    if (warp_in_cta == 0) {
        if (tig * 2 < kM4Ring) {  // fold 8 warp partials -> out_sm
            float v0 = 0.0f, v1 = 0.0f;
#pragma unroll
            for (int sp = 0; sp < 8; ++sp) {
                v0 += red[sp][lane][0];
                v1 += red[sp][lane][1];
            }
            out_sm[gid][tig * 2] = v0;
            out_sm[gid][tig * 2 + 1] = v1;
        }
        if (lane < kM4Ring) {  // fixed-order ss combine -> rms
            float sst = 0.0f;
#pragma unroll
            for (int wgt = 0; wgt < 8; ++wgt) {
                sst += ssred[wgt][lane];
            }
            rms_sm[lane] = rsqrtf(sst / kHid + kEps);
        }
        __syncwarp();

        // Flat parallel epilogue: one (position, channel) pair per lane.
        const int j = lane >> 3;  // draft position 0..3
        const int c = lane & 7;   // channel row within the tile
        const int row = row0 + c;
        if (qkv_tile) {
            const __nv_bfloat16* cw = conv_w + static_cast<size_t>(row) * 4;
            // This call's four positions for channel c, in the exact stored-word form.
            // Named scalars + select ladders (not an array): a runtime-indexed local
            // array would be demoted to stack.
            const __nv_bfloat16 pb0 = __float2bfloat16(out_sm[c][0] * rms_sm[0]);
            const __nv_bfloat16 pb1 = __float2bfloat16(out_sm[c][1] * rms_sm[1]);
            const __nv_bfloat16 pb2 = __float2bfloat16(out_sm[c][2] * rms_sm[2]);
            const __nv_bfloat16 pb3 = __float2bfloat16(out_sm[c][3] * rms_sm[3]);
            // Window C+j-3 .. C+j: history from the prefetched ring line, rest from pb*.
            float t[4];
#pragma unroll
            for (int tap = 0; tap < 4; ++tap) {
                const int off = j - 3 + tap;  // absolute position C + off
                __nv_bfloat16 v_b;
                if (off < 0) {
                    v_b = hist_sm[c][m4ctl::ConvSlot(cl + off)];
                } else if (off == 0) {
                    v_b = pb0;
                } else if (off == 1) {
                    v_b = pb1;
                } else if (off == 2) {
                    v_b = pb2;
                } else {
                    v_b = pb3;
                }
                t[tap] = __fmul_rn(__bfloat162float(v_b), __bfloat162float(cw[tap]));
            }
            const float s = __fadd_rn(__fadd_rn(t[0], t[2]), __fadd_rn(t[1], t[3]));
            qkv_out4[static_cast<size_t>(j) * kQkvRows + row] = __float2bfloat16(silu_f32(s));
            const __nv_bfloat16 pj = (j == 0) ? pb0 : (j == 1) ? pb1 : (j == 2) ? pb2 : pb3;
            conv_ring[static_cast<size_t>(row) * kConvSlots + m4ctl::ConvSlot(cl + j)] = pj;
        } else {
            z_epilogue(row, out_sm[c][j] * rms_sm[j], z_out4 + static_cast<size_t>(j) * kZRows);
        }
    }
}

extern "C" {
// qkvz_pack is the lane-major production prepack; the history ring and
// control block are M=4-only state.
int NexN25ProM4LinearFrontLaunch(const void* x4, const void* qkvz_pack, const void* qkvz_scales,
                                 const void* b_w, const void* a_w, const void* a_log_dt,
                                 const void* conv_w, void* conv_ring, const void* ctl,
                                 void* qkv_out4, void* z_out4, void* beta_out4, void* g_out4,
                                 void* stream) {
    M4LinearFrontKernel<<<dim3(kBRows + kARows + kFp8Rows / 8), dim3(256), 0,
                          static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(x4), static_cast<const __nv_fp8_e4m3*>(qkvz_pack),
        static_cast<const float*>(qkvz_scales), static_cast<const __nv_bfloat16*>(b_w),
        static_cast<const __nv_bfloat16*>(a_w), static_cast<const float*>(a_log_dt),
        static_cast<const __nv_bfloat16*>(conv_w), static_cast<__nv_bfloat16*>(conv_ring),
        static_cast<const int*>(ctl), static_cast<__nv_bfloat16*>(qkv_out4),
        static_cast<__nv_bfloat16*>(z_out4), static_cast<float*>(beta_out4),
        static_cast<float*>(g_out4));
    return static_cast<int>(cudaGetLastError());
}
}  // extern "C"

}  // namespace nexrt::sm90::nex_n2_5_pro::M4
