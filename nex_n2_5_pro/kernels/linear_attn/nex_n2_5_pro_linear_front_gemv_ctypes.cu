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

// Fused front projection for one nex_n2_5_pro linear-attn TP rank, single-token decode.
//
// Row space:
//   [0,1536)    fp8 qkv rows  -> conv+SiLU qkv_out and conv_state3 update
//   [1536,2560) fp8 z rows    -> z_out
//   16 bf16 b/a rows          -> beta_out / g_out, one CTA each
//
// Split-CTA scheduling with a 4-way K split on the fp8 groups: each GROUP
// (2 rows) gets a whole CTA whose 4 warps each stream 2 of the 8 K tiles of the
// same rows; partials combine in shared memory and warp 0 applies the fused
// input-RMSNorm + the epilogues. Grid 1280+16 CTAs at 128 threads ~= 39
// warps/SM, versus ~10 warps/SM for a one-warp-per-group mapping -- the fp8
// stream needs the extra in-flight loads, not more math. 16 BF16 CTAs each
// compute one b/a row with unrolled int4-pair loads (all loads in flight; a
// serial 4-byte-load loop makes the b/a rows a ~10 us latency chain). Row sums
// are fixed-order sums of 4 warp partials.
#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY bs1
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "nex_n2_5_pro_linear_front_common.cuh"

namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY {
__device__ __forceinline__ float dot_fp8x16_bf16x16(const __nv_fp8_e4m3* w, const float (&xf)[16]) {
    const int4 wraw = *reinterpret_cast<const int4*>(w);
    const __nv_fp8x4_e4m3* wp = reinterpret_cast<const __nv_fp8x4_e4m3*>(&wraw);
    const float4 w0 = float4(wp[0]);
    const float4 w1 = float4(wp[1]);
    const float4 w2 = float4(wp[2]);
    const float4 w3 = float4(wp[3]);
    return w0.x * xf[0] + w0.y * xf[1] + w0.z * xf[2] + w0.w * xf[3] + w1.x * xf[4] + w1.y * xf[5] +
           w1.z * xf[6] + w1.w * xf[7] + w2.x * xf[8] + w2.y * xf[9] + w2.z * xf[10] +
           w2.w * xf[11] + w3.x * xf[12] + w3.y * xf[13] + w3.z * xf[14] + w3.w * xf[15];
}

__device__ __forceinline__ void qkv_conv_epilogue_pair_warp(
    int row0, float acc0_lane0, float acc1_lane0, const __nv_bfloat16* __restrict__ conv_w,
    __nv_bfloat16* __restrict__ conv_state4, __nv_bfloat16* __restrict__ qkv_out, int lane) {
    if (lane >= 8) {
        return;
    }
    const int which = lane >> 2;
    const int tap = lane & 3;
    const int row = row0 + which;
    const float mixed0 = __shfl_sync(0xffffffffu, acc0_lane0, 0);
    const float mixed1 = __shfl_sync(0xffffffffu, acc1_lane0, 0);
    const float mixed = which == 0 ? mixed0 : mixed1;
    const __nv_bfloat16 mixed_b = __float2bfloat16(mixed);
    __nv_bfloat16* st = conv_state4 + static_cast<size_t>(row) * 4;
    const __nv_bfloat16* cw = conv_w + static_cast<size_t>(row) * 4;

    __nv_bfloat16 val_b = mixed_b;
    if (tap < 3) {
        val_b = st[tap];
    }
    float prod = __bfloat162float(val_b) * __bfloat162float(cw[tap]);
    const unsigned mask = which == 0 ? 0x0000000fu : 0x000000f0u;
    prod += __shfl_down_sync(mask, prod, 2, 4);
    prod += __shfl_down_sync(mask, prod, 1, 4);

    if (tap > 0) {
        st[tap - 1] = val_b;
    } else {
        qkv_out[row] = __float2bfloat16(silu_f32(prod));
    }
}

template <int kFp8CTAs, int kNumThreads>
__global__ __launch_bounds__(kNumThreads) void LinearFrontSplitCtaSk4Kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_fp8_e4m3* __restrict__ qkvz_w,
    const float* __restrict__ qkvz_scales, const __nv_bfloat16* __restrict__ b_w,
    const __nv_bfloat16* __restrict__ a_w, const float* __restrict__ a_log_dt,
    const __nv_bfloat16* __restrict__ conv_w, __nv_bfloat16* __restrict__ conv_state4,
    __nv_bfloat16* __restrict__ qkv_out, __nv_bfloat16* __restrict__ z_out,
    float* __restrict__ beta_out, float* __restrict__ g_out) {
    constexpr int kWarpsPerCTA = kNumThreads / 32;
    constexpr int kTilesPerWarp = kTilesK / kWarpsPerCTA;
    constexpr float kEps = 1e-6f;
    static_assert(kTilesK % kWarpsPerCTA == 0, "K split must cover whole tiles");
    static_assert(kFp8CTAs == kFp8Rows / 2, "sk4 expects one group per CTA");

    const int lane = threadIdx.x & 31;
    const int warp_in_cta = threadIdx.x >> 5;

    if (blockIdx.x < (kBRows + kARows)) {
        const int actual = blockIdx.x;
        const bool is_b = actual < kBRows;
        const int row = is_b ? actual : (actual - kBRows);
        const __nv_bfloat16* w = (is_b ? b_w : a_w) + static_cast<size_t>(row) * kHid;

        // Unrolled int4-pair loads keep every b/a load in flight (a strided
        // 4-byte-load loop would serialize into kHid/kNumThreads DRAM round trips).
        constexpr int kElemsPerIter = kNumThreads * 16;
        static_assert(kHid % kElemsPerIter == 0,
                      "b/a int4-pair loop expects a whole iteration count");
        float acc = 0.0f, ss = 0.0f;
#pragma unroll
        for (int it = 0; it < kHid / kElemsPerIter; ++it) {
            const int k = (it * kNumThreads + threadIdx.x) * 16;
            const int4 xraw0 = *reinterpret_cast<const int4*>(x + k);
            const int4 xraw1 = *reinterpret_cast<const int4*>(x + k + 8);
            const int4 wraw0 = *reinterpret_cast<const int4*>(w + k);
            const int4 wraw1 = *reinterpret_cast<const int4*>(w + k + 8);
            float xf[16], wf[16];
            unpack_bf16x16(xraw0, xraw1, xf);
            unpack_bf16x16(wraw0, wraw1, wf);
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                acc += xf[j] * wf[j];
                ss += xf[j] * xf[j];
            }
        }
        acc = warp_sum(acc);
        ss = warp_sum(ss);

        __shared__ float warp_partials[kWarpsPerCTA];
        __shared__ float warp_ss[kWarpsPerCTA];
        if (lane == 0) {
            warp_partials[warp_in_cta] = acc;
            warp_ss[warp_in_cta] = ss;
        }
        __syncthreads();

        if (warp_in_cta == 0) {
            float sum = (lane < kWarpsPerCTA) ? warp_partials[lane] : 0.0f;
            float sumss = (lane < kWarpsPerCTA) ? warp_ss[lane] : 0.0f;
            sum = warp_sum(sum);
            sumss = warp_sum(sumss);
            if (lane == 0) {
                const float rms = rsqrtf(sumss / kHid + kEps);
                const float y = __bfloat162float(__float2bfloat16(sum * rms));
                if (is_b) {
                    beta_out[row] = sigmoid_f32(y);
                } else {
                    const float a_log = a_log_dt[row];
                    const float dt_bias = a_log_dt[kARows + row];
                    g_out[row] = -expf(a_log) * softplus_f32(y + dt_bias);
                }
            }
        }
    } else {
        const int group = blockIdx.x - (kBRows + kARows);
        if (group < (kFp8Rows / 2)) {
            const int row0 = group * 2;
            const float* scales = qkvz_scales + static_cast<size_t>(row0 >> 7) * kScaleBlocksK;
            float acc0 = 0.0f, acc1 = 0.0f, ss = 0.0f;

#pragma unroll
            for (int t = 0; t < kTilesPerWarp; ++t) {
                const int tile = warp_in_cta * kTilesPerWarp + t;
                const int k = tile * 512 + lane * 16;
                const float scale = scales[tile * 4 + (lane >> 3)];
                const int4 xraw0 = *reinterpret_cast<const int4*>(x + k);
                const int4 xraw1 = *reinterpret_cast<const int4*>(x + k + 8);
                float xf[16];
                unpack_bf16x16(xraw0, xraw1, xf);
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    ss += xf[j] * xf[j];
                }

                const __nv_fp8_e4m3* w0 = qkvz_w + static_cast<size_t>(row0) * kHid + k;
                const __nv_fp8_e4m3* w1 = qkvz_w + static_cast<size_t>(row0 + 1) * kHid + k;
                acc0 += scale * dot_fp8x16_bf16x16(w0, xf);
                acc1 += scale * dot_fp8x16_bf16x16(w1, xf);
            }

            acc0 = warp_sum(acc0);
            acc1 = warp_sum(acc1);
            ss = warp_sum(ss);

            __shared__ float part0[kWarpsPerCTA];
            __shared__ float part1[kWarpsPerCTA];
            __shared__ float partss[kWarpsPerCTA];
            if (lane == 0) {
                part0[warp_in_cta] = acc0;
                part1[warp_in_cta] = acc1;
                partss[warp_in_cta] = ss;
            }
            __syncthreads();

            if (warp_in_cta == 0) {
                float a0 = 0.0f, a1 = 0.0f, sst = 0.0f;
#pragma unroll
                for (int wgt = 0; wgt < kWarpsPerCTA; ++wgt) {  // fixed order: deterministic sums
                    a0 += part0[wgt];
                    a1 += part1[wgt];
                    sst += partss[wgt];
                }
                const float rms = rsqrtf(sst / kHid + kEps);
                a0 *= rms;
                a1 *= rms;
                if (row0 < kQkvRows) {
                    qkv_conv_epilogue_pair_warp(row0, a0, a1, conv_w, conv_state4, qkv_out, lane);
                } else {
                    if (lane == 0) {
                        z_epilogue(row0, a0, z_out);
                        z_epilogue(row0 + 1, a1, z_out);
                    }
                }
            }
        }
    }
}

template <int kFp8CTAs, int kNumThreads>
int launch_front_splitcta_sk4(const void* x, const void* qkvz_w, const float* qkvz_scales,
                              const void* b_w, const void* a_w, const float* a_log_dt,
                              const void* conv_w, void* conv_state4, void* qkv_out, void* z_out,
                              float* beta_out, float* g_out, void* stream) {
    LinearFrontSplitCtaSk4Kernel<kFp8CTAs, kNumThreads>
        <<<kFp8CTAs + kBRows + kARows, kNumThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<const __nv_fp8_e4m3*>(qkvz_w), qkvz_scales,
            reinterpret_cast<const __nv_bfloat16*>(b_w),
            reinterpret_cast<const __nv_bfloat16*>(a_w), a_log_dt,
            reinterpret_cast<const __nv_bfloat16*>(conv_w),
            reinterpret_cast<__nv_bfloat16*>(conv_state4),
            reinterpret_cast<__nv_bfloat16*>(qkv_out), reinterpret_cast<__nv_bfloat16*>(z_out),
            beta_out, g_out);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
using namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY;
extern "C" int NexN25ProLinearFrontLaunch(const void* x, const void* qkvz_w,
                                          const float* qkvz_scales, const void* b_w,
                                          const void* a_w, const float* a_log_dt,
                                          const void* conv_w, void* conv_state4, void* qkv_out,
                                          void* z_out, float* beta_out, float* g_out,
                                          void* stream) {
    return launch_front_splitcta_sk4<kFp8Rows / 2, 128>(x, qkvz_w, qkvz_scales, b_w, a_w, a_log_dt,
                                                        conv_w, conv_state4, qkv_out, z_out,
                                                        beta_out, g_out, stream);
}

extern "C" const char* nex_n2_5_pro_linear_front_cuda_error_string(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
