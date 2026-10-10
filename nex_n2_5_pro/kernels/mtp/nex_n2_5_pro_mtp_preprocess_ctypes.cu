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

// MTP fused preprocess, LOCAL-WINDOW form: embed-select + Gemma norms + this rank's fc slice.
//
// The MTP fc is TP row-parallel over its 8192 input columns, so rank r only ever consumes
// cat[:, r*1024 : r*1024+1024] of the conceptual concat  cat = [ gemma(emb[ids], we) |
// gemma(gemma(y, wf), wh) ].  This kernel therefore emits ONLY the local [rows, 1024]
// window (contiguous, o_proj-ctx-shaped):
//   ranks 0..3: a window of the EMBED half   -- one norm  (we = mtp.pre_fc_norm_embedding)
//   ranks 4..7: a window of the HIDDEN half  -- two norms (wf = final norm, wh = mtp.pre_fc_norm_hidden)
// The hidden half applies TWO norms because the raw-hidden contract never materializes
// post-final-norm hiddens: extend rows pass y = the target's
// pre-model.norm residual with wf = model.norm; chain rows pass the MTP layer's own raw
// output with wf = mtp.norm. All norms are Gemma-style rms(x)*(1+w); math fp32, io bf16.
//
// ids[j] is read ON-DEVICE (graph-capturable): ctl+kEmittedIds for the extend,
// ctl+kChainSlotA / ctl+kChainSlotB for the chain steps (word 0 of a slot is the token).
// Out-of-range ids clamp to 0 defensively; every producer (argmax select, verify
// commit) is range-correct by construction.
//
// Grid <<<rows, 256>>>: full-row register pass for the rms sums (16 elems/thread), then
// a 4-elem/thread window re-read for the write.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M4
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace nexrt::sm90::nex_n2_5_pro::M4 {
constexpr int kHid = 4096;
constexpr int kThreads = 256;
constexpr int kPerThread = kHid / kThreads;  // 16 (rms pass)
constexpr int kWin = 1024;
constexpr int kWinPerThread = kWin / kThreads;  // 4  (window write)
constexpr int kVocab = 248320;
constexpr float kEps = 1e-6f;

__device__ float BlockSumOfSquares(const float* vals, int n, float* smem) {
    float acc = 0.f;
    for (int i = 0; i < n; ++i) acc += vals[i] * vals[i];
    smem[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    const float total = smem[0];
    __syncthreads();  // smem reused by the next reduction
    return total;
}

__global__ void MtpPreprocessKernel(const __nv_bfloat16* __restrict__ embed_w,
                                    const int* __restrict__ ids,
                                    const __nv_bfloat16* __restrict__ hidden,
                                    const __nv_bfloat16* __restrict__ final_norm_w,
                                    const __nv_bfloat16* __restrict__ pre_fc_hidden_w,
                                    const __nv_bfloat16* __restrict__ pre_fc_embed_w,
                                    __nv_bfloat16* __restrict__ cat_local, int rank) {
    __shared__ float smem[kThreads];
    const int row = blockIdx.x;
    const int base = threadIdx.x * kPerThread;
    __nv_bfloat16* out = cat_local + static_cast<int64_t>(row) * kWin;
    const int wbase = threadIdx.x * kWinPerThread;

    if (rank < 4) {
        // ---- embed half window: gemma(emb[tok], we)[win] ----
        int tok = ids[row];
        if (tok < 0 || tok >= kVocab) tok = 0;
        const __nv_bfloat16* erow = embed_w + static_cast<int64_t>(tok) * kHid;
        float e[kPerThread];
#pragma unroll
        for (int i = 0; i < kPerThread; ++i) e[i] = __bfloat162float(erow[base + i]);
        const float inv = rsqrtf(BlockSumOfSquares(e, kPerThread, smem) / kHid + kEps);
        const int win0 = rank * kWin;
#pragma unroll
        for (int i = 0; i < kWinPerThread; ++i) {
            const int c = win0 + wbase + i;
            const float w = __bfloat162float(pre_fc_embed_w[c]);
            out[wbase + i] = __float2bfloat16(__bfloat162float(erow[c]) * inv * (1.f + w));
        }
        return;
    }

    // ---- hidden half window: gemma(gemma(y, wf), wh)[win] ----
    const __nv_bfloat16* yrow = hidden + static_cast<int64_t>(row) * kHid;
    float y[kPerThread];
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) y[i] = __bfloat162float(yrow[base + i]);
    const float inv1 = rsqrtf(BlockSumOfSquares(y, kPerThread, smem) / kHid + kEps);
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) {
        const float w = __bfloat162float(final_norm_w[base + i]);
        y[i] = y[i] * inv1 * (1.f + w);  // post-final-norm value, registers only
    }
    const float inv2 = rsqrtf(BlockSumOfSquares(y, kPerThread, smem) / kHid + kEps);
    const int win0 = (rank - 4) * kWin;
#pragma unroll
    for (int i = 0; i < kWinPerThread; ++i) {
        const int c = win0 + wbase + i;
        const float wf = __bfloat162float(final_norm_w[c]);
        const float wh = __bfloat162float(pre_fc_hidden_w[c]);
        const float t = __bfloat162float(yrow[c]) * inv1 * (1.f + wf);
        out[wbase + i] = __float2bfloat16(t * inv2 * (1.f + wh));
    }
}

extern "C" int NexN25ProMtpPreprocessLaunch(const void* embed_w, const void* ids,
                                            const void* hidden, const void* final_norm_w,
                                            const void* pre_fc_hidden_w, const void* pre_fc_embed_w,
                                            void* cat_local, int rank, int rows, void* stream) {
    MtpPreprocessKernel<<<rows, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(embed_w), static_cast<const int*>(ids),
        static_cast<const __nv_bfloat16*>(hidden), static_cast<const __nv_bfloat16*>(final_norm_w),
        static_cast<const __nv_bfloat16*>(pre_fc_hidden_w),
        static_cast<const __nv_bfloat16*>(pre_fc_embed_w), static_cast<__nv_bfloat16*>(cat_local),
        rank);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M4