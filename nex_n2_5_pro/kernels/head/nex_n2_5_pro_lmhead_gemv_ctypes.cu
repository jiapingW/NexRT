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

// nex_n2_5_pro lm_head GEMV (one TP rank): logits_local[31040] = h @ W_local[31040,4096]^T
//
// Column/vocab-parallel: h (post model.norm) is REPLICATED [4096] bf16; W_local is this rank's vocab
// shard [31040,4096] bf16 (row-major). Each output logit is a complete dot over k=4096 -> NO cross-rank
// reduction. Streams the whole 254 MB shard once -> HBM-bandwidth-bound (~76 us wall @100% MBU).
//
// One CTA computes each row. CTA 0 also computes inv_rms while streaming h. W_fused contains the folded
// (1+norm_w) factor; inv_rms is applied downstream because positive scaling does not change top-k IDs.
// Output is FP32 (lm_head/sampler are precision-sensitive; the top-k consumes these logits directly).
#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY bs1
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY {
constexpr int kDim = 4096;
constexpr int kDimPairs = kDim / 2;  // 2048  (__nv_bfloat162 pairs)
constexpr int kRows = 31040;         // vocab / TP = 248320 / 8
constexpr float kInvDim = 1.0f / static_cast<float>(kDim);
static_assert(kDim % 2 == 0, "kDim must be even for bf16x2");
// One CTA per row + fused RMS in CTA 0 (raw logits + inv_rms; W is (1+norm_w)-folded).
template <int NT>
__global__ __launch_bounds__(NT) void LmHeadGemvRowRmsCta0Kernel(
    const __nv_bfloat16* __restrict__ h, const __nv_bfloat16* __restrict__ Wf,
    float* __restrict__ logits, float* __restrict__ inv_rms, float eps) {
    const int row = blockIdx.x;
    if (row >= kRows) {
        return;
    }
    const auto* __restrict__ h2 = reinterpret_cast<const __nv_bfloat162*>(h);
    const auto* __restrict__ w2 =
        reinterpret_cast<const __nv_bfloat162*>(Wf + static_cast<size_t>(row) * kDim);
    float acc = 0.0f, rms_acc = 0.0f;
    for (int p = threadIdx.x; p < kDimPairs; p += NT) {
        const float2 hf = __bfloat1622float2(h2[p]);
        const float2 wf = __bfloat1622float2(w2[p]);
        acc += hf.x * wf.x + hf.y * wf.y;
        if (row == 0) {
            rms_acc += hf.x * hf.x + hf.y * hf.y;
        }  // only CTA0 does the variance reduce
    }
    __shared__ typename cub::BlockReduce<float, NT>::TempStorage br_tmp;
    const float sum = cub::BlockReduce<float, NT>(br_tmp).Sum(acc);
    if (threadIdx.x == 0) {
        logits[row] = sum;
    }  // RAW (inv_rms applied downstream)
    if (row == 0) {
        __syncthreads();  // reuse br_tmp for the CTA0 variance reduce
        const float sq = cub::BlockReduce<float, NT>(br_tmp).Sum(rms_acc);
        if (threadIdx.x == 0) {
            inv_rms[0] = rsqrtf(sq * kInvDim + eps);
        }
    }
}

template <int NT>
int LaunchRowRms(const void* h, const void* Wf, void* logits, float* inv_rms, float eps,
                 void* stream) {
    LmHeadGemvRowRmsCta0Kernel<NT><<<dim3(kRows), dim3(NT), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(h), static_cast<const __nv_bfloat16*>(Wf),
        static_cast<float*>(logits), inv_rms, eps);
    return static_cast<int>(cudaGetLastError());
}
}  // namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
using namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY;
extern "C" {
int NexN25ProLmHeadGemvLaunch(const void* h, const void* Wf, void* o, float* r, float e, void* s) {
    return LaunchRowRms<256>(h, Wf, o, r, e, s);
}
#ifndef NEXRT_MTP_RUNTIME_ONLY
const char* NexN25ProLmHeadCudaErrorString(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
#endif
}
