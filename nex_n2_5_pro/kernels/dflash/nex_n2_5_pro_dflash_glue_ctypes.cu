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

// DFlash round glue: control-driven noise-block preparation (graph-capturable).
//
// Reads the M8 control block AFTER a verify unit and prepares the next draft
// forward's inputs with no host readback:
//   noise8[0]   = embed[pending_target_id]   (word 0 = the anchor)
//   noise8[1..7]= embed[mask_token_id]
//   df_meta     = { produced (=OUTPUT_COUNT), committed_len }  int32[2]
// Downstream ctl-driven kernels read df_meta[0] as `p` (feature rows) and
// ctl word 1 as `start`. INVARIANCE: absolute positions only.
// DF-family TU (reads the M8 ctl ABI via the shared header).

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY DF
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "../common/m8_control_block.cuh"

namespace nexrt::sm90::nex_n2_5_pro::DFlash {
constexpr int kHid = 4096;
constexpr int kVecs = kHid / 8;
constexpr int kM = nexrt::sm90::nex_n2_5_pro::M8::m8ctl::kM;
namespace m8ctl = nexrt::sm90::nex_n2_5_pro::M8::m8ctl;

__global__ void DfNoiseEmbedKernel(const __nv_bfloat16* __restrict__ embed_w,
                                   const int* __restrict__ ctl, int mask_token_id,
                                   __nv_bfloat16* __restrict__ noise8, int* __restrict__ df_meta) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif
    const int j = blockIdx.x;
    if (j >= kM) {
        return;
    }
    const int token = j == 0 ? ctl[m8ctl::kPendingTargetId] : mask_token_id;
    const int4* src = reinterpret_cast<const int4*>(embed_w + static_cast<size_t>(token) * kHid);
    int4* dst = reinterpret_cast<int4*>(noise8 + static_cast<size_t>(j) * kHid);
    for (int i = threadIdx.x; i < kVecs; i += blockDim.x) {
        dst[i] = src[i];
    }
    if (j == 0 && threadIdx.x == 0) {
        df_meta[0] = ctl[m8ctl::kOutputCount];
        df_meta[1] = ctl[m8ctl::kCommittedLen];
    }
}

}  // namespace nexrt::sm90::nex_n2_5_pro::DFlash
using namespace nexrt::sm90::nex_n2_5_pro::DFlash;
extern "C" {
int NexN25ProDFlashNoiseEmbedLaunch(const void* embed_w, const void* ctl, int mask_token_id,
                                    void* noise8, void* df_meta, void* stream) {
    DfNoiseEmbedKernel<<<dim3(kM), dim3(256), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(embed_w), static_cast<const int*>(ctl), mask_token_id,
        static_cast<__nv_bfloat16*>(noise8), static_cast<int*>(df_meta));
    return static_cast<int>(cudaGetLastError());
}
}

// Proposal commit: draft ids [7] -> ctl VERIFY_INPUT_IDS[1..7]; word [0] takes
// the anchor from pending_target_id. Single warp; graph-capturable.
namespace nexrt::sm90::nex_n2_5_pro::DFlash {
__global__ void DfProposalCommitKernel(const int* __restrict__ ids7, int* __restrict__ ctl) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif
    const int t = threadIdx.x;
    if (t == 0) {
        ctl[m8ctl::kVerifyInputIds] = ctl[m8ctl::kPendingTargetId];
    } else if (t < kM) {
        ctl[m8ctl::kVerifyInputIds + t] = ids7[t - 1];
    }
}
}  // namespace nexrt::sm90::nex_n2_5_pro::DFlash
extern "C" int NexN25ProDFlashProposalCommitLaunch(const void* ids7, void* ctl, void* stream) {
    DfProposalCommitKernel<<<dim3(1), dim3(32), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const int*>(ids7), static_cast<int*>(ctl));
    return static_cast<int>(cudaGetLastError());
}
