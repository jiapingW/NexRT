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

// MTP draft-loop glue: device-side bookkeeping kernels for one MTP round.
//
//   NexN25ProMtpPrepLaunch          post-verify: extend base, anchor copy, stop scan (1 warp)
//   NexN25ProMtpRowSelectLaunch     y_sel = x4[accepted_draft_count]  (the extend's g-select)
//   NexN25ProMtpStatusFoldLaunch    folds draft collective failures into verify status
//
// Everything reads/writes the 32-word v2 control block (m4_control_block.cuh); offsets are
// constexpr, single writer per field, same-stream visibility -- no atomics anywhere.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M4
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "../common/m4_control_block.cuh"

namespace nexrt::sm90::nex_n2_5_pro::M4 {
constexpr int kThreads = 256;
constexpr int kHid = 4096;
constexpr int kPerThread = kHid / kThreads;

// ---- post-verify prep -----------------------------------------------------------------
__global__ void MtpPrepKernel(int* __restrict__ ctl, int* __restrict__ ext_ctl,
                              const int* __restrict__ eos_ids, int n_eos) {
    if (threadIdx.x != 0) return;
    // ALWAYS stage the extend base, even on a fail-closed verify: the verify commit's
    // failure path publishes commit_count = 0 with C unchanged, so the base parks AT C
    // and the unconditional catch-up writes only beyond-frontier rows that are never
    // attended while C is frozen. A stale (or first-round zero) base would instead let
    // the catch-up overwrite committed MTP KV lines -- unrecoverable without a reload.
    ext_ctl[m4ctl::kCommittedLen] = ctl[m4ctl::kCommittedLen] - ctl[m4ctl::kCommitCount];
    if (ctl[m4ctl::kStatus] != m4ctl::kStatusOk) return;  // fail-closed: no anchor/stop updates
    ctl[m4ctl::kVerifyInputIds] = ctl[m4ctl::kPendingTargetId];
    int stop = -1;
    const int n = ctl[m4ctl::kOutputCount];
    for (int j = 0; j < n && stop < 0; ++j) {
        const int tok = ctl[m4ctl::kEmittedIds + j];
        for (int e = 0; e < n_eos; ++e) {
            if (tok == eos_ids[e]) {
                stop = j;
                break;
            }
        }
    }
    ctl[m4ctl::kStopPos] = stop;
    if (stop >= 0) {
        ctl[m4ctl::kFlags] |= m4ctl::kFlagStopHit;
        ctl[m4ctl::kOutputCount] = stop + 1;  // truncate the drained stream
    }
}

// ---- the extend's g-select: y_sel = x4[select_index] ----------------------------------
__global__ void MtpRowSelectKernel(const __nv_bfloat16* __restrict__ x4,
                                   const int* __restrict__ ctl, __nv_bfloat16* __restrict__ y_sel) {
    int sel = ctl[m4ctl::kAcceptedDraftCount];
    if (sel < 0) sel = 0;
    if (sel >= m4ctl::kM) sel = m4ctl::kM - 1;
    const __nv_bfloat16* src = x4 + static_cast<int64_t>(sel) * kHid;
    const int base = threadIdx.x * kPerThread;
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) y_sel[base + i] = src[base + i];
}

// ---- draft-side status fold ----------------------------------------------------------
// The MTP block's collectives (o_proj reduce, MoE reduce, head reducer) keep their own
// sticky status words that the verify commit's fail-closed decision never sees. Folding
// them into ws12_status (an existing sticky producer input) makes a draft-side
// collective failure veto the commit COLLECTIVELY, preserving the "drafts cannot
// corrupt output" invariant even under rank-skew/timeout on the draft path.
__global__ void MtpStatusFoldKernel(const int* __restrict__ a, const int* __restrict__ b,
                                    const int* __restrict__ c, int* __restrict__ dst) {
    if (threadIdx.x != 0) return;
    const int s = a[0] | b[0] | c[0];
    if (s != 0) dst[0] |= s;
}

extern "C" int NexN25ProMtpStatusFoldLaunch(const void* a, const void* b, const void* c, void* dst,
                                            void* stream) {
    MtpStatusFoldKernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const int*>(a), static_cast<const int*>(b), static_cast<const int*>(c),
        static_cast<int*>(dst));
    return static_cast<int>(cudaGetLastError());
}

extern "C" int NexN25ProMtpPrepLaunch(void* ctl, void* ext_ctl, const void* eos_ids, int n_eos,
                                      void* stream) {
    MtpPrepKernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<int*>(ctl), static_cast<int*>(ext_ctl), static_cast<const int*>(eos_ids),
        n_eos);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int NexN25ProMtpRowSelectLaunch(const void* x4, const void* ctl, void* y_sel,
                                           void* stream) {
    MtpRowSelectKernel<<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(x4), static_cast<const int*>(ctl),
        static_cast<__nv_bfloat16*>(y_sel));
    return static_cast<int>(cudaGetLastError());
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M4