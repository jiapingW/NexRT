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

// M=4 embedding gather -- four draft token embeddings in ONE operation.
//
// MTP target-verification consumes four ORDERED draft ids d0..d3 for one request. This kernel gathers
// all four embedding rows in a single launch (one CTA per draft row, same int4-vectorized row copy as
// the M1 nex_n2_5_pro_embed_gather kernel) and maintains a device-visible draft cursor proving consumption:
//
//   drafts  int32[4]          device-resident draft token ids (d0..d3, consumed in index order)
//   x4      [4, 4096] bf16    gathered embedding rows (row j = embedding of d_j, logical slot S+j)
//   cursor  int32[6]          [d0', d1', d2', d3', count, status] -- each CTA records the id it
//                             actually consumed and atomically bumps count; status is set to 1 if any
//                             id is outside [0, vocab). Serving clears cursor once per request;
//                             count then advances by four per graph replay and status stays sticky.
//
// Like the M1 kernel this is a pure replicated-table gather: every TP rank holds the full
// embed_tokens [vocab, 4096] bf16 table and gathers identical rows -- no cross-rank communication.
// Reads ids ON-DEVICE (no host token materialization, graph-capturable).

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M4
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace nexrt::sm90::nex_n2_5_pro::M4 {
constexpr int kHid = 4096;
constexpr int kVecs = kHid / 8;  // 512 int4 (8 bf16 each)
constexpr int kM = 4;
constexpr int kVocab = 248320;

__global__ void M4EmbedGatherKernel(const __nv_bfloat16* __restrict__ embed_w,
                                    const int* __restrict__ drafts,
                                    __nv_bfloat16* __restrict__ x4_out, int* __restrict__ cursor) {
    const int j = blockIdx.x;  // draft row 0..3
    if (j >= kM) {
        return;
    }
    const int token = drafts[j];
    const bool bad = (token < 0) || (token >= kVocab);
    const int safe_token = bad ? 0 : token;  // never read out of bounds; badness lands in status
    const int4* src =
        reinterpret_cast<const int4*>(embed_w + static_cast<size_t>(safe_token) * kHid);
    int4* dst = reinterpret_cast<int4*>(x4_out + static_cast<size_t>(j) * kHid);
    for (int i = threadIdx.x; i < kVecs; i += blockDim.x) {
        dst[i] = src[i];
    }
    if (threadIdx.x == 0) {
        cursor[j] = token;          // the id this row actually consumed
        atomicAdd(&cursor[kM], 1);  // count -> 4 when all rows consumed
        if (bad) {
            atomicExch(&cursor[kM + 1], 1);
        }
    }
}

extern "C" {
// embed_w [vocab,4096] bf16 (full, replicated); drafts int32[4] device; x4_out [4,4096] bf16;
// cursor int32[6] (caller zeroes at the request boundary).
int NexN25ProM4EmbedGatherLaunch(const void* embed_w, const void* drafts, void* x4_out,
                                 void* cursor, void* stream) {
    M4EmbedGatherKernel<<<dim3(kM), dim3(256), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(embed_w), static_cast<const int*>(drafts),
        static_cast<__nv_bfloat16*>(x4_out), static_cast<int*>(cursor));
    return static_cast<int>(cudaGetLastError());
}
}  // extern "C"

}  // namespace nexrt::sm90::nex_n2_5_pro::M4