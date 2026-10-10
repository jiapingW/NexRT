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

// M=8 embedding gather -- eight target-row token embeddings in ONE launch.
//
// The M=8 verification unit consumes eight ORDERED externally supplied target input
// ids t0..t7 for one request. This kernel gathers all eight embedding rows in a single
// launch (one CTA per row, int4-vectorized row copy) and maintains a device-visible
// consumption cursor:
//
//   ctl     int32[32]         the M=8 control block; ids live at words
//                             m8ctl::kVerifyInputIds .. +7 (device-resident, so the
//                             captured graph replays against fresh ids with no host
//                             work between its nodes)
//   x8      [8, 4096] bf16    gathered embedding rows (row j = embedding of t_j,
//                             logical position committed_len + j)
//   cursor  int32[10]         [t0'..t7', count, status] -- each CTA records the id it
//                             actually consumed and atomically bumps count; status is
//                             set to 1 if any id is outside [0, vocab). The runtime
//                             clears cursor once per request; count then advances by
//                             eight per graph replay and status stays sticky.
//
// Pure replicated-table gather: every TP rank holds the full embed_tokens
// [vocab, 4096] bf16 table and gathers identical rows -- no cross-rank communication.
// Reads ids ON-DEVICE (no host token materialization, graph-capturable). Moving
// [8, 4096] bf16 is 64 KiB total; this launch sits at the launch floor, not at any
// bandwidth limit.
//
// M=8-family TU: no BS=1 or M=4 production symbol is included or launched here.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M8
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "../common/m8_control_block.cuh"

namespace nexrt::sm90::nex_n2_5_pro::M8 {
constexpr int kHid = 4096;
constexpr int kVecs = kHid / 8;  // 512 int4 (8 bf16 each)
constexpr int kM8 = m8ctl::kM;
constexpr int kVocab = 248320;
constexpr int kCursorLen = kM8 + 2;  // [ids 0..7, count, status]

__global__ void M8EmbedGatherKernel(const __nv_bfloat16* __restrict__ embed_w,
                                    const int* __restrict__ ctl, __nv_bfloat16* __restrict__ x8_out,
                                    int* __restrict__ cursor) {
    const int j = blockIdx.x;  // target row 0..7
    if (j >= kM8) {
        return;
    }
    const int token = ctl[m8ctl::kVerifyInputIds + j];
    const bool bad = (token < 0) || (token >= kVocab);
    const int safe_token = bad ? 0 : token;  // never read out of bounds; badness lands in status
    const int4* src =
        reinterpret_cast<const int4*>(embed_w + static_cast<size_t>(safe_token) * kHid);
    int4* dst = reinterpret_cast<int4*>(x8_out + static_cast<size_t>(j) * kHid);
    for (int i = threadIdx.x; i < kVecs; i += blockDim.x) {
        dst[i] = src[i];
    }
    if (threadIdx.x == 0) {
        cursor[j] = token;           // the id this row actually consumed
        atomicAdd(&cursor[kM8], 1);  // count -> 8 when all rows consumed
        if (bad) {
            atomicExch(&cursor[kM8 + 1], 1);
        }
    }
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M8
using namespace nexrt::sm90::nex_n2_5_pro::M8;
extern "C" {
// embed_w [vocab,4096] bf16 (full, replicated); ctl int32[32] M8 control block;
// x8_out [8,4096] bf16; cursor int32[10] (runtime zeroes at the request boundary).
int NexN25ProM8EmbedGatherLaunch(const void* embed_w, const void* ctl, void* x8_out, void* cursor,
                                 void* stream) {
    M8EmbedGatherKernel<<<dim3(kM8), dim3(256), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(embed_w), static_cast<const int*>(ctl),
        static_cast<__nv_bfloat16*>(x8_out), static_cast<int*>(cursor));
    return static_cast<int>(cudaGetLastError());
}
int NexN25ProM8EmbedGatherM() { return kM8; }
int NexN25ProM8EmbedGatherCursorLen() { return kCursorLen; }
const char* NexN25ProM8EmbedGatherCudaErrorString(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
}
