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

// nex_n2_5_pro decode embedding: token_id -> hidden x[4096]. The INPUT bookend of the one-graph decode loop.
//
// REPLICATED form (design choice): every TP rank holds the full embed_tokens [vocab, 4096] bf16 (~2 GB/rank)
// and all ranks see the same token_id, so the gather is a pure per-rank index-select -> x[4096] with NO
// cross-rank communication (the residual stream is replicated). token_id is read ON-DEVICE from slot[0] (the
// [token_id, S] decode slot the sampler writes), so a captured graph reads the fed-back token each replay.
//
// Pure copy of one 8 KB row: 4096 bf16 = 512 int4. One CTA, grid-stride over the 512 int4 vectors.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY bs1
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY {
constexpr int kHid = 4096;
constexpr int kVecs = kHid / 8;  // 512 int4 (8 bf16 each)

__global__ void EmbedGatherKernel(const __nv_bfloat16* __restrict__ embed_w,
                                  const int* __restrict__ slot, __nv_bfloat16* __restrict__ x_out) {
    // generate() may have queued one replay before observing a sampler error (-1).
    // Keep that replay in bounds while the host drains it and rejects the request.
    const int token = max(slot[0], 0);
    const int4* src = reinterpret_cast<const int4*>(embed_w + static_cast<size_t>(token) * kHid);
    int4* dst = reinterpret_cast<int4*>(x_out);
    for (int i = threadIdx.x; i < kVecs; i += blockDim.x) dst[i] = src[i];
}

}  // namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
using namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY;
extern "C" {
// embed_w [vocab, 4096] bf16 (full, replicated); slot int[2] = [token_id, S]; x_out [4096] bf16.
int NexN25ProEmbedGatherLaunch(const void* embed_w, const void* slot, void* x_out, void* stream) {
    EmbedGatherKernel<<<1, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(embed_w), static_cast<const int*>(slot),
        static_cast<__nv_bfloat16*>(x_out));
    return static_cast<int>(cudaGetLastError());
}
const char* NexN25ProEmbedGatherCudaErrorString(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
}
