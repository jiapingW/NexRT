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

// M=8 DFlash feature tap -- copy the residual x8 into one tap slot.
//
// The DFlash draft is conditioned on the target's hidden states at layers
// [1,9,17,25,33,41,49,57]: after each of those layers completes (post-MoE
// residual), this kernel snapshots x8 [8, 4096] bf16 into the tap ring slot
// for that layer. The draft's fc consumes taps[k][:produced] next round
// (produced read from the M8 control block by the consumer, not here).
//
// 64 KiB copy, one CTA per row -- launch-floor cost, PDL-ready (entry sync is
// a no-op on serial launches).
//
// M=8-family TU: no BS=1 or M=4 production symbol is included or launched here.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M8
#endif

#include <cuda_bf16.h>
#include "../common/pdl.cuh"
#include <cuda_runtime.h>

#include <cstdint>

#include "../common/m8_control_block.cuh"

namespace nexrt::sm90::nex_n2_5_pro::M8 {
constexpr int kHid = 4096;
constexpr int kVecs = kHid / 8;  // 512 int4 (8 bf16 each)
constexpr int kM8 = m8ctl::kM;

__global__ void M8FeatTapKernel(const __nv_bfloat16* __restrict__ x8,
                                __nv_bfloat16* __restrict__ tap_out) {
    NEXRT_PDL_ENTRY_SYNC();
    const int j = blockIdx.x;
    if (j >= kM8) {
        return;
    }
    const int4* src = reinterpret_cast<const int4*>(x8 + static_cast<size_t>(j) * kHid);
    int4* dst = reinterpret_cast<int4*>(tap_out + static_cast<size_t>(j) * kHid);
    for (int i = threadIdx.x; i < kVecs; i += blockDim.x) {
        dst[i] = src[i];
    }
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M8
using namespace nexrt::sm90::nex_n2_5_pro::M8;
extern "C" {
// x8 [8,4096] bf16; tap_out [8,4096] bf16 (one tap-layer slot).
int NexN25ProM8FeatTapLaunch(const void* x8, void* tap_out, void* stream) {
    M8FeatTapKernel<<<dim3(kM8), dim3(256), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(x8), static_cast<__nv_bfloat16*>(tap_out));
    return static_cast<int>(cudaGetLastError());
}

int NexN25ProM8FeatTapRows() { return kM8; }
}
