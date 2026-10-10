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

// Residual add for the decode graph: x_bf16 += out_fp32, in place. Replaces the eager torch
// `x.copy_((x.float() + out).to(bf16))` so NO torch op sits in the replayed graph — each decoder
// sub-block writes its reduced fp32 output, this folds it back into the bf16 residual stream.
//
// Numerics MUST match torch exactly: read x as fp32, add the fp32 out, round-to-nearest-even to bf16
// (__float2bfloat16 is RNE, same as torch's .to(bfloat16)). One CTA, grid-stride over n (=HID=4096).

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY common
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace nexrt::sm90::nex_n2_5_pro::common {
__global__ void ResidualAddKernel(__nv_bfloat16* __restrict__ x, const float* __restrict__ out,
                                  int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        x[i] = __float2bfloat16(__bfloat162float(x[i]) + out[i]);
}
}  // namespace nexrt::sm90::nex_n2_5_pro::common
using namespace nexrt::sm90::nex_n2_5_pro::common;
extern "C" {
// x [n] bf16 (in place, += out); out [n] fp32.
int NexN25ProResidualAddLaunch(void* x, const void* out, int n, void* stream) {
    ResidualAddKernel<<<1, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(x), static_cast<const float*>(out), n);
    return static_cast<int>(cudaGetLastError());
}
const char* NexN25ProResidualAddCudaErrorString(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
}
