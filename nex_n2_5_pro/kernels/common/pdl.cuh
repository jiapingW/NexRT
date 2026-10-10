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

// Programmatic Dependent Launch (PDL) support for SM90 kernels.
//
// A participating kernel calls NEXRT_PDL_ENTRY_SYNC() before consuming its
// predecessor's output. The matching host launcher uses
// cudaLaunchAttributeProgrammaticStreamSerialization, allowing CUDA to admit
// the successor grid while the predecessor drains without changing the
// same-stream memory dependency.
#pragma once

#include <cuda_runtime.h>

#if defined(__CUDACC__)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define NEXRT_PDL_ENTRY_SYNC() cudaGridDependencySynchronize()
#else
#define NEXRT_PDL_ENTRY_SYNC() \
    do {                       \
    } while (0)
#endif
#endif

template <typename... Expected, typename... Actual>
static inline cudaError_t NexrtPdlLaunchKernel(dim3 grid, dim3 block, size_t dynamic_smem,
                                               cudaStream_t stream, void (*kernel)(Expected...),
                                               Actual... args) {
    static_assert(sizeof...(Expected) == sizeof...(Actual),
                  "NexrtPdlLaunchKernel argument count mismatch");
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = dynamic_smem;
    config.stream = stream;
    cudaLaunchAttribute attributes[1];
    attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attributes[0].val.programmaticStreamSerializationAllowed = 1;
    config.attrs = attributes;
    config.numAttrs = 1;
    return cudaLaunchKernelEx(&config, kernel, static_cast<Expected>(args)...);
}
