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

// Control-block HOST MIRROR: the last node of a captured decode graph copies the int32 control words
// (M=4 ctl, or the bs=1 [token, S] slot) into host-mapped pinned memory and publishes a free-running
// generation word LAST, after a system-scope fence. The host learns a round's result by spinning on
// that word -- no cudaMemcpy, no event, no stream synchronization -- and keeps the next graph queued
// while it reads the previous words (see ctl_mirror.py, mtp/serve.py, DecodeEngine.generate).
//
// Layout: mirror = int32[2][words + 1]; slot = gen & 1; word `words` of a slot is its generation flag.
// `gen` is a device counter incremented once per launch and never reset (graph replays stay uniform),
// like the reduce-ring generations. One warp; ~6 us of graph time on H100.
#include <cuda_runtime.h>

namespace {

__global__ void CtlMirrorKernel(const int* __restrict__ ctl, int* __restrict__ mirror,
                                unsigned* __restrict__ gen, int words) {
    const unsigned g = *gen + 1u;  // every lane reads the same value
    int* dst = mirror + (g & 1u) * (words + 1);
    if (static_cast<int>(threadIdx.x) < words) dst[threadIdx.x] = ctl[threadIdx.x];
    __threadfence_system();  // words visible to the host before the flag
    __syncwarp();
    if (threadIdx.x == 0) {
        *reinterpret_cast<volatile int*>(dst + words) = static_cast<int>(g);
        *gen = g;
    }
}

}  // namespace

extern "C" {

// Host-mapped (zero-copy) mirror of 2 * (words + 1) int32, zeroed. Returns the host pointer; 0 on failure.
void* NexN25ProCtlMirrorAlloc(int words) {
    void* p = nullptr;
    const size_t bytes = sizeof(int) * 2 * static_cast<size_t>(words + 1);
    if (cudaHostAlloc(&p, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess)
        return nullptr;
    for (size_t i = 0; i < bytes / sizeof(int); ++i) static_cast<int*>(p)[i] = 0;
    return p;
}

void* NexN25ProCtlMirrorDevicePtr(void* host) {
    void* d = nullptr;
    if (cudaHostGetDevicePointer(&d, host, 0) != cudaSuccess) return nullptr;
    return d;
}

int NexN25ProCtlMirrorFree(void* host) { return static_cast<int>(cudaFreeHost(host)); }

// One warp; launch it as the LAST node of the round / step.
int NexN25ProCtlMirrorLaunch(const void* ctl, void* mirror_dev, void* gen, int words,
                             void* stream) {
    if (words < 1 || words > 32) return static_cast<int>(cudaErrorInvalidValue);
    CtlMirrorKernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const int*>(ctl), static_cast<int*>(mirror_dev), static_cast<unsigned*>(gen),
        words);
    return static_cast<int>(cudaGetLastError());
}

}  // extern "C"
