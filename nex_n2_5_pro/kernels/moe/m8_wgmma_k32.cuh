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

#pragma once
// K-major FP8 tiles, 128-byte swizzle, eight-row (1024-byte) atoms.
__device__ __forceinline__ int wg_sw(int row, int k) { return (row * 128 + k) ^ ((row & 7) << 4); }
__device__ __forceinline__ unsigned long long wg_desc(const void* p) {
    return ((unsigned long long)(__cvta_generic_to_shared(p) >> 4) & 0x3fff) | (1ull << 16) |
           (64ull << 32) | (1ull << 62);
}
__device__ __forceinline__ void wg_copy(void* d, const void* s) {
    unsigned a = (unsigned)__cvta_generic_to_shared(d);
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;" ::"r"(a), "l"(s));
}
__device__ __forceinline__ void wg_fence() {
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void wg_mma(float (&c)[4], unsigned long long a, unsigned long long b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
        "{%0,%1,%2,%3}, %4, %5, 1, 1, 1;"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "l"(a), "l"(b));
}
// Independent K32 result: overwrite D; do not read an uninitialized accumulator.
__device__ __forceinline__ void wg_mma_overwrite(float (&c)[4], unsigned long long a,
                                                 unsigned long long b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
        "{%0,%1,%2,%3}, %4, %5, 0, 1, 1;"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "l"(a), "l"(b));
}
__device__ __forceinline__ void wg_wait() {
    asm volatile("wgmma.commit_group.sync.aligned;\n wgmma.wait_group.sync.aligned 0;" ::
                     : "memory");
}
__device__ __forceinline__ void wg_proxy() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
