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

#include "m8_wgmma_k32.cuh"
constexpr int WG2Depth = 4;
constexpr int WG2Ring = 4096;
constexpr int WG2Out = WG2Ring + WG2Depth * 8192;
constexpr int WG2Dyn = WG2Out + 2048;
__device__ __forceinline__ void wg2_prefetch(char* stage, const __nv_fp8_e4m3* base) {
#pragma unroll
    for (int i = threadIdx.x; i < 512; i += 128) {
        const int row = i >> 3, col = (i & 7) * 16;
        wg_copy(stage + wg_sw(row, col), base + row * 128 + col);
    }
}
__device__ __forceinline__ void wg2_body(const M8RoutePlan* plan, const int32_t* packed8,
                                         const float* rawh8, const __nv_fp8_e4m3* w2_raw,
                                         const float* w2_scales, float* l2p) {
    const int desc = blockIdx.x >> 3, rb = blockIdx.x & 7;
    if (desc >= plan->n_desc) return;
    const int e = plan->expert_id[desc];
    extern __shared__ __align__(1024) char sm[];
    auto* sh = reinterpret_cast<__nv_fp8_e4m3*>(sm);
    int* tok = reinterpret_cast<int*>(sm + 1024);
    int* off = tok + 8;
    int* hsrc = off + 8;
    int* count = hsrc + 8;
    float* factors = reinterpret_cast<float*>(sm + 1152);
    float* ws = factors + 16;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (warp == 0) {
        int slot = -1;
        if (lane < 8) {
#pragma unroll
            for (int s = 0; s < 10; ++s)
                if (packed8[lane * 64 + s] == e) slot = s;
        }
        const bool active = lane < 8 && slot >= 0;
        unsigned mask = __ballot_sync(0xffffffffu, active);
        if (active) {
            int n = __popc(mask & ((1u << lane) - 1));
            tok[n] = lane;
            off[n] = (lane * 11 + slot) * 4096;
            hsrc[n] = lane * 1408 + slot * 128;
            factors[n * 2 + 1] = __int_as_float(packed8[lane * 64 + 10 + slot]);
        }
        if (lane == 0) count[0] = __popc(mask);
        if (lane < 4) ws[lane] = w2_scales[e * 32 + rb * 4 + lane];
    }
    const auto* wbase = w2_raw + (static_cast<size_t>(e) * 4096 + rb * 512) * 128;
#pragma unroll
    for (int st = 0; st < WG2Depth - 1; ++st) {
        wg2_prefetch(sm + WG2Ring + st * 8192, wbase + st * 8192);
        asm volatile("cp.async.commit_group;");
    }
    __syncthreads();
    const int c = count[0];
    cudaGridDependencySynchronize();
    const int n = tid >> 4, q = tid & 15;
    float v[8], amax = 1e-10f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = n < c ? __bfloat162float(__float2bfloat16(rawh8[hsrc[n] + q * 8 + j])) : 0.f;
        amax = fmaxf(amax, fabsf(v[j]));
    }
#pragma unroll
    for (int d = 8; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
    const float inv = __fdividef(448.f, amax);
#pragma unroll
    for (int j = 0; j < 8; ++j) sh[wg_sw(n, q * 8 + j)] = __nv_fp8_e4m3(v[j] * inv);
    if (q == 0) {
        factors[n * 2] = amax * (1.f / 448.f);
        if (n >= c) factors[n * 2 + 1] = 0.f;
    }
    wg_proxy();
    __syncthreads();
    const int col = (lane & 3) * 2, row = warp * 16 + (lane >> 2);
    const float sa = factors[col * 2], sb = factors[(col + 1) * 2];
    const float ra = factors[col * 2 + 1], rbv = factors[(col + 1) * 2 + 1];
    auto* sout = reinterpret_cast<float*>(sm + WG2Out);
#pragma unroll 1
    for (int tile = 0; tile < 8; ++tile) {
        if (tile + WG2Depth - 1 < 8)
            wg2_prefetch(sm + WG2Ring + ((tile + WG2Depth - 1) % WG2Depth) * 8192,
                         wbase + (tile + WG2Depth - 1) * 8192);
        asm volatile("cp.async.commit_group;\n cp.async.wait_group %0;" ::"n"(WG2Depth - 1));
        wg_proxy();
        __syncthreads();
        static_assert(DOT_CHUNK == 1);
        float parts[4 / DOT_CHUNK][4];
        wg_fence();
        const auto ad = wg_desc(sm + WG2Ring + (tile % WG2Depth) * 8192), bd = wg_desc(sh);
#pragma unroll
        for (int k = 0; k < 4; ++k) wg_mma_overwrite(parts[k / DOT_CHUNK], ad + k * 2, bd + k * 2);
        wg_wait();
        float acc[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[j] = parts[0][j];
#pragma unroll
            for (int p = 1; p < 4 / DOT_CHUNK; ++p) acc[j] = __fadd_rn(acc[j], parts[p][j]);
        }
        const float as = ws[tile >> 1] * sa, bs = ws[tile >> 1] * sb;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float f = acc[j] * ((j & 1) ? bs : as) * ((j & 1) ? rbv : ra);
            // Transpose to token-major, preserving the v2 expert-output rounding.
            sout[(col + (j & 1)) * 64 + row + (j >> 1) * 8] = __bfloat162float(__float2bfloat16(f));
        }
        __syncthreads();
        const int token = tid >> 4, r = (tid & 15) * 4;
        if (token < c)
            *reinterpret_cast<float4*>(l2p + off[token] + rb * 512 + tile * 64 + r) =
                *reinterpret_cast<float4*>(sout + token * 64 + r);
        __syncthreads();
    }
    asm volatile("cp.async.wait_group 0;");
}
