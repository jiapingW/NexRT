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
constexpr int WG1Depth = L1_DEPTH;
constexpr int WG1Half = L1_VALID / 2;
constexpr int WG1Ahead = WG1Depth - (L1_MODE == 1 ? L1_BATCH : (L1_MODE == 2 ? L1_PIPE : 1));
static_assert(WG1Ahead > 0);
__device__ __forceinline__ int out_offset(int n, int r) {
    return n * 64 + (L1_XOR ? (r ^ ((n >> 1) * 8)) : r);
}
__device__ __forceinline__ void operand_fence(float (&c)[4]) {
    asm volatile("" : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])::"memory");
}
constexpr int WG1Ring = 32768;
constexpr int WG1Meta = WG1Ring + WG1Depth * 8192;
constexpr int WG1Out = L1_COMPACT ? WG1Ring : WG1Meta + 2048;
constexpr int WG1Dyn = WG1Meta + (L1_COMPACT ? 2048 : 4096);
__device__ __forceinline__ void wg1_prefetch(char* stage, const CUtensorMap* map,
                                             unsigned long long* bars, int expert, int group,
                                             int cl) {
    if (threadIdx.x == 0) {
        unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(stage));
        unsigned bar = static_cast<unsigned>(__cvta_generic_to_shared(bars + cl % WG1Depth));
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 8192;" ::"r"(bar)
                     : "memory");
        const int k = cl * 128, g = expert * 256 + group * 32, u = g + 128;
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];" ::
                "r"(dst),
            "l"(map), "r"(k), "r"(g), "r"(bar)
            : "memory");
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];" ::
                "r"(dst + 4096),
            "l"(map), "r"(k), "r"(u), "r"(bar)
            : "memory");
    }
}
__device__ __forceinline__ void wait_weight(unsigned long long* bars, int cl) {
    if (!L1_LEADER || threadIdx.x == 0) {
        unsigned bar = static_cast<unsigned>(__cvta_generic_to_shared(bars + cl % WG1Depth));
        unsigned phase = (cl / WG1Depth) & 1;
        asm volatile(
            "{ .reg .pred ready; wait_tma: mbarrier.try_wait.parity.shared::cta.b64 ready, [%0], %1; @!ready bra wait_tma; }" ::
                "r"(bar),
            "r"(phase)
            : "memory");
    }
}
__device__ __forceinline__ void wg1_body(const M8RoutePlan* plan, const int32_t* packed8,
                                         const __nv_fp8_e4m3* w13_raw, const float* w13s,
                                         float* rawh8, const __nv_bfloat16* xq8, const float* xs8,
                                         const CUtensorMap* w_map) {
    const int desc = blockIdx.x / L1_GROUPS, group = blockIdx.x % L1_GROUPS;
    if (desc >= plan->n_desc) return;
    const int e = plan->expert_id[desc];
    extern __shared__ __align__(1024) char sm[];
    int* tok = reinterpret_cast<int*>(sm + WG1Meta);
    int* hoff = tok + 8;
    int* count = hoff + 8;
    float* ws = reinterpret_cast<float*>(sm + WG1Meta + 128);
    float* xs = ws + 64;
    auto* bars = reinterpret_cast<unsigned long long*>(sm + WG1Meta + 1536);
    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < WG1Depth; ++i) {
            unsigned bar = static_cast<unsigned>(__cvta_generic_to_shared(bars + i));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar) : "memory");
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
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
            const int n = __popc(mask & ((1u << lane) - 1));
            tok[n] = lane;
            hoff[n] = lane * 1408 + slot * 128;
        }
        if (lane == 0) count[0] = __popc(mask);
    }
    if (tid < 64) ws[tid] = w13s[e * 64 + tid];
    __syncthreads();
    const int c = count[0];
    const auto* q = reinterpret_cast<const __nv_fp8_e4m3*>(xq8);
#pragma unroll 1
    for (int i = tid; i < 2048; i += 128) {
        int cl = i >> 6, n = (i >> 3) & 7, k = (i & 7) * 16;
        if (n < c)
            wg_copy(sm + cl * 1024 + wg_sw(n, k), q + tok[n] * 4096 + cl * 128 + k);
        else
            *reinterpret_cast<uint4*>(sm + cl * 1024 + wg_sw(n, k)) = make_uint4(0, 0, 0, 0);
    }
    for (int i = tid; i < 256; i += 128) {
        int n = i >> 5, cl = i & 31;
        xs[n * 33 + cl] = n < c ? xs8[tok[n] * 32 + cl] : 0.f;
    }
    const auto* wbase = w13_raw + static_cast<size_t>(e) * 256 * 4096;
    asm volatile("cp.async.commit_group;");
    if constexpr (!L1_EARLY) {
        asm volatile("cp.async.wait_group 0;");
        wg_proxy();
        __syncthreads();
    }
#pragma unroll
    for (int st = 0; st < WG1Ahead; ++st) {
        wg1_prefetch(sm + WG1Ring + st * 8192, w_map, bars, e, group, st);
    }
    if constexpr (L1_EARLY) {
        asm volatile("cp.async.wait_group %0;" ::"n"(WG1Ahead));
        wg_proxy();
        __syncthreads();
    }
    const int col = (lane & 3) * 2, row = warp * 16 + (lane >> 2);
    float total[4] = {0, 0, 0, 0};
    float acc[32][4];
    // Preserve the original per-K128 accumulators; buffer only p1/p2/p3.
    float tmp[L1_PIPE][3][4];
#pragma unroll
    for (int cl = 0; cl < 24; ++cl) {
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[cl][j] = 0.f;
        operand_fence(acc[cl]);
    }
    wg_fence();
#pragma unroll
    for (int cl = 0; cl < 32; ++cl) {
        if (cl + WG1Ahead < 32)
            wg1_prefetch(sm + WG1Ring + ((cl + WG1Ahead) % WG1Depth) * 8192, w_map, bars, e, group,
                         cl + WG1Ahead);
        wait_weight(bars, cl);
        wg_proxy();
        __syncthreads();
        if (cl == L1_TRIGGER) cudaTriggerProgrammaticLaunchCompletion();
        wg_fence();  // Explicit uniform register fence after producer/wait divergence.
        const auto ad = wg_desc(sm + WG1Ring + (cl % WG1Depth) * 8192),
                   bd = wg_desc(sm + cl * 1024);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            if (k == 0)
                wg_mma(acc[cl], ad, bd);
            else
                wg_mma_overwrite(tmp[cl % L1_PIPE][k - 1], ad + k * 2, bd + k * 2);
        }
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        if (cl >= L1_PIPE - 1) {
            asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(L1_PIPE - 1) : "memory");
            const int kcl = cl - L1_PIPE + 1;
            const float s = ws[((warp * 16 / WG1Half) & 1) * 32 + kcl];
            const float sa = s * xs[col * 33 + kcl], sb = s * xs[(col + 1) * 33 + kcl];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc[kcl][j] = __fadd_rn(acc[kcl][j], tmp[kcl % L1_PIPE][0][j]);
                acc[kcl][j] = __fadd_rn(acc[kcl][j], tmp[kcl % L1_PIPE][1][j]);
                acc[kcl][j] = __fadd_rn(acc[kcl][j], tmp[kcl % L1_PIPE][2][j]);
                total[j] += acc[kcl][j] * ((j & 1) ? sb : sa);
            }
        }
        // Produce the future p0 after consuming the previous group.
        // The next existing per-iteration WGMMA fence covers this initialization.
        if (cl + 24 < 32) {
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[cl + 24][j] = 0.f;
            operand_fence(acc[cl + 24]);
        }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
    for (int kcl = 32 - L1_PIPE + 1; kcl < 32; ++kcl) {
        const float s = ws[((warp * 16 / WG1Half) & 1) * 32 + kcl];
        const float sa = s * xs[col * 33 + kcl], sb = s * xs[(col + 1) * 33 + kcl];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[kcl][j] = __fadd_rn(acc[kcl][j], tmp[kcl % L1_PIPE][0][j]);
            acc[kcl][j] = __fadd_rn(acc[kcl][j], tmp[kcl % L1_PIPE][1][j]);
            acc[kcl][j] = __fadd_rn(acc[kcl][j], tmp[kcl % L1_PIPE][2][j]);
            total[j] += acc[kcl][j] * ((j & 1) ? sb : sa);
        }
    }

    asm volatile("cp.async.wait_group 0;");
    auto* sout = reinterpret_cast<float*>(sm + WG1Out);
#pragma unroll
    for (int j = 0; j < 4; ++j) sout[out_offset(col + (j & 1), row + (j >> 1) * 8)] = total[j];
    __syncthreads();
    for (int i = tid; i < 8 * WG1Half; i += 128) {
        const int n = i / WG1Half, r = i % WG1Half;
        if (n < c) {
            const float g = __bfloat162float(__float2bfloat16(sout[out_offset(n, r)]));
            const float u = __bfloat162float(__float2bfloat16(sout[out_offset(n, r + WG1Half)]));
            rawh8[hoff[n] + group * WG1Half + r] =
                __bfloat162float(__float2bfloat16(wm_silu(g) * u));
        }
    }
}
