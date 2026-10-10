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

// DFlash M=8 attention front for TP8 rank-local weights.
//
// Projection (two physical launches with independent input ownership):
//   qkv_proj:        h8_raw   [8,4096] x folded [768,4096] -> 96 CTAs
//   context_kv_proj: ctx8_raw [8,4096] x folded [256,4096] -> 32 CTAs
//
// The pack is FT16 lane-major BF16: every m16n8k16 A fragment is a real
// output row.  Each 16-row tile has two K=2048 CTAs.  CTA 0 of each launch
// computes its own eight input RMS inverse scales.  The context buffer is fixed
// M=8 for CUDA Graph capture; df_meta.p controls which leading context rows are
// published, while rows >= p remain private scratch.  The two FP32 K halves are
// folded and BF16-rounded once by the row-owned post kernel.
//
// Post (8 CTAs x 8 warps): each CTA owns one token/context row; its warps own
// Q0..Q3, draft K, context K, draft V, context V.  Q/K/CK apply the input
// scalar before head RMSNorm and RoPE; V/CV apply it before cache publication.
//
// Absolute cache contract for df_meta={p,start}:
//   context row c < p -> start-p+c
//   draft row   j < 8 -> start+j
// RoPE always uses those absolute positions.  The linear launcher publishes
// to a linear cache; the ring launcher translates only the K/V destination to
// `position&8191`.  Invalid metadata latches status bit 0
// and publishes no Q/K/V output.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY DF
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace nexrt::sm90::nex_n2_5_pro::DFlash {
constexpr int kDfHid = 4096;
constexpr int kDfM = 8;
constexpr int kDfHd = 128;
constexpr int kDfQHeads = 4;
constexpr int kDfQRows = kDfQHeads * kDfHd;
constexpr int kDfQkvRows = kDfQRows + 2 * kDfHd;
constexpr int kDfCkvRows = 2 * kDfHd;
constexpr int kDfRot = kDfHd / 2;

__device__ __forceinline__ float df_front_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ void df_front_mma_m16n8k16_bf16(float (&c)[4],
                                                           const unsigned int (&a)[4],
                                                           const unsigned int (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int kWarpK>
__device__ __forceinline__ void df_front_stage_x_issue(const __nv_bfloat16* __restrict__ x8,
                                                       char* __restrict__ shared_warp, int lane,
                                                       int k0) {
    constexpr int kPitch = kWarpK * 2 + 16;
    constexpr int kBytes = kWarpK * 2;
#pragma unroll
    for (int row = 0; row < kDfM; ++row) {
        for (int byte = lane * 16; byte < kBytes; byte += 32 * 16) {
            const unsigned int dst = static_cast<unsigned int>(
                __cvta_generic_to_shared(shared_warp + row * kPitch + byte));
            const void* src = x8 + static_cast<size_t>(row) * kDfHid + k0 + byte / 2;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
        }
    }
    asm volatile("cp.async.commit_group;");
}

constexpr int kDfProjSplits = 2;
constexpr int kDfProjTileRows = 16;
constexpr int kDfProjWarpK = kDfHid / (kDfProjSplits * 8);
constexpr int kDfProjSteps = kDfProjWarpK / 16;
constexpr int kDfProjPitch = kDfProjWarpK * 2 + 16;

struct __align__(16) DfFrontProjectionShared {
    char x[8][kDfM][kDfProjPitch];
    float partial[8][32][4];
};

template <int kOutputRows>
__device__ __forceinline__ void df_m8_folded_projection_body(
    const __nv_bfloat16* __restrict__ x8, const __nv_bfloat16* __restrict__ wpack,
    float* __restrict__ partial_out, float* __restrict__ scale8, float eps,
    DfFrontProjectionShared& shared) {
    constexpr int kSplits = 2;
    constexpr int kTileRows = 16;
    constexpr int kWarpK = kDfHid / (kSplits * 8);  // 256
    constexpr int kSteps = kWarpK / 16;             // 16
    constexpr int kPitch = kWarpK * 2 + 16;
    static_assert(kOutputRows % kTileRows == 0);
    static_assert(kSplits == kDfProjSplits && kTileRows == kDfProjTileRows);
    static_assert(kWarpK == kDfProjWarpK && kSteps == kDfProjSteps);
    static_assert(kPitch == kDfProjPitch);

    const int tile = blockIdx.x >> 1;
    const int k_split = blockIdx.x & 1;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    const int k0 = k_split * (kDfHid / kSplits) + warp * kWarpK;
    char* const shared_warp = &shared.x[warp][0][0];

    df_front_stage_x_issue<kWarpK>(x8, shared_warp, lane, k0);
    const uint4* packed = reinterpret_cast<const uint4*>(wpack) +
                          (static_cast<size_t>(tile) * (kDfHid / 16) + (k0 >> 4)) * 32 + lane;
    uint4 weight_pipe[kSteps];
#pragma unroll
    for (int step = 0; step < kSteps; ++step) {
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(weight_pipe[step].x), "=r"(weight_pipe[step].y),
                       "=r"(weight_pipe[step].z), "=r"(weight_pipe[step].w)
                     : "l"(packed + static_cast<size_t>(step) * 32));
    }
    asm volatile("cp.async.wait_group 0;");
    __syncwarp();

    float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int step = 0; step < kSteps; ++step) {
        const uint4 weight = weight_pipe[step];
        const char* x_ptr = shared_warp + group * kPitch + step * 32 + thread_in_group * 4;
        unsigned int b[2];
        b[0] = *reinterpret_cast<const unsigned int*>(x_ptr);
        b[1] = *reinterpret_cast<const unsigned int*>(x_ptr + 16);
        const unsigned int a[4] = {weight.x, weight.y, weight.z, weight.w};
        df_front_mma_m16n8k16_bf16(accum, a, b);
    }

#pragma unroll
    for (int item = 0; item < 4; ++item) {
        shared.partial[warp][lane][item] = accum[item];
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float value[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int k_warp = 0; k_warp < 8; ++k_warp) {
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                value[item] += shared.partial[k_warp][lane][item];
            }
        }
        const int row_lo = tile * kTileRows + group;
        const int token0 = thread_in_group * 2;
        partial_out[(static_cast<size_t>(k_split) * kDfM + token0) * kOutputRows + row_lo] =
            value[0];
        partial_out[(static_cast<size_t>(k_split) * kDfM + token0 + 1) * kOutputRows + row_lo] =
            value[1];
        partial_out[(static_cast<size_t>(k_split) * kDfM + token0) * kOutputRows + row_lo + 8] =
            value[2];
        partial_out[(static_cast<size_t>(k_split) * kDfM + token0 + 1) * kOutputRows + row_lo + 8] =
            value[3];
    }

    // Only CTA 0 of this projection launch performs its input side reduction.
    if (blockIdx.x == 0) {
        const __nv_bfloat162* row =
            reinterpret_cast<const __nv_bfloat162*>(x8) + static_cast<size_t>(warp) * (kDfHid / 2);
        float square_sum = 0.0f;
        for (int pair = lane; pair < kDfHid / 2; pair += 32) {
            const float2 value = __bfloat1622float2(row[pair]);
            square_sum += value.x * value.x + value.y * value.y;
        }
        square_sum = df_front_warp_sum(square_sum);
        if (lane == 0) {
            scale8[warp] = rsqrtf(square_sum * (1.0f / static_cast<float>(kDfHid)) + eps);
        }
    }
}

__global__ __launch_bounds__(256) void DfM8FoldedQkvFt16Split2ProjKernel(
    const __nv_bfloat16* __restrict__ h8, const __nv_bfloat16* __restrict__ qkv_wpack,
    float* __restrict__ qkv_partial, float* __restrict__ attn_scale8, float attn_eps) {
    __shared__ DfFrontProjectionShared shared;
    df_m8_folded_projection_body<kDfQkvRows>(h8, qkv_wpack, qkv_partial, attn_scale8, attn_eps,
                                             shared);
}

__global__ __launch_bounds__(256) void DfM8FoldedContextKvFt16Split2ProjKernel(
    const __nv_bfloat16* __restrict__ ctx_raw8, const __nv_bfloat16* __restrict__ context_kv_wpack,
    float* __restrict__ context_kv_partial, float* __restrict__ hidden_scale8, float hidden_eps) {
    __shared__ DfFrontProjectionShared shared;
    df_m8_folded_projection_body<kDfCkvRows>(ctx_raw8, context_kv_wpack, context_kv_partial,
                                             hidden_scale8, hidden_eps, shared);
}

enum class DfPostWork : int {
    kQuery = 0,
    kDraftKey = 1,
    kContextKey = 2,
    kDraftValue = 3,
    kContextValue = 4,
};

__global__ __launch_bounds__(256) void DfQkvCkvPostRowsSplit2Kernel(
    const float* __restrict__ qkv_partial, const float* __restrict__ ckv_partial,
    const float* __restrict__ attn_scale8, const float* __restrict__ hidden_scale8,
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight, const float* __restrict__ cos_table,
    const float* __restrict__ sin_table, __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_cache, __nv_bfloat16* __restrict__ v_cache,
    int* __restrict__ status, const int* __restrict__ df_meta, int position_capacity,
    int ring_capacity, float qk_eps) {
    const int p = df_meta[0];
    const int start = df_meta[1];
    const bool ring_valid = ring_capacity == 0 || ring_capacity == 8192;
    const bool valid = p >= 0 && p <= kDfM && start >= p && ring_valid &&
                       position_capacity >= kDfM && start <= position_capacity - kDfM;
    if (!valid) {
        if (threadIdx.x == 0 && status != nullptr) {
            atomicOr(status, 1);
        }
        return;
    }

    const int row_group = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int work_index = warp < 4    ? row_group * 4 + warp
                           : warp == 4 ? 32 + row_group
                           : warp == 5 ? 40 + row_group
                           : warp == 6 ? 48 + row_group
                                       : 56 + row_group;
    int row = 0;
    int head = 0;
    int position = 0;
    DfPostWork work;
    if (work_index < 32) {
        work = DfPostWork::kQuery;
        row = work_index >> 2;
        head = work_index & 3;
        position = start + row;
    } else if (work_index < 40) {
        work = DfPostWork::kDraftKey;
        row = work_index - 32;
        position = start + row;
    } else if (work_index < 48) {
        work = DfPostWork::kContextKey;
        row = work_index - 40;
        if (row >= p) {
            return;
        }
        position = start - p + row;
    } else if (work_index < 56) {
        work = DfPostWork::kDraftValue;
        row = work_index - 48;
        position = start + row;
    } else {
        work = DfPostWork::kContextValue;
        row = work_index - 56;
        if (row >= p) {
            return;
        }
        position = start - p + row;
    }

    const bool query = work == DfPostWork::kQuery;
    const bool context = work == DfPostWork::kContextKey || work == DfPostWork::kContextValue;
    const bool value_work = work == DfPostWork::kDraftValue || work == DfPostWork::kContextValue;
    const int cache_position = ring_capacity == 8192 ? (position & 8191) : position;
    const int output_rows = context ? kDfCkvRows : kDfQkvRows;
    const int base =
        query ? head * kDfHd
              : (context ? (value_work ? kDfHd : 0) : kDfQRows + (value_work ? kDfHd : 0));
    const float* partial = context ? ckv_partial : qkv_partial;
    const float input_scale = context ? hidden_scale8[row] : attn_scale8[row];
    float output[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int output_d = base + lane + item * 32;
        float total = partial[static_cast<size_t>(row) * output_rows + output_d] +
                      partial[(static_cast<size_t>(kDfM) + row) * output_rows + output_d];
        // Projection visibility is BF16 before the input row scalar is applied.
        output[item] = __bfloat162float(__float2bfloat16(total)) * input_scale;
    }

    if (!value_work) {
        float square_sum = 0.0f;
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            square_sum += output[item] * output[item];
        }
        square_sum = df_front_warp_sum(square_sum);
        const float head_rinv = rsqrtf(square_sum * (1.0f / static_cast<float>(kDfHd)) + qk_eps);
        const __nv_bfloat16* norm_weight = query ? q_norm_weight : k_norm_weight;
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int d = lane + item * 32;
            output[item] *= head_rinv * __bfloat162float(norm_weight[d]);
        }
        const float* cos_row = cos_table + static_cast<size_t>(position) * kDfRot;
        const float* sin_row = sin_table + static_cast<size_t>(position) * kDfRot;
        float rotated[4];
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int d = lane + item * 32;
            const int pair = d < kDfRot ? d : d - kDfRot;
            const int partner_item = d < kDfRot ? item + 2 : item - 2;
            const float cos_value = __bfloat162float(__float2bfloat16(cos_row[pair]));
            const float sin_value = __bfloat162float(__float2bfloat16(sin_row[pair]));
            rotated[item] = output[item] * cos_value +
                            (d < kDfRot ? -output[partner_item] : output[partner_item]) * sin_value;
        }
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            output[item] = rotated[item];
        }
    }

#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int d = lane + item * 32;
        const __nv_bfloat16 result = __float2bfloat16(output[item]);
        if (query) {
            q_out[static_cast<size_t>(row) * kDfQRows + head * kDfHd + d] = result;
        } else if (value_work) {
            v_cache[static_cast<size_t>(cache_position) * kDfHd + d] = result;
        } else {
            k_cache[static_cast<size_t>(cache_position) * kDfHd + d] = result;
        }
    }
}

}  // namespace nexrt::sm90::nex_n2_5_pro::DFlash

using namespace nexrt::sm90::nex_n2_5_pro::DFlash;

extern "C" {

int NexN25ProDFlashQkvProjLaunch(const void* h8, const void* qkv_wpack, void* qkv_partial,
                                 float* attn_scale8, float attn_eps, void* stream) {
    constexpr int kQkvCtas = (kDfQkvRows / kDfProjTileRows) * kDfProjSplits;
    static_assert(kQkvCtas == 96);
    DfM8FoldedQkvFt16Split2ProjKernel<<<dim3(kQkvCtas), dim3(256), 0,
                                        static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(h8), static_cast<const __nv_bfloat16*>(qkv_wpack),
        static_cast<float*>(qkv_partial), attn_scale8, attn_eps);
    return static_cast<int>(cudaGetLastError());
}

int NexN25ProDFlashContextKvProjLaunch(const void* ctx_raw8, const void* context_kv_wpack,
                                       void* context_kv_partial, float* hidden_scale8,
                                       float hidden_eps, void* stream) {
    constexpr int kContextKvCtas = (kDfCkvRows / kDfProjTileRows) * kDfProjSplits;
    static_assert(kContextKvCtas == 32);
    DfM8FoldedContextKvFt16Split2ProjKernel<<<dim3(kContextKvCtas), dim3(256), 0,
                                              static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(ctx_raw8),
        static_cast<const __nv_bfloat16*>(context_kv_wpack),
        static_cast<float*>(context_kv_partial), hidden_scale8, hidden_eps);
    return static_cast<int>(cudaGetLastError());
}

int NexN25ProDFlashQkvCkvPostRowsCtlLaunch(const void* qkv_partial, const void* ckv_partial,
                                           const float* attn_scale8, const float* hidden_scale8,
                                           const void* q_norm_weight, const void* k_norm_weight,
                                           const void* cos_table, const void* sin_table,
                                           void* q_out, void* k_cache, void* v_cache, int* status,
                                           const void* df_meta, int cache_capacity, float qk_eps,
                                           void* stream) {
    DfQkvCkvPostRowsSplit2Kernel<<<dim3(kDfM), dim3(256), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(qkv_partial), static_cast<const float*>(ckv_partial), attn_scale8,
        hidden_scale8, static_cast<const __nv_bfloat16*>(q_norm_weight),
        static_cast<const __nv_bfloat16*>(k_norm_weight), static_cast<const float*>(cos_table),
        static_cast<const float*>(sin_table), static_cast<__nv_bfloat16*>(q_out),
        static_cast<__nv_bfloat16*>(k_cache), static_cast<__nv_bfloat16*>(v_cache), status,
        static_cast<const int*>(df_meta), cache_capacity, 0, qk_eps);
    return static_cast<int>(cudaGetLastError());
}

int NexN25ProDFlashQkvCkvPostRowsCtlRingLaunch(const void* qkv_partial, const void* ckv_partial,
                                               const float* attn_scale8, const float* hidden_scale8,
                                               const void* q_norm_weight, const void* k_norm_weight,
                                               const void* cos_table, const void* sin_table,
                                               void* q_out, void* k_cache, void* v_cache,
                                               int* status, const void* df_meta,
                                               int position_capacity, int ring_capacity,
                                               float qk_eps, void* stream) {
    DfQkvCkvPostRowsSplit2Kernel<<<dim3(kDfM), dim3(256), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(qkv_partial), static_cast<const float*>(ckv_partial), attn_scale8,
        hidden_scale8, static_cast<const __nv_bfloat16*>(q_norm_weight),
        static_cast<const __nv_bfloat16*>(k_norm_weight), static_cast<const float*>(cos_table),
        static_cast<const float*>(sin_table), static_cast<__nv_bfloat16*>(q_out),
        static_cast<__nv_bfloat16*>(k_cache), static_cast<__nv_bfloat16*>(v_cache), status,
        static_cast<const int*>(df_meta), position_capacity, ring_capacity, qk_eps);
    return static_cast<int>(cudaGetLastError());
}

int NexN25ProDFlashQkvProjRows() { return kDfQkvRows; }
int NexN25ProDFlashContextKvProjRows() { return kDfCkvRows; }

}  // extern "C"
