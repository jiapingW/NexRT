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

#define GATE_XN_ROLES 2
#ifndef GATE_XN_ROLES
#define GATE_XN_ROLES 2
#endif
static_assert(GATE_XN_ROLES == 1 || GATE_XN_ROLES == 2);
#include <cuda_bf16.h>
#include "../common/pdl.cuh"
#include <cuda_runtime.h>

#include <cstdint>

namespace nexrt::sm90::nex_n2_5_pro::M4 {
constexpr int kDim = 4096;
constexpr int kGateRowsPhysical = 528;
constexpr int kRoutedExperts = 512;
constexpr int kTopK = 10;
constexpr int kOutputWords = 64;
constexpr int kSharedSlot = 63;
constexpr int kRows = 4;

__device__ __forceinline__ float Sigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }

__device__ __forceinline__ uint32_t OrderedFloat(float x) {
    const uint32_t bits = __float_as_uint(x);
    return bits ^ ((bits >> 31) ? 0xffffffffu : 0x80000000u);
}

__device__ __forceinline__ uint32_t PackedOrderedBf16Key(float x, int id) {
    return (OrderedFloat(x) & 0xffff0000u) | (0xffffu - static_cast<uint32_t>(id));
}

__device__ __forceinline__ int ExpertIdFromPackedKey(uint32_t key) {
    return static_cast<int>(0xffffu - (key & 0xffffu));
}

__device__ __forceinline__ uint32_t ReduxMax32(uint32_t x) {
    uint32_t out;
    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(out) : "r"(x));
    return out;
}

__device__ __forceinline__ uint32_t ReduxMax16(uint32_t x) {
    uint32_t out;
    asm volatile("redux.sync.max.u32 %0, %1, 0xffff;" : "=r"(out) : "r"(x));
    return out;
}

__device__ __forceinline__ void MaterializeXn(const __nv_bfloat16* x, const float* gamma,
                                              __nv_bfloat16* xn, const float* rms_scale,
                                              int vector) {
    const float scale = rms_scale[0];
    const int base = vector * 8;
    const uint4 xv = *reinterpret_cast<const uint4*>(x + base);
    const float4 g0 = *reinterpret_cast<const float4*>(gamma + base);
    const float4 g1 = *reinterpret_cast<const float4*>(gamma + base + 4);
    const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&xv);
    const float gs[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
    __nv_bfloat16 out[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i] = __float2bfloat16(__bfloat162float(xb[i]) * scale * gs[i]);
    }
    *reinterpret_cast<uint4*>(xn + base) = *reinterpret_cast<const uint4*>(out);
}

__global__ __launch_bounds__(512) void M4GateTopKPackedKeyRawKernel(
    const __nv_bfloat16* __restrict__ logits4, const float* __restrict__ rms4,
    const __nv_bfloat16* __restrict__ x4, const float* __restrict__ gamma,
    int32_t* __restrict__ packed4, __nv_bfloat16* __restrict__ xn4) {
    NEXRT_PDL_ENTRY_SYNC();
    constexpr int kWarps = kRoutedExperts / 32;
    static_assert(kDim == 512 * 8, "xn path assumes 8 elements per thread at 512 threads");
    __shared__ float s_logit[kRoutedExperts];
    __shared__ uint32_t s_winner[kTopK][kWarps];

    const int row = blockIdx.x % kRows;
    const __nv_bfloat16* logits = logits4 + static_cast<size_t>(row) * kGateRowsPhysical;
    const float* rms_scale = rms4 + row;
    const __nv_bfloat16* x = x4 + static_cast<size_t>(row) * kDim;
    int32_t* packed = packed4 + static_cast<size_t>(row) * kOutputWords;
    __nv_bfloat16* xn = xn4 + static_cast<size_t>(row) * kDim;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

#if GATE_XN_ROLES == 2
    if (blockIdx.x >= kRows) {
        MaterializeXn(x, gamma, xn, rms_scale, tid);
        return;
    }
#endif

    if (tid < kOutputWords) {
        packed[tid] = 0;
    }

    const float logit = __bfloat162float(logits[tid]);
    s_logit[tid] = logit;
    const uint32_t key = PackedOrderedBf16Key(logit, tid);

    bool alive = true;
#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
        const uint32_t best = ReduxMax32(alive ? key : 0u);
        if (alive && key == best) {
            s_winner[rank][warp] = best;
            alive = false;
        }
    }
    __syncthreads();

    // Preserve Xn arithmetic; assign work independently of the selector warp.
#if GATE_XN_ROLES == 1
    // Warp 0 can merge immediately; 480 threads cover 512 eight-element vectors.
    if (warp != 0) {
        for (int vector = tid - 32; vector < 512; vector += 480)
            MaterializeXn(x, gamma, xn, rms_scale, vector);
    }
#endif

    if (warp == 0 && lane < kWarps) {
        int pos = 0;
        uint32_t head = s_winner[0][lane];
        int my_id = -1;

#pragma unroll
        for (int rank = 0; rank < kTopK; ++rank) {
            const uint32_t best = ReduxMax16(head);
            if (lane == rank) {
                my_id = ExpertIdFromPackedKey(best);
            }
            if (rank != kTopK - 1 && head == best) {
                ++pos;
                head = s_winner[pos][lane];
            }
        }

        const float sel_scale = rms_scale[0];
        const float selected =
            (lane < kTopK) ? s_logit[my_id] * sel_scale : -3.4028234663852886e38f;
        const float max_logit = __shfl_sync(0xffffu, selected, 0, 16);
        float weight = (lane < kTopK) ? __expf(selected - max_logit) : 0.0f;
        float sum = weight;
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffu, sum, offset, 16);
        }

        if (lane < kTopK) {
            packed[lane] = my_id;
            packed[kTopK + lane] = __float_as_int(weight / sum);
        }
        if (lane == 0) {
            const float shared_raw = __bfloat162float(logits[kRoutedExperts]);
            packed[kSharedSlot] = __float_as_int(Sigmoid(shared_raw * sel_scale));
        }
    }
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M4
using namespace nexrt::sm90::nex_n2_5_pro::M4;
extern "C" {
int NexN25ProM4GateTopKPackedKeyRawLaunch(const void* logits4, const void* rms4, const void* x4,
                                          const void* gamma, void* packed4, void* xn4,
                                          void* stream) {
    M4GateTopKPackedKeyRawKernel<<<kRows*(GATE_XN_ROLES == 2 ? 2 : 1), 512, 0,
                                   reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits4), reinterpret_cast<const float*>(rms4),
        reinterpret_cast<const __nv_bfloat16*>(x4), reinterpret_cast<const float*>(gamma),
        reinterpret_cast<int32_t*>(packed4), reinterpret_cast<__nv_bfloat16*>(xn4));
    return static_cast<int>(cudaGetLastError());
}
}
extern "C" int NexN25ProM4GateTopKPackedKeyRawPdlLaunch(const void* logits4, const void* rms4,
                                                        const void* x4, const void* gamma,
                                                        void* packed4, void* xn4, void* stream) {
    return static_cast<int>(NexrtPdlLaunchKernel(
        dim3(kRows * (GATE_XN_ROLES == 2 ? 2 : 1)), dim3(512), 0,
        reinterpret_cast<cudaStream_t>(stream), M4GateTopKPackedKeyRawKernel,
        reinterpret_cast<const __nv_bfloat16*>(logits4), reinterpret_cast<const float*>(rms4),
        reinterpret_cast<const __nv_bfloat16*>(x4), reinterpret_cast<const float*>(gamma),
        reinterpret_cast<int32_t*>(packed4), reinterpret_cast<__nv_bfloat16*>(xn4)));
}

extern "C" int GateRolesAttributes(int* out) {
    cudaFuncAttributes a;
    auto e = cudaFuncGetAttributes(&a, M4GateTopKPackedKeyRawKernel);
    if (e != cudaSuccess) return int(e);
    out[0] = a.numRegs;
    out[1] = a.sharedSizeBytes;
    out[2] = a.localSizeBytes;
    return 0;
}

extern "C" int NexN25ProM4GateTopKLaunch(const void* logits, const void* rms, const void* x,
                                         const void* gamma, void* packed, void* xn, void* stream) {
    return NexN25ProM4GateTopKPackedKeyRawPdlLaunch(logits, rms, x, gamma, packed, xn, stream);
}
