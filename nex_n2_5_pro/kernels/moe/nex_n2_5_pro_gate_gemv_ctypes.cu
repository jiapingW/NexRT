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

// MoE router GEMV and top-k kernels selected for the nex_n2_5_pro TP=8 backend.
#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY bs1
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY {
constexpr int kDim = 4096;
constexpr int kGateRowsActive = 513;
constexpr int kRoutedExperts = 512;
constexpr int kTopK = 10;
constexpr int kDimPairs = kDim / 2;

__device__ __forceinline__ float WarpSum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

template <int kNumThreads>
__device__ __forceinline__ float BlockSum(float v) {
    constexpr int kNumWarps = kNumThreads / 32;
    __shared__ float warp_sums[kNumWarps];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = WarpSum(v);
    if (lane == 0) {
        warp_sums[warp] = v;
    }
    __syncthreads();

    v = (threadIdx.x < kNumWarps) ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        v = WarpSum(v);
    }
    return v;
}

__device__ __forceinline__ float Sigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }

__device__ __forceinline__ uint32_t OrderedFloat(float x) {
    const uint32_t bits = __float_as_uint(x);
    return bits ^ ((bits >> 31) ? 0xffffffffu : 0x80000000u);
}

// Every routed score is loaded from BF16, so the low 16 bits of its exact FP32
// conversion are zero. Reuse those bits for the inverse expert id: one unsigned
// max then selects the larger score and, on a tie, the lower expert id -- no
// ballot/ffs owner election or separate id shuffle in any selection round.
// (Bitwise-verified over all 65536 BF16 score patterns.)
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

template <int kNumThreads>
__global__ __launch_bounds__(kNumThreads) void GateGemvRowRawAndRmsCta0Kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ gate_w_prefused,
    __nv_bfloat16* __restrict__ logits, float* __restrict__ rms_scale, float eps,
    const char* __restrict__ pf_base,  // optional: L2-warm after the row is stored
    long long pf_bytes) {
    const int row = blockIdx.x;
    if (row >= kGateRowsActive) {
        return;
    }

    const auto* __restrict__ x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    const auto* __restrict__ w2 =
        reinterpret_cast<const __nv_bfloat162*>(gate_w_prefused + static_cast<size_t>(row) * kDim);

    float acc = 0.0f;
    float rms_acc = 0.0f;
    for (int pair = threadIdx.x; pair < kDimPairs; pair += kNumThreads) {
        const float2 xf = __bfloat1622float2(x2[pair]);
        const float2 wf = __bfloat1622float2(w2[pair]);
        acc += xf.x * wf.x + xf.y * wf.y;
        if (row == 0) {
            rms_acc += xf.x * xf.x + xf.y * xf.y;
        }
    }

    const float sum = BlockSum<kNumThreads>(acc);
    if (threadIdx.x == 0) {
        // Raw prefused dot. TopK applies the positive rms_scale for softmax
        // weights and shared sigmoid; selected expert IDs are scale-invariant.
        logits[row] = __float2bfloat16(sum);
    }

    if (row == 0) {
        const float sq_sum = BlockSum<kNumThreads>(rms_acc);
        if (threadIdx.x == 0) {
            rms_scale[0] = rsqrtf(sq_sum * (1.0f / static_cast<float>(kDim)) + eps);
        }
    }

    // Shared-expert warm: after its row is stored, each drained CTA issues
    // per-thread L2 hints over the shared expert's L1 gate|up weights (the only
    // L1 payload whose address does not depend on the top-k selection, so no
    // handoff is needed). Fire-and-forget -- the kernel never waits on the
    // fills; the top-k kernel plus two launch gaps shadow them before the L1
    // kernel consumes the range. 513 CTAs give the hints issue parallelism a
    // single latency-island kernel cannot (one SM issues ~1 L2 request/cycle).
    if (pf_bytes > 0) {
        const long long lines = pf_bytes >> 7;
        const long long gtid = static_cast<long long>(row) * kNumThreads + threadIdx.x;
        constexpr long long kStride = static_cast<long long>(kGateRowsActive) * kNumThreads;
        for (long long line = gtid; line < lines; line += kStride) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(pf_base + (line << 7)));
        }
    }
}

#ifndef NEXRT_MTP_RUNTIME_ONLY
template <int kOutputWords, int kSharedSlot>
__global__ __launch_bounds__(512) void GateTopKSingleCtaNoLogitRmsScaleKernel(
    const __nv_bfloat16* __restrict__ logits, const float* __restrict__ rms_scale,
    int32_t* __restrict__ packed) {
    constexpr int kWarps = kRoutedExperts / 32;
    __shared__ float s_logit[kRoutedExperts];
    __shared__ uint32_t s_key[kTopK][kWarps];
    __shared__ int s_val[kTopK][kWarps];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    if (tid < kOutputWords) {
        packed[tid] = 0;
    }

    const float logit = __bfloat162float(logits[tid]);
    s_logit[tid] = logit;
    const uint32_t okey = OrderedFloat(logit);

    bool alive = true;
#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
        const uint32_t best = ReduxMax32(alive ? okey : 0u);
        const uint32_t owners = __ballot_sync(0xffffffffu, alive && okey == best);
        const int owner = __ffs(owners) - 1;
        if (lane == owner) {
            s_key[rank][warp] = okey;
            s_val[rank][warp] = tid;
            alive = false;
        }
    }
    __syncthreads();

    if (warp == 0 && lane < 16) {
        int pos = 0;
        uint32_t head = (lane < kWarps) ? s_key[0][lane] : 0u;
        int hval = (lane < kWarps) ? s_val[0][lane] : -1;
        int my_id = -1;

#pragma unroll
        for (int rank = 0; rank < kTopK; ++rank) {
            const uint32_t best = ReduxMax16(head);
            const uint32_t owners = __ballot_sync(0xffffu, head == best);
            const int owner = __ffs(owners) - 1;
            const int winner = __shfl_sync(0xffffu, hval, owner, 16);
            if (lane == rank) {
                my_id = winner;
            }
            if (rank != kTopK - 1 && lane == owner && lane < kWarps) {
                ++pos;
                head = s_key[pos][lane];
                hval = s_val[pos][lane];
            }
        }

        const float scale = rms_scale[0];
        const float selected = (lane < kTopK) ? s_logit[my_id] * scale : -3.4028234663852886e38f;
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
            packed[kSharedSlot] = __float_as_int(Sigmoid(shared_raw * scale));
        }
    }
}
#endif

// Xn variant for the RAW-weight (sglang-shared-layout) L1 path: identical top-k / route-weight /
// shared-gate math to the kernel above, plus it materializes xn = x * rms_scale * gamma into a
// [kDim] bf16 buffer, where gamma = (1 + post_attention_layernorm) fp32. The xn work runs on ALL
// 16 warps right after the phase-1 sync -- warps 1..15 are otherwise idle while warp 0 finalizes
// -- so it is hidden inside this kernel's existing latency. The downstream raw-weight L1 kernel
// reads xn with UNFOLDED checkpoint weights and applies no rms scaling of its own.
// Selection uses the packed 32-bit key (score | inverse id): every argmax round is ONE redux with
// no owner election, and ties select the lower expert id -- bitwise-identical selections and
// route weights to the ballot/ffs form it replaced.
template <int kOutputWords, int kSharedSlot>
__global__ __launch_bounds__(512) void GateTopKSingleCtaNoLogitRmsScaleXnKernel(
    const __nv_bfloat16* __restrict__ logits, const float* __restrict__ rms_scale,
    const __nv_bfloat16* __restrict__ x, const float* __restrict__ gamma,
    int32_t* __restrict__ packed, __nv_bfloat16* __restrict__ xn) {
    constexpr int kWarps = kRoutedExperts / 32;
    static_assert(kDim == 512 * 8, "xn path assumes 8 elements per thread at 512 threads");
    __shared__ float s_logit[kRoutedExperts];
    __shared__ uint32_t s_winner[kTopK][kWarps];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

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

    {  // xn = x * rms * gamma, vectorized 8/thread (16B x, 32B gamma, 16B out per thread)
        const float scale = rms_scale[0];
        const int base = tid * 8;
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

        const float scale = rms_scale[0];
        const float selected = (lane < kTopK) ? s_logit[my_id] * scale : -3.4028234663852886e38f;
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
            packed[kSharedSlot] = __float_as_int(Sigmoid(shared_raw * scale));
        }
    }
}

template <int kNumThreads>
int LaunchRowRawAndRmsCta0(const void* x, const void* gate_w_prefused, void* logits,
                           float* rms_scale, float eps, const void* pf_base, long long pf_bytes,
                           void* stream) {
    GateGemvRowRawAndRmsCta0Kernel<kNumThreads>
        <<<kGateRowsActive, kNumThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<const __nv_bfloat16*>(gate_w_prefused),
            reinterpret_cast<__nv_bfloat16*>(logits), rms_scale, eps,
            reinterpret_cast<const char*>(pf_base), pf_bytes);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
using namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY;
// pf_base/pf_bytes (nullable): weight range each drained CTA L2-warms after storing
// its row -- the raw path passes the shared expert's stacked L1 gate|up weights.
extern "C" int NexN25ProGateGemvLaunch(const void* x, const void* gate_w_prefused, void* logits,
                                       float* rms_scale, float eps, const void* pf_base,
                                       long long pf_bytes, void* stream) {
    return LaunchRowRawAndRmsCta0<256>(x, gate_w_prefused, logits, rms_scale, eps, pf_base,
                                       pf_bytes, stream);
}

#ifndef NEXRT_MTP_RUNTIME_ONLY
extern "C" int NexN25ProGateTopKLaunch(const void* logits, const float* rms_scale, int32_t* packed,
                                       void* stream) {
    GateTopKSingleCtaNoLogitRmsScaleKernel<64, 63>
        <<<1, 512, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits), rms_scale, packed);
    return static_cast<int>(cudaGetLastError());
}
#endif

// Bridge / raw-weight path: same top-k as NexN25ProGateTopKLaunch, and additionally writes
// xn = x * rms_scale * gamma (gamma = 1 + post_attention_layernorm) for the w13-xn L1 kernel.
extern "C" int NexN25ProGateTopKXnLaunch(const void* logits, const float* rms_scale, const void* x,
                                         const float* gamma, int32_t* packed, void* xn,
                                         void* stream) {
    GateTopKSingleCtaNoLogitRmsScaleXnKernel<64, 63>
        <<<1, 512, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits), rms_scale,
            reinterpret_cast<const __nv_bfloat16*>(x), gamma, packed,
            reinterpret_cast<__nv_bfloat16*>(xn));
    return static_cast<int>(cudaGetLastError());
}

#ifndef NEXRT_MTP_RUNTIME_ONLY
extern "C" const char* NexN25ProGateGemvCudaErrorString(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
#endif
