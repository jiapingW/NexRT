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

// DfTargetHiddenProj: the draft's target-hidden projection (fc) as an M=8 GEMM
// plus its TP reduce over the LL peer-inbox wire. Two kernels, one fixed form.
//
//   K1 DfTargetHiddenProjGemm   Cᵀ = W_r · X8ᵀ in the TRANSPOSED mma orientation:
//      the weight tile is the A operand (m16 = 16 output channels per warp tile),
//      the 8 tokens are N=8, K is the reduction dim. mma.sync.m16n8k16 bf16->fp32,
//      the WM16_4W law of the M=8 MoE L1: weights consumed ONCE through a depth-5
//      cp.async ring of 2 KiB chunks per warp from an A-fragment-major, CTA-major
//      prepack (dflash/weights.py): each thread's 16 B cp.async lands in mma fragment
//      order (LDS.128 -> mma, no ldmatrix, no swizzle) and the CTA's four warps
//      read ONE contiguous 8 KiB burst per chunk. CTA = 2 row-tiles x 2 K-halves
//      = 4 warps, 128 CTAs = one wave; x8 [8, 4096] staged once per CTA as the
//      stationary B operand (natural n x k, k-contiguous layout of mma .col); the
//      K-half pair reduces through smem in fixed order; fp32 partials stored to
//      global.
//   K2 DfTargetHiddenProjReduce the M8 L2-tail LL form: one packet owner per
//      thread (8 tok x 4096 cols = 32768 fp32 packets -> 128 CTAs x 256 thr);
//      reads its partial, publishes ONE self-tagged packet into every rank's
//      inbox (multimem.st replication, [parity][src][pid]), JOINT-polls the PEERS
//      only with __nanosleep backoff, own value from registers, sums in fixed
//      ascending source order (identical on every rank), stores the sum as
//      BF16 straight into ctx_raw (the reference-visible rounding; the fp32
//      intermediate staging and copy). RMSNorm + gamma are NOT here: they
//      fuse into context_kv_proj / norm_rope downstream.
//
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../common/pdl.cuh"
#include "ll_wire.cuh"

namespace nexrt::sm90::nex_n2_5_pro::DFlash {
constexpr int kTok = 8;                               // tokens = mma N
constexpr int kHid = 4096;                            // K and output channels
constexpr int kTileRows = 16;                         // mma M
constexpr int kTiles = kHid / kTileRows;              // 256 row tiles
constexpr int kKSteps = kHid / 16;                    // 256 k16 steps
constexpr int kHalfSteps = kKSteps / 2;               // 128 per K-half
constexpr int kStepsPerChunk = 4;                     // 2 KiB chunk = 4 k16 steps
constexpr int kChunkElems = kStepsPerChunk * 32 * 8;  // 1024 bf16
constexpr int kChunkBytes = kChunkElems * 2;          // 2048
constexpr int kChunks = kHalfSteps / kStepsPerChunk;  // 32 per warp
constexpr int kDepth = 5;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kCtas = kTiles / 2;            // 128
constexpr int kXPitch = kHid + 8;            // bf16 elems; +16 B row skew -> conflict-free B loads
constexpr int kXBytes = kTok * kXPitch * 2;  // 65,664
constexpr int kRingBytes = kWarps * kDepth * kChunkBytes;  // 40,960
constexpr int kSmemBytes = kXBytes + kRingBytes;           // 106,624

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void mma16816(float (&c)[4], const uint4& a, const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b[0]), "r"(b[1]));
}

__global__ __launch_bounds__(kThreads) void DfTargetHiddenProjGemmKernel(
    const __nv_bfloat16* __restrict__ x8,     // [8, 4096] this rank's tap rows
    const __nv_bfloat16* __restrict__ wpack,  // CTA-major [128][32][4][4][32][8]
    float* __restrict__ part) {               // [8, 4096] fp32 partial
    NEXRT_PDL_ENTRY_SYNC();
    extern __shared__ __align__(16) unsigned char smem[];
    __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(smem);
    unsigned char* ring = smem + kXBytes;
    __shared__ float red[2][kTileRows][kTok];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tin = warp >> 1;  // tile within CTA
    const int tile = blockIdx.x * 2 + tin;
    const int khalf = warp & 1;
    const int g = lane >> 2, c4 = lane & 3;

    // stage x8 (B operand): cooperative 16 B cp.async, coalesced
    for (int i = threadIdx.x; i < kTok * (kHid / 8); i += kThreads) {
        const int j = i / (kHid / 8), k8 = i % (kHid / 8);
        cp16(smem_u32(sx + j * kXPitch + k8 * 8), x8 + j * kHid + k8 * 8);
    }
    asm volatile("cp.async.commit_group;");
    // this warp's weight stream: chunk c of the CTA's four warps is ONE contiguous
    // 8 KiB burst [chunk][warp][4 steps][32 lanes][8]
    const __nv_bfloat16* wsrc =
        wpack + static_cast<size_t>(blockIdx.x) * kChunks * kWarps * kChunkElems;
    unsigned char* wring = ring + warp * kDepth * kChunkBytes;
    auto issue = [&](int chunk) {
        const __nv_bfloat16* src =
            wsrc + (static_cast<size_t>(chunk) * kWarps + warp) * kChunkElems;
        unsigned char* dst = wring + (chunk % kDepth) * kChunkBytes;
#pragma unroll
        for (int s = 0; s < kStepsPerChunk; ++s) {
            cp16(smem_u32(dst + (s * 32 + lane) * 16), src + (s * 32 + lane) * 8);
        }
    };
#pragma unroll
    for (int c = 0; c < kDepth - 1; ++c) {
        issue(c);
        asm volatile("cp.async.commit_group;");
    }
    // x8 was staged COOPERATIVELY: cp.async groups are per-thread, so drain this
    // thread's x group and barrier the CTA before anyone reads sx. The weight ring
    // is lane-private (each lane copies and consumes its own fragments).
    asm volatile("cp.async.wait_group %0;" ::"n"(kDepth - 1));
    __syncthreads();

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int chunk = 0; chunk < kChunks; ++chunk) {
        asm volatile("cp.async.wait_group %0;" ::"n"(kDepth - 2));
        __syncwarp();
        const unsigned char* buf = wring + (chunk % kDepth) * kChunkBytes;
        const int step0 = khalf * kHalfSteps + chunk * kStepsPerChunk;
#pragma unroll
        for (int s = 0; s < kStepsPerChunk; ++s) {
            const uint4 a = *reinterpret_cast<const uint4*>(buf + (s * 32 + lane) * 16);
            const __nv_bfloat16* xr = sx + g * kXPitch + (step0 + s) * 16 + c4 * 2;
            uint32_t b[2];
            b[0] = *reinterpret_cast<const uint32_t*>(xr);
            b[1] = *reinterpret_cast<const uint32_t*>(xr + 8);
            mma16816(acc, a, b);
        }
        __syncwarp();
        if (chunk + kDepth - 1 < kChunks) {
            issue(chunk + kDepth - 1);
        }
        asm volatile("cp.async.commit_group;");  // empty groups keep the cadence
    }
    // K-half reduce, fixed order: half0 + half1
    if (khalf == 1) {
        red[tin][g][c4 * 2] = acc[0];
        red[tin][g][c4 * 2 + 1] = acc[1];
        red[tin][g + 8][c4 * 2] = acc[2];
        red[tin][g + 8][c4 * 2 + 1] = acc[3];
    }
    __syncthreads();
    if (khalf == 0) {
        acc[0] += red[tin][g][c4 * 2];
        acc[1] += red[tin][g][c4 * 2 + 1];
        acc[2] += red[tin][g + 8][c4 * 2];
        acc[3] += red[tin][g + 8][c4 * 2 + 1];
        const int col0 = tile * kTileRows + g, col1 = col0 + 8;  // output channels
        const int t0 = c4 * 2, t1 = t0 + 1;                      // tokens
        part[t0 * kHid + col0] = acc[0];
        part[t1 * kHid + col0] = acc[1];
        part[t0 * kHid + col1] = acc[2];
        part[t1 * kHid + col1] = acc[3];
    }
}

constexpr int kRedThreads = 256;
constexpr int kRedCtas = kTok * kHid / kRedThreads;  // 128
constexpr int kPkt = kTok * kHid;                    // 32768 per source
static_assert(2 * kLlMaxRanks * kPkt == kLlWireFloats, "wire holds 2 parities x 8 sources x 32768");
constexpr int kRedMaxSpins = 1 << 20;

__global__ __launch_bounds__(kRedThreads, 1) void DfTargetHiddenProjReduceKernel(
    const float* __restrict__ part,  // [8, 4096] this rank's fp32 partial
    unsigned long long inbox_mc, const uint64_t* __restrict__ inbox_local,
    unsigned int* __restrict__ ll_ctr,  // [>= kRedCtas]
    int* __restrict__ status,
    __nv_bfloat16* __restrict__ out,  // [8, 4096] bf16 reduced sum (== ctx_raw)
    int rank, int num_ranks) {
    NEXRT_PDL_ENTRY_SYNC();
    __shared__ unsigned cta_gen;
    if (threadIdx.x == 0) {
        unsigned g = ll_ctr[blockIdx.x] + 1u;
        if (g == 0u) g = 2u;  // reserve epoch 0 across the wrap
        ll_ctr[blockIdx.x] = g;
        cta_gen = g;
    }
    __syncthreads();
    const unsigned epoch = cta_gen;
    const unsigned parity = (epoch - 1u) & 1u;
    const int pid = blockIdx.x * kRedThreads + threadIdx.x;  // packet = (tok, col)
    const float mine = part[pid];
    const unsigned long long pkt_self =
        (static_cast<unsigned long long>(epoch) << 32) | __float_as_uint(mine);
    if (num_ranks > 1) {
        df_mm_st_rel_sys_b64(
            inbox_mc +
                ((static_cast<unsigned long long>(parity) * num_ranks + rank) * kPkt + pid) * 8ull,
            pkt_self);
    }
    const uint64_t* const mybox =
        inbox_local + static_cast<unsigned long long>(parity) * num_ranks * kPkt;
    const unsigned peer_full = ((1u << num_ranks) - 1u) & ~(1u << rank);
    uint64_t pk[kLlMaxRanks];
#pragma unroll
    for (int r = 0; r < kLlMaxRanks; ++r) pk[r] = 0ull;
    unsigned okm = 0u;
    int spins = 0;
    while (true) {  // JOINT poll: all peer loads per pass
        okm = 0u;
#pragma unroll
        for (int r = 0; r < kLlMaxRanks; ++r) {
            if (r < num_ranks && r != rank) {
                pk[r] = df_ld_rel_sys_b64(mybox + static_cast<unsigned long long>(r) * kPkt + pid);
                okm |= (static_cast<unsigned>(pk[r] >> 32) == epoch) ? (1u << r) : 0u;
            }
        }
        if (okm == peer_full) break;
        if (++spins > kRedMaxSpins) {
            atomicOr(status, 8);
            break;
        }  // fail-closed
        __nanosleep(64);
    }
    float sum = 0.f;
#pragma unroll
    for (int src = 0; src < kLlMaxRanks; ++src) {  // ascending source order
        if (src >= num_ranks) break;
        if (src == rank)
            sum += mine;
        else if ((okm >> src) & 1u)
            sum += __uint_as_float(static_cast<unsigned>(pk[src] & 0xffffffffull));
    }
    out[pid] =
        __float2bfloat16(sum);  // reference-visible bf16 rounding, folded (was ctx_raw.copy_)
}

}  // namespace nexrt::sm90::nex_n2_5_pro::DFlash
using namespace nexrt::sm90::nex_n2_5_pro::DFlash;
extern "C" {

int NexN25ProDFlashTargetHiddenProjGemmLaunch(const void* x8, const void* wpack, void* part,
                                              void* stream) {
    static bool ready = false;
    if (!ready) {
        const int rc = static_cast<int>(
            cudaFuncSetAttribute(reinterpret_cast<const void*>(DfTargetHiddenProjGemmKernel),
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
        if (rc) return rc;
        ready = true;
    }
    DfTargetHiddenProjGemmKernel<<<dim3(kCtas), dim3(kThreads), kSmemBytes,
                                   static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(x8), static_cast<const __nv_bfloat16*>(wpack),
        static_cast<float*>(part));
    return static_cast<int>(cudaGetLastError());
}

int NexN25ProDFlashTargetHiddenProjReduceLaunch(const void* part, unsigned long long inbox_mc,
                                                const void* inbox_local, void* ll_ctr, void* status,
                                                void* out, int rank, int num_ranks, void* stream) {
    if (num_ranks < 1 || num_ranks > kLlMaxRanks) return static_cast<int>(cudaErrorInvalidValue);
    DfTargetHiddenProjReduceKernel<<<dim3(kRedCtas), dim3(kRedThreads), 0,
                                     static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(part), inbox_mc, static_cast<const uint64_t*>(inbox_local),
        static_cast<unsigned int*>(ll_ctr), static_cast<int*>(status),
        static_cast<__nv_bfloat16*>(out), rank, num_ranks);
    return static_cast<int>(cudaGetLastError());
}
}
