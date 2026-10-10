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

// DfMlp (tensor-core form): the draft's dense gated MLP using the M=8 MoE laws
// (L1 -> L2-local -> L2-tail) minus gate/top-k/plan. TP=8: gate/up rows and
// down columns are rank-sharded (kInterR = 1536), ONE cross-rank reduce at the
// end, fused with the residual add exactly like the M8 MoE L2 tail.
//
//   K0 DfRmsNorm           xn = bf16(gamma * bf16(h * rsqrt(mean(h^2) + eps)))
//                          8 CTAs (one row each) x 512 thr; the row's full
//                          sum-of-squares is finished in-kernel (fixed order).
//                          Rounding chain == the reference _rms (fp32 normalise,
//                          bf16 cast, bf16 gamma multiply).
//   K1 DfMlpGateUpGemm     inter8 = bf16(bf16(silu(bf16 g)) * bf16 u), with
//                          g|u = xn . Wg_r^T | xn . Wu_r^T in the TRANSPOSED
//                          mma.m16n8k16 orientation of the M8 MoE L1 (WM16_4W):
//                          weight tile = A operand (m16 = 16 inter rows), the 8
//                          tokens = N, bf16 -> fp32 accumulate. CTA = one 16-row
//                          tile, warps = (gate|up) x K-split; weights stream once
//                          through a per-warp cp.async ring of 2 KiB chunks from
//                          an A-fragment-major, CTA-major prepack (dflash/weights.py):
//                          one contiguous (warps x 2 KiB) burst per chunk, the
//                          refill issued BEFORE the chunk's mma (MoE order). xn
//                          staged once per CTA as the stationary B operand.
//                          K-split partials reduce through smem in fixed order.
//   K2 DfMlpDownGemm       part = inter8 . Wd_r^T (fp32 [8, 4096], this rank's
//                          K = 1536 slice): same form, CTA = tiles x K-split.
//   (DfRmsNorm also serves the FINAL norm before the draft head: same op,
//   gamma = norm.weight; renamed from DfMlpRmsNorm when the head chain landed.)
//   PDL form (pdl=1 at the launchers): K1/K2/K3 are
//   launched with programmatic stream serialization; K1/K2 issue their static
//   weight-ring prologue BEFORE cudaGridDependencySynchronize() and trigger
//   their dependent at 3/4 of the K loop; K3 bumps its epoch and reads the
//   residual before the sync. The MoE L1 -> L2-local pair mechanism.
//   K3 DfMlpDownReduce     the M8 MoE L2-tail LL form: 128 CTAs x 128
//                          thr, one thread owns ONE 8 B packet [u32 epoch |
//                          bf16x2] = (token, column pair); ONE multimem.st into
//                          every rank's inbox [parity][src][pid]; self kept in
//                          registers; JOINT poll of the peers; fixed ascending
//                          source fold with self at its rank (identical on every
//                          rank); residual fold h = bf16(f(h) + sum). Fail-closed
//                          spin guard (status |= 16).
//
// ROW/POSITION INVARIANCE: every reduction has a fixed, row-independent order;
// no committed_len or unit count enters any range. DF-family TU (draft only).
#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY DF
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "../common/pdl.cuh"
#include "ll_wire.cuh"

namespace nexrt::sm90::nex_n2_5_pro::DFlash {
constexpr int kTok = 8;  // draft rows = mma N
constexpr int kHid = 4096;
constexpr int kInterR = 1536;  // 12288 / 8 ranks
constexpr float kEps = 1e-6f;
constexpr int kTileRows = 16;                         // mma M
constexpr int kStepsPerChunk = 4;                     // 2 KiB chunk = 4 k16 steps
constexpr int kChunkElems = kStepsPerChunk * 32 * 8;  // 1024 bf16
constexpr int kChunkBytes = kChunkElems * 2;          // 2048

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

// ------------------------------- K0: RMSNorm ---------------------------------
constexpr int kNormThreads = kHid / 8;  // 512: one 16 B vector per thread

__global__ __launch_bounds__(kNormThreads) void DfRmsNormKernel(
    const __nv_bfloat16* __restrict__ h,      // [8, 4096]
    const __nv_bfloat16* __restrict__ gamma,  // [4096]
    __nv_bfloat16* __restrict__ xn) {         // [8, 4096]
    NEXRT_PDL_ENTRY_SYNC();
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();  // no-op unless a PDL dependent follows
#endif
    __shared__ float red[kNormThreads / 32];
    const int j = blockIdx.x, t = threadIdx.x;
    const uint4 v4 = *reinterpret_cast<const uint4*>(h + j * kHid + t * 8);
    const uint4 g4 = *reinterpret_cast<const uint4*>(gamma + t * 8);
    const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&v4);
    const __nv_bfloat16* g = reinterpret_cast<const __nv_bfloat16*>(&g4);
    float x[8];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        x[i] = __bfloat162float(v[i]);
        ss += x[i] * x[i];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((t & 31) == 0) red[t >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int w = 0; w < kNormThreads / 32; ++w) tot += red[w];  // fixed order, all threads alike
    const float rinv = rsqrtf(tot * (1.0f / kHid) + kEps);
    __align__(16) __nv_bfloat16 o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const __nv_bfloat16 xb = __float2bfloat16(x[i] * rinv);  // reference: x.to(bf16)
        o[i] = __float2bfloat16(__bfloat162float(g[i]) *
                                __bfloat162float(xb));  // reference: w * x (bf16)
    }
    *reinterpret_cast<uint4*>(xn + j * kHid + t * 8) = *reinterpret_cast<const uint4*>(o);
}

// ------------------------- shared GEMM loop (K1 / K2) ------------------------
// One warp: its K-range of ONE 16-row A tile, kChunks chunks of 4 k16 steps,
// depth-kDepth cp.async ring, B fragments from the staged activations sx
// (row pitch xpitch bf16 = 16 B skew -> conflict-free 32-bit B loads).
// Weight-ring prologue: kDepth-1 chunks in flight. Weights are graph-static, so
// this is issued BEFORE the grid-dependency sync (PDL arm: it overlaps the
// predecessor's tail, the MoE L2-local mechanism).
template <int kDepth, int kWarps>
__device__ __forceinline__ void df_ring_prologue(const __nv_bfloat16* __restrict__ wsrc,
                                                 unsigned char* __restrict__ wring, int warp,
                                                 int lane) {
#pragma unroll
    for (int c = 0; c < kDepth - 1; ++c) {
        const __nv_bfloat16* src = wsrc + (static_cast<size_t>(c) * kWarps + warp) * kChunkElems;
        unsigned char* dst = wring + c * kChunkBytes;
#pragma unroll
        for (int s = 0; s < kStepsPerChunk; ++s) {
            cp16(smem_u32(dst + (s * 32 + lane) * 16), src + (s * 32 + lane) * 8);
        }
        asm volatile("cp.async.commit_group;");
    }
}

// Main stream. Precondition: ring prologue issued and x staged (both committed);
// this drains EVERY pending group (x is the newest group) and barriers once.
template <int kDepth, int kWarps, int kChunks, int kXPitch>
__device__ __forceinline__ void df_mma_stream(
    const __nv_bfloat16* __restrict__ wsrc,  // this CTA's pack base
    unsigned char* __restrict__ wring,       // this warp's ring
    const __nv_bfloat16* __restrict__ sx,    // staged B rows
    int warp, int lane, int step_base, float (&acc)[4]) {
    const int g = lane >> 2, c4 = lane & 3;
    auto issue = [&](int chunk, int slot) {
        const __nv_bfloat16* src =
            wsrc + (static_cast<size_t>(chunk) * kWarps + warp) * kChunkElems;
        unsigned char* dst = wring + slot * kChunkBytes;
#pragma unroll
        for (int s = 0; s < kStepsPerChunk; ++s) {
            cp16(smem_u32(dst + (s * 32 + lane) * 16), src + (s * 32 + lane) * 8);
        }
    };
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    int rslot = 0, islot = kDepth - 1;
#pragma unroll 2
    for (int chunk = 0; chunk < kChunks; ++chunk) {
        if (chunk == (kChunks * 3) / 4) cudaTriggerProgrammaticLaunchCompletion();
        asm volatile("cp.async.wait_group %0;" ::"n"(kDepth - 2));
        const unsigned char* buf = wring + rslot * kChunkBytes;
        uint4 a[kStepsPerChunk];
#pragma unroll
        for (int s = 0; s < kStepsPerChunk; ++s) {
            a[s] =
                *reinterpret_cast<const uint4*>(buf + (s * 32 + lane) * 16);  // lane-private bytes
        }
        if (chunk + kDepth - 1 < kChunks)
            issue(chunk + kDepth - 1, islot);    // refill BEFORE compute
        asm volatile("cp.async.commit_group;");  // empty groups keep cadence
        rslot = (rslot + 1 == kDepth) ? 0 : rslot + 1;
        islot = (islot + 1 == kDepth) ? 0 : islot + 1;
        const int step0 = step_base + chunk * kStepsPerChunk;
#pragma unroll
        for (int s = 0; s < kStepsPerChunk; ++s) {
            const __nv_bfloat16* xr = sx + g * kXPitch + (step0 + s) * 16 + c4 * 2;
            uint32_t b[2];
            b[0] = *reinterpret_cast<const uint32_t*>(xr);
            b[1] = *reinterpret_cast<const uint32_t*>(xr + 8);
            mma16816(acc, a[s], b);
        }
    }
}

// Stage a [kTok][K] bf16 activation block into smem (pitch xpitch) with 16 B cp.async.
template <int kThreads, int K, int kXPitch>
__device__ __forceinline__ void df_stage_rows(const __nv_bfloat16* __restrict__ src,
                                              __nv_bfloat16* sx) {
    for (int i = threadIdx.x; i < kTok * (K / 8); i += kThreads) {
        const int j = i / (K / 8), k8 = i % (K / 8);
        cp16(smem_u32(sx + j * kXPitch + k8 * 8), src + j * K + k8 * 8);
    }
    asm volatile("cp.async.commit_group;");
}

// Accumulator -> smem red[warp][row][tok] (fixed positions).
__device__ __forceinline__ void df_put_acc(float* rw, int lane, const float (&acc)[4]) {
    const int g = lane >> 2, c4 = lane & 3;
    rw[g * kTok + c4 * 2] = acc[0];
    rw[g * kTok + c4 * 2 + 1] = acc[1];
    rw[(g + 8) * kTok + c4 * 2] = acc[2];
    rw[(g + 8) * kTok + c4 * 2 + 1] = acc[3];
}

// ------------------------------ K1: gate|up ----------------------------------
template <int kDepth, int kSplit>
struct GateUpCfg {
    static constexpr int kWarps = 2 * kSplit;  // (gate|up) x K-split
    static constexpr int kThreads = kWarps * 32;
    static constexpr int kKSteps = kHid / 16;  // 256
    static constexpr int kSplitSteps = kKSteps / kSplit;
    static constexpr int kChunks = kSplitSteps / kStepsPerChunk;
    static constexpr int kCtas = kInterR / kTileRows;  // 96
    static constexpr int kXPitch = kHid + 8;
    static constexpr int kXBytes = kTok * kXPitch * 2;  // 65,664
    static constexpr int kRingBytes = kWarps * kDepth * kChunkBytes;
    static constexpr int kRedBytes = kWarps * kTileRows * kTok * 4;
    static constexpr int kSmem = kXBytes + kRingBytes + kRedBytes;
    static_assert(kSplitSteps % kStepsPerChunk == 0, "K-split must be chunk-aligned");
};

template <int kDepth, int kSplit>
__global__ __launch_bounds__(GateUpCfg<kDepth, kSplit>::kThreads) void DfMlpGateUpGemmKernel(
    const __nv_bfloat16* __restrict__ xn,     // [8, 4096]
    const __nv_bfloat16* __restrict__ wpack,  // [96][chunks][warps][4][32][8]
    __nv_bfloat16* __restrict__ inter8) {     // [8, 1536]
    using C = GateUpCfg<kDepth, kSplit>;
    extern __shared__ __align__(16) unsigned char smem[];
    __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(smem);
    unsigned char* ring = smem + C::kXBytes;
    float* red = reinterpret_cast<float*>(smem + C::kXBytes + C::kRingBytes);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int ks = warp % kSplit;  // warp = branch * kSplit + ks
    const __nv_bfloat16* wsrc =
        wpack + static_cast<size_t>(blockIdx.x) * C::kChunks * C::kWarps * kChunkElems;
    unsigned char* wring = ring + warp * kDepth * kChunkBytes;
    df_ring_prologue<kDepth, C::kWarps>(wsrc, wring, warp,
                                        lane);  // static weights: before the sync
    NEXRT_PDL_ENTRY_SYNC();                     // xn is the predecessor's output
    df_stage_rows<C::kThreads, kHid, C::kXPitch>(xn, sx);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    df_mma_stream<kDepth, C::kWarps, C::kChunks, C::kXPitch>(wsrc, wring, sx, warp, lane,
                                                             ks * C::kSplitSteps, acc);
    df_put_acc(red + warp * (kTileRows * kTok), lane, acc);
    __syncthreads();
    if (threadIdx.x < kTileRows * kTok) {  // one (row, tok) per thread
        const int r = threadIdx.x >> 3, tok = threadIdx.x & 7;
        float gs = 0.f, us = 0.f;
#pragma unroll
        for (int q = 0; q < kSplit; ++q) {  // ascending K order
            gs += red[q * (kTileRows * kTok) + r * kTok + tok];
            us += red[(kSplit + q) * (kTileRows * kTok) + r * kTok + tok];
        }
        // reference-visible rounding chain: bf16 g, bf16 u, bf16 silu, bf16 product
        const float gb = __bfloat162float(__float2bfloat16(gs));
        const float ub = __bfloat162float(__float2bfloat16(us));
        const float sb = __bfloat162float(__float2bfloat16(gb / (1.0f + __expf(-gb))));
        inter8[tok * kInterR + blockIdx.x * kTileRows + r] = __float2bfloat16(sb * ub);
    }
}

// -------------------------------- K2: down -----------------------------------
template <int kDepth, int kSplit, int kTiles>
struct DownCfg {
    static constexpr int kWarps = kTiles * kSplit;
    static constexpr int kThreads = kWarps * 32;
    static constexpr int kKSteps = kInterR / 16;  // 96
    static constexpr int kSplitSteps = kKSteps / kSplit;
    static constexpr int kChunks = kSplitSteps / kStepsPerChunk;
    static constexpr int kCtas = kHid / kTileRows / kTiles;  // 128 (2 tiles) / 256 (1 tile)
    static constexpr int kXPitch = kInterR + 8;              // 1544: 772 words = 4 mod 32
    static constexpr int kXBytes = kTok * kXPitch * 2;       // 24,704
    static constexpr int kRingBytes = kWarps * kDepth * kChunkBytes;
    static constexpr int kRedBytes = kWarps * kTileRows * kTok * 4;
    static constexpr int kSmem = kXBytes + kRingBytes + kRedBytes;
    static_assert(kSplitSteps % kStepsPerChunk == 0, "K-split must be chunk-aligned");
    static_assert(kThreads >= kTileRows * kTok, "epilogue needs 128 threads");
};

template <int kDepth, int kSplit, int kTiles>
__global__ __launch_bounds__(DownCfg<kDepth, kSplit, kTiles>::kThreads) void DfMlpDownGemmKernel(
    const __nv_bfloat16* __restrict__ inter8,  // [8, 1536]
    const __nv_bfloat16* __restrict__ wpack,   // [ctas][chunks][warps][4][32][8]
    float* __restrict__ part) {                // [8, 4096] fp32
    using C = DownCfg<kDepth, kSplit, kTiles>;
    extern __shared__ __align__(16) unsigned char smem[];
    __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(smem);
    unsigned char* ring = smem + C::kXBytes;
    float* red = reinterpret_cast<float*>(smem + C::kXBytes + C::kRingBytes);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int ks = warp % kSplit;  // warp = tile * kSplit + ks
    const __nv_bfloat16* wsrc =
        wpack + static_cast<size_t>(blockIdx.x) * C::kChunks * C::kWarps * kChunkElems;
    unsigned char* wring = ring + warp * kDepth * kChunkBytes;
    df_ring_prologue<kDepth, C::kWarps>(wsrc, wring, warp,
                                        lane);  // static weights: before the sync
    NEXRT_PDL_ENTRY_SYNC();                     // inter8 is the predecessor's output
    df_stage_rows<C::kThreads, kInterR, C::kXPitch>(inter8, sx);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    df_mma_stream<kDepth, C::kWarps, C::kChunks, C::kXPitch>(wsrc, wring, sx, warp, lane,
                                                             ks * C::kSplitSteps, acc);
    df_put_acc(red + warp * (kTileRows * kTok), lane, acc);
    __syncthreads();
    if (threadIdx.x < kTileRows * kTok) {
        const int r = threadIdx.x >> 3, tok = threadIdx.x & 7;
#pragma unroll
        for (int t = 0; t < kTiles; ++t) {
            float s = 0.f;
#pragma unroll
            for (int q = 0; q < kSplit; ++q) {  // ascending K order
                s += red[(t * kSplit + q) * (kTileRows * kTok) + r * kTok + tok];
            }
            part[tok * kHid + (blockIdx.x * kTiles + t) * kTileRows + r] = s;
        }
    }
}

// ---------------------- K3: LL peer-inbox tail + residual --------------------
constexpr int kRedCtas = 128;
constexpr int kRedThreads = 128;
constexpr int kPktPerSrc = kTok * kHid / 2;  // 16384 bf16x2 packets
constexpr int kRedMaxSpins = kLlMaxSpins;
static_assert(kRedCtas * kRedThreads == kPktPerSrc, "one owner per packet");
static_assert(2L * kLlMaxRanks * kPktPerSrc <= kLlWireFloats, "wire holds 2 parities x 8 sources");

__global__ __launch_bounds__(kRedThreads, 1) void DfMlpDownReduceKernel(
    const float* __restrict__ part,  // [8, 4096] this rank's partial
    unsigned long long inbox_mc, const uint64_t* __restrict__ inbox_local,
    unsigned int* __restrict__ ll_ctr,  // [>= kRedCtas]
    int* __restrict__ status,
    __nv_bfloat16* __restrict__ h,  // [8, 4096] += reduced
    int rank, int num_ranks) {
    // Pre-sync work depends only on protocol state and on h (last written by
    // DfOProjLL, complete before this layer's MLP chain launched): epoch bump,
    // residual read. The partial (K2's output) is read after the sync.
    __shared__ unsigned cta_gen;
    if (threadIdx.x == 0) {
        unsigned g = ll_ctr[blockIdx.x] + 1u;
        if (g == 0u) g = 2u;  // reserve epoch 0 across the wrap
        ll_ctr[blockIdx.x] = g;
        cta_gen = g;
    }
    const int pid = blockIdx.x * kRedThreads + threadIdx.x;  // (tok, column pair)
    __nv_bfloat162* const d = reinterpret_cast<__nv_bfloat162*>(h) + pid;
    const __nv_bfloat162 rv = *d;  // residual issued first
    NEXRT_PDL_ENTRY_SYNC();
    __syncthreads();
    const unsigned epoch = cta_gen;
    const unsigned parity = (epoch - 1u) & 1u;
    // Keep the producer's output load after PDL synchronization. CUDA 13 can
    // hoist an ordinary const/restrict load above cudaGridDependencySynchronize.
    float2 p2;
    asm volatile("ld.volatile.global.v2.f32 {%0,%1}, [%2];"
                 : "=f"(p2.x), "=f"(p2.y)
                 : "l"(part + 2 * pid)
                 : "memory");
    const __nv_bfloat162 pv = __float22bfloat162_rn(p2);
    const unsigned payload = *reinterpret_cast<const unsigned*>(&pv);
    const unsigned long long pkt_self = (static_cast<unsigned long long>(epoch) << 32) | payload;
    if (num_ranks > 1) {
        df_mm_st_rel_sys_b64(
            inbox_mc +
                ((static_cast<unsigned long long>(parity) * num_ranks + rank) * kPktPerSrc + pid) *
                    8ull,
            pkt_self);
    }
    const uint64_t* const mybox =
        inbox_local + static_cast<unsigned long long>(parity) * num_ranks * kPktPerSrc;
    const unsigned peer_full = ((1u << num_ranks) - 1u) & ~(1u << rank);
    uint64_t pk[kLlMaxRanks];
#pragma unroll
    for (int r = 0; r < kLlMaxRanks; ++r) pk[r] = 0ull;
    unsigned okm = 0u;
    int spins = 0;
    while (true) {  // JOINT poll: every peer per pass
        okm = 0u;
#pragma unroll
        for (int r = 0; r < kLlMaxRanks; ++r) {
            if (r < num_ranks && r != rank) {
                pk[r] = df_ld_rel_sys_b64(mybox + static_cast<unsigned long long>(r) * kPktPerSrc +
                                          pid);
                okm |= (static_cast<unsigned>(pk[r] >> 32) == epoch) ? (1u << r) : 0u;
            }
        }
        if (okm == peer_full) break;
        if (++spins > kRedMaxSpins) {
            atomicOr(status, 16);
            break;
        }  // fail-closed
        __nanosleep(64);
    }
    float r0 = 0.f, r1 = 0.f;
#pragma unroll
    for (int s = 0; s < kLlMaxRanks; ++s) {  // ascending source, self at rank
        if (s >= num_ranks) break;
        if (s == rank) {
            r0 += __bfloat162float(pv.x);
            r1 += __bfloat162float(pv.y);
        } else if ((okm >> s) & 1u) {
            const unsigned w = static_cast<unsigned>(pk[s] & 0xffffffffull);
            const __nv_bfloat162 u = *reinterpret_cast<const __nv_bfloat162*>(&w);
            r0 += __bfloat162float(u.x);
            r1 += __bfloat162float(u.y);
        }
    }
    *d = __nv_bfloat162(__float2bfloat16(__bfloat162float(rv.x) + r0),
                        __float2bfloat16(__bfloat162float(rv.y) + r1));
}

// ------------------------------- launch glue ---------------------------------
template <int kDepth, int kSplit>
static int launch_gateup(const void* xn, const void* wpack, void* inter8, void* stream) {
    using C = GateUpCfg<kDepth, kSplit>;
    auto* kern = &DfMlpGateUpGemmKernel<kDepth, kSplit>;
    static bool ready = false;
    if (!ready) {
        const int rc = static_cast<int>(
            cudaFuncSetAttribute(reinterpret_cast<const void*>(kern),
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmem));
        if (rc) return rc;
        ready = true;
    }
    const auto* x = static_cast<const __nv_bfloat16*>(xn);
    const auto* w = static_cast<const __nv_bfloat16*>(wpack);
    auto* o = static_cast<__nv_bfloat16*>(inter8);
    return static_cast<int>(NexrtPdlLaunchKernel(dim3(C::kCtas), dim3(C::kThreads), C::kSmem,
                                                 static_cast<cudaStream_t>(stream), kern, x, w, o));
}

template <int kDepth, int kSplit, int kTiles>
static int launch_down(const void* inter8, const void* wpack, void* part, void* stream) {
    using C = DownCfg<kDepth, kSplit, kTiles>;
    auto* kern = &DfMlpDownGemmKernel<kDepth, kSplit, kTiles>;
    static bool ready = false;
    if (!ready) {
        const int rc = static_cast<int>(
            cudaFuncSetAttribute(reinterpret_cast<const void*>(kern),
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmem));
        if (rc) return rc;
        ready = true;
    }
    const auto* x = static_cast<const __nv_bfloat16*>(inter8);
    const auto* w = static_cast<const __nv_bfloat16*>(wpack);
    auto* o = static_cast<float*>(part);
    return static_cast<int>(NexrtPdlLaunchKernel(dim3(C::kCtas), dim3(C::kThreads), C::kSmem,
                                                 static_cast<cudaStream_t>(stream), kern, x, w, o));
}
}  // namespace nexrt::sm90::nex_n2_5_pro::DFlash
using namespace nexrt::sm90::nex_n2_5_pro::DFlash;
extern "C" {

int NexN25ProDFlashRmsNormLaunch(const void* h, const void* gamma, void* xn, void* stream) {
    DfRmsNormKernel<<<dim3(kTok), dim3(kNormThreads), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(h), static_cast<const __nv_bfloat16*>(gamma),
        static_cast<__nv_bfloat16*>(xn));
    return static_cast<int>(cudaGetLastError());
}

// (depth, ksplit, pdl) selects a static instantiation; the pack layout depends on ksplit only.
// pdl != 0 launches with programmatic stream serialization (the kernel then waits for
// its predecessor only at the xn staging, after its weight-ring prologue).
int NexN25ProDFlashMlpGateUpGemmLaunch(const void* xn, const void* wpack, void* inter8, int depth,
                                       int ksplit, int pdl, void* stream) {
    if (depth == 5 && ksplit == 2 && pdl) return launch_gateup<5, 2>(xn, wpack, inter8, stream);
    return static_cast<int>(cudaErrorInvalidValue);  // the form is fixed
}

// (depth, ksplit, tiles, pdl) selects a static instantiation; the pack depends on (ksplit, tiles).
int NexN25ProDFlashMlpDownGemmLaunch(const void* inter8, const void* wpack, void* part, int depth,
                                     int ksplit, int tiles, int pdl, void* stream) {
    if (depth == 5 && ksplit == 2 && tiles == 2 && pdl)
        return launch_down<5, 2, 2>(inter8, wpack, part, stream);
    return static_cast<int>(cudaErrorInvalidValue);  // the form is fixed
}

int NexN25ProDFlashMlpDownReduceLaunch(const void* part, unsigned long long inbox_mc,
                                       const void* inbox_local, void* ll_ctr, void* status, void* h,
                                       int rank, int num_ranks, int pdl, void* stream) {
    if (num_ranks < 1 || num_ranks > kLlMaxRanks || rank < 0 || rank >= num_ranks ||
        status == nullptr || (inbox_mc == 0ull && num_ranks > 1)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (!pdl) return static_cast<int>(cudaErrorInvalidValue);
    return static_cast<int>(NexrtPdlLaunchKernel(
        dim3(kRedCtas), dim3(kRedThreads), 0, static_cast<cudaStream_t>(stream),
        DfMlpDownReduceKernel, static_cast<const float*>(part), inbox_mc,
        static_cast<const uint64_t*>(inbox_local), static_cast<unsigned int*>(ll_ctr),
        static_cast<int*>(status), static_cast<__nv_bfloat16*>(h), rank, num_ranks));
}

int NexN25ProDFlashMlpTcInterR() { return kInterR; }
}
