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

// M=8 MoE L2-local: row-major FP8 W2 with H-fragment and scale reuse.
// Uses the existing PDL dependency and LL peer-inbox tail.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace nexrt::sm90::nex_n2_5_pro::M8 {
constexpr int kDim = 4096;
constexpr int kTopK = 10;
constexpr int kM8 = 8;
constexpr int kRankInter = 128;
constexpr int kPackedWords = 64;
// Eight token lanes each read four adjacent IDs: 4-word row padding
// gives each lane a distinct four-bank group. Keep global rows at 64 words.
constexpr int kPackedSharedWords = kPackedWords + 4;
constexpr int kHRows = kTopK * kRankInter + kRankInter;
constexpr int kSlots = kTopK + 1;
constexpr int kMaxDesc = kM8 * kTopK + 1;
constexpr int kMaxRanks = 8;

#ifndef M8L2TC_SPIN_CAP
#define M8L2TC_SPIN_CAP (1 << 30)  // gate builds override; timed binary never does
#endif
constexpr int kMaxSpins = M8L2TC_SPIN_CAP;

// ---- LL tail geometry ----
constexpr int kPktPerSrc = kM8 * (kDim / 2);
constexpr int kTailLlCtas = 128;
constexpr int kTailLlThreads = 128;         // 16384 packet owners
constexpr int kTailPairsPerTok = kDim / 2;  // 2048 row pairs per token
constexpr int kTailCtrInts = 128;           // per-CTA epoch counters
static_assert(kTailLlCtas * kTailLlThreads == kPktPerSrc, "LL: one owner per packet");
static_assert(kTailCtrInts >= kTailLlCtas, "counters");

constexpr int kTcRows = 512;
constexpr int kTcBlocks = kDim / kTcRows;
constexpr int kTcGrid = kMaxDesc * kTcBlocks;
constexpr int kTcThreads = 128;
constexpr int kTcTilesPerWarp = kTcRows / 16 / 4;
constexpr int kTcTileBytes = 16 * kRankInter;
constexpr int kTcDepth = 4;
constexpr int kTcHPitch = kRankInter * 2 + 16;

constexpr int kTcPackedOff = 0;
constexpr int kTcHOff = ((kM8 * kPackedSharedWords * 4 + 127) / 128) * 128;
constexpr int kTcRingOff = kTcHOff + kM8 * kTcHPitch;
constexpr int kTcSoutOff = kTcRingOff + 4 * kTcDepth * kTcTileBytes;
constexpr int kTcDescOff = kTcSoutOff + 4 * kM8 * 16 * 4;
constexpr int kTcDyn = kTcDescOff + 192;

__device__ __forceinline__ void tc_cp_async16_ca(void* smem_dst, const void* gmem_src) {
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(d), "l"(gmem_src));
}
__device__ __forceinline__ void tc_mma_m16n8k16_bf16(float (&c)[4], const unsigned (&a)[4],
                                                     const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void native_fp8_mma(float (&c)[4], const unsigned (&a)[4],
                                               const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

struct alignas(128) M8RoutePlan {
    int32_t n_desc;
    int32_t reserved[31];
    alignas(128) int32_t expert_id[128];
};

__device__ __forceinline__ void tc_prefetch16(char* stage, const __nv_fp8_e4m3* source, int lane) {
    const unsigned base = static_cast<unsigned>(__cvta_generic_to_shared(stage));
    const unsigned word = (lane * 16) ^ ((lane >> 3) * 16);
    const unsigned even = base + word;
    const unsigned odd = base + (word ^ 64);
    asm volatile(
        "cp.async.cg.shared.global.L2::256B [%0], [%2], 16;\n"
        "cp.async.cg.shared.global.L2::256B [%1+512], [%2+512], 16;\n"
        "cp.async.cg.shared.global.L2::256B [%0+1024], [%2+1024], 16;\n"
        "cp.async.cg.shared.global.L2::256B [%1+1536], [%2+1536], 16;" ::"r"(even),
        "r"(odd), "l"(source));
}

__device__ __forceinline__ uint4 tc_fragment16(const char* stage, int pair, int lane) {
    union {
        uint4 vector;
        uint16_t pair[8];
    } fragment;
    const int row = (lane >> 2);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int col = pair * 32 + half * 16 + (lane & 3) * 2;
        const int offset = (row * 128 + col) ^ ((row & 7) << 4);
        const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(stage + offset));
        unsigned p0, p1, p2, p3;
        // Row +8 and column +8 are fixed offsets under this swizzle.
        asm volatile(
            "ld.shared.u16 %0, [%4];\n"
            "ld.shared.u16 %1, [%4+1024];\n"
            "ld.shared.u16 %2, [%4+8];\n"
            "ld.shared.u16 %3, [%4+1032];\n"
            : "=r"(p0), "=r"(p1), "=r"(p2), "=r"(p3)
            : "r"(addr));
        fragment.pair[half * 4] = p0;
        fragment.pair[half * 4 + 1] = p1;
        fragment.pair[half * 4 + 2] = p2;
        fragment.pair[half * 4 + 3] = p3;
    }
    return fragment.vector;
}

// MMA scalar stores and float4 transpose loads both use distinct banks.
__device__ __forceinline__ int tc_sout_offset(int row, int col) {
    return (row * 16 + col) ^ ((row >> 1) * 8);
}

__device__ __forceinline__ void l2_local_body(
    const M8RoutePlan* __restrict__ plan, const int32_t* __restrict__ packed8,
    const float* __restrict__ rawh8, const __nv_fp8_e4m3* __restrict__ w2_raw,
    const float* __restrict__ w2_scales, const __nv_fp8_e4m3* __restrict__ shared_w2_raw,
    const float* __restrict__ shared_w2_scales, float* __restrict__ l2p) {
    const int desc = blockIdx.x >> 3;
    const int rb = blockIdx.x & 7;
    const int n_desc = plan->n_desc;
    if (desc >= n_desc) return;
    const int e = -1;  // Shared-expert fallback only.
    const bool routed = e >= 0;

    extern __shared__ __align__(1024) char sm[];
    const int32_t* const pk = reinterpret_cast<const int32_t*>(sm + kTcPackedOff);
    char* const sh = sm + kTcHOff;
    int* const s_tok = reinterpret_cast<int*>(sm + kTcDescOff);
    int* const s_off = reinterpret_cast<int*>(sm + kTcDescOff + 32);
    int* const s_hsrc = reinterpret_cast<int*>(sm + kTcDescOff + 64);
    int* const s_c = reinterpret_cast<int*>(sm + kTcDescOff + 96);
    float* const sscl = reinterpret_cast<float*>(sm + kTcDescOff + 128);

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int gid = lane >> 2;
    const int tig = lane & 3;
    char* const ring = sm + kTcRingOff + warp * (kTcDepth * kTcTileBytes);
    float* const sout = reinterpret_cast<float*>(sm + kTcSoutOff) + warp * (kM8 * 16);

    const __nv_fp8_e4m3* const wbase =
        (routed ? w2_raw + static_cast<size_t>(e) * (kDim * kRankInter) : shared_w2_raw) +
        static_cast<size_t>(rb) * (kTcRows * kRankInter);

    if (warp == 0) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            tc_cp_async16_ca(sm + kTcPackedOff + (j * 2 + (lane >> 4)) * (kPackedSharedWords * 4) +
                                 (lane & 15) * 16,
                             reinterpret_cast<const char*>(packed8) + j * 512 + lane * 16);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncwarp();
        int slot = routed ? -1 : kTopK;
        if (lane < kM8 && routed) {
#pragma unroll
            for (int s = 0; s < kTopK; ++s) {
                if (pk[lane * kPackedSharedWords + s] == e) slot = s;
            }
        }
        const bool act = (lane < kM8) && (slot >= 0);
        const unsigned amask = __ballot_sync(0xffffffffu, act);
        if (act) {
            const int pos = __popc(amask & ((1u << lane) - 1u));
            s_tok[pos] = lane;
            s_off[pos] = (lane * kSlots + slot) * kDim;
            s_hsrc[pos] = lane * kHRows + slot * kRankInter;
        }
        if (lane == 0) s_c[0] = __popc(amask);
        if (lane < 4) {
            sscl[lane] = routed ? w2_scales[e * (kDim / 128) + rb * 4 + lane]
                                : shared_w2_scales[rb * 4 + lane];
        }
    } else {
#pragma unroll
        for (int st = 0; st < kTcDepth - 1; ++st) {
            tc_prefetch16(ring + st * kTcTileBytes,
                          wbase + (warp * kTcTilesPerWarp + st) * kTcTileBytes + lane * 16, lane);
            asm volatile("cp.async.commit_group;");
        }
    }
    __syncthreads();
    const int c = s_c[0];

    if (warp == 0) {
#pragma unroll
        for (int st = 0; st < kTcDepth - 1; ++st) {
            tc_prefetch16(ring + st * kTcTileBytes, wbase + st * kTcTileBytes + lane * 16, lane);
            asm volatile("cp.async.commit_group;");
        }
    }
    cudaGridDependencySynchronize();

    {
        const int n = threadIdx.x >> 4, q = threadIdx.x & 15;
        float v[8], amax = 1e-10f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[j] = (n < c) ? __bfloat162float(__float2bfloat16(rawh8[s_hsrc[n] + q * 8 + j])) : 0.f;
            amax = fmaxf(amax, fabsf(v[j]));
        }
        if (routed) {
#pragma unroll
            for (int off = 8; off; off >>= 1)
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        }
        const float inv = routed ? __fdividef(448.f, amax) : 1.f;
        __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(sh + n * kTcHPitch) + q * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (routed)
                reinterpret_cast<__nv_fp8_e4m3*>(sh + n * kTcHPitch)[q * 8 + j] =
                    __nv_fp8_e4m3(v[j] * inv);
            else
                dst[j] = __float2bfloat16(v[j]);
        }
        if (q == 0) {
            float* factors = reinterpret_cast<float*>(sh + n * kTcHPitch + 256);
            factors[0] = routed ? amax * (1.f / 448.f) : 1.f;
            const int slot = (n < c) ? s_hsrc[n] % kHRows / 128 : 0;
            factors[1] = (routed && n < c)
                             ? __int_as_float(pk[s_tok[n] * kPackedSharedWords + 10 + slot])
                             : 1.f;
        }
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    // H is invariant over the eight output tiles assigned to this warp.
    // Keep 16 operand registers; source still comes from the CTA's H tile.
    unsigned hfrag[4][4];
    if (routed) {
#pragma unroll
        for (int pp = 0; pp < 4; ++pp) {
            const char* hp = sh + gid * kTcHPitch + pp * 32 + tig * 4;
            hfrag[pp][0] = *reinterpret_cast<const unsigned*>(hp);
            hfrag[pp][1] = *reinterpret_cast<const unsigned*>(hp + 16);
        }
    } else {
#pragma unroll
        for (int pp = 0; pp < 4; ++pp) {
            const unsigned haddr = static_cast<unsigned>(
                __cvta_generic_to_shared(sh + (lane & 7) * kTcHPitch + pp * 64 + (lane >> 3) * 16));
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                "{%0,%1,%2,%3}, [%4];"
                : "=r"(hfrag[pp][0]), "=r"(hfrag[pp][1]), "=r"(hfrag[pp][2]), "=r"(hfrag[pp][3])
                : "r"(haddr));
        }
    }

    // Each warp covers 128 output rows, exactly one scale group.
    // Reuse the same scale value; CTA rows/warps/work are unchanged.
    static_assert(kTcTilesPerWarp * 16 == 128);
    const float wsc = sscl[warp];
    const float* fa = reinterpret_cast<const float*>(sh + (tig * 2) * kTcHPitch + 256);
    const float* fb = reinterpret_cast<const float*>(sh + (tig * 2 + 1) * kTcHPitch + 256);
    const float scale_a = wsc * fa[0], scale_b = wsc * fb[0];
    const float route_a = fa[1], route_b = fb[1];
    const int tile0 = warp * kTcTilesPerWarp;
    int rslot = 0;
    int islot = kTcDepth - 1;
#pragma unroll 1
    for (int tw = 0; tw < kTcTilesPerWarp; ++tw) {
        if (tw + kTcDepth - 1 < kTcTilesPerWarp) {
            tc_prefetch16(ring + islot * kTcTileBytes,
                          wbase + (tile0 + tw + kTcDepth - 1) * kTcTileBytes + lane * 16, lane);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group %0;" ::"n"(kTcDepth - 1));
        __syncwarp();
        float cacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const char* const wt = ring + rslot * kTcTileBytes;
        rslot = (rslot + 1 == kTcDepth) ? 0 : rslot + 1;
        islot = (islot + 1 == kTcDepth) ? 0 : islot + 1;
        if (routed) {
#pragma unroll
            for (int pp = 0; pp < 4; ++pp) {
                unsigned a[4];
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const int row = gid + ((p & 1) ? 8 : 0),
                              col = pp * 32 + ((p & 2) ? 16 : 0) + tig * 4;
                    const int off = (row * 128 + col) ^ ((row & 7) << 4);
                    a[p] = *reinterpret_cast<const unsigned*>(wt + off);
                }
                unsigned b[2] = {hfrag[pp][0], hfrag[pp][1]};
                native_fp8_mma(cacc, a, b);
            }
        } else {
#pragma unroll
            for (int pp = 0; pp < 4; ++pp) {
                const uint4 wq = tc_fragment16(wt, pp, lane);
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const uint2 wv = (hh == 0) ? make_uint2(wq.x, wq.y) : make_uint2(wq.z, wq.w);
                    const __nv_fp8x2_e4m3* pw = reinterpret_cast<const __nv_fp8x2_e4m3*>(&wv);
                    unsigned a[4];
#pragma unroll
                    for (int p_ = 0; p_ < 4; ++p_) {
                        const __nv_bfloat162 w2b =
                            __float22bfloat162_rn(static_cast<float2>(pw[p_]));
                        a[p_] = *reinterpret_cast<const unsigned*>(&w2b);
                    }
                    unsigned b[2] = {hfrag[pp][hh * 2], hfrag[pp][hh * 2 + 1]};
                    tc_mma_m16n8k16_bf16(cacc, a, b);
                }
            }
        }
        float out[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float f = cacc[j] * ((j & 1) ? scale_b : scale_a) * ((j & 1) ? route_b : route_a);
            out[j] = routed ? __bfloat162float(__float2bfloat16(f)) : f;
        }
        sout[tc_sout_offset(tig * 2, gid)] = out[0];
        sout[tc_sout_offset(tig * 2 + 1, gid)] = out[1];
        sout[tc_sout_offset(tig * 2, gid + 8)] = out[2];
        sout[tc_sout_offset(tig * 2 + 1, gid + 8)] = out[3];
        __syncwarp();
        const int n = lane >> 2;
        const int q = lane & 3;
        if (n < c) {
            const float4 v = *reinterpret_cast<const float4*>(sout + tc_sout_offset(n, q * 4));
            *reinterpret_cast<float4*>(l2p + s_off[n] + (rb * (kTcRows / 16) + tile0 + tw) * 16 +
                                       q * 4) = v;
        }
        __syncwarp();
    }
}

#ifndef DOT_CHUNK
#define DOT_CHUNK 1
#endif

#include "m8_wgmma_k32_l2.cuh"
static_assert(kTcDyn >= WG2Dyn);

__global__ __launch_bounds__(kTcThreads) void M8MoeL2LocalKernel(
    const M8RoutePlan* __restrict__ plan, const int32_t* __restrict__ packed8,
    const float* __restrict__ rawh8, const __nv_fp8_e4m3* __restrict__ w2_raw,
    const float* __restrict__ w2_scales, const __nv_fp8_e4m3* __restrict__ shared_w2_raw,
    const float* __restrict__ shared_w2_scales, float* __restrict__ l2p) {
    if ((blockIdx.x >> 3) >= plan->n_desc) return;
    if (plan->expert_id[blockIdx.x >> 3] >= 0)
        wg2_body(plan, packed8, rawh8, w2_raw, w2_scales, l2p);
    else
        l2_local_body(plan, packed8, rawh8, w2_raw, w2_scales, shared_w2_raw, shared_w2_scales,
                      l2p);
}

// ----------------------------- LL peer-inbox tail -----------------------------
// 8 B packet primitives: a naturally aligned 8 B access is single-copy atomic,
// so a fresh epoch tag can never pair with stale payload. Nothing here relies
// on 16 B atomicity. multimem.st = REPLICATION only (never multimem.red /
// ld_reduce: fabric arithmetic would change the reduction order).
__device__ __forceinline__ void ll_mc_st_pkt(unsigned long long mc, unsigned long long v) {
    asm volatile("multimem.st.relaxed.sys.global.u64 [%0], %1;" ::"l"(mc), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long ll_ld_pkt(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

// 128 CTAs on 132 SMs: minBlocksPerMultiprocessor = 1 costs no occupancy and
// keeps the eleven fold loads and the seven peer packets in registers.
__global__ __launch_bounds__(kTailLlThreads, 1) void M8MoeL2TailLlKernel(
    const float* __restrict__ l2p,  // [8][11][4096] fp32
    int* __restrict__ cur_gen,
    const unsigned long long* __restrict__ inbox_ptrs,  // unicast peer VAs
    unsigned long long inbox_mc,        // NVLS multicast alias (0 only at num_ranks == 1)
    unsigned int* __restrict__ ll_ctr,  // [>= kTailLlCtas]
    int rank, int num_ranks,
    int* __restrict__ fuse_status,          // non-null (launcher-checked)
    __nv_bfloat16* __restrict__ fuse_x8) {  // [8, 4096] bf16 residual
    __shared__ unsigned cta_gen;
    if (threadIdx.x == 0) {
        unsigned g = ll_ctr[blockIdx.x] + 1u;
        if (g == 0u) g = 2u;     // reserve epoch 0; +2 keeps
        ll_ctr[blockIdx.x] = g;  // parity alternation across
        cta_gen = g;             // the 2^32 wrap
        if (blockIdx.x == 0) {
            cur_gen[0] = static_cast<int>(g);
        }
    }
    __syncthreads();
    const unsigned gen = cta_gen;
    const unsigned parity = (gen - 1u) & 1u;
    const unsigned epoch = gen;
    const int gid = blockIdx.x * kTailLlThreads + threadIdx.x;  // packet index
    const int t = gid / kTailPairsPerTok;                       // token
    const int pr = gid - t * kTailPairsPerTok;                  // row pair -> rows 2pr, 2pr+1

    // residual RMW issued first: its latency is spent under fold/publish/poll
    __nv_bfloat162* const d = reinterpret_cast<__nv_bfloat162*>(fuse_x8) + gid;
    const __nv_bfloat162 rv = *d;

    // Eleven-slot fold in fixed slot order.
    float2 v[kSlots];
#pragma unroll
    for (int s = 0; s < kSlots; ++s) {
        v[s] = *reinterpret_cast<const float2*>(l2p + (static_cast<size_t>(t) * kSlots + s) * kDim +
                                                2 * pr);
    }
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int s = 0; s < kSlots; ++s) {
        a0 += v[s].x;
        a1 += v[s].y;
    }
    const __nv_bfloat162 pv = __float22bfloat162_rn(make_float2(a0, a1));
    const unsigned payload = *reinterpret_cast<const unsigned*>(&pv);
    const unsigned long long pkt_self = (static_cast<unsigned long long>(epoch) << 32) | payload;

    // ONE publication into [parity][rank][gid] of EVERY rank's inbox.
    if (num_ranks > 1) {
        ll_mc_st_pkt(
            inbox_mc +
                ((static_cast<unsigned long long>(parity) * num_ranks + rank) * kPktPerSrc + gid) *
                    8ull,
            pkt_self);
    }

    const unsigned long long* const mybox =
        reinterpret_cast<const unsigned long long*>(inbox_ptrs[rank]) +
        static_cast<unsigned long long>(parity) * num_ranks * kPktPerSrc;
    const unsigned peer_full =
        ((num_ranks >= 32) ? 0xffffffffu : ((1u << num_ranks) - 1u)) & ~(1u << rank);

    // JOINT poll: one round issues every peer load independently.
    unsigned long long pk[kMaxRanks];
#pragma unroll
    for (int r = 0; r < kMaxRanks; ++r) pk[r] = 0ull;
    unsigned okm = 0u;  // peers whose packet is fresh
    int spins = 0;
    while (true) {
        okm = 0u;
#pragma unroll
        for (int r = 0; r < kMaxRanks; ++r) {
            if (r < num_ranks && r != rank) {
                pk[r] = ll_ld_pkt(mybox + static_cast<unsigned long long>(r) * kPktPerSrc + gid);
                okm |= (static_cast<unsigned>(pk[r] >> 32) == epoch) ? (1u << r) : 0u;
            }
        }
        if (okm == peer_full) break;
        if (++spins > kMaxSpins) {         // fail-closed: unmatched
            atomicCAS(fuse_status, 0, 1);  // sources fold exactly 0
            break;
        }
        __nanosleep(64);
    }

    // fixed ascending source order, self at index rank (bitwise == directed-scout)
    float r0 = 0.f, r1 = 0.f;
#pragma unroll
    for (int s = 0; s < kMaxRanks; ++s) {
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

}  // namespace nexrt::sm90::nex_n2_5_pro::M8
using namespace nexrt::sm90::nex_n2_5_pro::M8;
extern "C" {

int NexN25ProM8MoeL2LocalLaunch(const void* plan, const void* packed8, const void* rawh8,
                                const void* w2_raw, const void* w2_scales,
                                const void* shared_w2_raw, const void* shared_w2_scales, void* l2p,
                                void* stream) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(kTcGrid);
    cfg.blockDim = dim3(kTcThreads);
    cfg.dynamicSmemBytes = kTcDyn;
    cfg.stream = static_cast<cudaStream_t>(stream);
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    return static_cast<int>(cudaLaunchKernelEx(
        &cfg, M8MoeL2LocalKernel, static_cast<const M8RoutePlan*>(plan),
        static_cast<const int32_t*>(packed8), static_cast<const float*>(rawh8),
        static_cast<const __nv_fp8_e4m3*>(w2_raw), static_cast<const float*>(w2_scales),
        static_cast<const __nv_fp8_e4m3*>(shared_w2_raw),
        static_cast<const float*>(shared_w2_scales), static_cast<float*>(l2p)));
}

int NexN25ProM8MoeL2TailLlLaunch(const void* l2p, void* cur_gen, const void* inbox_ptrs,
                                 unsigned long long inbox_mc, void* ll_ctr, int rank, int num_ranks,
                                 void* status, void* x8_resid, void* stream) {
    if (num_ranks <= 0 || num_ranks > kMaxRanks || inbox_ptrs == nullptr || l2p == nullptr ||
        cur_gen == nullptr || ll_ctr == nullptr || x8_resid == nullptr ||
        status == nullptr  // fail-closed needs it
        || rank < 0 || rank >= num_ranks || (inbox_mc == 0ull && num_ranks > 1)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    M8MoeL2TailLlKernel<<<kTailLlCtas, kTailLlThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(l2p), static_cast<int*>(cur_gen),
        static_cast<const unsigned long long*>(inbox_ptrs), inbox_mc,
        static_cast<unsigned int*>(ll_ctr), rank, num_ranks, static_cast<int*>(status),
        static_cast<__nv_bfloat16*>(x8_resid));
    return static_cast<int>(cudaGetLastError());
}

int NexN25ProM8MoeL2PartialFloats() { return kM8 * kSlots * kDim; }
long long NexN25ProM8MoeL2InboxBytes() {
    return 2ll * kMaxRanks * kPktPerSrc * 8ll;  // 2 MiB per rank
}
int NexN25ProM8MoeL2TailCtrInts() { return kTailCtrInts; }
int NexN25ProM8MoeL2LocalCtas() { return kTcGrid; }
int NexN25ProM8MoeL2LocalSmem() { return kTcDyn; }
int NexN25ProM8MoeL2LocalBlocksPerSm() {
    int blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, M8MoeL2LocalKernel, kTcThreads, kTcDyn);
    return blocks;
}
}  // extern "C"
