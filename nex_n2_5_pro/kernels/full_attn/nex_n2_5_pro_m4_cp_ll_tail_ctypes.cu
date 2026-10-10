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

// M=4 CP tail: local attention combine, owner-directed push, and four-rank
// LSE merge using 8-byte payload/generation packets.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M4
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace nexrt::sm90::nex_n2_5_pro::M4 {
constexpr int RANKS = 4, ROWS = 4, OHEADS = 4, HD = 256;
constexpr int CPKT = HD / 4;          // 64 groups of two bf16-pair packets
constexpr int PKT_PER_HR = CPKT + 2;  // padded storage for the LSE packet
constexpr int DT8 = 8, NHCP = 16, CBLK = 256;

// Parity, device epochs, and optional gating.
// Graph-capture-safe forms of the LL tail for the CP integration module:
//  - inbox gains a PARITY dim [2][src][row][oh][PKT_PER_HR] (one-step cross-
//    rank skew tolerance: rank A's produce N+1 may run before rank B's merge N
//    consumed parity (N&1); N+1 writes the other parity).
//  - gen comes from a rank-local device epoch word (ENTRY-sampled by every
//    CTA so one launch never mixes generations); the last merge CTA advances
//    it after all merge CTAs complete. Cross-rank gen agreement holds by the lockstep
//    per-layer launch sequence; nothing host-side changes across replays.
//  - the merge optionally emits the gate-once x_eff epilogue: xeff[d] =
//    bf16(f32(bf16(merged)) * sigmoid(f32(gate[d]))) -- reading back the
//    ROUNDED bf16 (the exact-boundary contract), so the o_proj stages
//    precomputed values bit-identical to what it would have computed itself.

__device__ __forceinline__ size_t pk_off_p(int par, int src, int row, int oh, int pkt) {
    return ((((static_cast<size_t>(par) * RANKS + src) * ROWS + row) * OHEADS + oh) * PKT_PER_HR +
            pkt);
}
__device__ __forceinline__ float sigmoid_ll(float x) { return 1.0f / (1.0f + __expf(-x)); }

// Pack one 32-bit payload and its generation into a single 64-bit access.
// System-scope loads and stores use the same packet granularity. Each data
// group occupies two packets; the LSE uses the first packet of its group.

__device__ __forceinline__ void st_rel_sys_b64(uint64_t* p, uint64_t v) {
    asm volatile("st.relaxed.sys.global.b64 [%0], %1;\n" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ uint64_t ld_rel_sys_b64(const uint64_t* p) {
    uint64_t v;
    asm volatile("ld.relaxed.sys.global.b64 %0, [%1];\n" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint64_t ll8_pack(uint32_t pay, uint32_t gen) {
    return static_cast<uint64_t>(pay) | (static_cast<uint64_t>(gen) << 32);
}

__global__ void cp_ll_combine_push_ep8(const float* __restrict__ ws_m,
                                       const float* __restrict__ ws_l,
                                       const __nv_bfloat16* __restrict__ ws_acc,
                                       uint64_t* __restrict__ inbox_ptrs,
                                       const int* __restrict__ epoch, int my_rank, int cp_base,
                                       int ns, float scale2) {
    const int o0 = blockIdx.x * DT8;
    const int h = o0 / HD, d0 = o0 % HD, row = blockIdx.y, tid = threadIdx.x;
    const int u = row * NHCP + h;
    const uint32_t gen = static_cast<uint32_t>(epoch[0]) + 1u;  // entry-sampled
    const int par = static_cast<int>(gen & 1u);
    extern __shared__ __align__(16) char csm[];
    __nv_bfloat16* sm = reinterpret_cast<__nv_bfloat16*>(csm);
    float* m_sh = reinterpret_cast<float*>(sm + ns * DT8);
    float* l_sh = m_sh + ns;
    __shared__ float red[CBLK], den_red[CBLK];
    __shared__ float out_sh[DT8];
    __shared__ float lse_sh;
    for (int s = tid; s < ns; s += CBLK)
        *reinterpret_cast<uint4*>(sm + s * DT8) =
            *reinterpret_cast<const uint4*>(ws_acc + (static_cast<size_t>(u) * ns + s) * HD + d0);
    for (int s = tid; s < ns; s += CBLK) {
        m_sh[s] = ws_m[static_cast<size_t>(u) * ns + s];
        l_sh[s] = ws_l[static_cast<size_t>(u) * ns + s];
    }
    __syncthreads();
    float lm = -1.0e30f;
    for (int s = tid; s < ns; s += CBLK) lm = fmaxf(lm, m_sh[s]);
#pragma unroll
    for (int o = 16; o; o >>= 1) lm = fmaxf(lm, __shfl_xor_sync(~0u, lm, o));
    if ((tid & 31) == 0) red[tid >> 5] = lm;
    __syncthreads();
    float gm = -1.0e30f;
#pragma unroll
    for (int i = 0; i < CBLK / 32; ++i) gm = fmaxf(gm, red[i]);
    const int d = tid % DT8, group = tid / DT8;
    float num = 0.0f, den = 0.0f;
    for (int s = group; s < ns; s += CBLK / DT8) {
        const float w = exp2f((m_sh[s] - gm) * scale2) * l_sh[s];
        num += w * __bfloat162float(sm[s * DT8 + d]);
        den += w;
    }
    __syncthreads();
    red[tid] = num;
    den_red[tid] = den;
    __syncthreads();
    if (tid < DT8) {
        float n = 0.0f, z = 0.0f;
#pragma unroll
        for (int g = 0; g < CBLK / DT8; ++g) {
            n += red[g * DT8 + tid];
            z += den_red[g * DT8 + tid];
        }
        out_sh[tid] = n / (z + 1.0e-20f);
        if (tid == 0) lse_sh = (gm * scale2 + log2f(z + 1.0e-20f)) * 0.6931471805599453f;
    }
    __syncthreads();
    const int owner = h / OHEADS, oh = h % OHEADS;
    uint64_t* base = reinterpret_cast<uint64_t*>(inbox_ptrs[cp_base + owner]);
    if (tid < 4) {
        const int pk = d0 / 4 + (tid >> 1);
        const int hf = tid & 1;
        __nv_bfloat162 pr = {__float2bfloat16(out_sh[(tid >> 1) * 4 + hf * 2 + 0]),
                             __float2bfloat16(out_sh[(tid >> 1) * 4 + hf * 2 + 1])};
        st_rel_sys_b64(base + pk_off_p(par, my_rank, row, oh, pk) * 2 + hf,
                       ll8_pack(*reinterpret_cast<uint32_t*>(&pr), gen));
    }
    if (d0 == 0 && tid == 0) {
        st_rel_sys_b64(base + pk_off_p(par, my_rank, row, oh, CPKT) * 2,
                       ll8_pack(__float_as_uint(lse_sh), gen));
    }
}

__global__ void cp_ll_tail_merge_ep8(
    uint64_t* __restrict__ inbox_ptrs, int* __restrict__ epoch,
    unsigned int* __restrict__ done_ctr,  // rank-LOCAL monotone counter
    int my_rank, int cp_base, __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ gate,  // optional (gate-once)
    __nv_bfloat16* __restrict__ xeff,        // optional BF16 [R][OH][HD]
    unsigned long long* __restrict__ dbg) {
    const int row = blockIdx.x / OHEADS, oh = blockIdx.x % OHEADS;
    const int tid = threadIdx.x;
    const uint32_t gen = static_cast<uint32_t>(epoch[0]) + 1u;  // entry-sampled
    const int par = static_cast<int>(gen & 1u);
    const unsigned long long t0 = clock64();
    uint64_t* base = reinterpret_cast<uint64_t*>(inbox_ptrs[cp_base + my_rank]);
    __shared__ float s_lse[RANKS];
    if (tid < RANKS) {
        uint64_t v;
        do {
            v = ld_rel_sys_b64(base + pk_off_p(par, tid, row, oh, CPKT) * 2);
        } while (static_cast<uint32_t>(v >> 32) != gen);
        s_lse[tid] = __uint_as_float(static_cast<uint32_t>(v));
    }
    __syncthreads();
    const unsigned long long t1 = clock64();
    float m = fmaxf(fmaxf(s_lse[0], s_lse[1]), fmaxf(s_lse[2], s_lse[3]));
    float w[RANKS], den = 0.0f;
#pragma unroll
    for (int s = 0; s < RANKS; ++s) {
        w[s] = __expf(s_lse[s] - m);
        den += w[s];
    }
    const float inv = 1.0f / den;
    const int pkt = tid >> 1, half = tid & 1;
    float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
    for (int s = 0; s < RANKS; ++s) {
        uint64_t v;
        do {
            v = ld_rel_sys_b64(base + pk_off_p(par, s, row, oh, pkt) * 2 + half);
        } while (static_cast<uint32_t>(v >> 32) != gen);
        const uint32_t payw = static_cast<uint32_t>(v);
        const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(&payw);
        acc0 += w[s] * __bfloat162float(b2.x);
        acc1 += w[s] * __bfloat162float(b2.y);
    }
    const int d = pkt * 4 + half * 2;
    const size_t o0 = (static_cast<size_t>(row) * OHEADS + oh) * HD + d;
    const __nv_bfloat16 r0 = __float2bfloat16(acc0 * inv);
    const __nv_bfloat16 r1 = __float2bfloat16(acc1 * inv);
    out[o0] = r0;
    out[o0 + 1] = r1;
    if (xeff) {  // Match the non-CP combine's BF16 input to the shared O-projection.
        xeff[o0] = __float2bfloat16(__bfloat162float(r0) * sigmoid_ll(__bfloat162float(gate[o0])));
        xeff[o0 + 1] =
            __float2bfloat16(__bfloat162float(r1) * sigmoid_ll(__bfloat162float(gate[o0 + 1])));
    }
    if (dbg && tid == 0) {
        atomicMax(dbg, static_cast<unsigned long long>(t1 - t0));
        atomicMax(dbg + 1, static_cast<unsigned long long>(clock64() - t0));
    }
    __syncthreads();
    // Epoch advance must be ordered AFTER every CTA's entry read (a late CTA
    // reading an advanced epoch would wait for packets that never come). The
    // LAST of the 16 CTAs to finish advances: rank-local monotone counter,
    // no reset, no cross-rank traffic (entry-sample rule respected).
    if (tid == 0) {
        if (atomicAdd(done_ctr, 1u) + 1u == static_cast<unsigned int>(ROWS * OHEADS) * gen) {
            epoch[0] = static_cast<int>(gen);
        }
    }
}

extern "C" int CpLLCombinePushEpoch8Launch(const void* ws_m, const void* ws_l, const void* ws_acc,
                                           void* inbox_ptrs, const void* epoch, int my_rank,
                                           int cp_base, int ns, float scale2, void* stream) {
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    const int smem = ns * DT8 * 2 + 2 * ns * 4;
    dim3 g(NHCP * HD / DT8, ROWS), b(CBLK);
    cp_ll_combine_push_ep8<<<g, b, smem, st>>>(
        static_cast<const float*>(ws_m), static_cast<const float*>(ws_l),
        static_cast<const __nv_bfloat16*>(ws_acc), static_cast<uint64_t*>(inbox_ptrs),
        static_cast<const int*>(epoch), my_rank, cp_base, ns, scale2);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int CpLLTailMergeEpoch8Launch(void* inbox_ptrs, void* epoch, void* done_ctr, int my_rank,
                                         int cp_base, void* out, const void* gate, void* xeff,
                                         void* dbg, void* stream) {
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    cp_ll_tail_merge_ep8<<<ROWS * OHEADS, 128, 0, st>>>(
        static_cast<uint64_t*>(inbox_ptrs), static_cast<int*>(epoch),
        static_cast<unsigned int*>(done_ctr), my_rank, cp_base, static_cast<__nv_bfloat16*>(out),
        static_cast<const __nv_bfloat16*>(gate), static_cast<__nv_bfloat16*>(xeff),
        static_cast<unsigned long long*>(dbg));
    return static_cast<int>(cudaGetLastError());
}

}  // namespace nexrt::sm90::nex_n2_5_pro::M4
