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

// Bridge handoff kernels: sglang token-pool <-> nexrt dense caches. Complete requests use one
// fused gather; streamed prefill uses one KV append launch per completed chunk plus one finalize.
//
// The naive torch handoff is ~200 host dispatches (8-9 ms at S=512, pure overhead). These
// kernels operate directly on boot-time device pointer tables:
//   gather  (sglang -> nexrt): k/v_buffer[L][idx[r]] -> k/v_cache[L][r] for 15 full-attn
//            layers, mamba conv/temporal[slot] -> conv_state4/state_t for 45 GDN layers,
//            writes the [token_id, S] slot and zeroes every block's status word.
//   append  (sglang -> nexrt): only k/v_buffer[L][idx[r]] -> k/v_cache[L][dst+r] for a
//            newly completed prefill chunk. GDN state and the decode slot are finalized
//            separately after the last chunk so slot.S never exposes an incomplete row.
//   scatter (nexrt -> sglang): cache rows [row_start, row_start+n) -> pool rows idx[0..n),
//            conv_state4/state_t -> mamba conv/temporal[slot] (the multi-turn hand-back).
//
// Layout facts this relies on (byte-verified by the bridge work):
//   - one KV row is 256 bf16 = 512 B in BOTH the pool ([slots,1,256]) and the cache ([S,256]);
//   - mamba temporal[slot] [8,128,128] fp32 == nexrt state_t (no transpose);
//   - mamba conv[slot] [1536,3] bf16 == nexrt conv_state4[:, 0:3] (col 3 zeroed on gather).
//
// Pointer tables (int64 device tensors, built once at boot; all addresses process-stable):
//   kv_table [4*n_kv]  : srcK[0..n) | srcV[0..n) | dstK[0..n) | dstV[0..n)
//   gdn_table [4*n_gdn]: convPool[0..n) | temporalPool[0..n) | conv4[0..n) | state_t[0..n)
//     (pool entries are the per-layer VIEW bases; the kernel adds slot * fixed stride)
//   status_table [n_status]: int32* words to zero on gather (engine reset)
#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY common
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace nexrt::sm90::nex_n2_5_pro::common {
constexpr int kRowBytes = 512;                             // one KV row (256 bf16)
constexpr int kRowVecs = kRowBytes / 16;                   // 32 x uint4
constexpr int kRowsPerCta = 128;                           // 8 warps x 16 rows
constexpr long long kTemporalBytes = 8LL * 128 * 128 * 4;  // 512 KiB fp32 per layer state
constexpr int kTemporalChunkBytes = 128 * 1024;            // 4 CTAs per temporal state
constexpr int kTemporalChunks = kTemporalBytes / kTemporalChunkBytes;
constexpr int kConvRows = 1536;
constexpr long long kConvBytes = kConvRows * 3 * 2;  // [1536,3] bf16 = 9216 B

__device__ __forceinline__ void copy_row_512(const uint4* __restrict__ src, uint4* __restrict__ dst,
                                             int lane) {
    dst[lane] = src[lane];  // 32 lanes x 16 B = 512 B
}

// blockIdx decode, shared by both kernels:
//   [0, kvCtas)                        : KV tensors; tensor t = b / ctasPerTensor (0..2*n_kv),
//                                        t < n_kv -> K[t], else V[t - n_kv]
//   [kvCtas, +n_gdn*kTemporalChunks)   : temporal chunks
//   [.., +n_gdn)                       : conv states
//   last CTA (gather only)             : slot + statuses
__global__ void BridgeGatherKernel(const int64_t* __restrict__ kv_table, int n_kv,
                                   const int32_t* __restrict__ idx, int s_len,
                                   const int64_t* __restrict__ gdn_table, int n_gdn, int mamba_slot,
                                   int first_token, int32_t* __restrict__ slot,
                                   const int64_t* __restrict__ status_table, int n_status) {
    const int ctas_per_tensor = (s_len + kRowsPerCta - 1) / kRowsPerCta;
    const int kv_ctas = 2 * n_kv * ctas_per_tensor;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    int b = blockIdx.x;

    if (b < kv_ctas) {
        const int tensor = b / ctas_per_tensor;
        const int chunk = b - tensor * ctas_per_tensor;
        const bool is_v = tensor >= n_kv;
        const int layer = is_v ? tensor - n_kv : tensor;
        const char* src = reinterpret_cast<const char*>(kv_table[(is_v ? n_kv : 0) + layer]);
        char* dst = reinterpret_cast<char*>(kv_table[(is_v ? 3 * n_kv : 2 * n_kv) + layer]);
        const int row0 = chunk * kRowsPerCta;
#pragma unroll
        for (int i = 0; i < kRowsPerCta / 8; ++i) {  // 8 warps -> 16 rows each
            const int row = row0 + i * 8 + warp;
            if (row < s_len) {
                const long long src_row = static_cast<long long>(idx[row]) * kRowBytes;
                copy_row_512(
                    reinterpret_cast<const uint4*>(src + src_row),
                    reinterpret_cast<uint4*>(dst + static_cast<long long>(row) * kRowBytes), lane);
            }
        }
        return;
    }
    b -= kv_ctas;

    if (b < n_gdn * kTemporalChunks) {
        const int layer = b / kTemporalChunks;
        const int chunk = b - layer * kTemporalChunks;
        const char* src = reinterpret_cast<const char*>(gdn_table[n_gdn + layer]) +
                          static_cast<long long>(mamba_slot) * kTemporalBytes +
                          static_cast<long long>(chunk) * kTemporalChunkBytes;
        char* dst = reinterpret_cast<char*>(gdn_table[3 * n_gdn + layer]) +
                    static_cast<long long>(chunk) * kTemporalChunkBytes;
        const uint4* s4 = reinterpret_cast<const uint4*>(src);
        uint4* d4 = reinterpret_cast<uint4*>(dst);
        for (int i = threadIdx.x; i < kTemporalChunkBytes / 16; i += blockDim.x) {
            d4[i] = s4[i];
        }
        return;
    }
    b -= n_gdn * kTemporalChunks;

    if (b < n_gdn) {
        // pool conv [1536,3] bf16 -> conv_state4 [1536,4], col 3 zeroed
        const __nv_bfloat16* src =
            reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const char*>(gdn_table[b]) +
                                                   static_cast<long long>(mamba_slot) * kConvBytes);
        __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(gdn_table[2 * n_gdn + b]);
        for (int r = threadIdx.x; r < kConvRows; r += blockDim.x) {
            dst[r * 4 + 0] = src[r * 3 + 0];
            dst[r * 4 + 1] = src[r * 3 + 1];
            dst[r * 4 + 2] = src[r * 3 + 2];
            dst[r * 4 + 3] = __nv_bfloat16(0.f);
        }
        return;
    }
    b -= n_gdn;

    if (b == 0) {  // slot + engine status reset
        if (threadIdx.x == 0) {
            slot[0] = first_token;
            slot[1] = s_len;
        }
        for (int i = threadIdx.x; i < n_status; i += blockDim.x) {
            *reinterpret_cast<int32_t*>(status_table[i]) = 0;
        }
    }
}

// Incremental prefill append. idx contains only the newly completed logical rows and the
// destination is the corresponding contiguous range in nexrt's cache. No GDN or control
// state is touched here: callers launch BridgeFinalizeKernel after the final chunk.
__global__ void BridgeAppendKvKernel(const int64_t* __restrict__ kv_table, int n_kv,
                                     const int32_t* __restrict__ idx, int n_rows, int dst_start) {
    const int ctas_per_tensor = (n_rows + kRowsPerCta - 1) / kRowsPerCta;
    const int kv_ctas = 2 * n_kv * ctas_per_tensor;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    int b = blockIdx.x;

    if (b >= kv_ctas) return;
    const int tensor = b / ctas_per_tensor;
    const int chunk = b - tensor * ctas_per_tensor;
    const bool is_v = tensor >= n_kv;
    const int layer = is_v ? tensor - n_kv : tensor;
    const char* src = reinterpret_cast<const char*>(kv_table[(is_v ? n_kv : 0) + layer]);
    char* dst = reinterpret_cast<char*>(kv_table[(is_v ? 3 * n_kv : 2 * n_kv) + layer]);
    const int row0 = chunk * kRowsPerCta;
#pragma unroll
    for (int i = 0; i < kRowsPerCta / 8; ++i) {
        const int row = row0 + i * 8 + warp;
        if (row < n_rows) {
            const long long src_row = static_cast<long long>(idx[row]) * kRowBytes;
            const long long dst_row = static_cast<long long>(dst_start + row) * kRowBytes;
            copy_row_512(reinterpret_cast<const uint4*>(src + src_row),
                         reinterpret_cast<uint4*>(dst + dst_row), lane);
        }
    }
}

// Finalize a streamed prefill handoff without recopying KV. GDN state is per request rather
// than per token, so it is copied once from the final SGLang mamba slot. The last CTA writes
// the decode feedback slot and clears nexrt's sticky status words.
__global__ void BridgeFinalizeKernel(const int64_t* __restrict__ gdn_table, int n_gdn,
                                     int mamba_slot, int first_token, int s_len,
                                     int32_t* __restrict__ slot,
                                     const int64_t* __restrict__ status_table, int n_status) {
    int b = blockIdx.x;
    if (b < n_gdn * kTemporalChunks) {
        const int layer = b / kTemporalChunks;
        const int chunk = b - layer * kTemporalChunks;
        const char* src = reinterpret_cast<const char*>(gdn_table[n_gdn + layer]) +
                          static_cast<long long>(mamba_slot) * kTemporalBytes +
                          static_cast<long long>(chunk) * kTemporalChunkBytes;
        char* dst = reinterpret_cast<char*>(gdn_table[3 * n_gdn + layer]) +
                    static_cast<long long>(chunk) * kTemporalChunkBytes;
        const uint4* s4 = reinterpret_cast<const uint4*>(src);
        uint4* d4 = reinterpret_cast<uint4*>(dst);
        for (int i = threadIdx.x; i < kTemporalChunkBytes / 16; i += blockDim.x) {
            d4[i] = s4[i];
        }
        return;
    }
    b -= n_gdn * kTemporalChunks;

    if (b < n_gdn) {
        const __nv_bfloat16* src =
            reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const char*>(gdn_table[b]) +
                                                   static_cast<long long>(mamba_slot) * kConvBytes);
        __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(gdn_table[2 * n_gdn + b]);
        for (int r = threadIdx.x; r < kConvRows; r += blockDim.x) {
            dst[r * 4 + 0] = src[r * 3 + 0];
            dst[r * 4 + 1] = src[r * 3 + 1];
            dst[r * 4 + 2] = src[r * 3 + 2];
            dst[r * 4 + 3] = __nv_bfloat16(0.f);
        }
        return;
    }
    b -= n_gdn;

    if (b == 0) {
        if (threadIdx.x == 0) {
            slot[0] = first_token;
            slot[1] = s_len;
        }
        for (int i = threadIdx.x; i < n_status; i += blockDim.x) {
            *reinterpret_cast<int32_t*>(status_table[i]) = 0;
        }
    }
}

__global__ void BridgeScatterKernel(const int64_t* __restrict__ kv_table, int n_kv,
                                    const int32_t* __restrict__ idx, int n_rows, int row_start,
                                    const int64_t* __restrict__ gdn_table, int n_gdn,
                                    int mamba_slot) {
    const int ctas_per_tensor = (n_rows + kRowsPerCta - 1) / kRowsPerCta;
    const int kv_ctas = 2 * n_kv * ctas_per_tensor;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    int b = blockIdx.x;

    if (b < kv_ctas) {
        const int tensor = b / ctas_per_tensor;
        const int chunk = b - tensor * ctas_per_tensor;
        const bool is_v = tensor >= n_kv;
        const int layer = is_v ? tensor - n_kv : tensor;
        // roles swapped vs gather: cache rows [row_start+r] -> pool rows idx[r]
        char* pool = reinterpret_cast<char*>(kv_table[(is_v ? n_kv : 0) + layer]);
        const char* cache =
            reinterpret_cast<const char*>(kv_table[(is_v ? 3 * n_kv : 2 * n_kv) + layer]);
        const int row0 = chunk * kRowsPerCta;
#pragma unroll
        for (int i = 0; i < kRowsPerCta / 8; ++i) {
            const int row = row0 + i * 8 + warp;
            if (row < n_rows) {
                const long long dst_row = static_cast<long long>(idx[row]) * kRowBytes;
                copy_row_512(reinterpret_cast<const uint4*>(
                                 cache + static_cast<long long>(row_start + row) * kRowBytes),
                             reinterpret_cast<uint4*>(pool + dst_row), lane);
            }
        }
        return;
    }
    b -= kv_ctas;

    if (b < n_gdn * kTemporalChunks) {
        const int layer = b / kTemporalChunks;
        const int chunk = b - layer * kTemporalChunks;
        const char* src = reinterpret_cast<const char*>(gdn_table[3 * n_gdn + layer]) +
                          static_cast<long long>(chunk) * kTemporalChunkBytes;
        char* dst = reinterpret_cast<char*>(gdn_table[n_gdn + layer]) +
                    static_cast<long long>(mamba_slot) * kTemporalBytes +
                    static_cast<long long>(chunk) * kTemporalChunkBytes;
        const uint4* s4 = reinterpret_cast<const uint4*>(src);
        uint4* d4 = reinterpret_cast<uint4*>(dst);
        for (int i = threadIdx.x; i < kTemporalChunkBytes / 16; i += blockDim.x) {
            d4[i] = s4[i];
        }
        return;
    }
    b -= n_gdn * kTemporalChunks;

    if (b < n_gdn) {  // conv_state4[:, 0:3] -> pool conv [1536,3]
        const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(gdn_table[2 * n_gdn + b]);
        __nv_bfloat16* dst =
            reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<char*>(gdn_table[b]) +
                                             static_cast<long long>(mamba_slot) * kConvBytes);
        for (int r = threadIdx.x; r < kConvRows; r += blockDim.x) {
            dst[r * 3 + 0] = src[r * 4 + 0];
            dst[r * 3 + 1] = src[r * 4 + 1];
            dst[r * 3 + 2] = src[r * 4 + 2];
        }
    }
}

int grid_size(int n_kv, int n_gdn, int rows, bool gather) {
    const int ctas_per_tensor = (rows + kRowsPerCta - 1) / kRowsPerCta;
    return 2 * n_kv * ctas_per_tensor + n_gdn * kTemporalChunks + n_gdn + (gather ? 1 : 0);
}
int append_grid_size(int n_kv, int rows) {
    const int ctas_per_tensor = (rows + kRowsPerCta - 1) / kRowsPerCta;
    return 2 * n_kv * ctas_per_tensor;
}

int finalize_grid_size(int n_gdn) { return n_gdn * kTemporalChunks + n_gdn + 1; }
}  // namespace nexrt::sm90::nex_n2_5_pro::common
using namespace nexrt::sm90::nex_n2_5_pro::common;
extern "C" int nex_n2_5_pro_bridge_gather_launch(const void* kv_table, int n_kv, const void* idx,
                                                 int s_len, const void* gdn_table, int n_gdn,
                                                 int mamba_slot, int first_token, void* slot,
                                                 const void* status_table, int n_status,
                                                 void* stream) {
    if (s_len <= 0 || n_kv < 0 || n_gdn < 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    BridgeGatherKernel<<<grid_size(n_kv, n_gdn, s_len, true), 256, 0,
                         reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const int64_t*>(kv_table), n_kv, reinterpret_cast<const int32_t*>(idx),
        s_len, reinterpret_cast<const int64_t*>(gdn_table), n_gdn, mamba_slot, first_token,
        reinterpret_cast<int32_t*>(slot), reinterpret_cast<const int64_t*>(status_table), n_status);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int nex_n2_5_pro_bridge_append_launch(const void* kv_table, int n_kv, const void* idx,
                                                 int n_rows, int dst_start, void* stream) {
    if (n_rows <= 0 || n_kv <= 0 || dst_start < 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    BridgeAppendKvKernel<<<append_grid_size(n_kv, n_rows), 256, 0,
                           reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const int64_t*>(kv_table), n_kv, reinterpret_cast<const int32_t*>(idx),
        n_rows, dst_start);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int nex_n2_5_pro_bridge_finalize_launch(const void* gdn_table, int n_gdn, int mamba_slot,
                                                   int first_token, int s_len, void* slot,
                                                   const void* status_table, int n_status,
                                                   void* stream) {
    if (n_gdn < 0 || mamba_slot < 0 || first_token < 0 || s_len <= 0 || n_status < 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    BridgeFinalizeKernel<<<finalize_grid_size(n_gdn), 256, 0,
                           reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const int64_t*>(gdn_table), n_gdn, mamba_slot, first_token, s_len,
        reinterpret_cast<int32_t*>(slot), reinterpret_cast<const int64_t*>(status_table), n_status);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int nex_n2_5_pro_bridge_scatter_launch(const void* kv_table, int n_kv, const void* idx,
                                                  int n_rows, int row_start, const void* gdn_table,
                                                  int n_gdn, int mamba_slot, void* stream) {
    if (n_rows <= 0 || n_kv < 0 || n_gdn < 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    BridgeScatterKernel<<<grid_size(n_kv, n_gdn, n_rows, false), 256, 0,
                          reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const int64_t*>(kv_table), n_kv, reinterpret_cast<const int32_t*>(idx),
        n_rows, row_start, reinterpret_cast<const int64_t*>(gdn_table), n_gdn, mamba_slot);
    return static_cast<int>(cudaGetLastError());
}
