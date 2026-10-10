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

// M=4-ONLY device control block: the canonical ABI shared by every M=4 kernel.
// v1 = the frozen int32[16] verify region (words 0..15); v2 extends the SAME allocation
// to int32[32] (kExtWords) with the MTP-chain words documented below.
//
// SCOPE. This block belongs to the M=4 verification path ALONE. It does not replace,
// alias, or modify DecodeEngine.slot, which stays the production BS1 [token_id, S]
// feedback slot with its own kernels, sampler and captured graphs.
//
// LAYOUT of the FROZEN v1 verify region -- words 0..15 of what is now a 128 B (kExtWords)
// allocation; the v2 MTP-chain extension in words 16..31 is documented below:
//
//   word  0      pending_target_id      last emitted token; the next unit's t0
//   word  1      committed_len          C: absolute count of target rows committed
//   words 2      commit_count           1..4 rows committed by the last verification
//   word  3      output_count           number of valid emitted_ids entries (1..4)
//   words 4..7   verify_input_ids[4]    [t0,t1,t2,t3] fed to the four target rows
//   words 8..11  emitted_ids[4]         greedy output token stream for this unit
//   word  12     accepted_draft_count   0..3 == commit_count - 1
//   word  13     flags                  kM4CtlFlag* bits
//   word  14     status                 0 == OK; nonzero is fail-closed
//   word  15     step_epoch             monotonic verification counter
//
// WHY THIS LAYOUT:
//   * words 0/1 keep the familiar [token_id, S] prefix, so an M=4 helper that already
//     reads a two-word slot (embed gather, attention QKV/RoPE/core) can be handed
//     `ctl` unchanged and read committed_len exactly where it used to read S. That
//     reuse is the reason the two fields lead the block and must not be reordered.
//   * verify_input_ids and emitted_ids start at 16 B-aligned byte offsets 16 and 32,
//     so each is a single int4 load/store.
//   * every hot dynamic field lives in ONE allocation, so no M=4 kernel needs a CPU
//     state update between graph replays.
//
// WRITER DISCIPLINE. committed_len has exactly ONE logical writer, the final M=4
// verify/commit CTA, which publishes it LAST. A same-stream kernel boundary supplies
// visibility to every later M=4 kernel, so no atomic is required. Any nonzero status
// leaves committed_len unchanged (fail closed).

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M4
#endif

#pragma once

#include <cstdint>

namespace nexrt::sm90::nex_n2_5_pro::M4 {
namespace m4ctl {

// ---- canonical word offsets (mirrored by m4_control.py; drift is a build-time error) ----
constexpr int kPendingTargetId = 0;
constexpr int kCommittedLen = 1;
constexpr int kCommitCount = 2;
constexpr int kOutputCount = 3;
constexpr int kVerifyInputIds = 4;  // [4]
constexpr int kEmittedIds = 8;      // [4]
constexpr int kAcceptedDraftCount = 12;
constexpr int kFlags = 13;
constexpr int kStatus = 14;
constexpr int kStepEpoch = 15;
constexpr int kWords = 16;  // size of the FROZEN v1 verify region above

// ---- v2 extension: words 16..31 (second 64 B line of the SAME allocation; MTP-chain state) ----
//
// The v1 region (words 0..15) is FROZEN: offsets never move, v1 kernels keep reading it
// unchanged. The allocation now spans kExtWords words.
//
//   words 16..17  chain_slot_a   [token_id, S]  input slot for MTP draft-decode #1
//   words 18..19  chain_slot_b   [token_id, S]  input slot for MTP draft-decode #2
//   word  20      stop_pos       emitted_ids index of the first stop token this epoch, else -1
//   words 21..31  reserved       zero until assigned by a future freeze
//
// SLOT S CONVENTION (verbatim from DecodeEngine.slot, decode_engine.py): S is the number of
// rows already cached BEFORE the step. The consuming kernel processes its token at position S,
// writes cache row S, attends rows 0..S, and never bumps S itself. For the MTP chain each
// round (C = committed_len already advanced by the verify/commit CTA):
//   chain_slot_a = [d1, C  ]    -> draft-decode #1 writes MTP line C
//   chain_slot_b = [d2, C+1]    -> draft-decode #2 writes MTP line C+1
//
// WRITER DISCIPLINE, v2 words (single writer per field; same-stream kernel boundaries supply
// visibility, no atomics anywhere):
//   chain_slot_a         MtpDraftSelectExt (extend epilogue; also verify_input_ids[1] = d1)
//   chain_slot_b         MtpDraftSelectD1  (decode #1 epilogue; also verify_input_ids[2] = d2)
//   verify_input_ids[3]  MtpDraftSelectD2  (decode #2 epilogue; = d3)
//   verify_input_ids[0]  MtpPrep, post-verify (copy of pending_target_id)
//   stop_pos             MtpPrep, post-verify (with kFlagStopHit; -1 when no stop). NOTE
//                        flags therefore has TWO same-stream ordered writers (verify
//                        commit assigns, MtpPrep ORs the stop bit) -- ordered, never
//                        concurrent.
constexpr int kChainSlotA = 16;  // [2]  [token_id, S]
constexpr int kChainSlotB = 18;  // [2]  [token_id, S]
constexpr int kStopPos = 20;
constexpr int kExtWords = 32;  // full v2 allocation: two 64 B lines

constexpr int kM = 4;               // target rows per verification unit
constexpr int kConvRingSlots = 8;   // conv history ring depth: position p -> p & 7
constexpr int kStateRingSlots = 4;  // recurrent state ring depth: position p -> p & 3

// ---- flags ----
constexpr int kFlagVerified = 1 << 0;    // a verification decision was published this epoch
constexpr int kFlagFullAccept = 1 << 1;  // all three drafts matched (commit_count == 4)
constexpr int kFlagStopHit = 1 << 2;     // a stop token id landed in emitted_ids this epoch

// ---- status bits (0 == OK; any nonzero bit means "do not commit") ----
constexpr int kStatusOk = 0;
constexpr int kStatusUpstreamFailed = 1 << 0;     // target forward / row reduction reported failure
constexpr int kStatusBadResolution = 1 << 1;      // a resolved row id was missing or out of range
constexpr int kStatusBadCommitCount = 1 << 2;     // commit_count escaped 1..4
constexpr int kStatusBadSamplingConfig = 1 << 3;  // temperature/top-p is invalid
constexpr int kStatusEmptySamplingMass = 1
                                         << 4;  // a row's retained sampling mass was empty/invalid

// ---- ABI assertions: the reuse and alignment arguments above are checked, not assumed ----
static_assert(kPendingTargetId == 0 && kCommittedLen == 1,
              "words 0/1 must stay [token_id, S]-shaped for M=4 slot-reading helpers");
static_assert(kVerifyInputIds % 4 == 0, "verify_input_ids must be int4-aligned");
static_assert(kEmittedIds % 4 == 0, "emitted_ids must be int4-aligned");
static_assert(kVerifyInputIds + kM <= kEmittedIds, "verify_input_ids overlaps emitted_ids");
static_assert(kEmittedIds + kM <= kAcceptedDraftCount, "emitted_ids overlaps the tail scalars");
static_assert(kStepEpoch == kWords - 1, "step_epoch must be the last v1 word");
static_assert(kWords * sizeof(int32_t) == 64,
              "the FROZEN v1 verify region must stay one 64 B line");
static_assert(kChainSlotA == kWords, "the v2 extension begins exactly at the frozen v1 boundary");
static_assert(kChainSlotA % 2 == 0 && kChainSlotB == kChainSlotA + 2,
              "chain slots are adjacent int2-aligned [token_id, S] pairs");
static_assert(kStopPos == kChainSlotB + 2, "stop_pos follows the chain slots");
static_assert(kExtWords * sizeof(int32_t) == 128, "the v2 control block is exactly two 64 B lines");

// ---- accessors: kernels use these instead of open-coded indices ----
__device__ __forceinline__ int Load(const int* __restrict__ ctl, int word) { return ctl[word]; }
__device__ __forceinline__ void Store(int* __restrict__ ctl, int word, int v) { ctl[word] = v; }

__device__ __forceinline__ int CommittedLen(const int* __restrict__ ctl) {
    return ctl[kCommittedLen];
}
// Absolute input position of target row j for a unit starting at committed_len.
__device__ __forceinline__ int RowPosition(int committed_len, int j) { return committed_len + j; }
__device__ __forceinline__ int ConvSlot(int position) { return position & (kConvRingSlots - 1); }
__device__ __forceinline__ int StateSlot(int position) { return position & (kStateRingSlots - 1); }

}  // namespace m4ctl
}  // namespace nexrt::sm90::nex_n2_5_pro::M4