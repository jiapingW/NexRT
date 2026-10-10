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

// M=8-ONLY device control block: the canonical ABI shared by every M=8 kernel.
// One frozen int32[32] verification region (two 64 B lines, one allocation).
//
// SCOPE. This block belongs to the M=8 verification family ALONE. It does not
// replace, alias, or modify DecodeEngine.slot (the production BS1 [token_id, S]
// feedback slot), nor the M=4 control block (kernels/common/m4_control_block.cuh).
// The three families never share mutable control state.
//
// This ABI is VERIFY-ONLY by design: eight externally supplied target input ids
// in, one sampled-prefix commit decision out. There are deliberately NO draft
// chain slots, stop-token words, or serving-state words -- reserved words stay
// zero until a future freeze of THIS verification ABI assigns them.
//
// LAYOUT (int32 words; the whole allocation is 128 B = two 64 B lines):
//
//   word  0       pending_target_id      last emitted token; the next unit's t0
//   word  1       committed_len          C: absolute count of target rows committed
//   word  2       commit_count           1..8 rows committed by the last verification
//   word  3       output_count           number of valid emitted_ids entries (1..8)
//   words 4..11   verify_input_ids[8]    [t0..t7] fed to the eight target rows
//   words 12..19  emitted_ids[8]         verified output token stream for this unit
//   word  20      accepted_draft_count   0..7 == commit_count - 1
//   word  21      flags                  kFlag* bits
//   word  22      status                 0 == OK; nonzero is fail-closed
//   word  23      step_epoch             monotonic verification counter
//   words 24..31  reserved               zero until assigned by a future freeze
//
// WHY THIS LAYOUT:
//   * words 0/1 keep the familiar [token_id, S] prefix: every M=8 kernel that
//     needs the unit base position reads committed_len at word 1, exactly where
//     the engine slot keeps S. The two fields lead the block and must not move.
//   * verify_input_ids starts at byte offset 16 and emitted_ids at byte offset
//     48 -- both int4-aligned, so each eight-word array is two int4 loads.
//   * every hot dynamic field lives in ONE allocation, so no M=8 kernel needs a
//     CPU state update between graph replays.
//
// WRITER DISCIPLINE. committed_len has exactly ONE logical writer, the final
// M=8 sample/verify/commit CTA, which publishes it LAST (after a threadfence
// over the other decision words). A same-stream kernel boundary supplies
// visibility to every later M=8 kernel, so no atomic is required. Any nonzero
// status leaves committed_len unchanged (fail closed).
//
// FAILURE CONTRACT: a fail-closed decision publishes commit_count = 0, bumps
// step_epoch, and leaves committed_len untouched. A failed unit is RETRYABLE
// without reloading the request: both rings are sized so that a unit never
// overwrites any state its own re-execution would need (see RING SIZING), so
// the base recurrent state and the convolution history both survive intact.
// The caller must still clear the sticky protocol status latches before the
// retry -- see M8Verify.reset_unit() -- because those are level-triggered and
// would otherwise fail the next unit closed as well.
//
// RING SIZING (the no-alias law). Within one unit the live logical positions
// are the base plus the eight rows. A ring is safe exactly when it has at
// least that many slots, because then the live span maps to distinct residues:
//   * recurrent state: base (C-1) + rows C..C+7  = kM + 1  = 9 positions
//   * convolution:     history (C-3..C-1) + rows = kM + 3  = 11 positions
// Both bounds are asserted below rather than left to comment prose. Positions
// are always >= 0 here: init_control_block refuses committed_len < kConvHistory,
// so the smallest index ever formed is committed_len - kConvHistory >= 0.

#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY M8
#endif

#pragma once

#include <cstdint>

namespace nexrt::sm90::nex_n2_5_pro::M8 {
namespace m8ctl {

// ---- canonical word offsets (mirrored by m8_control.py; drift is an import-time error) ----
constexpr int kPendingTargetId = 0;
constexpr int kCommittedLen = 1;
constexpr int kCommitCount = 2;
constexpr int kOutputCount = 3;
constexpr int kVerifyInputIds = 4;  // [8]
constexpr int kEmittedIds = 12;     // [8]
constexpr int kAcceptedDraftCount = 20;
constexpr int kFlags = 21;
constexpr int kStatus = 22;
constexpr int kStepEpoch = 23;
constexpr int kReserved = 24;  // [8] zero until a future freeze
constexpr int kWords = 32;     // full allocation: two 64 B lines

constexpr int kM = 8;  // target rows per verification unit
// Width-4 causal convolution: three history positions before the unit base are
// read by the GDN front. Mirrored by m8_control.py CONV_HISTORY.
constexpr int kConvHistory = 3;
// Conv history ring: positions C-3 .. C+7 (kM + kConvHistory = 11 logical
// positions) are simultaneously live inside one unit; 16 is the smallest power
// of two that holds them without aliasing, so the three history taps survive
// the unit intact and a FAILED unit can be retried against them directly.
constexpr int kConvRingSlots = 16;  // position p -> slot p & 15
// Recurrent state ring: the snapshot after position p lives at slot p % 9.
// Positions C-1 (base) .. C+7 are kM + 1 = 9 consecutive integers, so they take
// nine DISTINCT residues mod 9: no row ever overwrites the base. That is what
// makes a failed unit retryable -- committed_len stays C, and the base snapshot
// the retry reads is still the one written before this unit began.
// A successful unit committing k rows moves the base to C+k-1, whose slot row
// k-1 wrote during the unit, so partial acceptance still needs no copy.
// Nine is not a power of two; the modulo runs once per kernel entry and once
// per token against a 512 KiB slot stride, so it is not measurable.
constexpr int kStateRingSlots = 9;  // position p -> slot p % 9

// ---- flags ----
constexpr int kFlagVerified = 1 << 0;    // a verification decision was published this epoch
constexpr int kFlagFullAccept = 1 << 1;  // all seven drafts matched (commit_count == 8)

// ---- status bits (0 == OK; any nonzero bit means "do not commit") ----
constexpr int kStatusOk = 0;
constexpr int kStatusUpstreamFailed = 1 << 0;     // target forward / row reduction reported failure
constexpr int kStatusBadResolution = 1 << 1;      // a resolved row id was missing or out of range
constexpr int kStatusBadCommitCount = 1 << 2;     // commit_count escaped 1..8
constexpr int kStatusBadSamplingConfig = 1 << 3;  // temperature/top-p is invalid
constexpr int kStatusEmptySamplingMass = 1
                                         << 4;  // a row's retained sampling mass was empty/invalid

// ---- ABI assertions: the layout arguments above are checked, not assumed ----
static_assert(kPendingTargetId == 0 && kCommittedLen == 1,
              "words 0/1 must stay [token_id, S]-shaped for M=8 base-position readers");
static_assert(kVerifyInputIds % 4 == 0, "verify_input_ids must be int4-aligned");
static_assert(kEmittedIds % 4 == 0, "emitted_ids must be int4-aligned");
static_assert(kVerifyInputIds + kM == kEmittedIds, "verify_input_ids must abut emitted_ids");
static_assert(kEmittedIds + kM == kAcceptedDraftCount, "emitted_ids overlaps the tail scalars");
static_assert(kStepEpoch + 1 == kReserved, "step_epoch must be the last assigned word");
static_assert(kReserved + kM == kWords, "reserved region must pad the block to kWords");
static_assert(kWords * sizeof(int32_t) == 128, "the M=8 control block is exactly two 64 B lines");
// The no-alias law: a ring must hold every logical position live inside one
// unit, or a row would clobber state that unit's own retry still needs.
static_assert(kStateRingSlots >= kM + 1,
              "recurrent ring must hold the base plus all kM row snapshots");
static_assert(kConvRingSlots >= kM + kConvHistory,
              "conv ring must hold the history taps plus all kM row inputs");

// ---- accessors: kernels use these instead of open-coded indices ----
__device__ __forceinline__ int Load(const int* __restrict__ ctl, int word) { return ctl[word]; }
__device__ __forceinline__ void Store(int* __restrict__ ctl, int word, int v) { ctl[word] = v; }

__device__ __forceinline__ int CommittedLen(const int* __restrict__ ctl) {
    return ctl[kCommittedLen];
}
// Absolute input position of target row j for a unit starting at committed_len.
__device__ __forceinline__ int RowPosition(int committed_len, int j) { return committed_len + j; }
__device__ __forceinline__ int ConvSlot(int position) { return position & (kConvRingSlots - 1); }
// kStateRingSlots is 9, so this is a real modulo, not a mask. Positions are
// non-negative by the init_control_block precondition documented above.
__device__ __forceinline__ int StateSlot(int position) { return position % kStateRingSlots; }

}  // namespace m8ctl
}  // namespace nexrt::sm90::nex_n2_5_pro::M8