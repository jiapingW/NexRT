#!/usr/bin/env python3
# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""M=4-ONLY device control block: the Python side of the canonical ABI
(v1 = frozen int32[16] verify region; v2 = the int32[32] extension with the MTP-chain words).

The offsets below MIRROR ``kernels/common/m4_control_block.cuh``. They are not
maintained by hand on two sides: :func:`assert_abi_matches_cuda` parses the header at
import time and fails loudly if the two ever disagree, so a kernel-side edit cannot
silently desynchronize the Python launchers (or vice versa).

SCOPE. This block is M=4-only. ``DecodeEngine.slot`` -- the production BS1
``[token_id, S]`` feedback slot -- keeps its tensor, ABI, sampler, and captured
graphs; no M=4 kernel READS it. The M=4 commit kernels optionally WRITE a
``[token_id, S]`` compatibility mirror into it for harnesses that inspect the
engine slot. Words 0/1 of this block are *shaped* like that slot so M=4 helpers
which already read a two-word slot can be handed ``ctl`` unchanged, but the two
allocations are independent.

Static sampling configuration (temperature/top-p) and the uint64 draw seed deliberately
stay in their own typed device buffers; forcing fp32/u64 values through an int32 ABI to
save one allocation would trade a real type hazard for nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CUH = HERE / "kernels" / "common" / "m4_control_block.cuh"

# ---- canonical word offsets (mirror of m4_control_block.cuh) ----
PENDING_TARGET_ID = 0
COMMITTED_LEN = 1
COMMIT_COUNT = 2
OUTPUT_COUNT = 3
VERIFY_INPUT_IDS = 4  # [4]
EMITTED_IDS = 8  # [4]
ACCEPTED_DRAFT_COUNT = 12
FLAGS = 13
STATUS = 14
STEP_EPOCH = 15
WORDS = 16  # size of the FROZEN v1 verify region

# ---- v2 extension (words 16..31, second 64 B line of the same allocation) ----
# S convention (verbatim from DecodeEngine.slot): S = rows already cached BEFORE the step;
# the consuming kernel processes its token at position S, writes cache row S, attends 0..S.
CHAIN_SLOT_A = 16  # [2]: [token_id, S] for MTP draft-decode #1 = [d1, C]
CHAIN_SLOT_B = 18  # [2]: [token_id, S] for MTP draft-decode #2 = [d2, C+1]
STOP_POS = 20  # emitted_ids index of the first stop token this epoch, else -1
EXT_WORDS = 32  # full v2 allocation: two 64 B lines

M = 4  # target rows per verification unit
CONV_RING_SLOTS = 8  # conv history ring depth: position p -> p & 7
STATE_RING_SLOTS = 4  # recurrent state ring depth: position p -> p & 3

FLAG_VERIFIED = 1 << 0
FLAG_FULL_ACCEPT = 1 << 1
FLAG_STOP_HIT = 1 << 2

STATUS_OK = 0
STATUS_UPSTREAM_FAILED = 1 << 0
STATUS_BAD_RESOLUTION = 1 << 1
STATUS_BAD_COMMIT_COUNT = 1 << 2
STATUS_BAD_SAMPLING_CONFIG = 1 << 3
STATUS_EMPTY_SAMPLING_MASS = 1 << 4

_CUDA_NAMES = {
    "kPendingTargetId": "PENDING_TARGET_ID",
    "kCommittedLen": "COMMITTED_LEN",
    "kCommitCount": "COMMIT_COUNT",
    "kOutputCount": "OUTPUT_COUNT",
    "kVerifyInputIds": "VERIFY_INPUT_IDS",
    "kEmittedIds": "EMITTED_IDS",
    "kAcceptedDraftCount": "ACCEPTED_DRAFT_COUNT",
    "kFlags": "FLAGS",
    "kStatus": "STATUS",
    "kStepEpoch": "STEP_EPOCH",
    "kWords": "WORDS",
    "kChainSlotA": "CHAIN_SLOT_A",
    "kChainSlotB": "CHAIN_SLOT_B",
    "kStopPos": "STOP_POS",
    "kExtWords": "EXT_WORDS",
    "kM": "M",
    "kConvRingSlots": "CONV_RING_SLOTS",
    "kStateRingSlots": "STATE_RING_SLOTS",
    "kFlagVerified": "FLAG_VERIFIED",
    "kFlagFullAccept": "FLAG_FULL_ACCEPT",
    "kFlagStopHit": "FLAG_STOP_HIT",
    "kStatusOk": "STATUS_OK",
    "kStatusUpstreamFailed": "STATUS_UPSTREAM_FAILED",
    "kStatusBadResolution": "STATUS_BAD_RESOLUTION",
    "kStatusBadCommitCount": "STATUS_BAD_COMMIT_COUNT",
    "kStatusBadSamplingConfig": "STATUS_BAD_SAMPLING_CONFIG",
    "kStatusEmptySamplingMass": "STATUS_EMPTY_SAMPLING_MASS",
}

_CONST = re.compile(r"^constexpr\s+int\s+(k\w+)\s*=\s*([^;]+);", re.MULTILINE)


def _eval_cuda_constant(expr: str) -> int:
    """Evaluate the small constant grammar the header actually uses: ints and `1 << n`."""
    expr = expr.strip()
    match = re.fullmatch(r"(\d+)\s*<<\s*(\d+)", expr)
    if match:
        return int(match.group(1)) << int(match.group(2))
    return int(expr, 0)


def assert_abi_matches_cuda() -> None:
    """Fail loudly if the CUDA header and this module disagree on any shared constant."""
    text = CUH.read_text(encoding="utf-8")
    found = {name: _eval_cuda_constant(value) for name, value in _CONST.findall(text)}
    missing = sorted(set(_CUDA_NAMES) - set(found))
    if missing:
        raise RuntimeError(f"{CUH.name} is missing M=4 control constants: {missing}")
    drift = {
        cuda_name: (found[cuda_name], globals()[py_name])
        for cuda_name, py_name in _CUDA_NAMES.items()
        if found[cuda_name] != globals()[py_name]
    }
    if drift:
        raise RuntimeError(
            f"M=4 control-block ABI drift between {CUH.name} and {Path(__file__).name}: {drift}"
        )


assert_abi_matches_cuda()


def new_control_block(dev: torch.device) -> torch.Tensor:
    """Allocate the zeroed v2 device control block. int32[32] == two 64 B lines; the frozen
    v1 verify region occupies words 0..15, so every v1 kernel reads it unchanged."""
    ctl = torch.zeros(EXT_WORDS, dtype=torch.int32, device=dev)
    # The int4 regions are only single-instruction loads if the base is 16 B aligned.
    # Torch's caching allocator returns 512 B-aligned blocks; assert rather than assume.
    if ctl.data_ptr() % 16 != 0:
        raise RuntimeError("M=4 control block is not 16 B aligned; int4 regions are unsafe")
    return ctl


def init_control_block(ctl: torch.Tensor, committed_len: int, verify_input_ids) -> None:
    """Seed the block for a fresh request: known committed_len and the first quartet.

    Clears every derived field so a stale commit decision cannot survive a reload. The
    protocol epoch is also reset here because a reload starts a new request, not a new
    step of the current one.
    """
    if ctl.numel() not in (WORDS, EXT_WORDS) or ctl.dtype != torch.int32:
        raise ValueError(
            f"control block must be int32[{WORDS}] (v1) or int32[{EXT_WORDS}] (v2), "
            f"got {ctl.dtype} {tuple(ctl.shape)}"
        )
    ids = [int(t) for t in verify_input_ids]
    if len(ids) != M:
        raise ValueError(f"verify_input_ids must hold exactly {M} ids, got {len(ids)}")
    if committed_len < M - 1:
        raise ValueError(
            f"committed_len={committed_len} is below the {M - 1} positions of convolution "
            "history the M=4 front reads; seed from a real prefill length"
        )
    host = torch.zeros(ctl.numel(), dtype=torch.int32)
    host[PENDING_TARGET_ID] = ids[0]
    host[COMMITTED_LEN] = committed_len
    host[VERIFY_INPUT_IDS : VERIFY_INPUT_IDS + M] = torch.tensor(ids, dtype=torch.int32)
    if ctl.numel() == EXT_WORDS:
        # Hand-off seeding makes replay #1 identical to every later replay: chain_slot_a is
        # what the DRAFT_EXTEND epilogue would have written -- [d1, C] under the S convention.
        host[CHAIN_SLOT_A] = ids[1]
        host[CHAIN_SLOT_A + 1] = committed_len
        host[STOP_POS] = -1
    ctl.copy_(host)


def read_control_block(ctl: torch.Tensor) -> dict:
    """Host-side snapshot for tests and reporting. Never called inside run()."""
    w = ctl.tolist()
    out = {
        "pending_target_id": w[PENDING_TARGET_ID],
        "committed_len": w[COMMITTED_LEN],
        "commit_count": w[COMMIT_COUNT],
        "output_count": w[OUTPUT_COUNT],
        "verify_input_ids": w[VERIFY_INPUT_IDS : VERIFY_INPUT_IDS + M],
        "emitted_ids": w[EMITTED_IDS : EMITTED_IDS + M],
        "accepted_draft_count": w[ACCEPTED_DRAFT_COUNT],
        "flags": w[FLAGS],
        "status": w[STATUS],
        "step_epoch": w[STEP_EPOCH],
    }
    if len(w) == EXT_WORDS:
        out["chain_slot_a"] = w[CHAIN_SLOT_A : CHAIN_SLOT_A + 2]
        out["chain_slot_b"] = w[CHAIN_SLOT_B : CHAIN_SLOT_B + 2]
        out["stop_pos"] = w[STOP_POS]
    return out


def conv_slot(position: int) -> int:
    """Ring slot holding the raw projected convolution input at absolute position p."""
    return position & (CONV_RING_SLOTS - 1)


def state_slot(position: int) -> int:
    """Ring slot holding the complete recurrent state after consuming absolute position p."""
    return position & (STATE_RING_SLOTS - 1)
