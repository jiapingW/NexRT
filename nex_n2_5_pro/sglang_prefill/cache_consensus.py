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

"""Pure host-side resolution for compact TP cache-lifecycle control records."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Iterable, Sequence

from nex_n2_5_pro.sglang_prefill.cache_lifecycle import (
    CacheBoundary,
    CacheDecision,
    CachePlan,
)


_DECISION_TO_CODE = {
    CacheDecision.PREFILL: 0,
    CacheDecision.APPEND: 1,
    CacheDecision.HIT: 2,
}
_CODE_TO_DECISION = {code: decision for decision, code in _DECISION_TO_CODE.items()}


class CacheConsensusError(RuntimeError):
    """TP ranks cannot safely enter the same cache-lifecycle transition."""


class FatalTPInvariantError(CacheConsensusError):
    """A TP invariant or backend collective failed; the worker must terminate."""


class RecoverableRequestError(CacheConsensusError):
    """All TP ranks rejected one request at an aligned, reset-safe boundary."""


@dataclass(frozen=True)
class RequestConsensus:
    generation: int
    decision: CacheDecision
    base_length: int
    target_length: int
    invalid_ranks: tuple[int, ...]

    @property
    def forced_prefill(self) -> bool:
        return bool(self.invalid_ranks)


def request_record(
    valid: bool,
    plan: CachePlan,
    fingerprint: int,
    *,
    fatal: bool = False,
    recoverable_failure: bool = False,
) -> tuple[int, int, int, int, int, int]:
    # 0 means that only this rank rejected its live cached boundary and permits a
    # collective PREFILL. -1 means that the local proposal itself is unusable.
    validity = -1 if fatal else (-2 if recoverable_failure else int(bool(valid)))
    return (
        validity,
        _DECISION_TO_CODE[plan.decision],
        int(plan.generation),
        int(plan.base_length),
        int(plan.target_length),
        int(fingerprint),
    )


def request_fingerprint(
    *,
    owner: str | None,
    tokens: Iterable[int],
    model_epoch: str,
    decode_kind: str,
    max_new_tokens: int,
    seed: int,
    temperature: float,
    top_p: float,
    handoff_mode: str,
    top_k: int = 20,
) -> int:
    """Build a stable signed-64 request identity for TP lifecycle consensus."""
    digest = hashlib.blake2b(digest_size=8, person=b"NexRT-TP")

    def add_bytes(value: bytes) -> None:
        digest.update(struct.pack("<Q", len(value)))
        digest.update(value)

    add_bytes(b"\xff" if owner is None else str(owner).encode("utf-8"))
    add_bytes(str(model_epoch).encode("utf-8"))
    add_bytes(str(decode_kind).encode("utf-8"))
    add_bytes(str(handoff_mode).encode("utf-8"))
    token_values = tuple(int(token) for token in tokens)
    digest.update(struct.pack("<Q", len(token_values)))
    for token in token_values:
        digest.update(struct.pack("<q", token))
    digest.update(
        struct.pack(
            "<qQddq",
            int(max_new_tokens),
            int(seed) & 0xFFFFFFFFFFFFFFFF,
            float(temperature),
            float(top_p),
            int(top_k),
        )
    )
    return int.from_bytes(digest.digest(), byteorder="little", signed=True)


def boundary_record(
    valid: bool,
    lease_generation: int,
    target_length: int,
    boundary: CacheBoundary | None,
) -> tuple[int, int, int, int, int]:
    if boundary is None:
        return (int(bool(valid)), int(lease_generation), int(target_length), -1, -1)
    return (
        int(bool(valid)),
        int(boundary.generation),
        int(boundary.sglang_token_length),
        int(boundary.nexrt_token_length),
        int(boundary.pending_token),
    )


def retain_record(retain: bool, generation: int, target_length: int):
    return (int(bool(retain)), int(generation), int(target_length), 0, 0)


def _records(
    values: Iterable[Sequence[int]],
    *,
    width: int,
    validity_values: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    rows = tuple(tuple(int(value) for value in row) for row in values)
    if not rows or any(len(row) != width for row in rows):
        raise CacheConsensusError(f"cache consensus requires non-empty {width}-word records")
    if any(row[0] not in validity_values for row in rows):
        raise CacheConsensusError(f"invalid cache consensus validity bit: {rows}")
    return rows


def resolve_request(records: Iterable[Sequence[int]]) -> RequestConsensus:
    rows = _records(records, width=6, validity_values=(-2, -1, 0, 1))
    generations = {row[2] for row in rows}
    targets = {row[4] for row in rows}
    fingerprints = {row[5] for row in rows}
    if len(generations) != 1 or len(targets) != 1 or len(fingerprints) != 1:
        raise FatalTPInvariantError(
            f"TP cache generation/target/request metadata disagree before request: {rows}"
        )
    fatal = tuple(rank for rank, row in enumerate(rows) if row[0] == -1)
    if any(row[0] == -1 for row in rows):
        raise FatalTPInvariantError(f"fatal TP cache proposals from ranks {fatal}: {rows}")
    recoverable = tuple(rank for rank, row in enumerate(rows) if row[0] == -2)
    if recoverable:
        if len(recoverable) != len(rows):
            raise FatalTPInvariantError(f"TP cache proposal failure disagrees across ranks: {rows}")
        raise RecoverableRequestError(f"collectively agreed request proposal failure: {rows}")
    invalid = tuple(rank for rank, row in enumerate(rows) if not row[0])
    generation = rows[0][2]
    target = rows[0][4]
    valid_rows = tuple(row for row in rows if row[0] == 1)
    if generation < 1 or target < 1:
        raise FatalTPInvariantError(f"invalid cache request proposal: {rows}")
    if any(row[1] not in _CODE_TO_DECISION or row[3] < 0 or row[3] > target for row in valid_rows):
        raise FatalTPInvariantError(f"invalid cache request proposal: {rows}")
    proposals = {(row[1], row[3]) for row in valid_rows}
    if len(proposals) > 1:
        raise FatalTPInvariantError(f"valid TP cache proposals disagree: {rows}")
    if invalid:
        return RequestConsensus(generation, CacheDecision.PREFILL, 0, target, invalid)
    if len(proposals) != 1:
        raise FatalTPInvariantError(f"valid TP cache proposals disagree: {rows}")
    decision_code, base = next(iter(proposals))
    return RequestConsensus(generation, _CODE_TO_DECISION[decision_code], base, target, ())


def resolve_boundary(records: Iterable[Sequence[int]], stage: str) -> tuple[int, int, int, int]:
    rows = _records(records, width=5, validity_values=(0, 1))
    invalid = tuple(rank for rank, row in enumerate(rows) if not row[0])
    if invalid:
        raise CacheConsensusError(f"{stage} boundary rejected on TP ranks {invalid}: {rows}")
    observations = {row[1:] for row in rows}
    if len(observations) != 1:
        raise CacheConsensusError(f"{stage} boundary disagrees across TP ranks: {rows}")
    generation, sglang_length, nexrt_length, pending_token = rows[0][1:]
    if generation < 1 or sglang_length < 1 or sglang_length != nexrt_length or pending_token < 0:
        raise CacheConsensusError(f"invalid {stage} boundary: {rows}")
    return generation, sglang_length, nexrt_length, pending_token


def resolve_retain(records: Iterable[Sequence[int]]) -> tuple[bool, tuple[int, ...]]:
    rows = _records(records, width=5, validity_values=(0, 1))
    lifecycle = {(row[1], row[2]) for row in rows}
    if len(lifecycle) != 1:
        raise FatalTPInvariantError(f"TP restore generations/targets disagree: {rows}")
    rejected = tuple(rank for rank, row in enumerate(rows) if not row[0])
    return not rejected, rejected


def failure_record(
    recoverable: bool | None,
    generation: int,
    phase: int,
    active_generation: int,
    request_kind: int = 0,
) -> tuple[int, int, int, int, int]:
    """Describe a request failure before deciding whether reset is safe.

    Every field after ``recoverable`` is lifecycle state. Reset is allowed only when
    all ranks report the same state and all ranks classify the failure as recoverable.
    """
    # 2 = request completed successfully, 1 = collectively resettable failure,
    # 0 = fatal local failure. Mixed outcomes are fatal in resolve_request_outcome.
    outcome = 2 if recoverable is None else int(bool(recoverable))
    return (
        outcome,
        int(generation),
        int(phase),
        int(active_generation),
        int(request_kind),
    )


def resolve_failure(records: Iterable[Sequence[int]]) -> tuple[int, int, int, int]:
    rows = _records(records, width=5, validity_values=(0, 1))
    if any(row[0] != 1 for row in rows):
        raise FatalTPInvariantError(f"request failure is fatal on TP ranks: {rows}")
    lifecycle = {row[1:] for row in rows}
    if len(lifecycle) != 1:
        raise FatalTPInvariantError(f"TP request lifecycle disagrees during failure reset: {rows}")
    return rows[0][1], rows[0][2], rows[0][3], rows[0][4]


def resolve_request_outcome(
    records: Iterable[Sequence[int]],
) -> tuple[str, tuple[int, int, int, int]]:
    rows = _records(records, width=5, validity_values=(0, 1, 2))
    outcomes = {row[0] for row in rows}
    lifecycle = {row[1:] for row in rows}
    if len(lifecycle) != 1:
        raise FatalTPInvariantError(f"TP request lifecycle disagrees at outcome: {rows}")
    if outcomes == {2}:
        return "success", (rows[0][1], rows[0][2], rows[0][3], rows[0][4])
    if outcomes != {1}:
        raise FatalTPInvariantError(f"request outcome disagrees across TP ranks: {rows}")
    return "reset", (rows[0][1], rows[0][2], rows[0][3], rows[0][4])
