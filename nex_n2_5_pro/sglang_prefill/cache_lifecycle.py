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

"""Lifecycle metadata for NexRT's single physical reusable cache session.

The metadata becomes READY only after the serving layer synchronizes the CUDA stream and
supplies observations from both live backends. Exact token IDs are retained because a digest
alone is not a correctness proof for recurrent-state reuse.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional


class CachePhase(str, Enum):
    EMPTY = "empty"
    FILLING = "filling"
    READY = "ready"
    DECODING = "decoding"


class CacheDecision(str, Enum):
    PREFILL = "prefill"
    APPEND = "append"
    HIT = "hit"


class CacheLeaseError(RuntimeError):
    """The lease is stale or does not match the cache's current lifecycle."""


class CacheBusyError(RuntimeError):
    """The physical cache is currently filling or decoding."""


@dataclass(frozen=True)
class CacheLease:
    generation: int
    owner: Optional[str]
    request_id: str
    decision: CacheDecision
    base_length: int
    target_length: int


@dataclass(frozen=True)
class CachePlan:
    """A request decision inspected without mutating the cache lifecycle."""

    generation: int
    decision: CacheDecision
    base_length: int
    target_length: int


@dataclass(frozen=True)
class CacheBoundary:
    """Backend observations made after the handoff stream has synchronized."""

    generation: int
    sglang_token_length: int
    nexrt_token_length: int
    pending_token: int
    request_pool_index: int
    mamba_slot: int
    request_identity: int


@dataclass(frozen=True)
class CachedSessionSnapshot:
    phase: CachePhase
    generation: int
    owner: Optional[str]
    model_epoch: Optional[str]
    committed_tokens: tuple[int, ...]
    token_digest: Optional[str]
    valid_kv_length: int
    boundary: Optional[CacheBoundary]
    last_error: Optional[str]


def _tokens(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _token_digest(tokens: tuple[int, ...]) -> Optional[str]:
    if not tokens:
        return None
    digest = hashlib.blake2b(digest_size=16)
    for token in tokens:
        digest.update(int(token).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


class CachedSession:
    """Validate reuse of the one SGLang/NexRT physical cache session.

    APPEND and HIT require the same non-empty owner, model epoch, exact token prefix, and a
    fresh live observation equal to the observation stored at the previous restore boundary.
    A mismatch replaces the cache with PREFILL. An in-flight cache is never evicted.
    """

    def __init__(self, capacity: int, model_epoch: str):
        if int(capacity) < 1:
            raise ValueError("capacity must be >= 1")
        if not str(model_epoch):
            raise ValueError("model_epoch must not be empty")
        self.capacity = int(capacity)
        self.model_epoch = str(model_epoch)
        self._phase = CachePhase.EMPTY
        self._generation = 0
        self._owner: Optional[str] = None
        self._session_epoch: Optional[str] = None
        self._committed_tokens: tuple[int, ...] = ()
        self._token_digest: Optional[str] = None
        self._valid_kv_length = 0
        self._boundary: Optional[CacheBoundary] = None
        self._last_error: Optional[str] = None
        self._active_decision: Optional[CacheDecision] = None
        self._target_tokens: tuple[int, ...] = ()
        self._lock = threading.RLock()

    def _snapshot(self) -> CachedSessionSnapshot:
        return CachedSessionSnapshot(
            phase=self._phase,
            generation=self._generation,
            owner=self._owner,
            model_epoch=self._session_epoch,
            committed_tokens=self._committed_tokens,
            token_digest=self._token_digest,
            valid_kv_length=self._valid_kv_length,
            boundary=self._boundary,
            last_error=self._last_error,
        )

    def snapshot(self) -> CachedSessionSnapshot:
        with self._lock:
            return self._snapshot()

    def _clear(self, reason: Optional[str]) -> None:
        self._phase = CachePhase.EMPTY
        self._owner = None
        self._session_epoch = None
        self._committed_tokens = ()
        self._token_digest = None
        self._valid_kv_length = 0
        self._boundary = None
        self._active_decision = None
        self._target_tokens = ()
        self._last_error = reason

    def _ready_is_reusable(
        self,
        owner: Optional[str],
        epoch: str,
        target: tuple[int, ...],
        live_boundary: Optional[CacheBoundary],
    ) -> bool:
        length = len(self._committed_tokens)
        return (
            owner is not None
            and self._phase == CachePhase.READY
            and self._owner == owner
            and self._session_epoch == epoch
            and length > 0
            and self._valid_kv_length == length
            and self._boundary is not None
            and live_boundary == self._boundary
            and length <= len(target)
            and target[:length] == self._committed_tokens
        )

    def _request_values(
        self,
        owner: Optional[str],
        tokens: Iterable[int],
        model_epoch: Optional[str],
    ) -> tuple[Optional[str], tuple[int, ...], str]:
        target = _tokens(tokens)
        if not target:
            raise ValueError("cache request must contain at least one token")
        if len(target) > self.capacity:
            raise ValueError(f"request length {len(target)} exceeds cache capacity {self.capacity}")
        owner = None if owner is None else str(owner)
        if owner == "":
            raise ValueError("owner must be non-empty or None")
        epoch = self.model_epoch if model_epoch is None else str(model_epoch)
        if not epoch:
            raise ValueError("model_epoch must not be empty")
        return owner, target, epoch

    def _plan_request_locked(
        self,
        owner: Optional[str],
        target: tuple[int, ...],
        epoch: str,
        live_boundary: Optional[CacheBoundary],
    ) -> CachePlan:
        if self._phase in (CachePhase.FILLING, CachePhase.DECODING):
            raise CacheBusyError(f"cache is currently {self._phase.value}")
        reusable = self._ready_is_reusable(owner, epoch, target, live_boundary)
        if reusable:
            base = len(self._committed_tokens)
            decision = CacheDecision.HIT if base == len(target) else CacheDecision.APPEND
        else:
            decision, base = CacheDecision.PREFILL, 0
        return CachePlan(
            generation=self._generation + 1,
            decision=decision,
            base_length=base,
            target_length=len(target),
        )

    def plan_request(
        self,
        owner: Optional[str],
        tokens: Iterable[int],
        *,
        model_epoch: Optional[str] = None,
        live_boundary: Optional[CacheBoundary] = None,
    ) -> CachePlan:
        """Inspect the next decision without entering FILLING.

        The serving layer uses this immutable proposal for TP-wide consensus before any rank
        clears allocator pools or enters SGLang work.
        """
        owner, target, epoch = self._request_values(owner, tokens, model_epoch)
        with self._lock:
            return self._plan_request_locked(owner, target, epoch, live_boundary)

    def begin_request(
        self,
        owner: Optional[str],
        tokens: Iterable[int],
        request_id: str,
        *,
        model_epoch: Optional[str] = None,
        live_boundary: Optional[CacheBoundary] = None,
        force_prefill: bool = False,
    ) -> CacheLease:
        owner, target, epoch = self._request_values(owner, tokens, model_epoch)

        with self._lock:
            if force_prefill:
                plan = CachePlan(
                    generation=self._generation + 1,
                    decision=CacheDecision.PREFILL,
                    base_length=0,
                    target_length=len(target),
                )
            else:
                plan = self._plan_request_locked(owner, target, epoch, live_boundary)
            if plan.decision == CacheDecision.PREFILL:
                self._clear(None)
                self._owner = owner
                self._session_epoch = epoch

            self._generation = plan.generation
            self._phase = CachePhase.FILLING
            self._active_decision = plan.decision
            self._target_tokens = target
            self._last_error = None
            return CacheLease(
                generation=self._generation,
                owner=owner,
                request_id=str(request_id),
                decision=plan.decision,
                base_length=plan.base_length,
                target_length=plan.target_length,
            )

    def _leased(self, lease: CacheLease, phase: CachePhase) -> None:
        if self._generation != int(lease.generation):
            raise CacheLeaseError(
                f"stale lease generation {lease.generation}, current {self._generation}"
            )
        if self._phase != phase:
            raise CacheLeaseError(f"cache is {self._phase.value}, expected {phase.value}")
        if self._owner != lease.owner:
            raise CacheLeaseError(f"cache owner changed from {lease.owner!r} to {self._owner!r}")

    @staticmethod
    def _validate_boundary(lease: CacheLease, boundary: CacheBoundary) -> None:
        if int(boundary.generation) != int(lease.generation):
            raise CacheLeaseError(
                f"boundary generation {boundary.generation} does not match lease "
                f"generation {lease.generation}"
            )
        if (
            int(boundary.sglang_token_length) != lease.target_length
            or int(boundary.nexrt_token_length) != lease.target_length
        ):
            raise CacheLeaseError(
                "SGLang and NexRT did not reach the requested token boundary: "
                f"sglang={boundary.sglang_token_length}, "
                f"nexrt={boundary.nexrt_token_length}, target={lease.target_length}"
            )
        if (
            int(boundary.pending_token) < 0
            or int(boundary.request_pool_index) < 0
            or int(boundary.mamba_slot) < 0
            or int(boundary.request_identity) <= 0
        ):
            raise CacheLeaseError(f"invalid backend boundary observation: {boundary}")

    def record_append(self, lease: CacheLease, start: int, end: int) -> None:
        start, end = int(start), int(end)
        with self._lock:
            self._leased(lease, CachePhase.FILLING)
            if lease.decision == CacheDecision.HIT:
                raise CacheLeaseError("a cache HIT has no rows to append")
            if start != self._valid_kv_length or end <= start or end > lease.target_length:
                raise CacheLeaseError(
                    f"non-contiguous append [{start}, {end}); expected start "
                    f"{self._valid_kv_length}, target {lease.target_length}"
                )
            self._valid_kv_length = end

    def _validate_fill_locked(self, lease: CacheLease, boundary: CacheBoundary) -> None:
        self._leased(lease, CachePhase.FILLING)
        if self._active_decision != lease.decision or not self._target_tokens:
            raise CacheLeaseError("lease does not match the active fill")
        if self._valid_kv_length != lease.target_length:
            raise CacheLeaseError(
                f"cache has KV through {self._valid_kv_length}, expected {lease.target_length}"
            )
        self._validate_boundary(lease, boundary)

    def validate_fill_boundary(self, lease: CacheLease, boundary: CacheBoundary) -> None:
        """Validate a fill boundary without publishing READY."""
        with self._lock:
            self._validate_fill_locked(lease, boundary)

    def commit_fill(self, lease: CacheLease, boundary: CacheBoundary) -> CachedSessionSnapshot:
        with self._lock:
            self._validate_fill_locked(lease, boundary)
            self._committed_tokens = self._target_tokens
            self._token_digest = _token_digest(self._committed_tokens)
            self._boundary = boundary
            self._phase = CachePhase.READY
            self._active_decision = None
            self._target_tokens = ()
            self._last_error = None
            return self._snapshot()

    def abort_fill(self, lease: CacheLease, reason: str) -> CachedSessionSnapshot:
        with self._lock:
            self._leased(lease, CachePhase.FILLING)
            self._clear(str(reason))
            return self._snapshot()

    def begin_decode(self, lease: CacheLease) -> CachedSessionSnapshot:
        with self._lock:
            self._leased(lease, CachePhase.READY)
            self._phase = CachePhase.DECODING
            return self._snapshot()

    def validate_decode_boundary(self, lease: CacheLease, boundary: CacheBoundary) -> None:
        """Validate a restored decode boundary without publishing READY."""
        with self._lock:
            self._leased(lease, CachePhase.DECODING)
            self._validate_boundary(lease, boundary)

    def finish_decode(self, lease: CacheLease, boundary: CacheBoundary) -> CachedSessionSnapshot:
        with self._lock:
            self._leased(lease, CachePhase.DECODING)
            try:
                self._validate_boundary(lease, boundary)
            except CacheLeaseError:
                self._generation += 1
                self._clear("decode boundary restore was not observed")
                raise
            self._boundary = boundary
            self._phase = CachePhase.READY
            self._last_error = None
            return self._snapshot()

    def abort_decode(self, lease: CacheLease, reason: str) -> CachedSessionSnapshot:
        with self._lock:
            self._leased(lease, CachePhase.DECODING)
            self._generation += 1
            self._clear(str(reason))
            return self._snapshot()

    def invalidate(
        self, reason: str, *, expected_generation: Optional[int] = None
    ) -> CachedSessionSnapshot:
        with self._lock:
            if expected_generation is not None and self._generation != int(expected_generation):
                raise CacheLeaseError(
                    f"refusing to invalidate generation {self._generation}; "
                    f"expected {expected_generation}"
                )
            self._generation += 1
            self._clear(str(reason))
            return self._snapshot()
