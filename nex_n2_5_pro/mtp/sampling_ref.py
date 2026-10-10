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

"""CPU checks of MTP sampled decisions, RNG, and committed draft prefixes."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

M = 4
NUM_DRAFTS = 3

_MASK64 = (1 << 64) - 1


def splitmix64(value: int) -> int:
    """Exact splitmix64 (the M4-local rederivation of the position-keyed spec)."""
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def seeded_uniform(seed: int, position: int) -> np.float32:
    """uniform for absolute position = hash(request_seed, position) in [0, 1)."""
    hashed = splitmix64((seed & _MASK64) ^ splitmix64(position & 0xFFFFFFFF))
    return np.float32(np.float32(hashed >> 40) * np.float32(1.0 / 16777216.0))


class Resolution(NamedTuple):
    accepted_draft_count: int
    commit_count: int
    emitted_ids: tuple
    pending_target_id: int


def resolve_prefix(target_ids, verify_ids) -> Resolution:
    """Pure prefix comparison: verify_ids = [t0, d1, d2, d3] (t0 is NOT a draft)."""
    assert len(target_ids) == M and len(verify_ids) == M
    accepted = 0
    for row in range(NUM_DRAFTS):
        if accepted == row and int(target_ids[row]) == int(verify_ids[row + 1]):
            accepted = row + 1
    final = int(target_ids[accepted])
    emitted = tuple(int(verify_ids[row + 1]) for row in range(accepted)) + (final,)
    return Resolution(accepted, accepted + 1, emitted, final)


class Expected(NamedTuple):
    accepted_draft_count: int
    commit_count: int
    emitted_ids: tuple
    pending_target_id: int
    boundary_rows: tuple


def verify_round_from_export(
    top_ids,
    top_probs,
    target_ids,
    verify_ids,
    *,
    seed: int,
    committed_len: int,
    temperature: float = 0.6,
) -> Expected:
    """Bridge live-round check from the kernel's own exports.

    Hard checks: every sampled id is a kept (probability > 0) member of that
    row's exported top-k, the draw column matches the integer-exact RNG replay
    against the exported renormalized CDF (an adjacent column is tolerated only
    inside a 4e-7 CDF boundary window), and the commit fields are recomputed
    EXACTLY from (verify_ids, target_ids).  Raises RuntimeError on any hard
    mismatch; the caller compares the returned fields against the control block.
    """
    top_ids = np.asarray(top_ids, dtype=np.int64)
    top_probs = np.asarray(top_probs, dtype=np.float32)
    target_ids = np.asarray(target_ids, dtype=np.int64)
    assert top_ids.ndim == 2 and top_ids.shape[0] == M
    top_k = top_ids.shape[1]
    assert 1 <= top_k <= 28 and top_probs.shape == top_ids.shape
    assert target_ids.shape == (M,)
    boundary_rows = []
    for row in range(M):
        sampled = int(target_ids[row])
        where = np.nonzero(top_ids[row] == sampled)[0]
        if where.size != 1:
            raise RuntimeError(
                f"sampled id {sampled} not exactly once in exported top-k of "
                f"row {row}: {top_ids[row].tolist()}"
            )
        column = int(where[0])
        if temperature == 0.0:
            if column != 0:
                raise RuntimeError(f"greedy row {row} did not take the top-1 id")
            continue
        if not top_probs[row, column] > 0.0:
            raise RuntimeError(f"sampled id of row {row} has zero nucleus mass")
        uniform = seeded_uniform(seed, committed_len + row)
        cdf = np.float32(0.0)
        expected_column = None
        near_boundary = False
        for k in range(top_k):
            cdf = np.float32(cdf + top_probs[row, k])
            if expected_column is None and cdf >= uniform:
                expected_column = k
                near_boundary = abs(float(cdf) - float(uniform)) < 4e-7
                break
        if expected_column is None:  # fp mass ended below u
            expected_column = int(np.max(np.nonzero(top_probs[row] > 0.0)[0]))
            near_boundary = True
        if column != expected_column:
            kernel_cdf = float(np.float32(np.sum(top_probs[row, : column + 1], dtype=np.float32)))
            if not (near_boundary or abs(kernel_cdf - float(uniform)) < 4e-7):
                raise RuntimeError(
                    f"row {row} draw mismatch: kernel column {column}, "
                    f"reference column {expected_column}, u={float(uniform)}"
                )
            boundary_rows.append(row)
        elif near_boundary:
            boundary_rows.append(row)
    resolution = resolve_prefix(target_ids, verify_ids)
    return Expected(
        resolution.accepted_draft_count,
        resolution.commit_count,
        resolution.emitted_ids,
        resolution.pending_target_id,
        tuple(boundary_rows),
    )
