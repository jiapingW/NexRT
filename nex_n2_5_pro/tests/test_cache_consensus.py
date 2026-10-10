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

"""Host-only tests for TP cache-lifecycle consensus resolution."""

import pytest

from nex_n2_5_pro.sglang_prefill.cache_consensus import (
    CacheConsensusError,
    FatalTPInvariantError,
    RecoverableRequestError,
    boundary_record,
    failure_record,
    request_record,
    request_fingerprint,
    resolve_boundary,
    resolve_failure,
    resolve_request,
    resolve_request_outcome,
    resolve_retain,
    retain_record,
)
from nex_n2_5_pro.sglang_prefill.cache_lifecycle import (
    CacheBoundary,
    CacheDecision,
    CachePlan,
)


def _plan(decision=CacheDecision.HIT, generation=4, base=8, target=8):
    return CachePlan(generation, decision, base, target)


def _boundary(pending=71):
    return CacheBoundary(4, 8, 8, pending, 3, 7, 101)


def _record(valid=True, plan=None, fingerprint=101, **kwargs):
    return request_record(valid, _plan() if plan is None else plan, fingerprint, **kwargs)


def test_unanimous_request_proposal_is_preserved():
    record = _record()
    result = resolve_request([record] * 8)
    assert result.decision == CacheDecision.HIT
    assert result.generation == 4
    assert result.base_length == result.target_length == 8
    assert result.invalid_ranks == ()


def test_one_invalid_rank_forces_prefill_on_all_ranks():
    records = [_record() for _ in range(8)]
    records[3] = _record(False, _plan(CacheDecision.PREFILL, base=0))
    result = resolve_request(records)
    assert result.decision == CacheDecision.PREFILL
    assert result.base_length == 0
    assert result.invalid_ranks == (3,)
    assert result.forced_prefill


def test_valid_rank_disagreement_fails_before_lifecycle_mutation():
    records = [_record() for _ in range(8)]
    records[5] = _record(True, _plan(CacheDecision.APPEND, base=4))
    with pytest.raises(FatalTPInvariantError, match="proposals disagree"):
        resolve_request(records)


def test_generation_disagreement_cannot_be_hidden_by_invalid_rank():
    records = [_record() for _ in range(8)]
    records[2] = _record(False, _plan(CacheDecision.PREFILL, generation=5, base=0))
    with pytest.raises(FatalTPInvariantError, match="metadata disagree"):
        resolve_request(records)


def test_post_handoff_boundary_rejects_one_rank_and_value_mismatch():
    records = [boundary_record(True, 4, 8, _boundary()) for _ in range(8)]
    invalid = list(records)
    invalid[6] = boundary_record(False, 4, 8, None)
    with pytest.raises(CacheConsensusError, match=r"ranks \(6,\)"):
        resolve_boundary(invalid, "post-handoff")

    mismatch = list(records)
    mismatch[1] = boundary_record(True, 4, 8, _boundary(pending=72))
    with pytest.raises(CacheConsensusError, match="disagrees"):
        resolve_boundary(mismatch, "post-handoff")


def test_restore_retain_is_false_if_any_rank_rejects_it():
    records = [retain_record(True, 4, 8) for _ in range(8)]
    records[7] = retain_record(False, 4, 8)
    retain, rejected = resolve_retain(records)
    assert not retain
    assert rejected == (7,)


def test_collective_prefill_only_uses_boundary_rejection():
    records = [_record() for _ in range(8)]
    records[3] = _record(False, _plan(CacheDecision.PREFILL, base=0))
    result = resolve_request(records)
    assert result.forced_prefill
    assert result.invalid_ranks == (3,)


def test_collectively_agreed_nonfatal_failure_allows_reset():
    records = [failure_record(True, 4, 1, 4, 0) for _ in range(8)]
    assert resolve_failure(records) == (4, 1, 4, 0)


def test_collective_plan_failure_is_resettable_but_mixed_plan_failure_is_fatal():
    plan = _plan(CacheDecision.PREFILL, base=0)
    records = [_record(False, plan, recoverable_failure=True) for _ in range(8)]
    with pytest.raises(RecoverableRequestError, match="collectively agreed"):
        resolve_request(records)
    records[4] = _record()
    with pytest.raises(FatalTPInvariantError, match="proposal failure disagrees"):
        resolve_request(records)


def test_failure_lifecycle_disagreement_is_fatal_not_reset():
    records = [failure_record(True, 4, 1, 4, 0) for _ in range(8)]
    records[2] = failure_record(True, 4, 2, 4, 0)
    with pytest.raises(FatalTPInvariantError, match="lifecycle disagrees"):
        resolve_failure(records)


def test_local_fatal_proposal_terminates_collectively():
    records = [_record() for _ in range(8)]
    records[6] = _record(False, _plan(CacheDecision.PREFILL, base=0), fatal=True)
    with pytest.raises(FatalTPInvariantError, match="fatal TP cache proposals"):
        resolve_request(records)


def test_request_outcome_distinguishes_success_and_collective_reset():
    reset = [failure_record(True, 4, 1, 4, 0) for _ in range(8)]
    assert resolve_request_outcome(reset)[0] == "reset"
    success = [failure_record(None, 4, 3, 4, 0) for _ in range(8)]
    assert resolve_request_outcome(success)[0] == "success"


def test_request_outcome_mixed_reset_and_success_is_fatal():
    rows = [failure_record(None, 4, 3, 4, 0) for _ in range(8)]
    rows[1] = failure_record(True, 4, 3, 4, 0)
    with pytest.raises(FatalTPInvariantError, match="outcome disagrees"):
        resolve_request_outcome(rows)


def test_request_metadata_disagreement_is_fatal_even_for_prefill_fallback():
    records = [_record() for _ in range(8)]
    records[3] = _record(False, _plan(CacheDecision.PREFILL, base=0), fingerprint=202)
    with pytest.raises(FatalTPInvariantError, match="request metadata disagree"):
        resolve_request(records)


def test_invalid_valid_proposal_cannot_hide_behind_prefill_fallback():
    records = [_record() for _ in range(8)]
    records[0] = (records[0][0], 99, *records[0][2:])
    records[3] = _record(False, _plan(CacheDecision.PREFILL, base=0))
    with pytest.raises(FatalTPInvariantError, match="invalid cache request proposal"):
        resolve_request(records)


def test_request_fingerprint_covers_cache_and_generation_inputs():
    values = dict(
        owner="alice",
        tokens=[1, 2, 3],
        model_epoch="model-v1|target",
        decode_kind="target",
        max_new_tokens=8,
        seed=7,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        handoff_mode="append",
    )
    baseline = request_fingerprint(**values)
    assert baseline == request_fingerprint(**values)
    for key, changed in (
        ("owner", "bob"),
        ("tokens", [1, 2, 4]),
        ("model_epoch", "model-v2|target"),
        ("decode_kind", "mtp:noncp"),
        ("max_new_tokens", 9),
        ("seed", 8),
        ("temperature", 0.7),
        ("top_p", 0.9),
        ("top_k", 28),
        ("handoff_mode", "gather"),
    ):
        candidate = dict(values)
        candidate[key] = changed
        assert request_fingerprint(**candidate) != baseline
