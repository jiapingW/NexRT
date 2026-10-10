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

"""Host-only tests for the single physical cached-session lifecycle."""

from nex_n2_5_pro.sglang_prefill.cache_lifecycle import (
    CacheBoundary,
    CacheBusyError,
    CacheDecision,
    CacheLeaseError,
    CachePhase,
    CachePlan,
    CachedSession,
)


def _boundary(lease, pending=None, *, length=None, request_identity=101):
    length = lease.target_length if length is None else int(length)
    return CacheBoundary(
        generation=lease.generation,
        sglang_token_length=length,
        nexrt_token_length=length,
        pending_token=9000 + length if pending is None else int(pending),
        request_pool_index=3,
        mamba_slot=7,
        request_identity=request_identity,
    )


def _commit(cache, owner, tokens, request_id, *, live_boundary=None, epoch=None):
    lease = cache.begin_request(
        owner, tokens, request_id, live_boundary=live_boundary, model_epoch=epoch
    )
    if lease.decision != CacheDecision.HIT:
        cache.record_append(lease, lease.base_length, lease.target_length)
    snap = cache.commit_fill(lease, _boundary(lease))
    return lease, snap


def main():
    cache = CachedSession(capacity=32, model_epoch="weights-v1")

    plan = cache.plan_request("alice", [1, 2, 3])
    assert plan == CachePlan(1, CacheDecision.PREFILL, 0, 3)
    assert cache.snapshot().phase == CachePhase.EMPTY

    prefill, ready = _commit(cache, "alice", [1, 2, 3], "a0")
    assert prefill.decision == CacheDecision.PREFILL
    assert ready.phase == CachePhase.READY and ready.valid_kv_length == 3
    assert ready.boundary is not None and ready.boundary.nexrt_token_length == 3
    digest = ready.token_digest

    planned_hit = cache.plan_request("alice", [1, 2, 3], live_boundary=ready.boundary)
    assert planned_hit.decision == CacheDecision.HIT
    forced = cache.begin_request(
        "alice",
        [1, 2, 3],
        "forced-prefill",
        live_boundary=ready.boundary,
        force_prefill=True,
    )
    assert forced.decision == CacheDecision.PREFILL and forced.base_length == 0
    cache.record_append(forced, 0, 3)
    ready = cache.commit_fill(forced, _boundary(forced))

    append = cache.begin_request("alice", [1, 2, 3, 4, 5], "a1", live_boundary=ready.boundary)
    assert append.decision == CacheDecision.APPEND and append.base_length == 3
    cache.record_append(append, 3, 4)
    cache.record_append(append, 4, 5)
    ready = cache.commit_fill(append, _boundary(append))
    assert ready.committed_tokens == (1, 2, 3, 4, 5)
    assert ready.token_digest is not None and ready.token_digest != digest

    hit = cache.begin_request("alice", [1, 2, 3, 4, 5], "a2", live_boundary=ready.boundary)
    assert hit.decision == CacheDecision.HIT
    hit_ready = cache.commit_fill(hit, _boundary(hit))
    cache.begin_decode(hit)
    hit_ready = cache.finish_decode(hit, _boundary(hit))
    assert hit_ready.phase == CachePhase.READY

    # A missing observation or stale backend identity makes exact tokens non-reusable.
    replace = cache.begin_request("alice", [1, 2, 3, 4, 5], "a3")
    assert replace.decision == CacheDecision.PREFILL
    cache.record_append(replace, 0, 5)
    ready = cache.commit_fill(replace, _boundary(replace))

    wrong_identity = CacheBoundary(
        generation=ready.boundary.generation,
        sglang_token_length=5,
        nexrt_token_length=5,
        pending_token=9005,
        request_pool_index=3,
        mamba_slot=7,
        request_identity=202,
    )
    replace = cache.begin_request("alice", [1, 2, 3, 4, 5], "a4", live_boundary=wrong_identity)
    assert replace.decision == CacheDecision.PREFILL
    cache.record_append(replace, 0, 5)
    ready = cache.commit_fill(replace, _boundary(replace))

    # A stale physical length also forces a full replacement.
    stale_live = CacheBoundary(
        generation=ready.boundary.generation,
        sglang_token_length=5,
        nexrt_token_length=4,
        pending_token=9005,
        request_pool_index=3,
        mamba_slot=7,
        request_identity=101,
    )
    replace = cache.begin_request("alice", [1, 2, 3, 4, 5], "a5", live_boundary=stale_live)
    assert replace.decision == CacheDecision.PREFILL
    cache.record_append(replace, 0, 5)
    ready = cache.commit_fill(replace, _boundary(replace))

    # A token mismatch and an execution-arm/model epoch change both require PREFILL.
    mismatch = cache.begin_request("alice", [1, 2, 99], "a6", live_boundary=ready.boundary)
    assert mismatch.decision == CacheDecision.PREFILL
    cache.record_append(mismatch, 0, 3)
    ready = cache.commit_fill(mismatch, _boundary(mismatch))
    epoch = cache.begin_request(
        "alice",
        [1, 2, 99, 8],
        "a7",
        model_epoch="weights-v2",
        live_boundary=ready.boundary,
    )
    assert epoch.decision == CacheDecision.PREFILL
    cache.record_append(epoch, 0, 4)
    ready = cache.commit_fill(epoch, _boundary(epoch))

    # Injected append failure: a partial physical write is never advertised as READY.
    failed_append = cache.begin_request(
        "alice",
        [1, 2, 99, 8, 10],
        "a8",
        model_epoch="weights-v2",
        live_boundary=ready.boundary,
    )
    cache.record_append(failed_append, 4, 5)
    failed = cache.abort_fill(failed_append, "injected append launch failure")
    assert failed.phase == CachePhase.EMPTY and failed.boundary is None

    # Injected finalize/synchronization failure: caller-supplied lengths cannot self-prove READY.
    bad_finalize = cache.begin_request("alice", [1, 2], "a9", model_epoch="weights-v2")
    cache.record_append(bad_finalize, 0, 2)
    try:
        cache.commit_fill(bad_finalize, _boundary(bad_finalize, length=1))
        raise AssertionError("mismatched backend boundary was accepted")
    except CacheLeaseError:
        pass
    assert cache.snapshot().phase == CachePhase.FILLING
    failed = cache.abort_fill(bad_finalize, "injected finalize observation failure")
    assert failed.phase == CachePhase.EMPTY

    # A failed post-decode restore invalidates the committed session inside finish_decode.
    decode_lease, ready = _commit(cache, "alice", [4, 5, 6], "a10", epoch="weights-v2")
    cache.begin_decode(decode_lease)
    try:
        cache.finish_decode(decode_lease, _boundary(decode_lease, length=2))
        raise AssertionError("mismatched decode restore was accepted")
    except CacheLeaseError:
        pass
    assert cache.snapshot().phase == CachePhase.EMPTY

    # Invalidation increments generation, so an old callback cannot mutate the session.
    stale, ready = _commit(cache, "alice", [7], "a11", epoch="weights-v2")
    cache.invalidate("test invalidate")
    try:
        cache.begin_decode(stale)
        raise AssertionError("stale lease was accepted")
    except CacheLeaseError:
        pass

    busy = CachedSession(capacity=4, model_epoch="weights-v1")
    busy.begin_request("alice", [1], "busy-a")
    try:
        busy.begin_request("bob", [2], "busy-b")
        raise AssertionError("an in-flight physical cache was replaced")
    except CacheBusyError:
        pass

    print("PASS test_cache_lifecycle: observed boundaries, strict reuse, failure invalidation")


if __name__ == "__main__":
    main()
