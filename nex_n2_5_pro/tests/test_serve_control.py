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

"""Host-only tests for request policy and backend failure classification."""

import pytest

from nex_n2_5_pro.sglang_prefill.cache_consensus import (
    FatalTPInvariantError,
    RecoverableRequestError,
)
from nex_n2_5_pro.sglang_prefill.cache_lifecycle import (
    CacheDecision,
    CachePhase,
    CachedSession,
)
from nex_n2_5_pro.sglang_prefill.serve_control import (
    CacheReusePolicyError,
    FatalBackendError,
    RequestResetError,
    cache_key_for_request,
    dispatch_port_request,
    execute_request_with_consensus,
    is_fatal_backend_error,
    reset_failed_request,
)


class _Status:
    def __init__(self, value=1):
        self.value = value

    def zero_(self):
        self.value = 0


class _Block:
    def __init__(self):
        self.status = _Status()


class _CpBlock:
    def __init__(self):
        self.cp_status = _Status()


class _Pool:
    def __init__(self):
        self.clear_count = 0

    def clear(self):
        self.clear_count += 1


class _Tap:
    def __init__(self):
        self.abort_count = 0

    def abort(self):
        self.abort_count += 1


class DistBackendError(RuntimeError):
    """Stand-in for torch.distributed.DistBackendError in the host-only test."""


def test_collective_request_reset_is_not_misclassified_as_backend_fatal():
    error = RecoverableRequestError("collectively agreed request proposal failure")
    assert not is_fatal_backend_error(error)


@pytest.mark.parametrize(
    "error",
    (
        FatalBackendError("typed fatal backend failure"),
        RuntimeError("CUDA error: illegal memory access"),
        RuntimeError("ProcessGroupNCCL communicator aborted"),
        RuntimeError("c10d error while reducing status"),
        DistBackendError("backend communicator timed out"),
    ),
)
def test_cuda_and_nccl_failures_are_backend_fatal(error):
    assert is_fatal_backend_error(error)


def test_ordinary_request_failure_is_not_a_backend_failure():
    assert not is_fatal_backend_error(ValueError("prompt exceeds configured capacity"))


def test_untyped_request_failure_is_fatal_without_outcome_or_reset():
    calls = []

    def request():
        raise ValueError("rank-local append failure")

    def must_not_continue(*_args):
        calls.append(True)
        raise AssertionError("fatal request continued into outcome/reset handling")

    with pytest.raises(FatalTPInvariantError, match="cannot be reset safely"):
        execute_request_with_consensus(
            request,
            lifecycle_state=must_not_continue,
            gather_outcome=must_not_continue,
            reset_request=must_not_continue,
        )

    assert calls == []


def test_collective_recoverable_dispatch_resets_state_and_next_request_prefills():
    cache = CachedSession(capacity=16, model_epoch="model-v1")
    lease = cache.begin_request("alice", [1, 2, 3], "request-1")
    cache.record_append(lease, 0, 1)
    runtime_state = {"runtime_session": object(), "active_lease": lease}

    engine = type("Engine", (), {})()
    engine.head = _Block()
    engine.gdn = {0: _Block()}
    engine.attn = {1: _Block()}
    engine.moe = {2: _Block()}
    engine.attn_cp = {1: _CpBlock()}
    draft = type("Draft", (), {"attn": _Block(), "moe": _Block()})()
    m4 = type(
        "M4",
        (),
        {
            "ws_status": _Status(),
            "ws_moe_status": _Status(),
            "ws12_status": _Status(),
            "ws21_status": _Status(),
        },
    )()
    tap = _Tap()
    mtp = {"tap": tap, "arms": {"noncp": {"blk": draft, "m4": m4}}}
    model_runner = type("Runner", (), {})()
    model_runner.token_to_kv_pool_allocator = _Pool()
    model_runner.req_to_token_pool = _Pool()
    synchronize_count = [0]

    def synchronize():
        synchronize_count[0] += 1

    def lifecycle_state():
        snapshot = cache.snapshot()
        assert snapshot.phase == CachePhase.FILLING
        return snapshot.generation, 1, lease.generation, 0

    def request():
        raise RecoverableRequestError("collectively reproducible request failure")

    def gather_outcome(record):
        return [record] * 8

    def reset_request(reason):
        reset_failed_request(
            reason,
            cache=cache,
            runtime_state=runtime_state,
            model_runner=model_runner,
            engine=engine,
            mtp=mtp,
            synchronize=synchronize,
        )

    with pytest.raises(RequestResetError, match="collectively reproducible"):
        execute_request_with_consensus(
            request,
            lifecycle_state=lifecycle_state,
            gather_outcome=gather_outcome,
            reset_request=reset_request,
        )

    assert cache.snapshot().phase == CachePhase.EMPTY
    assert cache.snapshot().last_error == "collectively reproducible request failure"
    assert runtime_state == {"runtime_session": None, "active_lease": None}
    assert model_runner.token_to_kv_pool_allocator.clear_count == 1
    assert model_runner.req_to_token_pool.clear_count == 1
    assert tap.abort_count == 1
    assert synchronize_count == [2]
    assert engine.head.status.value == 0
    assert engine.gdn[0].status.value == 0
    assert engine.attn[1].status.value == 0
    assert engine.moe[2].status.value == 0
    assert draft.attn.status.value == 0
    assert draft.moe.status.value == 0
    assert all(
        getattr(m4, name).value == 0
        for name in ("ws_status", "ws_moe_status", "ws12_status", "ws21_status")
    )
    assert engine.attn_cp[1].cp_status.value == 0
    next_plan = cache.plan_request("alice", [1, 2, 3])
    assert next_plan.decision == CacheDecision.PREFILL
    assert next_plan.base_length == 0


def test_target_decode_accepts_reusable_cache_key():
    assert cache_key_for_request("session-a", speculative=False) == "session-a"


def test_speculative_decode_without_cache_key_is_one_shot():
    assert cache_key_for_request(None, speculative=True) is None


def test_speculative_cache_reuse_is_rejected_explicitly():
    with pytest.raises(CacheReusePolicyError, match="MTP/DFlash cache reuse is out of scope"):
        cache_key_for_request("session-a", speculative=True)


class _FakeConnection:
    def __init__(self, events):
        self.events = events
        self.writes = []

    def sendall(self, payload):
        self.events.append("socket-write")
        self.writes.append(payload)

    def close(self):
        self.events.append("socket-close")


class _FakeEventStream:
    def __init__(self, events):
        self.events = events

    def finish(self, _result):
        self.events.append("stream-finish")

    def fail(self, _error):
        self.events.append("stream-fail")

    def cancel(self):
        self.events.append("stream-cancel")

    def close(self):
        self.events.append("stream-close")


def test_streaming_production_dispatch_reports_reset_only_after_collective_cleanup():
    events = []
    conn = _FakeConnection(events)
    stream = _FakeEventStream(events)

    def recoverable_request():
        raise RecoverableRequestError("request validation failed on every rank")

    def reset_request(_reason):
        events.append("collective-reset")

    def run_request():
        return execute_request_with_consensus(
            recoverable_request,
            lifecycle_state=lambda: (1, 1, 1, 0),
            gather_outcome=lambda row: [row] * 8,
            reset_request=reset_request,
        )

    def must_not_terminate(*_args):
        raise AssertionError("recoverable dispatch terminated the serving group")

    result = dispatch_port_request(
        run_request,
        tp_rank=0,
        conn=conn,
        event_stream=stream,
        report=[],
        terminate_fatally=must_not_terminate,
        log_reset=lambda _error: events.append("reset-log"),
    )

    assert result is None
    assert events == [
        "collective-reset",
        "reset-log",
        "stream-fail",
        "stream-close",
        "socket-close",
    ]
    assert conn.writes == []


def test_streaming_production_dispatch_cancels_writer_before_fatal_termination():
    events = []
    conn = _FakeConnection(events)
    stream = _FakeEventStream(events)
    reset_calls = []

    def unexpected_request_failure():
        raise ValueError("rank-local append failure")

    def must_not_continue(*_args):
        raise AssertionError("fatal request entered lifecycle consensus")

    def run_request():
        return execute_request_with_consensus(
            unexpected_request_failure,
            lifecycle_state=must_not_continue,
            gather_outcome=must_not_continue,
            reset_request=lambda reason: reset_calls.append(reason),
        )

    def terminate_fatally(error, direct_conn):
        assert isinstance(error, FatalTPInvariantError)
        assert direct_conn is None
        events.append("fatal-terminate")
        raise SystemExit(2)

    with pytest.raises(SystemExit) as stopped:
        dispatch_port_request(
            run_request,
            tp_rank=0,
            conn=conn,
            event_stream=stream,
            report=[],
            terminate_fatally=terminate_fatally,
        )

    assert stopped.value.code == 2
    assert events == [
        "stream-cancel",
        "fatal-terminate",
        "stream-close",
        "socket-close",
    ]
    assert conn.writes == []
    assert reset_calls == []
