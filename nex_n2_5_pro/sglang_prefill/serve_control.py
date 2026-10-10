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

"""Host-only serving policy and backend-failure classification."""

from __future__ import annotations

import json

from nex_n2_5_pro.sglang_prefill.cache_consensus import (
    FatalTPInvariantError,
    RecoverableRequestError,
    failure_record,
    resolve_failure,
    resolve_request_outcome,
)


class FatalBackendError(RuntimeError):
    """The CUDA/NCCL process state is no longer safe for request recovery."""


class RequestResetError(RuntimeError):
    """The current request was rejected after a collective, non-fatal reset."""


class CacheReusePolicyError(ValueError):
    """A request asked for cache reuse outside the supported target path."""


def cache_key_for_request(cache_key, *, speculative: bool):
    """Return the target cache key or reject speculative cache reuse explicitly."""
    if speculative and cache_key is not None:
        raise CacheReusePolicyError(
            "MTP/DFlash cache reuse is out of scope; omit cache_key for speculative decode"
        )
    return None if speculative else cache_key


def clear_pools(model_runner) -> None:
    """Release the current physical SGLang allocator arena."""
    for pool in (
        getattr(model_runner, "token_to_kv_pool_allocator", None),
        getattr(model_runner, "req_to_token_pool", None),
    ):
        if pool is not None and hasattr(pool, "clear"):
            pool.clear()


def clear_request_statuses(engine, mtp=None) -> None:
    """Clear request-scoped target, speculative, and CP execution status."""
    buffers = (
        engine.head,
        *engine.gdn.values(),
        *engine.attn.values(),
        *engine.moe.values(),
    )
    if mtp is not None:
        for arm in mtp["arms"].values():
            buffers += (
                arm["m4"].ws_status,
                arm["m4"].ws_moe_status,
                arm["m4"].ws12_status,
                arm["m4"].ws21_status,
                arm["blk"].attn.status,
                arm["blk"].moe.status,
            )
    for buffer in buffers:
        (buffer.status if hasattr(buffer, "status") else buffer).zero_()
    for block in engine.attn_cp.values():
        block.cp_status.zero_()


def reset_failed_request(
    reason,
    *,
    cache,
    runtime_state,
    model_runner,
    engine,
    mtp=None,
    synchronize,
) -> None:
    """Reset only a request already approved as collectively recoverable."""
    try:
        if mtp is not None:
            mtp["tap"].abort()
        synchronize()
        clear_pools(model_runner)
        cache.invalidate(reason)
        runtime_state["runtime_session"] = None
        runtime_state["active_lease"] = None
        clear_request_statuses(engine, mtp)
        synchronize()
    except Exception as error:  # noqa: BLE001
        raise FatalBackendError(
            f"failed to reset request state after {reason}: {error!r}"
        ) from error


def is_fatal_backend_error(error: BaseException) -> bool:
    """Recognize typed failures and concrete external CUDA/NCCL diagnostics.

    Operations owned by the bridge raise ``FatalBackendError`` directly. The
    narrow name/message fallback covers exceptions raised inside PyTorch itself.
    It intentionally does not treat the generic word "collective" as fatal.
    """
    if isinstance(error, FatalBackendError):
        return True
    type_name = f"{type(error).__module__}.{type(error).__qualname__}".lower()
    if any(
        marker in type_name
        for marker in (
            "cuda",
            "nccl",
            "c10d",
            "distbackenderror",
            "distnetworkerror",
        )
    ):
        return True
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "cudaerror=",
            "cuda error:",
            "cuda runtime error",
            "cuda driver error",
            "cuda initialization error",
            "cuda context",
            "context is destroyed",
            "context has been destroyed",
            "device-side assert",
            "illegal memory access",
            "unspecified launch failure",
            "processgroupnccl",
            "nccl error",
            "nccl communicator",
            "ncclunhandled",
            "c10d error",
        )
    )


def execute_request_with_consensus(
    request,
    *,
    lifecycle_state,
    gather_outcome,
    reset_request,
):
    """Execute one request and dispatch its collective success/reset/fatal outcome."""
    local_error = None
    try:
        result = request()
    except FatalTPInvariantError:
        raise
    except RecoverableRequestError as error:
        local_error = error
        result = None
    except Exception as error:  # noqa: BLE001
        if is_fatal_backend_error(error):
            if isinstance(error, FatalBackendError):
                raise
            raise FatalBackendError(f"fatal backend failure during request: {error!r}") from error
        raise FatalTPInvariantError(
            f"unexpected request failure cannot be reset safely: {error!r}"
        ) from error

    generation, phase, active_generation, request_kind = lifecycle_state()
    outcome_rows = gather_outcome(
        failure_record(
            True if local_error is not None else None,
            generation,
            phase,
            active_generation,
            request_kind,
        )
    )
    outcome, _ = resolve_request_outcome(outcome_rows)
    if outcome == "reset":
        resolve_failure(outcome_rows)
        reason = str(local_error or "collective request reset")
        reset_request(reason)
        raise RequestResetError(reason)
    if local_error is not None or outcome != "success":
        raise FatalTPInvariantError(f"request completion outcome disagrees: {outcome_rows}")
    return result


def dispatch_port_request(
    request,
    *,
    tp_rank,
    conn,
    event_stream,
    report,
    terminate_fatally,
    log_reset=None,
):
    """Send one port result without violating collective failure policy."""

    def send_json_line(payload):
        conn.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode())

    try:
        result = request()
        if tp_rank == 0:
            report.append(result)
            if event_stream is not None:
                event_stream.finish(result)
            else:
                send_json_line(result)
        return result
    except RequestResetError as error:
        if log_reset is not None:
            log_reset(error)
        if tp_rank == 0:
            if event_stream is not None:
                event_stream.fail(repr(error))
            else:
                send_json_line({"error": repr(error)})
        return None
    except FatalTPInvariantError as error:
        if tp_rank == 0 and event_stream is not None:
            event_stream.cancel()
        terminate_fatally(error, None if event_stream is not None else conn)
        raise AssertionError("terminate_fatally returned")
    except Exception as error:  # noqa: BLE001
        if tp_rank == 0 and event_stream is not None:
            event_stream.cancel()
        terminate_fatally(error, None if event_stream is not None else conn)
        raise AssertionError("terminate_fatally returned")
    finally:
        if tp_rank == 0 and conn is not None:
            if event_stream is not None:
                event_stream.close()
            conn.close()
