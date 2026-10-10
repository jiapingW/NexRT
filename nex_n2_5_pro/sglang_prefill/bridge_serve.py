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

"""Example server for SGLang prefill, device handoff, and NexRT decode.

Select 1-batch decoding, MTP, or DFlash at startup. One resident engine processes prompts
from --lens, --files, or the --listen socket; bridge_client.py sends requests.
Launch with the run-nexrt serve.sh script or torchrun --standalone --nproc-per-node=8.
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
NEX_N2_5_PRO = HERE.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(NEX_N2_5_PRO.parent))

from nex_n2_5_pro.sampling import resolve_sampling_config  # noqa: E402
from nex_n2_5_pro.sglang_prefill.weight_loading import load_model  # noqa: E402

from sglang.srt.distributed.parallel_state import (  # noqa: E402
    destroy_distributed_environment,
    destroy_model_parallel,
)
from sglang.srt.entrypoints.engine import _set_envs_and_config  # noqa: E402
from sglang.srt.layers.moe import initialize_moe_config  # noqa: E402
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config  # noqa: E402
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config  # noqa: E402

from nex_n2_5_pro.sglang_prefill.bridge import (  # noqa: E402
    B,
    Handoff,
    chunked_prefill,
    layer_sets,
    make_server_args,
    CP_THRESHOLD,
    MODEL,
    PortArgs,
)
from nex_n2_5_pro.sglang_prefill.cache_consensus import (  # noqa: E402
    CacheConsensusError,
    FatalTPInvariantError,
    boundary_record,
    request_record,
    request_fingerprint,
    resolve_boundary,
    resolve_request,
    resolve_retain,
    retain_record,
)
from nex_n2_5_pro.sglang_prefill.process_supervisor import torchrun_rank  # noqa: E402
from nex_n2_5_pro.sglang_prefill.serve_control import (  # noqa: E402
    FatalBackendError,
    RequestResetError,
    cache_key_for_request,
    clear_pools,
    dispatch_port_request,
    execute_request_with_consensus,
    is_fatal_backend_error,
    reset_failed_request,
)
from nex_n2_5_pro.sglang_prefill.cache_lifecycle import (  # noqa: E402
    CacheBoundary,
    CacheDecision,
    CachePlan,
    CachePhase,
    CachedSession,
)
from nex_n2_5_pro.sglang_prefill.stream_protocol import (  # noqa: E402
    BridgeStreamWriter,
    decode_visible_text,
)

# Stop the generation at end-of-turn or a tool-call opener (the model may emit any of these).
STOP_TOKEN_STRS = (
    "<tool_call>",
    "<|tool_call_begin|>",
    "<|tool_calls_begin|>",
    "<function_call>",
    "<|action_start|>",
)


def _checked_sampling_config(temperature, top_p):
    """Validate and canonicalize the graph-stable M=4 sampler inputs."""
    temperature = float(temperature)
    top_p = float(top_p)
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError(f"temperature must be finite and >= 0, got {temperature}")
    if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be finite and in (0, 1], got {top_p}")
    return temperature, top_p


def _seed_as_int64(seed):
    """Preserve a request seed's low 64 bits in a signed torch.int64 scalar."""
    value = int(seed) & 0xFFFFFFFFFFFFFFFF
    return value if value < (1 << 63) else value - (1 << 64)


def _request_text(request: dict) -> str:
    """Accept inline prompt text without reading client-supplied server paths."""
    if "file" in request:
        raise ValueError("send prompt text; bridge_client.py --file reads files locally")
    text = request.get("text")
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return text


def _validated_prompt_ids(ids, max_prompt, min_length=1):
    """Return a concrete token list, rejecting prompts that would be truncated."""
    tokens = [int(token) for token in ids]
    if not tokens:
        raise ValueError("empty prompt")
    if len(tokens) < min_length:
        raise ValueError(f"prompt must contain at least {min_length} tokens")
    if len(tokens) > max_prompt:
        raise ValueError(
            f"prompt has {len(tokens)} tokens; maximum is {max_prompt} (input was not truncated)"
        )
    return tokens


def _stop_ids(tok, model_path):
    """Resolve model EOS, tokenizer EOS, and the bridge's tool-call stops once."""
    ids = set()
    config_path = Path(model_path) / "generation_config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    for eos in (config.get("eos_token_id"), getattr(tok, "eos_token_id", None)):
        if eos is not None:
            ids.update([int(eos)] if isinstance(eos, int) else map(int, eos))
    for s in STOP_TOKEN_STRS:
        try:
            i = tok.convert_tokens_to_ids(s)
        except Exception:  # noqa: BLE001
            i = None
        unk = getattr(tok, "unk_token_id", None)
        if i is not None and i >= 0 and i != unk:
            ids.add(int(i))
    return ids


def _sync_cuda():
    try:
        torch.cuda.synchronize()
    except Exception as error:  # noqa: BLE001
        raise FatalBackendError(f"CUDA synchronize failed: {error!r}") from error


def worker(server_args, port_args, gpu_id, tp_rank, cli):
    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)
    initialize_fp4_gemm_config(server_args)
    B.configure_logger(server_args, prefix=f" TP{tp_rank}")
    B.TreeCacheNamespace.available_and_evictable_str = lambda self: "(dummy tree cache)"
    B.TreeCacheNamespace.pretty_print = lambda self: None
    rp = print if tp_rank == 0 else (lambda *a, **k: None)

    # ---- boot: sglang (full 60 layers) + nexrt (weights ALIASED from sglang, cp graph) ----
    t0 = time.time()
    model_runner, tok = load_model(server_args, port_args, gpu_id, tp_rank)
    dev = torch.device("cuda", torch.cuda.current_device())
    rp(f"@@ BOOT sglang load={time.time() - t0:.1f}s", flush=True)
    B.extend(B.prepare_synthetic_inputs_for_latency_test(1, 128), model_runner)  # JIT warm
    _sync_cuda()
    clear_pools(model_runner)

    nlayers = 60
    FULL_ATTN, GDN = layer_sets(nlayers)
    os.environ["NEXRT_NLAYERS"] = str(nlayers)
    from nex_n2_5_pro.decode_engine import DecodeEngine
    from nex_n2_5_pro.sglang_prefill.ext_weights import SglangExt
    from nex_n2_5_pro.real_prompts import build_real_prompt
    from nex_n2_5_pro.attn import CACHE_LEN

    t0 = time.time()
    eng = DecodeEngine(
        tp_rank, server_args.tp_size, dev, cp=not cli.no_cp, ext=SglangExt(model_runner.model)
    )
    _sync_cuda()
    torch.cuda.empty_cache()
    rp(
        f"@@ BOOT nexrt build={time.time() - t0:.1f}s (ext=sglang-alias, cp={not cli.no_cp})",
        flush=True,
    )
    h = Handoff(eng, model_runner, FULL_ATTN, GDN, dev)
    model_epoch = (
        f"{Path(server_args.model_path).resolve()}|"
        f"quant={getattr(server_args, 'quantization', None)}|layers={nlayers}"
    )
    cache = CachedSession(h.cache_len, model_epoch)
    r2t = model_runner.req_to_token_pool
    # warm the handoff kernel once (first launch pays ~60 ms of CUDA module load otherwise)
    _w = B.extend(B.prepare_synthetic_inputs_for_latency_test(1, 128), model_runner)
    h.gather(
        int(_w[2].req_pool_indices[0]),
        128,
        int(r2t.get_mamba_indices(_w[2].req_pool_indices)[0].item()),
        0,
    )
    _sync_cuda()
    clear_pools(model_runner)
    stop_ids = _stop_ids(tok, server_args.model_path)
    rp(f"@@ BOOT stop_ids={sorted(stop_ids)}", flush=True)

    mtp = None
    dflash = None
    if cli.dflash:
        from nex_n2_5_pro.sglang_prefill.bridge_dflash import DFlashBridge

        t0 = time.time()
        dflash = DFlashBridge(eng, model_runner, server_args, port_args, dev)
        _sync_cuda()
        torch.cuda.empty_cache()
        rp(f"@@ BOOT dflash build={time.time() - t0:.1f}s", flush=True)
    if cli.mtp:
        from nex_n2_5_pro.m4_verify import M4VerifyNonCP, M4VerifyCp, M4_CP_DISPATCH_S
        from nex_n2_5_pro import m4_control as ctlabi
        from nex_n2_5_pro.mtp.block import MtpDraft
        from nex_n2_5_pro.mtp.serve import capture_round, generate_mtp
        from nex_n2_5_pro.mtp.sampling_ref import verify_round_from_export
        from nex_n2_5_pro.config import LAYERS_DIR
        from nex_n2_5_pro.sglang_prefill.sglang_mtp import HiddenTap, MtpPrefill, draft_checkpoint

        t0 = time.time()
        cp_threshold = cli.mtp_cp_threshold or M4_CP_DISPATCH_S

        def _build_arm(arm, share_from=None):
            # Both arms share the selected Tensor Core O-projections; only the
            # target attention core changes. M4VerifyCp requires a CP-enabled engine.
            m4 = M4VerifyNonCP(eng, dev) if arm == "noncp" else M4VerifyCp(eng, dev)
            blk = MtpDraft(
                eng,
                m4,
                dev,
                eos_ids=sorted(stop_ids),
                sampling=cli.mtp_sampling == "sample",
                temperature=cli.mtp_temperature,
                top_p=cli.mtp_top_p,
                share_weights_from=share_from["blk"] if share_from else None,
            )
            if share_from is not None:
                # Both arms pack the SAME shared engine weights into the same
                # lane-major layouts; point the second arm's lazy pack caches at
                # the first arm's containers so each prepack exists once.
                s = share_from["m4"]
                m4._bf16tc_packs = s._bf16tc_packs
                m4._gdnfront_packs = s._gdnfront_packs
                m4._oproj_packs = s._oproj_packs
                m4._gate_packs = s._gate_packs
                m4._head_pack = s._head_pack
            return {"m4": m4, "blk": blk, "graph": None}

        arm_names = ("noncp", "cp") if cli.mtp_arm == "auto" else (cli.mtp_arm,)
        arms = {}
        for a in arm_names:
            arms[a] = _build_arm(a, share_from=next(iter(arms.values()), None))
        tap = HiddenTap(model_runner.model)
        # SGLang computes the MTP layer over the prompt (its EAGLE/NEXTN draft
        # extend), writing MTP KV at the target's token slots.
        pref = MtpPrefill(
            model_runner,
            server_args,
            port_args,
            tp_rank,
            draft_checkpoint(server_args.model_path, LAYERS_DIR),
        )
        _sync_cuda()
        free, _ = torch.cuda.mem_get_info(dev)
        rp(
            f"@@ BOOT mtp build={time.time() - t0:.1f}s arms={sorted(arms)} "
            f"cp_threshold={cp_threshold} free={free / 2**30:.2f}GiB",
            flush=True,
        )
        mtp = {
            "arms": arms,
            "tap": tap,
            "pref": pref,
            "ctl": ctlabi,
            "capture": capture_round,
            "generate": generate_mtp,
            "sampling_ref": verify_round_from_export,
            "pick": (
                lambda Lp: (
                    cli.mtp_arm
                    if cli.mtp_arm != "auto"
                    else ("cp" if Lp >= cp_threshold else "noncp")
                )
            ),
        }

    state = {
        "g_noncp": None,
        "g_cp": None,
        "active_lease": None,
        "runtime_session": None,
    }

    def gather_cache_control(record, elapsed_ms):
        """Collect one compact lifecycle record from every TP rank."""
        started = time.perf_counter()
        local = torch.tensor(record, dtype=torch.int64, device=dev)
        gathered = torch.empty(server_args.tp_size * local.numel(), dtype=torch.int64, device=dev)
        try:
            dist.all_gather_into_tensor(gathered, local)
            rows = gathered.view(server_args.tp_size, local.numel()).cpu().tolist()
        except Exception as error:  # noqa: BLE001
            raise FatalBackendError(f"TP cache-control collective failed: {error!r}") from error
        elapsed_ms[0] += (time.perf_counter() - started) * 1000.0
        return rows

    def all_reduce_max(tensor, stage):
        try:
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        except Exception as error:  # noqa: BLE001
            raise FatalBackendError(f"TP {stage} all-reduce failed: {error!r}") from error

    def broadcast_from_rank0(tensor, stage):
        try:
            dist.broadcast(tensor, src=0)
        except Exception as error:  # noqa: BLE001
            raise FatalBackendError(f"TP {stage} broadcast failed: {error!r}") from error

    def agree_boundary(stage, lease, boundary, local_error, enabled, elapsed_ms):
        """Fail together before publishing a fill or restore boundary."""
        if local_error is not None and is_fatal_backend_error(local_error):
            if isinstance(local_error, FatalBackendError):
                raise local_error
            raise FatalBackendError(
                f"fatal backend failure while validating {stage} boundary: {local_error!r}"
            ) from local_error
        if not enabled:
            if local_error is not None:
                raise local_error
            return boundary
        rows = gather_cache_control(
            boundary_record(
                local_error is None and boundary is not None,
                lease.generation,
                lease.target_length,
                boundary,
            ),
            elapsed_ms,
        )
        try:
            resolve_boundary(rows, stage)
        except CacheConsensusError as error:
            fatal = FatalTPInvariantError(f"{stage} boundary consensus failed: {error}")
            if local_error is not None:
                raise fatal from local_error
            raise fatal from error
        return boundary

    def observe_boundary(generation, session):
        """Read the synchronized SGLang and NexRT prompt boundary."""
        if tuple(session.tokens) == ():
            raise RuntimeError("cannot cache an empty SGLang session")
        if len(session.batch.reqs) != 1 or session.batch.reqs[0] is not session.req:
            raise RuntimeError("retained SGLang request is not the live batch request")
        batch_rp_idx = int(session.batch.req_pool_indices[0])
        request_rp_idx = getattr(session.req, "req_pool_idx", None)
        if (
            request_rp_idx is None
            or batch_rp_idx != int(session.req_pool_index)
            or batch_rp_idx != int(request_rp_idx)
        ):
            raise RuntimeError(
                "SGLang request-pool identity changed: "
                f"session={session.req_pool_index}, batch={batch_rp_idx}, "
                f"request={request_rp_idx}"
            )
        mslot = int(r2t.get_mamba_indices(session.batch.req_pool_indices)[0].item())
        request_mslot = getattr(session.req, "mamba_pool_idx", None)
        request_mslot_value = (
            None
            if request_mslot is None
            else int(request_mslot.reshape(-1)[0].item())
            if isinstance(request_mslot, torch.Tensor)
            else int(request_mslot)
        )
        if request_mslot_value != mslot:
            raise RuntimeError(
                "SGLang mamba-pool identity changed: "
                f"request={request_mslot_value}, allocator={mslot}"
            )
        batch_length = int(session.batch.seq_lens.reshape(-1)[0].item())
        request_committed = int(session.req.kv_committed_len)
        request_allocated = int(session.req.kv_allocated_len)
        if not (batch_length == request_committed == request_allocated == len(session.tokens)):
            raise RuntimeError(
                "SGLang did not retain one fully committed prompt boundary: "
                f"batch={batch_length}, committed={request_committed}, "
                f"allocated={request_allocated}, session={len(session.tokens)}"
            )
        pending_token, nexrt_length = [int(value) for value in eng.slot.tolist()]
        sglang_pending = int(session.next_token_ids[0])
        if pending_token != sglang_pending:
            raise RuntimeError(
                "NexRT pending token does not match the retained SGLang session: "
                f"nexrt={pending_token}, sglang={sglang_pending}"
            )
        return CacheBoundary(
            generation=int(generation),
            sglang_token_length=batch_length,
            nexrt_token_length=nexrt_length,
            pending_token=pending_token,
            request_pool_index=batch_rp_idx,
            mamba_slot=mslot,
            request_identity=id(session.req),
        )

    def finish_request_cache(
        lease,
        mslot,
        prompt_len,
        first_token,
        retain,
        consensus_enabled,
        consensus_ms,
    ):
        """Return the physical session to READY, or invalidate all of its live state."""
        if consensus_enabled:
            rows = gather_cache_control(
                retain_record(retain, lease.generation, lease.target_length), consensus_ms
            )
            try:
                retain, rejected_ranks = resolve_retain(rows)
            except CacheConsensusError as error:
                raise FatalTPInvariantError(f"restore retain consensus failed: {error}") from error
            if rejected_ranks:
                rp(
                    f"@@ CACHE restore disabled by TP ranks {rejected_ranks}",
                    flush=True,
                )
        if retain:
            boundary = None
            local_error = None
            try:
                h.finalize(mslot, prompt_len, first_token)
                _sync_cuda()
                runtime = state["runtime_session"]
                if runtime is None:
                    raise RuntimeError("live SGLang session disappeared before cache restore")
                if int(runtime["generation"]) != int(lease.generation):
                    raise RuntimeError(
                        "live SGLang session generation changed before cache restore: "
                        f"runtime={runtime['generation']}, lease={lease.generation}"
                    )
                boundary = observe_boundary(lease.generation, runtime["prefill"])
                cache.validate_decode_boundary(lease, boundary)
            except Exception as error:  # noqa: BLE001
                local_error = error
            boundary = agree_boundary(
                "restore", lease, boundary, local_error, consensus_enabled, consensus_ms
            )
            cache.finish_decode(lease, boundary)
        else:
            cache.abort_decode(lease, "request result is not reusable")
            state["runtime_session"] = None
            clear_pools(model_runner)
        state["active_lease"] = None
        return bool(retain)

    def _handle_request_impl(
        name,
        tokens,
        max_new,
        seed,
        temperature=None,
        top_p=None,
        cache_key=None,
        handoff_mode=None,
        consensus_enabled=True,
        consensus_fault_rank=None,
        measure_consensus=False,
        event_stream=None,
        top_k=None,
    ):
        """The one-turn pipeline; the outer wrapper closes TP success/failure collectively."""
        request_started = time.perf_counter()
        handoff_mode = cli.handoff_mode if handoff_mode is None else str(handoff_mode)
        if handoff_mode not in ("append", "gather"):
            raise ValueError(f"unsupported handoff mode {handoff_mode!r}")
        Lp = len(tokens)
        if Lp < 1 or Lp > cli.max_prompt:
            raise ValueError(f"prompt length must be in [1, {cli.max_prompt}], got {Lp}")
        if (mtp is not None or dflash is not None) and Lp < 3:
            raise ValueError("speculative decode requires at least three prompt tokens")
        if mtp is None and dflash is None:
            temperature, top_p, top_k = resolve_sampling_config(
                eng.head.sampling_defaults, temperature, top_p, top_k
            )
        else:
            temperature, top_p = _checked_sampling_config(
                cli.mtp_temperature if temperature is None else temperature,
                cli.mtp_top_p if top_p is None else top_p,
            )
            top_k = 20
        armname = mtp["pick"](Lp) if mtp is not None else None
        decode_kind = (
            "dflash"
            if dflash is not None
            else (f"mtp:{armname}" if armname is not None else "target")
        )
        request_epoch = model_epoch + "|" + decode_kind
        consensus_ms = [0.0]
        # Reusable prompt state is intentionally limited to the ordinary target path.
        # MTP owns draft state with different alignment and never reuses this cache.
        reuse_cache = mtp is None and dflash is None
        lifecycle_cache_key = cache_key_for_request(cache_key, speculative=not reuse_cache)
        request_identity = request_fingerprint(
            owner=lifecycle_cache_key,
            tokens=tokens,
            model_epoch=request_epoch,
            decode_kind=decode_kind,
            max_new_tokens=max_new,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            handoff_mode=handoff_mode,
        )
        cached = state["runtime_session"] if reuse_cache else None
        live_boundary = None
        local_cache_valid = True
        local_proposal_fatal = False
        local_proposal_recoverable = False
        snapshot = cache.snapshot()
        if reuse_cache and cached is not None:
            try:
                if snapshot.phase != CachePhase.READY:
                    raise RuntimeError(
                        f"live runtime exists while cache metadata is {snapshot.phase.value}"
                    )
                if int(cached["generation"]) != int(snapshot.generation):
                    raise RuntimeError(
                        "cache metadata/runtime generation mismatch: "
                        f"metadata={snapshot.generation}, runtime={cached['generation']}"
                    )
                _sync_cuda()
                live_boundary = observe_boundary(snapshot.generation, cached["prefill"])
            except Exception as error:  # noqa: BLE001
                local_cache_valid = False
                local_proposal_fatal = is_fatal_backend_error(error)
                rp(f"@@ CACHE live boundary rejected: {error!r}", flush=True)
                if local_proposal_fatal:
                    if isinstance(error, FatalBackendError):
                        raise
                    raise FatalBackendError(
                        f"fatal backend failure while observing cached boundary: {error!r}"
                    ) from error
        elif reuse_cache and snapshot.phase == CachePhase.READY:
            local_cache_valid = False
            rp("@@ CACHE READY metadata has no live runtime session", flush=True)
        if consensus_fault_rank is not None and tp_rank == int(consensus_fault_rank):
            live_boundary = None
            local_cache_valid = False
        try:
            plan = cache.plan_request(
                lifecycle_cache_key,
                tokens,
                model_epoch=request_epoch,
                live_boundary=live_boundary,
            )
        except Exception as error:  # noqa: BLE001
            local_cache_valid = False
            # A plan failure is not a boundary rejection: the rank cannot prove a
            # compatible proposal. A uniform, non-backend plan failure can be reset;
            # a mixed failure is promoted to a fatal TP invariant by the resolver.
            local_proposal_fatal = is_fatal_backend_error(error)
            if local_proposal_fatal:
                if isinstance(error, FatalBackendError):
                    raise
                raise FatalBackendError(
                    f"fatal backend failure while planning cache request: {error!r}"
                ) from error
            local_proposal_recoverable = not local_proposal_fatal
            plan = CachePlan(snapshot.generation + 1, CacheDecision.PREFILL, 0, Lp)
            rp(f"@@ CACHE request proposal rejected: {error!r}", flush=True)
        consensus = None
        if consensus_enabled:
            rows = gather_cache_control(
                request_record(
                    local_cache_valid,
                    plan,
                    request_identity,
                    fatal=local_proposal_fatal,
                    recoverable_failure=local_proposal_recoverable,
                ),
                consensus_ms,
            )
            # A unanimous proposal failure raises the only reset-safe exception here,
            # after every rank has completed this collective and before cache mutation.
            consensus = resolve_request(rows)
            if consensus.invalid_ranks:
                rp(
                    f"@@ CACHE forcing TP-wide PREFILL; invalid ranks={consensus.invalid_ranks}",
                    flush=True,
                )
        consensus_invalid_ranks = list(consensus.invalid_ranks) if consensus is not None else []
        lease = cache.begin_request(
            lifecycle_cache_key,
            tokens,
            name,
            model_epoch=request_epoch,
            live_boundary=live_boundary,
            force_prefill=(
                not reuse_cache or bool(consensus is not None and consensus.forced_prefill)
            ),
        )
        if consensus is not None and (
            lease.generation != consensus.generation
            or lease.decision != consensus.decision
            or lease.base_length != consensus.base_length
            or lease.target_length != consensus.target_length
        ):
            raise FatalTPInvariantError(
                f"activated cache lease {lease} differs from TP consensus {consensus}"
            )
        state["active_lease"] = lease
        if lease.decision == CacheDecision.PREFILL:
            # The current backend has one physical slot and one SGLang allocator arena. A full
            # replacement therefore releases that arena before rebuilding slot zero.
            clear_pools(model_runner)
            state["runtime_session"] = None
            cached = None
        elif cached is None:
            raise RuntimeError(
                f"cache metadata selected {lease.decision.value}, "
                "but its live SGLang session is missing"
            )
        elif cached["prefill"].tokens != tuple(tokens[: lease.base_length]):
            raise RuntimeError(
                f"live SGLang session does not match its {lease.base_length}-token committed prefix"
            )
        rp(
            f"@@ CACHE {name} key={lifecycle_cache_key!r} gen={lease.generation} "
            f"decision={lease.decision.value} "
            f"base={lease.base_length} target={lease.target_length}",
            flush=True,
        )
        torch.cuda.empty_cache()
        _sync_cuda()
        if reuse_cache or mtp is not None:
            first_temperature = (
                0.0 if mtp is not None and cli.mtp_sampling == "greedy" else temperature
            )
            eng.head.set_sampling_config(first_temperature, top_p, top_k, seed=seed)
        else:
            eng.head.seed_dev[0] = _seed_as_int64(seed)

        t0 = time.time()
        appended_to = [lease.base_length]
        append_events = []
        mtp_replay_start = None

        def _append_chunk(s, e, _next_tok, chunk_batch):
            """Queue only this SGLang extend's KV rows in NexRT's contiguous cache."""
            s, e = int(s), int(e)
            if s != appended_to[0]:
                raise RuntimeError(
                    f"SGLang KV append is not contiguous: expected start {appended_to[0]}, got {s}"
                )
            rp_idx = int(chunk_batch.req_pool_indices[0])
            if handoff_mode == "append":
                append_start = torch.cuda.Event(True)
                append_end = torch.cuda.Event(True)
                append_start.record()
                h.append(rp_idx, s, e - s, dst_start=s)
                append_end.record()
                append_events.append((append_start, append_end))
            cache.record_append(lease, s, e)
            appended_to[0] = e

        if dflash is not None:
            dflash.begin(Lp)
            try:
                nti, ntl, batch, session = chunked_prefill(
                    model_runner,
                    tokens,
                    4096,
                    on_chunk=dflash.consume,
                    on_chunk_batch=_append_chunk,
                    return_session=True,
                    request_id=lease.generation,
                )
                dflash.end()
            finally:
                dflash.tap.abort()
        elif mtp is not None:
            if lease.decision != CacheDecision.PREFILL or lease.base_length != 0:
                raise FatalTPInvariantError(
                    f"speculative decode received reusable cache lease {lease}"
                )
            # SGLang MTP draft-prefill: after each target extend chunk, SGLang's MTP layer
            # extends over the same rows (ids shifted by one) from that chunk's post-norm
            # hidden. HiddenTap holds one chunk, never the prompt.
            A = mtp["arms"][armname]
            blk, pref, tap = A["blk"], mtp["pref"], mtp["tap"]
            if cli.mtp_sampling == "sample":
                A["m4"].set_sampling_config(temperature, top_p, seed=seed)
            mtp_replay_start = 0
            tap.arm()

            def _mtp_chunk(s, e, next_tok, chunk_batch):
                hid = tap.pop()
                if hid.shape[0] != e - s:
                    raise RuntimeError(f"tap rows {hid.shape[0]} != chunk {e - s}")
                _append_chunk(s, e, next_tok, chunk_batch)
                pref.extend(chunk_batch, hid, tokens[e] if e < Lp else next_tok)

            try:
                nti, ntl, batch, session = chunked_prefill(
                    model_runner,
                    tokens,
                    4096,
                    on_chunk_batch=_mtp_chunk,
                    resume=None,
                    return_session=True,
                    request_id=lease.generation,
                    sample_next_token=lambda logits: eng.head.sample_prefill(logits, eng.slot, Lp),
                )
            except BaseException:
                # A hook left armed after a failed request captures the next request's
                # first chunk and makes recovery impossible. Drop only this transient.
                tap.abort()
                raise
            else:
                tap.disarm()
        else:
            if lease.decision == CacheDecision.HIT:
                session = cached["prefill"]
                nti, ntl, batch = (session.next_token_ids, session.next_token_logits, session.batch)
            else:
                nti, ntl, batch, session = chunked_prefill(
                    model_runner,
                    tokens,
                    4096,
                    on_chunk_batch=_append_chunk,
                    resume=cached["prefill"] if cached is not None else None,
                    return_session=True,
                    request_id=lease.generation,
                )
            nti = eng.head.sample_prefill(ntl, eng.slot, Lp)
            session.next_token_ids = nti
        _sync_cuda()
        prefill_wall_s = time.time() - t0
        first_token = int(nti[0])
        rp_idx = int(batch.req_pool_indices[0])
        mslot = int(r2t.get_mamba_indices(batch.req_pool_indices)[0].item())
        if appended_to[0] != Lp:
            raise RuntimeError(f"SGLang KV append stopped at {appended_to[0]}, expected {Lp}")

        append_ms = sum(start.elapsed_time(end) for start, end in append_events)
        finalize_ms = 0.0
        if lease.decision != CacheDecision.HIT:
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            if handoff_mode == "append":
                # KV rows were appended chunk-by-chunk. Finalize copies only the final GDN state,
                # publishes [first_token, Lp], and clears NexRT status words.
                h.finalize(mslot, Lp, first_token)
            else:
                # Matched full-pipeline baseline: suffix prefill still resumes in SGLang, then the
                # legacy gather recopies the complete prompt into NexRT.
                h.gather(rp_idx, Lp, mslot, first_token)
            e1.record()
            e1.synchronize()
            finalize_ms = e0.elapsed_time(e1)
        else:
            # READY already proves that the previous decode restored this exact boundary.
            # Re-finalizing an exact HIT only repeats GDN copies and status publication.
            _sync_cuda()
        # Keep prefill_s compatible with the old full-gather path: it measures SGLang's
        # prefill work, while gather_ms reports the streamed append plus finalization.
        prefill_s = max(0.0, prefill_wall_s - append_ms / 1000.0)
        gather_ms = append_ms + finalize_ms

        if mtp is not None:
            m4, blk, ctl = A["m4"], A["blk"], mtp["ctl"]
            mtp_k, mtp_v, y_sel, d1 = mtp["pref"].finish(rp_idx, Lp)
            # Rounds only write draft cache positions >= Lp, so the prompt rows are
            # copied once and later re-seeds pass None.
            blk.seed(mtp_k, mtp_v, Lp, y_sel)
            del mtp_k, mtp_v

        boundary = None
        local_error = None
        try:
            boundary = observe_boundary(lease.generation, session)
            cache.validate_fill_boundary(lease, boundary)
        except Exception as error:  # noqa: BLE001
            local_error = error
        boundary = agree_boundary(
            "post-handoff", lease, boundary, local_error, consensus_enabled, consensus_ms
        )
        cache.commit_fill(lease, boundary)
        state["runtime_session"] = {
            "generation": lease.generation,
            "prefill": session,
            "mamba_slot": mslot,
            "first_token": first_token,
        }
        cache.begin_decode(lease)
        ttft_s = time.perf_counter() - request_started
        consensus_ttft_local_ms = consensus_ms[0]
        rank_max = torch.tensor(
            [
                prefill_s,
                gather_ms / 1000.0,
                ttft_s,
                consensus_ttft_local_ms / 1000.0,
            ],
            dtype=torch.float64,
            device=dev,
        )
        all_reduce_max(rank_max, "request timing")
        prefill_s, gather_s, ttft_s, consensus_ttft_s = [
            float(value) for value in rank_max.tolist()
        ]
        gather_ms = gather_s * 1000.0
        consensus_ttft_ms = consensus_ttft_s * 1000.0

        def consensus_metrics():
            total_ms = consensus_ms[0]
            restore_ms = max(0.0, total_ms - consensus_ttft_local_ms)
            if measure_consensus:
                rank_max_ms = torch.tensor([total_ms, restore_ms], dtype=torch.float64, device=dev)
                all_reduce_max(rank_max_ms, "cache consensus timing")
                total_ms, restore_ms = [float(value) for value in rank_max_ms.tolist()]
            return total_ms, restore_ms

        if event_stream is not None:
            event_stream.event(
                {
                    "type": "start",
                    "prompt_tokens": Lp,
                    "ttft_s": round(ttft_s, 6),
                    "prefill_s": round(prefill_s, 6),
                    "gather_ms": round(gather_ms, 3),
                }
            )
            event_stream.tokens([first_token])

        if dflash is not None:
            graph = dflash.capture(Lp, first_token, lambda: h.finalize(mslot, Lp, first_token))
            _sync_cuda()
            decode_start = time.perf_counter()
            toks, accepts, status, checks = dflash.generate(
                graph,
                max_new,
                stop_ids,
                cli.verify,
                on_tokens=event_stream.tokens if event_stream is not None else None,
            )
            _sync_cuda()
            elapsed = time.perf_counter() - decode_start
            gen = [first_token] + toks
            finish_request_cache(
                lease, mslot, Lp, first_token, False, consensus_enabled, consensus_ms
            )
            rp(
                f"@@ SERVE {name}: DFlash tokens={len(gen)} rounds={len(accepts)} "
                f"verify={checks} status={status}",
                flush=True,
            )
            return {
                "name": name,
                "prompt_tokens": Lp,
                "mode": "dflash",
                "handoff_mode": handoff_mode,
                "prefill_s": round(prefill_s, 6),
                "gather_ms": round(gather_ms, 3),
                "ttft_s": round(ttft_s, 6),
                "decode_us_per_tok": elapsed * 1e6 / max(1, len(toks)),
                "accept_mean": sum(accepts) / max(1, len(accepts)),
                "rounds": len(accepts),
                "verified_rounds": checks if cli.verify else None,
                "cache_reuse_enabled": False,
                "status": status,
                "stop": "eos"
                if gen[-1] in stop_ids
                else ("max-tokens" if len(gen) >= max_new else "context-limit"),
                "generated_tokens": len(gen),
                "gen_token_ids": gen,
                "generated_text": decode_visible_text(tok, gen, stop_ids),
            }
        if mtp is not None:
            prefill_finished = first_token in stop_ids or max_new == 1

            def seed_mtp_state(y_sel, d1):
                """Per-request M=4 seeding: rings from the finalized bs=1 state (the
                M4Verify.load scatter, minus the disk load h.gather already replaced),
                sticky statuses, ctl, and the draft block. Draft K/V rows [0,Lp) are
                already resident (streamed during prefill); rounds only write cache
                positions >= Lp, so re-seeding never has to restore them."""
                for layer, gdn in eng.gdn.items():
                    ring = m4.conv_ring[layer]
                    ring.zero_()
                    for kk in range(3):
                        ring[:, ctl.conv_slot(Lp - 3 + kk)].copy_(gdn.conv_state4[:, kk])
                    sring = m4.state_ring[layer]
                    sring.zero_()
                    sring[ctl.state_slot(Lp - 1)].copy_(gdn.state_t)
                for s in (
                    m4.ws_status,
                    m4.ws_moe_status,
                    m4.ws12_status,
                    m4.ws21_status,
                    blk.attn.status,
                    blk.moe.status,
                ):
                    s.zero_()
                # The embed cursor's invalid-token word is sticky across graph replays.
                # A new request must not inherit a previous request's fail-closed status;
                # its diagnostic count is request-local too.
                m4.cursor.zero_()
                m4.sampling_status.zero_()
                ctl.init_control_block(m4.ctl, Lp, [first_token, d1, 0, 0])
                blk.seed(None, None, Lp, y_sel)

            seed_mtp_state(y_sel, d1)
            if A["graph"] is None and not prefill_finished:
                A["graph"] = mtp["capture"](blk)
                h.finalize(mslot, Lp, first_token)  # capture advanced: restore prompt boundary
                seed_mtp_state(y_sel, d1)
            _sync_cuda()
            t0 = time.time()
            sampled_checks = [0]

            def _verify_sampled_round(before, after):
                """Exact four-row sample-and-verify decision check (--verify only)."""
                # The whole-round observer runs before draft_phase and after
                # extend_phase.  At decision time anchor/d1 are therefore the
                # pre-round words, while d2/d3 are the tail words left by this
                # round's draft phase (extend advances only anchor/d1).
                verify_ids = [
                    before[ctl.VERIFY_INPUT_IDS],
                    before[ctl.VERIFY_INPUT_IDS + 1],
                    after[ctl.VERIFY_INPUT_IDS + 2],
                    after[ctl.VERIFY_INPUT_IDS + 3],
                ]
                expected = mtp["sampling_ref"](
                    m4.sv_top_ids.cpu().to(torch.int64),
                    m4.sv_top_probs.cpu(),
                    m4.sv_target_ids.cpu().to(torch.int64),
                    verify_ids,
                    seed=seed,
                    committed_len=before[ctl.COMMITTED_LEN],
                    temperature=temperature,
                )
                emitted = tuple(after[ctl.EMITTED_IDS : ctl.EMITTED_IDS + expected.commit_count])
                if not (
                    after[ctl.STATUS] == ctl.STATUS_OK
                    and after[ctl.ACCEPTED_DRAFT_COUNT] == expected.accepted_draft_count
                    and after[ctl.COMMIT_COUNT] == expected.commit_count
                    and emitted == expected.emitted_ids
                    and after[ctl.PENDING_TARGET_ID] == expected.pending_target_id
                    and after[ctl.COMMITTED_LEN]
                    == before[ctl.COMMITTED_LEN] + expected.commit_count
                ):
                    raise RuntimeError(
                        "M=4 sampled decision disagrees with CPU reference: "
                        f"expected={expected}, before={before}, after={after}"
                    )
                sampled_checks[0] += 1

            observer = (
                _verify_sampled_round if cli.verify and cli.mtp_sampling == "sample" else None
            )
            if prefill_finished:
                toks, accepts, st_mtp = [], [], 0
            else:
                toks, accepts, st_mtp = mtp["generate"](
                    blk,
                    A["graph"],
                    max_new,
                    context_limit=CACHE_LEN,
                    round_observer=observer,
                    on_tokens=event_stream.tokens if event_stream is not None else None,
                )
            wall = time.time() - t0
            gen = [first_token] + toks
            us_tok = wall * 1e6 / max(1, len(toks))
            ac = sum(accepts) / max(1, len(accepts))

            agree = -1
            if cli.verify and cli.mtp_sampling == "greedy" and not prefill_finished:
                if state["g_noncp"] is None:
                    state["g_noncp"] = eng.capture(warmup=8)
                h.finalize(mslot, Lp, first_token)
                n_chk = min(16, len(gen) - 1)
                agree = 0
                for i in range(n_chk):
                    eng.slot[0] = gen[i]  # teacher-force bs=1 greedy argmax
                    state["g_noncp"].replay()
                    _sync_cuda()
                    t = int(eng.head.out_id[int(eng.head.out_val.argmax().item())].item())
                    agree += int(t == gen[i + 1])
                agree = (agree, n_chk)

            sampled_ok = (
                not cli.verify or cli.mtp_sampling != "sample" or sampled_checks[0] == len(accepts)
            )
            # ``max_new=1`` legitimately returns only the already-produced first
            # target token, so an empty post-first-token list is still success.
            ok = (
                st_mtp == 0
                and len(gen) > 0
                and sampled_ok
                and (agree == -1 or agree[0] >= agree[1] - 1)
            )
            cache_retained = finish_request_cache(
                lease,
                mslot,
                Lp,
                first_token,
                reuse_cache and lifecycle_cache_key is not None and ok,
                consensus_enabled,
                consensus_ms,
            )
            consensus_total_ms, consensus_restore_ms = consensus_metrics()
            free, _ = torch.cuda.mem_get_info(dev)
            ag = (
                f"sample-exact:{sampled_checks[0]}/{len(accepts)}"
                if cli.verify and cli.mtp_sampling == "sample"
                else (f"{agree[0]}/{agree[1]}" if agree != -1 else "-")
            )
            print(
                f"@@ R{tp_rank} MTP  {name:28s} L={Lp:6d} arm={armname} "
                f"mode={handoff_mode} ttft={ttft_s:7.3f}s prefill={prefill_s:7.2f}s "
                f"gather={gather_ms:7.3f}ms mtp={us_tok:7.1f}us/tok x{len(toks):4d} "
                f"AC={ac:.2f} rounds={len(accepts)} sampling={cli.mtp_sampling} "
                f"T={temperature:g} P={top_p:g} verify={ag} status={st_mtp} "
                f"free={free / 2**30:.2f}GiB -> {'PASS' if ok else 'CHECK'}",
                flush=True,
            )
            return {
                "name": name,
                "prompt_tokens": Lp,
                "arm": armname,
                "handoff_mode": handoff_mode,
                "ttft_s": round(ttft_s, 6),
                "prefill_s": round(prefill_s, 6),
                "gather_ms": round(gather_ms, 3),
                "stop": "eos"
                if gen[-1] in stop_ids
                else ("max-tokens" if len(gen) >= max_new else "context-limit"),
                "mtp_us_per_tok": round(us_tok, 1),
                "accept_mean": round(ac, 3),
                "rounds": len(accepts),
                "generated_tokens": len(gen),
                "forced_agree": ag,
                "seed": seed,
                "sampling": cli.mtp_sampling,
                "temperature": temperature,
                "top_p": top_p,
                "cache_consensus_enabled": consensus_enabled,
                "cache_reuse_enabled": reuse_cache,
                "cache_consensus_invalid_ranks": consensus_invalid_ranks,
                "cache_consensus_ttft_ms": round(consensus_ttft_ms, 3),
                "cache_consensus_restore_ms": round(consensus_restore_ms, 3),
                "cache_consensus_ms": round(consensus_total_ms, 3),
                "mtp_replay_start": mtp_replay_start,
                "cache_generation": lease.generation,
                "cache_decision": lease.decision.value,
                "cache_base_tokens": lease.base_length,
                "cache_retained": cache_retained,
                "status": st_mtp,
                "gen_token_ids_head": gen[:16],
                "gen_token_ids": gen,
                "generated_text": decode_visible_text(tok, gen, stop_ids),
            }

        if state["g_noncp"] is None:
            state["g_noncp"] = eng.capture(warmup=8)
            h.finalize(mslot, Lp, first_token)
        if state["g_cp"] is None and Lp + max_new >= CP_THRESHOLD:
            state["g_cp"] = eng.capture(warmup=8, step_fn=eng.step_cp)
            h.finalize(mslot, Lp, first_token)
        g_noncp, g_cp = state["g_noncp"], state["g_cp"]

        _sync_cuda()
        t0 = time.time()
        gen = eng.generate(
            g_noncp,
            g_cp,
            max_new_tokens=max_new,
            eos_id=stop_ids,
            cp_threshold=CP_THRESHOLD,
            on_tokens=event_stream.tokens if event_stream is not None else None,
        )  # lagged host-mirror drain (decode_engine)
        decode_us = (time.time() - t0) * 1e6 / max(1, len(gen) - 1)
        stop = (
            "stop-token"
            if gen[-1] in stop_ids
            else "max-tokens"
            if len(gen) >= max_new
            else "kv-budget"
        )
        st = eng.status_max()

        agree = -1
        if cli.verify:  # teacher-forced vs sglang's own decode
            h.finalize(mslot, Lp, first_token)  # clean pre-B.decode seed
            cur = nti
            sgl_ref = [first_token]
            for _ in range(8):
                cur, _ = B.decode(cur, batch, model_runner)
                sgl_ref.append(int(cur[0]))
            agree = 0
            for i in range(8):
                eng.slot[0] = sgl_ref[i]
                (g_cp if (g_cp is not None and Lp + i >= CP_THRESHOLD) else g_noncp).replay()
                _sync_cuda()
                t = int(eng.head.out_id[int(eng.head.out_val.argmax().item())].item())
                agree += int(t == sgl_ref[i + 1])

        # B.decode advances the retained SGLang request during --verify, so that state cannot be
        # advertised as committed at Lp. Normal serving keeps it and restores NexRT to Lp.
        cache_retained = finish_request_cache(
            lease,
            mslot,
            Lp,
            first_token,
            reuse_cache and lifecycle_cache_key is not None and not cli.verify and st == 0,
            consensus_enabled,
            consensus_ms,
        )
        consensus_total_ms, consensus_restore_ms = consensus_metrics()
        free, _ = torch.cuda.mem_get_info(dev)
        ok = st == 0 and (
            agree < 0 or agree >= 7
        )  # permit one mismatch in eight teacher-forced tokens
        print(
            f"@@ R{tp_rank} SERVE {name:24s} L={Lp:6d} mode={handoff_mode} "
            f"ttft={ttft_s:7.3f}s prefill={prefill_s:7.2f}s "
            f"gather={gather_ms:7.3f}ms decode={decode_us:7.1f}us/tok x{len(gen) - 1:4d} "
            f"({'CP' if Lp >= CP_THRESHOLD else 'nonCP'}) stop={stop:10s} "
            f"forced={agree if agree >= 0 else '-'}/8 seed={seed} status={st} "
            f"free={free / 2**30:.2f}GiB -> {'PASS' if ok else 'CHECK'}",
            flush=True,
        )
        return {
            "name": name,
            "prompt_tokens": Lp,
            "handoff_mode": handoff_mode,
            "ttft_s": round(ttft_s, 6),
            "prefill_s": round(prefill_s, 6),
            "gather_ms": round(gather_ms, 3),
            "decode_us_per_tok": round(decode_us, 1),
            "generated_tokens": len(gen),
            "stop": stop,
            "forced_agree": agree,
            "seed": seed,
            "cache_consensus_enabled": consensus_enabled,
            "cache_reuse_enabled": reuse_cache,
            "cache_consensus_invalid_ranks": consensus_invalid_ranks,
            "cache_consensus_ttft_ms": round(consensus_ttft_ms, 3),
            "cache_consensus_restore_ms": round(consensus_restore_ms, 3),
            "cache_consensus_ms": round(consensus_total_ms, 3),
            "cache_generation": lease.generation,
            "cache_decision": lease.decision.value,
            "cache_base_tokens": lease.base_length,
            "cache_retained": cache_retained,
            "status": st,
            "gen_token_ids_head": gen[:16],
            "gen_token_ids": gen,
            "generated_text": decode_visible_text(tok, gen, stop_ids),
        }  # FULL completion (reply stays one JSON line)

    _PHASE_CODE = {
        CachePhase.EMPTY: 0,
        CachePhase.FILLING: 1,
        CachePhase.READY: 2,
        CachePhase.DECODING: 3,
    }

    def terminate_fatally(error, conn=None):
        """Leave the serving loop with a non-zero status on every TP rank."""
        print(f"@@ R{tp_rank} FATAL TP TERMINATION: {error!r}", flush=True)
        if tp_rank == 0 and conn is not None:
            try:
                conn.sendall((json.dumps({"error": "fatal TP invariant failure"}) + "\n").encode())
            except Exception:  # noqa: BLE001
                pass
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
        raise SystemExit(2)

    def handle_request(*args, **kwargs):
        """Run one request, then close the TP outcome collectively."""

        def lifecycle_state():
            snapshot = cache.snapshot()
            lease = state["active_lease"]
            return (
                snapshot.generation,
                _PHASE_CODE[snapshot.phase],
                -1 if lease is None else lease.generation,
                0,
            )

        def reset_request(reason):
            if dflash is not None:
                dflash.tap.abort()
            reset_failed_request(
                reason,
                cache=cache,
                runtime_state=state,
                model_runner=model_runner,
                engine=eng,
                mtp=mtp,
                synchronize=_sync_cuda,
            )

        return execute_request_with_consensus(
            lambda: _handle_request_impl(*args, **kwargs),
            lifecycle_state=lifecycle_state,
            gather_outcome=lambda record: gather_cache_control(record, [0.0]),
            reset_request=reset_request,
        )

    # ---- queue mode: request list identical on every rank (same files/args, same tokenizer) ----
    report = []
    idx = 0
    queued = []
    for L in [int(x) for x in cli.lens.split(",")] if cli.lens else []:
        queued.append(("len%d" % L, [int(t) for t in build_real_prompt(L, tok)], cli.gen))
    for pat in cli.files or []:
        for f in sorted(glob.glob(pat)):
            try:
                ids = _validated_prompt_ids(
                    tok(Path(f).read_text())["input_ids"],
                    cli.max_prompt,
                    min_length=3 if cli.mtp or cli.dflash else 1,
                )
            except ValueError as error:
                rp(f"@@ QUEUE {Path(f).name}: REJECT {error}", flush=True)
                continue
            queued.append((Path(f).name, ids, cli.gen_files))
    rp(f"@@ BOOT {len(queued)} queued requests", flush=True)
    for name, tokens, max_new in queued:
        try:
            r = handle_request(
                name,
                tokens,
                max_new,
                cli.seed_base + idx,
                cache_key=None if mtp is not None or dflash is not None else f"queue:{name}",
            )
            if tp_rank == 0:
                report.append(r)
        except FatalTPInvariantError as e:
            terminate_fatally(e)
        except RequestResetError as e:
            print(f"@@ R{tp_rank} SERVE {name}: RESET {e}", flush=True)
        except Exception as e:  # noqa: BLE001
            terminate_fatally(e)
        idx += 1

    for pair_index, (base_length, target_length) in enumerate(cli.cache_e2e or ()):
        e2e_tokens = [int(token) for token in build_real_prompt(target_length, tok)]

        def run_sequence(repetition, handoff_mode, consensus_enabled, record_results):
            nonlocal idx
            phase = "measure" if record_results else "warmup"
            variant = "consensus" if consensus_enabled else "baseline"
            key = f"cache-e2e:{pair_index}:{variant}:{handoff_mode}:{phase}:{repetition}"
            request_seed = cli.seed_base + max(0, repetition)
            cases = (
                ("prefill", e2e_tokens[:base_length], key, CacheDecision.PREFILL),
                ("hit", e2e_tokens[:base_length], key, CacheDecision.HIT),
                ("append", e2e_tokens, key, CacheDecision.APPEND),
                ("oracle", e2e_tokens, key + ":oracle", CacheDecision.PREFILL),
            )
            results = {}
            for role, case_tokens, case_key, expected in cases:
                idx += 1
                result = handle_request(
                    f"cache-e2e-{variant}-{handoff_mode}-{repetition}-{role}",
                    case_tokens,
                    cli.cache_e2e_gen,
                    request_seed,
                    cache_key=case_key,
                    handoff_mode=handoff_mode,
                    consensus_enabled=consensus_enabled,
                    measure_consensus=record_results,
                )
                if result["cache_decision"] != expected.value:
                    raise RuntimeError(
                        f"cache e2e {role} selected {result['cache_decision']}, "
                        f"expected {expected.value}"
                    )
                if role == "hit" and result["gather_ms"] != 0.0:
                    raise RuntimeError("exact HIT unexpectedly executed a handoff kernel")
                result["cache_e2e_role"] = role
                result["cache_e2e_pair"] = pair_index
                result["cache_e2e_repetition"] = repetition
                result["cache_e2e_phase"] = phase
                result["cache_consensus_variant"] = variant
                results[role] = result
                if tp_rank == 0 and record_results:
                    report.append(result)
            if results["prefill"]["gen_token_ids"] != results["hit"]["gen_token_ids"]:
                raise RuntimeError("exact HIT generation differs from its PREFILL baseline")
            if results["append"]["gen_token_ids"] != results["oracle"]["gen_token_ids"]:
                raise RuntimeError("suffix APPEND generation differs from full-PREFILL oracle")
            rp(
                f"@@ CACHE_E2E consensus={variant} mode={handoff_mode} "
                f"phase={phase} repetition={repetition} "
                f"base={base_length} target={target_length} -> PASS",
                flush=True,
            )
            return results

        if cli.cache_consensus_benchmark:
            for consensus_enabled in (False, True):
                for handoff_mode in ("gather", "append"):
                    run_sequence(-1, handoff_mode, consensus_enabled, False)

        for repetition in range(cli.cache_e2e_repetitions):
            consensus_order = (
                ((False, True) if repetition % 2 == 0 else (True, False))
                if cli.cache_consensus_benchmark
                else (True,)
            )
            for consensus_enabled in consensus_order:
                for handoff_mode in ("gather", "append"):
                    run_sequence(repetition, handoff_mode, consensus_enabled, True)

    if cli.cache_consensus_fault_rank is not None:
        base_length, _ = cli.cache_e2e[0]
        fault_tokens = [int(token) for token in build_real_prompt(base_length, tok)]
        fault_key = "cache-e2e:single-rank-fault"
        baseline = handle_request(
            "cache-consensus-fault-baseline",
            fault_tokens,
            cli.cache_e2e_gen,
            cli.seed_base,
            cache_key=fault_key,
            measure_consensus=True,
        )
        fallback = handle_request(
            "cache-consensus-fault-fallback",
            fault_tokens,
            cli.cache_e2e_gen,
            cli.seed_base,
            cache_key=fault_key,
            consensus_fault_rank=cli.cache_consensus_fault_rank,
            measure_consensus=True,
        )
        if baseline["cache_decision"] != CacheDecision.PREFILL.value:
            raise RuntimeError("fault-injection baseline did not select PREFILL")
        if fallback["cache_decision"] != CacheDecision.PREFILL.value:
            raise RuntimeError("single-rank invalid boundary did not force collective PREFILL")
        if fallback["cache_consensus_invalid_ranks"] != [cli.cache_consensus_fault_rank]:
            raise RuntimeError("fault-injection request did not identify exactly the injected rank")
        if baseline["gen_token_ids"] != fallback["gen_token_ids"]:
            raise RuntimeError("fault-injection PREFILL differs from its clean baseline")
        if tp_rank == 0:
            baseline["cache_e2e_role"] = "fault-baseline"
            fallback["cache_e2e_role"] = "fault-fallback"
            report.extend((baseline, fallback))
        rp(
            f"@@ CACHE_CONSENSUS_FAULT rank={cli.cache_consensus_fault_rank} "
            "-> collective PREFILL PASS",
            flush=True,
        )

    # ---- listen mode: rank 0 owns a TCP socket; header+ids broadcast keeps ranks in lockstep ----
    if cli.listen:
        import socket

        HEARTBEAT_S = 60.0  # idle re-sync period; MUST be << the 10-min NCCL watchdog timeout
        # [kind(0=req,1=quit,2=beat), token_count, max_new, seed, cache_key_utf8_bytes]
        hdr = torch.zeros(5, dtype=torch.int64, device=dev)
        sampling_cfg = torch.zeros(
            3, dtype=torch.float32, device=dev
        )  # [temperature, top_p, top_k]
        srv = conn = None
        if tp_rank == 0:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((cli.host, cli.listen))
            srv.listen(1)
            srv.settimeout(HEARTBEAT_S)
            print(
                f"@@ LISTENING on {cli.host}:{cli.listen} (one request per connection, JSON line in/out)",
                flush=True,
            )
        while True:
            # Every iteration broadcasts EXACTLY one header: request, quit, or heartbeat. Ranks 1-7
            # park in this broadcast while rank 0 waits in accept(); without the heartbeat the NCCL
            # watchdog kills the parked collective after 10 idle minutes.
            req = None
            stream_requested = False
            if tp_rank == 0:
                hdr[0] = 2
                hdr[4] = 0
                conn = None
                try:
                    conn, _addr = srv.accept()
                except socket.timeout:  # no client this beat -> heartbeat broadcast
                    pass
                if conn is not None:
                    try:
                        conn.settimeout(600.0)  # a stalled client must not park rank 0 forever
                        buf = b""
                        while not buf.endswith(b"\n"):
                            chunk = conn.recv(65536)
                            if not chunk:
                                break
                            buf += chunk
                        req = json.loads(buf.decode())
                        if not isinstance(req, dict):
                            raise ValueError("request body must be a JSON object")
                        if req.get("cmd") == "shutdown":
                            hdr[0] = 1
                        else:
                            stream_requested = req.get("stream", False)
                            if not isinstance(stream_requested, bool):
                                raise ValueError("stream must be a boolean")
                            if not cli.mtp and not cli.dflash:
                                req_temperature, req_top_p, req_top_k = resolve_sampling_config(
                                    eng.head.sampling_defaults,
                                    req.get("temperature"),
                                    req.get("top_p"),
                                    req.get("top_k"),
                                )
                            else:
                                if "top_k" in req:
                                    raise ValueError("per-request top_k requires 1-batch decode")
                                if (not cli.mtp or cli.mtp_sampling != "sample") and (
                                    "temperature" in req or "top_p" in req
                                ):
                                    raise ValueError(
                                        "per-request temperature/top_p require 1-batch decode "
                                        "or --mtp --mtp-sampling sample"
                                    )
                                req_temperature, req_top_p = _checked_sampling_config(
                                    req.get("temperature", cli.mtp_temperature),
                                    req.get("top_p", cli.mtp_top_p),
                                )
                                req_top_k = 20
                            if "file" in req:
                                _request_text(req)
                            prompt_fields = sum(key in req for key in ("text", "messages"))
                            if prompt_fields != 1:
                                raise ValueError(
                                    "request must contain exactly one of text or messages"
                                )
                            if "messages" in req:
                                messages = req["messages"]
                                if not isinstance(messages, list) or not messages:
                                    raise ValueError("messages must be a non-empty array")
                                if not all(
                                    isinstance(message, dict)
                                    and isinstance(message.get("role"), str)
                                    and "content" in message
                                    for message in messages
                                ):
                                    raise ValueError(
                                        "each message must contain string role and content"
                                    )
                                text = tok.apply_chat_template(
                                    messages,
                                    tokenize=False,
                                    add_generation_prompt=True,
                                )
                                ids = tok(text, add_special_tokens=False)["input_ids"]
                            else:
                                text = _request_text(req)
                            if "messages" not in req and req.get("chat"):
                                text = tok.apply_chat_template(
                                    [{"role": "user", "content": text}],
                                    tokenize=False,
                                    add_generation_prompt=True,
                                )
                                ids = tok(text, add_special_tokens=False)["input_ids"]
                            elif "messages" not in req:
                                ids = tok(text)["input_ids"]
                            # A hard list[int] makes odd tokenizer returns a clean reject.
                            ids = _validated_prompt_ids(
                                ids,
                                cli.max_prompt,
                                min_length=3 if cli.mtp or cli.dflash else 1,
                            )
                            hdr[0], hdr[1] = 0, len(ids)
                            request_max_new = int(req.get("max_new", cli.gen_files))
                            if request_max_new < 1:
                                raise ValueError(f"max_new must be >= 1, got {request_max_new}")
                            hdr[2] = request_max_new
                            # The transport tensor is signed int64; preserve the
                            # caller's complete low-64-bit seed pattern.
                            hdr[3] = _seed_as_int64(req.get("seed", cli.seed_base + idx + 1))
                            cache_key = req.get("cache_key")
                            if cache_key is not None:
                                cache_key_for_request(cache_key, speculative=cli.mtp or cli.dflash)
                                cache_key = str(cache_key)
                                if not cache_key:
                                    raise ValueError("cache_key must be non-empty or null")
                                cache_key_bytes = cache_key.encode("utf-8")
                                if len(cache_key_bytes) > 4096:
                                    raise ValueError("cache_key exceeds 4096 UTF-8 bytes")
                                hdr[4] = len(cache_key_bytes)
                            sampling_cfg[0] = req_temperature
                            sampling_cfg[1] = req_top_p
                            sampling_cfg[2] = req_top_k
                    except Exception as e:  # noqa: BLE001  (bad request: reject, then heartbeat)
                        hdr[0] = 2
                        req = None
                        try:
                            conn.sendall(
                                (json.dumps({"error": f"bad request: {e!r}"}) + "\n").encode()
                            )
                        except Exception:  # noqa: BLE001
                            pass
                        conn.close()
                        conn = None
            broadcast_from_rank0(hdr, "request header")
            kind = int(hdr[0])
            if kind == 2:  # heartbeat: nothing to serve, re-park
                continue
            if kind == 1:
                if tp_rank == 0:
                    conn.sendall(b'{"ok": "shutdown"}\n')
                    conn.close()
                    srv.close()
                break
            n, max_new, seed = int(hdr[1]), int(hdr[2]), int(hdr[3])
            cache_key_len = int(hdr[4])
            broadcast_from_rank0(sampling_cfg, "sampling config")
            req_temperature, req_top_p, req_top_k = sampling_cfg.tolist()
            ids_t = torch.zeros(n, dtype=torch.int32, device=dev)
            if tp_rank == 0:
                ids_t.copy_(torch.tensor(ids, dtype=torch.int32))
            broadcast_from_rank0(ids_t, "request tokens")
            cache_key = None
            if cache_key_len:
                cache_key_t = torch.zeros(cache_key_len, dtype=torch.uint8, device=dev)
                if tp_rank == 0:
                    cache_key_t.copy_(torch.tensor(list(cache_key_bytes), dtype=torch.uint8))
                broadcast_from_rank0(cache_key_t, "cache key")
                cache_key = bytes(cache_key_t.cpu().tolist()).decode("utf-8")
            idx += 1  # all ranks: keep request names/counters in lockstep
            event_stream = (
                BridgeStreamWriter(conn, tok, stop_ids)
                if tp_rank == 0 and stream_requested
                else None
            )
            dispatch_port_request(
                lambda: handle_request(
                    f"port#{idx}",
                    [int(t) for t in ids_t.tolist()],
                    max_new,
                    seed,
                    req_temperature,
                    req_top_p,
                    cache_key=cache_key,
                    event_stream=event_stream,
                    top_k=int(req_top_k),
                ),
                tp_rank=tp_rank,
                conn=conn,
                event_stream=event_stream,
                report=report,
                terminate_fatally=terminate_fatally,
                log_reset=lambda error: print(
                    f"@@ R{tp_rank} SERVE port#{idx}: RESET {error}", flush=True
                ),
            )

    if tp_rank == 0 and report:
        out = Path(os.environ.get("NEXRT_SERVE_REPORT", str(NEX_N2_5_PRO / "serve_report.jsonl")))
        with open(out, "w") as f:
            for r in report:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"@@ REPORT written: {out}", flush=True)
    print(f"@@ RANK{tp_rank}_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", type=str, default=None, help="comma synthetic prompt lengths")
    ap.add_argument("--files", type=str, nargs="*", default=None, help="prompt file globs")
    ap.add_argument("--gen", type=int, default=64, help="max new tokens for --lens requests")
    ap.add_argument(
        "--gen-files", type=int, default=256, help="max new tokens for --files / port requests"
    )
    ap.add_argument(
        "--max-prompt",
        type=int,
        default=262080,
        help="cap file/port prompts (context_length 262144 minus generation headroom)",
    )
    ap.add_argument(
        "--no-cp", action="store_true", help="do not build the context-parallel attention arm"
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="check target decoding against SGLang, or speculative commits against CPU resolution",
    )
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--dflash", action="store_true", help="DFlash M=8 greedy decoding")
    modes.add_argument(
        "--mtp",
        action="store_true",
        help="M=4 MTP decode path: bridge-side draft-prefill + MtpDraft round graph "
        "(four-row sampling by default)",
    )
    ap.add_argument(
        "--mtp-sampling",
        choices=("sample", "greedy"),
        default="sample",
        help="M=4 verification policy; greedy is the compatibility mode",
    )
    ap.add_argument(
        "--mtp-temperature",
        type=float,
        default=0.6,
        help="sampler temperature (0 is greedy-compatible)",
    )
    ap.add_argument(
        "--mtp-top-p",
        type=float,
        default=0.95,
        help="sampler nucleus threshold; top-k is fixed at 20",
    )
    ap.add_argument(
        "--mtp-arm",
        choices=("auto", "noncp", "cp"),
        default="auto",
        help="M4 arm: auto dispatches per request on prompt length vs the "
        "CP threshold; noncp/cp force one arm (debug/validation)",
    )
    ap.add_argument(
        "--mtp-cp-threshold",
        type=int,
        default=None,
        help="auto-dispatch CP threshold in tokens "
        "(default: m4_verify.M4_CP_DISPATCH_S, the measured MTP crossover)",
    )
    ap.add_argument("--host", default="127.0.0.1", help="listen address (default: local host only)")
    ap.add_argument(
        "--listen",
        type=int,
        default=None,
        help="after the queued requests, serve on this TCP port (rank 0; one JSON line "
        'request per connection; {"cmd": "shutdown"} stops the server)',
    )
    ap.add_argument(
        "--handoff-mode",
        choices=("append", "gather"),
        default="append",
        help="append is production; gather is the matched full-copy benchmark baseline",
    )
    ap.add_argument(
        "--cache-e2e",
        type=str,
        action="append",
        default=None,
        metavar="BASE,TARGET",
        help="run matched gather/append PREFILL-HIT-APPEND-oracle repetitions",
    )
    ap.add_argument(
        "--cache-e2e-repetitions", type=int, default=3, help="raw repetition count for --cache-e2e"
    )
    ap.add_argument(
        "--cache-e2e-gen", type=int, default=8, help="new-token budget for each cache e2e request"
    )
    ap.add_argument(
        "--cache-consensus-benchmark",
        action="store_true",
        help="within --cache-e2e, warm up and alternate matched consensus-off/on sequences",
    )
    ap.add_argument(
        "--cache-consensus-fault-rank",
        type=int,
        default=None,
        help="test-only: reject one rank's READY boundary and require collective PREFILL",
    )
    ap.add_argument(
        "--seed-base",
        type=int,
        default=20260712,
        help="request i samples with seed base+i unless the request specifies one",
    )
    ap.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.72,
        help="SGLang static-memory budget fraction",
    )
    ap.add_argument("--max-total-tokens", type=int, default=263168, help="SGLang KV token capacity")
    ap.add_argument(
        "--max-running-requests", type=int, default=2, help="SGLang request/Mamba slot capacity"
    )
    cli = ap.parse_args()
    rank = torchrun_rank(8)
    try:
        cli.mtp_temperature, cli.mtp_top_p = _checked_sampling_config(
            cli.mtp_temperature, cli.mtp_top_p
        )
    except ValueError as error:
        ap.error(str(error))
    if cli.gen < 1 or cli.gen_files < 1:
        ap.error("--gen and --gen-files must both be >= 1")
    if not 1 <= cli.max_prompt <= 262080:
        ap.error("--max-prompt must be in [1, 262080]")
    if (
        not cli.dflash
        and cli.no_cp
        and cli.max_prompt + max(cli.gen, cli.gen_files, cli.cache_e2e_gen) >= CP_THRESHOLD
    ):
        ap.error("--no-cp requires max prompt + generation to stay below the CP threshold")
    if (cli.mtp or cli.dflash) and cli.max_prompt < 3:
        ap.error("speculative decoding requires --max-prompt >= 3")
    if cli.mtp:
        mtp_weights = (
            Path(os.environ.get("NEXRT_LAYERS_DIR", "/path/to/nexrt-layers")) / "mtp.safetensors"
        )
        if not mtp_weights.is_file():
            ap.error(
                f"--mtp requires {mtp_weights}; prepare it with "
                "tools/extract_layers.py --mtp-src /path/to/mtp-checkpoint"
            )
    if cli.no_cp and cli.mtp and cli.mtp_arm != "noncp":
        ap.error("--no-cp with --mtp requires --mtp-arm noncp")
    if cli.cache_e2e_repetitions < 1 or cli.cache_e2e_gen < 2:
        ap.error("--cache-e2e-repetitions must be >= 1 and --cache-e2e-gen must be >= 2")
    if (
        cli.cache_consensus_benchmark or cli.cache_consensus_fault_rank is not None
    ) and not cli.cache_e2e:
        ap.error("cache consensus benchmark/fault injection requires --cache-e2e")
    if (cli.mtp or cli.dflash) and (
        cli.cache_e2e or cli.cache_consensus_benchmark or cli.cache_consensus_fault_rank is not None
    ):
        ap.error("cache-e2e and cache-consensus tests require ordinary target decoding")

    def parse_length_pair(option, value):
        try:
            lengths = tuple(int(item) for item in value.split(","))
        except ValueError:
            ap.error(f"--{option.replace('_', '-')} must contain two integer lengths")
        if len(lengths) != 2 or not 1 <= lengths[0] < lengths[1] <= cli.max_prompt:
            ap.error(
                f"--{option.replace('_', '-')} must satisfy 1 <= BASE < TARGET <= --max-prompt"
            )
        if cli.mtp and lengths[0] < 3:
            ap.error(f"--{option.replace('_', '-')} BASE must be >= 3 with --mtp")
        return lengths

    if cli.cache_e2e is not None:
        cli.cache_e2e = [parse_length_pair("cache_e2e", value) for value in cli.cache_e2e]
    if not 0 < cli.mem_fraction_static < 1:
        ap.error("--mem-fraction-static must be between 0 and 1")
    if cli.max_total_tokens <= cli.max_prompt or cli.max_running_requests < 1:
        ap.error("KV capacity must exceed --max-prompt and request capacity must be positive")
    server_args = make_server_args(
        MODEL,
        mem_fraction_static=cli.mem_fraction_static,
        max_total_tokens=cli.max_total_tokens,
        max_running_requests=cli.max_running_requests,
    )
    if cli.cache_consensus_fault_rank is not None and not (
        0 <= cli.cache_consensus_fault_rank < server_args.tp_size
    ):
        ap.error(f"--cache-consensus-fault-rank must be in [0, {server_args.tp_size - 1}]")
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)

    try:
        worker(server_args, port_args, rank, rank, cli)
    finally:
        if torch.distributed.is_initialized():
            # SGLang groups also own communicators and shared-memory broadcast buffers.
            try:
                destroy_model_parallel()
            finally:
                destroy_distributed_environment()


if __name__ == "__main__":
    main()
