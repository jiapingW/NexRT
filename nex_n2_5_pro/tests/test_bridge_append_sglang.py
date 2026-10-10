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

"""Real SGLang-pool integration test for streamed bridge append/finalize.

This intentionally stops at the handoff boundary: SGLang runs real chunked prefill, while
small tensors stand in for NexRT's resident contiguous caches. That isolates the transfer
path while still exercising the checkpoint's real model layers and SGLang cache pools.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[2]
PREFILL = ROOT / "nex_n2_5_pro" / "sglang_prefill"
NEX_N2_5_PRO = ROOT / "nex_n2_5_pro"
sys.path[:0] = [str(ROOT), str(PREFILL), str(NEX_N2_5_PRO)]

from nex_n2_5_pro.sglang_prefill.weight_loading import load_model  # noqa: E402
from nex_n2_5_pro.sglang_prefill.process_supervisor import torchrun_rank  # noqa: E402

import sglang.bench_one_batch as B  # noqa: E402
from sglang.srt.entrypoints.engine import _set_envs_and_config  # noqa: E402
from sglang.srt.layers.moe import initialize_moe_config  # noqa: E402
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config  # noqa: E402
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config  # noqa: E402

from bridge import Handoff, PortArgs, chunked_prefill, layer_sets, make_server_args  # noqa: E402
from real_prompts import build_real_prompt  # noqa: E402


def _fake_engine(full_attn, gdn_layers, cache_len, dev):
    status = lambda: torch.full((1,), 17, dtype=torch.int32, device=dev)
    attn = {
        layer: SimpleNamespace(
            k_cache=torch.full((cache_len, 256), -1, dtype=torch.bfloat16, device=dev),
            v_cache=torch.full((cache_len, 256), -1, dtype=torch.bfloat16, device=dev),
            status=status(),
        )
        for layer in full_attn
    }
    gdn = {
        layer: SimpleNamespace(
            conv_state4=torch.full((1536, 4), -1, dtype=torch.bfloat16, device=dev),
            state_t=torch.full((8, 128, 128), -1, dtype=torch.float32, device=dev),
            status=status(),
        )
        for layer in gdn_layers
    }
    moe = {layer: SimpleNamespace(status=status()) for layer in (*full_attn, *gdn_layers)}
    return SimpleNamespace(
        attn=attn,
        gdn=gdn,
        moe=moe,
        attn_cp={},
        head=SimpleNamespace(status=status()),
        slot=torch.full((2,), -1, dtype=torch.int32, device=dev),
    )


def _worker(
    server_args,
    port_args,
    gpu_id,
    rank,
    prompt_len,
    chunk,
    resume_base,
    nlayers,
    bench_iters,
    quant_method,
):
    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)
    initialize_fp4_gemm_config(server_args)
    B.configure_logger(server_args, prefix=f" TP{rank}")
    B.TreeCacheNamespace.available_and_evictable_str = lambda self: "(dummy tree cache)"
    B.TreeCacheNamespace.pretty_print = lambda self: None

    model_runner, tok = load_model(server_args, port_args, gpu_id, rank)
    dev = torch.device("cuda", torch.cuda.current_device())
    full_attn, gdn_layers = layer_sets(nlayers)
    eng = _fake_engine(full_attn, gdn_layers, prompt_len + 16, dev)
    handoff = Handoff(eng, model_runner, full_attn, gdn_layers, dev)
    tokens = [int(token) for token in build_real_prompt(prompt_len, tok)]

    appended_to = 0
    append_events = []
    idx_lifetimes = []

    def append_chunk(start, end, _next_token, batch):
        nonlocal appended_to
        assert start == appended_to, (start, appended_to)
        begin, done = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record()
        idx_lifetimes.append(
            handoff.append(int(batch.req_pool_indices[0]), start, end - start, dst_start=start)
        )
        done.record()
        append_events.append((start, end, begin, done))
        appended_to = end

    wall_start = time.time()
    first_len = resume_base or prompt_len
    next_ids, _, batch, session = chunked_prefill(
        model_runner,
        tokens[:first_len],
        chunk,
        on_chunk_batch=append_chunk,
        return_session=True,
    )
    first_token = int(next_ids[0])
    r2t = model_runner.req_to_token_pool
    first_mamba_slot = int(r2t.get_mamba_indices(batch.req_pool_indices)[0].item())
    handoff.finalize(first_mamba_slot, first_len, first_token)

    if resume_base:
        suffix_events_start = len(append_events)
        next_ids, _, batch, session = chunked_prefill(
            model_runner,
            tokens,
            chunk,
            on_chunk_batch=append_chunk,
            resume=session,
            return_session=True,
        )
        assert [(s, e) for s, e, _, _ in append_events[suffix_events_start:]][0][0] == resume_base

        def reject_hit_chunk(*_args):
            raise AssertionError("exact hit unexpectedly ran a prefill chunk")

        hit_ids, _, hit_batch, hit_session = chunked_prefill(
            model_runner,
            tokens,
            chunk,
            on_chunk_batch=reject_hit_chunk,
            resume=session,
            return_session=True,
        )
        assert int(hit_ids[0]) == int(next_ids[0])
        assert hit_batch is batch and hit_session.req is session.req

    first_token = int(next_ids[0])
    req_idx = int(batch.req_pool_indices[0])
    mamba_slot = int(r2t.get_mamba_indices(batch.req_pool_indices)[0].item())
    assert mamba_slot == first_mamba_slot, "suffix extend changed the SGLang mamba slot"
    handoff.finalize(mamba_slot, prompt_len, first_token)
    torch.cuda.synchronize()
    wall_s = time.time() - wall_start

    assert appended_to == prompt_len
    src_rows = r2t.req_to_token[req_idx, :prompt_len].long()
    pool = model_runner.token_to_kv_pool
    for layer in full_attn:
        src_k = pool.get_key_buffer(layer)[src_rows].reshape(prompt_len, 256)
        src_v = pool.get_value_buffer(layer)[src_rows].reshape(prompt_len, 256)
        assert torch.equal(eng.attn[layer].k_cache[:prompt_len], src_k), f"K mismatch layer {layer}"
        assert torch.equal(eng.attn[layer].v_cache[:prompt_len], src_v), f"V mismatch layer {layer}"
        assert bool(torch.all(eng.attn[layer].k_cache[prompt_len:] == -1))
        assert bool(torch.all(eng.attn[layer].v_cache[prompt_len:] == -1))

    for layer in gdn_layers:
        cache = r2t.mamba2_layer_cache(layer)
        assert torch.equal(eng.gdn[layer].conv_state4[:, :3], cache.conv[0][mamba_slot])
        assert bool(torch.all(eng.gdn[layer].conv_state4[:, 3] == 0))
        assert torch.equal(eng.gdn[layer].state_t, cache.temporal[mamba_slot])

    assert torch.equal(eng.slot.cpu(), torch.tensor([first_token, prompt_len]))
    statuses = [eng.head.status]
    statuses += [block.status for block in eng.gdn.values()]
    statuses += [block.status for block in eng.attn.values()]
    statuses += [block.status for block in eng.moe.values()]
    assert all(int(value.item()) == 0 for value in statuses)

    if resume_base:
        # Independent full-prefill oracle: clear the retained request, rebuild from row zero, and
        # compare its live pool/GDN state against the cache produced by prefix+suffix resume.
        model_runner.token_to_kv_pool_allocator.clear()
        model_runner.req_to_token_pool.clear()
        ref_ids, _, ref_batch = chunked_prefill(model_runner, tokens, chunk)
        ref_req_idx = int(ref_batch.req_pool_indices[0])
        ref_mamba_slot = int(r2t.get_mamba_indices(ref_batch.req_pool_indices)[0].item())
        ref_rows = r2t.req_to_token[ref_req_idx, :prompt_len].long()
        assert int(ref_ids[0]) == first_token, "suffix resume changed the first target token"
        for layer in full_attn:
            ref_k = pool.get_key_buffer(layer)[ref_rows].reshape(prompt_len, 256)
            ref_v = pool.get_value_buffer(layer)[ref_rows].reshape(prompt_len, 256)
            assert torch.equal(eng.attn[layer].k_cache[:prompt_len], ref_k), (
                f"resumed K != full-prefill K at layer {layer}"
            )
            assert torch.equal(eng.attn[layer].v_cache[:prompt_len], ref_v), (
                f"resumed V != full-prefill V at layer {layer}"
            )
        for layer in gdn_layers:
            ref_cache = r2t.mamba2_layer_cache(layer)
            assert torch.equal(
                eng.gdn[layer].conv_state4[:, :3], ref_cache.conv[0][ref_mamba_slot]
            ), f"resumed conv != full-prefill conv at layer {layer}"
            assert torch.equal(eng.gdn[layer].state_t, ref_cache.temporal[ref_mamba_slot]), (
                f"resumed state != full-prefill state at layer {layer}"
            )
        req_idx, mamba_slot = ref_req_idx, ref_mamba_slot

    append_ms = sum(begin.elapsed_time(done) for _, _, begin, done in append_events)
    print(
        f"@@ R{rank} REAL_APPEND L={prompt_len} resume_base={resume_base or '-'} "
        f"chunks={[(s, e) for s, e, _, _ in append_events]} "
        f"kv_layers={len(full_attn)} gdn_layers={len(gdn_layers)} "
        f"append={append_ms:.3f}ms wall={wall_s:.3f}s -> PASS",
        flush=True,
    )

    def bench(operation):
        samples = []
        keepalive = []
        for _ in range(3):
            keepalive.extend(operation())
        torch.cuda.synchronize()
        for _ in range(bench_iters):
            begin, done = torch.cuda.Event(True), torch.cuda.Event(True)
            begin.record()
            keepalive.extend(operation())
            done.record()
            samples.append((begin, done))
        torch.cuda.synchronize()
        values = [begin.elapsed_time(done) for begin, done in samples]
        return sum(values) / len(values), min(values), max(values)

    def full_gather():
        return [handoff.gather(req_idx, prompt_len, mamba_slot, first_token)]

    def chunk_append():
        indices = []
        for start in range(0, prompt_len, chunk):
            end = min(start + chunk, prompt_len)
            indices.append(handoff.append(req_idx, start, end - start, dst_start=start))
        handoff.finalize(mamba_slot, prompt_len, first_token)
        return indices

    gather_stats = bench(full_gather)
    append_stats = bench(chunk_append)
    if rank == 0:
        ratio = append_stats[0] / gather_stats[0]
        print(
            f"@@ HANDOFF_BENCH quantization={quant_method or 'none'} L={prompt_len} "
            f"layers={nlayers} iters={bench_iters} "
            f"gather_avg={gather_stats[0]:.3f}ms min={gather_stats[1]:.3f}ms "
            f"max={gather_stats[2]:.3f}ms append_avg={append_stats[0]:.3f}ms "
            f"min={append_stats[1]:.3f}ms max={append_stats[2]:.3f}ms "
            f"append/gather={ratio:.3f}x",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lens", type=int, default=16384)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument(
        "--resume-base",
        type=int,
        default=0,
        help="first commit this exact prefix, then append only the suffix",
    )
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--quantization", choices=("fp8",), default=None)
    parser.add_argument("--bench-iters", type=int, default=20)
    parser.add_argument("--mem-fraction-static", type=float, default=0.72)
    args = parser.parse_args()
    rank = torchrun_rank(8)
    if args.layers < 4 or args.layers > 60:
        parser.error("--layers must be in [4, 60] so the test includes full-attention KV")
    if args.bench_iters <= 0:
        parser.error("--bench-iters must be positive")
    if args.resume_base and not 0 < args.resume_base < args.lens:
        parser.error("--resume-base must be zero or in [1, --lens)")
    if not 0 < args.mem_fraction_static < 1:
        parser.error("--mem-fraction-static must be between 0 and 1")

    with open(Path(args.model) / "config.json", encoding="utf-8") as config_file:
        quant_method = json.load(config_file).get("quantization_config", {}).get("quant_method")

    server_args = make_server_args(
        args.model,
        mem_fraction_static=args.mem_fraction_static,
        context_length=max(32768, args.lens + 1024),
        quantization=args.quantization,
    )
    server_args.max_total_tokens = max(32768, args.lens + 1024)
    server_args.max_running_requests = 1
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)
    try:
        _worker(
            server_args,
            port_args,
            rank,
            rank,
            args.lens,
            args.chunk,
            args.resume_base,
            args.layers,
            args.bench_iters,
            quant_method,
        )
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
