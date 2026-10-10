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

"""The MTP serving wrapper: capture one round, replay until budget/stop.

Derived from DecodeEngine.capture/generate but kept as a SEPARATE path (bs=1 stays
untouched). Per-round host work mirrors bs=1 exactly: one graph.replay() dispatch plus one
read of the round's 128 B control block from its host mirror (published by the round's last
node; see ctl_mirror.py). The next round is queued BEFORE the previous round's words are read
whenever those words cannot end the loop by budget or context limit, so the GPU never drains
between rounds. No other host work is in the loop.

Contract (same as the bs=1 capture contract): capture records launches after warmup, so
the caller MUST re-load/seed state after capture_round() and before generate_mtp().
Context accounting: the KV allocation already carries slack (CACHE_LEN = 256K + 1024),
so the loop only enforces the ordinary budget/context stop -- no special headroom logic.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from nex_n2_5_pro import m4_control as ctlabi


def capture_round(blk, warmup: int = 3) -> torch.cuda.CUDAGraph:
    """Warm up blk.round() on a side stream, then capture it as ONE CUDA graph."""
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        for _ in range(warmup):
            blk.round()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        blk.round()
    torch.cuda.synchronize()
    dist.barrier()
    return g


def generate_mtp(
    blk,
    graph: torch.cuda.CUDAGraph,
    max_new_tokens: int,
    context_limit: int | None = None,
    round_observer=None,
    on_tokens=None,
):
    """Replay rounds until the token budget, a stop token, or the context limit.

    Round k+1 is queued before round k's control words are read whenever round k cannot end
    the loop by budget (a round emits at most M tokens) or context limit (it commits at most
    M rows); a stop token or a nonzero status is therefore learned one round late, the
    speculative round runs to completion and its tokens are not emitted (device state has
    advanced by that one round, exactly as after a forced round). The stream, accepts and
    status are identical to a drain-every-round loop, and the stream is synchronized before
    returning, so the caller sees a quiescent device.

    Returns (tokens, accepts, status): `tokens` are the emitted ids AFTER the pending
    anchor t0 (the caller holds t0 from the hand-off; the full stream is [t0] + tokens,
    so t0 counts against max_new_tokens exactly like bs=1's first_token). `accepts` is
    the per-round commit_count list; nonzero `status` means a fail-closed round (its
    round emitted nothing and state did not advance).

    ``round_observer`` is an opt-in correctness hook, never used by the production
    fast path.  When provided, it is called as ``observer(before_words, after_words)``
    once per replay.  The pre-replay snapshot is necessary because the captured MTP
    extend phase has already installed the *next* draft quartet by the time the normal
    post-replay control-block drain completes.

    ``on_tokens`` receives each round's committed output IDs after host-mirror publication.
    It excludes the pending anchor held by the caller and must not block.
    """
    if round_observer is not None:  # the observer reads live device buffers
        return _generate_mtp_observed(
            blk, graph, max_new_tokens, context_limit, round_observer, on_tokens
        )
    mirror = blk.mirror
    tokens: list[int] = []
    accepts: list[int] = []
    budget = max_new_tokens - 1  # t0 already counts as one token
    status = 0
    if budget <= 0:
        return tokens, accepts, status
    # The context guard needs the committed length before the first round; the device is
    # quiescent at entry, so this one 128 B read is off the round path.
    c_known = int(blk.m4.ctl[ctlabi.COMMITTED_LEN].item()) if context_limit is not None else 0
    gen = mirror.last_gen()
    graph.replay()
    queued = 1
    while True:
        ahead = budget > ctlabi.M and (
            context_limit is None or c_known + 2 * ctlabi.M <= context_limit
        )
        if ahead:
            graph.replay()  # round k+1 queued behind round k
            queued = 2
        gen += 1
        w = mirror.wait(gen)  # round k's words, from host memory
        queued -= 1
        status = w[ctlabi.STATUS]
        if status != 0:
            break
        take = min(w[ctlabi.OUTPUT_COUNT], budget)
        emitted = w[ctlabi.EMITTED_IDS : ctlabi.EMITTED_IDS + take]
        tokens.extend(emitted)
        if on_tokens is not None and emitted:
            on_tokens(emitted)
        accepts.append(w[ctlabi.COMMIT_COUNT])
        budget -= take
        c_known = w[ctlabi.COMMITTED_LEN]
        if w[ctlabi.FLAGS] & ctlabi.FLAG_STOP_HIT:
            break
        if context_limit is not None and c_known + ctlabi.M > context_limit:
            break
        if budget <= 0:
            break
        if not ahead:
            graph.replay()
            queued = 1
    if queued:  # a speculative round may still be running
        torch.cuda.current_stream().synchronize()
    return tokens, accepts, status


def _generate_mtp_observed(
    blk, graph, max_new_tokens, context_limit, round_observer, on_tokens=None
):
    """The drain-every-round loop, kept for the correctness observer (never the fast path)."""
    ctl = blk.m4.ctl
    tokens: list[int] = []
    accepts: list[int] = []
    budget = max_new_tokens - 1
    status = 0
    while budget > 0:
        before = ctl.tolist()
        graph.replay()
        w = ctl.tolist()  # ONE 128 B D2H; completes the round
        round_observer(before, w)
        status = w[ctlabi.STATUS]
        if status != 0:
            break
        take = min(w[ctlabi.OUTPUT_COUNT], budget)
        emitted = w[ctlabi.EMITTED_IDS : ctlabi.EMITTED_IDS + take]
        tokens.extend(emitted)
        if on_tokens is not None and emitted:
            on_tokens(emitted)
        accepts.append(w[ctlabi.COMMIT_COUNT])
        budget -= take
        if w[ctlabi.FLAGS] & ctlabi.FLAG_STOP_HIT:
            break
        if context_limit is not None and w[ctlabi.COMMITTED_LEN] + ctlabi.M > context_limit:
            break
    return tokens, accepts, status
