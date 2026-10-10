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

"""Request loading, graph capture, and replay loop for DFlash block-8 decode."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable

import torch

from nex_n2_5_pro import m8_control as control
from nex_n2_5_pro.dflash.block import DFlashBlock
from nex_n2_5_pro.dflash.config import M


def capture_request(
    block: DFlashBlock, state_dir: str | Path, first_token: int, warmup: int = 3
) -> tuple[torch.cuda.CUDAGraph, int]:
    """Load, warm/capture one request-base graph, then restore pristine state.

    The draft attention launch bakes the request's absolute cache base into the
    graph. A graph can be replayed for all rounds of that request; a different
    prefill length must be captured separately.
    """
    committed_len = block.load_request(state_dir, first_token)
    graph = block.capture(warmup=warmup)
    restored_len = block.load_request(state_dir, first_token)
    if restored_len != committed_len:
        raise RuntimeError(
            f"prefill length changed across capture: {committed_len} -> {restored_len}"
        )
    return graph, committed_len


def generate_dflash(
    block: DFlashBlock,
    graph: torch.cuda.CUDAGraph,
    max_new_tokens: int,
    *,
    context_limit: int | None = None,
    eos_ids: tuple[int, ...] | list[int],
    on_round: Callable[[dict, dict], None] | None = None,
    on_tokens: Callable[[list[int]], None] | None = None,
) -> tuple[list[int], list[int], int]:
    """Replay complete DFlash rounds until budget, EOS, capacity, or failure.

    Returns ``(tokens, accepts, status)``. ``tokens`` excludes the pending
    anchor already produced by prefill, while ``max_new_tokens`` includes that
    anchor (the same convention as ``mtp.serve.generate_mtp``). ``accepts`` is
    the target commit count for each completed round.

    The caller supplies the target request's stop IDs; draft EOS is not a
    substitute for the target tokenizer's chat termination policy.

    The graph contains no host-mirror publication node, so this loop drains the
    128-byte M=8 control block after each replay. The readback does not add a
    kernel to the captured round.
    """
    max_new_tokens = int(max_new_tokens)
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    budget = max_new_tokens - 1
    if budget <= 0:
        return [], [], 0

    hard_limit = min(block.cache_capacity, block.config.max_position_embeddings)
    if context_limit is not None:
        hard_limit = min(hard_limit, int(context_limit))
    eos = set(int(token) for token in eos_ids)
    if int(block.verifier.ctl[control.PENDING_TARGET_ID].item()) in eos:
        return [], [], 0

    tokens: list[int] = []
    accepts: list[int] = []
    status = 0
    committed_len = int(block.verifier.ctl[control.COMMITTED_LEN].item())
    while budget > 0 and committed_len + M <= hard_limit:
        before = control.read_control_block(block.verifier.ctl) if on_round else None
        graph.replay()
        snapshot = control.read_control_block(block.verifier.ctl)
        status = int(snapshot["status"])
        if status:
            break
        count = int(snapshot["output_count"])
        commit = int(snapshot["commit_count"])
        if not 1 <= count <= M or commit != count:
            status = control.STATUS_BAD_COMMIT_COUNT
            break
        if on_round is not None:
            on_round(before, snapshot)
        emitted = [int(token) for token in snapshot["emitted_ids"][:count]]
        take = min(count, budget)
        stop = False
        published = []
        for token in emitted[:take]:
            tokens.append(token)
            published.append(token)
            budget -= 1
            if token in eos:
                stop = True
                break
        if on_tokens is not None and published:
            on_tokens(published)
        accepts.append(commit)
        committed_len = int(snapshot["committed_len"])
        if stop:
            break
    return tokens, accepts, status
