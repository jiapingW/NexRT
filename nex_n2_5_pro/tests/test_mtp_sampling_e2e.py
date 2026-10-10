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

"""Real 60-layer TP8 gate for native M=4 four-row sample-and-verify.

The synthetic sampler test proves full-vocabulary top-k/top-p and fault handling.
This gate exercises the other boundary: the sampler installed in a real
``MtpDraft.round`` (draft -> four-row target verify -> sampled commit -> catch-up),
both eagerly and as one captured graph.  Every round is checked against the
independent CPU resolver using the actual four device distributions produced by
the CUDA tail.

Run on eight H100 ranks::

    NEXRT_NLAYERS=60 NEXRT_MOE_RAW=1 \
    torchrun --standalone --nproc_per_node=8 tests/test_mtp_sampling_e2e.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import m4_control as ctlabi  # noqa: E402
from decode_engine import DecodeEngine  # noqa: E402
from m4_verify import M4VerifyNonCP  # noqa: E402
from mtp.block import MtpDraft  # noqa: E402
from mtp.sampling_ref import verify_round_from_export  # noqa: E402
from mtp.serve import capture_round, generate_mtp  # noqa: E402

TEMPERATURE = 0.6
TOP_P = 0.95
SEED = 20260808


def load_and_seed(m4, blk, state_dir, hand, rank, seed, temperature, top_p):
    """Restore every request-varying input while retaining graph-stable buffers."""
    kv_head = rank // 4
    seed_bits = int(seed) & 0xFFFFFFFFFFFFFFFF
    seed_i64 = seed_bits if seed_bits < (1 << 63) else seed_bits - (1 << 64)
    blk.eng.head.seed_dev.fill_(seed_i64)
    m4.set_sampling_config(temperature, top_p, seed=seed)
    length = m4.load(str(state_dir), [hand["t0"], hand["d1"], 0, 0])
    if length != hand["L"]:
        raise AssertionError(f"state L {length} != handoff L {hand['L']}")
    blk.seed(
        hand["k"][kv_head].to(blk.dev),
        hand["v"][kv_head].to(blk.dev),
        length,
        hand["y_sel_raw"].to(blk.dev),
    )


class RoundGate:
    def __init__(self, m4, rank, world, seed, temperature):
        self.m4 = m4
        self.rank = rank
        self.world = world
        self.seed = seed
        self.temperature = temperature
        self.rounds = 0
        self.stream: list[int] = []
        self.accepts: list[int] = []
        self.failures: list[str] = []

    def __call__(self, before, after):
        top_ids = self.m4.sv_top_ids.cpu().to(torch.int64)
        top_probs = self.m4.sv_top_probs.cpu()
        target_ids = self.m4.sv_target_ids.cpu().to(torch.int64)
        # The observer brackets the *whole* MTP round.  At its start only the
        # anchor/d1 words are final; draft_phase writes d2/d3.  After verification,
        # extend_phase advances only anchor/d1 for the next round, leaving the d2/d3
        # just consumed in place.  Reconstruct the decision-time quartet from that
        # single-writer table instead of auditing the stale pre-draft tail.
        verify_ids = [
            before[ctlabi.VERIFY_INPUT_IDS],
            before[ctlabi.VERIFY_INPUT_IDS + 1],
            after[ctlabi.VERIFY_INPUT_IDS + 2],
            after[ctlabi.VERIFY_INPUT_IDS + 3],
        ]
        committed_len = before[ctlabi.COMMITTED_LEN]
        try:
            expected = verify_round_from_export(
                top_ids,
                top_probs,
                target_ids,
                verify_ids,
                seed=self.seed,
                committed_len=committed_len,
                temperature=self.temperature,
            )
        except RuntimeError as error:
            self.failures.append(f"round {self.rounds}: {error}")
            self.rounds += 1
            return

        observed = tuple(after[ctlabi.EMITTED_IDS : ctlabi.EMITTED_IDS + expected.commit_count])
        checks = {
            "status": after[ctlabi.STATUS] == ctlabi.STATUS_OK,
            "accepted": (after[ctlabi.ACCEPTED_DRAFT_COUNT] == expected.accepted_draft_count),
            "commit": after[ctlabi.COMMIT_COUNT] == expected.commit_count,
            "output": after[ctlabi.OUTPUT_COUNT] == expected.commit_count,
            "emitted": observed == expected.emitted_ids,
            "pending": after[ctlabi.PENDING_TARGET_ID] == expected.pending_target_id,
            "length": (after[ctlabi.COMMITTED_LEN] == committed_len + expected.commit_count),
            "epoch": after[ctlabi.STEP_EPOCH] == before[ctlabi.STEP_EPOCH] + 1,
        }

        control = torch.tensor(after, dtype=torch.int32, device=self.m4.dev)
        gathered_ctl = [torch.empty_like(control) for _ in range(self.world)]
        dist.all_gather(gathered_ctl, control)
        checks["rank_control"] = all(
            torch.equal(item, gathered_ctl[0]) for item in gathered_ctl[1:]
        )

        gathered_ids = [torch.empty_like(self.m4.sv_top_ids) for _ in range(self.world)]
        gathered_probs = [torch.empty_like(self.m4.sv_top_probs) for _ in range(self.world)]
        gathered_targets = [torch.empty_like(self.m4.sv_target_ids) for _ in range(self.world)]
        dist.all_gather(gathered_ids, self.m4.sv_top_ids)
        dist.all_gather(gathered_probs, self.m4.sv_top_probs)
        dist.all_gather(gathered_targets, self.m4.sv_target_ids)
        checks["rank_distributions"] = (
            all(torch.equal(item, gathered_ids[0]) for item in gathered_ids[1:])
            and all(torch.equal(item, gathered_probs[0]) for item in gathered_probs[1:])
            and all(torch.equal(item, gathered_targets[0]) for item in gathered_targets[1:])
        )

        if not all(checks.values()):
            self.failures.append(
                f"round {self.rounds}: checks={checks} verify={verify_ids} "
                f"expected={expected} observed_ctl={after}"
            )
        self.stream.extend(observed)
        self.accepts.append(expected.commit_count)
        self.rounds += 1


def eager_rounds(blk, gate, rounds):
    ctl = blk.m4.ctl
    for _ in range(rounds):
        before = ctl.tolist()
        blk.round()
        after = ctl.tolist()
        gate(before, after)
        if after[ctlabi.STATUS] != ctlabi.STATUS_OK:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", default="states/prefill_state_60_8192")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--no-stream-match",
        action="store_true",
        help="do not require the eager and captured token streams to "
        "match: an arbitrary seed can place a draw on a CDF knife "
        "edge where the ulp-level difference between eager and "
        "captured reduce arithmetic legitimately forks the "
        "streams. Per-round exactness against the CPU verifier "
        "(the actual sampling contract) is still required for "
        "BOTH executions.",
    )
    args = parser.parse_args()
    state_dir = HERE / args.state_dir

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl")
    if world != 8:
        raise SystemExit(f"MTP sampling E2E gate requires TP8, got {world}")

    hand = torch.load(state_dir / "mtp_handoff.pt", map_location="cpu", weights_only=False)
    eng = DecodeEngine(rank, world, dev)
    m4 = M4VerifyNonCP(eng, dev)
    blk = MtpDraft(
        eng,
        m4,
        dev,
        eos_ids=[],
        sampling=True,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Eager integration: real draft and target models, exact sampled decision per round.
    load_and_seed(m4, blk, state_dir, hand, rank, args.seed, args.temperature, args.top_p)
    torch.cuda.synchronize()
    dist.barrier()
    eager = RoundGate(m4, rank, world, args.seed, args.temperature)
    eager_rounds(blk, eager, args.rounds)

    # Capture advances model and collective state. Restore the request, then exercise
    # the real serving wrapper with its opt-in pre/post-round observer.
    load_and_seed(m4, blk, state_dir, hand, rank, args.seed, args.temperature, args.top_p)
    graph = capture_round(blk)
    load_and_seed(m4, blk, state_dir, hand, rank, args.seed, args.temperature, args.top_p)
    torch.cuda.synchronize()
    dist.barrier()
    captured = RoundGate(m4, rank, world, args.seed, args.temperature)
    token_budget = 1 + args.rounds * ctlabi.M
    tokens, accepts, status = generate_mtp(
        blk,
        graph,
        max_new_tokens=token_budget,
        round_observer=captured,
    )

    # Same request seed/state must not depend on eager vs graph execution or on the
    # sampler collective generation/parity reached during capture warmup.
    comparable = min(len(eager.stream), len(captured.stream))
    replay_equal = eager.stream[:comparable] == captured.stream[:comparable] and comparable > 0
    wrapper_consistent = tokens == captured.stream[: len(tokens)] and accepts == captured.accepts
    local_ok = (
        not eager.failures
        and not captured.failures
        and eager.rounds == args.rounds
        and captured.rounds >= args.rounds
        and status == ctlabi.STATUS_OK
        and (replay_equal or args.no_stream_match)
        and wrapper_consistent
    )
    flag = torch.tensor([1 if local_ok else 0], dtype=torch.int32, device=dev)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    ok = int(flag.item()) == 1

    if rank == 0:
        print(
            f"eager rounds={eager.rounds} accepts={eager.accepts} stream_head={eager.stream[:16]}",
            flush=True,
        )
        print(
            f"graph rounds={captured.rounds} accepts={captured.accepts} "
            f"stream_head={captured.stream[:16]} status={status}",
            flush=True,
        )
        print(
            f"exact_rounds={not eager.failures and not captured.failures} "
            f"eager_graph_same_seed={replay_equal} wrapper={wrapper_consistent}",
            flush=True,
        )
        for failure in eager.failures + captured.failures:
            print(f"FAIL: {failure}", flush=True)
        print("PASS" if ok else "FAIL", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
