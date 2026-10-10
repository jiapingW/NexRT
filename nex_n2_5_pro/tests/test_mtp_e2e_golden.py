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

"""E2E golden gate: the complete MTP loop on the REAL 60-layer engine vs the golden stream.

One replay = draft_phase (2x T=1 chain) -> m4.run() (real M=4 verify + commit)
-> extend_phase (prep + T=4 catch-up + next d1). Under greedy verification the emitted
stream must be BIT-IDENTICAL to the teacher-forced golden decode stream regardless of
draft quality -- that is the speculative-decoding invariant this test gates.

Also reports the measured accept lengths (the observed accept lengths of
nex_n2_5_pro's MTP head on this engine) and, with --bench, graph-captures one round and
times replays.

Run: NEXRT_NLAYERS=60 NEXRT_MOE_RAW=1 \
     torchrun --standalone --nproc_per_node=8 tests/test_mtp_e2e_golden.py \
       --state-dir states/prefill_state_60_8192 [--rounds 24] [--bench]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import m4_control as ctlabi  # noqa: E402
from decode_engine import DecodeEngine  # noqa: E402
from m4_verify import M4VerifyNonCP  # noqa: E402
from mtp.block import MtpDraft  # noqa: E402


def load_and_seed(m4, blk, state_dir, hand, rank):
    kv_head = rank // 4
    length = m4.load(str(state_dir), [hand["t0"], hand["d1"], 0, 0])
    assert length == hand["L"], f"state L {length} != handoff L {hand['L']}"
    blk.seed(
        hand["k"][kv_head].to(blk.dev),
        hand["v"][kv_head].to(blk.dev),
        length,
        hand["y_sel_raw"].to(blk.dev),
    )
    return length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default="states/prefill_state_60_8192")
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--bench", action="store_true")
    cli = ap.parse_args()
    state_dir = HERE / cli.state_dir

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl")
    assert world == 8

    hand = torch.load(state_dir / "mtp_handoff.pt", map_location="cpu", weights_only=False)
    golden = torch.load(state_dir / "golden.pt", map_location="cpu", weights_only=False)
    gold_tokens = [int(t) for t in golden["tokens"]]
    assert gold_tokens[0] == hand["t0"]

    eng = DecodeEngine(rank, world, dev)
    m4 = M4VerifyNonCP(eng, dev)
    blk = MtpDraft(eng, m4, dev)
    load_and_seed(m4, blk, state_dir, hand, rank)
    torch.cuda.synchronize()
    dist.barrier()

    # ---- correctness loop (eager) -------------------------------------------------------
    stream = [hand["t0"]]
    accepts = []
    status = 0
    for _ in range(cli.rounds):
        blk.round()
        torch.cuda.synchronize()
        ctl = ctlabi.read_control_block(m4.ctl)
        status = max(status, ctl["status"])
        if status:
            break
        L = ctl["output_count"]
        stream.extend(ctl["emitted_ids"][:L])
        accepts.append(ctl["commit_count"])
        if ctl["flags"] & ctlabi.FLAG_STOP_HIT or len(stream) >= len(gold_tokens):
            break

    ncmp = min(len(stream), len(gold_tokens))
    mismatch = next((i for i in range(ncmp) if stream[i] != gold_tokens[i]), -1)
    ok = status == 0 and mismatch < 0 and ncmp >= len(gold_tokens)
    if rank == 0:
        print(f"accept lengths: {accepts}  mean {sum(accepts) / max(1, len(accepts)):.2f}")
        print(
            f"stream[{len(stream)}] vs golden[{len(gold_tokens)}]: "
            f"{'exact prefix match' if mismatch < 0 else f'MISMATCH at {mismatch}: {stream[mismatch]} != {gold_tokens[mismatch]}'}"
            f"{' (short)' if mismatch < 0 and ncmp < len(gold_tokens) else ''}  status={status}"
        )
    flag = torch.tensor([1 if ok else 0], device=dev)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    ok = int(flag.item()) == 1
    if rank == 0:
        print("PASS" if ok else "FAIL")

    # ---- optional bench: capture one round, time replays --------------------------------
    if ok and cli.bench:
        load_and_seed(m4, blk, state_dir, hand, rank)
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            for _ in range(3):
                blk.round()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        dist.barrier()
        load_and_seed(m4, blk, state_dir, hand, rank)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            blk.round()
        torch.cuda.synchronize()
        dist.barrier()
        load_and_seed(m4, blk, state_dir, hand, rank)
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        reps = 30
        for _ in range(reps):
            g.replay()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / reps
        acc = torch.tensor([dt], device=dev)
        dist.all_reduce(acc, op=dist.ReduceOp.MAX)
        if rank == 0:
            ac = sum(accepts) / max(1, len(accepts))
            print(
                f"[bench] round (graph replay): {acc.item() * 1e3:.3f} ms  "
                f"-> at measured AC {ac:.2f}: {ac / acc.item():.1f} tok/s"
            )
    dist.barrier()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
