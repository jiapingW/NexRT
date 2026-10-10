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

"""Serving-wrapper gate: captured-graph generate_mtp vs the golden stream, EXACT length.

The E2E gate requires [t0] + tokens to equal golden.pt tokens
elementwise AND in length (the budget stops generation at exactly the golden count, so
over-generation or a wrong tail token fails). Also times the replay loop wall-clock,
which now includes the real per-round 128 B drain -- the serving-loop throughput number.

Run: NEXRT_NLAYERS=60 NEXRT_MOE_RAW=1 \
     torchrun --standalone --nproc_per_node=8 tests/test_mtp_serve.py
"""

import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from decode_engine import DecodeEngine  # noqa: E402
from m4_verify import M4VerifyNonCP  # noqa: E402
from mtp.block import MtpDraft  # noqa: E402
from mtp.serve import capture_round, generate_mtp  # noqa: E402

STATE = HERE / os.environ.get("NEXRT_MTP_STATE", "states/prefill_state_60_8192")


def load_and_seed(m4, blk, hand, rank):
    kv_head = rank // 4
    length = m4.load(str(STATE), [hand["t0"], hand["d1"], 0, 0])
    blk.seed(
        hand["k"][kv_head].to(blk.dev),
        hand["v"][kv_head].to(blk.dev),
        length,
        hand["y_sel_raw"].to(blk.dev),
    )


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl")
    assert world == 8

    hand = torch.load(STATE / "mtp_handoff.pt", map_location="cpu", weights_only=False)
    gold = [
        int(t)
        for t in torch.load(STATE / "golden.pt", map_location="cpu", weights_only=False)["tokens"]
    ]

    eng = DecodeEngine(rank, world, dev)
    m4 = M4VerifyNonCP(eng, dev)
    blk = MtpDraft(eng, m4, dev)

    load_and_seed(m4, blk, hand, rank)
    torch.cuda.synchronize()
    dist.barrier()
    graph = capture_round(blk)

    load_and_seed(m4, blk, hand, rank)  # capture contract: reset state
    torch.cuda.synchronize()
    dist.barrier()
    t0_wall = time.perf_counter()
    tokens, accepts, status = generate_mtp(blk, graph, max_new_tokens=len(gold))
    wall = time.perf_counter() - t0_wall

    stream = [hand["t0"]] + tokens
    ok = status == 0 and stream == gold  # exact length AND content
    if not ok and rank == 0:
        n = min(len(stream), len(gold))
        mism = next((i for i in range(n) if stream[i] != gold[i]), None)
        print(f"len {len(stream)} vs {len(gold)}; first mismatch {mism}; status {status}")

    flag = torch.tensor([1 if ok else 0], device=dev)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    ok = int(flag.item()) == 1
    w = torch.tensor([wall], device=dev)
    dist.all_reduce(w, op=dist.ReduceOp.MAX)
    if rank == 0:
        rounds = len(accepts)
        toks = len(tokens)
        print(
            f"accepts {accepts} mean {sum(accepts) / max(1, rounds):.2f} | "
            f"{toks} tokens in {rounds} rounds, wall {w.item() * 1e3:.1f} ms "
            f"-> serving {toks / w.item():.1f} tok/s (incl. per-round drain)"
        )
        print("PASS" if ok else "FAIL")
    dist.barrier()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
