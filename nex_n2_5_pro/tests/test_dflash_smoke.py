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

"""TP=8 complete-round DFlash smoke gate (run on the target GPU host).

This constructs the real six-layer draft, captures proposal + M=8 verification
as one graph, restores the request, and replays multiple dynamic-accept rounds.

Run:

    NEXRT_NLAYERS=60 NEXRT_MOE_RAW=1 \
      torchrun --standalone --nproc_per_node=8 \
      tests/test_dflash_smoke.py --state-dir states/prefill_state_60_8192
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

import m8_control as control  # noqa: E402
from decode_engine import DecodeEngine  # noqa: E402
from dflash.block import DFlashBlock  # noqa: E402
from dflash.config import DFlashConfig  # noqa: E402
from dflash.serve import capture_request, generate_dflash  # noqa: E402
from m8_verify import M8Verify  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", default="states/prefill_state_60_8192")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument(
        "--check-golden", action="store_true", help="require the returned stream to match golden.pt"
    )
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    if world != 8:
        raise RuntimeError(f"DFlash requires TP=8, got {world}")

    state_dir = HERE / args.state_dir
    golden = torch.load(state_dir / "golden.pt", map_location="cpu", weights_only=False)
    golden_tokens = [int(token) for token in golden["tokens"]]
    first_token = golden_tokens[0]

    config = DFlashConfig.load()
    engine = DecodeEngine(rank, world, device)
    verifier = M8Verify(engine, device, dflash_tap_layers=config.target_layer_ids)
    block = DFlashBlock.from_checkpoint(engine, verifier, device, config=config)
    graph, initial_length = capture_request(block, state_dir, first_token, warmup=3)
    torch.cuda.synchronize()
    dist.barrier()

    max_tokens = min(len(golden_tokens), 1 + args.rounds * control.M)
    tokens, accepts, status = generate_dflash(
        block,
        graph,
        max_new_tokens=max_tokens,
        context_limit=config.max_position_embeddings,
        eos_ids=(),
    )
    stream = [first_token] + tokens
    snapshot = control.read_control_block(verifier.ctl)
    local_ok = (
        status == 0
        and len(tokens) == max_tokens - 1
        and bool(accepts)
        and all(1 <= accepted <= control.M for accepted in accepts)
        and snapshot["committed_len"] > initial_length
    )
    if args.check_golden:
        local_ok = local_ok and stream == golden_tokens[:max_tokens]

    controls = [torch.empty_like(verifier.ctl) for _ in range(world)]
    dist.all_gather(controls, verifier.ctl)
    local_ok = local_ok and all(bool(torch.equal(controls[0], value)) for value in controls[1:])
    flag = torch.tensor([1 if local_ok else 0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    ok = int(flag.item()) == 1
    if rank == 0:
        print(
            f"DFlash rounds={len(accepts)} accepts={accepts} "
            f"tokens={len(tokens)} status={status} "
            f"C={initial_length}->{snapshot['committed_len']}",
            flush=True,
        )
        print("DFLASH SMOKE PASS" if ok else "DFLASH SMOKE FAIL", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
