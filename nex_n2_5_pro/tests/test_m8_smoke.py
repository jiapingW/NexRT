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

"""M=8 smoke gate: boot the engine, run one eager M=8 unit, sanity-check the
control block, statuses, and logits; then a captured-graph replay repeats the
same checks. Wiring/launch validation only -- numerical gates live in the
external-reference and unit suites.

Run:  NEXRT_NLAYERS=4 NEXRT_MOE_RAW=1 torchrun --standalone --nproc_per_node=8 \
          tests/test_m8_smoke.py --state-dir states/prefill_state_60_512

Add ``--cp`` to exercise the context-parallel attention graph.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default="states/prefill_state_60_512")
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--replays", type=int, default=4)
    ap.add_argument(
        "--cp", action="store_true", help="exercise the M=8 context-parallel attention arm"
    )
    cli = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl")
    assert world == 8

    from decode_engine import DecodeEngine
    from m8_verify import M8Verify
    import m8_control as ctl8

    eng = DecodeEngine(rank, world, dev, cp=cli.cp)
    if cli.cp:
        from m8_cp import M8CpVerify

        m8 = M8CpVerify(eng, dev)
    else:
        m8 = M8Verify(eng, dev)
    m8.set_seed(cli.seed)

    state_dir = HERE / cli.state_dir
    # Arbitrary teacher-forced ids (valid vocab range) -- semantics do not matter
    # for the smoke gate, only wiring and the decision contract.
    ids = [11, 22, 33, 44, 55, 66, 77, 88]
    length = m8.load(state_dir, ids)

    fails = []

    def check(name, ok, detail=""):
        flag = torch.tensor([0 if ok else 1], device=dev)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(f"   [{'OK' if flag.item() == 0 else 'FAIL'}] {name} {detail}", flush=True)
        if flag.item() != 0:
            fails.append(name)

    # ---- eager unit ----
    m8.run()
    torch.cuda.synchronize()
    out = m8.outputs()
    c = out["m8_control"]
    check("eager status == 0", out["status"] == 0, f"(status={out['status']})")
    check("eager commit in 1..8", 1 <= c["commit_count"] <= 8, f"(commit={c['commit_count']})")
    check(
        "eager committed_len advanced",
        c["committed_len"] == length + c["commit_count"],
        f"(C {length} -> {c['committed_len']})",
    )
    check("eager epoch == 1", c["step_epoch"] == 1)
    check("eager cursor consumed", int(m8.cursor[8].item()) == 8 and int(m8.cursor[9].item()) == 0)
    check("eager logits finite", bool(torch.isfinite(out["logits8_true"]).all()))
    check(
        "eager top_ids valid",
        bool((m8.sv_top_ids[:, :20] >= 0).all()) and bool((m8.sv_top_ids[:, :20] < 248320).all()),
    )
    # rank identity of the whole control block
    ctl_all = [torch.zeros_like(m8.ctl) for _ in range(world)]
    dist.all_gather(ctl_all, m8.ctl)
    same = all(bool(torch.equal(ctl_all[0], t)) for t in ctl_all)
    check("eager ctl rank-identical", same)

    # ---- three more eager units (epoch/ring advance) ----
    for _ in range(3):
        m8.run()
    torch.cuda.synchronize()
    c = ctl8.read_control_block(m8.ctl)
    check("eager x4 epochs", c["step_epoch"] == 4)
    check("eager status still 0", c["status"] == 0)

    # ---- captured graph: capture once, replay across changing control values ----
    m8.load(state_dir, ids)
    g = m8.capture(warmup=3)
    length = m8.load(state_dir, ids)
    for _ in range(cli.replays):
        g.replay()
    torch.cuda.synchronize()
    c = ctl8.read_control_block(m8.ctl)
    check("graph status == 0", max(c["status"], m8.status_max()) == 0, f"(status={c['status']})")
    check("graph epochs == replays", c["step_epoch"] == cli.replays, f"(epoch={c['step_epoch']})")
    check(
        "graph committed_len grew",
        c["committed_len"] > length,
        f"(C {length} -> {c['committed_len']})",
    )
    ctl_all = [torch.zeros_like(m8.ctl) for _ in range(world)]
    dist.all_gather(ctl_all, m8.ctl)
    same = all(bool(torch.equal(ctl_all[0], t)) for t in ctl_all)
    check("graph ctl rank-identical", same)

    if rank == 0:
        verdict = "PASS" if not fails else f"FAIL ({fails})"
        arm = "CP" if cli.cp else "non-CP"
        print(
            f"== M8 {arm} smoke ({len(eng.layers)} layers, state {cli.state_dir}): {verdict} ==",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
