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

"""Same-boot BS=1 whole-step vs MTP whole-round benchmark (TP=8 rank-max).

The acceptance-independent engineering comparison:

    overhead(S) = T_mtp_full_round(S) / T_bs1_full_step(S)

Both series are measured in ONE boot from the same seeded states, interleaved
per trial (MTP then BS=1, --trials times), as pure captured-graph replays:

  T_bs1_full_step  = one BS=1 graph replay (embed, 60 layers, lm_head, in-graph
                     sampler/commit, next-token state), dispatched by the BS=1
                     policy (CP graph at S >= its 48K crossover).
  T_mtp_full_round = one captured MtpDraft round replay (draft chain, complete
                     M=4 target verification, four-row lm_head, SAMPLED
                     verify/commit tail, T=4 extension, next-round draft/control
                     state), dispatched by the MTP policy (CP verification arm
                     at S >= M4_CP_DISPATCH_S).

Prefill, handoff, graph capture, and host readback are excluded (both loops are
event-timed back-to-back replays). Each length re-seeds from a real prefill
state; lengths above the largest state use its deterministic repeated-KV
extension (timing fixture).

Run: NEXRT_NLAYERS=60 NEXRT_MOE_RAW=1 torchrun --standalone \
     --nproc_per_node=8 tests/bench_mtp_vs_bs1.py --state-root <states> \
     --lengths 512,1024,4096,8192,16384,32768,65536,131072
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import m4_control as ctlabi  # noqa: E402
from decode_engine import DecodeEngine, CP_DISPATCH_S  # noqa: E402
from m4_verify import M4VerifyNonCP, M4VerifyCp, M4_CP_DISPATCH_S  # noqa: E402


def wall_us(graph, reps, dev):
    """Block-averaged wall per replay, MAX across ranks."""
    torch.cuda.synchronize()
    dist.barrier()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        graph.replay()
    e.record()
    torch.cuda.synchronize()
    t = torch.tensor([s.elapsed_time(e) * 1000.0 / reps], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="512,1024,4096,8192,16384,32768,65536,131072")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--state-root", type=Path, default=HERE / "states")
    ap.add_argument("--out", default="", help="write the result JSON here (else stdout)")
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",") if x]

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl")
    assert world == 8

    from mtp.block import MtpDraft  # noqa: E402  (after path setup)

    real_states = sorted(
        int(p.name.rsplit("_", 1)[1]) for p in args.state_root.glob("prefill_state_60_*")
    )

    def state_for(L):
        if L in real_states:
            return args.state_root / f"prefill_state_60_{L}", L, "real_prefill"
        seed = max(s for s in real_states if s <= L)
        return (
            args.state_root / f"prefill_state_60_{seed}",
            seed,
            "deterministic_repeated_kv_extension",
        )

    eng = DecodeEngine(rank, world, dev, cp=True)

    need_cp_arm = any(L >= M4_CP_DISPATCH_S for L in lengths)
    arms = {}
    for name in ("noncp", "cp") if need_cp_arm else ("noncp",):
        m4 = M4VerifyNonCP(eng, dev) if name == "noncp" else M4VerifyCp(eng, dev)
        blk = MtpDraft(eng, m4, dev, eos_ids=[], sampling=True)
        arms[name] = (m4, blk)

    handoffs = {}

    def extend_kv_prefix(seed_len, L):
        for block in list(eng.attn.values()):
            for cache in (block.k_cache, block.v_cache):
                w = seed_len
                while w < L:
                    c = min(seed_len, L - w)
                    cache[w : w + c].copy_(cache[:c])
                    w += c

    def mtp_extend(blk, seed_len, L):
        for cache in (blk.attn.k_cache, blk.attn.v_cache):
            w = seed_len
            while w < L:
                c = min(seed_len, L - w)
                cache[w : w + c].copy_(cache[:c])
                w += c

    def relocate(m4, L, quartet):
        ctlabi.init_control_block(m4.ctl, L, quartet)
        eng.slot[0] = quartet[0]
        eng.slot[1] = L
        m4.S0 = L
        for layer, gdn in eng.gdn.items():
            ring = m4.conv_ring[layer]
            ring.zero_()
            for k in range(3):
                ring[:, ctlabi.conv_slot(L - 3 + k)].copy_(gdn.conv_state4[:, k])
            state = m4.state_ring[layer]
            state.zero_()
            state[ctlabi.state_slot(L - 1)].copy_(gdn.state_t)
        m4.cp_slots4.zero_()

    def reset(arm, L, load=True):
        """Seed (from disk when load=True) and relocate both families to length L."""
        m4, blk = arms[arm]
        sd, seed_len, kind = state_for(L)
        if load:
            if sd not in handoffs:
                handoffs[sd] = torch.load(
                    sd / "mtp_handoff.pt", map_location="cpu", weights_only=False
                )
            hand = handoffs[sd]
            quartet = [int(hand["t0"]), int(hand["d1"]), 0, 0]
            Lp = m4.load(str(sd), quartet)
            assert Lp == seed_len
            blk.seed(
                hand["k"][rank // 4].to(dev),
                hand["v"][rank // 4].to(dev),
                Lp,
                hand["y_sel_raw"].to(dev),
            )
            if L > seed_len:
                extend_kv_prefix(seed_len, L)
                mtp_extend(blk, seed_len, L)
        hand = handoffs[sd]
        blk.y_sel.copy_(hand["y_sel_raw"].to(dev))
        relocate(m4, L, (int(hand["t0"]), int(hand["d1"]), 0, 0))
        for s in (
            m4.ws_status,
            m4.ws_moe_status,
            m4.ws12_status,
            m4.ws21_status,
            blk.attn.status,
            blk.moe.status,
        ):
            s.zero_()
        m4.cursor.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        return kind

    # ---- capture: BS=1 graphs once, one MTP round graph per arm ----
    first = lengths[0]
    reset("noncp", first)
    g_bs1_noncp = eng.capture(warmup=8)
    reset("noncp", first)
    g_bs1_cp = eng.capture(warmup=8, step_fn=eng.step_cp)
    g_round = {}
    for name, (m4, blk) in arms.items():
        L0 = first if name == "noncp" else max(lengths)
        reset(name, L0)
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            for _ in range(3):
                blk.round()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        dist.barrier()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            blk.round()
        torch.cuda.synchronize()
        dist.barrier()
        g_round[name] = g

    rows = []
    for L in lengths:
        arm = "cp" if (need_cp_arm and L >= M4_CP_DISPATCH_S) else "noncp"
        bs1_graph = g_bs1_cp if L >= CP_DISPATCH_S else g_bs1_noncp
        mtp_us, bs1_us = [], []
        reset(arm, L)  # one disk seed per length
        for _ in range(args.trials):  # interleave MTP / BS=1
            reset(arm, L, load=False)
            for _ in range(args.warmup):
                g_round[arm].replay()
            mtp_us.append(wall_us(g_round[arm], args.reps, dev))
            eng.slot[1] = L  # stage the BS=1 step at S=L
            for _ in range(args.warmup):
                bs1_graph.replay()
            bs1_us.append(wall_us(bs1_graph, args.reps, dev))
        row = {
            "length": L,
            "fixture": state_for(L)[2],
            "mtp_arm": arm,
            "bs1_graph": "cp" if L >= CP_DISPATCH_S else "noncp",
            "bs1_us": round(statistics.median(bs1_us), 1),
            "mtp_round_us": round(statistics.median(mtp_us), 1),
            "bs1_trials": [round(v, 1) for v in bs1_us],
            "mtp_trials": [round(v, 1) for v in mtp_us],
        }
        row["overhead"] = round(row["mtp_round_us"] / row["bs1_us"], 3)
        rows.append(row)
        if rank == 0:
            print(
                f"[S={L:>6}] bs1 {row['bs1_us']:7.1f} us/step ({row['bs1_graph']:5s})  "
                f"mtp {row['mtp_round_us']:7.1f} us/round ({arm:5s})  "
                f"overhead {row['overhead']:.3f}x",
                flush=True,
            )

    if rank == 0:
        result = {
            "rows": [
                {
                    "ctx": str(r["length"]),
                    "bs1_tok_s": round(1e6 / r["bs1_us"], 1),
                    "bs1_us": r["bs1_us"],
                    "mtp_round_us": r["mtp_round_us"],
                    "mtp_arm": r["mtp_arm"],
                    "bs1_graph": r["bs1_graph"],
                    "overhead": r["overhead"],
                }
                for r in rows
            ],
            "trials": rows,
        }
        text = json.dumps(result, indent=1)
        if args.out:
            Path(args.out).write_text(text)
            print(f"wrote {args.out}", flush=True)
        else:
            print(text, flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
