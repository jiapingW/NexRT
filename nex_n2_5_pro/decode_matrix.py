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

"""Whole-step decode benchmark: load the engine and capture once, then sweep context lengths.

The 60-layer weight load/transform is paid ONCE and amortized across every length; each length is then just
load_prefill + replay. If a state directory contains `golden.pt`, the driver also compares the engine's argmax
against that saved reference token stream. Speed is pure free-running replay with the device-resident sequence
slot and no per-token host work.

Run:  NEXRT_NLAYERS=60 torchrun --standalone --nproc_per_node=8 nex_n2_5_pro/decode_matrix.py \
        [--lengths 512,1024,2048,4096,8192] [--steps 32]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(HERE.parent))

from nex_n2_5_pro.decode_engine import DecodeEngine, CP_MIN_S, CP_DISPATCH_S  # noqa: E402
from nex_n2_5_pro.attn import CACHE_LEN  # noqa: E402  (compile-time KV budget -> the guard)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=str, default="512,1024,2048,4096,8192")
    ap.add_argument("--steps", type=int, default=32, help="decode tokens per context length")
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument(
        "--reps", type=int, default=300, help="pure-replay reps for the speed measurement"
    )
    ap.add_argument(
        "--trials", type=int, default=3, help="same-process timing trials; report median of TP-max"
    )
    ap.add_argument(
        "--state-root",
        type=Path,
        default=HERE,
        help="directory containing prefill_state_60_<length> folders",
    )
    ap.add_argument(
        "--cp",
        action="store_true",
        help="also build and benchmark the CP long-context graph; "
        "when golden.pt is present, validate the combined non-CP-to-CP trajectory",
    )
    ap.add_argument(
        "--cp-threshold",
        type=int,
        default=CP_DISPATCH_S,
        help="S at/above which the run switches to "
        "the CP graph (measured post-WGMMA E2E crossover: 48K on 60-layer TP8)",
    )
    cli = ap.parse_args()
    N = cli.steps
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl")
    assert world == 8
    if cli.cp:
        assert cli.cp_threshold > CP_MIN_S, (
            f"--cp-threshold must be > CP_MIN_S={CP_MIN_S} (CP invalid at small S)"
        )

    eng = DecodeEngine(rank, world, dev, cp=cli.cp)  # the init wall -- paid ONCE

    # Resolve testable lengths: prefill state present and L+N within the compile-time KV budget.
    avail = []
    for L in [int(x) for x in cli.lengths.split(",")]:
        sd = cli.state_root / f"prefill_state_60_{L}"
        if not (sd / f"prefill_state_rank{rank}.pt").exists():
            if rank == 0:
                print(f"   L={L}: prefill state missing -> skip", flush=True)
        elif L + N > CACHE_LEN:
            if rank == 0:
                print(
                    f"   L={L}: L+N={L + N} > CACHE_LEN={CACHE_LEN} -> skip (raise CACHE_LEN to test)",
                    flush=True,
                )
        else:
            avail.append((L, sd))
    if not avail:
        if rank == 0:
            print("no testable lengths on disk", flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return

    eng.load_prefill(state_dir=avail[0][1])  # warm up from the first state
    capture_stats = []

    def _capture(label, fn):
        torch.cuda.synchronize()
        free0, _ = torch.cuda.mem_get_info()
        t0 = time.perf_counter()
        graph = eng.capture(warmup=cli.warmup, step_fn=fn)
        torch.cuda.synchronize()
        free1, _ = torch.cuda.mem_get_info()
        capture_stats.append((label, time.perf_counter() - t0, (free0 - free1) / (1024**2)))
        return graph

    g = _capture("selected non-CP", eng.step)  # device-S -> serves all L
    g_cp = _capture("CP", eng.step_cp) if cli.cp else None  # shares model/cache buffers
    if rank == 0:
        a0 = next(iter(eng.attn.values()))
        print(
            f"   runtime wiring: non-CP=WGMMA/{a0.nctas} CTAs; "
            f"MoE L1 split-K=4; CP={'mega' if eng.attn_cp else 'disabled'}",
            flush=True,
        )
        print(
            "   graph capture: "
            + "; ".join(
                f"{label} {sec:.2f}s, incremental HBM {mib:+.1f}MiB"
                for label, sec, mib in capture_stats
            ),
            flush=True,
        )

    def argmax_tok():  # engine's OWN top-1 over its topk (not the sampler)
        return int(eng.head.out_id[int(eng.head.out_val.argmax().item())].item())

    NEAR_TIE = 0.30  # |top1-top2| below this = fp8 coin-flip, not a bug

    def _bench(graph, sd):  # pure in-graph replay us/tok (real whole-step E2E)
        Lp, _ = eng.load_prefill(state_dir=sd)
        samples = []
        for _ in range(cli.trials):
            eng.slot[1] = Lp
            for _ in range(10):
                graph.replay()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            dist.barrier()
            e0.record()
            for _ in range(cli.reps):
                graph.replay()
            e1.record()
            e1.synchronize()
            u = torch.tensor([e0.elapsed_time(e1) * 1e3 / cli.reps], device=dev)
            dist.all_reduce(u, op=dist.ReduceOp.MAX)
            samples.append(u.item())
        return sorted(samples)[len(samples) // 2]

    results = []
    for L, sd in avail:
        reference_path = sd / "golden.pt"
        exact = clean = None
        flips = []
        n_flip, max_gap, n_cp, st = 0, 0.0, 0, 0
        if reference_path.exists():
            reference = torch.load(reference_path, map_location="cpu")[
                "tokens"
            ]  # [first, dec0, dec1, ...]
            reference_dec = reference[1 : N + 1]
            # Teacher forcing isolates each numerical difference instead of allowing one token to change the
            # rest of the trajectory. A small top-logit gap identifies an FP8 near-tie.
            eng.load_prefill(state_dir=sd)
            for step in range(N):
                use_cp = cli.cp and (L + step) >= cli.cp_threshold
                n_cp += int(use_cp)
                (g_cp if use_cp else g).replay()
                torch.cuda.synchronize()
                token = argmax_tok()
                if step < len(reference_dec) and token != reference_dec[step]:
                    values, ids = eng.head.out_val.flatten(), eng.head.out_id.flatten()
                    top1 = float(values.max())
                    hit = (ids == reference_dec[step]).nonzero().flatten()
                    if hit.numel() > 0:
                        reference_logit = float(values[int(hit[0])])
                        reference_rank = int((values > reference_logit).sum().item()) + 1
                        gap = top1 - reference_logit
                    else:
                        reference_rank, gap = 99, 9.9
                    flips.append((step, reference_rank, gap))
                eng.slot[0] = reference[step + 1] if step + 1 < len(reference) else token
            st = eng.status_max()
            n_flip = len(flips)
            max_gap = max((gap for _, _, gap in flips), default=0.0)
            exact = n_flip == 0 and st == 0
            clean = exact or (all(gap < NEAR_TIE for _, _, gap in flips) and st == 0)
        # --- speed: pure free-run replay (real whole-step E2E, no host work); both graphs when --cp ---
        us = _bench(g, sd)
        us_cp = _bench(g_cp, sd) if (cli.cp and g_cp is not None) else None
        results.append((L, exact, clean, n_flip, max_gap, flips, us, st, n_cp, us_cp))

    if rank == 0:
        mode = "COMBINED non-CP->CP dispatch" if cli.cp else "non-CP"
        print(
            f"\n== nexrt decode matrix (TP=8, 60 layers, N={N} tok/length, {mode}; init+capture once) ==",
            flush=True,
        )
        print(
            f"   {'ctx_len':>8} {'reference':>9} {'us/tok':>9} {'tok/s':>7}   "
            f"saved-token comparison (difference = reference rank / gap to engine #1)",
            flush=True,
        )
        for L, exact, clean, n_flip, max_gap, flips, us, st, n_cp, us_cp in results:
            if clean is None:
                verdict, v = "N/A", "no golden.pt; speed only"
            elif st != 0:
                verdict, v = "FAIL", f"status={st}"
            elif exact:
                verdict, v = "PASS", f"all {N} match"
            else:
                desc = ", ".join(
                    f"step{s}:#{gr if gr < 99 else '>topk'} gap{gg:.3f}" for s, gr, gg in flips
                )
                kind = "near-tie" if clean else "REAL"
                v = f"{n_flip}/{N} {kind} flip(s) [{desc}]"
                verdict = "PASS" if clean else "FAIL"
            if us_cp is None:
                extra = ""
            elif clean is None:
                extra = f" | CP graph {us_cp:.0f}us ({1e6 / us_cp:.0f} tok/s)"
            else:
                extra = f" | {n_cp}/{N} reference steps on CP, CP graph {us_cp:.0f}us ({1e6 / us_cp:.0f} tok/s)"
            print(f"   {L:>8} {verdict:>9} {us:>9.1f} {1e6 / us:>7.0f}   {v}{extra}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
