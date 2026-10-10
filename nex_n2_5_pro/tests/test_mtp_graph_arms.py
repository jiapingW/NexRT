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

"""Check MTP graph-family routing and compare CP/non-CP round replay.

Both attention arms share one target engine. The test checks the round and
verification graphs, then replays them over saved prefill states. Extended KV
fixtures are for timing only; they do not validate generated answers.

Run from the repository root with TP=8 and full weight/state fixtures.
"""

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

DEFAULT_LENGTHS = "8192,16384,32768,49152,65536,131072,262080"
REAL_STATES = (8192, 16384, 32768, 65536, 131072)

# Device-kernel routing assertions per arm (count ranges per ROUND replay).
# 15 full-attn verify layers + 1 extend layer; the T=1 drafts run the
# MTP-family clone kernels (never 1-batch entry points).

# 1-batch production global kernels: NONE may appear in ANY MTP graph.
BS1_KERNELS = [
    r"\bQkvGroupSplitKKernel",
    r"\bNormRopeFusedKernel",
    r"\bNormRopeFusedCpKernel",
    r"\bwgmma_fd4_mega",
    r"\bOProjFusedPushKernel",
    r"\bWholeVectorReduceKernel",
    r"\bGateGemvRowRawAndRmsCta0Kernel",
    r"\bGateTopKSingleCtaNoLogit",
    r"\bL1K2PairPacked64",
    r"\bFusedL2ReducePushKernel",
    r"\bLmHeadGemvRowRmsCta0Kernel",
    r"\bEmbedGatherKernel",
    r"\bResidualAddKernel",
    r"\bLinearFrontSplitCtaSk4Kernel",
    r"\bRecurrentCoreTransposedSharedNormKernel",
    r"\bLinearFinalOProjFusedPushKernel",
    r"\bDupQOnlyGroupSplitKKernel",
    r"\bRaw16HopperWsStageSwitchPublishDynKernel",
    r"\bSummaryThenPartialSlotProducerWaitKernel",
    r"\bProducerKernel",
    r"\bReducerKernel",
    r"\bSamplerKernel",
]

# M4/MTP-family global kernels (spelling markers plus the family kernels whose
# names carry no marker): NONE may appear in either 1-batch graph.
M4_MTP_KERNELS = [
    r"M4",
    r"Mtp",
    r"m4_",
    r"mtp_",
    r"\bStageCpSlotsKernel",
    r"\bFoldCpStatusKernel",
    r"\bcp_ll_",
    r"\bMultiRow",
    r"\bquad",
    r"\bQuad",
]

# Positive 1-batch sentinels: a 1-batch step graph that profiles as empty (or
# as something else entirely) must FAIL, not slide through the absence checks.
BS1_SENTINELS = {
    # Present in BOTH 1-batch graphs (the non-CP qkv front is absent from the
    # CP graph, so it cannot be a sentinel here).
    r"\bEmbedGatherKernel": 1,
    r"\bFusedL2ReducePushKernel": 1,
    r"\bLinearFrontSplitCtaSk4Kernel": 1,
}

# MTP-family T=1 draft clone kernels: one round = 2 chain decodes + 1 extend
# head, and the o-proj clone also carries the two T=1 fc launches. The
# verify-only graph must contain none of these.
DRAFT_CLONES = {
    r"\bMtpQkvGroupSplitKKernel": (2, 2),
    r"\bMtpNormRopeFusedKernel": (2, 2),
    r"\bmtp_wgmma_fd4_mega": (2, 2),
    r"\bMtpOProjFusedPushKernel": (4, 4),
    r"\bMtpGateGemvRowRawAndRmsCta0Kernel": (2, 2),
    r"\bMtpGateTopKSingleCtaNoLogitRmsScaleXnKernel": (2, 2),
    r"\bMtpL1K2PairPacked64UnweightedXbf16V4SplitKW13XnKernel": (2, 2),
    r"\bMtpFusedL2ReducePushKernel": (2, 2),
    r"\bMtpLmHeadGemvRowRmsCta0Kernel": (3, 3),
    r"\bMtpFcT4Kernel": (1, 1),
}

EXPECT = {
    "noncp": {
        "present": {
            r"\bm4_attn_partial_quad_dyn": (14, 18),
            r"\bM4AttnOProjKernel": (16, 16),
            r"\bM4GdnOProjKernel": (45, 45),
            r"\bM4AttnQkvKernel": (14, 18),
        },
        "absent": [
            r"\bm4_cp_quad_partial",
            r"\bcp_ll_combine_push",
            r"\bM4AttnOProjCpKernel",
            r"\bM4GdnOProjCpKernel",
            r"\bM4GdnOProjNonCpKernel",
            r"\bMultiRowStageSwitchPublishKernel",
            r"\bM4CpMerge4GateOnceKernel",
            r"\bm4_attn_partial_rowmaj",
        ],
    },
    "cp8": {
        "present": {
            r"\bm4_cp_quad_partial": (14, 16),
            r"\bcp_ll_combine_push_ep8\(": (14, 16),
            r"\bcp_ll_tail_merge_ep8\(": (14, 16),
            r"\bM4AttnOProjKernel": (16, 16),
            r"\bM4GdnOProjKernel": (45, 45),
            r"\bm4_attn_partial_quad_dyn": (1, 2),
        },
        "absent": [
            r"\bM4AttnOProjCpKernel",
            r"\bM4GdnOProjCpKernel",
            r"\bM4GdnOProjNonCpKernel",
            r"\bcp_ll_combine_push_ep\(",
            r"\bcp_ll_tail_merge_ep<",
            r"\bMultiRowStageSwitchPublishKernel",
            r"\bM4CpMerge4GateOnceKernel",
        ],
    },
}


def capture(fn, warmup=3):
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    return g


def wall_block_us(g, reps):
    """Block-averaged wall per replay (the established [graph] basis)."""
    torch.cuda.synchronize()
    dist.barrier()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / reps


def sampled_us(g, reps):
    """Per-replay samples (sync per replay) for spread; NOT the headline basis."""
    torch.cuda.synchronize()
    dist.barrier()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    out = []
    for _ in range(reps):
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) * 1000.0)
    return out


def kernel_agg(g, reps=3):
    """rank0: {kernel_name: (count_per_replay, us_per_replay)}; {} elsewhere."""
    rank = dist.get_rank()
    for _ in range(2):
        g.replay()
    torch.cuda.synchronize()
    dist.barrier()
    if rank != 0:
        for _ in range(reps):
            g.replay()
        torch.cuda.synchronize()
        dist.barrier()
        return {}
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            g.replay()
        torch.cuda.synchronize()
    agg = {}
    for evt in prof.key_averages():
        t = float(evt.self_device_time_total)
        if t > 0.0:
            agg[evt.key] = (evt.count / reps, t / reps)
    dist.barrier()
    return agg


def check_symbols(arm, agg, *, round_graph=True):
    """Evaluate EXPECT[arm] plus the two-way family rules against a rank0 agg.

    Every MTP graph (round or verify-only) must contain ZERO 1-batch
    production kernels; round graphs must additionally contain exactly the
    expected MTP draft-clone launch counts, and verify-only graphs none."""
    bad = []

    def count_of(pat):
        return sum(c for k, (c, _t) in agg.items() if re.search(pat, k))

    if not agg:
        bad.append(f"{arm}: empty kernel profile (no kernels captured)")
    for pat, (lo, hi) in EXPECT[arm]["present"].items():
        c = count_of(pat)
        if round_graph and not (lo <= c <= hi):
            bad.append(f"{arm}: expected {pat} x[{lo},{hi}] per round, saw {c:.1f}")
        # Verify-only graphs must contain the arm's PER-LAYER verify kernels
        # (round lo >= 14); low-count patterns belong to the draft/extend
        # phases and are legitimately absent from the verify-only capture.
        if not round_graph and lo >= 14 and c < 1:
            bad.append(f"{arm}: verify graph missing its core kernel {pat}")
    for pat in EXPECT[arm]["absent"]:
        c = count_of(pat)
        if c > 0:
            bad.append(f"{arm}: forbidden {pat} present x{c:.1f}")
    for pat in BS1_KERNELS:
        c = count_of(pat)
        if c > 0:
            bad.append(f"{arm}: 1-batch production kernel in MTP graph: {pat} x{c:.1f}")
    for pat, (lo, hi) in DRAFT_CLONES.items():
        c = count_of(pat)
        if round_graph and not (lo <= c <= hi):
            bad.append(f"{arm}: draft clone {pat} x{c:.1f} not in [{lo},{hi}]")
        if not round_graph and c > 0:
            bad.append(f"{arm}: draft clone in verify-only graph: {pat} x{c:.1f}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default=DEFAULT_LENGTHS)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--state-root", type=Path, default=HERE / "states")
    ap.add_argument(
        "--out", type=str, default="", help="optional JSON output path (default: print to stdout)"
    )
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",") if x]

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl")
    assert world == 8

    from decode_engine import DecodeEngine
    from m4_verify import M4VerifyCp, M4VerifyNonCP
    from mtp.block import MtpDraft
    import m4_control as ctlabi

    def extend_kv_prefix(engine, seed_len, target_len):
        """Repeat the real seed KV prefix in-place to form a matched long fixture."""
        if target_len <= seed_len:
            return
        for block in engine.attn.values():
            for cache in (block.k_cache, block.v_cache):
                write = seed_len
                while write < target_len:
                    count = min(seed_len, target_len - write)
                    cache[write : write + count].copy_(cache[:count])
                    write += count
        torch.cuda.synchronize()
        dist.barrier()

    def relocate_loaded_m4(obj, target_len, quartet):
        """Relocate the real seed's logical state to an arbitrary committed length."""
        ctlabi.init_control_block(obj.ctl, target_len, quartet)
        obj.eng.slot[0] = quartet[0]
        obj.eng.slot[1] = target_len
        obj.S0 = target_len
        for layer, gdn in obj.eng.gdn.items():
            conv = obj.conv_ring[layer]
            conv.zero_()
            for offset in range(3):
                conv[:, ctlabi.conv_slot(target_len - 3 + offset)].copy_(gdn.conv_state4[:, offset])
            state = obj.state_ring[layer]
            state.zero_()
            state[ctlabi.state_slot(target_len - 1)].copy_(gdn.state_t)
        cp_slots = getattr(obj, "cp_slots4", None)
        if cp_slots is not None:
            cp_slots.zero_()

    def status_max(obj):
        value = int(obj.eng.status_max())
        for name in ("ws_status", "ws_moe_status", "ws12_status", "ws21_status", "cursor_status"):
            tensor = getattr(obj, name, None)
            if tensor is not None:
                value = max(value, int(tensor.item()))
        return value

    eng = DecodeEngine(rank, world, dev, cp=True)

    def build(arm, shared_draft=None):
        if arm == "noncp":
            m4 = M4VerifyNonCP(eng, dev)
        else:
            m4 = M4VerifyCp(eng, dev)
        m4.use_forced_commit = True
        m4.forced_commit.fill_(4)
        blk = MtpDraft(eng, m4, dev, share_weights_from=shared_draft)
        assert m4.oproj_xeff4.dtype == torch.bfloat16
        assert m4.oproj_xeff4.numel() == 4 * 1024
        assert blk.fc4_rep.data_ptr() != m4.ws12_rep.data_ptr()
        assert blk.fc4_it.data_ptr() != m4.ws12_it.data_ptr()
        if shared_draft is not None:
            for name in ("fc4_rep", "fc4_bar", "fc4_it"):
                assert getattr(blk, name).data_ptr() == getattr(shared_draft, name).data_ptr()
            assert blk.fc4_rep_mc == shared_draft.fc4_rep_mc
            assert blk.fc4_bar_mc == shared_draft.fc4_bar_mc
        return m4, blk

    arms = {"noncp": build("noncp")}
    arms["cp8"] = build("cp8", arms["noncp"][1])

    handoffs = {}

    def state_for(L):
        if L in REAL_STATES:
            return args.state_root / f"prefill_state_60_{L}", L, "real_prefill"
        seed = max(s for s in REAL_STATES if s <= L)
        return (
            args.state_root / f"prefill_state_60_{seed}",
            seed,
            "deterministic_repeated_kv_extension",
        )

    def mtp_extend(blk, seed_len, L):
        for cache in (blk.attn.k_cache, blk.attn.v_cache):
            w = seed_len
            while w < L:
                c = min(seed_len, L - w)
                cache[w : w + c].copy_(cache[:c])
                w += c

    def reset(arm, L):
        m4, blk = arms[arm]
        sd, seed_len, kind = state_for(L)
        if sd not in handoffs:
            handoffs[sd] = torch.load(sd / "mtp_handoff.pt", map_location="cpu", weights_only=False)
        hand = handoffs[sd]
        quartet = [int(hand["t0"]), int(hand["d1"]), 0, 0]
        Lp = int(m4.load(str(sd), quartet))
        assert Lp == seed_len, f"seed mismatch {Lp} != {seed_len}"
        blk.seed(
            hand["k"][rank // 4].to(dev),
            hand["v"][rank // 4].to(dev),
            Lp,
            hand["y_sel_raw"].to(dev),
        )
        if L > seed_len:
            extend_kv_prefix(eng, seed_len, L)
            mtp_extend(blk, seed_len, L)
            relocate_loaded_m4(m4, L, tuple(quartet))
        m4.use_forced_commit = True
        m4.forced_commit.fill_(4)
        torch.cuda.synchronize()
        dist.barrier()
        return kind

    # ---- capture once per arm at the first length (graphs are length-agnostic:
    # every kernel reads committed_len from the device control block) ----------
    g_round, g_verify = {}, {}
    for arm in arms:
        reset(arm, lengths[0])
        m4, blk = arms[arm]
        g_round[arm] = capture(blk.round)
        reset(arm, lengths[0])
        g_verify[arm] = capture(m4.run)

    # ---- routing assertions from the profiler (rank0 evaluates, all ranks act) --
    dump_aggs = {}
    fail = torch.zeros(1, dtype=torch.int32, device=dev)
    for arm in arms:
        reset(arm, lengths[0])
        agg = kernel_agg(g_round[arm])
        if rank == 0:
            dump_aggs[arm] = {k: [c, t] for k, (c, t) in agg.items()}
            bad = check_symbols(arm, agg)
            for b in bad:
                print(f"[SYMBOL-FAIL] {b}", flush=True)
            if bad:
                fail.fill_(1)
    for arm in arms:
        reset(arm, lengths[0])
        agg = kernel_agg(g_verify[arm])
        if rank == 0:
            bad = check_symbols(arm, agg, round_graph=False)
            for b in bad:
                print(f"[SYMBOL-FAIL] verify-{arm}: {b}", flush=True)
            if bad:
                fail.fill_(1)
    dist.all_reduce(fail)
    if int(fail.item()):
        raise SystemExit("routing assertions failed; see [SYMBOL-FAIL] lines")
    if rank == 0:
        print(
            "[symbols] all arm routing assertions passed "
            "(round + verify graphs, two-way family isolation)",
            flush=True,
        )

    # ---- BS=1 family isolation: neither bs=1 graph may contain an M4/MTP kernel ----
    fail.zero_()
    reset("noncp", lengths[0])
    g_bs1_noncp = eng.capture(warmup=3)
    reset("noncp", lengths[0])
    g_bs1_cp = eng.capture(warmup=3, step_fn=eng.step_cp)
    for tag, g in (("noncp", g_bs1_noncp), ("cp", g_bs1_cp)):
        agg = kernel_agg(g)
        if rank == 0:
            bad = [
                f"contains an M4/MTP kernel: {k}"
                for k in agg
                for pat in M4_MTP_KERNELS
                if re.search(pat, k)
            ]
            if not agg:
                bad.append("empty kernel profile (no kernels captured)")
            for pat, lo in BS1_SENTINELS.items():
                c = sum(cnt for k, (cnt, _t) in agg.items() if re.search(pat, k))
                if c < lo:
                    bad.append(f"missing 1-batch sentinel {pat} (saw {c:.1f})")
            for b in bad:
                print(f"[SYMBOL-FAIL] bs1-{tag} graph {b}", flush=True)
            if bad:
                fail.fill_(1)
    dist.all_reduce(fail)
    if int(fail.item()):
        raise SystemExit("bs=1 graphs contain M4/MTP kernels; see [SYMBOL-FAIL] lines")
    if rank == 0:
        print("[symbols] bs=1 graphs contain no M4/MTP kernels", flush=True)

    # ---- interleaved per-length measurement --------------------------------------
    rows = []
    order = ["noncp", "cp8", "noncp"]
    for L in lengths:
        row = {"length": L, "arms": {}, "fixture": state_for(L)[2]}
        for slot, arm in enumerate(order):
            m4, blk = arms[arm]
            torch.cuda.reset_peak_memory_stats()
            kind = reset(arm, L)
            for _ in range(3):
                g_round[arm].replay()
            torch.cuda.synchronize()
            dist.barrier()
            bw = wall_block_us(g_round[arm], args.iters)
            samples = sampled_us(g_round[arm], args.samples)
            vw = wall_block_us(g_verify[arm], args.iters)
            # correctness surface after timing: sticky statuses + control block
            w = m4.ctl.tolist()
            ok_commit = w[ctlabi.COMMIT_COUNT] == 4
            ok_status = w[ctlabi.STATUS] == 0 and status_max(m4) == 0
            gathered = [torch.empty_like(m4.ctl) for _ in range(world)]
            dist.all_gather(gathered, m4.ctl)
            ok_identical = all(torch.equal(g_, gathered[0]) for g_ in gathered)
            t = torch.tensor(
                [
                    bw,
                    statistics.median(samples),
                    sorted(samples)[max(0, int(0.1 * len(samples)) - 1)],
                    sorted(samples)[min(len(samples) - 1, int(0.9 * len(samples)))],
                    vw,
                    float(ok_commit and ok_status and ok_identical),
                    torch.cuda.max_memory_allocated() / 2**20,
                    torch.cuda.memory_reserved() / 2**20,
                ],
                dtype=torch.float64,
                device=dev,
            )
            allt = [torch.zeros_like(t) for _ in range(world)]
            dist.all_gather(allt, t)
            if rank == 0:
                mx = [max(x[i].item() for x in allt) for i in range(6)]
                ok_all = min(x[5].item() for x in allt) > 0.5
                key = arm if slot < len(order) - 1 else "noncp2"
                row["arms"][key] = {
                    "round_wall_us": mx[0],
                    "round_med_us": mx[1],
                    "round_p10_us": mx[2],
                    "round_p90_us": mx[3],
                    "verify_wall_us": mx[4],
                    "ok": ok_all,
                    "fixture": kind,
                    "hbm_peak_alloc_mb": max(x[6].item() for x in allt),
                    "hbm_reserved_mb": max(x[7].item() for x in allt),
                }
            dist.barrier()
        if rank == 0:
            a = row["arms"]
            drift = abs(a["noncp"]["round_wall_us"] - a["noncp2"]["round_wall_us"]) / (
                (a["noncp"]["round_wall_us"] + a["noncp2"]["round_wall_us"]) / 2
            )
            row["drift_pct"] = 100.0 * drift
            nc = min(a["noncp"]["round_wall_us"], a["noncp2"]["round_wall_us"])
            row["winner"] = "cp" if a["cp8"]["round_wall_us"] < nc else "noncp"
            print(
                f"[row] L={L:6d} [{row['fixture'][:4]}]  "
                f"noncp {nc:7.1f}  cp8 {a['cp8']['round_wall_us']:7.1f}  "
                f"d(cp8-noncp) {a['cp8']['round_wall_us'] - nc:+7.1f}  "
                f"win={row['winner']}  drift {row['drift_pct']:.2f}%  "
                f"ok={all(v['ok'] for v in a.values())}",
                flush=True,
            )
            rows.append(row)
        dist.barrier()

    # Fail closed: any not-ok row (sticky status, commit!=4, rank-divergent
    # control block) must fail the PROCESS, not just annotate the JSON.
    fail.zero_()
    if rank == 0 and any(not v["ok"] for r in rows for v in r["arms"].values()):
        fail.fill_(1)
    dist.all_reduce(fail)
    row_fail = int(fail.item())

    if rank == 0:
        dump = {
            "basis": "graph (production CUDA-graph replay, rank-max, forced commit 4)",
            "arms": {
                "noncp": "M4VerifyNonCP (short-context arm)",
                "cp8": "M4VerifyCp (8 B LL wire)",
            },
            "env": {k: v for k, v in os.environ.items() if k.startswith("NEXRT")},
            "iters": args.iters,
            "samples": args.samples,
            "round_graph": "2xT1 draft -> M4 verify -> T4 extend (one graph)",
            "rows": rows,
            "kernel_aggs_first_length": dump_aggs,
        }
        if args.out:
            Path(args.out).write_text(json.dumps(dump, indent=1) + "\n")
            print(f"[graph-arms] wrote {args.out}", flush=True)
        else:
            print(json.dumps(dump, indent=1), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    if row_fail:
        raise SystemExit(
            "graph A/B: at least one arm-row failed its "
            "correctness surface (ok=False); see the JSON rows"
        )


if __name__ == "__main__":
    main()
