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

"""M=8 backbone vs the sequential BS=1 oracle.

From one prefill snapshot, the frozen BS=1 production engine is teacher-forced
through the SAME eight token ids the M=8 unit consumes, capturing after each step:
the rank-local TRUE logits shard, the provisional K/V rows, and the GDN conv/
recurrent states. The state is then re-seeded from disk and ONE eager M=8 unit
runs on [t0..t7]; all eight ordered rows are compared at every boundary.

BS=1 here is an out-of-path ORACLE: none of its kernels or state appear in the
M=8 unit. Numerics class: the M=8 family stack (bf16tc qkv/front, tc8 gate, tc16
head, quad core, push reduces) vs the BS=1 production stack differ by the accepted
family/reorder classes, so gates are the repo-declared ceilings (relL2 <= 0.25,
argmax agree or 0.30 near-tie, top-20 overlap >= 17), not bit equality.

Run:  NEXRT_NLAYERS=4 NEXRT_MOE_RAW=1 torchrun --standalone \
          --nproc_per_node=8 tests/test_m8_vs_bs1_seq.py \
          --state-dir states/prefill_state_60_512 [--ids t0,...,t7]
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

NEAR_TIE = 0.30  # nex_n2_5_pro/decode_matrix.py fp8 near-tie margin (repo-declared)
REL_GATE = 0.25  # established M=4 family ceiling
TOP20_MIN = 17


def rel_l2(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    denom = torch.linalg.vector_norm(b) + 1e-30
    return float(torch.linalg.vector_norm(a - b) / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default="states/prefill_state_60_512")
    ap.add_argument("--ids", default="")
    ap.add_argument("--out", default="")
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

    eng = DecodeEngine(rank, world, dev)
    m8 = M8Verify(eng, dev)
    state_dir = HERE / cli.state_dir

    # ---- build the eight-token chain: t0 = the snapshot's first token, then
    # seven greedy argmax transitions for the sequential BS=1 oracle ----
    Lp, first = eng.load_prefill(state_dir=state_dir)
    if cli.ids:
        ids = [int(t) for t in cli.ids.split(",")]
        assert len(ids) == 8
        teacher_forced = True
    else:
        ids = [int(first)]
        teacher_forced = False

    ref_logits = []
    ref_kv = {L: [] for L in eng.attn}
    ref_conv, ref_rec = {}, {}
    for j in range(8):
        if teacher_forced:
            tok = ids[j]
        else:
            tok = ids[j]  # ids grows below via argmax
        eng.slot[0] = tok
        eng.slot[1] = Lp + j
        eng.step()
        torch.cuda.synchronize()
        # hd.logits is the RAW GEMV output; the BS=1 top-k reducer applies inv_rms
        # downstream (top-k is scale-invariant). The M8 head emits TRUE logits, so
        # scale the reference by the same CTA0 scalar for a like-for-like compare.
        ref_logits.append((eng.head.logits.float() * eng.head.inv_rms.float()).clone())
        for L, a in eng.attn.items():
            ref_kv[L].append((a.k_cache[Lp + j].clone(), a.v_cache[Lp + j].clone()))
        if j == 7:
            for L, g in eng.gdn.items():
                ref_conv[L] = g.conv_state4[:, :3].clone()
                ref_rec[L] = g.state_t.clone()
        if not teacher_forced and j < 7:
            nxt = int(eng.head.out_id[int(eng.head.out_val.argmax().item())].item())
            ids.append(nxt)
    ref_logits = torch.stack(ref_logits)  # [8, 31040] this rank's shard

    # ---- reset from disk and run ONE eager M=8 unit on the same ids ----
    m8.set_seed(20260813)
    m8.use_forced_commit = True  # commit all 8: full-state handback
    m8.forced_commit.fill_(8)
    length = m8.load(state_dir, ids)
    assert length == Lp
    m8.run()
    torch.cuda.synchronize()
    out = m8.outputs()  # handback selects C+8 state
    c = out["m8_control"]

    fails = []

    def gate(name, ok, detail=""):
        flag = torch.tensor([0 if ok else 1], device=dev)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(f"   [{'OK' if flag.item() == 0 else 'FAIL'}] {name} {detail}", flush=True)
        if flag.item() != 0:
            fails.append(name)

    gate("m8 status == 0", out["status"] == 0, f"(status={out['status']})")
    gate("m8 commit == 8 (forced)", c["commit_count"] == 8)

    # ---- logits: per-row relL2, global argmax (near-tie tolerant), top-20 ----
    worst_rel = 0.0
    argmax_bad = 0
    top20_min_seen = 20
    for j in range(8):
        r = rel_l2(m8.logits8[j], ref_logits[j])
        worst_rel = max(worst_rel, r)
        # global argmax needs the full vocab: all-gather both shards
        mine = [torch.zeros_like(m8.logits8[j]) for _ in range(world)]
        refs = [torch.zeros_like(ref_logits[j]) for _ in range(world)]
        dist.all_gather(mine, m8.logits8[j].contiguous())
        dist.all_gather(refs, ref_logits[j].contiguous())
        mine_full = torch.cat(mine)
        ref_full = torch.cat(refs)
        am, ar = int(mine_full.argmax()), int(ref_full.argmax())
        if am != ar and float(ref_full[am]) < float(ref_full[ar]) - NEAR_TIE:
            argmax_bad += 1
        t_m = set(torch.topk(mine_full, 20).indices.tolist())
        t_r = set(torch.topk(ref_full, 20).indices.tolist())
        top20_min_seen = min(top20_min_seen, len(t_m & t_r))
    rel_t = torch.tensor([worst_rel], device=dev)
    dist.all_reduce(rel_t, op=dist.ReduceOp.MAX)
    gate("logits relL2 <= 0.25", rel_t.item() <= REL_GATE, f"(worst {rel_t.item():.4g})")
    gate("argmax rows agree (0.30 near-tie)", argmax_bad == 0, f"(bad rows {argmax_bad})")
    gate(f"top-20 overlap >= {TOP20_MIN}", top20_min_seen >= TOP20_MIN, f"(min {top20_min_seen})")

    # ---- provisional K/V rows C..C+7 (BS=1 wrote the same rows; M8 rewrote) ----
    worst_kv = 0.0
    for L, a in eng.attn.items():
        for j in range(8):
            rk = rel_l2(a.k_cache[Lp + j], ref_kv[L][j][0])
            rv = rel_l2(a.v_cache[Lp + j], ref_kv[L][j][1])
            worst_kv = max(worst_kv, rk, rv)
    kv_t = torch.tensor([worst_kv], device=dev)
    dist.all_reduce(kv_t, op=dist.ReduceOp.MAX)
    gate("kv relL2 <= 0.25", kv_t.item() <= REL_GATE, f"(worst {kv_t.item():.4g})")

    # ---- GDN state after C+8 (handback already selected it, forced commit 8) ----
    worst_conv = worst_rec = 0.0
    for L, g in eng.gdn.items():
        worst_conv = max(worst_conv, rel_l2(g.conv_state4[:, :3], ref_conv[L]))
        worst_rec = max(worst_rec, rel_l2(g.state_t, ref_rec[L]))
    s_t = torch.tensor([worst_conv, worst_rec], device=dev)
    dist.all_reduce(s_t, op=dist.ReduceOp.MAX)
    gate("conv relL2 <= 0.25", s_t[0].item() <= REL_GATE, f"(worst {s_t[0].item():.4g})")
    gate("rec relL2 <= 0.25", s_t[1].item() <= REL_GATE, f"(worst {s_t[1].item():.4g})")

    if rank == 0:
        verdict = "PASS" if not fails else f"FAIL ({fails})"
        mode = "teacher-forced" if teacher_forced else "argmax-chain"
        print(
            f"== M8 vs sequential BS=1 oracle ({len(eng.layers)}L, {mode}, L={Lp}): {verdict} ==",
            flush=True,
        )
        print(f"   ids: {ids}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
