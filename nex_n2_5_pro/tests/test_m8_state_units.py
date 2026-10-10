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

"""M=8 ring/state unit gates.

Sections (each prints one OK/FAIL line; process fails on any):
  A. COMMIT MATRIX: for every commit_count 1..8 (forced), one M=8 unit's
     handback state (conv window + recurrent) must match the frozen BS=1 engine
     teacher-forced through the same committed input tokens (anchored relL2 --
     the M8 family stack is not bitwise vs the BS=1 production stack).
  B. RESIDUES x WRAP: sequences of forced commits walk committed_len across
     ring residues (conv &15, state %9, including 16-ring wraparound); after
     each sequence the handback must still match the BS=1 oracle fed the same
     committed tokens.
  C. GRAPH REPLAY, no host writes: captured unit replayed N times with forced
     commit 1 advances committed_len by exactly N and step_epoch by N, status 0.
  D. COLLECTIVE FAIL-CLOSED + RETRY: a sticky upstream status on EXACTLY ONE
     rank must fail all eight ranks closed (commit 0, committed_len unchanged,
     control rank-identical). Then reset_unit() (latches only -- NO reload) must
     let the SAME committed_len be re-verified, and the resulting state must
     still match the BS=1 oracle: that is what proves the failed unit did not
     damage the recurrent base its own retry re-reads. Checking only status/
     commit_count here would pass even on a silently corrupted base.
  E. KV PREFIX: after a partial commit, re-speculating from C' with different
     ids overwrites the rejected provisional KV rows; the committed prefix rows
     stay bit-identical (they are never rewritten by the next unit).

Token identity is irrelevant to state mechanics; what matters is that the M=8
path and the BS=1 oracle consume the SAME tokens.

Run:  NEXRT_NLAYERS=4 NEXRT_MOE_RAW=1 torchrun --standalone \
          --nproc_per_node=8 tests/test_m8_state_units.py \
          --state-dir states/prefill_state_60_512
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

REL_GATE = 0.05  # state-boundary ceiling (measured family envelope ~2e-3)
# [1]*9 walks committed_len through nine consecutive values, so it visits EVERY
# residue of the 9-slot recurrent ring; the others cross the 16-slot conv wrap.
WRAP_SEQUENCES = [[1] * 9, [8, 8], [3, 5, 1, 8, 2], [7, 1, 6, 2]]


def rel_l2(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return float(torch.linalg.vector_norm(a - b) / (torch.linalg.vector_norm(b) + 1e-30))


def bs1_states(eng, state_dir, tokens, Lp):
    """Teacher-force the frozen BS=1 engine through `tokens`; return final
    conv/rec states per GDN layer for the external-oracle comparison."""
    eng.load_prefill(state_dir=state_dir)
    for j, tok in enumerate(tokens):
        eng.slot[0] = int(tok)
        eng.slot[1] = Lp + j
        eng.step()
    torch.cuda.synchronize()
    return (
        {L: g.conv_state4[:, :3].clone() for L, g in eng.gdn.items()},
        {L: g.state_t.clone() for L, g in eng.gdn.items()},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default="states/prefill_state_60_512")
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

    eng = DecodeEngine(rank, world, dev)
    m8 = M8Verify(eng, dev)
    m8.set_seed(97)
    state_dir = HERE / cli.state_dir

    # deterministic token pool (valid ids, non-trivial)
    pool = [(1000 + 37 * i) % 200000 for i in range(256)]
    fails = []

    def gate(name, ok, detail=""):
        flag = torch.tensor([0 if ok else 1], device=dev)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(f"   [{'OK' if flag.item() == 0 else 'FAIL'}] {name} {detail}", flush=True)
        if flag.item() != 0:
            fails.append(name)

    def run_m8_sequence(commits, token_stream):
        """Load once, then one eager unit per commit entry (teacher-forced ids,
        forced commit). Returns (final committed_len, committed input tokens)."""
        Lp = m8.load(state_dir, token_stream[0:8])
        m8.use_forced_commit = True
        committed = []
        cursor = 0
        for cc in commits:
            ids = token_stream[cursor : cursor + 8]
            m8.verify_ids.copy_(torch.tensor(ids, dtype=torch.int32, device=dev))
            m8.ctl[ctl8.PENDING_TARGET_ID] = ids[0]
            m8.forced_commit.fill_(cc)
            m8.run()
            committed.extend(ids[:cc])
            cursor += cc
        torch.cuda.synchronize()
        c = ctl8.read_control_block(m8.ctl)
        return Lp, committed, c

    # ---- A: commit matrix 1..8 ----
    worst = {}
    for cc in range(1, 9):
        Lp, committed, c = run_m8_sequence([cc], pool)
        ok_len = c["committed_len"] == Lp + cc and c["commit_count"] == cc and c["status"] == 0
        m8.handback_bs1_views(c["committed_len"])
        m8_conv = {L: g.conv_state4[:, :3].clone() for L, g in eng.gdn.items()}
        m8_rec = {L: g.state_t.clone() for L, g in eng.gdn.items()}
        ref_conv, ref_rec = bs1_states(eng, state_dir, committed, Lp)
        wc = max(rel_l2(m8_conv[L], ref_conv[L]) for L in eng.gdn)
        wr = max(rel_l2(m8_rec[L], ref_rec[L]) for L in eng.gdn)
        worst[cc] = (wc, wr)
        gate(
            f"A commit={cc}",
            ok_len and wc <= REL_GATE and wr <= REL_GATE,
            f"(conv {wc:.2e} rec {wr:.2e})",
        )

    # ---- B: residue walks + 16-ring wrap ----
    for si, seq in enumerate(WRAP_SEQUENCES):
        Lp, committed, c = run_m8_sequence(seq, pool)
        ok_len = c["committed_len"] == Lp + sum(seq) and c["status"] == 0
        m8.handback_bs1_views(c["committed_len"])
        m8_conv = {L: g.conv_state4[:, :3].clone() for L, g in eng.gdn.items()}
        m8_rec = {L: g.state_t.clone() for L, g in eng.gdn.items()}
        ref_conv, ref_rec = bs1_states(eng, state_dir, committed, Lp)
        wc = max(rel_l2(m8_conv[L], ref_conv[L]) for L in eng.gdn)
        wr = max(rel_l2(m8_rec[L], ref_rec[L]) for L in eng.gdn)
        gate(
            f"B seq={seq}",
            ok_len and wc <= REL_GATE and wr <= REL_GATE,
            f"(conv {wc:.2e} rec {wr:.2e})",
        )

    # ---- C: captured replay, zero host writes between replays ----
    m8.load(state_dir, pool[:8])
    m8.use_forced_commit = True
    m8.forced_commit.fill_(1)
    g = m8.capture(warmup=3)
    Lp = m8.load(state_dir, pool[:8])
    replays = 20  # crosses the 16-slot conv-ring wrap
    for _ in range(replays):
        g.replay()
    torch.cuda.synchronize()
    c = ctl8.read_control_block(m8.ctl)
    gate(
        "C replay x20 (ring wrap)",
        c["committed_len"] == Lp + replays and c["step_epoch"] == replays and c["status"] == 0,
        f"(C {Lp}->{c['committed_len']}, epoch {c['step_epoch']})",
    )

    # ---- D: collective fail-closed from ONE rank ----
    Lp = m8.load(state_dir, pool[:8])
    m8.use_forced_commit = False
    if rank == 3:
        m8.op_status.fill_(7)
    m8.run()
    torch.cuda.synchronize()
    c = ctl8.read_control_block(m8.ctl)
    ctl_all = [torch.zeros_like(m8.ctl) for _ in range(world)]
    dist.all_gather(ctl_all, m8.ctl)
    same = all(bool(torch.equal(ctl_all[0], t)) for t in ctl_all)
    gate(
        "D one-rank failure fails all closed",
        c["status"] != 0 and c["commit_count"] == 0 and c["committed_len"] == Lp and same,
        f"(status={c['status']} commit={c['commit_count']})",
    )
    # Retry the SAME committed_len after clearing only the sticky latches. No
    # reload: the 9-slot recurrent ring and 16-slot conv ring both hold a unit's
    # whole live span, so the failed unit left the base snapshot and history taps
    # intact. Verify that against the BS=1 oracle, not just against status.
    m8.reset_unit()
    m8.use_forced_commit = True
    m8.forced_commit.fill_(3)
    retry_ids = pool[:8]
    m8.verify_ids.copy_(torch.tensor(retry_ids, dtype=torch.int32, device=dev))
    m8.ctl[ctl8.PENDING_TARGET_ID] = retry_ids[0]
    m8.run()
    torch.cuda.synchronize()
    c = ctl8.read_control_block(m8.ctl)
    m8.handback_bs1_views(c["committed_len"])
    retry_conv = {L: g.conv_state4[:, :3].clone() for L, g in eng.gdn.items()}
    retry_rec = {L: g.state_t.clone() for L, g in eng.gdn.items()}
    ref_conv, ref_rec = bs1_states(eng, state_dir, retry_ids[:3], Lp)
    wc = max(rel_l2(retry_conv[L], ref_conv[L]) for L in eng.gdn)
    wr = max(rel_l2(retry_rec[L], ref_rec[L]) for L in eng.gdn)
    gate(
        "D retry after reset_unit commits",
        c["status"] == 0 and c["commit_count"] == 3 and c["committed_len"] == Lp + 3,
        f"(status={c['status']} commit={c['commit_count']} C {Lp}->{c['committed_len']})",
    )
    gate(
        "D retry state matches BS=1 oracle (base survived failed unit)",
        wc <= REL_GATE and wr <= REL_GATE,
        f"(conv {wc:.2e} rec {wr:.2e})",
    )

    # ---- E: rejected-KV overwrite + committed-prefix stability ----
    m8.use_forced_commit = True
    Lp, _, c = run_m8_sequence([2], pool)  # commit rows C, C+1
    kv_before = {
        L: (a.k_cache[Lp : Lp + 2].clone(), a.v_cache[Lp : Lp + 2].clone())
        for L, a in eng.attn.items()
    }
    spec_before = {L: a.k_cache[Lp + 2 : Lp + 8].clone() for L, a in eng.attn.items()}
    # re-speculate from C+2 with different ids
    alt = [(5 + 91 * i) % 200000 for i in range(8)]
    m8.verify_ids.copy_(torch.tensor(alt, dtype=torch.int32, device=dev))
    m8.forced_commit.fill_(1)
    m8.run()
    torch.cuda.synchronize()
    prefix_ok = all(
        bool(torch.equal(kv_before[L][0], a.k_cache[Lp : Lp + 2]))
        and bool(torch.equal(kv_before[L][1], a.v_cache[Lp : Lp + 2]))
        for L, a in eng.attn.items()
    )
    overwrote = any(
        not bool(torch.equal(spec_before[L], a.k_cache[Lp + 2 : Lp + 8]))
        for L, a in eng.attn.items()
    )
    gate("E committed prefix bit-stable", prefix_ok)
    gate("E rejected rows overwritten", overwrote)

    if rank == 0:
        verdict = "PASS" if not fails else f"FAIL ({fails})"
        print(f"== M8 state units ({len(eng.layers)}L): {verdict} ==", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
