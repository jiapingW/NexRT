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

"""nexrt CP (context-parallel) full-attention decode block. Each 4-rank group [0-3]/[4-7] holds 1
replicated KV head; instead of each rank attending its 4 q-heads over the FULL KV (4x redundant read),
each rank computes ALL 16 q-heads over its 1/4 KV shard, then the group LSE-merges + reduce-scatters so
each rank keeps its own 4 heads. This trades additional projection and attention compute for
less redundant KV traffic.

Chain: qgate(16Q) -> norm_rope(16) -> cooperative Hopper partial+combine+publish megakernel
       -> NexN25ProCpTail -> NexN25ProOProjWholeReduce (same o-proj as non-CP).

The standalone path uses private buffers at a fixed position. The device-S decode path shares its KV cache,
slot, and reduction workspace with the non-CP block so the host can select between two captured graphs.

Run: torchrun --standalone --nproc_per_node=8 attn_cp.py --layer 3 --past 65535
"""

import argparse
import ctypes
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

HERE = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(HERE.parent))

from nex_n2_5_pro.attn import (  # noqa: E402
    HID,
    HEAD_DIM,
    LOCAL,
    EPS,
    CACHE_LEN,
    build,
    build_cp,
    convert_weights_cp,
    rope_tables,
    _p,
    _sp,
    FullAttn,
)

CP = 4  # ranks per CP group (share 1 KV head)
NH_CP = 16  # q heads per KV head (all computed per rank under CP)
NUM_SPLIT = 128  # fixed partial count used by the selected cooperative megakernel

_CP_GROUPS = None


def _cp_group(rank):
    """The 4-rank subgroup [0-3] or [4-7] this rank belongs to (dist.new_group is collective -> build both)."""
    global _CP_GROUPS
    if _CP_GROUPS is None:
        _CP_GROUPS = [dist.new_group([0, 1, 2, 3]), dist.new_group([4, 5, 6, 7])]
    return _CP_GROUPS[rank // CP]


class FullAttnCP:
    def __init__(self, layer, rank, world, dev, share_noncp=None, share_cp=None):
        # share_noncp: a FullAttn block whose k_cache/v_cache/slot + 8-rank reduce symm-mem we reuse (no
        # HBM/NVLS dup for the two-graph). share_cp: the first FullAttnCP block whose 4-rank CP channel we reuse.
        self.rank, self.world, self.dev = rank, world, dev
        self.cp_rank, self.cp_gid = rank % CP, rank // CP
        vp, ci, cf, cu = ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_ulonglong
        # --- kernels (CP ones via build_cp gencode; norm_rope/o_proj via build) ---
        self.qgl = ctypes.CDLL(str(build_cp("nex_n2_5_pro_cp_qgate_ctypes")))
        self.qgl.NexN25ProCpQGateLaunch.argtypes = [vp] * 7 + [ci, vp]
        self.qgl.NexN25ProCpQGateDynLaunch.argtypes = [vp] * 9  # device-S (pos from slot)
        self.nrl = ctypes.CDLL(str(build("nex_n2_5_pro_norm_rope_ctypes")))
        self.nrl.NexN25ProCpNormRopeLaunch.argtypes = [vp] * 7 + [ci] + [vp] * 4 + [ci, cf, vp]
        self.nrl.NexN25ProCpNormRopeDynLaunch.argtypes = [vp] * 12 + [ci, cf, vp]  # device-S
        self.megatl = ctypes.CDLL(str(build_cp("nex_n2_5_pro_cp_hopper_mega_ctypes")))
        self.megatl.NexN25ProCpAttnMegaDynLaunch.argtypes = [vp] * 9 + [ci] * 2 + [cu] + [vp] * 4
        self.megatl.NexN25ProCpAttnMegaDynLaunch.restype = ci
        self.tll = ctypes.CDLL(str(build_cp("nex_n2_5_pro_cp_tail_ctypes")))
        self.tll.NexN25ProCpTailLaunch.argtypes = [vp] * 12 + [ci] * 4 + [vp]
        self.opl = ctypes.CDLL(str(build("nex_n2_5_pro_o_proj_reduce_ctypes")))
        self.opl.NexN25ProOProjWholeReduceLaunch.argtypes = [vp, cu] + [vp] * 4 + [ci, vp]
        self.opl.NexN25ProOProjWholeReduceResidLaunch.argtypes = [vp, cu] + [vp] * 4 + [ci, vp, vp]
        # --- weights + rope tables ---
        # CP-specific projection weights and RoPE tables are owned by this block. The KV cache and TP-reduction
        # workspace are shared with the non-CP twin below. Reference-only conversion tensors are not retained.
        self.w, _wr, _iln = convert_weights_cp(layer, rank, dev)
        self.cos_tab, self.sin_tab = rope_tables(dev)
        # --- per-block working buffers (transient; reused across the graph) ---
        self.q_group = torch.empty(NH_CP * HEAD_DIM, dtype=torch.bfloat16, device=dev)  # qgate 16 Q
        self.gate = torch.empty(
            LOCAL, dtype=torch.bfloat16, device=dev
        )  # qgate 4 gate (rank's own)
        self.q_out = torch.empty(
            NH_CP, HEAD_DIM, dtype=torch.bfloat16, device=dev
        )  # normed+roped 16 Q
        self.raw16_op = torch.empty(NH_CP, NUM_SPLIT, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        self.raw16_glse = torch.empty(NH_CP, NUM_SPLIT, dtype=torch.bfloat16, device=dev)
        # Private reset-free sense/counter state. DecodeEngine captures one CP
        # graph per block and replays only one graph at a time; any future
        # concurrent graph/stream use must allocate a distinct FullAttnCP (and
        # therefore a distinct counter) per in-flight graph.
        self.mega_grid_counter = torch.zeros(1, dtype=torch.int32, device=dev)
        self.summary = torch.empty(LOCAL, dtype=torch.bfloat16, device=dev)  # tail scratch
        self.final_out = torch.empty(HID, dtype=torch.float32, device=dev)
        # --- KV cache + device-S slot + 8-rank reduce: SHARE the non-CP block's (no HBM / NVLS dup) ---
        # The CP path's split reduce (tail partial + WholeReduce) binds slot [128] of the twin's
        # 256-slot arrival buffer: the non-CP fused push reduce owns slots [0:128] (one per CTA),
        # so interleaved non-CP/CP replays never see each other's signals and the tail's
        # generation-from-counter derivation counts only CP steps.
        if share_noncp is not None:
            fa = share_noncp
            self.k_cache, self.v_cache, self.slot = fa.k_cache, fa.v_cache, fa.slot
            self.partial2, self.peer_partial2, self.partial2_mc = (
                fa.partial2,
                fa.peer_partial2,
                fa.partial2_mc,
            )
            self.tp_bar, self.tp_bar_mc = fa.bar[128:], fa.bar_mc + 128 * 4
            self.tp_gen, self.tp_status = fa.cur_gen, fa.status
        else:
            self.k_cache = torch.zeros(CACHE_LEN, HEAD_DIM, dtype=torch.bfloat16, device=dev)
            self.v_cache = torch.zeros(CACHE_LEN, HEAD_DIM, dtype=torch.bfloat16, device=dev)
            self.slot = torch.zeros(2, dtype=torch.int32, device=dev)
            wname = dist.group.WORLD.group_name
            self.partial2, self.peer_partial2, self.partial2_mc = self._symm(
                2 * HID, torch.float32, wname
            )
            self.partial2.zero_()
            self.tp_bar, _, self.tp_bar_mc = self._symm(128, torch.int32, wname)
            self.tp_bar.zero_()
            self.tp_gen = torch.zeros(1, dtype=torch.int32, device=dev)
            self.tp_status = torch.zeros(1, dtype=torch.int32, device=dev)
        # --- 4-rank CP channel: allocate on the first CP block, SHARE across the rest (one pool) ---
        if share_cp is not None:
            c = share_cp
            self.cp_ctx, self.peer_ctx, self.cp_lse, self.peer_lse = (
                c.cp_ctx,
                c.peer_ctx,
                c.cp_lse,
                c.peer_lse,
            )
            self.cp_bar, self.cp_bar_mc = c.cp_bar, c.cp_bar_mc
            self.cp_status, self.cp_it, self.cp_gen = c.cp_status, c.cp_it, c.cp_gen
        else:
            self.cp_group = _cp_group(rank)
            self.cp_ctx, self.peer_ctx, _ = self._symm(
                NH_CP * HEAD_DIM, torch.bfloat16, self.cp_group.group_name
            )
            self.cp_lse, self.peer_lse, _ = self._symm(
                NH_CP, torch.float32, self.cp_group.group_name
            )
            self.cp_bar, _, self.cp_bar_mc = self._symm(128, torch.int32, self.cp_group.group_name)
            self.cp_bar.zero_()
            self.cp_status = torch.zeros(1, dtype=torch.int32, device=dev)
            self.cp_it = torch.zeros(1, dtype=torch.int32, device=dev)
            self.cp_gen = torch.zeros(1, dtype=torch.int32, device=dev)
        if self.cp_bar_mc == 0 or self.partial2_mc == 0 or self.tp_bar_mc == 0:
            raise RuntimeError("multicast unavailable (NVLS required)")
        torch.cuda.synchronize()
        dist.barrier()

    def _symm(self, numel, dtype, group_name):
        x = symm_mem.empty(numel, dtype=dtype, device=self.dev)
        h = symm_mem.rendezvous(x, group_name)
        peers = torch.tensor(list(h.buffer_ptrs), dtype=torch.int64, device=self.dev)
        return x, peers, int(getattr(h, "multicast_ptr", 0) or 0)

    def forward_cp(self, x, pos):
        """Fixed-pos CP full-attn -> reduced o-proj out [HID], using the production megakernel."""
        w = self.w
        ci, cf, cu = ctypes.c_int, ctypes.c_float, ctypes.c_ulonglong
        # 1. qgate: 16 Q heads (duplicate) + 4 gate + K/V scatter to cache[pos]
        self.qgl.NexN25ProCpQGateLaunch(
            _p(x),
            _p(w["qgate_w"]),
            _p(w["qgate_scales"]),
            _p(self.q_group),
            _p(self.gate),
            _p(self.k_cache),
            _p(self.v_cache),
            ci(pos),
            _sp(),
        )
        # 2. norm_rope (16 Q + 1 K + s/v/4-gate scale)
        self.nrl.NexN25ProCpNormRopeLaunch(
            _p(self.q_group),
            _p(self.k_cache),
            _p(w["qn"]),
            _p(w["kn"]),
            _p(self.cos_tab),
            _p(self.sin_tab),
            _p(self.q_out),
            ci(pos),
            _p(x),
            _p(self.v_cache),
            _p(self.gate),
            _p(self.gate),
            ci(HID),
            cf(EPS),
            _sp(),
        )
        # 3. cooperative partial + local combine + CP publish over this rank's 1/4 KV shard.
        self.slot[1] = pos
        rc = self.megatl.NexN25ProCpAttnMegaDynLaunch(
            _p(self.q_out),
            _p(self.k_cache),
            _p(self.v_cache),
            _p(self.raw16_op),
            _p(self.raw16_glse),
            _p(self.cp_ctx),
            _p(self.cp_lse),
            _p(self.mega_grid_counter),
            _p(self.slot),
            ci(CACHE_LEN),
            ci(self.cp_rank),
            cu(self.cp_bar_mc),
            _p(self.cp_status),
            _p(self.cp_it),
            _p(self.cp_gen),
            _sp(),
        )
        if rc != 0:
            raise RuntimeError(f"NexN25ProCpAttnMegaDynLaunch rc={rc}")
        # 5. tail: wait CP peers, merge 4 shards for this rank's 4 heads, gate, o-proj partial -> partial2[rank]
        self.tll.NexN25ProCpTailLaunch(
            _p(self.peer_ctx),
            _p(self.peer_lse),
            _p(self.gate),
            _p(self.summary),
            _p(w["o_w"]),
            _p(w["o_scales"]),
            _p(self.peer_partial2),
            _p(self.cp_bar),
            _p(self.cp_status),
            _p(self.cp_gen),
            _p(self.tp_bar),
            _p(self.tp_gen),
            ci(self.cp_rank),
            ci(CP),
            ci(self.rank),
            ci(self.world),
            _sp(),
        )
        # 6. final 8-rank TP reduce
        self.opl.NexN25ProOProjWholeReduceLaunch(
            _p(self.peer_partial2),
            cu(self.tp_bar_mc),
            _p(self.tp_bar),
            _p(self.final_out),
            _p(self.tp_status),
            _p(self.tp_gen),
            self.world,
            _sp(),
        )
        return self.final_out

    def forward_dyn_cp(self, x, resid=None):
        """Device-S CP full-attn: slot[1] drives both pos and the 1/4 KV shard. Same math as forward_cp;
        this is the graph-replay path for the growing-S decode. Stages 1-3 are dyn (read slot);
        combine/tail/reduce are pos-independent (identical to forward_cp)."""
        w = self.w
        ci, cf, cu = ctypes.c_int, ctypes.c_float, ctypes.c_ulonglong
        self.qgl.NexN25ProCpQGateDynLaunch(  # qgate 16Q, K/V -> cache[slot[1]]
            _p(x),
            _p(w["qgate_w"]),
            _p(w["qgate_scales"]),
            _p(self.q_group),
            _p(self.gate),
            _p(self.k_cache),
            _p(self.v_cache),
            _p(self.slot),
            _sp(),
        )
        self.nrl.NexN25ProCpNormRopeDynLaunch(  # 16-head norm_rope (device-S)
            _p(self.q_group),
            _p(self.k_cache),
            _p(w["qn"]),
            _p(w["kn"]),
            _p(self.cos_tab),
            _p(self.sin_tab),
            _p(self.q_out),
            _p(self.slot),
            _p(x),
            _p(self.v_cache),
            _p(self.gate),
            _p(self.gate),
            ci(HID),
            cf(EPS),
            _sp(),
        )
        rc = self.megatl.NexN25ProCpAttnMegaDynLaunch(
            _p(self.q_out),
            _p(self.k_cache),
            _p(self.v_cache),
            _p(self.raw16_op),
            _p(self.raw16_glse),
            _p(self.cp_ctx),
            _p(self.cp_lse),
            _p(self.mega_grid_counter),
            _p(self.slot),
            ci(CACHE_LEN),
            ci(self.cp_rank),
            cu(self.cp_bar_mc),
            _p(self.cp_status),
            _p(self.cp_it),
            _p(self.cp_gen),
            _sp(),
        )
        if rc != 0:
            raise RuntimeError(f"NexN25ProCpAttnMegaDynLaunch rc={rc}")
        self.tll.NexN25ProCpTailLaunch(
            _p(self.peer_ctx),
            _p(self.peer_lse),
            _p(self.gate),
            _p(self.summary),
            _p(w["o_w"]),
            _p(w["o_scales"]),
            _p(self.peer_partial2),
            _p(self.cp_bar),
            _p(self.cp_status),
            _p(self.cp_gen),
            _p(self.tp_bar),
            _p(self.tp_gen),
            ci(self.cp_rank),
            ci(CP),
            ci(self.rank),
            ci(self.world),
            _sp(),
        )
        if resid is None:  # standalone check -> fp32 out
            self.opl.NexN25ProOProjWholeReduceLaunch(
                _p(self.peer_partial2),
                cu(self.tp_bar_mc),
                _p(self.tp_bar),
                _p(self.final_out),
                _p(self.tp_status),
                _p(self.tp_gen),
                self.world,
                _sp(),
            )
            return self.final_out
        self.opl.NexN25ProOProjWholeReduceResidLaunch(  # graph path -> fold residual (r += reduced)
            _p(self.peer_partial2),
            cu(self.tp_bar_mc),
            _p(self.tp_bar),
            _p(self.final_out),
            _p(self.tp_status),
            _p(self.tp_gen),
            self.world,
            _p(resid),
            _sp(),
        )
        return resid


def _time(fn, dev, warmup=20, reps=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    e1.synchronize()
    t = torch.tensor([e0.elapsed_time(e1) * 1e3 / reps], device=dev)  # us/call
    dist.all_reduce(t, op=dist.ReduceOp.MAX)  # slowest rank (TP barrier waits for it)
    return t.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--past", type=int, default=65535)
    ap.add_argument(
        "--bench", action="store_true", help="also time CP vs non-CP full-attn at 32K + 64K"
    )
    cli = ap.parse_args()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl")
    assert world == 8
    pos = cli.past
    assert CP * 64 <= pos + 1 <= CACHE_LEN

    cpa = FullAttnCP(cli.layer, rank, world, dev)
    ref = FullAttn(cli.layer, rank, world, dev, keep_ref=False)  # non-CP baseline (own buffers)

    # replicated residual x; KV replicated within each kv-head group (ranks 0-3 share head 0, 4-7 head 1)
    xg = torch.Generator(device=dev).manual_seed(1234)
    x = torch.randn(HID, dtype=torch.bfloat16, device=dev, generator=xg)
    kg = torch.Generator(device=dev).manual_seed(4321 + rank // 4)
    pk = (
        torch.randn(pos, HEAD_DIM, dtype=torch.bfloat16, device=dev, generator=kg) * 0.1
    ).contiguous()
    pv = (
        torch.randn(pos, HEAD_DIM, dtype=torch.bfloat16, device=dev, generator=kg) * 0.1
    ).contiguous()
    for a in (cpa, ref):
        a.k_cache[:pos].copy_(pk)
        a.v_cache[:pos].copy_(pv)

    cp_out = cpa.forward_cp(x, pos).clone()
    ref_out = ref.forward(x, pos).clone()  # non-CP full-attn (same token)
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        d = (cp_out - ref_out).norm() / ref_out.norm().clamp_min(1e-9)
        print(f"== CP vs non-CP full-attn @ S={pos + 1} (layer {cli.layer}) ==", flush=True)
        print(
            f"   cp_status={int(cpa.cp_status.item())} tp_status={int(cpa.tp_status.item())}",
            flush=True,
        )
        print(
            f"   relL2(cp, noncp) = {d.item():.3e}   cp[absmax]={cp_out.abs().max():.3f} ref[absmax]={ref_out.abs().max():.3f}",
            flush=True,
        )
        print(
            f"   VERDICT: {'PASS' if d.item() < 5e-2 else 'FAIL'} (fp8-band expected ~1e-2)",
            flush=True,
        )

    # full device-S chain: forward_dyn_cp (slot-driven pos + shard) vs forward_cp
    cpa.slot[1] = pos
    dyn_out = cpa.forward_dyn_cp(x).clone()
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        dd = (dyn_out - cp_out).norm() / cp_out.norm().clamp_min(1e-9)
        print(
            f"   forward_dyn_cp vs forward_cp: relL2={dd.item():.3e} "
            f"{'PASS' if dd.item() < 1e-2 else 'FAIL'}",
            flush=True,
        )

    if cli.bench:
        if rank == 0:
            print("== CP vs non-CP full-attn timing (us/call, MAX across ranks) ==", flush=True)
        for S in (32768, 65536):
            p = S - 1
            kg2 = torch.Generator(device=dev).manual_seed(4321 + rank // 4)
            pk2 = (
                torch.randn(p, HEAD_DIM, dtype=torch.bfloat16, device=dev, generator=kg2) * 0.1
            ).contiguous()
            pv2 = (
                torch.randn(p, HEAD_DIM, dtype=torch.bfloat16, device=dev, generator=kg2) * 0.1
            ).contiguous()
            for a in (cpa, ref):
                a.k_cache[:p].copy_(pk2)
                a.v_cache[:p].copy_(pv2)
            t_cp = _time(lambda: cpa.forward_cp(x, p), dev)
            t_nc = _time(lambda: ref.forward(x, p), dev)
            if rank == 0:
                who = "CP wins" if t_cp < t_nc else "non-CP wins"
                print(
                    f"   S={S:6d}: CP {t_cp:7.1f} us   non-CP {t_nc:7.1f} us   delta {t_nc - t_cp:+7.1f} us  ({who})",
                    flush=True,
                )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
