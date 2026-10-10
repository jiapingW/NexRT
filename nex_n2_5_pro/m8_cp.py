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

"""M=8 context-parallel full-attention arm.

M8CpVerify subclasses M8Verify and overrides the attention path and supplies its CP status to the M8 sampling tail.
`use_cp=False` delegates to the non-CP implementation verbatim, so the same
object can drive either production graph without sharing their mutable state.

WHAT IS NEW HERE, AND WHY.
The CP arm owns its own front, partial, local split reduction, publication,
barrier/generation state and four-rank merge, all in the M=8-only TU
`kernels/full_attn/nex_n2_5_pro_m8_cp_attn_ctypes.cu`. Nothing from the BS=1 CP arm
(`nex_n2_5_pro_cp_hopper_mega_ctypes.cu`, `nex_n2_5_pro_cp_tail_ctypes.cu`) or the M=4 CP arm
(`nex_n2_5_pro_m4_cp_control_ctypes.cu`) is called: those were protocol references. The
M=8 CP graphs therefore contain no BS=1 or M=4 production kernel symbols; the
graph-isolation test enforces that property from captured kernel names.

WHAT IS SHARED, AND WHY THAT IS SAFE.
Only immutable engine weights, the CP block's own o-projection/qn/kn shards, the
rope tables, and the REPLICATED KV cache of the non-CP twin. No second KV cache
and no duplicated expert or attention weights are allocated. The KV cache is
legitimately shared because CP shards READ WINDOWS, not storage: every rank in a
CP group holds the same replica and writes the same provisional rows C..C+7.

THE PUBLISH/MERGE PAIR IS A COLLECTIVE OVER THE FOUR-RANK GROUP.
Every rank must invoke `_attn_cp_core` the SAME NUMBER OF TIMES per replay: the
arrival bumps a monotonic generation and adds 1 to every peer's barrier, and the
merge waits for generation*CP. A captured graph guarantees this in production
(all four ranks replay the same graph), but any host-side driver that varies the
invocation count per rank -- e.g. a test whose loop bound depends on cp_rank --
desynchronises the generations and the merge blocks until its spin cap. That is
correct fail-closed behaviour, not a bug, but it is a real usage constraint;
`assert_core_balanced()` exists so a driver can turn it into a named failure
instead of a hang.

THE TWO-LEVEL REDUCTION IS TWO LAUNCHES, ON PURPOSE.
  level 1  M8CpLocalCombineKernel  ns flash partials -> (ctx, lse) per (row,head)
  level 2  M8CpMergeKernel         four ranks -> the owning rank's four heads
with a one-thread arrival kernel between them. The arrival is its own launch so
that the same-stream kernel boundary -- not a hand-rolled software fence -- is
what makes every published byte visible before the release-ordered multicast
add. The split also gives publication a kernel-boundary system-ordering point
before the peer merge.
"""

import ctypes
from pathlib import Path

import torch

from nex_n2_5_pro.m8_verify import HID, LOCAL, M, M8Verify, pack_gemm_bf16tc_m8

HERE = Path(__file__).resolve().parent

CP = 4  # ranks per CP group (share one KV head)
NHCP = 16  # q heads computed per rank under CP
NHL = 4  # heads the rank keeps after the merge
HEAD_DIM = 256
CP_QGATE_ROWS = 5632  # 16*256 q | 4*256 gate | 256 k | 256 v

GUARD = 2048  # canary elements on each side of every CP workspace
GUARD_F = -1.2345678e30  # exactly representable in fp32 and bf16
GUARD_BF = -1.2345678e30


class M8CpVerify(M8Verify):
    def __init__(self, eng, dev, *, use_cp=True, dflash_tap_layers=None):
        super().__init__(eng, dev, dflash_tap_layers=dflash_tap_layers)
        self.use_cp = bool(use_cp)
        # cp_status is word 0 of a guarded allocation: words 1.. hold a fixed
        # canary so that an out-of-bounds protocol write cannot hide behind a
        # reporting bug. cp_guard_ok() is checked by the correctness gate.
        self._cp_status_buf = torch.zeros(64, dtype=torch.int32, device=dev)
        self._cp_status_buf[1:].fill_(0x5A5A5A5A)
        self.cp_status = self._cp_status_buf[:1]
        self.cp_sampling_upstream = self.sampling_upstream.clone()
        self.cp_sampling_upstream[4] = self.cp_status.data_ptr()
        self.inject_cp_fail_rank = -1
        self.cp_core_calls = 0
        self._cp_by_attn = {}
        self._cp_packs = {}
        if not use_cp:
            return

        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem
        from nex_n2_5_pro.kernel_build import build_kernel

        missing = sorted(set(eng.attn) - set(getattr(eng, "attn_cp", {})))
        if missing:
            raise RuntimeError(
                f"M8 CP arm requires DecodeEngine(cp=True); missing CP blocks for layers {missing}"
            )
        self._cp_by_attn = {id(eng.attn[layer]): eng.attn_cp[layer] for layer in eng.attn}

        vp, ci, cf, cu, cl = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_ulonglong,
            ctypes.c_long,
        )
        self.cpl = ctypes.CDLL(
            str(
                build_kernel(
                    HERE / "kernels" / "full_attn",
                    "nex_n2_5_pro_m8_cp_attn_ctypes",
                    hopper=True,
                    link_cuda=True,
                )
            )
        )
        self.cpl.NexN25ProM8CpQGateLaunch.argtypes = [vp] * 9
        self.cpl.NexN25ProM8CpNormRopeLaunch.argtypes = [vp] * 11 + [ci, cf, ci, vp]
        self.cpl.NexN25ProM8CpPartialLaunch.argtypes = [vp] * 4 + [ci, cf, ci] + [vp] * 4
        self.cpl.NexN25ProM8CpLocalCombineLaunch.argtypes = [vp] * 5 + [ci, cf] + [vp] * 2
        self.cpl.NexN25ProM8CpArriveLaunch.argtypes = [cu] + [vp] * 4
        self.cpl.NexN25ProM8CpMergeLaunch.argtypes = [vp] * 5 + [vp] * 3 + [ci, vp, cl, vp]

        # Fixed production geometry: coalesced partial, 132 CTAs, and the
        # split combine/arrival publication sequence.
        self.cp_hs = 2
        self.cp_ns = 132

        self.cp_rank = eng.rank % CP
        self.cp_gid = eng.rank // CP
        self.cp_merge_pf_mb = 4

        # ---- CP-private working buffers (transient, reused across the graph) ----
        # Every one of them is GUARDED: allocated with a canary pad on each side
        # and handed to the kernels as the middle slice, so an out-of-bounds
        # write by a new kernel is caught by cp_guard_ok() instead of silently
        # corrupting whatever the allocator put next. The pads cost 1.5 MB total.
        self._guards = []
        self.cp_q_raw8 = self._guarded((M, NHCP * HEAD_DIM), torch.bfloat16, dev)
        # head-group-major [kHS][8][16/kHS][256] so one TMA box feeds one CTA
        self.cp_q_out8 = self._guarded((M * NHCP, HEAD_DIM), torch.bfloat16, dev)
        self.cp_ws_m = self._guarded((M * self.cp_ns * NHCP,), torch.float32, dev)
        self.cp_ws_l = self._guarded((M * self.cp_ns * NHCP,), torch.float32, dev)
        self.cp_ws_acc = self._guarded((M * self.cp_ns * NHCP * HEAD_DIM,), torch.bfloat16, dev)

        # ---- the 4-rank CP channel: ONE pool for every CP layer ----
        from nex_n2_5_pro.attn_cp import _cp_group

        group = _cp_group(eng.rank)
        self.cp_group_name = group.group_name

        def _symm(numel, dtype):
            x = symm_mem.empty(numel, dtype=dtype, device=dev)
            h = symm_mem.rendezvous(x, group.group_name)
            peers = torch.tensor(list(h.buffer_ptrs), dtype=torch.int64, device=dev)
            return x, peers, int(getattr(h, "multicast_ptr", 0) or 0)

        # Symmetric buffers carry a TRAILING guard only: peers address them from
        # the rendezvous base, so a leading pad would shift every peer pointer.
        # cp_ctx/cp_lse are double buffers indexed by generation parity.
        # The combine publishes generation g into slice g & 1 and the merge
        # reads slice g & 1, so the slice a straggling peer is still reading
        # (the other parity) is never the one being overwritten -- the WAR race
        # a peer would need to lag two full generations before it could observe
        # an overwritten slice; the merge carries a fail-closed tripwire.
        ctx_full, self.cp_peer_ctx, _ = _symm(2 * M * NHCP * HEAD_DIM + GUARD, torch.bfloat16)
        lse_full, self.cp_peer_lse, _ = _symm(2 * M * NHCP + GUARD, torch.float32)
        self.cp_bar, _, self.cp_bar_mc = _symm(128, torch.int32)
        ctx_full.zero_()
        lse_full.zero_()
        self.cp_bar.zero_()
        self.cp_ctx_full = ctx_full[: 2 * M * NHCP * HEAD_DIM].view(2, M * NHCP * HEAD_DIM)
        self.cp_lse_full = lse_full[: 2 * M * NHCP].view(2, M * NHCP)
        ctx_full[2 * M * NHCP * HEAD_DIM :].fill_(GUARD_BF)
        lse_full[2 * M * NHCP :].fill_(GUARD_F)
        self._guards.append((ctx_full[2 * M * NHCP * HEAD_DIM :], GUARD_BF))
        self._guards.append((lse_full[2 * M * NHCP :], GUARD_F))
        if self.cp_bar_mc == 0:
            raise RuntimeError("M8 CP publication requires NVLS multicast")
        self.cp_it = torch.zeros(1, dtype=torch.int32, device=dev)
        self.cp_gen = torch.zeros(1, dtype=torch.int32, device=dev)
        torch.cuda.synchronize()
        dist.barrier()

    # ---- current-generation compatibility views ----------------------------
    # Tests read cp_ctx/cp_lse as "the slice this rank just
    # published". Under parity buffering that is slice (cp_gen & 1), so expose
    # them as properties resolving the CURRENT generation. Host-side only
    # (they synchronize); kernels always receive the FULL double buffer and
    # pick the slice on device, which is what keeps graph replay correct.
    @property
    def cp_ctx(self):
        return self.cp_ctx_full[int(self.cp_gen[0].item()) & 1]

    @property
    def cp_lse(self):
        return self.cp_lse_full[int(self.cp_gen[0].item()) & 1]

    def _guarded(self, shape, dtype, dev):
        """Allocate shape with a GUARD-element canary pad on both sides and
        return the middle view. Registers the pads for cp_guard_ok()."""
        n = 1
        for s in shape:
            n *= s
        full = torch.empty(n + 2 * GUARD, dtype=dtype, device=dev)
        fill = GUARD_F if dtype == torch.float32 else GUARD_BF
        full[:GUARD].fill_(fill)
        full[GUARD + n :].fill_(fill)
        self._guards.append((full[:GUARD], fill))
        self._guards.append((full[GUARD + n :], fill))
        return full[GUARD : GUARD + n].view(*shape)

    # ---- prepacks -----------------------------------------------------------
    def _cp_pack(self, W):
        key = W.data_ptr()
        pk = self._cp_packs.get(key)
        if pk is None:
            if W.shape[0] != CP_QGATE_ROWS:
                raise RuntimeError(
                    f"CP qgate weight has {W.shape[0]} rows, expected {CP_QGATE_ROWS}"
                )
            pk = pack_gemm_bf16tc_m8(W, HID)
            self._cp_packs[key] = pk
        return pk

    def build_packs(self):
        super().build_packs()
        if self.use_cp:
            for cp in self._cp_by_attn.values():
                self._cp_pack(cp.w["qgate_w"])

    # ---- the CP attention layer --------------------------------------------
    def _attn_layer(self, attn):
        if not self.use_cp:
            return super()._attn_layer(attn)
        cp = self._cp_by_attn.get(id(attn))
        if cp is None:
            raise RuntimeError("M8 CP arm received an unmapped full-attention block")
        p, sp = self._p, self._sp
        ci, cf = ctypes.c_int, ctypes.c_float
        w = cp.w
        hg_heads = NHCP // self.cp_hs

        # 1. CP qgate: 16 group q heads + this rank's 4 gate heads, and the
        #    provisional K/V for rows C..C+7 written ONCE each.
        self._ck(
            self.cpl.NexN25ProM8CpQGateLaunch(
                p(self.x8),
                p(self._cp_pack(w["qgate_w"])),
                p(w["qgate_scales"]),
                p(self.cp_q_raw8),
                p(self.gate8),
                p(attn.k_cache),
                p(attn.v_cache),
                p(self.ctl),
                sp(),
            ),
            "m8 cp qgate",
        )
        # 2. CP norm + partial RoPE, 16 q heads, head-group-major q_out.
        self._ck(
            self.cpl.NexN25ProM8CpNormRopeLaunch(
                p(self.cp_q_raw8),
                p(attn.k_cache),
                p(w["qn"]),
                p(w["kn"]),
                p(cp.cos_tab),
                p(cp.sin_tab),
                p(self.cp_q_out8),
                p(self.ctl),
                p(self.x8),
                p(attn.v_cache),
                p(self.gate8),
                ci(HID),
                cf(self._eps),
                ci(hg_heads),
                sp(),
            ),
            "m8 cp norm rope",
        )
        self._attn_cp_core(attn)
        # 7. the UNCHANGED M=8 o-projection consumes xeff8 exactly as in non-CP.
        self._ck(
            self.aol.NexN25ProM8AttnOProjLaunch(
                p(self.xeff8),
                p(self._pack_of(self._oproj_packs, attn.w["o_w"], LOCAL)),
                p(attn.w["o_scales"]),
                ctypes.c_ulonglong(self.op_rep_mc),
                p(self.op_rep),
                p(self.op_it),
                p(self.op_status),
                p(self.x8),
                self.eng.rank,
                self.eng.world,
                sp(),
            ),
            "m8 cp attn oproj",
        )

    def assert_core_balanced(self):
        """Raise if the ranks have not invoked the CP core the same number of
        times. Cheap insurance against the one usage error that manifests as a
        device hang instead of an exception."""
        import torch.distributed as dist

        v = torch.tensor(
            [self.cp_core_calls, -self.cp_core_calls], dtype=torch.int64, device=self.dev
        )
        dist.all_reduce(v, op=dist.ReduceOp.MAX)
        hi, lo = int(v[0].item()), -int(v[1].item())
        if hi != lo:
            raise RuntimeError(
                f"CP core invocation count diverged across ranks ({lo}..{hi}); "
                "the publish/merge pair is a four-rank collective and every rank "
                "must call it the same number of times per replay"
            )
        return hi

    def cp_protocol_reset(self):
        """Resynchronise the four-rank CP channel. reset_unit() clears the
        LEVEL-TRIGGERED status latches, but a round in which a rank failed to
        publish leaves that group's barrier permanently one arrival short, so
        every later round would also time out. A protocol reset is therefore
        part of the retry contract after an ARRIVAL failure (it is not needed
        after an ordinary compute failure, which still publishes).

        Host-side only, outside capture: it must never be baked into a graph."""
        import torch.distributed as dist

        if not self.use_cp:
            return
        torch.cuda.synchronize()
        dist.barrier()
        self.cp_bar.zero_()
        self.cp_it.zero_()
        self.cp_gen.zero_()
        self.cp_status.zero_()
        self._cp_status_buf[1:].fill_(0x5A5A5A5A)
        self.cp_core_calls = 0
        torch.cuda.synchronize()
        dist.barrier()

    def _attn_cp_core(self, attn):
        """Steps 3-6: partial, local split reduction, publication, four-rank
        merge. Split out from the front so a test can drive the PARTITION with a
        caller-supplied q_out and an untouched K/V cache -- the CP front rewrites
        K/V at rows C..C+7, which would silently defeat a poison or sentinel
        probe placed there."""
        p, sp = self._p, self._sp
        ci, cf, cu, cl = (ctypes.c_int, ctypes.c_float, ctypes.c_ulonglong, ctypes.c_long)
        self.cp_core_calls += 1
        # 3. CP partial over THIS rank's per-row shard windows.
        self._ck(
            self.cpl.NexN25ProM8CpPartialLaunch(
                p(self.cp_q_out8),
                p(attn.k_cache),
                p(attn.v_cache),
                p(self.ctl),
                ci(self.cp_ns),
                cf(self._scaling),
                ci(self.cp_rank),
                p(self.cp_ws_m),
                p(self.cp_ws_l),
                p(self.cp_ws_acc),
                sp(),
            ),
            "m8 cp partial",
        )
        # 4. level 1: local split reduction -> this rank's (ctx, lse) slice.
        bar = cu(self.cp_bar_mc if self.inject_cp_fail_rank != self.eng.rank else 0)
        self._ck(
            self.cpl.NexN25ProM8CpLocalCombineLaunch(
                p(self.cp_ws_m),
                p(self.cp_ws_l),
                p(self.cp_ws_acc),
                p(self.cp_ctx_full),
                p(self.cp_lse_full),
                ci(self.cp_ns),
                cf(self._scaling),
                p(self.cp_it),
                sp(),
            ),
            "m8 cp local combine",
        )
        # 5. publication arrival; the stream boundary is the visibility fence.
        self._ck(
            self.cpl.NexN25ProM8CpArriveLaunch(
                bar, p(self.cp_it), p(self.cp_gen), p(self.cp_status), sp()
            ),
            "m8 cp arrive",
        )
        # 6. level 2: four-rank LSE merge -> this rank's four heads, gated.
        oproj_weight = self._pack_of(self._oproj_packs, attn.w["o_w"], LOCAL)
        pf_bytes = (
            min(self.cp_merge_pf_mb << 20, oproj_weight.numel() * oproj_weight.element_size())
            if self.cp_merge_pf_mb > 0
            else 0
        )
        self._ck(
            self.cpl.NexN25ProM8CpMergeLaunch(
                p(self.cp_peer_ctx),
                p(self.cp_peer_lse),
                p(self.gate8),
                p(self.ctx8),
                p(self.xeff8),
                p(self.cp_bar),
                p(self.cp_status),
                p(self.cp_gen),
                ci(self.cp_rank),
                p(oproj_weight) if pf_bytes else None,
                cl(pf_bytes),
                sp(),
            ),
            "m8 cp merge",
        )

    # ---- state hygiene ------------------------------------------------------
    def seed(self, length, verify_input_ids):
        super().seed(length, verify_input_ids)
        self.cp_status.zero_()
        self._cp_status_buf[1:].fill_(0x5A5A5A5A)
        return length

    def reset_unit(self):
        super().reset_unit()
        if self.use_cp:
            self.cp_status.zero_()
            self._cp_status_buf[1:].fill_(0x5A5A5A5A)

    def cp_status_max(self):
        """Max cp_status across the whole TP world, reduced ON DEVICE in the
        status word's own dtype -- no host round-trip and no int64 promotion."""
        import torch.distributed as dist

        v = self.cp_status.detach().clone()
        dist.all_reduce(v, op=dist.ReduceOp.MAX)
        return int(v[0].item())

    def cp_memory_report(self):
        """Actual bytes the CP arm adds, measured from the allocations rather
        than estimated. The KV cache is deliberately absent: CP shards read
        windows, not storage, so it is SHARED with the non-CP twin and no second
        cache exists."""
        if not self.use_cp:
            return {}

        def nb(t):
            return t.numel() * t.element_size()

        priv = sum(
            nb(t)
            for t in (self.cp_q_raw8, self.cp_q_out8, self.cp_ws_m, self.cp_ws_l, self.cp_ws_acc)
        )
        symm = sum(nb(t) for t in (self.cp_ctx_full, self.cp_lse_full, self.cp_bar))
        guards = sum(nb(g) for g, _ in self._guards)
        packs = sum(nb(v) for v in self._cp_packs.values())
        wts, seen = 0, set()
        for cp in self._cp_by_attn.values():
            for k in ("qgate_w", "qgate_scales", "qn", "kn", "o_w", "o_scales"):
                tw = cp.w.get(k)
                if tw is not None and tw.data_ptr() not in seen:
                    seen.add(tw.data_ptr())
                    wts += nb(tw)
        return {
            "cp_workspaces_B": priv,
            "cp_symmetric_B": symm,
            "cp_guard_pads_B": guards,
            "cp_qgate_prepacks_B": packs,
            "cp_block_weights_B": wts,
            "kv_cache_B": 0,
            "total_B": priv + symm + guards + packs + wts,
            "note": "kv_cache_B is 0 BY DESIGN: the CP arm aliases the "
            "non-CP twin's replicated cache and allocates no second "
            "one. The dominant term is duplicated per-block weights "
            "(qgate_w plus its prepack, and the o_w/qn/kn copies "
            "FullAttnCP holds); the prepack source could be freed "
            "after packing and the o_w/qn/kn copies are the known "
            "deferred duplication noted in attn_cp.py.",
        }

    def _tail(self):
        self._launch_sampling(self.cp_sampling_upstream if self.use_cp else None)

    def cp_guard_ok(self):
        """True iff no CP kernel wrote outside any CP buffer it was handed."""
        if not bool((self._cp_status_buf[1:] == 0x5A5A5A5A).all()):
            return False
        for pad, fill in getattr(self, "_guards", []):
            if not bool((pad == fill).all()):
                return False
        return True


def build_m8_cp(rank, world, dev, eng=None, use_cp=True, *, dflash_tap_layers=None):
    if eng is None:
        from nex_n2_5_pro.decode_engine import DecodeEngine

        eng = DecodeEngine(rank, world, dev, cp=use_cp)
    return M8CpVerify(eng, dev, use_cp=use_cp, dflash_tap_layers=dflash_tap_layers)
