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

"""nexrt nex_n2_5_pro DecodeEngine (TP=8): token ID in -> token ID out in one whole-step graph.

Whole-step pipeline, initialized from the prefill state:
  embed(slot[0]) -> [GDN/full-attention mixer + MoE for each layer] -> final norm -> head
  -> sampler (writes slot[0]=next token, bumps slot[1]=S+1).

The [token_id, S] device slot carries the whole feedback loop: embed reads slot[0], the full-attn front
kernels read slot[1]=S (device-S growing KV), the sampler writes slot[0] + bumps slot[1]. The mixer/MoE/head
kernels FUSE their RMSNorms, so every block forward takes the RAW residual stream x.

Two run modes:
  eager (default): the step() chain run kernel-by-kernel — validates wiring/state-loading/token stream.
  --graph:        the WHOLE step() captured as ONE CUDA graph, replayed per token. The [token,S] slot
                  device-drives the loop (nothing host-side inside the graph); residual adds use a custom
                  kernel (no torch in the replayed path). The greedy (argmax) stream is bit-identical to
                  the eager path. The mixer/MoE push reduces sum in a fixed order (parity selects only
                  the buffer half, never the accumulation order); validation still uses argmax so a
                  knife-edge nucleus draw cannot flip the comparison.

Run:  torchrun --standalone --nproc_per_node=8 decode_engine.py --steps 32            # eager
      torchrun --standalone --nproc_per_node=8 decode_engine.py --steps 16 --graph    # one CUDA graph
"""

import argparse
import ctypes
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(HERE.parent))

from nex_n2_5_pro.gdn import Gdn  # noqa: E402
from nex_n2_5_pro.moe import Moe  # noqa: E402
from nex_n2_5_pro.attn import FullAttn, CACHE_LEN  # noqa: E402  (CACHE_LEN: KV budget -> generate() stop + cp-sweep guard)
from nex_n2_5_pro.attn_cp import FullAttnCP  # noqa: E402  (context-parallel full-attn -- long-context decode graph)
from nex_n2_5_pro.head import Head  # noqa: E402
from nex_n2_5_pro.kernel_build import build_kernel  # noqa: E402
from nex_n2_5_pro.ctl_mirror import CtlMirror  # noqa: E402  (host mirror of the [token, S] slot: generate() drain)
from nex_n2_5_pro.config import LAYERS, LAYER_IS_LINEAR  # noqa: E402
from nex_n2_5_pro.weights import load_embedding  # noqa: E402

HID = 4096
# CP_MIN_S is a safety floor for the partial-attention implementation.
# CP_DISPATCH_S controls when ordinary decode selects the CP graph.
CP_MIN_S = 1024
CP_DISPATCH_S = 48 * 1024
KEMBED = HERE / "kernels" / "embed"
KUTIL = HERE / "kernels" / "util"
STATE_DIR = HERE / os.environ.get(
    "NEXRT_STATE_DIR", "prefill_state"
)  # 60-layer: NEXRT_STATE_DIR=prefill_state_60


def build(kdir, name):
    return build_kernel(kdir, name)


def _p(t):
    return ctypes.c_void_p(t.data_ptr())


def _sp():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


class DecodeEngine:
    def __init__(self, rank, world, dev, cp=False, ext=None):
        """ext (bridge mode): an adapter exposing sglang's loaded per-rank tensors
        (sglang_prefill/ext_weights.SglangExt) -- blocks alias the bit-identical tensors and
        derive their folded privates on-GPU; embed comes from an NVLink all-gather of the vocab
        shards. The CP twin still reads from LAYERS_DIR (needs the group's 16 q heads, which a
        rank does not hold locally)."""
        self.rank, self.world, self.dev = rank, world, dev
        self.slot = torch.zeros(
            2, dtype=torch.int32, device=dev
        )  # [token_id, S] — the feedback slot
        self.x = torch.empty(HID, dtype=torch.bfloat16, device=dev)  # replicated residual stream

        self.layers = LAYERS  # full 60-layer model by default
        self.is_lin = LAYER_IS_LINEAR  # L -> True(GDN) / False(full-attn)
        # Build blocks. Two things make 60 layers fit:
        #  - keep_ref=False frees each block's raw fp8 after conversion (raw MoE is 6.6 GB/layer -> 396 GB/rank).
        #  - share_from=<first of type>: all blocks of a type SHARE one symmetric-memory reduce workspace. The
        #    reduces run sequentially (barrier-synced) and one NVLS multicast set per block (~300 rendezvous at
        #    60 layers) would exhaust the pool; the shared per-CTA epoch counters advance per reduce, so every
        #    launch derives a unique parity/arrival generation with no host resets.
        self.gdn, self.attn, self.moe = {}, {}, {}
        gdn0 = attn0 = moe0 = None
        for L in self.layers:
            if self.is_lin[L]:
                self.gdn[L] = Gdn(
                    L,
                    rank,
                    world,
                    dev,
                    keep_ref=False,
                    share_from=gdn0,
                    ext=ext.gdn(L) if ext else None,
                )
                gdn0 = gdn0 or self.gdn[L]
            else:
                self.attn[L] = FullAttn(
                    L,
                    rank,
                    world,
                    dev,
                    keep_ref=False,
                    share_from=attn0,
                    ext=ext.attn(L) if ext else None,
                )
                attn0 = attn0 or self.attn[L]
                self.attn[L].slot = self.slot  # share the [token,S] slot with every full-attn
            self.moe[L] = Moe(
                L,
                rank,
                world,
                dev,
                keep_ref=False,
                share_from=moe0,
                ext=ext.moe(L) if ext else None,
            )
            moe0 = moe0 or self.moe[L]
        # optional CP (context-parallel) full-attn blocks for the long-context (S>=threshold) decode graph. Each
        # SHARES its non-CP twin's k_cache/v_cache/slot + 8-rank o-proj reduce (no HBM / NVLS dup); the first CP
        # block owns one 4-rank CP channel that the rest share. Only the full-attn front + 1/4-shard partial differ.
        self.attn_cp, cp0 = {}, None
        if cp:
            for L in self.layers:
                if not self.is_lin[L]:
                    self.attn_cp[L] = FullAttnCP(
                        L, rank, world, dev, share_noncp=self.attn[L], share_cp=cp0
                    )
                    cp0 = cp0 or self.attn_cp[L]
        # Wait-window prefetches, baked into the captured graphs (the kernels treat (0, 0) as off):
        #  - each mixer's fused o_proj reduce L2-warms the SAME layer's MoE gate weights during its
        #    peer wait (the window absorbs the full 4.3 MB payload);
        #  - each MoE L2 reduce L2-warms the NEXT layer's front GEMV weights, CLAMPED to 4 MiB --
        #    the clamped head of the range is what the front kernel touches first.
        # The front target depends on the graph (non-CP qkv front vs CP qgate front), so step() /
        # step_cp() select which tuple each capture bakes; the non-CP tuple is also the default so
        # direct forward() paths (--profile/--scale) run the same config as the non-CP graph.
        front_pf_clamp = 4 << 20

        def _pf_range(w):
            return (w.data_ptr(), min(w.numel() * w.element_size(), front_pf_clamp))

        for L in self.layers:
            gw = self.moe[L].k["gate_w"]
            blk = self.gdn[L] if self.is_lin[L] else self.attn[L]
            blk.wait_pf = (gw.data_ptr(), gw.numel() * gw.element_size())
            self.moe[L].front_pf_noncp = self.moe[L].front_pf_cp = (0, 0)
        for i, L in enumerate(self.layers[:-1]):
            nxt = self.layers[i + 1]
            if self.is_lin[nxt]:
                self.moe[L].front_pf_noncp = self.moe[L].front_pf_cp = _pf_range(
                    self.gdn[nxt].w["qkvz_w"]
                )
            else:
                self.moe[L].front_pf_noncp = _pf_range(self.attn[nxt].w["qkv_w"])
                self.moe[L].front_pf_cp = (
                    _pf_range(self.attn_cp[nxt].w["qgate_w"])
                    if nxt in self.attn_cp
                    else self.moe[L].front_pf_noncp
                )
        for L in self.layers:
            self.moe[L].front_pf = self.moe[L].front_pf_noncp
        self.head = Head(rank, dev, ext=ext.head() if ext else None)
        vp = ctypes.c_void_p

        # [vocab, 4096] bf16, replicated per rank (bridge: NVLink all-gather of sglang's vocab shards)
        self.embed_w = ext.embed(dev) if ext else load_embedding(dev)
        self.embl = ctypes.CDLL(str(build(KEMBED, "nex_n2_5_pro_embed_gather_ctypes")))
        self.embl.NexN25ProEmbedGatherLaunch.argtypes = [vp] * 4
        self.rl = ctypes.CDLL(
            str(build(KUTIL, "nex_n2_5_pro_residual_add_ctypes"))
        )  # residual add (no torch in the graph)
        self.rl.NexN25ProResidualAddLaunch.argtypes = [vp, vp, ctypes.c_int, vp]
        self.slot_mirror = CtlMirror(
            self.slot, dev
        )  # published by step()/step_cp() as the last node
        torch.cuda.synchronize()
        dist.barrier()

    def load_prefill(self, state_dir=None):
        """Seed the block caches from this rank's prefill state; set slot = [first_token, L].
        state_dir overrides the env STATE_DIR (the matrix driver sweeps per-length dirs through one engine)."""
        sd = state_dir if state_dir is not None else STATE_DIR
        d = torch.load(sd / f"prefill_state_rank{self.rank}.pt", map_location="cpu")
        Lp = None
        for L in self.layers:
            ld = d["layers"][L]
            if self.is_lin[L]:
                self.gdn[L].conv_state4[:, 0:3].copy_(
                    ld["conv_state"].to(self.dev)
                )  # [1536,3] past conv inputs
                self.gdn[L].conv_state4[:, 3].zero_()
                # LA saves recurrent_state as [HEADS, K, V]; gdn.state_t is [HEADS, V, K] (its recurrence uses
                # s0 = state_t.transpose(1,2)), so load transposed.
                self.gdn[L].state_t.copy_(
                    ld["recurrent_state"].to(self.dev).transpose(1, 2).contiguous()
                )
            else:
                k, v = ld["k_cache"].to(self.dev), ld["v_cache"].to(self.dev)  # [1, L, 256] bf16
                Lp = k.shape[1]
                self.attn[L].k_cache[:Lp].copy_(k[0])
                self.attn[L].v_cache[:Lp].copy_(v[0])
        self.slot[0] = int(d["first_token"])
        self.slot[1] = Lp  # Lp = prompt len (same for all full-attn layers)
        for b in (self.head, *self.gdn.values(), *self.attn.values(), *self.moe.values()):
            b.status.zero_()  # clear any sticky error flag (reset-safe)
        for b in self.attn_cp.values():  # CP 4-rank merge handshake flag (empty if cp=False;
            b.cp_status.zero_()  # shared across CP blocks, so this over-zeroes harmlessly)
        return Lp, int(d["first_token"])

    def _resid(self, out):  # x += out (out fp32 [HID]) — our kernel, graph-safe
        if getattr(self, "skip_resid", False):  # profiling: isolate the residual cost
            return
        self.rl.NexN25ProResidualAddLaunch(_p(self.x), _p(out), ctypes.c_int(HID), _sp())

    def _head_slot(self, h_raw):
        self.head.forward_slot(h_raw, self.slot)

    def _dbg(self, tag, o=None):
        if getattr(self, "dbg", False) and self.rank == 0:
            om = f" | out absmax={o.abs().max().item():.4f}" if o is not None else ""
            print(
                f"      [{tag:6s}] x absmax={self.x.abs().max().item():.4f} mean={self.x.float().mean().item():+.5f}{om}",
                flush=True,
            )

    def _block(self, fn, x):
        """Run one block. self.fuse (default) -> the block's reduce FOLDS the residual add into its epilogue
        (x += reduced, in place); else forward + a separate residual-add kernel (the pre-fusion path)."""
        if getattr(self, "fuse", True):
            fn(x, resid=x)
        else:
            self._resid(fn(x))

    def step(self):
        """One decode token: reads slot[0]/slot[1], writes slot[0]=next token + slot[1]=S+1. Each block's
        residual add is fused into its reduce epilogue (no separate add kernel in the replayed graph)."""
        for L in self.layers:  # capture-time: bake the non-CP front-prefetch targets
            self.moe[L].front_pf = self.moe[L].front_pf_noncp
        self.embl.NexN25ProEmbedGatherLaunch(_p(self.embed_w), _p(self.slot), _p(self.x), _sp())
        self._dbg("embed")
        for L in self.layers:
            if self.is_lin[L]:
                self._block(self.gdn[L].forward, self.x)
                self._dbg(f"gdn{L}")  # GDN mixer + fused residual
            else:
                self._block(self.attn[L].forward_dyn, self.x)
                self._dbg(f"attn{L}")  # full-attn (device-S) + fused resid
            self._block(self.moe[L].forward, self.x)
            self._dbg(f"moe{L}")  # MoE + fused residual
        self._head_slot(self.x)  # final-norm+head+sampler -> slot
        self.slot_mirror.launch()  # publish [token, S] to the host

    def step_cp(self):
        """Long-context decode token: identical to step() except the full-attn blocks take the CP path
        (16Q front-end + 1/4-shard Hopper megakernel + 4-rank LSE-merge). Same shared slot / k_cache /
        residual stream / MoE / head -- captured as the SECOND CUDA graph, replayed when S >= threshold."""
        for L in self.layers:  # capture-time: bake the CP front-prefetch targets
            self.moe[L].front_pf = self.moe[L].front_pf_cp
        self.embl.NexN25ProEmbedGatherLaunch(_p(self.embed_w), _p(self.slot), _p(self.x), _sp())
        for L in self.layers:
            if self.is_lin[L]:
                self._block(self.gdn[L].forward, self.x)  # GDN mixer + fused residual
            else:
                self._block(
                    self.attn_cp[L].forward_dyn_cp, self.x
                )  # CP full-attn (device-S) + fused resid
            self._block(self.moe[L].forward, self.x)  # MoE + fused residual
        self._head_slot(self.x)
        self.slot_mirror.launch()  # publish [token, S] to the host

    def status_max(self):
        vals = [int(self.head.status.item())]
        vals += [int(b.status.item()) for b in self.gdn.values()]  # GDN reduce status (was missing)
        vals += [int(b.status.item()) for b in self.attn.values()]
        vals += [int(b.status.item()) for b in self.moe.values()]
        vals += [
            int(b.cp_status.item()) for b in self.attn_cp.values()
        ]  # CP 4-rank merge handshake (empty if cp=False)
        s = torch.tensor(vals, device=self.dev)
        dist.all_reduce(s, op=dist.ReduceOp.MAX)
        return int(s.max().item())

    def capture(self, warmup=15, step_fn=None):
        """Warm up step() (real execution -- stabilizes the caching allocator + each block's device-side
        generation flags), then capture the WHOLE step() as ONE CUDA graph. Capture only RECORDS the launches
        (no execution), so state is left at the post-warmup point; the caller resets (load_prefill) before
        replaying from a known start. The [token,S] slot device-drives the replayed loop (no host work inside).
        step_fn selects which chain to capture (self.step non-CP / self.step_cp CP) for the two-graph dispatch."""
        step_fn = step_fn or self.step
        for _ in range(warmup):
            step_fn()
        torch.cuda.synchronize()
        dist.barrier()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step_fn()
        torch.cuda.synchronize()
        dist.barrier()
        return g

    def generate(
        self,
        g_noncp,
        g_cp=None,
        max_new_tokens=256,
        eos_id=None,
        cp_threshold=CP_DISPATCH_S,
        on_tokens=None,
    ):
        """Custom decode loop: replay the dispatched graph per token until EOS or the KV budget, reading the
        in-graph sampler's token straight from the [token,S] slot. The slot is REPLICATED across all 8 ranks
        (the head reducer gathers the global top-k candidates), so every rank runs this same loop and checks EOS/max
        INDEPENDENTLY -- no broadcast, they stop on the same step. Comm kernels self-advance (monotonic gen, no
        per-step reset). Each step publishes [token,S] to a host mirror as its last node (ctl_mirror.py); the
        host reads it from host memory and keeps the NEXT step queued while it does, so the GPU never drains
        between steps. S advances by exactly one per step, so the graph choice (CP crossover) and the KV-budget
        stop of step k+1 are known before step k's token is read; only an EOS stop is learned one step late (the
        speculative step completes and its token is not emitted). `eos_id` is one id or a collection of stop ids.
        Assumes load_prefill() seeded slot=[first_token, Lp]. Returns the full token stream incl. first_token.
        ``on_tokens`` receives each post-first-token ID after the host mirror publishes it. The
        callback must not block; the caller owns transport and emits the seeded first token.
        Simple + fast by design: it IS the sampler kernel's stream (this model's inference config), not a
        general HF loop -- there is no host-side sampling/logits work in here."""
        stop_ids = (
            set()
            if eos_id is None
            else {int(eos_id)}
            if isinstance(eos_id, int)
            else {int(t) for t in eos_id}
        )
        mirror = self.slot_mirror
        tok, S = self.slot.tolist()  # [first_token, Lp]: prefill's first output + start S
        out = [tok]
        max_s = min(CACHE_LEN, S + max_new_tokens)  # never write past the KV budget
        steps = max_new_tokens - 1  # first_token already counts as 1
        if steps <= 0 or tok in stop_ids:
            return out

        def launch(s_now):
            (
                g_cp if (g_cp is not None and s_now >= cp_threshold) else g_noncp
            ).replay()  # host dispatch only

        gen = mirror.last_gen()
        launch(S)
        steps -= 1
        queued = 1
        while True:
            ahead = steps > 0 and S + 1 < max_s  # step k lands at S+1: it cannot hit the KV budget
            if ahead:
                launch(S + 1)  # step k+1 queued behind step k
                steps -= 1
                queued = 2
            gen += 1
            tok, S = mirror.wait(gen)  # step k's [token, S], from host memory
            queued -= 1
            if tok < 0:
                if queued:
                    torch.cuda.current_stream().synchronize()
                raise RuntimeError("sampling failed: invalid logits or sampling parameters")
            out.append(tok)
            if on_tokens is not None:
                on_tokens([tok])
            if S >= max_s or tok in stop_ids:
                break
            if not ahead:
                if steps <= 0:
                    break
                launch(S)
                steps -= 1
                queued = 1
        if queued:  # a speculative step may still be running
            torch.cuda.current_stream().synchronize()
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument(
        "--greedy",
        action="store_true",
        help="feed head top-1 (argmax) instead of the sampled id "
        "for deterministic eager/graph comparisons",
    )
    ap.add_argument("--debug", action="store_true", help="dump per-stage x magnitudes on step 0")
    ap.add_argument(
        "--graph",
        action="store_true",
        help="capture step() as ONE CUDA graph + replay loop; "
        "verify the graph stream == the eager stream, report us/tok",
    )
    ap.add_argument(
        "--profile",
        action="store_true",
        help="time cumulative pipeline prefixes and report successive differences",
    )
    ap.add_argument("--warmup", type=int, default=15, help="warmup steps before graph capture")
    ap.add_argument(
        "--scale",
        action="store_true",
        help="time a graph of one block repeated K times -> is the "
        "per-node graph overhead fixed or per-node (does it scale with layers)?",
    )
    ap.add_argument(
        "--cold", action="store_true", help="in --profile, evict L2 before every prefix replay"
    )
    ap.add_argument(
        "--cp",
        action="store_true",
        help="also build the CP (context-parallel) long-context "
        "full-attn graph; with --graph, dispatch g_cp when S>=--cp-threshold else g_noncp, switched "
        "host-side by S=Lp+step (one int compare, no device read-back -> free-run stays free)",
    )
    ap.add_argument(
        "--cp-threshold",
        type=int,
        default=CP_DISPATCH_S,
        help="context length at which decode selects the CP graph",
    )
    ap.add_argument(
        "--cp-sweep",
        type=str,
        default="",
        help="with --graph --cp: comma S values -> time BOTH graphs "
        "(whole-step E2E) at each context by overriding slot[1] on the seeded state -> the true crossover",
    )
    ap.add_argument(
        "--generate",
        action="store_true",
        help="run the custom generate() loop (real decode: replay "
        "until EOS or the KV budget) instead of the fixed-N validation; self-checks max-stop + EOS-stop",
    )
    ap.add_argument(
        "--eos-id",
        type=int,
        default=None,
        help="stop token id for --generate (default: run to --steps)",
    )
    cli = ap.parse_args()
    if cli.scale and len(LAYERS) < 4:
        ap.error("--scale requires NEXRT_NLAYERS>=4")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl")
    assert world == 8

    eng = DecodeEngine(rank, world, dev, cp=cli.cp)
    if cli.greedy:
        eng.head.set_sampling_config(temperature=0)

    if cli.generate:
        # ---- the REAL decode path: capture graph(s), then generate() until EOS or the KV budget ----
        if cli.cp:
            assert cli.cp_threshold > CP_MIN_S, (
                f"--cp-threshold {cli.cp_threshold} must be > CP_MIN_S={CP_MIN_S}"
            )
        Lp, first = eng.load_prefill()
        g_noncp = eng.capture(warmup=cli.warmup)
        g_cp = eng.capture(warmup=cli.warmup, step_fn=eng.step_cp) if cli.cp else None
        eng.load_prefill()
        toks = eng.generate(
            g_noncp,
            g_cp,
            max_new_tokens=cli.steps,
            eos_id=cli.eos_id,
            cp_threshold=cli.cp_threshold,
        )
        st = eng.status_max()
        # EOS-stop self-check: re-run with an early stream token as a synthetic EOS -> the loop must STOP at it
        k = min(3, len(toks) - 1)
        synth = toks[k]
        eng.load_prefill()
        toks_eos = eng.generate(
            g_noncp, g_cp, max_new_tokens=cli.steps, eos_id=synth, cp_threshold=cli.cp_threshold
        )
        # timing: the REAL loop us/tok (per-step replay + one slot readback = the honest production overhead,
        # unlike the pure device-chained free-run bench which has no per-token host sync)
        eng.load_prefill()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        dist.barrier()
        e0.record()
        eng.generate(
            g_noncp, g_cp, max_new_tokens=cli.steps, eos_id=None, cp_threshold=cli.cp_threshold
        )
        e1.record()
        e1.synchronize()
        us = torch.tensor([e0.elapsed_time(e1) * 1e3 / max(1, cli.steps - 1)], device=dev)
        dist.all_reduce(us, op=dist.ReduceOp.MAX)
        if rank == 0:
            eos_hit = cli.eos_id is not None and toks[-1] == cli.eos_id
            ok_eos = (
                toks_eos[-1] == synth and synth not in toks_eos[:-1]
            )  # stopped AT the first synthetic eos
            valid = (
                all(0 <= t < 248320 for t in toks) and len(set(toks)) > 1
            )  # non-degenerate stream
            ok = st == 0 and ok_eos and valid
            print(
                f"== nexrt generate() (TP=8, {len(eng.layers)}-layer{', CP' if cli.cp else ''}): L={Lp}, max_new={cli.steps} ==",
                flush=True,
            )
            print(
                f"   generated {len(toks)} tok (stop: {'EOS' if eos_hit else 'max/budget'}) | status {st}",
                flush=True,
            )
            print(f"   stream: {toks}", flush=True)
            print(
                f"   EOS-stop check (synthetic eos={synth}): stopped at {len(toks_eos)} tok, last=={synth} -> "
                f"{'OK' if ok_eos else 'FAIL'}",
                flush=True,
            )
            print(
                f"   [{'OK' if ok else 'CHECK'}]  {us.item():.1f} us/tok (real loop incl. per-step slot readback)",
                flush=True,
            )
        dist.barrier()
        dist.destroy_process_group()
        return

    if cli.scale:
        # Time a graph that repeats ONE op K times. latency(K)=a+b*K: b is the true PER-NODE marginal cost in a
        # chained graph, a is fixed overhead. If b == the op's own GPU time, the graph adds ZERO per-node overhead
        # (it would NOT grow with layers beyond the real work); if b > that, there is real per-node overhead.
        eng.load_prefill()
        eng.embl.NexN25ProEmbedGatherLaunch(_p(eng.embed_w), _p(eng.slot), _p(eng.x), _sp())
        torch.cuda.synchronize()

        def time_rep(fn, K, warm=10, iters=200):
            for _ in range(warm):
                for _ in range(K):
                    fn()
            torch.cuda.synchronize()
            dist.barrier()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(K):
                    fn()
            torch.cuda.synchronize()
            for _ in range(5):
                g.replay()
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            dist.barrier()
            a.record()
            for _ in range(iters):
                g.replay()
            b.record()
            b.synchronize()
            t = torch.tensor([a.elapsed_time(b) * 1e3 / iters], device=dev)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            return t.item()

        def three_real():  # 3 DIFFERENT layers (each weight read cold)
            for L in (0, 1, 2):
                eng.gdn[L].forward(eng.x, resid=eng.x)
                eng.moe[L].forward(eng.x, resid=eng.x)

        def same_x3():  # the SAME layer 3x (weights stay hot in L2)
            for _ in range(3):
                eng.gdn[0].forward(eng.x, resid=eng.x)
                eng.moe[0].forward(eng.x, resid=eng.x)

        ops = [
            (
                "residual(tiny 1-CTA)",
                lambda: eng.rl.NexN25ProResidualAddLaunch(
                    _p(eng.x), _p(eng.moe[3].out), ctypes.c_int(HID), _sp()
                ),
            ),
            ("gdn0.forward", lambda: eng.gdn[0].forward(eng.x, resid=eng.x)),
            ("moe0.forward", lambda: eng.moe[0].forward(eng.x, resid=eng.x)),
            ("layer(gdn0+moe0)", None),
            ("3 REAL layers L0+L1+L2", three_real),  # cold weights (like the real step)
            ("same layer L0 x3", same_x3),
        ]  # hot weights (like the isolated sum)
        Ks = (1, 8, 16, 32)
        if rank == 0:
            print(
                "== nexrt per-node graph-overhead probe (TP=8): latency(K) for K repeats in one graph ==",
                flush=True,
            )
        for name, fn in ops:
            if fn is None:
                fn = lambda: (
                    eng.gdn[0].forward(eng.x, resid=eng.x),
                    eng.moe[0].forward(eng.x, resid=eng.x),
                )
            ts = [time_rep(fn, K) for K in Ks]
            if rank == 0:
                slope = (ts[-1] - ts[0]) / (
                    Ks[-1] - Ks[0]
                )  # marginal us per repeat (per-node cost)
                fixed = ts[0] - slope * Ks[0]
                per = "  ".join(f"K={k}:{t:.1f}" for k, t in zip(Ks, ts))
                print(
                    f"   {name:22s} {per}   -> marginal {slope:.2f} us/rep, fixed {fixed:.1f} us",
                    flush=True,
                )
        dist.barrier()
        dist.destroy_process_group()
        return

    if cli.profile:
        # ---- per-block latency IN-CONTEXT via cumulative-prefix graphs ----
        # Timing each block in ITS OWN graph misjudges E2E two ways: HOT isolation keeps that block's weights
        # L2-resident so the block-sum UNDER-estimates E2E (forbidden); COLD isolation adds a per-block cold-start
        # + barrier-resync + relaunch cost so the block-sum over-estimates ~30% (distorts per-kernel judgement).
        # Instead time cumulative PREFIXES of the real step -- embed; embed+block0; embed+block0+block1; ... --
        # under the SAME back-to-back replay condition as the real graph. Block K's cost = t(prefix K+1) -
        # t(prefix K); the per-block SUM then EQUALS the real E2E BY CONSTRUCTION (it telescopes to the full-step
        # latency -- neither over nor under) while each block carries its real in-context cost. --cold flushes L2
        # before every replay for the cold-cache measurement; its sum equals the cold E2E, still exact.
        eng.load_prefill()
        eng.embl.NexN25ProEmbedGatherLaunch(_p(eng.embed_w), _p(eng.slot), _p(eng.x), _sp())
        torch.cuda.synchronize()
        seq = [
            (
                "embed",
                lambda: eng.embl.NexN25ProEmbedGatherLaunch(
                    _p(eng.embed_w), _p(eng.slot), _p(eng.x), _sp()
                ),
            )
        ]
        for L in eng.layers:
            if eng.is_lin[L]:
                seq.append((f"gdn{L}", lambda L=L: eng._block(eng.gdn[L].forward, eng.x)))
            else:
                seq.append((f"attn{L}", lambda L=L: eng._block(eng.attn[L].forward_dyn, eng.x)))
            seq.append((f"moe{L}", lambda L=L: eng._block(eng.moe[L].forward, eng.x)))
        seq.append(("head", lambda: eng._head_slot(eng.x)))
        from nex_n2_5_pro.bench import profile_pipeline  # vendored profiler (nex_n2_5_pro/bench.py)

        pr = profile_pipeline(
            seq, warmup=15, rep=100, flush_l2=cli.cold, tp_reduce=True, device=str(dev)
        )
        names, per, full = (
            pr.names,
            [v * 1e3 for v in pr.per_ms],
            pr.e2e_ms * 1e3,
        )  # ms -> us; sum(per)==full exact
        if rank == 0:
            mode = "COLD / L2-evicted" if cli.cold else "HOT back-to-back (real steady-state)"
            print(
                f"== nexrt per-block profile (TP=8, IN-CONTEXT prefix-increment, {mode}, S={int(eng.slot[1].item())}) ==",
                flush=True,
            )
            for name, v in zip(names, per):
                print(f"   {name:8s} {v:8.2f} us", flush=True)
            print(
                f"   ---- sum of blocks = {sum(per):.1f} us  ==  whole-step E2E {full:.1f} us   "
                f"(sum == E2E by construction: no over/under-estimate)",
                flush=True,
            )
        dist.barrier()
        dist.destroy_process_group()
        return

    if cli.graph and cli.cp and cli.cp_sweep:
        # ---- crossover sweep: time BOTH graphs (whole-step E2E) across contexts from ONE seeded state ----
        # The device-S graphs read slot[1] live, so I load the (e.g. 64K) state once and OVERRIDE slot[1]=S to
        # time each graph AS IF the context were S (reading a prefix of the seeded k_cache; correctness irrelevant
        # for timing). Non-full-attn work is identical across the two graphs, so where CP us/tok crosses non-CP
        # IS the true dispatch threshold. (The in-graph sampler drifts slot[1] by ~reps during timing -- negligible
        # vs S, and stays < CACHE_LEN.)
        Lp, _ = eng.load_prefill()
        g_noncp = eng.capture(warmup=cli.warmup)
        g_cp = eng.capture(warmup=cli.warmup, step_fn=eng.step_cp)
        # CP is only valid in (CP_MIN_S, CACHE_LEN - drift): S<=CP_MIN_S -> empty shards / no benefit; near CACHE_LEN
        # the sampler-driven slot[1] drift over the timed replays would write past the k_cache. Drop out-of-range S.
        req = [int(s) for s in cli.cp_sweep.split(",")]
        Svals = [s for s in req if CP_MIN_S < s <= CACHE_LEN - 256]
        if rank == 0 and len(Svals) < len(req):
            print(
                f"   [cp-sweep] dropped {sorted(set(req) - set(Svals))} (outside valid CP range "
                f"({CP_MIN_S}, {CACHE_LEN - 256}])",
                flush=True,
            )

        def bench_at(graph, S):
            eng.load_prefill()
            eng.slot[1] = S  # seeded to Lp; time as if context = S
            for _ in range(10):
                graph.replay()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            dist.barrier()
            e0.record()
            for _ in range(200):
                graph.replay()
            e1.record()
            e1.synchronize()
            u = torch.tensor([e0.elapsed_time(e1) * 1e3 / 200], device=dev)
            dist.all_reduce(u, op=dist.ReduceOp.MAX)
            return u.item()

        rows = [(S, bench_at(g_noncp, S), bench_at(g_cp, S)) for S in Svals]
        if rank == 0:
            print(
                f"== nexrt CP crossover sweep (TP=8, {len(eng.layers)}-layer, whole-step E2E us/tok, seeded @ {Lp}) ==",
                flush=True,
            )
            xover = None
            for S, un, uc in rows:
                win = "CP" if uc < un else "non-CP"
                if uc < un and xover is None:
                    xover = S
                print(
                    f"   S={S:>6}: non-CP {un:8.1f}  CP {uc:8.1f} us/tok  -> {win} wins (dCP {uc - un:+7.1f})",
                    flush=True,
                )
            print(f"   crossover (first S where CP wins): {xover}", flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return

    if cli.graph and cli.cp:
        # ---- two-graph dispatch: non-CP (S<thresh) + CP (S>=thresh), one graph replayed per token ----
        # The switch key S = Lp + step is known HOST-side (load_prefill returns Lp) -> a single int compare,
        # NO device read-back, so the free-run loop stays free. Both graphs SHARE k_cache / o-proj / 8-rank
        # reduce state; only one runs per token, so the monotonic reduce gen advances consistently across the
        # crossover. CP wins >=~48K (flat) vs non-CP (grows with S); dispatch picks the cheaper per token.
        def argmax_tok():
            return int(eng.head.out_id[int(eng.head.out_val.argmax().item())].item())

        thresh = cli.cp_threshold
        assert thresh > CP_MIN_S, (
            f"--cp-threshold {thresh} must be > CP_MIN_S={CP_MIN_S} (CP invalid at small S)"
        )
        # 1. eager DISPATCH reference (same host-side S switch, kernel-by-kernel)
        Lp, first = eng.load_prefill()
        eager = []
        for step in range(cli.steps):
            (eng.step_cp if (Lp + step) >= thresh else eng.step)()
            torch.cuda.synchronize()
            t = argmax_tok()
            eng.slot[0] = t
            eager.append(t)
        # 2. capture BOTH graphs (each warmed from a fresh reset state)
        eng.load_prefill()
        g_noncp = eng.capture(warmup=cli.warmup)
        eng.load_prefill()
        g_cp = eng.capture(warmup=cli.warmup, step_fn=eng.step_cp)
        # 3. dispatched replay -> graph stream (drives greedy via slot[0] override)
        eng.load_prefill()
        stream, n_cp = [], 0
        for step in range(cli.steps):
            use_cp = (Lp + step) >= thresh
            (g_cp if use_cp else g_noncp).replay()
            torch.cuda.synchronize()
            n_cp += int(use_cp)
            t = argmax_tok()
            eng.slot[0] = t
            stream.append(t)
        st = eng.status_max()
        S_end = int(eng.slot[1].item())

        # 4. steady-state us/tok for EACH graph -- pure in-graph replay (no host work) = a REAL whole-step E2E,
        #    never a per-block sum (a block sum could under-estimate, which is forbidden).
        def bench(g):
            eng.load_prefill()
            for _ in range(10):
                g.replay()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            dist.barrier()
            e0.record()
            for _ in range(200):
                g.replay()
            e1.record()
            e1.synchronize()
            u = torch.tensor([e0.elapsed_time(e1) * 1e3 / 200], device=dev)
            dist.all_reduce(u, op=dist.ReduceOp.MAX)
            return u.item()

        us_cp, us_noncp = bench(g_cp), bench(g_noncp)
        if rank == 0:
            match = eager == stream
            miss = next((i for i in range(len(eager)) if eager[i] != stream[i]), -1)
            ok = match and st == 0 and S_end == Lp + cli.steps
            print(
                f"== nexrt DecodeEngine (TP=8, {len(eng.layers)}-layer, TWO-GRAPH CP dispatch, greedy): "
                f"L={Lp}, {cli.steps} steps, thresh={thresh} ==",
                flush=True,
            )
            print(f"   eager(dispatch): {eager}", flush=True)
            print(f"   graph(dispatch): {stream}", flush=True)
            print(
                f"   graph==eager: {match} (first mismatch {miss}) | S {Lp}->{S_end} | status {st} | "
                f"{n_cp}/{cli.steps} tokens used CP",
                flush=True,
            )
            print(
                f"   [{'OK' if ok else 'CHECK'}]  non-CP graph {us_noncp:.2f} us/tok | CP graph {us_cp:.2f} us/tok "
                f"(each a real whole-step E2E replay, not a block sum)",
                flush=True,
            )
        dist.barrier()
        dist.destroy_process_group()
        return

    if cli.graph:
        # The whole step() is captured as one CUDA graph and replayed per token (slot-driven).
        # Validation uses greedy argmax: the sampler's nucleus draw is fp-knife-edge
        # sensitive, so it can flip on the (fp-epsilon) difference between an eager launch and a graph replay
        # of the same reduce -- the argmax does not. The in-graph sampler still runs each replay (bumps
        # slot[1]=S+1); the host overwrites slot[0] with the argmax to drive the greedy trajectory.
        def argmax_tok():
            return int(eng.head.out_id[int(eng.head.out_val.argmax().item())].item())

        # 1. Eager greedy reference.
        Lp, first = eng.load_prefill()
        eager = []
        for _ in range(cli.steps):
            eng.step()
            torch.cuda.synchronize()
            t = argmax_tok()
            eng.slot[0] = t
            eager.append(t)
        # 2. warm up + capture (warmup from a fresh initial state; capture records the launches)
        eng.load_prefill()
        g = eng.capture(warmup=cli.warmup)
        # 3. reset, replay the graph, drive greedy via slot[0] override -> graph greedy stream
        eng.load_prefill()
        graph_stream = []
        for _ in range(cli.steps):
            g.replay()
            torch.cuda.synchronize()
            t = argmax_tok()
            eng.slot[0] = t
            graph_stream.append(t)
        st = eng.status_max()
        S_end = int(eng.slot[1].item())
        # 4. pure-replay latency (in-graph sampler, no host work per step)
        eng.load_prefill()
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        dist.barrier()
        e0.record()
        for _ in range(200):
            g.replay()
        e1.record()
        e1.synchronize()
        us = torch.tensor([e0.elapsed_time(e1) * 1e3 / 200], device=dev)
        dist.all_reduce(us, op=dist.ReduceOp.MAX)
        if rank == 0:
            match = eager == graph_stream
            miss = next((i for i in range(len(eager)) if eager[i] != graph_stream[i]), -1)
            ok = match and st == 0 and S_end == Lp + cli.steps
            print(
                f"== nexrt DecodeEngine (TP=8, {len(eng.layers)}-layer, ONE CUDA GRAPH, greedy): L={Lp}, {cli.steps} steps ==",
                flush=True,
            )
            print(f"   eager(greedy) : {eager}", flush=True)
            print(f"   graph(greedy) : {graph_stream}", flush=True)
            print(
                f"   graph==eager: {match} (first mismatch idx {miss}) | S {Lp}->{S_end} | status {st}",
                flush=True,
            )
            print(
                f"   [{'OK' if ok else 'CHECK'}]  one-graph decode: {us.item():.2f} us/tok (pure in-graph sampled replay)",
                flush=True,
            )
        dist.barrier()
        dist.destroy_process_group()
        return

    Lp, first = eng.load_prefill()

    toks = [first]
    finite = True
    for _s in range(cli.steps):
        eng.dbg = cli.debug and _s == 0
        eng.step()  # samples -> slot[0], bumps slot[1]
        torch.cuda.synchronize()
        if cli.greedy:
            # reducer out_val/out_id are the UNSORTED top candidates (out_id[0] is just rank 0's first one,
            # always in vocab shard [0,31040)); the argmax is the MAX-val entry -- same extraction head.py uses.
            nxt = int(eng.head.out_id[int(eng.head.out_val.argmax().item())].item())
            eng.slot[0] = nxt  # overwrite the sampled token for the next embed
        else:
            nxt = int(eng.slot[0].item())
        toks.append(nxt)
        finite = finite and bool(torch.isfinite(eng.x).all())
    st = eng.status_max()
    S_end = int(eng.slot[1].item())

    if rank == 0:
        ok = finite and st == 0 and S_end == Lp + cli.steps and all(0 <= t < 248320 for t in toks)
        mode = "greedy(argmax)" if cli.greedy else "sampled"
        print(
            f"== nexrt DecodeEngine (TP=8, {len(eng.layers)} layers, eager, {mode}): "
            f"L={Lp}, {cli.steps} steps ==",
            flush=True,
        )
        print(f"   first_token={first}", flush=True)
        print(f"   token stream: {toks}", flush=True)
        print(
            f"   S: {Lp} -> {S_end} (expect {Lp + cli.steps}) | x finite={finite} | status={st}",
            flush=True,
        )
        print(f"   [{'OK' if ok else 'CHECK'}]", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
