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

"""nexrt nex_n2_5_pro decode MoE block (TP=8): gate -> top-k -> L1 SwiGLU -> fused L2 + TP reduce.

The MoE is intermediate-parallel under TP=8: each rank owns 128 of the 1024 expert-intermediate
(column-parallel gate/up, row-parallel down), computes a [4096] partial, and the fused-L2 kernel sums the
8 partials via a graph-safe symm-mem barrier (same family as the head top-k). Routing (softmax over top-10
logits) is exactly the real gate's softmax-over-512-then-top10-then-renormalize.

Weight conversion (runtime, per rank, timed): slice the real fp8 experts into the kernel layout
(intermediate shard) and fold gamma=(1+post_attention_layernorm) column-wise into gate/W1/W3 as
fp8(W_fp8.float()*gamma) (scale untouched; W2 not folded). rms_scale=rsqrt(mean(x^2)+eps) is computed at
runtime in the Gate CTA0 (rms-in-gate path), so the kernels read RAW x.

The standalone entry point checks parity against a torch reference and measures a captured graph.
Run:  torchrun --standalone --nproc_per_node=8 moe.py --layer 0
"""

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
KDIR = HERE / "kernels" / "moe"
if __package__ in (None, ""):
    sys.path.insert(0, str(HERE.parent))

from nex_n2_5_pro.kernel_build import build_kernel  # noqa: E402
from nex_n2_5_pro.weights import load_layer_norms, load_moe_weights  # noqa: E402

DIM, N_EXP, TOPK, RINTER = 4096, 512, 10, 128  # RINTER = moe_intermediate(1024) / TP(8)
GATE_ROWS = 528
L1_ROWS = (TOPK + 1) * RINTER
FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


def _act_quant_dequant(x):
    rows, cols = x.shape
    blocks = x.float().reshape(rows, cols // 128, 128)
    scale = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-10) / FP8_MAX
    quantized = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return (quantized.float() * scale).reshape(rows, cols)


def _fp8_linear(x, weight, scale, quantize_activation=True):
    rows, cols = weight.shape
    dequantized = (
        weight.float() * scale.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :cols]
    )
    activation = _act_quant_dequant(x) if quantize_activation else x.float()
    return (activation @ dequantized.T).to(x.dtype)


@torch.inference_mode()
def _moe_reference(x, weights, quantize_activation=False):
    """Single-device MoE reference used by the standalone parity check."""
    token_count = x.shape[0]
    probabilities = torch.softmax(F.linear(x, weights["gate_w"]).float(), dim=-1)
    top_weights, top_experts = probabilities.topk(TOPK, dim=-1)
    top_weights = (top_weights / top_weights.sum(-1, keepdim=True)).to(x.dtype)
    shared_gate = torch.sigmoid(F.linear(x, weights["sgate_w"]))
    out = torch.zeros(token_count, DIM, device=x.device, dtype=torch.bfloat16)
    for token in range(token_count):
        activation = x[token : token + 1]
        for index in range(TOPK):
            expert = int(top_experts[token, index])
            gate = _fp8_linear(
                activation, weights["e_g1"][expert], weights["e_s1"][expert], quantize_activation
            )
            up = _fp8_linear(
                activation, weights["e_g3"][expert], weights["e_s3"][expert], quantize_activation
            )
            hidden = F.silu(gate) * up
            down = _fp8_linear(
                hidden, weights["e_g2"][expert], weights["e_s2"][expert], quantize_activation
            )
            out[token] += (down[0] * top_weights[token, index]).to(torch.bfloat16)
    shared_gate_proj = _fp8_linear(x, weights["s_g1"], weights["s_s1"], quantize_activation)
    shared_up_proj = _fp8_linear(x, weights["s_g3"], weights["s_s3"], quantize_activation)
    shared = _fp8_linear(
        F.silu(shared_gate_proj) * shared_up_proj,
        weights["s_g2"],
        weights["s_s2"],
        quantize_activation,
    )
    return out + (shared_gate * shared).to(torch.bfloat16)


def build(name):
    return build_kernel(KDIR, name)


def _p(t):
    return ctypes.c_void_p(t.data_ptr())


def _poff(t, b):
    return ctypes.c_void_p(t.data_ptr() + b)


def _sp():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def _fold_requant(w_fp8, scale, gamma):
    """Dequant -> fold gamma column-wise -> requant to fp8 with FRESH per-col-block scales (never overflows,
    unlike folding into the fp8 values directly). w_fp8 [E,I,H] (or [I,H]) fp8; scale [E,NB] (or [NB]) fp32
    col-block; gamma [H] fp32. Returns (fp8 same rank, scale [E,NB] or [NB])."""
    two = w_fp8.dim() == 2
    if two:
        w_fp8, scale = w_fp8.unsqueeze(0), scale.unsqueeze(0)
    E, inter_dim, H = w_fp8.shape
    NB = H // 128
    w = (
        w_fp8.float() * scale.repeat_interleave(128, 1).view(E, 1, H) * gamma.view(1, 1, H)
    )  # dequant * gamma
    wb = w.view(E, inter_dim, NB, 128)
    ns = (wb.abs().amax(dim=(1, 3)) / 448.0).clamp_min(1e-12)  # [E,NB] fresh block scales
    wq = (wb / ns.view(E, 1, NB, 1)).clamp(-448.0, 448.0).to(FP8).view(E, inter_dim, H)
    return (wq[0].contiguous(), ns[0].contiguous()) if two else (wq.contiguous(), ns.contiguous())


def convert_layer(W, gamma, rank, dev):
    """Raw checkpoint weights plus gamma[(4096)] fp32 -> this rank's kernel-layout tensors.
    Slice the 128-intermediate shard; fold gamma column-wise into gate (bf16) and W1/W3 (fp8 requant); W2
    is NOT folded (it consumes the post-SwiGLU intermediate)."""
    lo, hi = rank * RINTER, (rank + 1) * RINTER
    gate_w = torch.zeros(GATE_ROWS, DIM, dtype=torch.bfloat16, device=dev)
    gate_w[:N_EXP] = (W["gate_w"].float() * gamma.view(1, DIM)).to(torch.bfloat16)
    gate_w[N_EXP] = (W["sgate_w"][0].float() * gamma).to(torch.bfloat16)
    W1, s1 = _fold_requant(
        W["e_g1"][:, lo:hi, :], W["e_s1"][:, rank, :], gamma
    )  # [512,128,4096],[512,32]
    W3, s3 = _fold_requant(W["e_g3"][:, lo:hi, :], W["e_s3"][:, rank, :], gamma)
    sW1, ss1 = _fold_requant(W["s_g1"][lo:hi, :], W["s_s1"][rank, :], gamma)  # [128,4096],[32]
    sW3, ss3 = _fold_requant(W["s_g3"][lo:hi, :], W["s_s3"][rank, :], gamma)
    return {
        "gate_w": gate_w.contiguous(),
        "W1": W1,
        "W3": W3,
        "w1_w3_scales": torch.cat([s1, s3], 1).contiguous(),  # [512,64]
        "W2": W["e_g2"][:, :, lo:hi].contiguous(),  # [512,4096,128]
        "w2_scales": W["e_s2"][:, :, rank].contiguous(),  # [512,32]
        "shared_W1": sW1,
        "shared_W3": sW3,
        "shared_w1_w3_scales": torch.cat([ss1, ss3]).contiguous(),  # [64]
        "shared_W2": W["s_g2"][:, lo:hi].contiguous(),  # [4096,128]
        "shared_w2_scales": W["s_s2"][:, rank].contiguous(),  # [32]
    }


def convert_layer_raw(W, gamma, rank, dev):
    """RAW checkpoint/sglang layout (the weight-SHARING path): NO gamma fold, no requant. w13
    [512,256,4096] stacks gate|up rows per expert == sglang FusedMoE w13_weight; scales are the
    ORIGINAL checkpoint block scales in the fused [E,2,32]->[E,64] view; W2 is byte-identical to
    the folded path (never folded). Only the router gate stays folded+padded (nexrt-private);
    the L1 kernel instead reads xn = x*rms*gamma written by the top-k Xn kernel."""
    lo, hi = rank * RINTER, (rank + 1) * RINTER
    gate_w = torch.zeros(GATE_ROWS, DIM, dtype=torch.bfloat16, device=dev)
    gate_w[:N_EXP] = (W["gate_w"].float() * gamma.view(1, DIM)).to(torch.bfloat16)
    gate_w[N_EXP] = (W["sgate_w"][0].float() * gamma).to(torch.bfloat16)
    return {
        "gate_w": gate_w.contiguous(),
        "w13": torch.cat([W["e_g1"][:, lo:hi, :], W["e_g3"][:, lo:hi, :]], 1).contiguous(),
        "w13_scales": torch.cat([W["e_s1"][:, rank, :], W["e_s3"][:, rank, :]], 1).contiguous(),
        "shared_gate_up": torch.cat([W["s_g1"][lo:hi, :], W["s_g3"][lo:hi, :]], 0).contiguous(),
        "shared_scales": torch.cat([W["s_s1"][rank, :], W["s_s3"][rank, :]]).contiguous(),
        "W2": W["e_g2"][:, :, lo:hi].contiguous(),  # [512,4096,128]
        "w2_scales": W["e_s2"][:, :, rank].contiguous(),  # [512,32]
        "shared_W2": W["s_g2"][:, lo:hi].contiguous(),  # [4096,128]
        "shared_w2_scales": W["s_s2"][:, rank].contiguous(),  # [32]
    }


def convert_layer_ext(ext, dev):
    """Bridge mode: build self.k directly from sglang's LOADED per-rank tensors (see
    sglang_prefill/ext_weights.SglangExt). Every fp8 tensor is a zero-copy ALIAS of the sglang
    param (bit-identical to the checkpoint slice, byte-verified); only the nexrt-private router
    gate is materialized (folded + padded). No safetensors I/O at all. Returns (k, gamma)."""
    gamma = 1.0 + ext["post_ln"].float()
    gate_w = torch.zeros(GATE_ROWS, DIM, dtype=torch.bfloat16, device=dev)
    gate_w[:N_EXP] = (ext["gate_w"].float() * gamma.view(1, DIM)).to(torch.bfloat16)
    gate_w[N_EXP] = (ext["sgate_w"][0].float() * gamma).to(torch.bfloat16)
    return {
        "gate_w": gate_w.contiguous(),
        "w13": ext["w13"],
        "w13_scales": ext["w13_scales"].view(N_EXP, 2 * 32),
        "shared_gate_up": ext["shared_gate_up"],
        "shared_scales": ext["shared_scales"].view(64),
        "W2": ext["w2"],
        "w2_scales": ext["w2_scales"].view(N_EXP, 32),
        "shared_W2": ext["shared_down"],
        "shared_w2_scales": ext["shared_down_scales"].view(32),
    }, gamma


class Moe:
    def __init__(self, layer, rank, world, dev, keep_ref=True, share_from=None, ext=None):
        self.rank, self.world, self.dev = rank, world, dev
        self.ext = ext
        # RAW mode = the weight-SHARING path (sglang/checkpoint layout, no fold): the top-k kernel
        # also writes xn = x*rms*gamma, and L1 reads xn + stacked w13 + original scales. ext (bridge)
        # implies raw -- the tensors ARE sglang's live params.
        self.raw = ext is not None or os.environ.get("NEXRT_MOE_RAW") == "1"
        vp, cf, ci, cu, cll = (
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_ulonglong,
            ctypes.c_longlong,
        )
        self.gl = ctypes.CDLL(str(build("nex_n2_5_pro_gate_gemv_ctypes")))
        self.gl.NexN25ProGateGemvLaunch.argtypes = [vp, vp, vp, vp, cf, vp, cll, vp]
        self.gl.NexN25ProGateTopKLaunch.argtypes = [vp, vp, vp, vp]
        self.gl.NexN25ProGateTopKXnLaunch.argtypes = [vp] * 7
        self.l1 = ctypes.CDLL(str(build("nex_n2_5_pro_l1_pair_ctypes")))
        self.l1.NexN25ProL1Launch.argtypes = [vp] * 11
        self.l1.NexN25ProL1W13XnLaunch.argtypes = [vp] * 8
        self.fl = ctypes.CDLL(str(build("nex_n2_5_pro_fused_l2_reduce_ctypes")))
        self.fl.NexN25ProFusedL2ReduceLaunch.argtypes = (
            [vp] * 7 + [cu, vp, cu] + [vp] * 4 + [ci, ci, vp]
        )
        self.fl.NexN25ProFusedL2ReduceResidLaunch.argtypes = (
            [vp] * 7 + [cu, vp, cu] + [vp] * 4 + [ci, ci, vp, vp, cll, vp]
        )
        self.fl.NexN25ProFusedL2ReduceErrorString.argtypes = [ci]
        self.fl.NexN25ProFusedL2ReduceErrorString.restype = ctypes.c_char_p
        # Wait-shadow prefetch of the NEXT layer's front GEMV weights, issued by the L2
        # kernel while its publish drains and the peer wait spins. The decode engine wires
        # (ptr, bytes) per captured graph -- clamped to what the wait window absorbs;
        # (0, 0) (standalone blocks, last layer) disables it with no other effect.
        self.front_pf = (0, 0)

        # weights (timed runtime conversion; ext mode aliases sglang's live tensors instead of
        # reloading them from the checkpoint)
        t0 = time.time()
        if ext is not None:
            self.Wr = None  # no raw reload; the ref is driven externally
            self.k, gamma = convert_layer_ext(ext, dev)
            self.post_ln = ext["post_ln"].float()
        else:
            _, post_ln = load_layer_norms(layer, dev)
            post_ln = post_ln.float()
            gamma = 1.0 + post_ln
            self.Wr = load_moe_weights(layer, dev)  # real fp8 experts (raw; for the torch ref)
            self.k = (convert_layer_raw if self.raw else convert_layer)(self.Wr, gamma, rank, dev)
            self.post_ln = post_ln
        torch.cuda.synchronize()
        self.convert_s = time.time() - t0
        if not keep_ref:  # decode engine (60 layers): drop the 6.6GB raw
            self.Wr = None  # experts — the converted self.k (0.81GB) is all
            torch.cuda.empty_cache()  # the kernels read. Keeping Wr OOMs at 60 layers.

        # scratch + symm buffers (fixed, pre-allocated)
        self.gate_logits = torch.empty(GATE_ROWS, dtype=torch.bfloat16, device=dev)
        self.packed = torch.empty(64, dtype=torch.int32, device=dev)
        self.rms_scale = torch.ones(1, dtype=torch.float32, device=dev)
        self.raw_h = torch.empty(TOPK + 1, RINTER, dtype=torch.float32, device=dev)
        if self.raw:
            self.gamma = gamma.contiguous()  # fp32 [DIM], read by the Xn top-k kernel
            self.xn = torch.empty(
                DIM, dtype=torch.bfloat16, device=dev
            )  # x*rms*gamma -> raw L1 input
        self.out = torch.empty(DIM, dtype=torch.float32, device=dev)
        self.status = torch.zeros(1, dtype=torch.int32, device=dev)
        # Symmetric-memory push-reduce workspace. Each rendezvous consumes a limited NVLS multicast
        # object, so at 60 layers per-block workspaces would exhaust the pool; the blocks run
        # SEQUENTIALLY (barrier-synced), so share_from lets all MoE blocks reuse ONE workspace.
        # rep = the replicated push buffer (every GPU holds all 8 ranks' partial stripes, written
        # through the multicast VA); bar = one NVLS arrival counter slot per CTA; it_ctr = per-CTA
        # epochs (monotonic device counters -- parity and spin targets derive from them, so graph
        # replays and layer reuse need no host resets).
        if share_from is None:
            self.rep, _, self.rep_mc = self._symm(8 * 2 * DIM, torch.float32)
            self.rep.zero_()
            self.bar, _, self.bar_mc = self._symm(128, torch.int32)  # one counter slot per CTA
            self.bar.zero_()
            if self.bar_mc == 0 or self.rep_mc == 0:
                raise RuntimeError("multicast unavailable (NVLS required)")
            self.it_ctr = torch.zeros(128, dtype=torch.int32, device=dev)
        else:
            s = share_from
            self.rep, self.rep_mc = s.rep, s.rep_mc
            self.bar, self.bar_mc = s.bar, s.bar_mc
            self.it_ctr = s.it_ctr
        torch.cuda.synchronize()
        dist.barrier()

    def _symm(self, numel, dtype):
        x = symm_mem.empty(numel, dtype=dtype, device=self.dev)
        h = symm_mem.rendezvous(x, dist.group.WORLD.group_name)
        peers = torch.tensor(list(h.buffer_ptrs), dtype=torch.int64, device=self.dev)
        return x, peers, int(getattr(h, "multicast_ptr", 0) or 0)

    def forward(self, x, resid=None):
        """x [DIM] bf16 (RAW residual-stream input; rms computed in Gate CTA0).  resid=None -> writes self.out
        [DIM] fp32.  resid=r_bf16 -> the L2 reduce FOLDS the residual add into its epilogue (r += reduced, in
        place, bf16), skipping the separate add kernel."""
        cf, cu, cll = ctypes.c_float, ctypes.c_ulonglong, ctypes.c_longlong
        if self.raw:  # weight-sharing path (bridge / NEXRT_MOE_RAW)
            # The drained gate CTAs L2-warm the shared expert's stacked gate|up L1 weights --
            # the only L1 payload whose address does not depend on the top-k selection.
            sp = self.k["shared_gate_up"]
            self.gl.NexN25ProGateGemvLaunch(
                _p(x),
                _p(self.k["gate_w"]),
                _p(self.gate_logits),
                _p(self.rms_scale),
                cf(1e-6),
                _p(sp),
                cll(sp.numel() * sp.element_size()),
                _sp(),
            )
            self.gl.NexN25ProGateTopKXnLaunch(
                _p(self.gate_logits),
                _p(self.rms_scale),
                _p(x),
                _p(self.gamma),
                _p(self.packed),
                _p(self.xn),
                _sp(),
            )
            self.l1.NexN25ProL1W13XnLaunch(
                _p(self.packed),
                _p(self.xn),
                _p(self.k["w13"]),
                _p(self.k["w13_scales"]),
                _p(self.k["shared_gate_up"]),
                _p(self.k["shared_scales"]),
                _p(self.raw_h),
                _sp(),
            )
        else:
            self.gl.NexN25ProGateGemvLaunch(
                _p(x),
                _p(self.k["gate_w"]),
                _p(self.gate_logits),
                _p(self.rms_scale),
                cf(1e-6),
                None,
                cll(0),
                _sp(),
            )
            self.gl.NexN25ProGateTopKLaunch(
                _p(self.gate_logits), _p(self.rms_scale), _p(self.packed), _sp()
            )
            self.l1.NexN25ProL1Launch(
                _p(self.packed),
                _p(x),
                _p(self.rms_scale),
                _p(self.k["W1"]),
                _p(self.k["W3"]),
                _p(self.k["w1_w3_scales"]),
                _p(self.k["shared_W1"]),
                _p(self.k["shared_W3"]),
                _p(self.k["shared_w1_w3_scales"]),
                _p(self.raw_h),
                _sp(),
            )
        if resid is None:
            self.fl.NexN25ProFusedL2ReduceLaunch(
                _p(self.packed),
                _poff(self.packed, TOPK * 4),
                _p(self.raw_h),
                _p(self.k["W2"]),
                _p(self.k["w2_scales"]),
                _p(self.k["shared_W2"]),
                _p(self.k["shared_w2_scales"]),
                cu(self.rep_mc),
                _p(self.rep),
                cu(self.bar_mc),
                _p(self.bar),
                _p(self.out),
                _p(self.status),
                _p(self.it_ctr),
                self.rank,
                self.world,
                _sp(),
            )
            return self.out
        pf_ptr, pf_bytes = self.front_pf
        self.fl.NexN25ProFusedL2ReduceResidLaunch(
            _p(self.packed),
            _poff(self.packed, TOPK * 4),
            _p(self.raw_h),
            _p(self.k["W2"]),
            _p(self.k["w2_scales"]),
            _p(self.k["shared_W2"]),
            _p(self.k["shared_w2_scales"]),
            cu(self.rep_mc),
            _p(self.rep),
            cu(self.bar_mc),
            _p(self.bar),
            _p(self.out),
            _p(self.status),
            _p(self.it_ctr),
            self.rank,
            self.world,
            _p(resid),
            ctypes.c_void_p(pf_ptr) if pf_ptr else None,
            cll(pf_bytes),
            _sp(),
        )
        return resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument(
        "--raw",
        action="store_true",
        help="test the RAW weight-sharing path (xn top-k + stacked-w13 L1) the bridge uses",
    )
    cli = ap.parse_args()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl")
    assert world == 8

    if cli.raw:
        os.environ["NEXRT_MOE_RAW"] = "1"
    moe = Moe(cli.layer, rank, world, dev)
    gen = torch.Generator(device=dev).manual_seed(7)  # same x on all ranks (replicated)
    x_raw = torch.randn(DIM, dtype=torch.bfloat16, device=dev, generator=gen)
    # the kernel reads RAW x and applies rms in-gate; the torch ref must see the SAME rms-normed input.
    xf = x_raw.float()
    inv_rms = torch.rsqrt(xf.pow(2).mean() + 1e-6)
    x_normed = (xf * inv_rms * (1.0 + moe.post_ln)).to(
        torch.bfloat16
    )  # (1+w) RMSNorm == what the kernel does

    # ---- torch reference: full MoE (non-TP) on the normed x ----
    # The kernels use BF16 activation (fp8 weights only) -- the standalone bench's own reference does too.
    # Use bf16 activations so the reference matches the kernel and isolates GEMV/routing/reduction behavior.
    ref = _moe_reference(x_normed.view(1, DIM), moe.Wr)[0].float()

    moe.forward(x_raw)
    torch.cuda.synchronize()
    dist.barrier()
    got = moe.out.clone()
    rel = ((got - ref).norm() / ref.norm().clamp_min(1e-9)).item()  # vs ORIGINAL weights (bf16-act)

    # converted-weights reference: SAME requant weights + bf16 activation the kernels use, matching the
    # kernel's rms-on-dot-output math -> isolates the kernel's arithmetic from the requant weight difference.
    def ref_conv(k, x_raw):
        xf = x_raw.float()
        rms = torch.rsqrt(xf.pow(2).mean() + 1e-6)
        pr = torch.softmax(rms * (k["gate_w"][:N_EXP].float() @ xf), -1)
        tw, ti = pr.topk(TOPK)
        tw = tw / tw.sum()
        gsh = torch.sigmoid(rms * (k["gate_w"][N_EXP].float() @ xf))
        part = torch.zeros(DIM, device=dev)
        for j in range(TOPK):
            e = int(ti[j])
            w1 = k["W1"][e].float() * k["w1_w3_scales"][e, :32].repeat_interleave(128).view(1, DIM)
            w3 = k["W3"][e].float() * k["w1_w3_scales"][e, 32:].repeat_interleave(128).view(1, DIM)
            w2 = k["W2"][e].float() * k["w2_scales"][e].repeat_interleave(128).view(DIM, 1)
            part += (w2 @ (torch.nn.functional.silu(rms * (w1 @ xf)) * (rms * (w3 @ xf)))) * tw[j]
        s1 = k["shared_W1"].float() * k["shared_w1_w3_scales"][:32].repeat_interleave(128).view(
            1, DIM
        )
        s3 = k["shared_W3"].float() * k["shared_w1_w3_scales"][32:].repeat_interleave(128).view(
            1, DIM
        )
        s2 = k["shared_W2"].float() * k["shared_w2_scales"].repeat_interleave(128).view(DIM, 1)
        part += s2 @ (torch.nn.functional.silu(rms * (s1 @ xf)) * (rms * (s3 @ xf)) * gsh)
        dist.all_reduce(part, op=dist.ReduceOp.SUM)  # sum the 8 intermediate-shard partials
        return part

    def ref_raw(k, x_raw, gamma):
        """RAW path's exact intended math: bf16 xn = x*rms*gamma, ORIGINAL weights/scales, no per-dot rms."""
        xf = x_raw.float()
        rms = torch.rsqrt(xf.pow(2).mean() + 1e-6)
        xn = (xf * rms * gamma).to(torch.bfloat16).float()
        pr = torch.softmax(rms * (k["gate_w"][:N_EXP].float() @ xf), -1)
        tw, ti = pr.topk(TOPK)
        tw = tw / tw.sum()
        gsh = torch.sigmoid(rms * (k["gate_w"][N_EXP].float() @ xf))
        part = torch.zeros(DIM, device=dev)
        for j in range(TOPK):
            e = int(ti[j])
            w1 = k["w13"][e, :RINTER].float() * k["w13_scales"][e, :32].repeat_interleave(128).view(
                1, DIM
            )
            w3 = k["w13"][e, RINTER:].float() * k["w13_scales"][e, 32:].repeat_interleave(128).view(
                1, DIM
            )
            w2 = k["W2"][e].float() * k["w2_scales"][e].repeat_interleave(128).view(DIM, 1)
            part += (w2 @ (F.silu(w1 @ xn) * (w3 @ xn))) * tw[j]
        s1 = k["shared_gate_up"][:RINTER].float() * k["shared_scales"][:32].repeat_interleave(
            128
        ).view(1, DIM)
        s3 = k["shared_gate_up"][RINTER:].float() * k["shared_scales"][32:].repeat_interleave(
            128
        ).view(1, DIM)
        s2 = k["shared_W2"].float() * k["shared_w2_scales"].repeat_interleave(128).view(DIM, 1)
        part += s2 @ (F.silu(s1 @ xn) * (s3 @ xn) * gsh)
        dist.all_reduce(part, op=dist.ReduceOp.SUM)
        return part

    refc = ref_raw(moe.k, x_raw, moe.gamma) if moe.raw else ref_conv(moe.k, x_raw)
    relc = (
        (got - refc).norm() / refc.norm().clamp_min(1e-9)
    ).item()  # vs CONVERTED/RAW weights (kernel arithmetic)
    st = torch.tensor([int(moe.status.item())], device=dev)
    dist.all_reduce(st, op=dist.ReduceOp.MAX)

    # ---- perf: captured-graph latency via the local benchmark helper (hot back-to-back, TP-max) ----
    from nex_n2_5_pro.bench import bench_graph  # vendored profiler

    us = (
        bench_graph(
            lambda: moe.forward(x_raw),
            warmup=30,
            rep=2000,
            flush_l2=False,
            tp_reduce=True,
            device=str(dev),
        ).median_ms
        * 1e3
    )
    cs = torch.tensor([moe.convert_s], device=dev)
    dist.all_reduce(cs, op=dist.ReduceOp.MAX)

    if rank == 0:
        ok = relc < 5e-3 and st.item() == 0
        mode = "RAW w13-xn (sglang-shared layout)" if moe.raw else "split-K=4"
        print(f"== nexrt MoE (TP=8, layer {cli.layer}, bf16 activation) ==", flush=True)
        print(
            f"   vs ORIGINAL-weights ref  : relL2 {rel:.3e}   (= fp8 requant-fold cost; collapses when baseline uses converted weights)",
            flush=True,
        )
        print(
            f"   vs {'RAW' if moe.raw else 'CONVERTED'}-weights ref : relL2 {relc:.3e}   (= kernel arithmetic only; proves numerics correct)",
            flush=True,
        )
        print(
            f"   [{'OK' if ok else 'CHECK'}]  status={st.item()}  latency(graph) {us:.2f} us/tok  "
            f"| L1 {mode} | weight-convert {cs.item():.2f} s/rank",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
