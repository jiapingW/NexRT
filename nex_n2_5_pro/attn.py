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

"""nexrt nex_n2_5_pro decode FULL-ATTENTION mixer block (TP=8), input-RMSNorm FUSED into the front.

Pipeline per rank (4 q heads / 1 kv head after TP=8):
  qkv_proj (fp8 GEMV, folded (1+input_ln)) -> q_raw[1024] gate_raw[1024] k_cache[pos] v_cache[pos]
  norm_rope_fused  -> q/k head-RMSNorm + partial-RoPE (s cancels), CTA-5 computes s and scales v & gate
  non-CP true-4-head WGMMA megakernel -> ctx[4,256]
  o_proj + TP reduce (fused gate: ctx * sigmoid(gate)) -> reduced out[4096]

Input RMSNorm handling:
  `(1+input_layernorm.weight)` is FOLDED (per in-channel) into q/k/v proj weights (fp8 requant, fresh
  block scales). The scalar `s = rsqrt(mean(x^2,HID)+eps_in)` CANCELS in q/k (eps-level: exact only for
  eps=0, negligible at eps=1e-6) because their head-dim RMSNorm (q_norm/k_norm) re-normalizes -- no
  nonlinearity sits between proj and norm. v has no norm and
  gate feeds a sigmoid, so BOTH need s; norm_rope's CTA-5 recomputes s from the raw residual x and scales
  v (in the cache) and gate in place. o_proj weights are NOT folded (row-parallel on the attn output).

Two references:
  reference()        = TRUE full-attn: input_layernorm(x) then fp8-activation attn_decode with ORIGINAL
                       weights (== HF); the "vs true" number folds in requant + fp8-activation cost.
  reference_folded() = the kernel's exact math (bf16 activation, FOLDED weights, s on v&gate only) --
                       isolates kernel arithmetic (should be ~1e-3 vs the kernel).

Run:  torchrun --standalone --nproc_per_node=8 attn.py --layer 3 --past 4096
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
KDIR = HERE / "kernels" / "full_attn"
if __package__ in (None, ""):
    sys.path.insert(0, str(HERE.parent))

from nex_n2_5_pro.kernel_build import build_kernel  # noqa: E402
from nex_n2_5_pro.weights import (  # noqa: E402
    load_attention_weights,
    load_layer_norms,
    shard_attention_weights,
)

HID, HEAD_DIM, N_HEADS, N_KV = 4096, 256, 32, 2
NHL, NKVL, GQA_LOCAL = 4, 1, 4  # per TP=8 rank: 4 q heads, 1 kv head, GQA 4:1
Q_GATE = 2
QROWS = NHL * HEAD_DIM * Q_GATE  # 2048  (query|gate interleaved per head)
KROWS = NKVL * HEAD_DIM  # 256
VROWS = NKVL * HEAD_DIM  # 256
TOTAL = QROWS + KROWS + VROWS  # 2560
LOCAL = NHL * HEAD_DIM  # 1024  (attn/gate width, o_proj input)
CACHE_LEN = (
    256 * 1024 + 1024
)  # pre-allocated KV-cache positions per full-attn layer: 256K ctx + 1K decode headroom
# (must match KV_CAPACITY in kernels/full_attn/nex_n2_5_pro_attn_fd4_wgmma_ctypes.cu;
#  the CP kernels take the capacity as a runtime arg)
EPS = 1e-6  # rms_norm_eps (input_ln and q/k norm)
ROPE_THETA, ROTARY_DIM = 10000000.0, 64
SCALING = HEAD_DIM**-0.5  # 1/16
NCTAS = 128
FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


def _act_quant_dequant(x):
    """Dynamic per-row, per-128-element FP8 activation quantization used by the parity reference."""
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


def _rms_norm(x, weight):
    xf = x.float()
    normalized = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return (normalized * (1.0 + weight.float())).to(x.dtype)


def _partial_rope(x, position):
    half = ROTARY_DIM // 2
    inv_freq = 1.0 / (
        ROPE_THETA ** (torch.arange(0, ROTARY_DIM, 2, device=x.device).float() / ROTARY_DIM)
    )
    angle = position * inv_freq
    cos, sin = torch.cos(angle).repeat(2), torch.sin(angle).repeat(2)
    rotary = x[..., :ROTARY_DIM].float()
    rotated = torch.cat([-rotary[..., half:], rotary[..., :half]], dim=-1)
    out = rotary * cos + rotated * sin
    return torch.cat([out.to(x.dtype), x[..., ROTARY_DIM:]], dim=-1)


def _attention_decode_local(x, weights, past_k, past_v, position, quantize_activation):
    """One TP rank's attention reference, including its row-parallel output projection."""
    qg = _fp8_linear(x, weights["q"], weights["qs"], quantize_activation).view(
        1, NHL, HEAD_DIM * Q_GATE
    )
    query, gate = qg.chunk(2, dim=-1)
    gate = gate.reshape(1, LOCAL)
    key = _fp8_linear(x, weights["k"], weights["ks"], quantize_activation).view(1, NKVL, HEAD_DIM)
    value = _fp8_linear(x, weights["v"], weights["vs"], quantize_activation).view(1, NKVL, HEAD_DIM)
    query = _partial_rope(_rms_norm(query, weights["qn"]), position)
    key = _partial_rope(_rms_norm(key, weights["kn"]), position)
    keys = torch.cat([past_k, key[0]], dim=0)
    values = torch.cat([past_v, value[0]], dim=0)
    scores = torch.einsum("hd,sd->hs", query[0].float(), keys.float()) * SCALING
    context = torch.einsum("hs,sd->hd", torch.softmax(scores, dim=-1), values.float())
    gated = context.reshape(1, LOCAL).to(x.dtype) * torch.sigmoid(gate)
    return _fp8_linear(gated, weights["o"], weights["os"], quantize_activation)


def build(name):
    return build_kernel(KDIR, name)


def build_cp(name):
    """Like build() but with EXPLICIT sm_90a gencode. The CP attn kernels use Hopper wgmma; on nvcc 12.9
    `-arch=sm_90a` mis-targets ptxas to `sm_90` (wgmma unsupported), while
    `-gencode arch=compute_90a,code=sm_90a` compiles it. (Non-wgmma kernels are unaffected either way.)"""
    return build_kernel(KDIR, name, hopper=True, link_cuda=True)


def _p(t):
    return ctypes.c_void_p(t.data_ptr())


def _sp():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def _fold_requant_2d(w_fp8, scale, gamma):
    """Fold gamma column-wise into an [R,C] fp8 weight (block-128 scale [R/128,C/128]) via
    dequant->fold->requant with fresh per-block scales (no e4m3 overflow). gamma [C] fp32."""
    R, C = w_fp8.shape
    RB, CB = R // 128, C // 128
    w = w_fp8.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1) * gamma.view(1, C)
    wb = w.view(RB, 128, CB, 128)
    ns = (wb.abs().amax(dim=(1, 3)) / 448.0).clamp_min(1e-12)  # [RB,CB]
    wq = (wb / ns.view(RB, 1, CB, 1)).clamp(-448.0, 448.0).to(FP8).view(R, C)
    return wq.contiguous(), ns.contiguous()


def rope_tables(dev):
    """cos/sin tables [CACHE_LEN,32] fp32 for partial RoPE (neox), indexed by position."""
    inv = 1.0 / (
        ROPE_THETA ** (torch.arange(0, ROTARY_DIM, 2, device=dev).float() / ROTARY_DIM)
    )  # [32]
    ang = torch.arange(CACHE_LEN, device=dev).float()[:, None] * inv[None, :]  # [16K,32]
    return torch.cos(ang).contiguous(), torch.sin(ang).contiguous()


def convert_weights_ext(ext):
    """Bridge mode: derive the folded front from sglang's loaded per-rank tensors without
    rereading checkpoint weights. sglang's qkv_proj [2560,4096] has EXACTLY nexrt's row layout
    (q|gate x4 heads, k, v -- byte-verified vs the checkpoint slice), so the fold is the same
    recipe on the same bytes. o_proj / q_norm / k_norm are bit-identical -> zero-copy aliases."""
    iln = ext["iln"].float()
    qkv_wf, qkv_sf = _fold_requant_2d(ext["qkv_w"], ext["qkv_scales"].float(), 1.0 + iln)
    kernel_w = {
        "qkv_w": qkv_wf,
        "qkv_scales": qkv_sf,
        "qn": ext["qn"].to(torch.bfloat16),
        "kn": ext["kn"].to(torch.bfloat16),
        "o_w": ext["o_w"],
        "o_scales": ext["o_scales"].float(),
    }
    return kernel_w, iln


def convert_weights(layer, rank, dev):
    """-> (kernel_w with (1+input_ln) folded into q/k/v, orig shard Wr for the true ref, input_ln_w)."""
    Wr = shard_attention_weights(
        load_attention_weights(layer, dev), rank, tp=8
    )  # q[2048,4096] k/v[256,4096]
    iln, _ = load_layer_norms(layer, dev)
    iln = iln.float()  # [4096]
    gamma = 1.0 + iln
    qkv_w = torch.cat([Wr["q"], Wr["k"], Wr["v"]], 0).contiguous()  # [2560,4096] fp8
    qkv_s = torch.cat([Wr["qs"], Wr["ks"], Wr["vs"]], 0).contiguous().float()  # [20,32]
    qkv_wf, qkv_sf = _fold_requant_2d(qkv_w, qkv_s, gamma)
    kernel_w = {
        "qkv_w": qkv_wf,
        "qkv_scales": qkv_sf,
        "qn": Wr["qn"].contiguous(),
        "kn": Wr["kn"].contiguous(),
        "o_w": Wr["o"].contiguous(),
        "o_scales": Wr["os"].contiguous().float(),
    }
    return kernel_w, Wr, iln


def convert_weights_cp(layer, rank, dev):
    """CP front-end weights: qgate W[5632,4096] = 16 group Q-heads + 4 rank gate-heads + 1 K + 1 V, with
    (1+input_ln) folded (fresh block-fp8 requant, same recipe as convert_weights). o_proj/qn/kn are the
    same shards as the non-CP path. group g=rank//4 owns q-heads [g*16,g*16+16) (duplicate-Q: every rank
    in the group computes all 16); the rank owns gate heads [rank*4, rank*4+4)."""
    W = load_attention_weights(layer, dev)  # q[16384,4096] k/v[512,4096] fp8 + scales
    Wr = shard_attention_weights(W, rank, tp=8)  # this rank's o_proj + qn/kn shards
    iln, _ = load_layer_norms(layer, dev)
    iln = iln.float()
    gamma = (1.0 + iln).view(1, HID)  # [1,4096] folded per in-channel

    def deq(w, s):
        return w.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)

    q_f = deq(W["q"], W["qs"])  # [16384,4096] (32 heads x (q256|gate256))
    k_f, v_f = deq(W["k"], W["ks"]), deq(W["v"], W["vs"])  # [512,4096] each (2 kv heads)
    g, HD, GH = rank // 4, HEAD_DIM, HEAD_DIM * Q_GATE  # group; 256; 512 (query|gate per head)
    q16 = torch.cat(
        [q_f[h * GH : h * GH + HD] for h in range(g * 16, g * 16 + 16)], 0
    )  # [4096,4096]
    gate4 = torch.cat(
        [q_f[h * GH + HD : h * GH + GH] for h in range(rank * 4, rank * 4 + 4)], 0
    )  # [1024,4096]
    k1, v1 = k_f[g * HD : g * HD + HD], v_f[g * HD : g * HD + HD]  # [256,4096] each
    w_raw = torch.cat([q16, gate4, k1, v1], 0) * gamma  # [5632,4096] folded
    R, C = w_raw.shape
    wb = w_raw.view(R // 128, 128, C // 128, 128)
    ns = (wb.abs().amax(dim=(1, 3)) / 448.0).clamp_min(1e-12)  # [44,32] fresh block scales
    wq = (wb / ns.view(R // 128, 1, C // 128, 1)).clamp(-448.0, 448.0).to(FP8).view(R, C)
    cp_w = {
        "qgate_w": wq.contiguous(),
        "qgate_scales": ns.contiguous().float(),
        "qn": Wr["qn"].contiguous(),
        "kn": Wr["kn"].contiguous(),
        "o_w": Wr["o"].contiguous(),
        "o_scales": Wr["os"].contiguous().float(),
    }
    return cp_w, Wr, iln


class FullAttn:
    def __init__(self, layer, rank, world, dev, keep_ref=True, share_from=None, ext=None):
        self.rank, self.world, self.dev = rank, world, dev
        self.ext = ext
        self.nctas = NCTAS
        vp, ci, cf, cu = ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_ulonglong
        self.ql = ctypes.CDLL(str(build("nex_n2_5_pro_qkv_proj_ctypes")))
        self.ql.NexN25ProQkvProjLaunch.argtypes = [vp] * 7 + [ci, vp]
        self.ql.NexN25ProQkvProjDynLaunch.argtypes = [vp] * 9  # ...,seqlen,stream
        self.nrl = ctypes.CDLL(str(build("nex_n2_5_pro_norm_rope_ctypes")))
        self.nrl.NexN25ProNormRopeLaunch.argtypes = [vp] * 7 + [ci] + [vp] * 5 + [ci, cf, vp]
        self.nrl.NexN25ProNormRopeDynLaunch.argtypes = [vp] * 12 + [
            ci,
            cf,
            vp,
        ]  # slot @ arg8, v_cache @ arg10
        self.wgml = ctypes.CDLL(str(build_cp("nex_n2_5_pro_attn_fd4_wgmma_ctypes")))
        self.wgml.NexN25ProAttnWgmmaLaunch.argtypes = [vp] * 3 + [ci] + [vp] * 6
        self.wgml.NexN25ProAttnWgmmaLaunch.restype = ci
        self.wgml.NexN25ProAttnWgmmaDynLaunch.argtypes = [vp] * 10
        self.wgml.NexN25ProAttnWgmmaDynLaunch.restype = ci
        self.opl = ctypes.CDLL(str(build("nex_n2_5_pro_o_proj_reduce_ctypes")))
        # Non-CP graph: ONE fused launch (gated o_proj + deterministic push reduce). The CP graph
        # keeps the split form -- its producer is the CP tail kernel -- via the WholeReduce pair.
        self.opl.NexN25ProOProjReduceLaunch.argtypes = (
            [vp] * 4 + [cu, vp, cu] + [vp] * 4 + [ci, ci, vp, vp, ctypes.c_longlong, vp]
        )
        self.opl.NexN25ProOProjWholeReduceLaunch.argtypes = [vp, cu] + [vp] * 4 + [ci, vp]
        self.opl.NexN25ProOProjWholeReduceResidLaunch.argtypes = [vp, cu] + [vp] * 4 + [ci, vp, vp]
        # Wait-shadow prefetch of the downstream MoE gate weights, issued by the fused o_proj
        # kernel during its peer wait; the decode engine wires (ptr, bytes). (0, 0) disables.
        self.wait_pf = (0, 0)
        if ext is not None:  # bridge: alias sglang's live tensors
            self.w, self.iln = convert_weights_ext(ext)
            self.Wr = None
        else:
            self.w, self.Wr, self.iln = convert_weights(layer, rank, dev)
        if not keep_ref:  # decode engine: drop raw fp8 (ref-only)
            self.Wr = None
        self.q_raw = torch.empty(
            LOCAL, dtype=torch.bfloat16, device=dev
        )  # qkv query out -> norm_rope q_raw
        self.gate = torch.empty(
            LOCAL, dtype=torch.bfloat16, device=dev
        )  # qkv gate out; scaled in place
        self.k_cache = torch.zeros(CACHE_LEN, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        self.v_cache = torch.zeros(CACHE_LEN, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        self.q_out = torch.empty(
            NHL, HEAD_DIM, dtype=torch.bfloat16, device=dev
        )  # normed+roped q -> attn core
        self.ctx = torch.empty(
            NHL, HEAD_DIM, dtype=torch.bfloat16, device=dev
        )  # attn ctx -> o_proj
        self.ws_m = torch.empty(self.nctas * NHL, dtype=torch.float32, device=dev)
        self.ws_l = torch.empty(self.nctas * NHL, dtype=torch.float32, device=dev)
        self.ws_acc = torch.empty(self.nctas * NHL * HEAD_DIM, dtype=torch.bfloat16, device=dev)
        self.wgmma_grid_counter = torch.zeros(1, dtype=torch.int32, device=dev)
        self.cos_tab, self.sin_tab = rope_tables(dev)
        self.final_out = torch.empty(HID, dtype=torch.float32, device=dev)
        self.slot = torch.zeros(
            2, dtype=torch.int32, device=dev
        )  # [token_id, S]: device-side decode slot
        self.status = torch.zeros(1, dtype=torch.int32, device=dev)
        # Shared sequential reduce workspace (see moe.py): one NVLS set per full-attn layer would
        # exhaust the pool. TWO reduce protocols live here because the two decode graphs differ:
        #  - non-CP push (rep + it_ctr): the fused o_proj kernel replicates its stripes through
        #    the multicast VA and combines locally; per-CTA epochs, no host resets.
        #  - CP split (partial2 + cur_gen): the CP tail kernel writes this rank's partial and
        #    derives the generation from its arrival counter; WholeReduce folds the 8 ranks.
        # The 256-slot bar keeps their arrival counters DISJOINT -- push CTAs use slots [0:128],
        # the CP whole-reduce uses slot [128] (bound in attn_cp) -- so interleaved non-CP/CP
        # replays never see each other's signals and each protocol's parity stays self-consistent.
        if share_from is None:
            self.rep, _, self.rep_mc = self._symm(8 * 2 * HID, torch.float32)
            self.rep.zero_()
            self.partial2, self.peer_partial2, self.partial2_mc = self._symm(2 * HID, torch.float32)
            self.bar, _, self.bar_mc = self._symm(256, torch.int32)
            self.bar.zero_()
            if self.bar_mc == 0 or self.partial2_mc == 0 or self.rep_mc == 0:
                raise RuntimeError("multicast unavailable (NVLS required)")
            self.it_ctr = torch.zeros(128, dtype=torch.int32, device=dev)
            self.cur_gen = torch.zeros(1, dtype=torch.int32, device=dev)
            self.partial2.zero_()
        else:
            s = share_from
            self.rep, self.rep_mc = s.rep, s.rep_mc
            self.partial2, self.peer_partial2, self.partial2_mc = (
                s.partial2,
                s.peer_partial2,
                s.partial2_mc,
            )
            self.bar, self.bar_mc = s.bar, s.bar_mc
            self.it_ctr = s.it_ctr
            self.cur_gen = s.cur_gen
        torch.cuda.synchronize()
        dist.barrier()

    def _symm(self, numel, dtype):
        x = symm_mem.empty(numel, dtype=dtype, device=self.dev)
        h = symm_mem.rendezvous(x, dist.group.WORLD.group_name)
        peers = torch.tensor(list(h.buffer_ptrs), dtype=torch.int64, device=self.dev)
        return x, peers, int(getattr(h, "multicast_ptr", 0) or 0)

    def forward(self, x, pos):
        w = self.w
        ci, cf, cu = ctypes.c_int, ctypes.c_float, ctypes.c_ulonglong
        vpos = ctypes.c_void_p(self.v_cache.data_ptr() + pos * HEAD_DIM * 2)  # &v_cache[pos] (bf16)
        # 1. qkv projection (bf16 activation, folded weights): scatter to q_raw / gate / k_cache[pos] / v_cache[pos]
        self.ql.NexN25ProQkvProjLaunch(
            _p(x),
            _p(w["qkv_w"]),
            _p(w["qkv_scales"]),
            _p(self.q_raw),
            _p(self.gate),
            _p(self.k_cache),
            _p(self.v_cache),
            ci(pos),
            _sp(),
        )
        # 2. norm_rope fused: q/k RMSNorm+RoPE (s cancels); CTA-5 computes s from x, scales v_cache[pos] & gate
        self.nrl.NexN25ProNormRopeLaunch(
            _p(self.q_raw),
            _p(self.k_cache),
            _p(w["qn"]),
            _p(w["kn"]),
            _p(self.cos_tab),
            _p(self.sin_tab),
            _p(self.q_out),
            ci(pos),
            _p(x),
            vpos,
            vpos,
            _p(self.gate),
            _p(self.gate),
            ci(HID),
            cf(EPS),
            _sp(),
        )
        # 3. attn core: 4 q heads vs the single kv head's K/V over [0, pos]  (split-K flash decode)
        rc = self.wgml.NexN25ProAttnWgmmaLaunch(
            _p(self.q_out),
            _p(self.k_cache),
            _p(self.v_cache),
            ci(pos + 1),
            _p(self.ws_m),
            _p(self.ws_l),
            _p(self.ws_acc),
            _p(self.ctx),
            _p(self.wgmma_grid_counter),
            _sp(),
        )
        if rc != 0:
            raise RuntimeError(f"NexN25ProAttnWgmmaLaunch rc={rc}")
        # 4. o_proj (fused gate: ctx * sigmoid(gate)) + deterministic push reduce, ONE launch
        pf_ptr, pf_bytes = self.wait_pf
        self.opl.NexN25ProOProjReduceLaunch(
            _p(self.ctx),
            _p(self.gate),
            _p(w["o_w"]),
            _p(w["o_scales"]),
            cu(self.rep_mc),
            _p(self.rep),
            cu(self.bar_mc),
            _p(self.bar),
            _p(self.final_out),
            _p(self.status),
            _p(self.it_ctr),
            self.rank,
            self.world,
            None,
            ctypes.c_void_p(pf_ptr) if pf_ptr else None,
            ctypes.c_longlong(pf_bytes),
            _sp(),
        )
        return self.final_out

    def _forward_dyn_front(self, x):
        w = self.w
        ci, cf = ctypes.c_int, ctypes.c_float
        self.ql.NexN25ProQkvProjDynLaunch(
            _p(x),
            _p(w["qkv_w"]),
            _p(w["qkv_scales"]),
            _p(self.q_raw),
            _p(self.gate),
            _p(self.k_cache),
            _p(self.v_cache),
            _p(self.slot),
            _sp(),
        )
        self.nrl.NexN25ProNormRopeDynLaunch(
            _p(self.q_raw),
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

    def _forward_dyn_finish(self, resid):
        w = self.w
        cu = ctypes.c_ulonglong
        pf_ptr, pf_bytes = self.wait_pf
        self.opl.NexN25ProOProjReduceLaunch(
            _p(self.ctx),
            _p(self.gate),
            _p(w["o_w"]),
            _p(w["o_scales"]),
            cu(self.rep_mc),
            _p(self.rep),
            cu(self.bar_mc),
            _p(self.bar),
            _p(self.final_out),
            _p(self.status),
            _p(self.it_ctr),
            self.rank,
            self.world,
            None if resid is None else _p(resid),
            ctypes.c_void_p(pf_ptr) if pf_ptr else None,
            ctypes.c_longlong(pf_bytes),
            _sp(),
        )
        return self.final_out if resid is None else resid

    def forward_dyn(self, x, resid=None):
        """Device-S forward: pos/S read from self.slot[1] on-device (nothing baked). Identical math to
        forward() at pos=slot[1]; this is the graph-replay path for the growing KV cache during decode.
        resid=None -> final_out fp32; resid=r_bf16 -> o_proj reduce folds the residual add (r += reduced)."""
        self._forward_dyn_front(x)
        rc = self.wgml.NexN25ProAttnWgmmaDynLaunch(
            _p(self.q_out),
            _p(self.k_cache),
            _p(self.v_cache),
            _p(self.slot),
            _p(self.ws_m),
            _p(self.ws_l),
            _p(self.ws_acc),
            _p(self.ctx),
            _p(self.wgmma_grid_counter),
            _sp(),
        )
        if rc != 0:
            raise RuntimeError(f"NexN25ProAttnWgmmaDynLaunch rc={rc}")
        return self._forward_dyn_finish(resid)

    def reference(self, x, past_k, past_v, pos, quantize_activation=True):
        """TRUE: input_layernorm(x) with (1+iln), then fp8-activation attn_decode (ORIGINAL weights)."""
        xf = x.float()
        xn = (xf * torch.rsqrt(xf.pow(2).mean() + EPS) * (1.0 + self.iln.float())).to(
            torch.bfloat16
        )
        o_partial = _attention_decode_local(
            xn.unsqueeze(0), self.Wr, past_k, past_v, pos, quantize_activation
        )
        o = torch.empty(
            HID, dtype=torch.float32, device=self.dev
        )  # fresh (o_partial is an inference tensor)
        o.copy_(o_partial.reshape(HID).float())
        dist.all_reduce(o, op=dist.ReduceOp.SUM)
        return o

    def reference_folded(self, x, past_k, past_v, pos):
        """Kernel's intended math: bf16 activation, FOLDED weights, s on v & gate only (q/k cancel). Note this
        ref rounds to bf16 after rms_norm and before RoPE, while the kernel keeps fp32 through rms->rope, so it
        carries one extra intermediate round (~1e-4) -- it bounds the kernel arithmetic, not a bit-match."""
        w = self.w
        xf = x.float()
        s = torch.rsqrt(xf.pow(2).mean() + EPS)  # CTA-5 scalar
        sc = w["qkv_scales"].repeat_interleave(128, 0).repeat_interleave(128, 1)  # [2560,4096]
        y = ((w["qkv_w"].float() * sc) @ xf).to(torch.bfloat16)  # bf16-x GEMV -> [2560]
        qg = y[:QROWS].view(NHL, HEAD_DIM * Q_GATE)
        q_raw, gate_raw = qg[:, :HEAD_DIM].contiguous(), qg[:, HEAD_DIM:].reshape(LOCAL)
        k_raw = y[QROWS : QROWS + KROWS].view(NKVL, HEAD_DIM)
        v_raw = y[QROWS + KROWS :].view(NKVL, HEAD_DIM)
        q = _partial_rope(_rms_norm(q_raw, w["qn"]), pos)  # [4,256]  NO s
        k = _partial_rope(_rms_norm(k_raw, w["kn"]), pos)  # [1,256]  NO s
        v = (v_raw.float() * s).to(torch.bfloat16)  # [1,256]  x s
        gate = (gate_raw.float() * s).to(torch.bfloat16)  # [1024]   x s
        Kf = torch.cat([past_k, k], 0)  # [pos+1,256]  single kv head
        Vf = torch.cat([past_v, v], 0)
        scores = (
            torch.einsum("hd,sd->hs", q.float(), Kf.float()) * SCALING
        )  # [4,pos+1]  GQA 4:1 broadcast
        ctx = torch.einsum("hs,sd->hd", torch.softmax(scores, -1), Vf.float())  # [4,256]
        attn = ctx.reshape(LOCAL).to(torch.bfloat16).float() * torch.sigmoid(gate.float())
        osc = w["o_scales"].repeat_interleave(128, 0).repeat_interleave(128, 1)  # [4096,1024]
        o = (w["o_w"].float() * osc) @ attn
        dist.all_reduce(o, op=dist.ReduceOp.SUM)
        return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3)  # first full-attention layer
    ap.add_argument(
        "--past", type=int, default=4096
    )  # KV-cache length S (past tokens); new token at pos=S
    cli = ap.parse_args()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl")
    assert world == 8
    pos = cli.past

    a = FullAttn(cli.layer, rank, world, dev)
    assert 0 <= pos < CACHE_LEN, (
        f"pos={pos} must be in [0, {CACHE_LEN}) (new token writes cache slot pos)"
    )
    # Coherent TP=8: the residual stream x is REPLICATED across all ranks; the KV cache is replicated within
    # each kv-head group (ranks 0-3 share kv head 0, 4-7 share kv head 1). Then the o_proj all-reduce sums the
    # 32 heads of ONE real token -> "vs true" is a genuine coherent TP=8 model-output check, not a per-rank sum.
    xg = torch.Generator(device=dev).manual_seed(1234)  # same on every rank
    x = torch.randn(
        HID, dtype=torch.bfloat16, device=dev, generator=xg
    )  # RAW residual (replicated)
    kg = torch.Generator(device=dev).manual_seed(
        4321 + rank // (world // N_KV)
    )  # per kv-head group
    past_k = (
        torch.randn(pos, HEAD_DIM, dtype=torch.bfloat16, device=dev, generator=kg) * 0.1
    ).contiguous()
    past_v = (
        torch.randn(pos, HEAD_DIM, dtype=torch.bfloat16, device=dev, generator=kg) * 0.1
    ).contiguous()
    a.k_cache[:pos].copy_(past_k)
    a.v_cache[:pos].copy_(past_v)  # seed the past cache

    exp_true = a.reference(x, past_k, past_v, pos)  # fp8 activation: requant-fold + fp8-act
    exp_reqt = a.reference(x, past_k, past_v, pos, quantize_activation=False)  # bf16 activation
    exp_fold = a.reference_folded(
        x, past_k, past_v, pos
    )  # kernel's exact math (bf16-act, folded, s-fusion)

    a.forward(x, pos)
    torch.cuda.synchronize()
    dist.barrier()

    def amax(v):  # max across ranks (worst-case reporting)
        t = torch.tensor([float(v)], device=dev, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.item()

    def rel(a_, b_):
        return amax(((a_ - b_).norm() / b_.norm().clamp_min(1e-9)).item())

    rel_true = rel(a.final_out, exp_true)
    rel_reqt = rel(a.final_out, exp_reqt)
    rel_fold = rel(a.final_out, exp_fold)
    st = int(amax(a.status.item()))
    # ---- device-S growth: ONE captured graph, S read from slot[1] each replay, KV cache accumulates ----
    P0, NGROW = pos, 8

    def seed():
        a.k_cache[:P0].copy_(past_k)
        a.v_cache[:P0].copy_(past_v)
        a.slot[1] = P0

    for _ in range(5):  # warmup (valid forward at pos=P0 each)
        seed()
        a.forward_dyn(x)
    torch.cuda.synchronize()
    seed()
    gdyn = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gdyn):
        a.forward_dyn(x)
    torch.cuda.synchronize()
    seed()  # reset the capture-replay's write at cache[P0]
    grow_worst = 0.0
    for s in range(NGROW):  # S = P0, P0+1, ... device-side
        pk = a.k_cache[: P0 + s].clone()
        pv = a.v_cache[: P0 + s].clone()  # accumulated past BEFORE this step
        gdyn.replay()  # writes cache[P0+s], attends [:P0+s+1]
        torch.cuda.synchronize()
        ref = a.reference_folded(x, pk, pv, P0 + s)
        grow_worst = max(
            grow_worst, ((a.final_out - ref).norm() / ref.norm().clamp_min(1e-9)).item()
        )
        a.slot[1] = P0 + s + 1  # (the sampler bumps S in-graph in the real loop)
    grow_worst = amax(grow_worst)
    grow_st = int(amax(a.status.item()))
    filled = int(
        (a.k_cache[P0 : P0 + NGROW].abs().sum(-1) > 0).sum().item()
    )  # cache slots actually written
    filled = int(amax(filled))

    # warmup, capture, and time via the local benchmark helper (hot back-to-back, TP-max)
    from nex_n2_5_pro.bench import bench_graph

    us = (
        bench_graph(
            lambda: a.forward(x, pos),
            warmup=30,
            rep=2000,
            flush_l2=False,
            tp_reduce=True,
            device=str(dev),
        ).median_ms
        * 1e3
    )
    if rank == 0:
        ok = rel_fold < 5e-3 and st == 0 and grow_worst < 5e-3 and grow_st == 0 and filled == NGROW
        print(
            f"== nexrt FULL-ATTN mixer (TP=8, layer {cli.layer}, S={pos}, fused input-RMSNorm) ==",
            flush=True,
        )
        print(
            f"   vs FOLDED-wts ref (kernel math)     : out relL2 {rel_fold:.3e}   (proves the wiring + rms fusion)",
            flush=True,
        )
        print(
            f"   vs TRUE bf16-act (requant-fold only): out relL2 {rel_reqt:.3e}   (requant cost; cf GDN 3.8e-2)",
            flush=True,
        )
        print(
            f"   vs TRUE fp8-act  (== HF)            : out relL2 {rel_true:.3e}   (+ fp8-activation gap)",
            flush=True,
        )
        print(
            f"   device-S growth ({NGROW} steps, one graph): worst relL2 {grow_worst:.3e}, status {grow_st}, "
            f"cache slots filled {filled}/{NGROW}",
            flush=True,
        )
        print(f"   status {st}", flush=True)
        print(f"   [{'OK' if ok else 'CHECK'}]  WGMMA mega: {us:.2f} us/tok (graph)", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
