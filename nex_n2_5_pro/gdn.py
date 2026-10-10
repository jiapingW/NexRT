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

"""nexrt nex_n2_5_pro decode GDN (linear-attention) mixer block (TP=8), input-RMSNorm FUSED into the front.

Pipeline per rank: front (in_proj qkv/z + a/b + conv + SiLU + gates, WITH the input-norm reduction fused)
  -> recurrent-core (Gated-DeltaNet) -> final o_proj (gated RMSNorm + out_proj + symm-mem TP reduce).

Input RMSNorm handling: the `(1+input_layernorm.weight)` scale is folded into the front
weights (qkvz fp8-requant, a/b bf16); the `rms_scale = rsqrt(mean(x^2)+eps)` reduction is recomputed by
every front CTA/warp from the x it already streams and applied to every
GEMV output before its nonlinearity. z comes out of the SAME front GEMV as mixed_qkv, so it is scaled there
too -> the final block's silu(z) needs nothing exported. Parity is vs the true GDN with input_layernorm.

Run:  torchrun --standalone --nproc_per_node=8 gdn.py --layer 0
"""

import argparse
import ctypes
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
KDIR = HERE / "kernels" / "linear_attn"
if __package__ in (None, ""):
    sys.path.insert(0, str(HERE.parent))

from nex_n2_5_pro.kernel_build import build_kernel  # noqa: E402
from nex_n2_5_pro.weights import (  # noqa: E402
    load_layer_norms,
    load_linear_attention_weights,
    shard_linear_attention_weights,
)

HID, QKV_ROWS, Z_ROWS, HEADS, QK_HEADS, GQA, DIM, LOCAL = 4096, 1536, 1024, 8, 2, 4, 128, 1024
Q_OFF, K_OFF, V_OFF = 0, QK_HEADS * DIM, 2 * QK_HEADS * DIM
EPS, SCALE = 1e-6, DIM**-0.5
FP8 = torch.float8_e4m3fn


def build(name):
    return build_kernel(KDIR, name)


def _p(t):
    return ctypes.c_void_p(t.data_ptr())


def _sp():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def _fold_requant_2d(w_fp8, scale, gamma):
    """Fold gamma column-wise into an [R,C] fp8 weight (block-128 scale [R/128,C/128]) via
    dequant->fold->requant with fresh per-block scales (no overflow). gamma [C] fp32."""
    R, C = w_fp8.shape
    RB, CB = R // 128, C // 128
    w = w_fp8.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1) * gamma.view(1, C)
    wb = w.view(RB, 128, CB, 128)
    ns = (wb.abs().amax(dim=(1, 3)) / 448.0).clamp_min(1e-12)  # [RB,CB]
    wq = (wb / ns.view(RB, 1, CB, 1)).clamp(-448.0, 448.0).to(FP8).view(R, C)
    return wq.contiguous(), ns.contiguous()


def convert_weights(layer, rank, dev):
    """-> (kernel_w with (1+input_ln) folded in, orig_w unfolded for the reference, input_ln_w)."""
    Wr = shard_linear_attention_weights(load_linear_attention_weights(layer, dev), rank, tp=8)
    iln, _ = load_layer_norms(layer, dev)
    iln = iln.float()  # [4096]
    gamma = 1.0 + iln
    qkvz_w = torch.cat([Wr["qkv"], Wr["z"]], 0).contiguous()  # [2560,4096] fp8
    qkvz_s = torch.cat([Wr["qkvs"], Wr["zs"]], 0).contiguous().float()  # [20,32]
    qkvz_wf, qkvz_sf = _fold_requant_2d(qkvz_w, qkvz_s, gamma)
    shared = {
        "a_log_dt": torch.cat([Wr["A_log"].float(), Wr["dt_bias"].float()], 0).contiguous(),
        "conv_w": Wr["conv_w"].reshape(QKV_ROWS, 4).contiguous(),
        "o_w": Wr["o"].contiguous(),
        "o_scales": Wr["os"].contiguous().float(),
        "norm_w": Wr["norm_w"].contiguous(),
    }
    kernel_w = {
        "qkvz_w": qkvz_wf,
        "qkvz_scales": qkvz_sf,
        "b_w": (Wr["b"].float() * gamma.view(1, HID)).to(torch.bfloat16).contiguous(),
        "a_w": (Wr["a"].float() * gamma.view(1, HID)).to(torch.bfloat16).contiguous(),
        **shared,
    }
    orig_w = {
        "qkvz_w": qkvz_w,
        "qkvz_scales": qkvz_s,
        "b_w": Wr["b"].contiguous(),
        "a_w": Wr["a"].contiguous(),
        **shared,
    }
    return kernel_w, orig_w, iln


def convert_weights_ext(ext):
    """Bridge mode: derive the folded front from sglang's loaded per-rank tensors without
    rereading checkpoint weights. sglang keeps in_proj_qkv [1536,4096] and in_proj_z [1024,4096]
    separate with the same per-rank q|k|v gather nexrt uses (byte-verified); the concat+fold is
    the same recipe.
    conv1d / A_log / dt_bias / norm / out_proj are bit-identical -> aliases (conv as a view)."""
    iln = ext["iln"].float()
    gamma = 1.0 + iln
    qkvz_w = torch.cat([ext["qkv_w"], ext["z_w"]], 0).contiguous()  # [2560,4096] fp8
    qkvz_s = torch.cat([ext["qkv_scales"], ext["z_scales"]], 0).contiguous().float()  # [20,32]
    qkvz_wf, qkvz_sf = _fold_requant_2d(qkvz_w, qkvz_s, gamma)
    kernel_w = {
        "qkvz_w": qkvz_wf,
        "qkvz_scales": qkvz_sf,
        "b_w": (ext["b_w"].float() * gamma.view(1, HID)).to(torch.bfloat16).contiguous(),
        "a_w": (ext["a_w"].float() * gamma.view(1, HID)).to(torch.bfloat16).contiguous(),
        "a_log_dt": torch.cat([ext["A_log"].float(), ext["dt_bias"].float()], 0).contiguous(),
        "conv_w": ext["conv_w"].reshape(QKV_ROWS, 4),
        "o_w": ext["o_w"],
        "o_scales": ext["o_scales"].float(),
        "norm_w": ext["norm_w"].to(torch.bfloat16),
    }
    return kernel_w, iln


class Gdn:
    def __init__(self, layer, rank, world, dev, keep_ref=True, share_from=None, ext=None):
        self.rank, self.world, self.dev = rank, world, dev
        self.ext = ext
        vp, ci, cu = ctypes.c_void_p, ctypes.c_int, ctypes.c_ulonglong
        self.fl = ctypes.CDLL(str(build("nex_n2_5_pro_linear_front_gemv_ctypes")))
        self.fl.NexN25ProLinearFrontLaunch.argtypes = [vp] * 13
        self.rl = ctypes.CDLL(str(build("nex_n2_5_pro_linear_recurrent_core_ctypes")))
        self.rl.NexN25ProLinearRecurrentLaunch.argtypes = [vp] * 6
        self.ol = ctypes.CDLL(str(build("nex_n2_5_pro_linear_final_o_proj_ctypes")))
        self.ol.NexN25ProLinearFinalOProjLaunch.argtypes = (
            [vp] * 5 + [cu, vp, cu] + [vp] * 4 + [ci, ci, vp, vp, ctypes.c_longlong, vp]
        )
        # Wait-shadow prefetch of the downstream MoE gate weights, issued by the o_proj
        # kernel during its peer wait; the decode engine wires (ptr, bytes). (0, 0)
        # (standalone blocks) disables it with no other effect.
        self.wait_pf = (0, 0)
        if ext is not None:  # bridge: alias sglang's live tensors
            self.w, self.iln = convert_weights_ext(ext)
            self.orig_w = None
        else:
            self.w, self.orig_w, self.iln = convert_weights(layer, rank, dev)
        if not keep_ref:  # decode engine: drop raw fp8 (ref-only)
            self.orig_w = None
        self.qkv_out = torch.empty(QKV_ROWS, dtype=torch.bfloat16, device=dev)
        self.z_out = torch.empty(Z_ROWS, dtype=torch.bfloat16, device=dev)
        self.beta_out = torch.empty(HEADS, dtype=torch.float32, device=dev)
        self.g_out = torch.empty(HEADS, dtype=torch.float32, device=dev)
        self.recurrent_out = torch.empty(HEADS, DIM, dtype=torch.bfloat16, device=dev)
        self.final_out = torch.empty(HID, dtype=torch.float32, device=dev)
        self.conv_state4 = torch.zeros(QKV_ROWS, 4, dtype=torch.bfloat16, device=dev)
        self.state_t = torch.zeros(HEADS, DIM, DIM, dtype=torch.float32, device=dev)
        self.status = torch.zeros(1, dtype=torch.int32, device=dev)
        # Shared sequential push-reduce workspace (see moe.py): one NVLS set per GDN layer would
        # exhaust the pool at 60 layers. rep = replicated push buffer ([8 ranks][2 parities][HID]
        # fp32, multicast-written); bar = one NVLS arrival slot per CTA; it_ctr = per-CTA epochs
        # (monotonic -- parity and spin targets derive from them, no host resets).
        if share_from is None:
            self.rep, _, self.rep_mc = self._symm(8 * 2 * HID, torch.float32)
            self.rep.zero_()
            self.bar, _, self.bar_mc = self._symm(128, torch.int32)
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
        """resid=None -> reduce writes final_out[HID] fp32 (standalone).  resid=x_bf16 -> the reduce FOLDS the
        residual add into its epilogue (x += reduced, in place, bf16), skipping the separate add kernel."""
        w = self.w
        self.fl.NexN25ProLinearFrontLaunch(
            _p(x),
            _p(w["qkvz_w"]),
            _p(w["qkvz_scales"]),
            _p(w["b_w"]),
            _p(w["a_w"]),
            _p(w["a_log_dt"]),
            _p(w["conv_w"]),
            _p(self.conv_state4),
            _p(self.qkv_out),
            _p(self.z_out),
            _p(self.beta_out),
            _p(self.g_out),
            _sp(),
        )
        self.rl.NexN25ProLinearRecurrentLaunch(
            _p(self.qkv_out),
            _p(self.beta_out),
            _p(self.g_out),
            _p(self.state_t),
            _p(self.recurrent_out),
            _sp(),
        )
        # ONE fused launch: gated RMSNorm + o_proj + deterministic push reduce (+ folded residual).
        cu = ctypes.c_ulonglong
        pf_ptr, pf_bytes = self.wait_pf
        self.ol.NexN25ProLinearFinalOProjLaunch(
            _p(self.recurrent_out),
            _p(self.z_out),
            _p(w["norm_w"]),
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


def _l2(x):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + EPS)


def reference(orig, iln, x, conv0, state0):
    """True GDN WITH input_layernorm: xn = rms(x)*(1+iln)*x, then the front on xn with ORIGINAL weights."""
    xf = x.float()
    xn = torch.rsqrt(xf.pow(2).mean() + EPS) * (1.0 + iln.float()) * xf  # input_layernorm(x)
    sc = orig["qkvz_scales"].repeat_interleave(128, 0).repeat_interleave(128, 1)
    y = orig["qkvz_w"].float().mul(sc).matmul(xn)
    mixed, z = y[:QKV_ROWS].to(torch.bfloat16), y[QKV_ROWS:].to(torch.bfloat16)
    st_old = conv0[:, 0:3]
    conv_in = torch.cat([st_old, mixed[:, None]], 1)
    new4 = conv0.clone()
    new4[:, 0], new4[:, 1], new4[:, 2] = st_old[:, 1], st_old[:, 2], mixed
    qkv = F.silu((conv_in.float() * orig["conv_w"].float()).sum(1)).to(torch.bfloat16)
    beta = torch.sigmoid((orig["b_w"].float() @ xn).to(torch.bfloat16)).float()
    a = (orig["a_w"].float() @ xn).to(torch.bfloat16)
    g = -torch.exp(orig["a_log_dt"][:HEADS]) * F.softplus(a.float() + orig["a_log_dt"][HEADS:])
    query = qkv[Q_OFF:K_OFF].reshape(QK_HEADS, DIM).repeat_interleave(GQA, 0)
    key = qkv[K_OFF:V_OFF].reshape(QK_HEADS, DIM).repeat_interleave(GQA, 0)
    value = qkv[V_OFF:].reshape(HEADS, DIM).float()
    q, k = _l2(query.float()) * SCALE, _l2(key.float())
    s0 = state0.transpose(1, 2).contiguous() * torch.exp(g).view(HEADS, 1, 1)
    kv = (s0 * k.unsqueeze(-1)).sum(-2)
    sn = s0 + k.unsqueeze(-1) * ((value - kv) * beta.view(HEADS, 1)).unsqueeze(-2)
    rout = (sn * q.unsqueeze(-1)).sum(-2).to(torch.bfloat16)
    xg = rout.float() * torch.rsqrt(rout.float().pow(2).mean(-1, keepdim=True) + EPS)
    core = (
        (orig["norm_w"] * xg.to(torch.bfloat16) * F.silu(z.reshape(HEADS, DIM).float()))
        .to(torch.bfloat16)
        .reshape(LOCAL)
    )
    o = orig["o_w"].float() * orig["o_scales"].repeat_interleave(128, 0).repeat_interleave(128, 1)
    local = o @ core.float()
    dist.all_reduce(local, op=dist.ReduceOp.SUM)
    return local, new4, sn.transpose(1, 2).contiguous()


def reference_folded(kw, x, conv0, state0):
    """Kernel's exact intended math: FOLDED weights + rms applied to the front outputs (raw x). Isolates
    the kernel arithmetic from the requant weight-representation difference (should be ~1e-4 vs the kernel)."""
    xf = x.float()
    rms = torch.rsqrt(xf.pow(2).mean() + EPS)
    sc = kw["qkvz_scales"].repeat_interleave(128, 0).repeat_interleave(128, 1)
    y = rms * (kw["qkvz_w"].float().mul(sc).matmul(xf))
    mixed, z = y[:QKV_ROWS].to(torch.bfloat16), y[QKV_ROWS:].to(torch.bfloat16)
    st_old = conv0[:, 0:3]
    qkv = F.silu((torch.cat([st_old, mixed[:, None]], 1).float() * kw["conv_w"].float()).sum(1)).to(
        torch.bfloat16
    )
    beta = torch.sigmoid((rms * (kw["b_w"].float() @ xf)).to(torch.bfloat16)).float()
    a = (rms * (kw["a_w"].float() @ xf)).to(torch.bfloat16)
    g = -torch.exp(kw["a_log_dt"][:HEADS]) * F.softplus(a.float() + kw["a_log_dt"][HEADS:])
    query = qkv[Q_OFF:K_OFF].reshape(QK_HEADS, DIM).repeat_interleave(GQA, 0)
    key = qkv[K_OFF:V_OFF].reshape(QK_HEADS, DIM).repeat_interleave(GQA, 0)
    value = qkv[V_OFF:].reshape(HEADS, DIM).float()
    q, k = _l2(query.float()) * SCALE, _l2(key.float())
    s0 = state0.transpose(1, 2).contiguous() * torch.exp(g).view(HEADS, 1, 1)
    kv = (s0 * k.unsqueeze(-1)).sum(-2)
    sn = s0 + k.unsqueeze(-1) * ((value - kv) * beta.view(HEADS, 1)).unsqueeze(-2)
    rout = (sn * q.unsqueeze(-1)).sum(-2).to(torch.bfloat16)
    xg = rout.float() * torch.rsqrt(rout.float().pow(2).mean(-1, keepdim=True) + EPS)
    core = (
        (kw["norm_w"] * xg.to(torch.bfloat16) * F.silu(z.reshape(HEADS, DIM).float()))
        .to(torch.bfloat16)
        .reshape(LOCAL)
    )
    o = kw["o_w"].float() * kw["o_scales"].repeat_interleave(128, 0).repeat_interleave(128, 1)
    local = o @ core.float()
    dist.all_reduce(local, op=dist.ReduceOp.SUM)
    return local


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    cli = ap.parse_args()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl")
    assert world == 8

    g = Gdn(cli.layer, rank, world, dev)
    gen = torch.Generator(device=dev).manual_seed(1234 + rank)
    x = torch.randn(
        HID, dtype=torch.bfloat16, device=dev, generator=gen
    )  # RAW residual-stream input
    conv0 = (
        torch.randn(QKV_ROWS, 4, dtype=torch.bfloat16, device=dev, generator=gen) * 0.1
    ).contiguous()
    state0 = (
        torch.randn(HEADS, DIM, DIM, dtype=torch.float32, device=dev, generator=gen) * 0.1
    ).contiguous()
    g.conv_state4.copy_(conv0)
    g.state_t.copy_(state0)
    exp_out, exp_conv, exp_state = reference(g.orig_w, g.iln, x, conv0, state0)

    g.forward(x)
    torch.cuda.synchronize()
    dist.barrier()
    out_rel = ((g.final_out - exp_out).norm() / exp_out.norm().clamp_min(1e-9)).item()
    st_rel = ((g.state_t - exp_state).norm() / exp_state.norm().clamp_min(1e-9)).item()
    cv_rel = (
        (g.conv_state4.float() - exp_conv.float()).norm() / exp_conv.float().norm().clamp_min(1e-9)
    ).item()
    st = torch.tensor([int(g.status.item())], device=dev)
    dist.all_reduce(st, op=dist.ReduceOp.MAX)
    for v in ([out_rel], [st_rel], [cv_rel]):
        t = torch.tensor(v, device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        v[0] = t.item()
    ec_out = reference_folded(g.w, x, conv0, state0)  # kernel's exact math (folded wts + rms)
    out_rel_c = ((g.final_out - ec_out).norm() / ec_out.norm().clamp_min(1e-9)).item()
    tc = torch.tensor([out_rel_c], device=dev)
    dist.all_reduce(tc, op=dist.ReduceOp.MAX)
    out_rel_c = tc.item()

    from nex_n2_5_pro.bench import bench_graph  # vendored profiler (hot back-to-back, TP-max)

    us = (
        bench_graph(
            lambda: g.forward(x),
            warmup=30,
            rep=2000,
            flush_l2=False,
            tp_reduce=True,
            device=str(dev),
        ).median_ms
        * 1e3
    )

    if rank == 0:
        ok = out_rel_c < 5e-3 and st.item() == 0  # kernel arithmetic vs its own (folded) weights
        print(f"== nexrt GDN mixer (TP=8, layer {cli.layer}, fused input-RMSNorm) ==", flush=True)
        print(
            f"   vs TRUE GDN (orig wts + input-norm): out relL2 {out_rel:.3e}   (= fp8 requant-fold cost)",
            flush=True,
        )
        print(
            f"   vs FOLDED-wts ref (kernel math)    : out relL2 {out_rel_c:.3e}   (proves the rms fusion correct)",
            flush=True,
        )
        print(
            f"   state relL2 {st_rel:.3e} | conv relL2 {cv_rel:.3e} | status {st.item()}",
            flush=True,
        )
        print(
            f"   [{'OK' if ok else 'CHECK'}]  GDN layer latency (graph): {us:.2f} us/tok",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
