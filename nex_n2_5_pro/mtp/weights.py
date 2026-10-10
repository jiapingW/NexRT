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

"""Per-rank weight preparation for the MTP draft block.

Produces the ``ext`` dictionaries the existing FullAttn/Moe aliasing path
consumes, plus the draft fc and norm tensors from
``$NEXRT_LAYERS_DIR/mtp.safetensors`` (keys ``mtp.*``, scales stored as
``weight_scale`` fp32 dequant multipliers -- the layer.N.safetensors
convention):

  attn_ext(rank, dev)   -> FullAttn(ext=...)   (per-rank unfolded qkv shards; the
                                                block folds (1+input_ln) itself)
  moe_ext(rank, dev)    -> Moe(ext=...)        (per-rank raw tensors in the
                                                weight-sharing layout)
  fc_shard(rank, dev)   -> (fc_w8 [4096,1024] fp8, fc_scales [32,8] fp32)  o_proj-shaped
  head_w_fused(lm_head, dev) -> resident target shard x (1 + mtp.norm), bf16
  norm_weights(dev, model_norm) -> preprocess norms, sharing the target's final norm
"""

from __future__ import annotations

import torch
from safetensors import safe_open

from nex_n2_5_pro.config import LAYERS_DIR as SNAP
from nex_n2_5_pro.weights import shard_attention_weights

MTP_ST = SNAP / "mtp.safetensors"
HID = 4096
N_EXP, RINTER = 512, 128
FC_COLS = 1024  # 8192 / TP8, this rank's input slice
FP8 = torch.float8_e4m3fn


def _open():
    return safe_open(str(MTP_ST), framework="pt")


def attn_ext(rank: int, dev) -> dict:
    """FullAttn ``ext`` dict from the mtp attention tensors (per-rank shards;
    FullAttn's own ext path folds (1+input_ln) with fresh block scales)."""
    with _open() as f:
        g = lambda n: f.get_tensor("mtp.layers.0.self_attn." + n).to(dev)  # noqa: E731
        W = dict(
            q=g("q_proj.weight"),
            qs=g("q_proj.weight_scale").float(),
            k=g("k_proj.weight"),
            ks=g("k_proj.weight_scale").float(),
            v=g("v_proj.weight"),
            vs=g("v_proj.weight_scale").float(),
            o=g("o_proj.weight"),
            os=g("o_proj.weight_scale").float(),
            qn=g("q_norm.weight").to(torch.bfloat16),
            kn=g("k_norm.weight").to(torch.bfloat16),
        )
        iln = f.get_tensor("mtp.layers.0.input_layernorm.weight").to(dev)
    Wr = shard_attention_weights(W, rank, tp=8)
    return {
        "iln": iln,
        "qkv_w": torch.cat([Wr["q"], Wr["k"], Wr["v"]], 0).contiguous(),
        "qkv_scales": torch.cat([Wr["qs"], Wr["ks"], Wr["vs"]], 0).contiguous(),
        "qn": Wr["qn"].contiguous(),
        "kn": Wr["kn"].contiguous(),
        "o_w": Wr["o"].contiguous(),
        "o_scales": Wr["os"].contiguous().float(),
    }


def moe_ext(rank: int, dev) -> dict:
    """Moe ``ext`` dict from the mtp MoE tensors, per-rank raw layout.

    Expert slices are taken per expert BEFORE stacking, so the CPU transient is
    the per-rank shard (~0.6 GB), never the full 512-expert stacks; only the
    final per-rank tensors are moved to the device."""
    lo, hi = rank * RINTER, (rank + 1) * RINTER
    with _open() as f:
        pre = "mtp.layers.0.mlp."
        g = lambda n: f.get_tensor(pre + n)  # noqa: E731  CPU tensors
        gate_w = g("gate.weight").to(torch.bfloat16)
        sgate_w = g("shared_expert_gate.weight").to(torch.bfloat16)
        s_g1, s_s1 = (
            g("shared_expert.gate_proj.weight"),
            g("shared_expert.gate_proj.weight_scale").float(),
        )
        s_g3, s_s3 = (
            g("shared_expert.up_proj.weight"),
            g("shared_expert.up_proj.weight_scale").float(),
        )
        s_g2, s_s2 = (
            g("shared_expert.down_proj.weight"),
            g("shared_expert.down_proj.weight_scale").float(),
        )
        e1, e3, e2 = [], [], []
        s1, s3, s2 = [], [], []
        for e in range(N_EXP):
            ge = lambda n: f.get_tensor(f"{pre}experts.{e}.{n}")  # noqa: E731  CPU
            e1.append(ge("gate_proj.weight")[lo:hi])
            s1.append(ge("gate_proj.weight_scale").float()[rank])
            e3.append(ge("up_proj.weight")[lo:hi])
            s3.append(ge("up_proj.weight_scale").float()[rank])
            e2.append(ge("down_proj.weight")[:, lo:hi])
            s2.append(ge("down_proj.weight_scale").float()[:, rank])
        post_ln = f.get_tensor("mtp.layers.0.post_attention_layernorm.weight")
    ext = {
        "post_ln": post_ln.float(),
        "gate_w": gate_w,
        "sgate_w": sgate_w,
        "w13": torch.cat([torch.stack(e1), torch.stack(e3)], 1),  # [512,256,4096]
        "w13_scales": torch.stack([torch.stack(s1), torch.stack(s3)], 1),  # [512,2,32]
        "shared_gate_up": torch.cat([s_g1[lo:hi], s_g3[lo:hi]], 0),  # [256,4096]
        "shared_scales": torch.cat([s_s1[rank], s_s3[rank]]),  # [64]
        "w2": torch.stack(e2),  # [512,4096,128]
        "w2_scales": torch.stack(s2),  # [512,32]
        "shared_down": s_g2[:, lo:hi],  # [4096,128]
        "shared_down_scales": s_s2[:, rank],  # [32]
    }
    out = {k: (v.to(dev).contiguous() if torch.is_tensor(v) else v) for k, v in ext.items()}
    torch.cuda.empty_cache()
    return out


def fc_shard(rank: int, dev):
    """This rank's fc input-column slice, fp8 block-128 quantized (o_proj kernel layout).

    fc is bf16 in the checkpoint; the draft path quantizes it so the existing o_proj
    kernels apply verbatim. Draft numerics affect only acceptance rate, never the
    emitted stream, so this is golden-gate-safe by construction.
    """
    with _open() as f:
        w = f.get_tensor("mtp.fc.weight").to(dev)  # [4096, 8192] bf16
    ws = w[:, rank * FC_COLS : (rank + 1) * FC_COLS].float()  # [4096, 1024]
    R, C = ws.shape
    wb = ws.view(R // 128, 128, C // 128, 128)
    ns = (wb.abs().amax(dim=(1, 3)) / 448.0).clamp_min(1e-12)  # [32, 8]
    wq = (wb / ns.view(R // 128, 1, C // 128, 1)).clamp(-448.0, 448.0).to(FP8).view(R, C)
    return wq.contiguous(), ns.contiguous().float()


def head_w_fused(lm_head: torch.Tensor, dev: torch.device | str) -> torch.Tensor:
    """W_fused for the DRAFT head: this rank's lm_head vocab shard x (1 + mtp.norm).
    Same fold as Head.W_fused but with mtp.norm instead of model.norm -- the draft head
    consumes the MTP layer's RAW output hidden with mtp.norm applied via the inv_rms trick."""
    if lm_head is None:
        raise RuntimeError("construct MtpDraft before capturing the target head")
    with _open() as f:
        mtp_norm = f.get_tensor("mtp.norm.weight").to(dev)
    return (lm_head.float() * (1.0 + mtp_norm.float())).to(torch.bfloat16).contiguous()


def norm_weights(dev: torch.device | str, model_norm: torch.Tensor) -> dict[str, torch.Tensor]:
    """The four norm vectors the fused preprocess kernel reads, bf16 [4096] each."""
    with _open() as f:
        pre_e = f.get_tensor("mtp.pre_fc_norm_embedding.weight").to(dev, torch.bfloat16)
        pre_h = f.get_tensor("mtp.pre_fc_norm_hidden.weight").to(dev, torch.bfloat16)
        mtp_norm = f.get_tensor("mtp.norm.weight").to(dev, torch.bfloat16)
    return {
        "pre_fc_embed": pre_e.contiguous(),
        "pre_fc_hidden": pre_h.contiguous(),
        "mtp_norm": mtp_norm.contiguous(),
        "model_norm": model_norm.contiguous(),
    }
