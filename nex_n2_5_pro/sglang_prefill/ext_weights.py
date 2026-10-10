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

"""Adapt live SGLang tensors to NexRT block inputs.

Routed MoE weights and scales alias the SGLang tensors. Blocks derive any
additional layouts needed by the selected mode from these detached inputs.
The embedding is gathered once from the rank-local vocabulary shards.
"""

import torch
import torch.distributed as dist

VOCAB, HID = 248320, 4096


def _d(t):
    """Detach without copying so weight preparation does not retain autograd state."""
    return t.detach()


class SglangExt:
    def __init__(self, model):
        self.model = model  # Rank-local SGLang model.
        self.lm = model.model  # text model: .layers / .embed_tokens / .norm

    def moe(self, L):
        mlp = self.lm.layers[L].mlp
        e = mlp.experts
        return {
            "post_ln": _d(self.lm.layers[L].post_attention_layernorm.weight),
            "gate_w": _d(mlp.gate.weight),  # [512,4096] bf16 replicated
            "sgate_w": _d(mlp.shared_expert_gate.weight),  # [1,4096] bf16
            "w13": _d(e.w13_weight),  # [512,256,4096] fp8 ALIAS
            "w13_scales": _d(e.w13_weight_scale_inv),  # [512,2,32] fp32 ALIAS
            "w2": _d(e.w2_weight),  # [512,4096,128] fp8 ALIAS
            "w2_scales": _d(e.w2_weight_scale_inv),  # [512,32,1] fp32 ALIAS
            "shared_gate_up": _d(mlp.shared_expert.gate_up_proj.weight),  # [256,4096] fp8 ALIAS
            "shared_scales": _d(mlp.shared_expert.gate_up_proj.weight_scale_inv),
            "shared_down": _d(mlp.shared_expert.down_proj.weight),  # [4096,128] fp8 ALIAS
            "shared_down_scales": _d(mlp.shared_expert.down_proj.weight_scale_inv),
        }

    def attn(self, L):
        lay = self.lm.layers[L]
        return {
            "iln": _d(lay.input_layernorm.weight),
            "qkv_w": _d(lay.qkv_proj.weight),  # [2560,4096] fp8 (fold source)
            "qkv_scales": _d(lay.qkv_proj.weight_scale_inv),  # [20,32]
            "qn": _d(lay.q_norm.weight),
            "kn": _d(lay.k_norm.weight),  # [256] bf16
            "o_w": _d(lay.o_proj.weight),  # [4096,1024] fp8 ALIAS
            "o_scales": _d(lay.o_proj.weight_scale_inv),  # [32,8] fp32 ALIAS
        }

    def gdn(self, L):
        la = self.lm.layers[L].linear_attn
        out = {
            "iln": _d(self.lm.layers[L].input_layernorm.weight),
            "conv_w": _d(la.conv1d.weight),  # [1536,1,4] bf16 ALIAS (viewed)
            "A_log": _d(la.A_log),
            "dt_bias": _d(la.dt_bias),  # [8]
            "norm_w": _d(la.norm.weight),  # [128] bf16 ALIAS
            "o_w": _d(la.out_proj.weight),  # [4096,1024] fp8 ALIAS
            "o_scales": _d(la.out_proj.weight_scale_inv),  # [32,8] fp32 ALIAS
        }
        if hasattr(la, "in_proj_qkv"):
            # Older sglang: separate qkv / z / b / a parameters.
            out.update(
                {
                    "qkv_w": _d(la.in_proj_qkv.weight),  # [1536,4096] fp8 (fold source)
                    "qkv_scales": _d(la.in_proj_qkv.weight_scale_inv),  # [12,32]
                    "z_w": _d(la.in_proj_z.weight),  # [1024,4096] fp8 (fold source)
                    "z_scales": _d(la.in_proj_z.weight_scale_inv),  # [8,32]
                    "b_w": _d(la.in_proj_b.weight),  # [8,4096] bf16 (fold source)
                    "a_w": _d(la.in_proj_a.weight),
                }
            )
        else:
            # sglang >= 0.5.15 fuses the projections: in_proj_qkvz is a
            # MergedColumnParallelLinear with output_sizes [k, k, v, v], so the
            # per-rank row order is [q|k|v (1536) ; z (1024)]; in_proj_ba merges
            # [b (8) ; a (8)]. Every segment boundary is 128-row aligned, so the
            # block-fp8 scale rows slice consistently ([12] qkv + [8] z). Row
            # slices are views -- aliasing is preserved.
            qkvz_w = _d(la.in_proj_qkvz.weight)  # [2560,4096] fp8
            qkvz_s = _d(la.in_proj_qkvz.weight_scale_inv)  # [20,32]
            ba_w = _d(la.in_proj_ba.weight)  # [16,4096] bf16
            out.update(
                {
                    "qkv_w": qkvz_w[:1536],
                    "qkv_scales": qkvz_s[:12],
                    "z_w": qkvz_w[1536:],
                    "z_scales": qkvz_s[12:],
                    "b_w": ba_w[:8],
                    "a_w": ba_w[8:],
                }
            )
        return out

    def head(self):
        return {
            "lm_head": _d(self.model.lm_head.weight),  # [31040,4096] bf16 rank shard
            "norm_w": _d(self.lm.norm.weight),
        }  # [4096] bf16

    @torch.no_grad()
    def embed(self, dev):
        shard = _d(self.lm.embed_tokens.weight)[: VOCAB // dist.get_world_size()].contiguous()
        full = torch.empty(VOCAB, HID, dtype=shard.dtype, device=dev)
        dist.all_gather_into_tensor(full, shard)  # rank order == vocab order
        return full.to(torch.bfloat16)
