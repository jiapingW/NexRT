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

"""Checkpoint loading and TP=8 sharding for the nex_n2_5_pro backend."""

import torch
from safetensors import safe_open

from nex_n2_5_pro.config import LAYERS_DIR

TP = 8
BLOCK = 128
N_EXPERTS = 512


def _get_tensor(handle, key):
    """Accept both checkpoint names for FP8 dequantization multipliers."""
    if key.endswith(".weight_scale") and key not in handle.keys():
        key += "_inv"
    return handle.get_tensor(key)


def _layer_file(layer):
    return LAYERS_DIR / f"layer.{layer}.safetensors"


def load_embedding(device):
    with safe_open(
        str(LAYERS_DIR / "embed_tokens.safetensors"), framework="pt", device=str(device)
    ) as f:
        return f.get_tensor("model.language_model.embed_tokens.weight").to(torch.bfloat16)


def load_final_norm(device):
    with safe_open(str(LAYERS_DIR / "norm.safetensors"), framework="pt", device=str(device)) as f:
        return f.get_tensor("model.language_model.norm.weight").to(torch.bfloat16)


def load_lm_head(device):
    with safe_open(
        str(LAYERS_DIR / "lm_head.safetensors"), framework="pt", device=str(device)
    ) as f:
        return f.get_tensor("lm_head.weight").to(torch.bfloat16)


def load_layer_norms(layer, device):
    prefix = f"model.language_model.layers.{layer}."
    with safe_open(str(_layer_file(layer)), framework="pt", device=str(device)) as f:
        return (
            f.get_tensor(prefix + "input_layernorm.weight").to(torch.bfloat16),
            f.get_tensor(prefix + "post_attention_layernorm.weight").to(torch.bfloat16),
        )


def load_attention_weights(layer, device):
    prefix = f"model.language_model.layers.{layer}.self_attn."
    with safe_open(str(_layer_file(layer)), framework="pt", device=str(device)) as f:
        get = lambda name: _get_tensor(f, prefix + name)
        return {
            "q": get("q_proj.weight"),
            "qs": get("q_proj.weight_scale").float(),
            "k": get("k_proj.weight"),
            "ks": get("k_proj.weight_scale").float(),
            "v": get("v_proj.weight"),
            "vs": get("v_proj.weight_scale").float(),
            "o": get("o_proj.weight"),
            "os": get("o_proj.weight_scale").float(),
            "qn": get("q_norm.weight").to(torch.bfloat16),
            "kn": get("k_norm.weight").to(torch.bfloat16),
        }


def shard_attention_weights(weights, rank, tp=TP):
    if tp != TP:
        raise ValueError(f"the nex_n2_5_pro backend requires TP={TP}, got {tp}")
    head_dim, q_heads_local, kv_heads, q_gate = 256, 4, 2, 2
    ranks_per_kv = tp // kv_heads
    kv_head = rank // ranks_per_kv
    q_rows = q_heads_local * head_dim * q_gate
    kv_rows = head_dim
    o_cols = q_heads_local * head_dim
    q_blocks, kv_blocks, o_blocks = q_rows // BLOCK, kv_rows // BLOCK, o_cols // BLOCK
    return {
        "q": weights["q"][rank * q_rows : (rank + 1) * q_rows],
        "qs": weights["qs"][rank * q_blocks : (rank + 1) * q_blocks],
        "k": weights["k"][kv_head * kv_rows : (kv_head + 1) * kv_rows],
        "ks": weights["ks"][kv_head * kv_blocks : (kv_head + 1) * kv_blocks],
        "v": weights["v"][kv_head * kv_rows : (kv_head + 1) * kv_rows],
        "vs": weights["vs"][kv_head * kv_blocks : (kv_head + 1) * kv_blocks],
        "o": weights["o"][:, rank * o_cols : (rank + 1) * o_cols],
        "os": weights["os"][:, rank * o_blocks : (rank + 1) * o_blocks],
        "qn": weights["qn"],
        "kn": weights["kn"],
    }


def load_linear_attention_weights(layer, device):
    prefix = f"model.language_model.layers.{layer}.linear_attn."
    with safe_open(str(_layer_file(layer)), framework="pt", device=str(device)) as f:
        get = lambda name: _get_tensor(f, prefix + name)
        return {
            "qkv": get("in_proj_qkv.weight"),
            "qkvs": get("in_proj_qkv.weight_scale").float(),
            "z": get("in_proj_z.weight"),
            "zs": get("in_proj_z.weight_scale").float(),
            "o": get("out_proj.weight"),
            "os": get("out_proj.weight_scale").float(),
            "b": get("in_proj_b.weight").to(torch.bfloat16),
            "a": get("in_proj_a.weight").to(torch.bfloat16),
            "conv_w": get("conv1d.weight").to(torch.bfloat16),
            "A_log": get("A_log").float(),
            "dt_bias": get("dt_bias").float(),
            "norm_w": get("norm.weight").to(torch.bfloat16),
        }


def _gather_qkv(tensor, rank, key_rows, value_rows, key_offset, value_offset):
    query = tensor[rank * key_rows : (rank + 1) * key_rows]
    key = tensor[key_offset + rank * key_rows : key_offset + (rank + 1) * key_rows]
    value = tensor[value_offset + rank * value_rows : value_offset + (rank + 1) * value_rows]
    return torch.cat([query, key, value], dim=0)


def shard_linear_attention_weights(weights, rank, tp=TP):
    if tp != TP:
        raise ValueError(f"the nex_n2_5_pro backend requires TP={TP}, got {tp}")
    key_dim, value_dim, value_heads = 2048, 8192, 64
    value_rows = value_dim // tp
    value_blocks = value_rows // BLOCK
    value_heads_local = value_heads // tp
    key_rows = key_dim // tp
    return {
        "qkv": _gather_qkv(weights["qkv"], rank, key_rows, value_rows, key_dim, 2 * key_dim),
        "qkvs": _gather_qkv(
            weights["qkvs"],
            rank,
            key_rows // BLOCK,
            value_blocks,
            key_dim // BLOCK,
            2 * key_dim // BLOCK,
        ),
        "z": weights["z"][rank * value_rows : (rank + 1) * value_rows],
        "zs": weights["zs"][rank * value_blocks : (rank + 1) * value_blocks],
        "b": weights["b"][rank * value_heads_local : (rank + 1) * value_heads_local],
        "a": weights["a"][rank * value_heads_local : (rank + 1) * value_heads_local],
        "conv_w": _gather_qkv(weights["conv_w"], rank, key_rows, value_rows, key_dim, 2 * key_dim),
        "A_log": weights["A_log"][rank * value_heads_local : (rank + 1) * value_heads_local],
        "dt_bias": weights["dt_bias"][rank * value_heads_local : (rank + 1) * value_heads_local],
        "norm_w": weights["norm_w"],
        "o": weights["o"][:, rank * value_rows : (rank + 1) * value_rows],
        "os": weights["os"][:, rank * value_blocks : (rank + 1) * value_blocks],
    }


def load_moe_weights(layer, device):
    prefix = f"model.language_model.layers.{layer}.mlp."
    with safe_open(str(_layer_file(layer)), framework="pt", device=str(device)) as f:
        get = lambda name: _get_tensor(f, prefix + name)
        weights = {
            "gate_w": get("gate.weight").to(torch.bfloat16),
            "sgate_w": get("shared_expert_gate.weight").to(torch.bfloat16),
            "s_g1": get("shared_expert.gate_proj.weight"),
            "s_s1": get("shared_expert.gate_proj.weight_scale").float(),
            "s_g3": get("shared_expert.up_proj.weight"),
            "s_s3": get("shared_expert.up_proj.weight_scale").float(),
            "s_g2": get("shared_expert.down_proj.weight"),
            "s_s2": get("shared_expert.down_proj.weight_scale").float(),
        }
        e_g1, e_s1, e_g3, e_s3, e_g2, e_s2 = ([] for _ in range(6))
        for expert in range(N_EXPERTS):
            expert_prefix = f"{prefix}experts.{expert}."
            e_g1.append(f.get_tensor(expert_prefix + "gate_proj.weight"))
            e_s1.append(_get_tensor(f, expert_prefix + "gate_proj.weight_scale").float())
            e_g3.append(f.get_tensor(expert_prefix + "up_proj.weight"))
            e_s3.append(_get_tensor(f, expert_prefix + "up_proj.weight_scale").float())
            e_g2.append(f.get_tensor(expert_prefix + "down_proj.weight"))
            e_s2.append(_get_tensor(f, expert_prefix + "down_proj.weight_scale").float())
        weights.update(
            {
                "e_g1": torch.stack(e_g1),
                "e_s1": torch.stack(e_s1),
                "e_g3": torch.stack(e_g3),
                "e_s3": torch.stack(e_s3),
                "e_g2": torch.stack(e_g2),
                "e_s2": torch.stack(e_s2),
            }
        )
        return weights
