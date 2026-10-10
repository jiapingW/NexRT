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

"""Rank-local DFlash checkpoint loading and immutable kernel prepacks.

Only the TP=8 shard needed by the current rank is materialized from the draft
checkpoint. Large projection packs are built once before graph capture; no
packing or allocation occurs in a replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from nex_n2_5_pro.dflash.config import (
    DFLASH_WEIGHTS,
    HEAD_DIM,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    LAYERS,
    TP,
)

BF16 = torch.bfloat16
QUERY_HEADS_PER_RANK = 4
QUERY_ROWS_PER_RANK = QUERY_HEADS_PER_RANK * HEAD_DIM
INTERMEDIATE_PER_RANK = INTERMEDIATE_SIZE // TP
QKV_ROWS = QUERY_ROWS_PER_RANK + 2 * HEAD_DIM
CONTEXT_KV_ROWS = 2 * HEAD_DIM
TILE = 16
KSTEP = 16
CHUNK_STEPS = 4

# Fixed production forms. These are constants, not runtime selectors.
HEAD_PDL = 0
MLP_CONFIG = (5, 2, 5, 2, 2, 1)  # gate depth/split, down depth/split/tiles, PDL
OPROJ_CONFIG = (5, 0)  # cp.async depth, PDL


@dataclass(frozen=True)
class DFlashLayerWeights:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    output: torch.Tensor
    query_norm: torch.Tensor
    key_norm: torch.Tensor
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor


@dataclass(frozen=True)
class DFlashWeightShard:
    target_hidden_projection: torch.Tensor
    hidden_norm: torch.Tensor
    final_norm: torch.Tensor
    layers: tuple[DFlashLayerWeights, ...]


def _shape(handle, key: str) -> tuple[int, ...]:
    return tuple(int(x) for x in handle.get_slice(key).get_shape())


def _require_shape(handle, key: str, expected: tuple[int, ...]) -> None:
    actual = _shape(handle, key)
    if actual != expected:
        raise ValueError(f"{key} must have shape {expected}, got {actual}")


def _require_bf16(tensor: torch.Tensor, key: str) -> torch.Tensor:
    if tensor.dtype != BF16:
        raise TypeError(f"{key} must be BF16, got {tensor.dtype}")
    return tensor.contiguous()


def load_dflash_shard(
    rank: int,
    device: torch.device,
    weights_path: str | Path = DFLASH_WEIGHTS,
) -> DFlashWeightShard:
    """Load one rank's TP=8 draft weights.

    The target LM-head belongs to the target engine and is packed once by its
    M=8 verifier; it is deliberately absent from the draft checkpoint shard.
    """
    rank = int(rank)
    if not 0 <= rank < TP:
        raise ValueError(f"DFlash requires a TP={TP} rank in [0,{TP}), got {rank}")

    path = Path(weights_path)
    row_q0 = rank * QUERY_ROWS_PER_RANK
    row_kv0 = rank * HEAD_DIM
    row_inter0 = rank * INTERMEDIATE_PER_RANK
    col_hidden0 = rank * HIDDEN_SIZE
    col_query0 = rank * QUERY_ROWS_PER_RANK
    col_inter0 = rank * INTERMEDIATE_PER_RANK

    with safe_open(str(path), framework="pt", device=str(device)) as handle:
        _require_shape(handle, "fc.weight", (HIDDEN_SIZE, TP * HIDDEN_SIZE))
        _require_shape(handle, "hidden_norm.weight", (HIDDEN_SIZE,))
        _require_shape(handle, "norm.weight", (HIDDEN_SIZE,))
        target_hidden_projection = _require_bf16(
            handle.get_slice("fc.weight")[:, col_hidden0 : col_hidden0 + HIDDEN_SIZE],
            "fc.weight shard",
        )
        hidden_norm = _require_bf16(handle.get_tensor("hidden_norm.weight"), "hidden_norm.weight")
        final_norm = _require_bf16(handle.get_tensor("norm.weight"), "norm.weight")

        layers: list[DFlashLayerWeights] = []
        for layer in range(LAYERS):
            prefix = f"layers.{layer}."
            attention = prefix + "self_attn."
            expected = {
                attention + "q_proj.weight": (TP * QUERY_ROWS_PER_RANK, HIDDEN_SIZE),
                attention + "k_proj.weight": (TP * HEAD_DIM, HIDDEN_SIZE),
                attention + "v_proj.weight": (TP * HEAD_DIM, HIDDEN_SIZE),
                attention + "o_proj.weight": (HIDDEN_SIZE, TP * QUERY_ROWS_PER_RANK),
                attention + "q_norm.weight": (HEAD_DIM,),
                attention + "k_norm.weight": (HEAD_DIM,),
                prefix + "input_layernorm.weight": (HIDDEN_SIZE,),
                prefix + "post_attention_layernorm.weight": (HIDDEN_SIZE,),
                prefix + "mlp.gate_proj.weight": (INTERMEDIATE_SIZE, HIDDEN_SIZE),
                prefix + "mlp.up_proj.weight": (INTERMEDIATE_SIZE, HIDDEN_SIZE),
                prefix + "mlp.down_proj.weight": (HIDDEN_SIZE, INTERMEDIATE_SIZE),
            }
            for key, shape in expected.items():
                _require_shape(handle, key, shape)

            def rows(key: str, start: int, count: int) -> torch.Tensor:
                return _require_bf16(
                    handle.get_slice(key)[start : start + count, :], key + " shard"
                )

            def columns(key: str, start: int, count: int) -> torch.Tensor:
                return _require_bf16(
                    handle.get_slice(key)[:, start : start + count], key + " shard"
                )

            layers.append(
                DFlashLayerWeights(
                    query=rows(attention + "q_proj.weight", row_q0, QUERY_ROWS_PER_RANK),
                    key=rows(attention + "k_proj.weight", row_kv0, HEAD_DIM),
                    value=rows(attention + "v_proj.weight", row_kv0, HEAD_DIM),
                    output=columns(attention + "o_proj.weight", col_query0, QUERY_ROWS_PER_RANK),
                    query_norm=_require_bf16(
                        handle.get_tensor(attention + "q_norm.weight"), attention + "q_norm.weight"
                    ),
                    key_norm=_require_bf16(
                        handle.get_tensor(attention + "k_norm.weight"), attention + "k_norm.weight"
                    ),
                    input_norm=_require_bf16(
                        handle.get_tensor(prefix + "input_layernorm.weight"),
                        prefix + "input_layernorm.weight",
                    ),
                    post_attention_norm=_require_bf16(
                        handle.get_tensor(prefix + "post_attention_layernorm.weight"),
                        prefix + "post_attention_layernorm.weight",
                    ),
                    gate=rows(prefix + "mlp.gate_proj.weight", row_inter0, INTERMEDIATE_PER_RANK),
                    up=rows(prefix + "mlp.up_proj.weight", row_inter0, INTERMEDIATE_PER_RANK),
                    down=columns(
                        prefix + "mlp.down_proj.weight", col_inter0, INTERMEDIATE_PER_RANK
                    ),
                )
            )

    return DFlashWeightShard(
        target_hidden_projection=target_hidden_projection,
        hidden_norm=hidden_norm,
        final_norm=final_norm,
        layers=tuple(layers),
    )


def _require_matrix(weight: torch.Tensor, rows: int, columns: int, name: str) -> None:
    if weight.dtype != BF16 or tuple(weight.shape) != (rows, columns):
        raise ValueError(
            f"{name} must be BF16 [{rows},{columns}], got {weight.dtype} {tuple(weight.shape)}"
        )


def _require_gamma(gamma: torch.Tensor, width: int, name: str) -> None:
    if gamma.dtype != BF16 or tuple(gamma.shape) != (width,):
        raise ValueError(f"{name} must be BF16 [{width}], got {gamma.dtype} {tuple(gamma.shape)}")


@torch.no_grad()
def fold_rmsnorm_weight(weight: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Fold a column-wise RMSNorm gamma into a BF16 projection in FP32."""
    if weight.ndim != 2:
        raise ValueError(f"weight must be a matrix, got {tuple(weight.shape)}")
    _require_gamma(gamma, weight.shape[1], "gamma")
    return (weight.float() * gamma.float().unsqueeze(0)).to(BF16).contiguous()


@torch.no_grad()
def pack_m8_bf16(weight: torch.Tensor) -> torch.Tensor:
    """Pack BF16 weights for M=8 with all 16 MMA A rows populated."""
    if weight.dtype != BF16 or weight.ndim != 2:
        raise ValueError(f"weight must be a BF16 matrix, got {weight.dtype}")
    rows, columns = weight.shape
    if rows % TILE or columns % KSTEP:
        raise ValueError(f"weight shape must be divisible by 16, got {(rows, columns)}")
    return (
        weight.contiguous()
        .view(rows // 16, 2, 8, columns // 16, 2, 4, 2)
        .permute(0, 3, 2, 5, 4, 1, 6)
        .contiguous()
        .view(-1)
    )


@torch.no_grad()
def pack_attention_front(
    layer: DFlashLayerWeights, hidden_norm: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return separate draft-QKV and target-context-KV packs for one layer."""
    _require_matrix(layer.query, QUERY_ROWS_PER_RANK, HIDDEN_SIZE, "query")
    _require_matrix(layer.key, HEAD_DIM, HIDDEN_SIZE, "key")
    _require_matrix(layer.value, HEAD_DIM, HIDDEN_SIZE, "value")
    _require_gamma(layer.input_norm, HIDDEN_SIZE, "input_norm")
    _require_gamma(hidden_norm, HIDDEN_SIZE, "hidden_norm")
    qkv = torch.cat((layer.query, layer.key, layer.value), dim=0)
    context_kv = torch.cat((layer.key, layer.value), dim=0)
    return (
        pack_m8_bf16(fold_rmsnorm_weight(qkv, layer.input_norm)),
        pack_m8_bf16(fold_rmsnorm_weight(context_kv, hidden_norm)),
    )


def _fragment_pack(weight: torch.Tensor) -> torch.Tensor:
    """[O,K] BF16 -> [O/16][K/16][32 lanes][8 BF16] MMA fragments."""
    output, inner = weight.shape
    if output % TILE or inner % KSTEP or weight.dtype != BF16:
        raise ValueError(f"invalid MMA weight shape {weight.dtype} {tuple(weight.shape)}")
    tiled = weight.view(output // TILE, TILE, inner // KSTEP, KSTEP).permute(0, 2, 1, 3)
    group = torch.arange(8, device=weight.device).repeat_interleave(4)
    column = torch.arange(4, device=weight.device).repeat(8)
    row_index = torch.stack(
        [group, group, group + 8, group + 8, group, group, group + 8, group + 8], 1
    )
    inner_index = torch.stack(
        [
            2 * column,
            2 * column + 1,
            2 * column,
            2 * column + 1,
            2 * column + 8,
            2 * column + 9,
            2 * column + 8,
            2 * column + 9,
        ],
        1,
    )
    return tiled[:, :, row_index, inner_index]


def pack_gate_up(gate: torch.Tensor, up: torch.Tensor, ksplit: int = 2) -> torch.Tensor:
    """Pack the two rank-local [1536,4096] MLP input projections."""
    _require_matrix(gate, INTERMEDIATE_PER_RANK, HIDDEN_SIZE, "gate")
    _require_matrix(up, INTERMEDIATE_PER_RANK, HIDDEN_SIZE, "up")
    gate_frag, up_frag = _fragment_pack(gate), _fragment_pack(up)
    output_tiles, steps = gate_frag.shape[:2]
    if steps % (ksplit * CHUNK_STEPS):
        raise ValueError("gate/up K steps do not divide the fixed split geometry")
    stacked = torch.stack((gate_frag, up_frag), 0)
    packed = stacked.view(
        2, output_tiles, ksplit, steps // ksplit // CHUNK_STEPS, CHUNK_STEPS, 32, 8
    )
    return packed.permute(1, 3, 0, 2, 4, 5, 6).contiguous()


def pack_down(weight: torch.Tensor, ksplit: int = 2, tiles: int = 2) -> torch.Tensor:
    """Pack a rank-local [4096,1536] MLP output projection."""
    _require_matrix(weight, HIDDEN_SIZE, INTERMEDIATE_PER_RANK, "down")
    fragments = _fragment_pack(weight)
    output_tiles, steps = fragments.shape[:2]
    if output_tiles % tiles or steps % (ksplit * CHUNK_STEPS):
        raise ValueError("down projection does not divide the fixed tile geometry")
    packed = fragments.view(
        output_tiles // tiles, tiles, ksplit, steps // ksplit // CHUNK_STEPS, CHUNK_STEPS, 32, 8
    )
    return packed.permute(0, 3, 1, 2, 4, 5, 6).contiguous()


def pack_output_projection(weight: torch.Tensor, ksplit: int = 2, tiles: int = 2) -> torch.Tensor:
    """Pack one rank-local [4096,512] attention output projection."""
    _require_matrix(weight, HIDDEN_SIZE, QUERY_ROWS_PER_RANK, "output")
    fragments = _fragment_pack(weight)
    output_tiles, steps = fragments.shape[:2]
    if output_tiles % tiles or steps % (ksplit * CHUNK_STEPS):
        raise ValueError("output projection does not divide the fixed tile geometry")
    packed = fragments.view(
        output_tiles // tiles, tiles, ksplit, steps // ksplit // CHUNK_STEPS, CHUNK_STEPS, 32, 8
    )
    return packed.permute(0, 3, 1, 2, 4, 5, 6).contiguous()


def pack_target_hidden_projection(weight: torch.Tensor) -> torch.Tensor:
    """Pack the rank-local [4096,4096] target-hidden projection shard."""
    _require_matrix(weight, HIDDEN_SIZE, HIDDEN_SIZE, "target_hidden_projection")
    fragments = _fragment_pack(weight)
    packed = fragments.view(HIDDEN_SIZE // TILE // 2, 2, 2, HIDDEN_SIZE // KSTEP // 8, 4, 32, 8)
    return packed.permute(0, 3, 1, 2, 4, 5, 6).contiguous()


def pack_tc16(weight: torch.Tensor) -> torch.Tensor:
    """Pack a generic BF16 [rows,K] matrix for the tc16 M=8 layout."""
    rows, inner = weight.shape
    if rows % 16 or inner % 16 or weight.dtype != BF16:
        raise ValueError(f"invalid tc16 matrix {weight.dtype} {tuple(weight.shape)}")
    device = weight.device
    tile = torch.arange(rows // 16, device=device).view(-1, 1, 1, 1, 1)
    k16 = torch.arange(inner // 16, device=device).view(1, -1, 1, 1, 1)
    lane = torch.arange(32, device=device).view(1, 1, 32, 1, 1)
    fragment = torch.arange(4, device=device).view(1, 1, 1, 4, 1)
    element = torch.arange(2, device=device).view(1, 1, 1, 1, 2)
    group, thread_in_group = lane >> 2, lane & 3
    row = tile * 16 + (fragment & 1) * 8 + group
    column = k16 * 16 + (fragment >> 1) * 8 + thread_in_group * 2 + element
    return weight.reshape(-1)[(row * inner + column).reshape(-1)].contiguous()
