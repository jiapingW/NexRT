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

"""SGLang-native DFlash prompt-context prefill for the bridge.

DFlash's draft attends to K/V rows that are projections of the TARGET's
features at every prompt position, not of tokens. SGLang's DFlash worker
materializes those rows during target prefill
(``DFlashWorkerV2._append_target_hidden_to_draft_kv_by_loc``): the captured
tap-layer features go through ``hidden_norm(fc(concat))``, then every draft
layer's K/V projection, K RMSNorm and RoPE run in one batched GEMM plus a
Triton kernel (``FusedKVMaterializeHelper``). The bridge uses that same model
code and the same SGLang-loaded weights through a draft ModelRunner
(``is_draft_worker=True``) and writes the rows straight into NexRT's draft
caches, so no draft KV pool, attention backend or CUDA graph is built.

SGLang keeps the complete ``fc`` on every rank and projects all taps at once,
so there is no cross-rank reduction in this path.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch

from nex_n2_5_pro.dflash.config import LAYERS, M, SLIDING_LAYERS, SLIDING_RING_CAPACITY


def write_context_rows(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    sliding: bool,
    positions: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    """Store one layer's prompt rows at their absolute positions.

    A sliding layer's cache is a ring of ``SLIDING_RING_CAPACITY`` rows; only the
    newest ring-capacity rows of a larger chunk are retained."""
    key = key.reshape(positions.numel(), -1)
    value = value.reshape(positions.numel(), -1)
    if sliding:
        offset = max(0, positions.numel() - SLIDING_RING_CAPACITY)
        indices = positions[offset:] % SLIDING_RING_CAPACITY
    else:
        offset, indices = 0, positions
    key_cache.index_copy_(0, indices, key[offset:])
    value_cache.index_copy_(0, indices, value[offset:])


class DFlashContextPrefill:
    """SGLang's DFlash draft model materializes the prompt context K/V rows."""

    def __init__(self, target, server_args, port_args, rank: int, ckpt: Path, block, device):
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.speculative.triton_ops.fused_kv_materialize import (
            FusedKVMaterializeHelper,
        )

        self.block = block
        sa = copy.copy(server_args)
        sa.speculative_algorithm = "DFLASH"
        sa.speculative_draft_model_path = str(ckpt)
        # The target may load from a sharded weight cache; the draft reads its
        # own checkpoint.
        sa.load_format = "auto"
        sa.model_loader_extra_config = {}
        moe_ep_rank = rank // (sa.tp_size // sa.ep_size)
        model_config = ModelConfig.from_server_args(
            sa,
            model_path=str(ckpt),
            is_draft_model=True,
            context_length=target.model_config.context_len,
        )
        if model_config.hf_config.architectures[0] != "DFlashDraftModel":
            raise RuntimeError(
                f"unexpected DFlash draft architecture {model_config.hf_config.architectures}"
            )
        # Weights only: the draft never runs a forward here, so its KV pool,
        # attention backend and graphs are not built.
        self.draft = ModelRunner(
            model_config=model_config,
            mem_fraction_static=sa.mem_fraction_static,
            gpu_id=rank,
            tp_rank=rank,
            tp_size=sa.tp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=sa.ep_size,
            pp_rank=0,
            pp_size=1,
            nccl_port=port_args.nccl_port,
            server_args=sa,
            is_draft_worker=True,
            req_to_token_pool=target.req_to_token_pool,
            token_to_kv_pool_allocator=target.token_to_kv_pool_allocator,
            memory_pool_config=target.memory_pool_config,
        )
        model = self.draft.model
        layers = model.layers
        attn = layers[0].self_attn
        if len(layers) != LAYERS or len(block.state.key_cache) != LAYERS:
            raise RuntimeError(f"draft has {len(layers)} layers, expected {LAYERS}")
        if attn.num_kv_heads != 1 or attn.head_dim != block.state.key_cache[0].shape[1]:
            raise RuntimeError(
                f"unexpected draft K/V shape: {attn.num_kv_heads} heads x {attn.head_dim}"
            )
        self.helper = FusedKVMaterializeHelper(
            layers=layers,
            rotary_emb=attn.rotary_emb,
            num_kv_heads=attn.num_kv_heads,
            head_dim=attn.head_dim,
            device=device,
            max_position_hint=target.model_config.context_len + M,
        )
        self.feature_width = int(model.fc.in_features)
        self.position = 0
        self.length = 0
        self._positions = None

    def begin(self, length: int) -> None:
        self.block.seed(length, context_start=0)
        self.position, self.length = 0, int(length)

    @torch.inference_mode()
    def consume(self, features: torch.Tensor, start: int, end: int) -> None:
        """Materialize prompt rows [start, end) from their concatenated tap features."""
        if start != self.position or not start < end <= self.length:
            raise ValueError("DFlash prefill chunks must cover the prompt in order")
        if tuple(features.shape) != (end - start, self.feature_width):
            raise ValueError(f"unexpected DFlash target feature shape: {tuple(features.shape)}")
        context = self.draft.model.project_target_hidden(features)
        self._positions = torch.arange(start, end, device=features.device, dtype=torch.int64)
        self.helper.materialize(context, self._positions, self._write_layer)
        self._positions = None
        self.position = end

    def _write_layer(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> None:
        state = self.block.state
        write_context_rows(
            state.key_cache[layer],
            state.value_cache[layer],
            layer < SLIDING_LAYERS,
            self._positions,
            key,
            value,
        )
