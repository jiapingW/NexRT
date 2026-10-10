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

"""Packed attention-front state and launch ABI for all six DFlash layers.

Draft QKV and target-context KV are deliberately separate graph nodes: they
consume different residual streams and fold different RMSNorm weights. Their
post kernel is shared and performs the dynamic-row position-controlled cache
publication.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent

from nex_n2_5_pro.kernel_build import build_kernel  # noqa: E402
from nex_n2_5_pro.dflash.config import DFlashConfig, LAYERS, M  # noqa: E402
from nex_n2_5_pro.dflash.weights import (  # noqa: E402
    CONTEXT_KV_ROWS,
    QKV_ROWS,
    DFlashWeightShard,
    pack_attention_front,
)

QUERY_HEADS_PER_RANK = 4


class DFlashAttentionFront:
    """Own immutable projection packs and reusable front-kernel scratch."""

    def __init__(
        self,
        config: DFlashConfig,
        weights: DFlashWeightShard,
        device: torch.device,
        cache_capacity: int,
        status: torch.Tensor,
    ) -> None:
        self.config = config
        self.device = device
        self.cache_capacity = int(cache_capacity)
        kernel_dir = HERE.parent / "kernels" / "dflash"
        self.library = ctypes.CDLL(
            str(build_kernel(kernel_dir, "nex_n2_5_pro_dflash_qkv_ckv_front_ctypes"))
        )
        void_p, int_t, float_t = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
        self.library.NexN25ProDFlashQkvProjLaunch.argtypes = [void_p] * 4 + [float_t, void_p]
        self.library.NexN25ProDFlashContextKvProjLaunch.argtypes = [void_p] * 4 + [float_t, void_p]
        self.library.NexN25ProDFlashQkvCkvPostRowsCtlLaunch.argtypes = [void_p] * 13 + [
            int_t,
            float_t,
            void_p,
        ]
        self.library.NexN25ProDFlashQkvCkvPostRowsCtlRingLaunch.argtypes = [void_p] * 13 + [
            int_t,
            int_t,
            float_t,
            void_p,
        ]

        self.layers: list[dict[str, torch.Tensor]] = []
        for layer in weights.layers:
            qkv_pack, context_kv_pack = pack_attention_front(layer, weights.hidden_norm)
            self.layers.append(
                {
                    "qkv_projection": qkv_pack,
                    "context_kv_projection": context_kv_pack,
                    "query_norm": layer.query_norm,
                    "key_norm": layer.key_norm,
                }
            )
        if len(self.layers) != LAYERS:
            raise ValueError(f"DFlash requires {LAYERS} attention layers")

        # A single scratch set is reused by the serial six-layer graph chain.
        self.qkv_partial = torch.empty(2, M, QKV_ROWS, dtype=torch.float32, device=device)
        self.context_kv_partial = torch.empty(
            2, M, CONTEXT_KV_ROWS, dtype=torch.float32, device=device
        )
        self.attention_scale = torch.empty(M, dtype=torch.float32, device=device)
        self.hidden_scale = torch.empty(M, dtype=torch.float32, device=device)
        if status.dtype != torch.int32 or status.numel() < 1 or status.device != device:
            raise ValueError("status must provide an int32 word on the DFlash device")
        self.status = status

    @staticmethod
    def _pointer(tensor: torch.Tensor) -> ctypes.c_void_p:
        return ctypes.c_void_p(tensor.data_ptr())

    def launch_qkv_projection(
        self, layer: int, draft_raw: torch.Tensor, stream: ctypes.c_void_p
    ) -> int:
        packed = self.layers[layer]
        return self.library.NexN25ProDFlashQkvProjLaunch(
            self._pointer(draft_raw),
            self._pointer(packed["qkv_projection"]),
            self._pointer(self.qkv_partial),
            self._pointer(self.attention_scale),
            ctypes.c_float(self.config.attn_rms_eps),
            stream,
        )

    def launch_context_kv_projection(
        self, layer: int, context_raw: torch.Tensor, stream: ctypes.c_void_p
    ) -> int:
        packed = self.layers[layer]
        return self.library.NexN25ProDFlashContextKvProjLaunch(
            self._pointer(context_raw),
            self._pointer(packed["context_kv_projection"]),
            self._pointer(self.context_kv_partial),
            self._pointer(self.hidden_scale),
            ctypes.c_float(self.config.hidden_rms_eps),
            stream,
        )

    def launch_post(
        self,
        layer: int,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        query_out: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        metadata: torch.Tensor,
        stream: ctypes.c_void_p,
        ring_capacity: int = 0,
    ) -> int:
        packed = self.layers[layer]
        args = (
            self._pointer(self.qkv_partial),
            self._pointer(self.context_kv_partial),
            self._pointer(self.attention_scale),
            self._pointer(self.hidden_scale),
            self._pointer(packed["query_norm"]),
            self._pointer(packed["key_norm"]),
            self._pointer(cos_table),
            self._pointer(sin_table),
            self._pointer(query_out),
            self._pointer(key_cache),
            self._pointer(value_cache),
            self._pointer(self.status),
            self._pointer(metadata),
        )
        if ring_capacity:
            return self.library.NexN25ProDFlashQkvCkvPostRowsCtlRingLaunch(
                *args,
                ctypes.c_int(self.cache_capacity),
                ctypes.c_int(ring_capacity),
                ctypes.c_float(self.config.qk_norm_eps),
                stream,
            )
        return self.library.NexN25ProDFlashQkvCkvPostRowsCtlLaunch(
            *args,
            ctypes.c_int(self.cache_capacity),
            ctypes.c_float(self.config.qk_norm_eps),
            stream,
        )
