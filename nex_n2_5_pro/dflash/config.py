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

"""Configuration and fixed geometry for Nex-N2.5-Pro DFlash block-8 decoding."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path


DFLASH_DIR = Path(os.environ.get("NEXRT_DFLASH_DIR", "/path/to/nexrt-dflash"))
DFLASH_CONFIG = Path(os.environ.get("NEXRT_DFLASH_CONFIG", str(DFLASH_DIR / "config.json")))
DFLASH_WEIGHTS = Path(os.environ.get("NEXRT_DFLASH_WEIGHTS", str(DFLASH_DIR / "model.safetensors")))

TP = 8
M = 8
HIDDEN_SIZE = 4096
HEAD_DIM = 128
QUERY_HEADS = 32
KV_HEADS = 8
LAYERS = 6
INTERMEDIATE_SIZE = 12288
SLIDING_LAYERS = 5
SLIDING_WINDOW = 4096
SLIDING_RING_CAPACITY = 8192
TARGET_TAPS = 8
VOCAB_SIZE = 248320
VOCAB_ROWS_PER_RANK = VOCAB_SIZE // TP
MAX_POSITION_EMBEDDINGS = 262144
CACHE_SLACK = 1024
RMS_NORM_EPS = 1.0e-6


@dataclass(frozen=True)
class DFlashConfig:
    """The checkpoint fields consumed by the fixed TP=8 DFlash kernels."""

    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    attn_rms_eps: float
    hidden_rms_eps: float
    qk_norm_eps: float
    rope_theta: float
    sliding_window: int
    layer_types: tuple[str, ...]
    max_position_embeddings: int
    vocab_size: int
    block_size: int
    mask_token_id: int
    target_layer_ids: tuple[int, ...]
    eos_token_id: int | None

    @classmethod
    def load(cls, path: str | Path = DFLASH_CONFIG) -> "DFlashConfig":
        source = Path(path)
        data = json.loads(source.read_text(encoding="utf-8"))
        dflash = data["dflash_config"]
        rope = data["rope_parameters"]
        eps = float(data["rms_norm_eps"])
        config = cls(
            hidden_size=int(data["hidden_size"]),
            num_layers=int(data["num_hidden_layers"]),
            num_heads=int(data["num_attention_heads"]),
            num_kv_heads=int(data["num_key_value_heads"]),
            head_dim=int(data["head_dim"]),
            intermediate_size=int(data["intermediate_size"]),
            rms_norm_eps=eps,
            attn_rms_eps=float(data.get("attn_rms_eps", eps)),
            hidden_rms_eps=float(data.get("hidden_rms_eps", eps)),
            qk_norm_eps=float(data.get("qk_norm_eps", eps)),
            rope_theta=float(rope["rope_theta"]),
            sliding_window=int(data["sliding_window"]),
            layer_types=tuple(data["layer_types"]),
            max_position_embeddings=int(data["max_position_embeddings"]),
            vocab_size=int(data["vocab_size"]),
            block_size=int(dflash["block_size"]),
            mask_token_id=int(dflash["mask_token_id"]),
            target_layer_ids=tuple(int(layer) for layer in dflash["target_layer_ids"]),
            eos_token_id=(None if data.get("eos_token_id") is None else int(data["eos_token_id"])),
        )
        config.validate_kernel_contract(source)
        return config

    @property
    def cache_capacity(self) -> int:
        return self.max_position_embeddings + CACHE_SLACK

    def validate_kernel_contract(self, source: str | Path = "DFlash config") -> None:
        expected = {
            "hidden_size": HIDDEN_SIZE,
            "num_layers": LAYERS,
            "num_heads": QUERY_HEADS,
            "num_kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "intermediate_size": INTERMEDIATE_SIZE,
            "sliding_window": SLIDING_WINDOW,
            "vocab_size": VOCAB_SIZE,
            "rms_norm_eps": RMS_NORM_EPS,
            "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
        }
        drift: dict[str, object] = {
            name: (getattr(self, name), value)
            for name, value in expected.items()
            if getattr(self, name) != value
        }
        expected_layers = ("sliding_attention",) * SLIDING_LAYERS + ("full_attention",)
        if self.layer_types != expected_layers:
            drift["layer_types"] = (self.layer_types, expected_layers)
        if len(self.target_layer_ids) != TARGET_TAPS:
            drift["target_layer_ids length"] = (len(self.target_layer_ids), TARGET_TAPS)
        elif len(set(self.target_layer_ids)) != TARGET_TAPS:
            drift["target_layer_ids uniqueness"] = self.target_layer_ids
        if self.block_size < M:
            drift["block_size minimum"] = (self.block_size, M)
        for name in ("attn_rms_eps", "hidden_rms_eps", "qk_norm_eps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                drift[name] = (value, "finite and positive")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0.0:
            drift["rope_theta"] = (self.rope_theta, "finite and positive")
        if not (0 <= self.mask_token_id < self.vocab_size):
            drift["mask_token_id range"] = (self.mask_token_id, f"[0,{self.vocab_size})")
        if self.eos_token_id is not None and not 0 <= self.eos_token_id < self.vocab_size:
            drift["eos_token_id range"] = (self.eos_token_id, f"[0,{self.vocab_size})")
        if drift:
            raise ValueError(
                f"{source} does not match the fixed Nex-N2.5-Pro DFlash kernel contract: {drift}"
            )


def round_positions(output_count: int, committed_len: int) -> tuple[range, range]:
    """Return the context and draft absolute-position ranges for one round."""
    output_count = int(output_count)
    committed_len = int(committed_len)
    if not 0 <= output_count <= M:
        raise ValueError(f"output_count must be in [0,{M}], got {output_count}")
    if committed_len < output_count:
        raise ValueError(f"committed_len={committed_len} is below output_count={output_count}")
    return (
        range(committed_len - output_count, committed_len),
        range(committed_len, committed_len + M),
    )
