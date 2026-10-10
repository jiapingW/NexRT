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

"""Configuration for nexrt's nex_n2_5_pro backend.

Set these paths for the local installation:
  NEXRT_LAYERS_DIR  per-layer fp8 safetensors plus embedding, final norm, and lm head, produced by
                      tools/extract_layers.py.
  NEXRT_STATE_DIR   (used by decode_engine) the prefill state dir to seed decode from.

The placeholder defaults are intentionally machine-independent; set the environment variables before running.
"""

import os
from pathlib import Path

# --- paths (override via env) ---
LAYERS_DIR = Path(os.environ.get("NEXRT_LAYERS_DIR", "/path/to/nexrt-layers"))

# --- model geometry (nex_n2_5_pro; a hybrid GDN + full-attn MoE) ---
HID = 4096  # hidden size
HEAD_DIM = 256  # attention head dim
N_HEADS = 32  # query heads (global)
N_KV = 2  # KV heads (global; one per 4-rank CP group under TP=8)
VOCAB = 248320  # lm_head vocab
N_LAYERS = 60  # decoder layers (45 GDN linear-attn + 15 full-attn, all MoE)
FULL_ATTN_INTERVAL = 4  # layer L is full-attn iff (L+1) % FULL_ATTN_INTERVAL == 0, else GDN
EPS = 1e-6  # rms_norm_eps (input_ln + q/k norm)

# The public runtime defaults to the complete supported model. A smaller prefix remains useful for focused tests.
ACTIVE_N_LAYERS = int(os.environ.get("NEXRT_NLAYERS", str(N_LAYERS)))
if not 1 <= ACTIVE_N_LAYERS <= N_LAYERS:
    raise ValueError(f"NEXRT_NLAYERS must be in [1, {N_LAYERS}], got {ACTIVE_N_LAYERS}")
LAYERS = tuple(range(ACTIVE_N_LAYERS))
LAYER_IS_LINEAR = {layer: (layer + 1) % FULL_ATTN_INTERVAL != 0 for layer in LAYERS}
