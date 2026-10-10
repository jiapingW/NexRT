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

"""Sampling parameters and the fixed 64-byte device-buffer layout."""

import json
import math
import operator
import os
import struct
from pathlib import Path

MAX_TOP_K = 28
SAMPLING_WORDS = 8
DEFAULT_SAMPLING = (0.6, 0.95, 20)


def checked_sampling_config(temperature, top_p, top_k):
    temperature, top_p = float(temperature), float(top_p)
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and >= 0")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be finite and in (0, 1]")
    if isinstance(top_k, bool):
        raise ValueError("top_k must be an integer in [1, 28]")
    try:
        top_k = operator.index(top_k)
    except TypeError as error:
        raise ValueError("top_k must be an integer in [1, 28]") from error
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError("top_k must be in [1, 28]")
    try:
        stored_t, stored_p = struct.unpack("<ff", struct.pack("<ff", temperature, top_p))
    except (OverflowError, struct.error) as error:
        raise ValueError("sampling parameters must fit float32") from error
    if (temperature > 0 and stored_t == 0) or stored_p == 0:
        raise ValueError("sampling parameters must not underflow float32")
    return stored_t, stored_p, top_k


def resolve_sampling_config(defaults, temperature=None, top_p=None, top_k=None):
    values = (temperature, top_p, top_k)
    return checked_sampling_config(
        *(default if value is None else value for default, value in zip(defaults, values))
    )


def sampling_defaults():
    model_dir = os.environ.get("NEXRT_MODEL_DIR") or os.environ.get("NEXRT_LAYERS_DIR")
    config = {}
    if model_dir:
        path = Path(model_dir) / "generation_config.json"
        if path.is_file():
            config = json.loads(path.read_text())
    temperature = config.get("temperature", DEFAULT_SAMPLING[0])
    top_p = config.get("top_p", DEFAULT_SAMPLING[1])
    top_k = config.get("top_k", DEFAULT_SAMPLING[2])
    if config.get("do_sample") is False:
        temperature, top_k = 0.0, 1
    return checked_sampling_config(temperature, top_p, top_k)


def sampling_words(seed, temperature, top_p, top_k):
    """Pack fixed words: seed[0], float32 temperature/top_p[1], top_k[2]."""
    temperature, top_p, top_k = checked_sampling_config(temperature, top_p, top_k)
    seed = int(seed) & ((1 << 64) - 1)
    signed_seed = seed if seed < (1 << 63) else seed - (1 << 64)
    tp_bits = struct.unpack("<q", struct.pack("<ff", temperature, top_p))[0]
    return [signed_seed, tp_bits, top_k] + [0] * (SAMPLING_WORDS - 3)
