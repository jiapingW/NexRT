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

"""CPU-only guard for fixed-M projection terminology and ABI names."""

from __future__ import annotations

from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

OLD_STEMS = (
    "nex_n2_5_pro_m4_lmhead_gemv_ctypes",
    "nex_n2_5_pro_m8_lmhead_gemv_ctypes",
    "nex_n2_5_pro_m4_gate_gemv_ctypes",
    "nex_n2_5_pro_m8_gate_gemv_ctypes",
)
NEW_STEMS = tuple(stem.replace("gemv", "gemm") for stem in OLD_STEMS)
# The M4 gate GEMM translation unit includes this GEMV body for its raw/RMS CTAs.
RETAINED_SOURCES = ("nex_n2_5_pro_m4_gate_gemv_ctypes",)
OLD_IDENTIFIERS = (
    "NexN25ProM4LmHeadGemv",
    "NexN25ProM8LmHeadGemv",
    "NexN25ProM4GateGemv",
    "NexN25ProM8GateGemv",
    "M4LmHeadGemvKernel",
    "M4GateGemvKernel",
    "M8GateGemvKernel",
    "pack_gemv_bf16tc",
)
NEW_IDENTIFIERS = (
    "NexN25ProM4LmHeadGemmLaunch",
    "NexN25ProM8LmHeadGemmTc16Launch",
    "NexN25ProM4GateGemmLaunch",
    "NexN25ProM8GateGemmLaunch",
    "M4LmHeadGemmKernel",
    "M4GateGemmKernel",
    "M8GateGemmKernel",
    "pack_gemm_bf16tc",
    "pack_gemm_bf16tc_m8",
)


def test_fixed_m_projection_names() -> None:
    this_file = Path(__file__).resolve()
    sources = tuple(
        source
        for pattern in ("*.py", "*.cu")
        for source in ROOT.rglob(pattern)
        if source.resolve() != this_file and source.stem not in RETAINED_SOURCES
    )
    names = {source.stem for source in sources}
    for stem in OLD_STEMS:
        assert stem not in names, stem
    for stem in NEW_STEMS:
        assert stem in names, stem

    text = "\n".join(source.read_text(encoding="utf-8") for source in sources)
    for identifier in OLD_IDENTIFIERS:
        assert identifier not in text, identifier
    for identifier in NEW_IDENTIFIERS:
        assert identifier in text, identifier


if __name__ == "__main__":
    test_fixed_m_projection_names()
    print("BATCH GEMM NAMING PASS", flush=True)
