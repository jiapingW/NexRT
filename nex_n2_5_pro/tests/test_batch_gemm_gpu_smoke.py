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

"""Build and launch the public M=4/M=8 batched GEMM kernel families."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from kernel_build import build_kernel  # noqa: E402


DIM = 4096
HEAD_ROWS = 31040
GATE_ROWS = 528
GATE_ACTIVE_ROWS = 513


def _pointer(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def _load(directory: str, stem: str, symbol: str, argtypes: list[type]):
    library = ctypes.CDLL(str(build_kernel(ROOT / "kernels" / directory, stem)))
    launch = getattr(library, symbol)
    launch.argtypes = argtypes
    launch.restype = ctypes.c_int
    return library, launch


def _check(code: int, label: str) -> None:
    if code:
        raise RuntimeError(f"{label} launch failed with CUDA error {code}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("the batched GEMM smoke test requires SM90")

    void_p = ctypes.c_void_p
    float_t = ctypes.c_float
    long_long = ctypes.c_longlong
    stream = void_p(torch.cuda.current_stream().cuda_stream)

    libraries = []
    library, head4 = _load(
        "head",
        "nex_n2_5_pro_m4_lmhead_gemm_ctypes",
        "NexN25ProM4LmHeadGemmLaunch",
        [void_p, void_p, void_p, void_p, float_t, void_p],
    )
    libraries.append(library)
    library, head8 = _load(
        "head",
        "nex_n2_5_pro_m8_lmhead_gemm_ctypes",
        "NexN25ProM8LmHeadGemmTc16Launch",
        [void_p, void_p, void_p, void_p],
    )
    libraries.append(library)
    library, gate4 = _load(
        "moe",
        "nex_n2_5_pro_m4_gate_gemm_ctypes",
        "NexN25ProM4GateGemmLaunch",
        [void_p, void_p, void_p, void_p, float_t, void_p, long_long, void_p],
    )
    libraries.append(library)
    library, gate8 = _load(
        "moe",
        "nex_n2_5_pro_m8_gate_gemm_ctypes",
        "NexN25ProM8GateGemmLaunch",
        [void_p, void_p, void_p, void_p, float_t, void_p, long_long, void_p],
    )
    libraries.append(library)

    # Both M families consume the same lane-major layouts. M=8 head GEMM accepts
    # already-normalized rows; M=4 applies inverse RMS after the matrix product.
    torch.manual_seed(20260903)
    head_weight = torch.empty(HEAD_ROWS * DIM, dtype=torch.bfloat16, device="cuda")
    head_weight.uniform_(-0.02, 0.02)
    gate_weight = torch.empty(GATE_ROWS * DIM, dtype=torch.bfloat16, device="cuda")
    gate_weight.uniform_(-0.02, 0.02)
    hidden8 = torch.empty(8, DIM, dtype=torch.bfloat16, device="cuda")
    hidden8.uniform_(-1.0, 1.0)
    logits4 = torch.full((4, HEAD_ROWS), torch.nan, dtype=torch.float32, device="cuda")
    logits8 = torch.full((8, HEAD_ROWS), torch.nan, dtype=torch.float32, device="cuda")
    inv4 = torch.empty(4, dtype=torch.float32, device="cuda")
    gate_logits4 = torch.full((4, GATE_ROWS), torch.nan, dtype=torch.bfloat16, device="cuda")
    gate_logits8 = torch.full((8, GATE_ROWS), torch.nan, dtype=torch.bfloat16, device="cuda")
    gate_rms4 = torch.empty(4, dtype=torch.float32, device="cuda")
    gate_rms8 = torch.empty(8, dtype=torch.float32, device="cuda")

    epsilon = float_t(1e-6)
    _check(
        head4(
            _pointer(hidden8[:4]),
            _pointer(head_weight),
            _pointer(logits4),
            _pointer(inv4),
            epsilon,
            stream,
        ),
        "M=4 lm-head GEMM",
    )
    _check(
        head8(_pointer(hidden8), _pointer(head_weight), _pointer(logits8), stream),
        "M=8 lm-head GEMM",
    )
    _check(
        gate4(
            _pointer(hidden8[:4]),
            _pointer(gate_weight),
            _pointer(gate_logits4),
            _pointer(gate_rms4),
            epsilon,
            None,
            long_long(0),
            stream,
        ),
        "M=4 router GEMM",
    )
    _check(
        gate8(
            _pointer(hidden8),
            _pointer(gate_weight),
            _pointer(gate_logits8),
            _pointer(gate_rms8),
            epsilon,
            None,
            long_long(0),
            stream,
        ),
        "M=8 router GEMM",
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(logits8[:4] * inv4[:, None], logits4, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        gate_logits8[:4, :GATE_ACTIVE_ROWS], gate_logits4[:, :GATE_ACTIVE_ROWS], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(gate_rms8[:4], gate_rms4, rtol=0.0, atol=0.0)
    assert torch.isfinite(logits8).all().item()
    assert torch.isfinite(gate_logits8[:, :GATE_ACTIVE_ROWS]).all().item()
    for scale in (inv4, gate_rms4, gate_rms8):
        assert torch.isfinite(scale).all().item() and torch.all(scale > 0).item()
    print(f"BATCH GEMM GPU SMOKE PASS ({torch.cuda.get_device_name()})", flush=True)


if __name__ == "__main__":
    main()
