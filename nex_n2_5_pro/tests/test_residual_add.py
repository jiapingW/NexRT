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

"""Unit test: the residual-add kernel (x_bf16 += out_fp32, in place) must be BIT-IDENTICAL to the eager
torch op it replaces in the decode graph -- (x.float() + out).to(bf16). Run with one GPU:
  python3 nex_n2_5_pro/tests/test_residual_add.py
"""

import ctypes
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from kernel_build import build_kernel  # noqa: E402

KDIR = HERE / "kernels" / "util"
HID = 4096


def build(name):
    return build_kernel(KDIR, name)


def main():
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    lib = ctypes.CDLL(str(build("nex_n2_5_pro_residual_add_ctypes")))
    lib.NexN25ProResidualAddLaunch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    p = lambda t: ctypes.c_void_p(t.data_ptr())
    sp = lambda: ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    bad = 0
    for scale in (0.01, 1.0, 100.0):  # small, unit, large-magnitude residuals
        x0 = (torch.randn(HID, dtype=torch.bfloat16) * scale).contiguous()
        out = (torch.randn(HID, dtype=torch.float32) * scale).contiguous()
        ref = (x0.float() + out).to(torch.bfloat16)  # the exact torch op the kernel replaces
        x = x0.clone()
        lib.NexN25ProResidualAddLaunch(p(x), p(out), ctypes.c_int(HID), sp())
        torch.cuda.synchronize()
        d = (x.float() - ref.float()).abs().max().item()
        ok = d == 0.0
        bad += not ok
        print(f"scale={scale:6.2f}  kernel-vs-torch max_abs={d:.3e}  {'OK' if ok else 'MISMATCH'}")
    print("ALL BIT-IDENTICAL" if bad == 0 else f"{bad} MISMATCH(es)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
