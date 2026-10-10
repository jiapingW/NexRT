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

"""QKV must preserve finite BF16 inputs beyond FP16's arithmetic range.

Run on one supported GPU: python -m nex_n2_5_pro.tests.test_qkv_range
"""

import ctypes
from pathlib import Path

import torch

from nex_n2_5_pro.kernel_build import build_kernel


def main():
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", 0)
    root = Path(__file__).resolve().parents[1] / "kernels"
    ptr = lambda tensor: ctypes.c_void_p(tensor.data_ptr())
    stream = lambda: ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    vp, ci = ctypes.c_void_p, ctypes.c_int
    count = 0
    for mode, component, stem, symbol, rows in (
        ("bs1", "full_attn", "qkv_proj", "QkvProj", 2560),
        ("cp", "full_attn", "cp_qgate", "CpQGate", 5632),
        ("mtp", "mtp", "mtp_qkv_proj", "MtpQkvProj", 2560),
    ):
        lib = ctypes.CDLL(str(build_kernel(root / component, f"nex_n2_5_pro_{stem}_ctypes")))
        dynamic = getattr(lib, f"NexN25Pro{symbol}DynLaunch")
        dynamic.argtypes, dynamic.restype = [vp] * 9, ci
        fixed = getattr(lib, f"NexN25Pro{symbol}Launch", None)
        if fixed is not None:
            fixed.argtypes, fixed.restype = [vp] * 7 + [ci, vp], ci
        for case in (
            "ordinary",
            "activation_range",
            "product_range",
            "partial_sum_range",
            "cancellation",
        ):
            x = torch.randn(4096, dtype=torch.bfloat16, device=device)
            w = (torch.randn(rows, 4096, device=device) * 32).to(torch.float8_e4m3fn)
            scales = torch.full((rows // 128, 32), 1 / 256, device=device)
            if case == "activation_range":
                x.mul_(2**18)
            elif case in ("product_range", "partial_sum_range"):
                x.fill_(512 if case == "product_range" else 64)
                w.fill_(448)
            elif case == "cancellation":
                x.fill_(2**18)
                x[1::2].neg_()
                w.fill_(448)
            # Input conversion, product and partial-sum overflow are distinct.
            # Reference dequantization and projection stay FP32 throughout.
            dense = w.float() * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
            projected = dense @ x.float()
            qrows = 4096 if mode == "cp" else 1024
            q = torch.empty(qrows, dtype=torch.bfloat16, device=device)
            gate = torch.empty(1024, dtype=torch.bfloat16, device=device)
            key = torch.full((4, 256), -17, dtype=torch.bfloat16, device=device)
            value = torch.full_like(key, -17)
            slot = torch.tensor([123, 1], dtype=torch.int32, device=device)
            args = [ptr(t) for t in (x, w, scales, q, gate, key, value)]
            if mode == "cp":
                expected = projected
            else:
                qgate = projected[:2048].reshape(4, 2, 256)
                expected = torch.cat(
                    (qgate[:, 0].flatten(), qgate[:, 1].flatten(), projected[2048:])
                )

            def check(position):
                actual = torch.cat((q, gate, key[position], value[position])).float()
                assert bool(torch.isfinite(actual).all()), (mode, case, "nonfinite")
                atol = max(1e-5, float(expected.abs().max()) * 1e-6)
                torch.testing.assert_close(actual, expected, rtol=0.005, atol=atol)
                assert bool((key[0] == -17).all() & (value[0] == -17).all())
                assert bool((key[3] == -17).all() & (value[3] == -17).all())

            if fixed is not None:
                assert fixed(*args, 1, stream()) == 0
                check(1)
            assert dynamic(*args, ptr(slot), stream()) == 0
            check(1)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                assert dynamic(*args, ptr(slot), stream()) == 0
            slot[1] = 2
            graph.replay()
            check(2)
            count += 1
            print(f"PASS {mode}/{case}: fixed/dynamic/graph as applicable", flush=True)
    print(f"PASS {count} QKV arithmetic cases", flush=True)


if __name__ == "__main__":
    main()
