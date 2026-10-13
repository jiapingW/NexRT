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

"""The production WGMMA attention megakernel must match between fixed-S and device-S launches.

Run on one Hopper GPU (including H20's non-cooperative fallback):
  python3 nex_n2_5_pro/tests/test_attn_core_dyn.py
"""

import ctypes
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from kernel_build import build_kernel  # noqa: E402

KDIR = HERE / "kernels" / "full_attn"
HD, NH, NS = 256, 4, 128
KV_CAPACITY = 256 * 1024 + 1024


def build():
    return build_kernel(KDIR, "nex_n2_5_pro_attn_fd4_wgmma_ctypes", hopper=True, link_cuda=True)


def main():
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    torch.manual_seed(0)
    lib = ctypes.CDLL(str(build()))
    vp, ci = ctypes.c_void_p, ctypes.c_int
    lib.NexN25ProAttnWgmmaLaunch.argtypes = [vp] * 3 + [ci] + [vp] * 6
    lib.NexN25ProAttnWgmmaLaunch.restype = ci
    lib.NexN25ProAttnWgmmaDynLaunch.argtypes = [vp] * 10
    lib.NexN25ProAttnWgmmaDynLaunch.restype = ci

    ptr = lambda tensor: ctypes.c_void_p(tensor.data_ptr())
    stream = lambda: ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    q = (torch.randn(NH, HD, dtype=torch.bfloat16, device=device) * 0.3).contiguous()
    k = torch.zeros(KV_CAPACITY, HD, dtype=torch.bfloat16, device=device)
    v = torch.zeros_like(k)
    max_s = 16384
    k[:max_s].normal_(std=0.3)
    v[:max_s].normal_(std=0.3)

    def buffers():
        return (
            torch.empty(NS * NH, dtype=torch.float32, device=device),
            torch.empty(NS * NH, dtype=torch.float32, device=device),
            torch.empty(NS * NH * HD, dtype=torch.bfloat16, device=device),
            torch.empty(NH, HD, dtype=torch.bfloat16, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
        )

    fixed = buffers()
    dynamic = buffers()
    slot = torch.tensor([12345, 0], dtype=torch.int32, device=device)
    failures = 0

    def launch_fixed(s, state):
        wm, wl, wa, ctx, counter = state
        rc = lib.NexN25ProAttnWgmmaLaunch(
            ptr(q),
            ptr(k),
            ptr(v),
            ci(s),
            ptr(wm),
            ptr(wl),
            ptr(wa),
            ptr(ctx),
            ptr(counter),
            stream(),
        )
        if rc:
            raise RuntimeError(f"fixed-S WGMMA launch returned CUDA error {rc}")

    def launch_dynamic(state):
        wm, wl, wa, ctx, counter = state
        rc = lib.NexN25ProAttnWgmmaDynLaunch(
            ptr(q),
            ptr(k),
            ptr(v),
            ptr(slot),
            ptr(wm),
            ptr(wl),
            ptr(wa),
            ptr(ctx),
            ptr(counter),
            stream(),
        )
        if rc:
            raise RuntimeError(f"device-S WGMMA launch returned CUDA error {rc}")

    for s in (1, 27, 127, 128, 129, 512, 1024, 2049, 4096, 16384):
        slot[1] = s - 1
        launch_fixed(s, fixed)
        launch_dynamic(dynamic)
        torch.cuda.synchronize()
        max_abs = float((fixed[3].float() - dynamic[3].float()).abs().max())
        ok = max_abs == 0.0
        failures += int(not ok)
        print(f"S={s:6d} fixed-vs-device max_abs={max_abs:.3e} {'OK' if ok else 'MISMATCH'}")
        # A separate FP32 reference catches errors shared by both launch paths,
        # including empty partials when a short prompt has fewer than 128 tokens.
        scores = q.float() @ k[:s].float().T / (HD**0.5)
        reference = scores.softmax(-1) @ v[:s].float()
        reference_error = float((dynamic[3].float() - reference).abs().max())
        reference_ok = reference_error < 2e-3
        failures += int(not reference_ok)
        print(
            f"S={s:6d} vs-fp32 max_abs={reference_error:.3e} {'OK' if reference_ok else 'MISMATCH'}"
        )

    slot[1] = 511
    launch_dynamic(dynamic)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch_dynamic(dynamic)
    torch.cuda.synchronize()
    for s in (1, 27, 129, 512, 2049, 4096):
        slot[1] = s - 1
        graph.replay()
        launch_fixed(s, fixed)
        torch.cuda.synchronize()
        max_abs = float((fixed[3].float() - dynamic[3].float()).abs().max())
        ok = max_abs == 0.0
        failures += int(not ok)
        print(f"graph S={s:6d} fixed-vs-device max_abs={max_abs:.3e} {'OK' if ok else 'MISMATCH'}")

    print("ALL BIT-IDENTICAL" if failures == 0 else f"{failures} MISMATCH(es)")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
