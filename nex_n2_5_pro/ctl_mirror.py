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

"""Host-mapped control words for CUDA-graph generation.

The graph's final node publishes control words and a generation counter into
pinned host memory. Serving loops poll the counter and can queue the next replay
while reading the previous result, avoiding a device-to-host copy per round.
"""

from __future__ import annotations

import ctypes
import time
from pathlib import Path

import torch

from nex_n2_5_pro.kernel_build import build_kernel

KUTIL = Path(__file__).resolve().parent / "kernels" / "util"


class CtlMirror:
    """Two-slot host mirror of an int32 device control tensor + its device generation counter."""

    def __init__(self, ctl: torch.Tensor, dev: torch.device):
        assert ctl.dtype == torch.int32 and ctl.is_cuda and 1 <= ctl.numel() <= 32
        lib = ctypes.CDLL(str(build_kernel(KUTIL, "nex_n2_5_pro_ctl_mirror_ctypes")))
        lib.NexN25ProCtlMirrorAlloc.restype = ctypes.c_void_p
        lib.NexN25ProCtlMirrorAlloc.argtypes = [ctypes.c_int]
        lib.NexN25ProCtlMirrorDevicePtr.restype = ctypes.c_void_p
        lib.NexN25ProCtlMirrorDevicePtr.argtypes = [ctypes.c_void_p]
        lib.NexN25ProCtlMirrorLaunch.restype = ctypes.c_int
        lib.NexN25ProCtlMirrorLaunch.argtypes = [ctypes.c_void_p] * 3 + [
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib, self.ctl, self.words = lib, ctl, ctl.numel()
        stride = self.words + 1
        self.host = lib.NexN25ProCtlMirrorAlloc(self.words)
        if not self.host:
            raise RuntimeError("cudaHostAlloc(mapped) failed for the control-block mirror")
        self.devptr = lib.NexN25ProCtlMirrorDevicePtr(self.host)
        if not self.devptr:
            raise RuntimeError("cudaHostGetDevicePointer failed for the control-block mirror")
        self.gen = torch.zeros(1, dtype=torch.int32, device=dev)  # device counter, never reset
        self._flag = [
            ctypes.cast(self.host + 4 * (s * stride + self.words), ctypes.POINTER(ctypes.c_int32))
            for s in (0, 1)
        ]
        self._words = [
            (ctypes.c_int32 * self.words).from_address(self.host + 4 * s * stride) for s in (0, 1)
        ]

    def launch(self) -> None:
        """Publish the current control words (call at the END of the round / step; capturable)."""
        rc = self.lib.NexN25ProCtlMirrorLaunch(
            ctypes.c_void_p(self.ctl.data_ptr()),
            ctypes.c_void_p(self.devptr),
            ctypes.c_void_p(self.gen.data_ptr()),
            self.words,
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        )
        if rc != 0:
            raise RuntimeError(f"NexN25ProCtlMirrorLaunch failed: cudaError {rc}")

    def last_gen(self) -> int:
        """Generation of the latest published round (call with the device quiescent)."""
        return max(self._flag[0][0], self._flag[1][0])

    def wait(self, gen: int, timeout_s: float = 120.0) -> list[int]:
        """Spin until generation `gen` is published; return a copy of its control words."""
        flag = self._flag[gen & 1]
        n, t0 = 0, None
        while flag[0] != gen:
            n += 1
            if (n & 4095) == 0:
                now = time.perf_counter()
                if t0 is None:
                    t0 = now
                elif now - t0 > timeout_s:
                    raise RuntimeError(
                        f"control-block mirror: generation {gen} not published "
                        f"within {timeout_s:.0f} s (flag {flag[0]})"
                    )
        return list(self._words[gen & 1])
