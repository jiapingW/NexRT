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

"""Symmetric peer-inbox setup for DFlash's in-kernel TP reductions.

Torch distributed participates only in one-time symmetric-memory rendezvous.
Every DFlash round reduction is performed by the custom LL CUDA kernels; there
is no Torch/NCCL tensor path in this module or in :mod:`dflash.block`.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symmetric_memory

WIRE_PACKETS = 2 * 8 * 128 * 32 * 8
DEFAULT_SITES = ("output_projection", "mlp_down", "target_hidden", "head")


class DFlashLlWires:
    def __init__(self, device: torch.device, status: torch.Tensor, sites=DEFAULT_SITES) -> None:
        self.device = device
        if status.dtype != torch.int32 or status.numel() < 1 or status.device != device:
            raise ValueError("status must provide an int32 device word on the DFlash device")
        # Alias the verifier's upstream-status word. The M=8 tail already folds
        # this word across ranks before publishing a commit, so a draft LL
        # timeout fails closed without an extra graph node or host readback.
        self.status = status
        self.wires = {}
        world = dist.group.WORLD
        if world is None:
            raise RuntimeError("torch.distributed must be initialized before DFlash LL setup")
        for site in sites:
            buffer = symmetric_memory.empty(WIRE_PACKETS, dtype=torch.int64, device=device)
            handle = symmetric_memory.rendezvous(buffer, world.group_name)
            multicast = int(getattr(handle, "multicast_ptr", 0) or 0)
            if multicast == 0:
                raise RuntimeError(f"multicast unavailable for DFlash LL wire {site!r}")
            buffer.zero_()
            self.wires[site] = {
                "buffer": buffer,
                "multicast": multicast,
                "local": int(handle.buffer_ptrs[dist.get_rank()]),
                "counter": torch.zeros(128, dtype=torch.int32, device=device),
            }
        dist.barrier()

    def args(self, site: str):
        wire = self.wires[site]
        return (wire["multicast"], wire["local"], wire["counter"], self.status)

    def reset_status(self) -> None:
        self.status.zero_()
