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

"""TP8 CP reduction: exact FP32 rank order, both parities and residual epilogues.

No model weights required. Run on the GPU host from the repository root:
    torchrun --standalone --nproc_per_node=8 -m nex_n2_5_pro.tests.test_cp_reduce_determinism
"""

import ctypes
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed import _symmetric_memory as symm_mem

from nex_n2_5_pro.kernel_build import build_kernel


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 8
    device = torch.device("cuda", torch.cuda.current_device())
    kdir = Path(__file__).resolve().parents[1] / "kernels" / "full_attn"
    lib = ctypes.CDLL(str(build_kernel(kdir, "nex_n2_5_pro_o_proj_reduce_ctypes")))
    vp, cu, ci = ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_int
    plain = lib.NexN25ProOProjWholeReduceLaunch
    resid = lib.NexN25ProOProjWholeReduceResidLaunch
    plain.argtypes = [vp, cu] + [vp] * 4 + [ci, vp]
    resid.argtypes = [vp, cu] + [vp] * 4 + [ci, vp, vp]
    plain.restype = resid.restype = ci

    partial = symm_mem.empty((2, 4096), dtype=torch.float32, device=device)
    partial_handle = symm_mem.rendezvous(partial, dist.group.WORLD.group_name)
    peers = torch.tensor(partial_handle.buffer_ptrs, dtype=torch.int64, device=device)
    barrier = symm_mem.empty(1, dtype=torch.int32, device=device)
    barrier_handle = symm_mem.rendezvous(barrier, dist.group.WORLD.group_name)
    multicast = int(barrier_handle.multicast_ptr)
    assert multicast, "TP8 reduction requires NVLS multicast"
    barrier.zero_()
    generation = torch.zeros(1, dtype=torch.int32, device=device)
    status = torch.zeros_like(generation)
    output = torch.empty(4096, dtype=torch.float32, device=device)
    residual = torch.empty(4096, dtype=torch.bfloat16, device=device)

    # Cancellation makes reordering observable: (2**24 + 1) - 2**24 differs
    # from (2**24 - 2**24) + 1. Rotate rank assignments across output lanes.
    values = torch.tensor([2**24, 1, -(2**24), 3, -2, 0.5, -0.25, 0.125])
    columns = torch.arange(4096)
    inputs = torch.stack([values[(columns + r) % 8] for r in range(8)])
    reference = inputs[0].clone()
    for r in range(1, 8):
        reference.add_(inputs[r])
    partial[0].copy_(inputs[rank])
    partial[1].copy_(-inputs[rank])
    references = [reference.to(device), (-reference).to(device)]
    residual_refs = [(ref + 0.75).to(torch.bfloat16) for ref in references]
    torch.cuda.synchronize()
    dist.barrier()

    ptr = lambda tensor: vp(tensor.data_ptr())

    def launch(fused, skew):
        generation.add_(1)
        # Change peer arrival order without changing any arithmetic input.
        torch.cuda._sleep(((rank + skew) % 8) * 10000)
        args = [
            ptr(peers),
            cu(multicast),
            ptr(barrier),
            ptr(output),
            ptr(status),
            ptr(generation),
            8,
        ]
        if fused:
            args.append(ptr(residual))
        args.append(vp(torch.cuda.current_stream().cuda_stream))
        rc = (resid if fused else plain)(*args)
        if rc:
            raise RuntimeError(f"CP reduce launch failed: {rc}")

    failures = 0
    epoch = 0
    for fused in (False, True):
        for skew in (0, 3, 7):
            residual.fill_(0.75)
            launch(fused, skew)
            epoch += 1
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(fused, skew)
            for _ in range(16):
                residual.fill_(0.75)
                graph.replay()
                epoch += 1
                parity = (epoch - 1) & 1
                actual = residual if fused else output
                expected = residual_refs[parity] if fused else references[parity]
                failures += int(
                    not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
                )
                failures += int(status.item() != 0 or generation.item() != epoch)
    result = torch.tensor([failures], device=device)
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    failures = int(result.item())
    if rank == 0:
        print(f"CP fixed-order reduction: 96 graph replays, failures={failures}")
    dist.destroy_process_group()
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
