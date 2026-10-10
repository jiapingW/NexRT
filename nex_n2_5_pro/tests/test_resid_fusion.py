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

"""Validate the fused-residual reduce (GDN so far): forward(x, resid=r) must equal r + forward(x) BIT-for-BIT
(the reduce folds the residual add into its epilogue, replacing the separate residual-add kernel).

Run:  torchrun --standalone --nproc_per_node=8 nex_n2_5_pro/tests/test_resid_fusion.py
"""

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from gdn import Gdn, HID, QKV_ROWS, HEADS, DIM  # noqa: E402


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl")
    assert world == 8
    g = Gdn(0, rank, world, dev)
    gen = torch.Generator(device=dev).manual_seed(7 + rank)
    x = torch.randn(HID, dtype=torch.bfloat16, device=dev, generator=gen)
    conv0 = (
        torch.randn(QKV_ROWS, 4, dtype=torch.bfloat16, device=dev, generator=gen) * 0.1
    ).contiguous()
    state0 = (
        torch.randn(HEADS, DIM, DIM, dtype=torch.float32, device=dev, generator=gen) * 0.1
    ).contiguous()

    # separate-residual reference: r + forward(x)
    g.conv_state4.copy_(conv0)
    g.state_t.copy_(state0)
    o = g.forward(x).clone()  # resid=None -> final_out fp32
    ref = (x.float() + o.float()).to(torch.bfloat16)  # the exact torch residual add
    st1 = g.state_t.clone()  # state after this forward (for state check)

    # fused-residual: forward(x, resid=xr) writes xr += reduced (in place, bf16)
    g.conv_state4.copy_(conv0)
    g.state_t.copy_(state0)
    xr = x.clone()
    g.forward(xr, resid=xr)
    torch.cuda.synchronize()
    dist.barrier()

    d = (xr.float() - ref.float()).abs().max().item()
    ds = (g.state_t - st1).abs().max().item()  # state update must be identical too
    t = torch.tensor([d, ds], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    if rank == 0:
        ok = t[0].item() == 0.0 and t[1].item() == 0.0
        print(
            f"GDN fused-residual vs (x + forward): out max_abs={t[0].item():.3e} state max_abs={t[1].item():.3e} "
            f"[{'OK bit-identical' if ok else 'MISMATCH'}]",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
