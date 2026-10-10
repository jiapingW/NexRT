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

"""CUDA smoke for the SGLang -> NexRT append/finalize handoff kernels.

This deliberately uses synthetic pool/cache buffers so it can validate the copy and
slot-publication contracts without loading the 397B checkpoint.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "nex_n2_5_pro"))


def ptr(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this smoke")

    from kernel_build import build_kernel

    lib = ctypes.CDLL(
        str(
            build_kernel(
                ROOT / "nex_n2_5_pro" / "kernels" / "util", "nex_n2_5_pro_bridge_handoff_ctypes"
            )
        )
    )
    vp, ci = ctypes.c_void_p, ctypes.c_int
    lib.nex_n2_5_pro_bridge_append_launch.argtypes = [vp, ci, vp, ci, ci, vp]
    lib.nex_n2_5_pro_bridge_finalize_launch.argtypes = [vp, ci, ci, ci, ci, vp, vp, ci, vp]

    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    # Use an arbitrary, non-contiguous slot mapping to exercise the indexed read.
    n_kv, pool_rows, n_rows, dst_start = 1, 64, 257, 19
    src_k = torch.arange(pool_rows * 256, device="cuda", dtype=torch.int32).to(torch.bfloat16)
    src_k = src_k.reshape(pool_rows, 256)
    src_v = src_k + 1
    dst_k = torch.full((dst_start + n_rows + 3, 256), -9, device="cuda", dtype=torch.bfloat16)
    dst_v = torch.full_like(dst_k, -9)
    idx = torch.tensor(
        [(i * 17 + 3) % pool_rows for i in range(n_rows)], device="cuda", dtype=torch.int32
    )
    kv_table = torch.tensor(
        [src_k.data_ptr(), src_v.data_ptr(), dst_k.data_ptr(), dst_v.data_ptr()],
        device="cuda",
        dtype=torch.int64,
    )
    rc = lib.nex_n2_5_pro_bridge_append_launch(
        ptr(kv_table), n_kv, ptr(idx), n_rows, dst_start, stream
    )
    if rc != 0:
        raise RuntimeError(f"append launch rc={rc}")
    torch.cuda.synchronize()
    assert torch.equal(dst_k[dst_start : dst_start + n_rows], src_k[idx])
    assert torch.equal(dst_v[dst_start : dst_start + n_rows], src_v[idx])
    assert bool(torch.all(dst_k[:dst_start] == -9))

    # Finalize copies one synthetic GDN state and publishes the decode slot.
    n_gdn, mamba_slot, first_token = 1, 2, 1234
    conv_pool = torch.arange(4 * 1536 * 3, device="cuda", dtype=torch.int32).to(torch.bfloat16)
    conv_pool = conv_pool.reshape(4, 1536, 3)
    temporal_pool = torch.arange(4 * 8 * 128 * 128, device="cuda", dtype=torch.float32)
    temporal_pool = temporal_pool.reshape(4, 8, 128, 128)
    conv4 = torch.full((1536, 4), -1, device="cuda", dtype=torch.bfloat16)
    state_t = torch.full((8, 128, 128), -1, device="cuda", dtype=torch.float32)
    slot = torch.full((2,), -7, device="cuda", dtype=torch.int32)
    status0 = torch.tensor([41], device="cuda", dtype=torch.int32)
    status1 = torch.tensor([42], device="cuda", dtype=torch.int32)
    gdn_table = torch.tensor(
        [conv_pool.data_ptr(), temporal_pool.data_ptr(), conv4.data_ptr(), state_t.data_ptr()],
        device="cuda",
        dtype=torch.int64,
    )
    status_table = torch.tensor(
        [status0.data_ptr(), status1.data_ptr()], device="cuda", dtype=torch.int64
    )
    rc = lib.nex_n2_5_pro_bridge_finalize_launch(
        ptr(gdn_table),
        n_gdn,
        mamba_slot,
        first_token,
        n_rows,
        ptr(slot),
        ptr(status_table),
        2,
        stream,
    )
    if rc != 0:
        raise RuntimeError(f"finalize launch rc={rc}")
    torch.cuda.synchronize()
    assert torch.equal(conv4[:, :3], conv_pool[mamba_slot])
    assert bool(torch.all(conv4[:, 3] == 0))
    assert torch.equal(state_t, temporal_pool[mamba_slot])
    assert torch.equal(slot.cpu(), torch.tensor([first_token, n_rows]))
    assert int(status0.item()) == 0 and int(status1.item()) == 0
    print("bridge handoff append/finalize smoke: PASS", flush=True)


if __name__ == "__main__":
    main()
