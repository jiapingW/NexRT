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

"""Benchmark full gather against chunk append + finalize at Nex-N2.5-Pro geometry."""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "nex_n2_5_pro"))


def _p(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens", type=int, default=16384)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    if args.lens <= 0 or args.chunk <= 0 or args.iters <= 0:
        parser.error("--lens, --chunk and --iters must be positive")

    from kernel_build import build_kernel

    lib = ctypes.CDLL(
        str(
            build_kernel(
                ROOT / "nex_n2_5_pro" / "kernels" / "util", "nex_n2_5_pro_bridge_handoff_ctypes"
            )
        )
    )
    vp, ci = ctypes.c_void_p, ctypes.c_int
    lib.nex_n2_5_pro_bridge_gather_launch.argtypes = [
        vp,
        ci,
        vp,
        ci,
        vp,
        ci,
        ci,
        ci,
        vp,
        vp,
        ci,
        vp,
    ]
    lib.nex_n2_5_pro_bridge_append_launch.argtypes = [vp, ci, vp, ci, ci, vp]
    lib.nex_n2_5_pro_bridge_finalize_launch.argtypes = [vp, ci, ci, ci, ci, vp, vp, ci, vp]

    dev = torch.device("cuda")
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    n_kv, n_gdn = 15, 45
    cache_len = args.lens + 16
    # The weight precision does not enter the handoff: nex_n2_5_pro geometry uses BF16 KV
    # for both BF16 and FP8 weights. These buffers reproduce the complete TP-rank layout.
    src_k = [torch.empty(cache_len, 256, dtype=torch.bfloat16, device=dev) for _ in range(n_kv)]
    src_v = [torch.empty_like(tensor) for tensor in src_k]
    dst_k = [torch.empty_like(tensor) for tensor in src_k]
    dst_v = [torch.empty_like(tensor) for tensor in src_k]
    idx = torch.arange(args.lens, dtype=torch.int32, device=dev)
    kv_table = torch.tensor(
        [tensor.data_ptr() for tensor in (*src_k, *src_v, *dst_k, *dst_v)],
        dtype=torch.int64,
        device=dev,
    )

    conv_src = [torch.empty(1, 1536, 3, dtype=torch.bfloat16, device=dev) for _ in range(n_gdn)]
    temporal_src = [
        torch.empty(1, 8, 128, 128, dtype=torch.float32, device=dev) for _ in range(n_gdn)
    ]
    conv_dst = [torch.empty(1536, 4, dtype=torch.bfloat16, device=dev) for _ in range(n_gdn)]
    temporal_dst = [torch.empty(8, 128, 128, dtype=torch.float32, device=dev) for _ in range(n_gdn)]
    gdn_table = torch.tensor(
        [tensor.data_ptr() for tensor in (*conv_src, *temporal_src, *conv_dst, *temporal_dst)],
        dtype=torch.int64,
        device=dev,
    )
    slot = torch.empty(2, dtype=torch.int32, device=dev)
    status = torch.ones(1, dtype=torch.int32, device=dev)
    status_table = torch.tensor([status.data_ptr()], dtype=torch.int64, device=dev)

    def gather():
        rc = lib.nex_n2_5_pro_bridge_gather_launch(
            _p(kv_table),
            n_kv,
            _p(idx),
            args.lens,
            _p(gdn_table),
            n_gdn,
            0,
            1,
            _p(slot),
            _p(status_table),
            1,
            stream,
        )
        if rc != 0:
            raise RuntimeError(f"gather launch rc={rc}")

    def append():
        for start in range(0, args.lens, args.chunk):
            rows = min(args.chunk, args.lens - start)
            rc = lib.nex_n2_5_pro_bridge_append_launch(
                _p(kv_table),
                n_kv,
                ctypes.c_void_p(idx.data_ptr() + start * 4),
                rows,
                start,
                stream,
            )
            if rc != 0:
                raise RuntimeError(f"append launch rc={rc}")
        rc = lib.nex_n2_5_pro_bridge_finalize_launch(
            _p(gdn_table),
            n_gdn,
            0,
            1,
            args.lens,
            _p(slot),
            _p(status_table),
            1,
            stream,
        )
        if rc != 0:
            raise RuntimeError(f"finalize launch rc={rc}")

    def benchmark(operation):
        for _ in range(10):
            operation()
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.iters):
            begin, done = torch.cuda.Event(True), torch.cuda.Event(True)
            begin.record()
            operation()
            done.record()
            samples.append((begin, done))
        torch.cuda.synchronize()
        values = [begin.elapsed_time(done) for begin, done in samples]
        return sum(values) / len(values), min(values), max(values)

    gather_stats = benchmark(gather)
    append_stats = benchmark(append)
    print(
        f"@@ KERNEL_HANDOFF_BENCH L={args.lens} chunk={args.chunk} "
        f"kv_layers={n_kv} gdn_layers={n_gdn} iters={args.iters} "
        f"gather_avg={gather_stats[0]:.3f}ms min={gather_stats[1]:.3f}ms "
        f"max={gather_stats[2]:.3f}ms append_avg={append_stats[0]:.3f}ms "
        f"min={append_stats[1]:.3f}ms max={append_stats[2]:.3f}ms "
        f"append/gather={append_stats[0] / gather_stats[0]:.3f}x",
        flush=True,
    )


if __name__ == "__main__":
    main()
