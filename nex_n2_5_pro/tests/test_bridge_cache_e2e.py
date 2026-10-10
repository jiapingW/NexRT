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

"""Launch the real TP8 SGLang -> NexRT cache integration matrix.

This is an explicit GPU integration entrypoint, not a mocked handoff test. The target serving
process checks PREFILL -> decode/restore -> exact HIT -> suffix APPEND against an independent
full-PREFILL oracle, and emits matched full-gather/chunk-append raw timing repetitions. This matrix covers target decoding.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "nex_n2_5_pro" / "sglang_prefill" / "bridge_serve.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("NEXRT_MODEL_DIR"))
    parser.add_argument("--layers-dir", default=os.environ.get("NEXRT_LAYERS_DIR"))
    parser.add_argument("--mode", choices=("target",), default="target")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--gen", type=int, default=8)
    parser.add_argument(
        "--consensus-benchmark",
        action="store_true",
        help="warm up and alternate matched consensus-off/on request sequences",
    )
    parser.add_argument(
        "--fault-rank",
        type=int,
        default=None,
        help="reject one rank's READY boundary and verify collective PREFILL fallback",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run only a 4K -> 8K non-CP pair; does not require extracted CP weights",
    )
    parser.add_argument("--report", type=Path, default=ROOT / "cache_e2e_report.jsonl")
    args = parser.parse_args()
    if not args.model:
        parser.error("--model or NEXRT_MODEL_DIR is required")
    if args.repetitions < 1 or args.gen < 2:
        parser.error("--repetitions must be >= 1 and --gen must be >= 2")
    if args.fault_rank is not None and not 0 <= args.fault_rank < 8:
        parser.error("--fault-rank must be in [0, 7]")
    if not args.quick and not args.layers_dir:
        parser.error("CP coverage requires --layers-dir or NEXRT_LAYERS_DIR")

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=8",
        str(SERVER),
        "--cache-e2e",
        "4096,8192",
        "--cache-e2e-repetitions",
        str(args.repetitions),
        "--cache-e2e-gen",
        str(args.gen),
    ]
    if args.quick:
        command.extend(("--no-cp", "--max-prompt", "16384"))
    else:
        command.extend(("--cache-e2e", "49152,53248", "--max-prompt", "65536"))
    if args.consensus_benchmark:
        command.append("--cache-consensus-benchmark")
    if args.fault_rank is not None:
        command.extend(("--cache-consensus-fault-rank", str(args.fault_rank)))

    environment = os.environ.copy()
    environment["NEXRT_MODEL_DIR"] = str(Path(args.model).resolve())
    environment["NEXRT_SERVE_REPORT"] = str(args.report.resolve())
    if args.layers_dir:
        environment["NEXRT_LAYERS_DIR"] = str(Path(args.layers_dir).resolve())

    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"TP8 cache integration failed with exit code {completed.returncode}")
    if not args.report.is_file():
        raise RuntimeError(f"TP8 cache integration did not write {args.report}")
    print(f"PASS test_bridge_cache_e2e: mode={args.mode} report={args.report}")


if __name__ == "__main__":
    main()
