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

"""Host-only gate for full-width (64-bit) request-seed handling.

The bridge transports request seeds through a signed torch.int64 header and a
device int64 scalar (head.seed_dev); the sampling kernels reinterpret those
bits as u64, and the CPU verification oracle replays splitmix64 on
``seed & (2**64 - 1)``. A seed at or above 2**63 therefore only round-trips if
the low 64 BITS are preserved exactly (two's complement), which is what
bridge_serve._seed_as_int64 provides. This test locks that contract, including
seeds the previous 63-bit mask would have altered.

Run: python3 tests/test_seed_width.py
"""

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from mtp.sampling_ref import seeded_uniform, splitmix64  # noqa: E402

MASK64 = (1 << 64) - 1


def _load_production_seed_helper():
    """Extract the REAL bridge_serve._seed_as_int64 without importing the module
    (bridge_serve imports sglang at load; this test must stay host-only). The
    function is compiled from the production source, so a regression there
    fails HERE, not just in a live serve."""
    import ast

    src = (HERE / "sglang_prefill" / "bridge_serve.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_seed_as_int64")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "bridge_serve.py", "exec"), ns)
    return ns["_seed_as_int64"]


_seed_as_int64 = _load_production_seed_helper()


def main():
    cases = [
        0,
        1,
        42,
        20260712,
        (1 << 62) + 3,
        (1 << 63) - 1,
        (1 << 63),
        (1 << 63) + 123,
        MASK64,
        MASK64 + 7,
        -1,
    ]
    failures = []
    for seed in cases:
        expect_bits = int(seed) & MASK64
        signed = _seed_as_int64(seed)
        # 1. int64 range (torch header/device scalar representable)
        if not (-(1 << 63) <= signed < (1 << 63)):
            failures.append(f"seed {seed}: signed value out of int64 range")
            continue
        # 2. tensor round-trip preserves the bit pattern
        t = torch.tensor([signed], dtype=torch.int64)
        got_bits = int(t.item()) & MASK64
        if got_bits != expect_bits:
            failures.append(f"seed {seed}: bit pattern {got_bits:#x} != {expect_bits:#x}")
            continue
        # 3. the oracle sees the same u64 the kernel reinterprets
        if splitmix64(got_bits) != splitmix64(expect_bits):
            failures.append(f"seed {seed}: splitmix64 mismatch")
            continue
        # 4. position-keyed draws are deterministic functions of those bits
        if seeded_uniform(got_bits, 12345) != seeded_uniform(expect_bits, 12345):
            failures.append(f"seed {seed}: seeded_uniform mismatch")
    # 5. a >=2**63 seed must NOT equal its 63-bit-masked variant (the old
    #    behavior) -- proves the full width actually reaches the RNG.
    big = (1 << 63) + 123
    if seeded_uniform(big, 7) == seeded_uniform(big & 0x7FFFFFFFFFFFFFFF, 7):
        failures.append("full-width and 63-bit-masked seeds draw identically")
    for f in failures:
        print("FAIL", f)
    print(
        f"{'PASS' if not failures else 'FAIL'} test_seed_width: "
        f"{len(cases)} seeds, {len(failures)} failures"
    )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
