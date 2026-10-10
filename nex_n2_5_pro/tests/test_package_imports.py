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

"""Host-only checks for deterministic lightweight package imports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


MODULES = (
    "nex_n2_5_pro",
    "nex_n2_5_pro.sglang_prefill",
    "nex_n2_5_pro.config",
    "nex_n2_5_pro.kernel_build",
    "nex_n2_5_pro.m4_control",
    "nex_n2_5_pro.m8_control",
    "nex_n2_5_pro.dflash.config",
    "nex_n2_5_pro.mtp.sampling_ref",
    "nex_n2_5_pro.sglang_prefill.cache_lifecycle",
    "nex_n2_5_pro.sglang_prefill.cache_consensus",
    "nex_n2_5_pro.sglang_prefill.process_supervisor",
    "nex_n2_5_pro.sglang_prefill.serve_control",
)


def test_lightweight_imports_outside_repository_cwd(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    assert repo_root not in (tmp_path, *tmp_path.parents)
    script = "\n".join(
        (
            "import importlib",
            "import sys",
            f"modules = {MODULES!r}",
            "for module in modules[:2]:",
            "    importlib.import_module(module)",
            "assert 'torch' not in sys.modules",
            "assert not any(name == 'sglang' or name.startswith('sglang.') "
            "for name in sys.modules)",
            "for module in modules[2:]:",
            "    importlib.import_module(module)",
            "print('\\n'.join(modules))",
        )
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == list(MODULES)
