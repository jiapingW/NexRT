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

"""Build sweep: compile and dlopen every CUDA translation unit in kernels/.

Every .cu builds standalone (files that other TUs #include are also valid
standalone TUs).  Each object is then dlopened, which proves the ELF links with
no unresolved symbols.  Run on the target machine (nvcc + libcuda required):

    python3 tests/build_sweep.py            # build all, from the current cache
    python3 tests/build_sweep.py --clean    # delete cached .so first (from-zero proof)
"""

import argparse
import ctypes
import errno
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(HERE.parent))

from nex_n2_5_pro.kernel_build import (  # noqa: E402
    KERNEL_COMPONENTS,
    build_kernel,
    validate_build_path,
)

# TUs using Hopper wgmma / the TMA descriptor driver API need the explicit
# sm_90a gencode and a libcuda link (see attn.build_cp).
HOPPER = {
    "nex_n2_5_pro_attn_fd4_wgmma_ctypes",
    "nex_n2_5_pro_cp_hopper_mega_ctypes",
    "nex_n2_5_pro_cp_qgate_ctypes",
    "nex_n2_5_pro_cp_tail_ctypes",
    "nex_n2_5_pro_m4_attn_core_quad_ctypes",
    "nex_n2_5_pro_m4_cp_quad_partial_ctypes",
    "nex_n2_5_pro_m8_attn_core_ctypes",
    "nex_n2_5_pro_m8_cp_attn_ctypes",
    "nex_n2_5_pro_mtp_attn_core_ctypes",
    "nex_n2_5_pro_dflash_attn_wgmma_ctypes",
}


def _is_generated_artifact(path):
    name = path.name
    if name.endswith(".so"):
        return True
    if not name.startswith(".") or not name.endswith(".tmp"):
        return False
    stem, separator, pid = name[1:-4].rpartition(".so.")
    return bool(stem and separator and pid.isdecimal())


def _enumerate_build_entries(build_root):
    """Enumerate without following links, rejecting any symlink before cleanup."""
    entries = []
    pending = [build_root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as iterator:
            children = sorted(iterator, key=lambda entry: entry.name)
        for child in children:
            path = Path(child.path)
            if child.is_symlink():
                raise RuntimeError(f"refusing symlink in build tree: {path}")
            entries.append(path)
            if child.is_dir(follow_symlinks=False):
                pending.append(path)
    return sorted(entries)


def clean_build_artifacts():
    """Remove only validated JIT artifacts and then empty managed directories."""
    build_root = validate_build_path()
    if not build_root.exists():
        return

    entries = _enumerate_build_entries(build_root)
    # Revalidate the ancestry after enumeration and before the first deletion.
    if validate_build_path() != build_root:
        raise RuntimeError(f"build root changed during cleanup: {build_root}")

    artifacts = []
    component_dirs = []
    for entry in entries:
        relative = entry.relative_to(build_root)
        if len(relative.parts) == 1:
            component = relative.parts[0]
            if component not in KERNEL_COMPONENTS or not entry.is_dir():
                raise RuntimeError(f"unexpected build-root entry: {entry}")
            if validate_build_path(component) != entry:
                raise RuntimeError(f"invalid build component path: {entry}")
            component_dirs.append(entry)
        elif len(relative.parts) == 2:
            component = relative.parts[0]
            if component not in KERNEL_COMPONENTS:
                raise RuntimeError(f"unexpected build component: {entry}")
            if not entry.is_file() or not _is_generated_artifact(entry):
                raise RuntimeError(f"unexpected non-artifact build entry: {entry}")
            artifacts.append(entry)
        else:
            raise RuntimeError(f"unexpected nested build entry: {entry}")

    for artifact in artifacts:
        validate_build_path(artifact.parent.name)
        if artifact.is_symlink() or not artifact.is_file():
            raise RuntimeError(f"build artifact changed during cleanup: {artifact}")
        artifact.unlink()
    for component_dir in reversed(component_dirs):
        validate_build_path(component_dir.name)
        component_dir.rmdir()
    validate_build_path()
    build_root.rmdir()
    validate_build_path()
    try:
        build_root.parent.rmdir()
    except OSError as err:
        if err.errno not in (errno.EEXIST, errno.ENOTEMPTY):
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--clean", action="store_true", help="delete the fixed build/sm90a artifact tree first"
    )
    cli = ap.parse_args()
    kroot = HERE / "kernels"
    if cli.clean:
        clean_build_artifacts()
    sources = sorted(p for p in kroot.rglob("*.cu"))
    failures = []
    for src in sources:
        name = src.stem
        try:
            so = build_kernel(src.parent, name, hopper=name in HOPPER, link_cuda=name in HOPPER)
            ctypes.CDLL(str(so))
        except Exception as err:  # noqa: BLE001  (report and continue the sweep)
            failures.append((src.relative_to(HERE), err))
            print(f"FAIL {src.relative_to(HERE)}: {err}", flush=True)
            continue
        print(f"ok   {src.relative_to(HERE)}", flush=True)
    print(f"BUILD SWEEP: {len(sources)} TUs, {len(failures)} failures", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
