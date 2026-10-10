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

"""Race-free NVCC build helper shared by nexrt's Python wrappers."""

import fcntl
import hashlib
import os
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
KERNEL_ROOT = REPO_ROOT / "nex_n2_5_pro" / "kernels"
BUILD_ROOT = REPO_ROOT / "build" / "sm90a"
BUILD_HELPER = Path(__file__).resolve()
LOCK_ROOT = Path("/tmp")
KERNEL_COMPONENTS = frozenset(
    {
        "dflash",
        "embed",
        "full_attn",
        "head",
        "linear_attn",
        "moe",
        "mtp",
        "util",
    }
)


def validate_build_path(component=None):
    """Return the fixed build root or component after rejecting path escapes."""
    repo_root = Path(REPO_ROOT)
    expected_root = repo_root / "build" / "sm90a"
    configured_root = Path(BUILD_ROOT)
    if configured_root != expected_root:
        raise ValueError(f"build root must be exactly {expected_root}, got {configured_root}")

    if component is not None:
        component = str(component)
        if component not in KERNEL_COMPONENTS:
            raise ValueError(f"unrecognized kernel build component: {component!r}")

    build_parent = repo_root / "build"
    selected = expected_root if component is None else expected_root / component
    candidates = [build_parent, expected_root]
    if component is not None:
        candidates.append(selected)
    for candidate in candidates:
        if candidate.is_symlink():
            raise RuntimeError(f"refusing symlinked build path: {candidate}")
        if candidate.exists() and not candidate.is_dir():
            raise RuntimeError(f"build path is not a directory: {candidate}")

    resolved_repo = repo_root.resolve()
    resolved_root = expected_root.resolve()
    try:
        resolved_root.relative_to(resolved_repo)
    except ValueError as err:
        raise RuntimeError(
            f"build root resolves outside repository: {expected_root} -> {resolved_root}"
        ) from err
    if component is not None:
        resolved_selected = selected.resolve()
        if resolved_selected.parent != resolved_root:
            raise RuntimeError(
                f"build component resolves outside fixed root: {selected} -> {resolved_selected}"
            )
    return selected


def _source_mtime(path, seen=None):
    """Return the newest mtime in a local quoted-include tree."""
    seen = set() if seen is None else seen
    path = path.resolve()
    if path in seen:
        return 0.0
    seen.add(path)
    try:
        newest = path.stat().st_mtime
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return float("inf")
    for match in re.finditer(r'^\s*#\s*include\s+"([^"]+)"', text, re.MULTILINE):
        dependency = path.parent / match.group(1)
        # Conditional profiler/debug includes may be absent in production builds.
        if dependency.is_file():
            newest = max(newest, _source_mtime(dependency, seen))
    return newest


def _freshness_mtime(src):
    """Return the newest source, quoted-header, or build-helper mtime."""
    try:
        helper_mtime = BUILD_HELPER.stat().st_mtime
    except (FileNotFoundError, OSError):
        return float("inf")
    return max(_source_mtime(src), helper_mtime)


def _fresh(src, so):
    try:
        if so.stat().st_mtime < _freshness_mtime(src):
            return False
        with so.open("rb") as handle:
            return handle.read(4) == b"\x7fELF"
    except (FileNotFoundError, OSError):
        return False


def _kernel_paths(kdir, name):
    """Resolve one direct kernel component source and its fixed build output."""
    kdir = Path(kdir).resolve()
    kernel_root = KERNEL_ROOT.resolve()
    if kdir.parent != kernel_root or kdir.name not in KERNEL_COMPONENTS:
        raise ValueError(
            f"kernel directory must be a recognized direct child of {kernel_root}: {kdir}"
        )
    name = str(name)
    if not name or Path(name).name != name:
        raise ValueError(f"kernel name must be a source stem, got {name!r}")
    src = kdir / f"{name}.cu"
    if not src.is_file():
        raise FileNotFoundError(f"CUDA source does not exist: {src}")
    so = validate_build_path(kdir.name) / f"{name}.so"
    if so.is_symlink():
        raise RuntimeError(f"refusing symlinked build output: {so}")
    if so.exists() and not so.is_file():
        raise RuntimeError(f"build output is not a regular file: {so}")
    return src, so


def build_kernel(kdir, name, *, hopper=False, link_cuda=False):
    """Compile one CUDA source once across TP ranks and atomically publish its ELF."""
    src, so = _kernel_paths(kdir, name)
    if _fresh(src, so):
        return _kernel_paths(kdir, name)[1]

    # Validate before and after creation so no pre-existing path component can
    # redirect mkdir or later compiler output outside the repository.
    component = src.parent.name
    component_dir = validate_build_path(component)
    component_dir.mkdir(parents=True, exist_ok=True)
    so = _kernel_paths(kdir, name)[1]
    digest = hashlib.sha1(str(so).encode()).hexdigest()[:12]
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    lock = LOCK_ROOT / f"nexrt-nvcc-{digest}.lock"
    with lock.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        so = _kernel_paths(kdir, name)[1]
        if _fresh(src, so):
            return _kernel_paths(kdir, name)[1]

        tmp = so.with_name(f".{so.name}.{os.getpid()}.tmp")
        if tmp.is_symlink():
            raise RuntimeError(f"refusing symlinked temporary build output: {tmp}")
        if tmp.exists() and not tmp.is_file():
            raise RuntimeError(f"temporary build output is not a regular file: {tmp}")
        tmp.unlink(missing_ok=True)
        arch = ["-gencode", "arch=compute_90a,code=sm_90a"] if hopper else ["-arch=sm_90a"]
        command = [
            "nvcc",
            "-O3",
            "-std=c++17",
            *arch,
            "--compiler-options",
            "-fPIC",
            "-shared",
            str(src),
            "-o",
            str(tmp),
        ]
        if link_cuda:
            command.append("-lcuda")
        try:
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(
                    f"nvcc {name} failed with rc={result.returncode}\n"
                    f"{result.stdout}{result.stderr}"
                )
            so = _kernel_paths(kdir, name)[1]
            if tmp.is_symlink() or not tmp.is_file():
                raise RuntimeError(f"compiler did not create a regular temporary file: {tmp}")
            os.replace(tmp, so)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    return _kernel_paths(kdir, name)[1]
