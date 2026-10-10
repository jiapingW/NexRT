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

"""Host-only tests for fixed-path, race-safe CUDA JIT publication."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest


from nex_n2_5_pro import kernel_build
from nex_n2_5_pro.tests import build_sweep


@pytest.fixture
def build_tree(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    kernel_root = repo_root / "nex_n2_5_pro" / "kernels"
    component = kernel_root / "moe"
    component.mkdir(parents=True)
    build_root = repo_root / "build" / "sm90a"
    helper = repo_root / "nex_n2_5_pro" / "kernel_build.py"
    helper.write_text("# test build helper\n", encoding="utf-8")
    monkeypatch.setattr(kernel_build, "REPO_ROOT", repo_root)
    monkeypatch.setattr(kernel_build, "KERNEL_ROOT", kernel_root)
    monkeypatch.setattr(kernel_build, "BUILD_ROOT", build_root)
    monkeypatch.setattr(kernel_build, "BUILD_HELPER", helper)
    monkeypatch.setattr(kernel_build, "LOCK_ROOT", tmp_path / "locks")
    return component, build_root, helper


def _source(component, text="// test kernel\n"):
    source = component / "example.cu"
    source.write_text(text, encoding="utf-8")
    return source


def _compiler(result_bytes=b"\x7fELFtest", returncode=0):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        output = Path(command[command.index("-o") + 1])
        output.write_bytes(result_bytes)
        return subprocess.CompletedProcess(command, returncode, "stdout", "stderr")

    return calls, run


def _set_mtime(path, nanoseconds):
    os.utime(path, ns=(nanoseconds, nanoseconds))


def test_kernel_maps_to_fixed_component_path(build_tree, monkeypatch):
    component, build_root, _helper = build_tree
    _source(component)
    _calls, compile_success = _compiler()
    monkeypatch.setattr(kernel_build.subprocess, "run", compile_success)

    result = kernel_build.build_kernel(component, "example")

    assert result == build_root / "moe" / "example.so"


def test_source_outside_recognized_kernel_tree_is_rejected(build_tree):
    _component, _build_root, _helper = build_tree
    outside = _component.parent.parent / "outside"
    outside.mkdir()
    _source(outside)

    with pytest.raises(ValueError, match="recognized direct child"):
        kernel_build.build_kernel(outside, "example")


def test_symlinked_build_directory_is_rejected_before_output(build_tree, monkeypatch):
    component, build_root, _helper = build_tree
    _source(component)
    external = Path(kernel_build.REPO_ROOT).parent / "external-build"
    external.mkdir()
    build_root.parent.symlink_to(external, target_is_directory=True)

    def unexpected_compile(*_args, **_kwargs):
        raise AssertionError("symlinked build path invoked the compiler")

    monkeypatch.setattr(kernel_build.subprocess, "run", unexpected_compile)
    with pytest.raises(RuntimeError, match="symlinked build path"):
        kernel_build.build_kernel(component, "example")

    assert not list(external.iterdir())
    assert not kernel_build.LOCK_ROOT.exists()


def test_symlinked_component_directory_is_rejected(build_tree, monkeypatch):
    component, build_root, _helper = build_tree
    _source(component)
    external = Path(kernel_build.REPO_ROOT).parent / "external-component"
    external.mkdir()
    build_root.mkdir(parents=True)
    (build_root / "moe").symlink_to(external, target_is_directory=True)

    def unexpected_compile(*_args, **_kwargs):
        raise AssertionError("symlinked component invoked the compiler")

    monkeypatch.setattr(kernel_build.subprocess, "run", unexpected_compile)
    with pytest.raises(RuntimeError, match="symlinked build path"):
        kernel_build.build_kernel(component, "example")

    assert not list(external.iterdir())
    assert not kernel_build.LOCK_ROOT.exists()


def test_fresh_elf_is_reused_without_compiler(build_tree, monkeypatch):
    component, build_root, helper = build_tree
    source = _source(component)
    output = build_root / "moe" / "example.so"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"\x7fELFcached")
    now = time.time_ns()
    _set_mtime(source, now)
    _set_mtime(helper, now)
    _set_mtime(output, now + 1_000_000)

    def unexpected_compile(*_args, **_kwargs):
        raise AssertionError("fresh ELF invoked the compiler")

    monkeypatch.setattr(kernel_build.subprocess, "run", unexpected_compile)
    assert kernel_build.build_kernel(component, "example") == output


def test_newer_quoted_header_makes_cached_elf_stale(build_tree, monkeypatch):
    component, build_root, helper = build_tree
    source = _source(component, '#include "local.cuh"\n')
    header = component / "local.cuh"
    header.write_text("// dependency\n", encoding="utf-8")
    output = build_root / "moe" / "example.so"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"\x7fELFold")
    now = time.time_ns()
    _set_mtime(source, now)
    _set_mtime(helper, now)
    _set_mtime(output, now + 1_000_000)
    _set_mtime(header, now + 2_000_000)
    calls, compile_success = _compiler(b"\x7fELFnew")
    monkeypatch.setattr(kernel_build.subprocess, "run", compile_success)

    assert kernel_build.build_kernel(component, "example") == output
    assert len(calls) == 1
    assert output.read_bytes() == b"\x7fELFnew"


def test_newer_build_helper_makes_cached_elf_stale(build_tree, monkeypatch):
    component, build_root, helper = build_tree
    source = _source(component)
    output = build_root / "moe" / "example.so"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"\x7fELFold")
    now = time.time_ns()
    _set_mtime(source, now)
    _set_mtime(output, now + 1_000_000)
    _set_mtime(helper, now + 2_000_000)
    calls, compile_success = _compiler(b"\x7fELFnew")
    monkeypatch.setattr(kernel_build.subprocess, "run", compile_success)

    assert kernel_build.build_kernel(component, "example") == output
    assert len(calls) == 1
    assert output.read_bytes() == b"\x7fELFnew"


def test_compiler_failure_removes_output_and_temporary_file(build_tree, monkeypatch):
    component, build_root, _helper = build_tree
    _source(component)
    _calls, compile_failure = _compiler(b"partial", returncode=1)
    monkeypatch.setattr(kernel_build.subprocess, "run", compile_failure)

    with pytest.raises(RuntimeError, match="nvcc example failed with rc=1"):
        kernel_build.build_kernel(component, "example")

    output_dir = build_root / "moe"
    assert not (output_dir / "example.so").exists()
    assert not list(output_dir.glob(".*.tmp"))


def test_successful_build_atomically_publishes_final_path(build_tree, monkeypatch):
    component, build_root, _helper = build_tree
    _source(component)
    observed_outputs = []

    def compile_success(command, **_kwargs):
        temporary = Path(command[command.index("-o") + 1])
        observed_outputs.append(temporary)
        assert temporary.parent == build_root / "moe"
        assert temporary.name.startswith(".example.so.")
        assert not (build_root / "moe" / "example.so").exists()
        temporary.write_bytes(b"\x7fELFpublished")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(kernel_build.subprocess, "run", compile_success)
    result = kernel_build.build_kernel(component, "example")

    assert observed_outputs
    assert result == build_root / "moe" / "example.so"
    assert result.read_bytes() == b"\x7fELFpublished"
    assert not observed_outputs[0].exists()


def test_safe_clean_removes_generated_artifacts(build_tree):
    _component, build_root, _helper = build_tree
    moe = build_root / "moe"
    head = build_root / "head"
    moe.mkdir(parents=True)
    head.mkdir()
    (moe / "example.so").write_bytes(b"\x7fELFartifact")
    (moe / ".example.so.123.tmp").write_bytes(b"temporary")
    (head / "other.so").write_bytes(b"\x7fELFartifact")

    build_sweep.clean_build_artifacts()

    assert not build_root.exists()
    assert not build_root.parent.exists()


def test_safe_clean_is_noop_when_build_tree_is_absent(build_tree):
    _component, build_root, _helper = build_tree

    build_sweep.clean_build_artifacts()

    assert not build_root.exists()


def test_safe_clean_rejects_unexpected_regular_file(build_tree):
    _component, build_root, _helper = build_tree
    output_dir = build_root / "moe"
    output_dir.mkdir(parents=True)
    artifact = output_dir / "example.so"
    unexpected = output_dir / "notes.txt"
    artifact.write_bytes(b"\x7fELFartifact")
    unexpected.write_text("do not delete\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unexpected non-artifact"):
        build_sweep.clean_build_artifacts()

    assert artifact.is_file()
    assert unexpected.read_text(encoding="utf-8") == "do not delete\n"


@pytest.mark.parametrize("location", ["build", "sm90a", "component", "content"])
def test_safe_clean_rejects_symlink_and_preserves_external_target(build_tree, location):
    _component, build_root, _helper = build_tree
    external = Path(kernel_build.REPO_ROOT).parent / "external-clean"
    external.mkdir()
    marker = external / "keep.txt"
    marker.write_text("preserve me\n", encoding="utf-8")

    if location == "build":
        link = build_root.parent
        link.symlink_to(external, target_is_directory=True)
    elif location == "sm90a":
        build_root.parent.mkdir()
        link = build_root
        link.symlink_to(external, target_is_directory=True)
    elif location == "component":
        build_root.mkdir(parents=True)
        link = build_root / "moe"
        link.symlink_to(external, target_is_directory=True)
    else:
        output_dir = build_root / "moe"
        output_dir.mkdir(parents=True)
        link = output_dir / "escape.so"
        link.symlink_to(marker)

    with pytest.raises(RuntimeError, match="symlink"):
        build_sweep.clean_build_artifacts()

    assert link.is_symlink()
    assert marker.read_text(encoding="utf-8") == "preserve me\n"
