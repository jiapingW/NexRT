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

"""Host-only validation of TP worker configuration and shutdown."""

import ast
from contextlib import nullcontext, suppress
from multiprocessing import shared_memory
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from nex_n2_5_pro.sglang_prefill.process_supervisor import torchrun_rank


def test_plain_python_requires_torchrun(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE", "custom://")
    with pytest.raises(ValueError, match="torchrun --standalone --nproc-per-node=8"):
        torchrun_rank(8)
    assert os.environ["SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE"] == "custom://"


@pytest.mark.parametrize("rank", range(8))
def test_torchrun_uses_the_existing_rank_and_store(monkeypatch, rank):
    for key, value in dict(LOCAL_RANK=rank, RANK=rank, WORLD_SIZE=8, LOCAL_WORLD_SIZE=8).items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.delenv("SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE", raising=False)
    assert torchrun_rank(8) == rank
    assert os.environ["SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE"] == "env://"


@pytest.mark.parametrize(
    "override",
    [
        {"WORLD_SIZE": 16},
        {"LOCAL_WORLD_SIZE": 4},
        {"RANK": 8},
        {"LOCAL_RANK": 8},
    ],
)
def test_rejects_incompatible_worker_groups(monkeypatch, override):
    values = dict(LOCAL_RANK=0, RANK=0, WORLD_SIZE=8, LOCAL_WORLD_SIZE=8)
    values.update(override)
    for key, value in values.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.delenv("SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE", raising=False)
    with pytest.raises(ValueError, match="one node with 8"):
        torchrun_rank(8)
    assert "SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE" not in os.environ


@pytest.fixture
def worker_exit():
    # Isolate the production exit path without importing the GPU runtime.
    source = Path(__file__).parents[1] / "sglang_prefill" / "bridge_serve.py"
    module = ast.parse(source.read_text())
    main = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    exit_block = main.body[-1]
    assert isinstance(exit_block, ast.Try) and exit_block.finalbody
    code = compile(ast.Module(body=[exit_block], type_ignores=[]), str(source), "exec")

    def run(worker, destroy_model, destroy_environment, initialized=True):
        namespace = dict.fromkeys(("server_args", "port_args", "rank", "cli"))
        namespace.update(
            worker=lambda *_args: worker(),
            torch=SimpleNamespace(
                distributed=SimpleNamespace(
                    is_initialized=lambda: initialized, destroy_process_group=lambda: None
                )
            ),
            destroy_model_parallel=destroy_model,
            destroy_distributed_environment=destroy_environment,
        )
        exec(code, namespace)

    return run


@pytest.mark.parametrize("error_type", (None, RuntimeError, KeyboardInterrupt))
def test_worker_exit_unlinks_owned_memory_before_destroying_environment(worker_exit, error_type):
    memory = shared_memory.SharedMemory(create=True, size=1)
    events = []

    def worker():
        if error_type is not None:
            raise error_type("worker stopped")

    def destroy_model():
        memory.close()
        memory.unlink()
        events.append("model")

    def destroy_environment():
        assert events == ["model"]
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=memory.name)
        events.append("environment")

    try:
        with pytest.raises(error_type, match="worker stopped") if error_type else nullcontext():
            worker_exit(worker, destroy_model, destroy_environment)
        assert events == ["model", "environment"]
    finally:
        memory.close()
        with suppress(FileNotFoundError):
            memory.unlink()


def test_worker_exit_destroys_environment_if_model_cleanup_fails(worker_exit):
    events = []

    def destroy_model():
        raise RuntimeError("cleanup failed")

    with pytest.raises(RuntimeError, match="cleanup failed"):
        worker_exit(lambda: None, destroy_model, lambda: events.append("environment"))
    assert events == ["environment"]


def test_worker_exit_before_distributed_initialization(worker_exit):
    def must_not_run():
        raise AssertionError("cleanup called before distributed initialization")

    worker_exit(lambda: None, must_not_run, must_not_run, initialized=False)
