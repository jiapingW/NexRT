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

"""Checkpoint prefetch is read-only and its temporary hook is restored after loading."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from nex_n2_5_pro.sglang_prefill import weight_loading


def test_prefault_preserves_checkpoint(tmp_path):
    checkpoint = tmp_path / "weights.safetensors"
    contents = b"original checkpoint" * ((16 << 20) // 19 + 1)
    checkpoint.write_bytes(contents)
    before = checkpoint.stat()
    weight_loading._prefault_checkpoint_file(str(checkpoint))
    assert checkpoint.read_bytes() == contents
    assert checkpoint.stat().st_mtime_ns == before.st_mtime_ns
    assert list(tmp_path.iterdir()) == [checkpoint]


@pytest.mark.parametrize("facade", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_load_restores_prefetch_helper(monkeypatch, fail, facade):
    modules = {
        name: ModuleType(name)
        for name in (
            "sglang",
            "sglang.bench_one_batch",
            "sglang.srt",
            "sglang.srt.model_loader",
            "sglang.srt.model_loader.weight_utils",
        )
    }
    bench = modules["sglang.bench_one_batch"]
    utilities = modules["sglang.srt.model_loader.weight_utils"]
    original = object()
    utilities._prefetch_checkpoint_file = original
    modules["sglang"].bench_one_batch = bench
    modules["sglang"].srt = modules["sglang.srt"]
    modules["sglang.srt"].model_loader = modules["sglang.srt.model_loader"]
    modules["sglang.srt.model_loader"].weight_utils = utilities
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    args = SimpleNamespace(model_path="checkpoint", weight_loader_prefetch_checkpoints=False)
    ports, result, tokenizer = object(), object(), object()
    adapted = []
    monkeypatch.setattr(weight_loading, "_patch_fp8_scale_names", lambda path: adapted.append(path))

    def load(server_args, port_args, gpu, rank):
        assert (server_args, port_args, gpu, rank) == (args, ports, 2, 3)
        assert utilities._prefetch_checkpoint_file is weight_loading._prefault_checkpoint_file
        if fail:
            raise RuntimeError("model load failed")
        return (SimpleNamespace(torch_runner=result) if facade else result), tokenizer

    bench.load_model = load
    if fail:
        with pytest.raises(RuntimeError, match="model load failed"):
            weight_loading.load_model(args, ports, 2, 3)
    else:
        assert weight_loading.load_model(args, ports, 2, 3) == (result, tokenizer)
    assert utilities._prefetch_checkpoint_file is original
    assert args.weight_loader_prefetch_checkpoints is True
    assert adapted == ["checkpoint"]
