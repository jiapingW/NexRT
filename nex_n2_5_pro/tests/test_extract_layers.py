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

"""Host-only fixture for the layer/MTP extraction contract."""

import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import ModuleType
from unittest.mock import patch


SCRIPT = Path(__file__).parents[2] / "tools" / "extract_layers.py"


class FakeTensor:
    def __init__(self, value, dtype):
        self.value = value
        self.dtype = dtype

    def float(self):
        return FakeTensor(self.value, "float32")

    def numel(self):
        return 1

    def element_size(self):
        return {"bfloat16": 2, "float32": 4, "float64": 8}[self.dtype]


class FakeSafeOpen:
    def __init__(self, tensors):
        self.tensors = tensors

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def get_tensor(self, key):
        return self.tensors[key]


def _write_index(directory, filename, tensors):
    directory.mkdir(parents=True)
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: filename for key in tensors}}),
        encoding="utf-8",
    )


def test_optional_mtp_conversion():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source"
        mtp_source = root / "mtp-source"
        plain_output = root / "plain-output"
        mtp_output = root / "mtp-output"
        base_tensors = {
            "model.language_model.layers.3.self_attn.q_proj.weight": FakeTensor(1, "bfloat16"),
            "model.language_model.layers.3.mlp.experts.0.weight": FakeTensor(2, "bfloat16"),
            "model.language_model.layers.3.self_attn.q_proj.weight_scale_inv": FakeTensor(
                0.125, "float64"
            ),
            "model.language_model.norm.weight": FakeTensor(3, "bfloat16"),
            "model.language_model.embed_tokens.weight": FakeTensor(3, "bfloat16"),
            "lm_head.weight": FakeTensor(4, "bfloat16"),
        }
        mtp_tensors = {
            "mtp.fc.weight": FakeTensor(5, "bfloat16"),
            "mtp.fc.weight_scale_inv": FakeTensor(6, "float64"),
            "ignored.weight": FakeTensor(7, "float32"),
        }
        _write_index(source, "base.safetensors", base_tensors)
        _write_index(mtp_source, "mtp.safetensors", mtp_tensors)
        registry = {
            str(source / "base.safetensors"): base_tensors,
            str(mtp_source / "mtp.safetensors"): mtp_tensors,
        }
        saved = {}

        def safe_open(path, **_kwargs):
            return FakeSafeOpen(registry[str(Path(path))])

        def save_file(tensors, path):
            saved[str(Path(path))] = dict(tensors)

        torch_module = ModuleType("torch")
        safetensors_module = ModuleType("safetensors")
        safetensors_module.__path__ = []
        safetensors_module.safe_open = safe_open
        safetensors_torch = ModuleType("safetensors.torch")
        safetensors_torch.save_file = save_file
        safetensors_module.torch = safetensors_torch
        modules = {
            "torch": torch_module,
            "safetensors": safetensors_module,
            "safetensors.torch": safetensors_torch,
        }
        with patch.dict(sys.modules, modules):
            script = runpy.run_path(str(SCRIPT), run_name="extract_layers_test")
            with patch.object(
                sys,
                "argv",
                [str(SCRIPT), "--layers", "3", "--src", str(source), "--dst", str(plain_output)],
            ):
                script["main"]()
            assert str(plain_output / "mtp.safetensors") not in saved

            with patch.object(
                sys,
                "argv",
                [
                    str(SCRIPT),
                    "--layers",
                    "3",
                    "--src",
                    str(source),
                    "--dst",
                    str(mtp_output),
                    "--mtp-src",
                    str(mtp_source),
                ],
            ):
                script["main"]()

            before = set(saved)
            with patch.object(
                sys,
                "argv",
                [
                    str(SCRIPT),
                    "--no-cp",
                    "--src",
                    "/unused",
                    "--dst",
                    str(root / "mtp-only"),
                    "--mtp-src",
                    str(mtp_source),
                ],
            ):
                script["main"]()
            assert set(saved) - before == {str(root / "mtp-only" / "mtp.safetensors")}

            with patch.object(
                sys,
                "argv",
                [
                    str(SCRIPT),
                    "--full",
                    "--layers",
                    "3",
                    "--src",
                    str(source),
                    "--dst",
                    str(root / "full"),
                ],
            ):
                script["main"]()
            full = saved[str(root / "full" / "layer.3.safetensors")]
            assert "model.language_model.layers.3.mlp.experts.0.weight" in full
            assert all(
                str(root / "full" / name) in saved
                for name in ("embed_tokens.safetensors", "norm.safetensors", "lm_head.safetensors")
            )

        layer = saved[str(plain_output / "layer.3.safetensors")]
        assert set(layer) == {
            "model.language_model.layers.3.self_attn.q_proj.weight",
            "model.language_model.layers.3.self_attn.q_proj.weight_scale",
        }
        assert layer["model.language_model.layers.3.self_attn.q_proj.weight_scale"].value == 0.125
        assert len(saved) == 8  # default exports, MTP-only, and four full diagnostic files
        mtp = saved[str(mtp_output / "mtp.safetensors")]
        assert set(mtp) == {"mtp.fc.weight", "mtp.fc.weight_scale"}
        assert mtp["mtp.fc.weight"].dtype == "bfloat16"
        assert mtp["mtp.fc.weight_scale"].dtype == "float32"
        assert mtp["mtp.fc.weight_scale"].value == 6
