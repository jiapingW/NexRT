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

"""Scale aliases retain dequantization values at the checkpoint boundary."""

import pytest
import torch
from safetensors.torch import save_file

from nex_n2_5_pro import weights


@pytest.mark.parametrize("suffix", ["weight_scale", "weight_scale_inv"])
def test_attention_scale_names(tmp_path, monkeypatch, suffix):
    prefix = "model.language_model.layers.3.self_attn."
    tensors = {}
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        tensors[prefix + name + ".weight"] = torch.zeros(128, 128).bfloat16()
        tensors[prefix + name + "." + suffix] = torch.tensor([[0.125]])
    for name in ("q_norm", "k_norm"):
        tensors[prefix + name + ".weight"] = torch.ones(128).bfloat16()
    save_file(tensors, str(tmp_path / "layer.3.safetensors"))
    monkeypatch.setattr(weights, "LAYERS_DIR", tmp_path)
    result = weights.load_attention_weights(3, "cpu")
    for name in ("qs", "ks", "vs", "os"):
        assert result[name].dtype == torch.float32
        assert result[name].item() == 0.125
