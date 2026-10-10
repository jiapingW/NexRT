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

"""MTP setup uses the resident target head and norm without target files."""

import pytest
import torch
from safetensors.torch import save_file

from nex_n2_5_pro.mtp import weights


def test_resident_target_weights(tmp_path, monkeypatch):
    torch.manual_seed(4004)
    tensors = {
        "mtp.norm.weight": torch.randn(16).bfloat16(),
        "mtp.pre_fc_norm_embedding.weight": torch.randn(16).bfloat16(),
        "mtp.pre_fc_norm_hidden.weight": torch.randn(16).bfloat16(),
    }
    checkpoint = tmp_path / "mtp.safetensors"
    save_file(tensors, str(checkpoint))
    monkeypatch.setattr(weights, "MTP_ST", checkpoint)
    head = torch.randn(8, 16).bfloat16()
    target_norm = torch.randn(16).bfloat16()
    original = head.clone()
    fused = weights.head_w_fused(head, "cpu")
    norms = weights.norm_weights("cpu", target_norm)
    expected = (original.float() * (1 + tensors["mtp.norm.weight"].float())).bfloat16()
    assert torch.equal(fused, expected)
    assert torch.equal(head, original)
    assert norms["model_norm"].data_ptr() == target_norm.data_ptr()
    assert torch.equal(norms["mtp_norm"], tensors["mtp.norm.weight"])


def test_released_target_head_is_rejected():
    with pytest.raises(RuntimeError, match="before capturing"):
        weights.head_w_fused(None, "cpu")
