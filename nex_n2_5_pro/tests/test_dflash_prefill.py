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

from types import SimpleNamespace

import pytest
import torch

from nex_n2_5_pro.dflash.config import SLIDING_RING_CAPACITY
from nex_n2_5_pro.sglang_prefill.bridge_dflash import TargetFeatureTap
from nex_n2_5_pro.sglang_prefill.sglang_dflash import write_context_rows


def test_feature_capture_concatenates_completed_residuals():
    class LanguageModel(torch.nn.Module):
        def set_dflash_layers_to_capture(self, layers):
            self.layers = layers

        def forward(self, x):
            return x, [x + index for index in range(8)]

    model = LanguageModel()
    tap = TargetFeatureTap(SimpleNamespace(model=model), tuple(range(8)))
    tap.arm()
    x = torch.zeros(5, 16)
    assert model(x) is x
    assert model.layers == list(range(1, 9))
    assert torch.equal(tap.pop(), torch.cat([x + index for index in range(8)], dim=-1))
    tap.abort()
    model(x)
    with pytest.raises(RuntimeError, match="hook did not run"):
        tap.pop()


def test_context_rows_follow_absolute_positions():
    ring = SLIDING_RING_CAPACITY
    sliding_k = torch.zeros(ring, 4)
    sliding_v = torch.zeros(ring, 4)
    full_k = torch.zeros(2 * ring, 4)
    full_v = torch.zeros(2 * ring, 4)
    # One chunk larger than the ring crosses the physical wrap; only the newest
    # ring-capacity rows survive on a sliding layer, every row on the full layer.
    start, end = 100, 100 + ring + 300
    positions = torch.arange(start, end)
    key = positions.float()[:, None, None].expand(-1, 1, 4).clone()
    value = -key
    write_context_rows(sliding_k, sliding_v, True, positions, key, value)
    write_context_rows(full_k, full_v, False, positions, key, value)
    newest = positions[300:]
    assert torch.equal(sliding_k[newest % ring, 0], newest.float())
    assert torch.equal(sliding_v[newest % ring, 0], -newest.float())
    assert torch.equal(full_k[positions, 0], positions.float())
    assert torch.equal(full_v[positions, 0], -positions.float())
    assert not bool(full_k[:start].any())
    # A later, smaller chunk overwrites only its own ring slots.
    later = torch.arange(end, end + 8)
    write_context_rows(
        sliding_k, sliding_v, True, later, later.float()[:, None].expand(-1, 4), torch.zeros(8, 4)
    )
    assert torch.equal(sliding_k[later % ring, 0], later.float())
    untouched = torch.arange(end - ring + 8, end)
    assert torch.equal(sliding_k[untouched % ring, 0], untouched.float())
