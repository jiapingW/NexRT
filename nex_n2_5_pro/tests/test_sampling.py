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

"""Host-side validation and the fixed sampling-buffer ABI."""

import json
import struct

import pytest

from nex_n2_5_pro.sampling import (
    checked_sampling_config,
    resolve_sampling_config,
    sampling_defaults,
    sampling_words,
)


@pytest.mark.parametrize("seed", [0, 42, (1 << 63) + 123, (1 << 64) - 1, -1])
def test_fixed_buffer_layout(seed):
    words = sampling_words(seed, 0.6, 0.95, 28)
    raw = struct.pack("<8q", *words)
    assert len(raw) == 64
    assert struct.unpack_from("<Q", raw, 0)[0] == seed & ((1 << 64) - 1)
    assert struct.unpack_from("<ff", raw, 8) == pytest.approx((0.6, 0.95))
    assert struct.unpack_from("<Q", raw, 16)[0] == 28
    assert raw[24:] == bytes(40)


@pytest.mark.parametrize(
    "t,p,k",
    [
        (-1, 1, 20),
        (float("nan"), 1, 20),
        (float("inf"), 1, 20),
        (0.6, 0, 20),
        (0.6, 1.1, 20),
        (0.6, float("nan"), 20),
        (0.6, 0.95, 0),
        (0.6, 0.95, 29),
        (0.6, 0.95, 20.0),
        (0.6, 0.95, True),
        (1e100, 0.95, 20),
        (1e-100, 0.95, 20),
        (0.6, 1e-100, 20),
    ],
)
def test_invalid_parameters_are_rejected(t, p, k):
    with pytest.raises(ValueError):
        checked_sampling_config(t, p, k)


def test_model_defaults_and_request_overrides(tmp_path, monkeypatch):
    path = tmp_path / "generation_config.json"
    path.write_text(json.dumps(dict(temperature=0.7, top_p=0.9, top_k=28)))
    monkeypatch.setenv("NEXRT_MODEL_DIR", str(tmp_path))
    defaults = sampling_defaults()
    assert defaults == pytest.approx((0.7, 0.9, 28))
    assert resolve_sampling_config(defaults, temperature=0) == (0, defaults[1], 28)
    assert resolve_sampling_config(defaults) == defaults
    path.write_text(json.dumps(dict(do_sample=False, temperature=1, top_k=50)))
    assert sampling_defaults()[::2] == (0, 1)
