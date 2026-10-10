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

"""M8 attention must retire shared K storage before its output transpose."""

import ctypes

import pytest
import torch

from nex_n2_5_pro import m8_control as control
from nex_n2_5_pro.kernel_build import KERNEL_ROOT, build_kernel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@torch.inference_mode()
def test_attention_graph_replay():
    torch.manual_seed(20260907)
    torch.cuda.set_device(0)
    library = ctypes.CDLL(
        str(
            build_kernel(
                KERNEL_ROOT / "full_attn",
                "nex_n2_5_pro_m8_attn_core_ctypes",
                hopper=True,
                link_cuda=True,
            )
        )
    )
    launch = library.NexN25ProM8AttnCoreLaunch
    launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_float] + [ctypes.c_void_p] * 7
    q = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(263168, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    gate = torch.randn_like(q)
    ctl = torch.zeros(control.WORDS, device="cuda", dtype=torch.int32)
    maximum = torch.empty(8 * 132 * 4, device="cuda")
    total = torch.empty_like(maximum)
    partial = torch.empty(8 * 132 * 4 * 256, device="cuda", dtype=torch.bfloat16)
    context = torch.empty_like(q)
    gated = torch.empty_like(q)

    def run():
        pointers = [ctypes.c_void_p(t.data_ptr()) for t in (q, k, v, ctl)]
        outputs = [
            ctypes.c_void_p(t.data_ptr()) for t in (maximum, total, partial, context, gate, gated)
        ]
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        assert launch(*pointers, 132, 1 / 16, *outputs, stream) == 0

    for length in (8192, 261888):
        ctl[control.COMMITTED_LEN] = length
        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        graph.replay()
        reference_context, reference_gated = context.clone(), gated.clone()
        for _ in range(32):
            context.fill_(float("nan"))
            gated.fill_(float("nan"))
            graph.replay()
            assert torch.equal(context, reference_context), f"context changed at length={length}"
            assert torch.equal(gated, reference_gated), f"gated output changed at length={length}"
        logits = (q.float() @ k[: length + 8].float().T) / 16
        positions = torch.arange(length + 8, device="cuda")
        limits = length + torch.arange(32, device="cuda") // 4
        logits.masked_fill_(positions[None, :] > limits[:, None], -torch.inf)
        expected = logits.softmax(-1) @ v[: length + 8].float()
        error = (context.float() - expected).norm() / expected.norm()
        assert float(error) < 0.006, f"attention relative L2={float(error)}"
        expected_gated = context.float() * gate.float().sigmoid()
        # Independent sigmoid implementations may straddle a BF16 rounding tie.
        torch.testing.assert_close(gated.float(), expected_gated, rtol=0.004, atol=1e-8)
