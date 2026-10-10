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

"""M8 MoE reads row-major expert weights without allocating weight packs."""

import ctypes
from types import SimpleNamespace

import pytest
import torch

from nex_n2_5_pro.kernel_build import KERNEL_ROOT, build_kernel
from nex_n2_5_pro.m8_verify import M8Verify


def quantized_input(x, *, normalize=False):
    """Run the fused production launch and check its independent BF16 output."""
    top = ctypes.CDLL(str(build_kernel(KERNEL_ROOT / "moe", "nex_n2_5_pro_m8_gate_topk_ctypes")))
    top.NexN25ProM8GateTopKPackedKeyLaunch.argtypes = [ctypes.c_void_p] * 9
    logits = torch.zeros(8, 528, dtype=torch.bfloat16, device=x.device)
    rms = (
        torch.linspace(0.5, 1.5, 8, device=x.device)
        if normalize
        else torch.ones(8, device=x.device)
    )
    gamma = (
        torch.linspace(0.5, 1.5, 4096, device=x.device)
        if normalize
        else torch.ones(4096, device=x.device)
    )
    packed = torch.empty(8, 64, dtype=torch.int32, device=x.device)
    normalized = torch.full_like(x, float("nan"))
    codes = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(8, 32, device=x.device)
    args = [
        ctypes.c_void_p(v.data_ptr())
        for v in (logits, rms, x, gamma, packed, normalized, codes, scales)
    ]
    graph = torch.cuda.CUDAGraph()
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    assert top.NexN25ProM8GateTopKPackedKeyLaunch(*args, stream) == 0
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        assert top.NexN25ProM8GateTopKPackedKeyLaunch(*args, stream) == 0
    for _ in range(3):
        normalized.fill_(float("nan"))
        scales.fill_(float("nan"))
        graph.replay()
    torch.cuda.synchronize()
    expected = (x.float() * rms[:, None] * gamma).bfloat16()
    assert torch.equal(normalized, expected)
    assert torch.equal(packed[:, :10], torch.arange(10, device=x.device).expand(8, -1))
    return codes, scales, expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@torch.inference_mode()
def test_moe_input_quantization_matches_sglang():
    from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

    torch.manual_seed(9)
    x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
    x[0].zero_()
    x[1].fill_(1e-12)
    x[2].fill_(-1e-12)
    x[3].mul_(100)
    x[4].mul_(0.01)
    codes, scales, normalized = quantized_input(x, normalize=True)
    expected_codes, expected_scales = sglang_per_token_group_quant_fp8(normalized, 128)
    assert torch.equal(codes.view(torch.uint8), expected_codes.view(torch.uint8))
    assert torch.equal(scales, expected_scales)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@torch.inference_mode()
def test_row_major_moe_pair():
    torch.manual_seed(20260905)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    libraries = [
        ctypes.CDLL(str(build_kernel(KERNEL_ROOT / "moe", name, hopper=True, link_cuda=True)))
        for name in ("nex_n2_5_pro_m8_moe_wm16_ctypes", "nex_n2_5_pro_m8_moe_l2tc_ctypes")
    ]
    l1, l2 = libraries
    l1.NexN25ProM8WmPlanLaunch.argtypes = [ctypes.c_void_p] * 3
    l1.NexN25ProM8WmMapCreate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    l1.NexN25ProM8WmMapDestroy.argtypes = [ctypes.c_void_p]
    l1.NexN25ProM8WmMapDestroy.restype = None
    l1.NexN25ProM8Wm4L1Launch.argtypes = [ctypes.c_void_p] * 12
    assert l1.NexN25ProM8WmPrepare() == 0
    l2.NexN25ProM8MoeL2LocalLaunch.argtypes = [ctypes.c_void_p] * 9

    def pointer(tensor):
        return ctypes.c_void_p(tensor.data_ptr())

    def fp8(shape):
        return (torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.025).to(
            torch.float8_e4m3fn
        )

    weights = {
        "w13": fp8((512, 256, 4096)),
        "W2": fp8((512, 4096, 128)),
        "shared_gate_up": fp8((256, 4096)),
        "shared_W2": fp8((4096, 128)),
    }
    map_handle = ctypes.c_void_p()
    assert l1.NexN25ProM8WmMapCreate(pointer(weights["w13"]), ctypes.byref(map_handle)) == 0
    verifier = M8Verify.__new__(M8Verify)
    verifier.eng = SimpleNamespace(moe={0: SimpleNamespace(k=weights)})
    before = torch.cuda.memory_allocated()
    assert verifier._wm_of(0)["w13"] is weights["w13"]
    assert verifier._wm_of(0)["shared_w13"] is weights["shared_gate_up"]
    assert verifier._l2_of(0)["w2"] is weights["W2"]
    assert verifier._l2_of(0)["shared_w2"] is weights["shared_W2"]
    assert torch.cuda.memory_allocated() == before
    scales = torch.rand(512, 64, device=device) * 0.5 + 0.5
    down_scales = torch.rand(512, 32, device=device) * 0.5 + 0.5
    shared_scales = torch.rand(64, device=device) * 0.5 + 0.5
    shared_down_scales = torch.rand(32, device=device) * 0.5 + 0.5
    x = torch.randn(8, 4096, device=device, dtype=torch.bfloat16)
    # Exercise the production fused top-k/quantization launch, then use frozen
    # routing below to cover every descriptor and token-slot ownership pattern.
    xq, xs, _ = quantized_input(x)
    plan = torch.zeros(160, device=device, dtype=torch.int32)

    def relative_error(actual, expected):
        return float((actual - expected).norm() / expected.norm().clamp_min(1e-20))

    for pattern in ("overlap", "disjoint", "rotated", "mixed"):
        if pattern == "overlap":
            ids = torch.arange(10, device=device).expand(8, -1)
        elif pattern == "disjoint":
            ids = torch.stack(
                [(torch.arange(10, device=device) + row * 61) % 512 for row in range(8)]
            )
        elif pattern == "rotated":
            ids = torch.stack(
                [torch.roll(torch.arange(10, device=device), row) for row in range(8)]
            )
        else:
            ids = torch.stack([torch.randperm(25, device=device)[:10] for _ in range(8)])
        route_weights = torch.rand(8, 10, device=device)
        route_weights /= route_weights.sum(-1, keepdim=True)
        route_weights[0, 0] = 0  # A zero-weight route still owns its output slot.
        routes = torch.zeros(8, 64, device=device, dtype=torch.int32)
        routes[:, :10] = ids
        routes[:, 10:20] = route_weights.view(torch.int32)
        routes[:, 63] = torch.tensor(0.5, device=device).view(torch.int32)
        hidden_storage = torch.full((8 * 1408 + 64,), 12345.0, device=device)
        output_storage = torch.full((8 * 11 * 4096 + 64,), 12345.0, device=device)
        hidden = hidden_storage[32:-32].view(8, 1408)
        output = output_storage[32:-32].view(8, 11, 4096)
        hidden.fill_(float("nan"))
        output.fill_(float("nan"))
        graph = torch.cuda.CUDAGraph()

        def launch():
            stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
            assert l1.NexN25ProM8WmPlanLaunch(pointer(routes), pointer(plan), stream) == 0
            assert (
                l1.NexN25ProM8Wm4L1Launch(
                    pointer(plan),
                    pointer(routes),
                    pointer(x),
                    pointer(weights["w13"]),
                    pointer(scales),
                    pointer(weights["shared_gate_up"]),
                    pointer(shared_scales),
                    pointer(hidden),
                    pointer(xq),
                    pointer(xs),
                    map_handle,
                    stream,
                )
                == 0
            )
            assert (
                l2.NexN25ProM8MoeL2LocalLaunch(
                    pointer(plan),
                    pointer(routes),
                    pointer(hidden),
                    pointer(weights["W2"]),
                    pointer(down_scales),
                    pointer(weights["shared_W2"]),
                    pointer(shared_down_scales),
                    pointer(output),
                    stream,
                )
                == 0
            )

        # Initialize the launchers before capture (L1 sets its dynamic shared-memory limit).
        launch()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            launch()
        for _ in range(3):
            hidden.fill_(float("nan"))
            output.fill_(float("nan"))
            graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(hidden).all() and torch.isfinite(output).all()
        for storage in (hidden_storage, output_storage):
            assert (storage[:32] == 12345).all() and (storage[-32:] == 12345).all()
        # Independent FP32 matrix products for every routed and shared result.
        for token in range(8):
            for slot in range(11):
                if slot == 10:
                    w13, s13 = weights["shared_gate_up"], shared_scales
                    w2, s2, route = weights["shared_W2"], shared_down_scales, 0.5
                else:
                    expert = int(ids[token, slot])
                    w13, s13 = weights["w13"][expert], scales[expert]
                    w2, s2 = weights["W2"][expert], down_scales[expert]
                    route = route_weights[token, slot]
                source = (
                    x[token].float()
                    if slot == 10
                    else (xq[token].float() * xs[token].repeat_interleave(128))
                )
                gate = (w13[:128].float() * s13[:32].repeat_interleave(128)) @ source
                up = (w13[128:].float() * s13[32:].repeat_interleave(128)) @ source
                if slot < 10:
                    gate, up = gate.bfloat16().float(), up.bfloat16().float()
                    expected = (torch.nn.functional.silu(gate) * up).bfloat16().float()
                else:
                    expected = torch.nn.functional.silu(gate) * up * route
                actual = hidden[token, slot * 128 : (slot + 1) * 128]
                assert relative_error(actual, expected) < 0.005
                down_input = actual.bfloat16().float()
                if slot < 10:
                    assert torch.equal(actual, down_input)
                    amax = down_input.abs().amax().clamp_min(1e-10)
                    scale = amax * (1.0 / 448.0)
                    down_input = (down_input * (448.0 / amax)).to(torch.float8_e4m3fn).float()
                    expected_down = (
                        ((w2.float() @ down_input) * (s2.repeat_interleave(128) * scale) * route)
                        .bfloat16()
                        .float()
                    )
                    assert torch.equal(output[token, slot], output[token, slot].bfloat16().float())
                else:
                    expected_down = (w2.float() * s2.repeat_interleave(128)[:, None]) @ down_input
                assert relative_error(output[token, slot], expected_down) < 0.003
    l1.NexN25ProM8WmMapDestroy(map_handle)
