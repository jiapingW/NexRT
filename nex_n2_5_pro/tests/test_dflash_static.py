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

"""CPU-only checks for the public DFlash geometry, pack law, and source surface."""

from __future__ import annotations

from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

from nex_n2_5_pro import m8_control as control  # noqa: E402
from nex_n2_5_pro.dflash.config import DFlashConfig, LAYERS, SLIDING_LAYERS, round_positions  # noqa: E402
from nex_n2_5_pro.dflash.serve import generate_dflash  # noqa: E402
from nex_n2_5_pro.dflash.weights import pack_m8_bf16  # noqa: E402


def _config() -> DFlashConfig:
    return DFlashConfig(
        hidden_size=4096,
        num_layers=6,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=12288,
        rms_norm_eps=1e-6,
        attn_rms_eps=1e-6,
        hidden_rms_eps=1e-6,
        qk_norm_eps=1e-6,
        rope_theta=10_000_000.0,
        sliding_window=4096,
        layer_types=("sliding_attention",) * SLIDING_LAYERS + ("full_attention",),
        max_position_embeddings=262144,
        vocab_size=248320,
        block_size=16,
        mask_token_id=248077,
        target_layer_ids=(1, 9, 17, 25, 33, 41, 49, 57),
        eos_token_id=248044,
    )


def test_config_and_positions() -> None:
    config = _config()
    config.validate_kernel_contract("unit config")
    assert config.cache_capacity == 262144 + 1024
    context, draft = round_positions(5, 8197)
    assert list(context) == [8192, 8193, 8194, 8195, 8196]
    assert list(draft) == list(range(8197, 8197 + 8))
    empty_context, first_draft = round_positions(0, 8192)
    assert not list(empty_context)
    assert list(first_draft) == list(range(8192, 8200))


def test_m8_pack_round_trip() -> None:
    source = torch.arange(16 * 32, dtype=torch.float32).to(torch.bfloat16).view(16, 32)
    packed = pack_m8_bf16(source)
    restored = (
        packed.view(1, 2, 8, 4, 2, 2, 2).permute(0, 5, 2, 1, 4, 3, 6).contiguous().view_as(source)
    )
    assert torch.equal(restored, source)


def test_public_kernel_surface() -> None:
    kernel_dir = ROOT / "kernels" / "dflash"
    sources = sorted(kernel_dir.glob("*.cu"))
    assert {source.name for source in sources} == {
        "nex_n2_5_pro_dflash_argmax_ctypes.cu",
        "nex_n2_5_pro_dflash_attn_wgmma_ctypes.cu",
        "nex_n2_5_pro_dflash_glue_ctypes.cu",
        "nex_n2_5_pro_dflash_head_ctypes.cu",
        "nex_n2_5_pro_dflash_mlp_tc_ctypes.cu",
        "nex_n2_5_pro_dflash_oproj_tc_ctypes.cu",
        "nex_n2_5_pro_dflash_qkv_ckv_front_ctypes.cu",
        "nex_n2_5_pro_dflash_target_hidden_proj_ctypes.cu",
    }
    text = "\n".join(source.read_text(encoding="utf-8") for source in sources)
    expected = (
        "NexN25ProDFlashNoiseEmbedLaunch",
        "NexN25ProDFlashTargetHiddenProjGemmLaunch",
        "NexN25ProDFlashQkvProjLaunch",
        "NexN25ProDFlashContextKvProjLaunch",
        "NexN25ProDFlashAttnWgmmaPartialCtlLaunch",
        "NexN25ProDFlashOProjTcGemmLaunch",
        "NexN25ProDFlashMlpGateUpGemmLaunch",
        "NexN25ProDFlashHeadGemmLaunch",
        "NexN25ProDFlashArgmaxLLLaunch",
    )
    for symbol in expected:
        assert symbol in text, symbol
    for forbidden in (
        "north" + "rt",
        "NORTH" + "RT",
        "q" + "wen35",
        "Q" + "wenDf",
        "split" + "8192",
        "global" + "timer",
        "stam" + "ps",
    ):
        assert forbidden not in text, forbidden
    tap = (ROOT / "kernels" / "embed" / "nex_n2_5_pro_m8_feature_tap_ctypes.cu").read_text(
        encoding="utf-8"
    )
    assert "NexN25ProM8FeatTapLaunch" in tap
    assert "NexN25ProM8FeatTapPdlLaunch" not in tap

    runtime = "\n".join(
        (ROOT / "dflash" / name).read_text(encoding="utf-8") for name in ("block.py", "ll.py")
    )
    for collective in ("all_reduce(", "all_gather(", "reduce_scatter("):
        assert collective not in runtime
    assert LAYERS == 6


def test_serve_boundaries() -> None:
    class Verifier:
        def __init__(self):
            self.ctl = torch.zeros(control.WORDS, dtype=torch.int32)
            self.ctl[control.COMMITTED_LEN] = 100

    class Block:
        def __init__(self):
            self.config = _config()
            self.cache_capacity = self.config.cache_capacity
            self.verifier = Verifier()

        @staticmethod
        def status() -> int:
            raise AssertionError("serving must use the folded M=8 status word")

    class Graph:
        def __init__(self, block, rounds):
            self.block = block
            self.rounds = iter(rounds)
            self.replays = 0

        def replay(self):
            emitted = next(self.rounds)
            self.replays += 1
            ctl = self.block.verifier.ctl
            count = len(emitted)
            ctl[control.COMMIT_COUNT] = count
            ctl[control.OUTPUT_COUNT] = count
            ctl[control.EMITTED_IDS : control.EMITTED_IDS + count] = torch.tensor(
                emitted, dtype=torch.int32
            )
            ctl[control.COMMITTED_LEN] += count

    block = Block()
    graph = Graph(block, ([11, 12, 13], [14, 15, 16]))
    emitted_chunks = []
    tokens, accepts, status = generate_dflash(
        block,
        graph,
        max_new_tokens=6,
        eos_ids=(),
        on_tokens=lambda chunk: emitted_chunks.append(list(chunk)),
    )
    assert tokens == [11, 12, 13, 14, 15]
    assert emitted_chunks == [[11, 12, 13], [14, 15]]
    assert accepts == [3, 3] and status == 0 and graph.replays == 2

    block = Block()
    graph = Graph(
        block,
        (
            [21, 248046, 23],
            [24],
        ),
    )
    tokens, accepts, status = generate_dflash(block, graph, max_new_tokens=9, eos_ids=(248046,))
    assert tokens == [21, 248046]
    assert accepts == [3] and status == 0 and graph.replays == 1

    for position in range(8):
        block = Block()
        emitted = list(range(10, 18))
        emitted[position] = 248046
        graph = Graph(block, (emitted,))
        tokens, _, status = generate_dflash(block, graph, 16, eos_ids=(248046,))
        assert tokens == emitted[: position + 1] and status == 0
        assert graph.replays == 1

    block = Block()
    block.verifier.ctl[control.PENDING_TARGET_ID] = 248046
    graph = Graph(block, ())
    assert generate_dflash(block, graph, 16, eos_ids=(248046,)) == ([], [], 0)
    assert graph.replays == 0

    block = Block()
    block.verifier.ctl[control.COMMITTED_LEN] = 200
    graph = Graph(block, ([31],))
    tokens, accepts, status = generate_dflash(
        block, graph, max_new_tokens=9, context_limit=207, eos_ids=()
    )
    assert not tokens and not accepts and status == 0 and graph.replays == 0


def main() -> None:
    test_config_and_positions()
    test_m8_pack_round_trip()
    test_public_kernel_surface()
    test_serve_boundaries()
    print("DFLASH STATIC PASS", flush=True)


if __name__ == "__main__":
    main()
