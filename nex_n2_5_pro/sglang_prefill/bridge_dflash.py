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

"""Chunked target-feature prefill and live handoff for DFlash decoding."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import hashlib
import os

import torch

from nex_n2_5_pro.dflash.block import DFlashBlock
from nex_n2_5_pro.dflash.config import (
    DFLASH_DIR,
    DFlashConfig,
    M,
    SLIDING_LAYERS,
    SLIDING_RING_CAPACITY,
)
from nex_n2_5_pro.dflash.serve import generate_dflash
from nex_n2_5_pro.dflash.weights import load_dflash_shard
from nex_n2_5_pro.decode_engine import DecodeEngine
from nex_n2_5_pro.m8_cp import M8CpVerify
from nex_n2_5_pro.m8_verify import M8Verify
from nex_n2_5_pro.sglang_prefill.sglang_dflash import DFlashContextPrefill


class TargetFeatureTap:
    """Use SGLang's residual-aware feature capture, retaining only one chunk."""

    def __init__(self, model: torch.nn.Module, layers: tuple[int, ...]) -> None:
        language_model = model.model
        if not hasattr(language_model, "set_dflash_layers_to_capture"):
            raise RuntimeError("DFlash prefill requires SGLang target feature capture support")
        # SGLang captures the previous layer's residual at the next layer's input.
        language_model.set_dflash_layers_to_capture([layer + 1 for layer in layers])
        self.chunk = None
        self.armed = False

        def hook(_module, _inputs, output):
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError("SGLang did not return requested DFlash target features")
            hidden, features = output
            if self.armed:
                if self.chunk is not None or len(features) != len(layers):
                    raise RuntimeError(
                        "DFlash target feature capture did not match the current chunk"
                    )
                # Every tap, concatenated: SGLang's draft fc consumes all of them.
                self.chunk = torch.cat([f.detach() for f in features], dim=-1)
            # The ordinary SGLang logits processor consumes the final hidden tensor.
            return hidden

        self.handle = language_model.register_forward_hook(hook)

    def arm(self) -> None:
        if self.chunk is not None:
            raise RuntimeError("unconsumed DFlash target features")
        self.armed = True

    def pop(self) -> torch.Tensor:
        if self.chunk is None:
            raise RuntimeError("DFlash target feature hook did not run")
        chunk, self.chunk = self.chunk, None
        return chunk

    def abort(self) -> None:
        self.chunk = None
        self.armed = False


class DFlashBridge:
    """One resident DFlash engine; each request supplies a fresh live prefill."""

    def __init__(
        self, engine: DecodeEngine, model_runner, server_args, port_args, device: torch.device
    ) -> None:
        config = DFlashConfig.load()
        verifier_type = M8CpVerify if engine.attn_cp else M8Verify
        self.verifier = verifier_type(engine, device, dflash_tap_layers=config.target_layer_ids)
        weights = load_dflash_shard(engine.rank, device)
        self.block = DFlashBlock(engine, self.verifier, device, config, weights)
        self.verifier.build_packs()
        # SGLang's DFlash draft model computes the prompt context rows.
        self.prefiller = DFlashContextPrefill(
            model_runner, server_args, port_args, engine.rank, DFLASH_DIR, self.block, device
        )
        self.tap = TargetFeatureTap(model_runner.model, config.target_layer_ids)
        self.config = config
        self.rank = engine.rank
        self.graphs = {}

    def begin(self, length: int) -> None:
        self.prefiller.begin(length)
        self.tap.arm()

    def consume(self, start: int, end: int, _next_token: int) -> None:
        self.prefiller.consume(self.tap.pop(), start, end)

    def end(self) -> None:
        self.tap.abort()
        if self.prefiller.position != self.prefiller.length:
            raise RuntimeError("DFlash prompt context is incomplete")
        if os.environ.get("NEXRT_DFLASH_FINGERPRINT") == "1":
            self._print_context_fingerprint()

    def _print_context_fingerprint(self) -> None:
        """Per-layer hashes of the live prompt rows (opt-in repeatability evidence)."""
        torch.cuda.synchronize()
        state, length = self.block.state, self.prefiller.length
        for layer, (key, value) in enumerate(zip(state.key_cache, state.value_cache)):
            if layer < SLIDING_LAYERS:
                start = max(0, length - self.config.sliding_window)
                rows = torch.arange(start, length, device=key.device) % SLIDING_RING_CAPACITY
            else:
                rows = torch.arange(0, length, device=key.device)
            digests = [
                hashlib.sha256(
                    t.index_select(0, rows).contiguous().view(torch.uint8).cpu().numpy().tobytes()
                ).hexdigest()[:16]
                for t in (key, value)
            ]
            print(
                f"@@ R{self.rank} DFLASH_CTX L={length} layer{layer} k={digests[0]} v={digests[1]}",
                flush=True,
            )

    def capture(
        self, length: int, first_token: int, restore_target: Callable[[], None]
    ) -> torch.cuda.CUDAGraph:
        verifier = self.verifier
        use_cp = bool(verifier.eng.attn_cp) and length >= 131072
        if hasattr(verifier, "use_cp"):
            verifier.use_cp = use_cp
        verifier.seed(length, [first_token] + [0] * (M - 1))
        if use_cp not in self.graphs:
            self.graphs[use_cp] = self.block.capture(warmup=3)
            restore_target()
            verifier.seed(length, [first_token] + [0] * (M - 1))
        # Prompt KV survives capture: writes start at length, and the sliding
        # ring has 4096 extra rows beyond its visible window.
        return self.graphs[use_cp]

    def generate(
        self,
        graph: torch.cuda.CUDAGraph,
        max_new: int,
        stop_ids: Iterable[int],
        verify: bool,
        on_tokens=None,
    ) -> tuple[list[int], list[int], int, int]:
        checks = 0

        def observe(before, after):
            nonlocal checks
            target = self.verifier.sv_target_ids.cpu().tolist()
            proposed = after["verify_input_ids"]
            accepted = 0
            for proposal, expected in zip(proposed[1:], target[:-1]):
                if proposal != expected:
                    break
                accepted += 1
            emitted = proposed[1 : accepted + 1] + [target[accepted]]
            if not (
                after["accepted_draft_count"] == accepted
                and after["commit_count"] == len(emitted)
                and after["emitted_ids"][: len(emitted)] == emitted
                and after["pending_target_id"] == emitted[-1]
                and after["committed_len"] == before["committed_len"] + len(emitted)
            ):
                raise RuntimeError("DFlash target commit disagrees with CPU greedy resolution")
            checks += 1

        tokens, accepts, status = generate_dflash(
            self.block,
            graph,
            max_new,
            eos_ids=tuple(stop_ids),
            on_round=observe if verify else None,
            on_tokens=on_tokens,
        )
        return tokens, accepts, status, checks
