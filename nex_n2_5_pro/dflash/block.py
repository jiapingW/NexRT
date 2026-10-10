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

"""Complete top-1 DFlash proposal block plus M=8 target verification.

One :meth:`DFlashBlock.round` call is graph-capturable and contains:

* control-driven anchor/mask embedding and target-hidden projection;
* six draft layers (separate QKV/context-KV fronts, attention, output
  projection, and dense MLP);
* the draft lm-head and TP-wide greedy argmax;
* proposal publication followed by the complete M=8 target verifier.

The runtime has no alternative kernel arms and no Torch collective data path.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent

from nex_n2_5_pro import m8_control as control  # noqa: E402
from nex_n2_5_pro.kernel_build import build_kernel  # noqa: E402
from nex_n2_5_pro.dflash.attention import DFlashAttentionFront  # noqa: E402
from nex_n2_5_pro.dflash.config import (  # noqa: E402
    DFLASH_WEIGHTS,
    HEAD_DIM,
    HIDDEN_SIZE,
    LAYERS,
    M,
    SLIDING_LAYERS,
    SLIDING_RING_CAPACITY,
    TP,
    VOCAB_ROWS_PER_RANK,
    DFlashConfig,
)
from nex_n2_5_pro.dflash.ll import DFlashLlWires  # noqa: E402
from nex_n2_5_pro.dflash.weights import (  # noqa: E402
    HEAD_PDL,
    MLP_CONFIG,
    OPROJ_CONFIG,
    DFlashWeightShard,
    load_dflash_shard,
    pack_down,
    pack_gate_up,
    pack_output_projection,
    pack_target_hidden_projection,
)

QUERY_HEADS_PER_RANK = 4
QUERY_ROWS_PER_RANK = QUERY_HEADS_PER_RANK * HEAD_DIM
INTERMEDIATE_PER_RANK = 1536
SLIDING_WORKERS = 66
GLOBAL_WORKERS = 256

DFLASH_KERNEL_TUS: tuple[tuple[str, dict[str, bool]], ...] = (
    ("nex_n2_5_pro_dflash_glue_ctypes", {}),
    ("nex_n2_5_pro_dflash_target_hidden_proj_ctypes", {}),
    ("nex_n2_5_pro_dflash_qkv_ckv_front_ctypes", {}),
    ("nex_n2_5_pro_dflash_attn_wgmma_ctypes", {"hopper": True, "link_cuda": True}),
    ("nex_n2_5_pro_dflash_oproj_tc_ctypes", {}),
    ("nex_n2_5_pro_dflash_mlp_tc_ctypes", {}),
    ("nex_n2_5_pro_dflash_head_ctypes", {}),
    ("nex_n2_5_pro_dflash_argmax_ctypes", {}),
)


def prebuild_dflash_kernels(verbose: bool = False):
    """Build the fixed DFlash translation units in one process."""
    kernel_dir = HERE.parent / "kernels" / "dflash"
    tap_name = "nex_n2_5_pro_m8_feature_tap_ctypes"
    tap_object = build_kernel(HERE.parent / "kernels" / "embed", tap_name)
    built = [(tap_name, tap_object)]
    if verbose:
        print(f"[dflash-build] ok {tap_name} -> {tap_object}", flush=True)
    for name, flags in DFLASH_KERNEL_TUS:
        shared_object = build_kernel(kernel_dir, name, **flags)
        built.append((name, shared_object))
        if verbose:
            print(f"[dflash-build] ok {name} -> {shared_object}", flush=True)
    return built


class _KernelState:
    """Rank-local persistent weights, RoPE tables, and six draft KV caches."""

    def __init__(
        self,
        config: DFlashConfig,
        weights: DFlashWeightShard,
        device: torch.device,
        cache_capacity: int,
    ) -> None:
        self.config = config
        self.device = device
        self.cache_capacity = int(cache_capacity)
        self.target_hidden_projection = weights.target_hidden_projection
        self.final_norm = weights.final_norm
        self.layers = tuple(
            {
                "output": layer.output,
                "post_attention_norm": layer.post_attention_norm,
                "gate": layer.gate,
                "up": layer.up,
                "down": layer.down,
            }
            for layer in weights.layers
        )

        inverse_frequency = 1.0 / (
            config.rope_theta ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM)
        )
        positions = torch.arange(self.cache_capacity, dtype=torch.float64)[:, None]
        frequencies = positions * inverse_frequency[None, :]
        self.cosine = frequencies.cos().float().to(device).contiguous()
        self.sine = frequencies.sin().float().to(device).contiguous()
        self.key_cache = [
            torch.zeros(
                SLIDING_RING_CAPACITY if layer < SLIDING_LAYERS else self.cache_capacity,
                HEAD_DIM,
                dtype=torch.bfloat16,
                device=device,
            )
            for layer in range(LAYERS)
        ]
        self.value_cache = [torch.zeros_like(cache) for cache in self.key_cache]

    def reset(self) -> None:
        for key, value in zip(self.key_cache, self.value_cache):
            key.zero_()
            value.zero_()


class DFlashBlock:
    """The fixed M=8 DFlash round bound to one target engine and verifier."""

    def __init__(
        self,
        engine,
        verifier,
        device: torch.device,
        config: DFlashConfig,
        weights: DFlashWeightShard,
        cache_capacity: int | None = None,
    ) -> None:
        config.validate_kernel_contract()
        if engine.world != TP:
            raise ValueError(f"DFlash requires TP={TP}, got world={engine.world}")
        if engine.rank < 0 or engine.rank >= TP:
            raise ValueError(f"invalid DFlash rank {engine.rank}")
        taps = getattr(verifier, "dflash_taps", None)
        if taps is None:
            raise ValueError("construct M8Verify with dflash_tap_layers=config.target_layer_ids")
        if tuple(taps) != config.target_layer_ids:
            raise ValueError(
                "M=8 DFlash tap layers do not match the draft checkpoint: "
                f"{tuple(taps)} != {config.target_layer_ids}"
            )

        self.engine = engine
        self.verifier = verifier
        self.device = device
        self.config = config
        self.cache_capacity = int(
            config.cache_capacity if cache_capacity is None else cache_capacity
        )
        if self.cache_capacity < config.max_position_embeddings:
            raise ValueError("DFlash cache_capacity must cover max_position_embeddings")
        # The draft head is greedy. Configure this verifier instance
        # explicitly without changing M8Verify's default for other users.
        verifier.set_sampling_config(0.0, 1.0)

        self.state = _KernelState(config, weights, device, self.cache_capacity)
        self.front = DFlashAttentionFront(
            config, weights, device, self.cache_capacity, verifier.op_status
        )
        self.wires = DFlashLlWires(device, verifier.op_status)

        kernel_dir = HERE.parent / "kernels" / "dflash"
        void_p = ctypes.c_void_p
        int_t = ctypes.c_int
        ulonglong_t = ctypes.c_ulonglong

        self.glue = ctypes.CDLL(str(build_kernel(kernel_dir, "nex_n2_5_pro_dflash_glue_ctypes")))
        self.glue.NexN25ProDFlashNoiseEmbedLaunch.argtypes = [
            void_p,
            void_p,
            int_t,
            void_p,
            void_p,
            void_p,
        ]
        self.glue.NexN25ProDFlashProposalCommitLaunch.argtypes = [void_p, void_p, void_p]

        self.target_hidden = ctypes.CDLL(
            str(build_kernel(kernel_dir, "nex_n2_5_pro_dflash_target_hidden_proj_ctypes"))
        )
        self.target_hidden.NexN25ProDFlashTargetHiddenProjGemmLaunch.argtypes = [void_p] * 4
        self.target_hidden.NexN25ProDFlashTargetHiddenProjReduceLaunch.argtypes = [
            void_p,
            ulonglong_t,
            void_p,
            void_p,
            void_p,
            void_p,
            int_t,
            int_t,
            void_p,
        ]

        self.attention = ctypes.CDLL(
            str(
                build_kernel(
                    kernel_dir, "nex_n2_5_pro_dflash_attn_wgmma_ctypes", hopper=True, link_cuda=True
                )
            )
        )
        self.attention.NexN25ProDFlashAttnWgmmaPartialCtlLaunch.argtypes = (
            [void_p] * 9 + [int_t] * 5 + [void_p]
        )
        self.attention.NexN25ProDFlashAttnWgmmaCombineLaunch.argtypes = (
            [void_p] * 6 + [int_t] * 3 + [void_p]
        )

        self.output_projection = ctypes.CDLL(
            str(build_kernel(kernel_dir, "nex_n2_5_pro_dflash_oproj_tc_ctypes"))
        )
        self.output_projection.NexN25ProDFlashOProjTcGemmLaunch.argtypes = [
            void_p,
            void_p,
            void_p,
            int_t,
            int_t,
            void_p,
        ]
        self.output_projection.NexN25ProDFlashOProjTcReduceLaunch.argtypes = [
            void_p,
            ulonglong_t,
            void_p,
            void_p,
            void_p,
            void_p,
            int_t,
            int_t,
            int_t,
            void_p,
        ]

        self.mlp = ctypes.CDLL(str(build_kernel(kernel_dir, "nex_n2_5_pro_dflash_mlp_tc_ctypes")))
        self.mlp.NexN25ProDFlashRmsNormLaunch.argtypes = [void_p] * 4
        self.mlp.NexN25ProDFlashMlpGateUpGemmLaunch.argtypes = [
            void_p,
            void_p,
            void_p,
            int_t,
            int_t,
            int_t,
            void_p,
        ]
        self.mlp.NexN25ProDFlashMlpDownGemmLaunch.argtypes = [
            void_p,
            void_p,
            void_p,
            int_t,
            int_t,
            int_t,
            int_t,
            void_p,
        ]
        self.mlp.NexN25ProDFlashMlpDownReduceLaunch.argtypes = [
            void_p,
            ulonglong_t,
            void_p,
            void_p,
            void_p,
            void_p,
            int_t,
            int_t,
            int_t,
            void_p,
        ]

        self.head = ctypes.CDLL(str(build_kernel(kernel_dir, "nex_n2_5_pro_dflash_head_ctypes")))
        self.head.NexN25ProDFlashHeadGemmLaunch.argtypes = [void_p, void_p, void_p, int_t, void_p]
        self.head.NexN25ProDFlashHeadArgmaxLaunch.argtypes = [void_p, void_p, void_p, int_t, void_p]

        self.argmax = ctypes.CDLL(
            str(build_kernel(kernel_dir, "nex_n2_5_pro_dflash_argmax_ctypes"))
        )
        self.argmax.NexN25ProDFlashArgmaxLLLaunch.argtypes = [
            void_p,
            void_p,
            ctypes.c_long,
            ulonglong_t,
            void_p,
            void_p,
            void_p,
            void_p,
            int_t,
            int_t,
            void_p,
        ]

        # Stable activation and reduction buffers used by every replay.
        self.noise = torch.empty(M, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
        self.metadata = torch.zeros(2, dtype=torch.int32, device=device)
        self.context_raw = torch.empty_like(self.noise)
        self.hidden = torch.empty_like(self.noise)
        self.query = torch.empty(M, QUERY_ROWS_PER_RANK, dtype=torch.bfloat16, device=device)
        self.attention_output = torch.empty_like(self.query)
        self.intermediate = torch.empty(
            M, INTERMEDIATE_PER_RANK, dtype=torch.bfloat16, device=device
        )
        self.proposal_ids = torch.empty(M - 1, dtype=torch.int32, device=device)

        self.target_hidden_partial = torch.empty(M, HIDDEN_SIZE, dtype=torch.float32, device=device)
        self.target_hidden_pack = pack_target_hidden_projection(self.state.target_hidden_projection)
        self.output_partial = torch.empty(M, HIDDEN_SIZE, dtype=torch.float32, device=device)
        self.output_packs = tuple(
            pack_output_projection(layer["output"]) for layer in self.state.layers
        )
        torch.cuda.synchronize()

        max_workers = max(SLIDING_WORKERS, GLOBAL_WORKERS)
        query_heads = M * QUERY_HEADS_PER_RANK
        self.attention_max = torch.empty(
            max_workers, query_heads, dtype=torch.float32, device=device
        )
        self.attention_sum = torch.empty_like(self.attention_max)
        self.attention_accumulator = torch.empty(
            max_workers, query_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
        )

        self.normalized = torch.empty_like(self.hidden)
        self.mlp_partial = torch.empty(M, HIDDEN_SIZE, dtype=torch.float32, device=device)
        _, gate_split, _, down_split, down_tiles, _ = MLP_CONFIG
        self.gate_up_packs = tuple(
            pack_gate_up(layer["gate"], layer["up"], gate_split) for layer in self.state.layers
        )
        self.down_packs = tuple(
            pack_down(layer["down"], down_split, down_tiles) for layer in self.state.layers
        )
        torch.cuda.synchronize()

        self.head_logits = torch.empty(M, VOCAB_ROWS_PER_RANK, dtype=torch.bfloat16, device=device)
        self.head_values = torch.empty(M - 1, dtype=torch.float32, device=device)
        self.head_indices = torch.empty(M - 1, dtype=torch.int64, device=device)
        # M=8 and DFlash use the same raw-shard tc16 layout. Reuse the verifier's
        # pack so the target lm_head is neither loaded nor packed a second time.
        self.head_pack = verifier.lm_head_pack()
        torch.cuda.synchronize()
        self.base_position: int | None = None

    @classmethod
    def from_checkpoint(
        cls,
        engine,
        verifier,
        device: torch.device,
        config: DFlashConfig | None = None,
        weights_path: str | Path = DFLASH_WEIGHTS,
        cache_capacity: int | None = None,
    ) -> "DFlashBlock":
        """Load one draft shard and construct its capture-ready block."""
        config = DFlashConfig.load() if config is None else config
        weights = load_dflash_shard(engine.rank, device, weights_path=weights_path)
        return cls(engine, verifier, device, config, weights, cache_capacity)

    @staticmethod
    def _pointer(tensor: torch.Tensor) -> ctypes.c_void_p:
        return ctypes.c_void_p(tensor.data_ptr())

    def seed(self, committed_len: int, *, context_start: int | None = None) -> None:
        """Reset the draft side for a request whose target prefill has this length."""
        committed_len = int(committed_len)
        if committed_len < control.CONV_HISTORY:
            raise ValueError("DFlash must be seeded from a real target prefill")
        if committed_len > self.cache_capacity - M:
            raise ValueError(f"committed_len={committed_len} leaves no block-8 cache headroom")
        context_start = committed_len if context_start is None else int(context_start)
        if not 0 <= context_start <= committed_len:
            raise ValueError("context_start must be between zero and the prefill length")
        self.base_position = context_start
        self.state.reset()
        self.front.status.zero_()
        self.wires.reset_status()

    def load_request(self, state_dir: str | Path, first_token: int) -> int:
        """Load target prefill state and reset the draft cache/control boundary."""
        verify_ids = [int(first_token)] + [0] * (M - 1)
        length = self.verifier.load(str(state_dir), verify_ids)
        self.seed(length)
        return length

    def _attention_layer(self, layer: int, window: int, stream: ctypes.c_void_p) -> None:
        if self.base_position is None:
            raise RuntimeError("seed() must be called before a DFlash round")
        check = self.verifier._ck
        int_t = ctypes.c_int
        sliding = window > 0
        workers = SLIDING_WORKERS if sliding else GLOBAL_WORKERS
        physical_capacity = SLIDING_RING_CAPACITY if sliding else self.cache_capacity
        cache_mode = 1 if sliding else 0
        check(
            self.attention.NexN25ProDFlashAttnWgmmaPartialCtlLaunch(
                self._pointer(self.query),
                self._pointer(self.state.key_cache[layer]),
                self._pointer(self.state.value_cache[layer]),
                self._pointer(self.attention_max),
                self._pointer(self.attention_sum),
                self._pointer(self.attention_accumulator),
                self._pointer(self.attention_output),
                self._pointer(self.metadata),
                self._pointer(self.front.status),
                int_t(self.base_position),
                int_t(window),
                int_t(physical_capacity),
                int_t(workers),
                int_t(cache_mode),
                stream,
            ),
            f"dflash attention partial L{layer}",
        )
        check(
            self.attention.NexN25ProDFlashAttnWgmmaCombineLaunch(
                self._pointer(self.attention_max),
                self._pointer(self.attention_sum),
                self._pointer(self.attention_accumulator),
                self._pointer(self.attention_output),
                self._pointer(self.metadata),
                self._pointer(self.front.status),
                int_t(self.base_position),
                int_t(window),
                int_t(workers),
                stream,
            ),
            f"dflash attention combine L{layer}",
        )

    def _output_projection_layer(self, layer: int, stream: ctypes.c_void_p) -> None:
        depth, pdl = OPROJ_CONFIG
        multicast, local, counter, status = self.wires.args("output_projection")
        int_t = ctypes.c_int
        check = self.verifier._ck
        check(
            self.output_projection.NexN25ProDFlashOProjTcGemmLaunch(
                self._pointer(self.attention_output),
                self._pointer(self.output_packs[layer]),
                self._pointer(self.output_partial),
                int_t(depth),
                int_t(pdl),
                stream,
            ),
            f"dflash output projection gemm L{layer}",
        )
        check(
            self.output_projection.NexN25ProDFlashOProjTcReduceLaunch(
                self._pointer(self.output_partial),
                ctypes.c_ulonglong(multicast),
                ctypes.c_void_p(local),
                self._pointer(counter),
                self._pointer(status),
                self._pointer(self.hidden),
                int_t(self.engine.rank),
                int_t(self.engine.world),
                int_t(pdl),
                stream,
            ),
            f"dflash output projection reduce L{layer}",
        )

    def _mlp_layer(self, layer: int, stream: ctypes.c_void_p) -> None:
        gate_depth, gate_split, down_depth, down_split, down_tiles, pdl = MLP_CONFIG
        int_t = ctypes.c_int
        check = self.verifier._ck
        state = self.state.layers[layer]
        check(
            self.mlp.NexN25ProDFlashRmsNormLaunch(
                self._pointer(self.hidden),
                self._pointer(state["post_attention_norm"]),
                self._pointer(self.normalized),
                stream,
            ),
            f"dflash mlp norm L{layer}",
        )
        check(
            self.mlp.NexN25ProDFlashMlpGateUpGemmLaunch(
                self._pointer(self.normalized),
                self._pointer(self.gate_up_packs[layer]),
                self._pointer(self.intermediate),
                int_t(gate_depth),
                int_t(gate_split),
                int_t(pdl),
                stream,
            ),
            f"dflash mlp gate/up L{layer}",
        )
        check(
            self.mlp.NexN25ProDFlashMlpDownGemmLaunch(
                self._pointer(self.intermediate),
                self._pointer(self.down_packs[layer]),
                self._pointer(self.mlp_partial),
                int_t(down_depth),
                int_t(down_split),
                int_t(down_tiles),
                int_t(pdl),
                stream,
            ),
            f"dflash mlp down L{layer}",
        )
        multicast, local, counter, status = self.wires.args("mlp_down")
        check(
            self.mlp.NexN25ProDFlashMlpDownReduceLaunch(
                self._pointer(self.mlp_partial),
                ctypes.c_ulonglong(multicast),
                ctypes.c_void_p(local),
                self._pointer(counter),
                self._pointer(status),
                self._pointer(self.hidden),
                int_t(self.engine.rank),
                int_t(self.engine.world),
                int_t(pdl),
                stream,
            ),
            f"dflash mlp down reduce L{layer}",
        )

    def _head(self, stream: ctypes.c_void_p) -> None:
        check = self.verifier._ck
        int_t = ctypes.c_int
        multicast, local, counter, status = self.wires.args("head")
        check(
            self.mlp.NexN25ProDFlashRmsNormLaunch(
                self._pointer(self.hidden),
                self._pointer(self.state.final_norm),
                self._pointer(self.normalized),
                stream,
            ),
            "dflash head norm",
        )
        check(
            self.head.NexN25ProDFlashHeadGemmLaunch(
                self._pointer(self.normalized),
                self._pointer(self.head_pack),
                self._pointer(self.head_logits),
                int_t(HEAD_PDL),
                stream,
            ),
            "dflash head gemm",
        )
        check(
            self.head.NexN25ProDFlashHeadArgmaxLaunch(
                self._pointer(self.head_logits),
                self._pointer(self.head_values),
                self._pointer(self.head_indices),
                int_t(HEAD_PDL),
                stream,
            ),
            "dflash head argmax",
        )
        check(
            self.argmax.NexN25ProDFlashArgmaxLLLaunch(
                self._pointer(self.head_values),
                self._pointer(self.head_indices),
                ctypes.c_long(self.engine.rank * VOCAB_ROWS_PER_RANK),
                ctypes.c_ulonglong(multicast),
                ctypes.c_void_p(local),
                self._pointer(counter),
                self._pointer(status),
                self._pointer(self.proposal_ids),
                int_t(self.engine.rank),
                int_t(self.engine.world),
                stream,
            ),
            "dflash argmax LL",
        )

    def round(self) -> None:
        """Launch one complete proposal + target-verification round."""
        if self.base_position is None:
            raise RuntimeError("seed() or load_request() must precede round()")
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        check = self.verifier._ck
        int_t = ctypes.c_int
        check(
            self.glue.NexN25ProDFlashNoiseEmbedLaunch(
                self._pointer(self.engine.embed_w),
                self._pointer(self.verifier.ctl),
                int_t(self.config.mask_token_id),
                self._pointer(self.noise),
                self._pointer(self.metadata),
                stream,
            ),
            "dflash noise embed",
        )

        tap_layer = self.config.target_layer_ids[self.engine.rank]
        tap = self.verifier.dflash_taps[tap_layer]
        multicast, local, counter, status = self.wires.args("target_hidden")
        check(
            self.target_hidden.NexN25ProDFlashTargetHiddenProjGemmLaunch(
                self._pointer(tap),
                self._pointer(self.target_hidden_pack),
                self._pointer(self.target_hidden_partial),
                stream,
            ),
            "dflash target-hidden projection gemm",
        )
        check(
            self.target_hidden.NexN25ProDFlashTargetHiddenProjReduceLaunch(
                self._pointer(self.target_hidden_partial),
                ctypes.c_ulonglong(multicast),
                ctypes.c_void_p(local),
                self._pointer(counter),
                self._pointer(status),
                self._pointer(self.context_raw),
                int_t(self.engine.rank),
                int_t(self.engine.world),
                stream,
            ),
            "dflash target-hidden projection reduce",
        )

        self.hidden.copy_(self.noise)
        for layer in range(LAYERS):
            window = self.config.sliding_window if layer < SLIDING_LAYERS else 0
            check(
                self.front.launch_qkv_projection(layer, self.hidden, stream),
                f"dflash qkv projection L{layer}",
            )
            check(
                self.front.launch_context_kv_projection(layer, self.context_raw, stream),
                f"dflash context kv projection L{layer}",
            )
            check(
                self.front.launch_post(
                    layer,
                    self.state.cosine,
                    self.state.sine,
                    self.query,
                    self.state.key_cache[layer],
                    self.state.value_cache[layer],
                    self.metadata,
                    stream,
                    ring_capacity=(SLIDING_RING_CAPACITY if window else 0),
                ),
                f"dflash qk/ck norm, rope, and v/cv scale L{layer}",
            )
            self._attention_layer(layer, window, stream)
            self._output_projection_layer(layer, stream)
            self._mlp_layer(layer, stream)

        self._head(stream)
        check(
            self.glue.NexN25ProDFlashProposalCommitLaunch(
                self._pointer(self.proposal_ids), self._pointer(self.verifier.ctl), stream
            ),
            "dflash proposal commit",
        )
        self.verifier.run()

    def capture(self, warmup: int = 3) -> torch.cuda.CUDAGraph:
        """Capture one round; reload the request after this destructive warmup."""
        if self.base_position is None:
            raise RuntimeError("seed() or load_request() must precede capture()")
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                self.round()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.round()
        torch.cuda.synchronize()
        return graph

    def status(self) -> int:
        """Return the local sticky DFlash/M=8 status outside captured work."""
        values = [
            int(self.wires.status.item()),
            int(self.front.status.item()),
            int(self.verifier.moe_status.item()),
            int(self.verifier.cursor_status.item()),
            int(self.verifier.ctl[control.STATUS].item()),
        ]
        if hasattr(self.verifier, "cp_status"):
            values.append(int(self.verifier.cp_status.item()))
        return max(values)
