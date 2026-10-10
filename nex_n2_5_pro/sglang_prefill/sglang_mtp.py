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

"""SGLang-native MTP draft prefill for the bridge.

SGLang computes the MTP layer over the prompt with the same machinery as its
EAGLE/NEXTN speculative worker (``EagleDraftWorker._draft_extend_for_prefill``):

  - a draft ModelRunner (``is_draft_worker=True``, architecture overridden to
    the checkpoint's MTP model) shares the target's req_to_token pool and KV
    allocator, so the MTP KV of prompt token p lives at the same token slot as
    the target KV of p, in the draft runner's one-layer pool;
  - after each target prefill chunk, the draft runs an EXTEND forward over the
    same rows with the input ids shifted left by one (tail = next prompt token,
    or the selected first output token on the final chunk) and the target's
    post-final-norm hidden states as ``spec_info.hidden_states``.

``HiddenTap`` holds exactly one chunk's post-norm target rows, the same tensor
SGLang returns under ``CaptureHiddenMode.FULL``. The decode hand-off
(``MtpDraft.seed``) needs, per rank: MTP KV rows [0, p) for this rank's KV head,
the raw (pre-``mtp.norm``) last-row hidden and the first draft token.
``MtpPrefill.finish`` returns exactly those.
"""

from __future__ import annotations

import copy
import hashlib
import tempfile
from pathlib import Path

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput

from nex_n2_5_pro.sglang_prefill.weight_loading import _patch_fp8_scale_names


def draft_checkpoint(model_path: Path, layers_dir: Path) -> Path:
    """A checkpoint directory holding only links to config.json + mtp.safetensors.

    SGLang's loader reads every safetensors file of its model path; pointing the
    draft at the full checkpoint would stream the whole target model again just
    to keep the ``mtp.*`` tensors. The links live in a temporary directory so
    the weight directories stay read-only."""
    sources = (
        ("config.json", (Path(model_path) / "config.json").resolve()),
        ("mtp.safetensors", (Path(layers_dir) / "mtp.safetensors").resolve()),
    )
    key = hashlib.sha256("\0".join(str(path) for _, path in sources).encode()).hexdigest()[:16]
    out_dir = Path(tempfile.gettempdir()) / f"nexrt-sglang-mtp-{key}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, target in sources:
        link = out_dir / name
        if not link.is_symlink() and not link.exists():
            try:
                link.symlink_to(target)
            except FileExistsError:  # another TP rank created it first
                pass
        if link.resolve() != target:
            raise RuntimeError(f"{link} -> {link.resolve()}, expected {target}")
    return out_dir


class HiddenTap:
    """Forward hook on the final norm module: exposes AT MOST the current prefill chunk.

    arm() -> [model chunk forward -> pop()]* -> disarm(). The hook never accumulates
    the prompt, so the transient stays chunk-bounded at every context length."""

    def __init__(self, model):
        norm = None
        for name, mod in model.named_modules():
            if name.endswith("language_model.norm") or name == "model.norm":
                norm = mod
        if norm is None:
            raise RuntimeError("final norm module not found for the hidden tap")
        self._chunk: torch.Tensor | None = None
        self._armed = False

        def hook(_mod, _inp, out):
            if not self._armed:
                return
            h = out[0] if isinstance(out, tuple) else out
            assert self._chunk is None, "hidden tap: previous chunk never consumed"
            self._chunk = h.detach().clone()

        norm.register_forward_hook(hook)

    def arm(self):
        assert self._chunk is None, "hidden tap: stale chunk at arm"
        self._armed = True

    def pop(self) -> torch.Tensor:
        h, self._chunk = self._chunk, None
        assert h is not None, "hidden tap: no chunk captured"
        return h

    def disarm(self):
        assert self._chunk is None, "hidden tap: unconsumed chunk at disarm"
        self._armed = False

    def abort(self):
        """Drop a partially captured chunk after a failed prefill request."""
        self._chunk = None
        self._armed = False


class MtpPrefill:
    """Draft ModelRunner over SGLang's MTP layer, extended after every target chunk."""

    def __init__(self, target, server_args, port_args, rank: int, ckpt: Path):
        self.target = target
        sa = copy.copy(server_args)
        sa.speculative_algorithm = "EAGLE"
        sa.speculative_draft_model_path = str(ckpt)
        # The target may load from a sharded weight cache; the draft reads its
        # own one-layer checkpoint.
        sa.load_format = "auto"
        sa.model_loader_extra_config = {}
        moe_ep_rank = rank // (sa.tp_size // sa.ep_size)
        model_config = ModelConfig.from_server_args(
            sa,
            model_path=str(ckpt),
            is_draft_model=True,
            context_length=target.model_config.context_len,
        )
        print(
            f"[rank{rank}] MTP fp8 scale-name remap on "
            f"{_patch_fp8_scale_names(str(ckpt), model_config.hf_config.architectures)} classes",
            flush=True,
        )
        with speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.draft = ModelRunner(
                model_config=model_config,
                mem_fraction_static=sa.mem_fraction_static,
                gpu_id=rank,
                tp_rank=rank,
                tp_size=sa.tp_size,
                moe_ep_rank=moe_ep_rank,
                moe_ep_size=sa.ep_size,
                pp_rank=0,
                pp_size=1,
                nccl_port=port_args.nccl_port,
                server_args=sa,
                is_draft_worker=True,
                req_to_token_pool=target.req_to_token_pool,
                token_to_kv_pool_allocator=target.token_to_kv_pool_allocator,
                memory_pool_config=target.memory_pool_config,
            )
            self.draft.alloc_memory_pool(target.memory_pool_config)
            self.draft.init_attention_backends()
            # Builds the eager runner; the draft prefill runs eagerly, as in
            # EagleDraftWorker.init_cuda_graphs.
            self.draft.init_cuda_graphs(capture_decode_cuda_graph=False)
        embed, head = target.model.get_embed_and_head()
        self.draft.model.set_embed_and_head(embed, head)
        lm_head = getattr(target.model, "lm_head", None)
        if lm_head is not None:
            self.draft.model.set_lm_head_from_target(lm_head)
        self.kv = self.draft.token_to_kv_pool.full_kv_pool
        if self.kv.k_buffer[0].shape[1] != 1:
            raise RuntimeError(
                f"expected one MTP KV head per rank, got {self.kv.k_buffer[0].shape}"
            )

        # Raw (pre-norm) MTP hidden: the layer ends in a fused add-norm
        # norm(hidden, residual); the raw row is hidden + residual.
        self._raw = None
        self._d1 = None

        def pre_norm(_mod, args):
            h = args[0]
            r = args[1] if len(args) > 1 else None
            self._raw = (h if r is None else h + r)[-1].detach().clone()

        self.draft.model.model.norm.register_forward_pre_hook(pre_norm)

    @torch.inference_mode()
    def extend(self, batch, hidden: torch.Tensor, tail_token: int):
        """Draft EXTEND over the rows of one completed target chunk.

        ``batch`` is the target's ScheduleBatch for that chunk, ``hidden`` its post-norm
        rows, ``tail_token`` the token after the chunk's last row."""
        if hidden.shape[0] != batch.input_ids.shape[0]:
            raise RuntimeError(f"tap rows {hidden.shape[0]} != chunk {batch.input_ids.shape[0]}")
        tail = torch.tensor(
            [int(tail_token)], dtype=batch.input_ids.dtype, device=batch.input_ids.device
        )
        batch.input_ids = torch.cat((batch.input_ids[1:], tail))
        batch.spec_info = EagleDraftExtendInput(
            hidden_states=hidden, num_tokens_per_req=1, num_tokens_for_logprob_per_req=1
        )
        batch.capture_hidden_mode = CaptureHiddenMode.LAST
        self._raw = None
        with speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            dfb = ForwardBatch.init_new(batch, self.draft)
            dfb.return_logprob = False
            dout = self.draft.forward(dfb).logits_output
        self._d1 = dout.next_token_logits[-1]

    def finish(self, req_pool_index: int, length: int):
        """(k [p,256], v [p,256], y_sel_raw [4096], d1) for this rank, rows [0, p)."""
        if self._raw is None or self._d1 is None:
            raise RuntimeError("MTP prefill did not run the final draft extend")
        locs = self.target.req_to_token_pool.req_to_token[req_pool_index, :length].long()
        k = self.kv.k_buffer[0][locs, 0]
        v = self.kv.v_buffer[0][locs, 0]
        y_raw, d1 = self._raw, int(self._d1.argmax())
        self._raw = self._d1 = None
        return k, v, y_raw, d1
