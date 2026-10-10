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

"""In-process sglang-prefill -> nexrt-decode bridge primitives (co-located 8xH100).

sglang loads the checkpoint and runs prefill; nexrt is built from sglang's live tensors (weight
aliasing, see ext_weights.SglangExt) and decodes by whole-step CUDA-graph replay. Prompt KV rows
are appended from sglang's pool into nexrt's contiguous cache as each prefill chunk completes;
the final Gated DeltaNet state and decode slot are committed once at the end. The handoff kernels
live in kernels/util/nex_n2_5_pro_bridge_handoff_ctypes.cu, replacing the offline extract -> disk -> reload
path. See BRIDGE.md.

This module holds the reusable pieces: the sglang ServerArgs factory, the layer split, the
chunked-prefill driver, and the Handoff pointer-table + append/finalize/gather/scatter driver.
bridge_serve.py is the sequential serving loop built on top; bridge_client.py is the matching
client. Cache ownership and lifecycle live in cache_lifecycle.py.

Runs only in an environment where SGLang is importable (the prefill engine); decode-only nexrt
does not need it.
"""

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
NEX_N2_5_PRO = HERE.parent

import sglang.bench_one_batch as B  # noqa: E402
from sglang.srt.server_args import PortArgs as PortArgs, ServerArgs  # noqa: E402  (re-exported for servers)
from nex_n2_5_pro.sglang_prefill.serve_control import FatalBackendError  # noqa: E402

MODEL = os.environ.get("NEXRT_MODEL_DIR", "/path/to/nex_n2_5_pro")

# S (context) at/above which the CP (context-parallel) decode graph replaces the non-CP graph.
# Matches decode_engine.CP_DISPATCH_S (the measured 48K whole-step crossover).
CP_THRESHOLD = 48 * 1024


def layer_sets(nlayers):
    """nex_n2_5_pro interleave: every 4th layer (3,7,11,...) is full attention; the rest are GDN."""
    full_attn = [L for L in range(nlayers) if (L + 1) % 4 == 0]
    gdn = [L for L in range(nlayers) if (L + 1) % 4 != 0]
    return full_attn, gdn


def make_server_args(
    model=None,
    mem_fraction_static=0.72,
    context_length=262144,
    quantization=None,
    max_total_tokens=263168,
    max_running_requests=2,
):
    """sglang ServerArgs for the co-located bridge: cuda-graph OFF (nexrt owns the decode graph),
    radix cache OFF (the bridge owns exact session reuse), fa3 attention (its post-RoPE post-norm
    bf16 KV pool is exactly nexrt's cache convention), 8K chunked prefill.
    Explicit KV and request-slot limits bound the prefill engine's cache allocation."""
    sa = ServerArgs(
        model_path=model or MODEL,
        tp_size=8,
        trust_remote_code=True,
        disable_cuda_graph=True,
        disable_radix_cache=True,
        mem_fraction_static=mem_fraction_static,
        context_length=context_length,
        attention_backend="fa3",
        quantization=quantization,
        chunked_prefill_size=8192,
        max_prefill_tokens=8192,
        weight_loader_prefetch_checkpoints=True,
        max_total_tokens=max_total_tokens,
        max_running_requests=max_running_requests,
    )
    sa.cuda_graph_max_bs = 1
    return sa


def _p(t):
    return ctypes.c_void_p(t.data_ptr())


def _sp():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


class Handoff:
    """SGLang pool -> NexRT cache handoff through boot-time device pointer tables.

    ``gather`` is the complete-request fallback. ``append`` copies only a newly completed
    prefill range into NexRT's contiguous cache, and ``finalize`` copies the per-request
    GDN state and publishes the decode slot after all KV appends have been queued. ``scatter``
    remains the optional multi-turn hand-back path.
    """

    def __init__(self, eng, model_runner, full_attn, gdn_layers, dev):
        from nex_n2_5_pro.kernel_build import build_kernel

        vp, ci = ctypes.c_void_p, ctypes.c_int
        self.lib = ctypes.CDLL(
            str(
                build_kernel(
                    NEX_N2_5_PRO / "kernels" / "util", "nex_n2_5_pro_bridge_handoff_ctypes"
                )
            )
        )
        self.lib.nex_n2_5_pro_bridge_gather_launch.argtypes = [
            vp,
            ci,
            vp,
            ci,
            vp,
            ci,
            ci,
            ci,
            vp,
            vp,
            ci,
            vp,
        ]
        self.lib.nex_n2_5_pro_bridge_gather_launch.restype = ci
        self.lib.nex_n2_5_pro_bridge_append_launch.argtypes = [vp, ci, vp, ci, ci, vp]
        self.lib.nex_n2_5_pro_bridge_append_launch.restype = ci
        self.lib.nex_n2_5_pro_bridge_finalize_launch.argtypes = [vp, ci, ci, ci, ci, vp, vp, ci, vp]
        self.lib.nex_n2_5_pro_bridge_finalize_launch.restype = ci
        self.lib.nex_n2_5_pro_bridge_scatter_launch.argtypes = [vp, ci, vp, ci, ci, vp, ci, ci, vp]
        self.lib.nex_n2_5_pro_bridge_scatter_launch.restype = ci
        self.eng, self.n_kv, self.n_gdn = eng, len(full_attn), len(gdn_layers)
        self.cache_len = int(eng.attn[full_attn[0]].k_cache.shape[0]) if self.n_kv else 0
        pool = model_runner.token_to_kv_pool
        r2t = model_runner.req_to_token_pool
        self.r2t = r2t
        # boot-time device pointer tables (all addresses are process-stable). Order matters -- it is
        # the [src | src | dst | dst] layout the gather/scatter kernels index.
        kv = (
            [pool.get_key_buffer(L).data_ptr() for L in full_attn]
            + [pool.get_value_buffer(L).data_ptr() for L in full_attn]
            + [eng.attn[L].k_cache.data_ptr() for L in full_attn]
            + [eng.attn[L].v_cache.data_ptr() for L in full_attn]
        )
        gd = (
            [r2t.mamba2_layer_cache(L).conv[0].data_ptr() for L in gdn_layers]
            + [r2t.mamba2_layer_cache(L).temporal.data_ptr() for L in gdn_layers]
            + [eng.gdn[L].conv_state4.data_ptr() for L in gdn_layers]
            + [eng.gdn[L].state_t.data_ptr() for L in gdn_layers]
        )
        stat = list(
            {
                t.data_ptr(): None
                for t in (
                    [eng.head.status]
                    + [b.status for b in eng.gdn.values()]
                    + [b.status for b in eng.attn.values()]
                    + [b.status for b in eng.moe.values()]
                    + [b.cp_status for b in eng.attn_cp.values()]
                )
            }
        )
        self.kv_t = torch.tensor(kv, dtype=torch.int64, device=dev)
        self.gdn_t = torch.tensor(gd, dtype=torch.int64, device=dev)
        self.stat_t = torch.tensor(stat, dtype=torch.int64, device=dev)
        # sanity: fixed row strides the kernel assumes (one KV row = 256 bf16 = 512 B, etc.)
        assert pool.get_key_buffer(full_attn[0]).stride(0) * 2 == 512
        assert r2t.mamba2_layer_cache(gdn_layers[0]).temporal.stride(0) * 4 == 8 * 128 * 128 * 4
        assert r2t.mamba2_layer_cache(gdn_layers[0]).conv[0].stride(0) * 2 == 1536 * 3 * 2

    def gather(self, rp_idx, s_len, mamba_slot, first_token):
        """sglang prompt state -> nexrt caches; also writes the [first_token, s_len] slot and
        zeroes every block status. Returns the idx tensor (keep alive until the stream syncs)."""
        rp_idx, s_len, mamba_slot, first_token = map(int, (rp_idx, s_len, mamba_slot, first_token))
        if s_len <= 0 or s_len > self.cache_len:
            raise ValueError(f"gather length {s_len} exceeds NexRT cache capacity {self.cache_len}")
        idx32 = self.r2t.req_to_token[rp_idx, :s_len].to(torch.int32).contiguous()
        idx32.record_stream(torch.cuda.current_stream())
        rc = self.lib.nex_n2_5_pro_bridge_gather_launch(
            _p(self.kv_t),
            self.n_kv,
            _p(idx32),
            s_len,
            _p(self.gdn_t),
            self.n_gdn,
            mamba_slot,
            first_token,
            _p(self.eng.slot),
            _p(self.stat_t),
            self.stat_t.numel(),
            _sp(),
        )
        if rc != 0:
            raise FatalBackendError(f"NexRT bridge gather launch failed: cudaError={rc}")
        return idx32

    def append(self, rp_idx, src_start, n_rows, dst_start=None):
        """Append SGLang KV rows into NexRT's contiguous cache.

        ``req_to_token`` may point at arbitrary SGLang pool slots; the append kernel reads
        those indexed rows and writes them to ``[dst_start, dst_start+n_rows)``. It does not
        touch GDN state or ``eng.slot``. Call :meth:`finalize` after the final prefill chunk.
        """
        rp_idx, src_start, n_rows = map(int, (rp_idx, src_start, n_rows))
        dst_start = src_start if dst_start is None else int(dst_start)
        table = self.r2t.req_to_token
        if rp_idx < 0 or rp_idx >= table.shape[0]:
            raise ValueError(f"request pool index {rp_idx} is out of range")
        if src_start < 0 or n_rows <= 0:
            raise ValueError(f"invalid source range [{src_start}, {src_start + n_rows})")
        if src_start + n_rows > table.shape[1]:
            raise ValueError(
                f"source range [{src_start}, {src_start + n_rows}) exceeds SGLang request table"
            )
        if dst_start < 0 or dst_start + n_rows > self.cache_len:
            raise ValueError(
                f"destination range [{dst_start}, {dst_start + n_rows}) exceeds "
                f"NexRT cache capacity {self.cache_len}"
            )
        idx32 = table[rp_idx, src_start : src_start + n_rows].to(torch.int32).contiguous()
        idx32.record_stream(torch.cuda.current_stream())
        rc = self.lib.nex_n2_5_pro_bridge_append_launch(
            _p(self.kv_t), self.n_kv, _p(idx32), n_rows, dst_start, _sp()
        )
        if rc != 0:
            raise FatalBackendError(f"NexRT bridge append launch failed: cudaError={rc}")
        return idx32

    def finalize(self, mamba_slot, s_len, first_token):
        """Copy final GDN state and publish ``eng.slot=[first_token, s_len]``."""
        mamba_slot, s_len, first_token = map(int, (mamba_slot, s_len, first_token))
        if s_len <= 0 or s_len > self.cache_len:
            raise ValueError(f"final length {s_len} exceeds NexRT cache capacity {self.cache_len}")
        rc = self.lib.nex_n2_5_pro_bridge_finalize_launch(
            _p(self.gdn_t),
            self.n_gdn,
            mamba_slot,
            first_token,
            s_len,
            _p(self.eng.slot),
            _p(self.stat_t),
            self.stat_t.numel(),
            _sp(),
        )
        if rc != 0:
            raise FatalBackendError(f"NexRT bridge finalize launch failed: cudaError={rc}")

    def scatter(self, dst_slots_i32, row_start, n_rows, mamba_slot):
        """NexRT caches -> SGLang pools (multi-turn hand-back). dst_slots_i32 = destination pool
        rows; cache rows [row_start, row_start+n_rows) are written back."""
        rc = self.lib.nex_n2_5_pro_bridge_scatter_launch(
            _p(self.kv_t),
            self.n_kv,
            _p(dst_slots_i32),
            n_rows,
            row_start,
            _p(self.gdn_t),
            self.n_gdn,
            mamba_slot,
            _sp(),
        )
        if rc != 0:
            raise FatalBackendError(f"NexRT bridge scatter launch failed: cudaError={rc}")


def _stage_extend(req, tokens, consumed, e):
    """Declare the next extend window [consumed, e) on the request, on either Req API.

    sglang >= 0.5.15: token containers are array('q') (list+array concat inside
    Req._refresh_fill_ids raises TypeError otherwise), full_untruncated_fill_ids is
    ALIASED to origin_input_ids and extended in place (the upstream bench convention),
    and the window is declared with set_extend_range(start, end); prepare_for_extend
    consumes get_fill_ids()[len(prefix_indices):]. Older sglang keeps the
    fill_ids / set_extend_input_len form."""
    if hasattr(req, "set_extend_range"):
        from array import array

        have = len(req.full_untruncated_fill_ids)
        if have < e:
            req.full_untruncated_fill_ids.extend(array("q", tokens[have:e]))
        req.set_extend_range(consumed, e)
    else:
        req.fill_ids = tokens[:e]
        req.set_extend_input_len(e - consumed)


@dataclass
class PrefillSession:
    """The live SGLang request state retained for an exact future suffix extend."""

    tokens: tuple[int, ...]
    req: object
    next_token_ids: object
    next_token_logits: object
    batch: object
    req_pool_index: int


def chunked_prefill(
    model_runner,
    tokens,
    chunk,
    on_chunk=None,
    on_chunk_batch=None,
    resume=None,
    return_session=False,
    request_id=0,
    sample_next_token=None,
):
    """One continued sglang prefill request in <=chunk-token extends -> (next_token_ids, logits, batch).
    Prompts are streamed as consecutive extends sharing one request slot. If ``resume`` is a
    :class:`PrefillSession`, only the exact suffix after its committed token prefix is evaluated.

    on_chunk(s, e, next_tok): called after each extend chunk covering prompt rows [s, e),
    OUTSIDE the model forward (collectives inside the callback stay at top level and in TP
    lockstep). next_tok is the next token after row e-1, selected greedily unless the
    final row uses ``sample_next_token``. ``on_chunk_batch(s, e, next_tok, batch)`` is
    an optional companion callback for handoff users that need the current request-pool index.
    It runs immediately after the same ``B.extend`` returns, before the next chunk is staged.

    ``sample_next_token(logits)`` optionally selects the final next-token tensor before
    either callback consumes it. Intermediate chunks keep their greedy next token."""

    def notify(s, e, next_tok, batch):
        if on_chunk is not None:
            on_chunk(s, e, next_tok)
        if on_chunk_batch is not None:
            on_chunk_batch(s, e, next_tok, batch)

    from array import array
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    tokens = [int(token) for token in tokens]
    n, chunk = len(tokens), int(chunk)
    if n < 1:
        raise ValueError("prefill requires at least one token")
    if chunk < 1:
        raise ValueError("chunk must be >= 1")

    if resume is not None:
        if not isinstance(resume, PrefillSession):
            raise TypeError("resume must be a PrefillSession")
        consumed = len(resume.tokens)
        if consumed > n or tuple(tokens[:consumed]) != resume.tokens:
            raise ValueError("resume tokens are not an exact prefix of the new prompt")
        req = resume.req
        rp = int(resume.req_pool_index)
        nti, ntl, batch = resume.next_token_ids, resume.next_token_logits, resume.batch
        if consumed == n:
            if sample_next_token is not None:
                nti = sample_next_token(ntl)
            session = PrefillSession(tuple(tokens), req, nti, ntl, batch, rp)
            result = (nti, ntl, batch)
            return result + (session,) if return_session else result
    else:
        consumed = 0
        first_end = min(chunk, n)
        new_api = hasattr(Req, "set_extend_range")
        first = array("q", tokens[:first_end]) if new_api else tokens[:first_end]
        req = Req(
            rid=int(request_id),
            origin_input_text="",
            origin_input_ids=first,
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        if new_api:
            req.full_untruncated_fill_ids = req.origin_input_ids  # upstream alias convention
        req.logprob_start_len = -1
        rp = -1

    while consumed < n:
        e = min(consumed + chunk, n)
        if consumed:
            req.prefix_indices = model_runner.req_to_token_pool.req_to_token[rp, :consumed].to(
                req.prefix_indices.dtype
            )
        _stage_extend(req, tokens, consumed, e)
        nti, ntl, batch = B.extend([req], model_runner)
        if e == n and sample_next_token is not None:
            nti = sample_next_token(ntl)
        rp = int(batch.req_pool_indices[0])
        notify(consumed, e, int(nti[0]), batch)
        consumed = e
    session = PrefillSession(tuple(tokens), req, nti, ntl, batch, rp)
    result = (nti, ntl, batch)
    return result + (session,) if return_session else result
