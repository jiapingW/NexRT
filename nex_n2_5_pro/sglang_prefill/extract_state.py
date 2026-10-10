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

"""SGLang prefill -> nexrt prefill state (per-rank, TP=8), with long-context support.

Offline ModelRunner path (reuses sglang.bench_one_batch load_model/extend). Per rank:
  - full-attn layers: paired hooks -- rotary_emb pre-hook captures PRE-rope k (post qk-norm) + positions;
    RadixAttention pre-hook captures v + SGLang's POST-rope k. k_cache is rebuilt with nexrt's OWN
    partial_rope (convention-safe; RoPE is a forward rotation we re-apply). Accumulated PER CHUNK so
    L > chunked prefill still rebuilds a contiguous [1, L, 256] per layer.
  - GDN layers: read the mamba pool after prefill -> conv[0][slot] (conv [1536,3]) + temporal[slot]
    (recurrent [8,128,128], TRANSPOSED [V,K]->[K,V]).
Also: (A) prints the RoPE re-apply delta (nexrt_rope(kpre) vs SGLang post-rope k) -- must be ~0.
      (B) --validate-chunk compares single-extend vs chunked-extend state (KV reconstruction + GDN carry).
      (C) --Ls sweeps lengths up to 262080, chunking to bound memory, reporting time + size.

Run: torchrun --standalone --nproc-per-node=8 extract_state.py --L 512 --out <dir>
"""

import argparse
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from nex_n2_5_pro.sglang_prefill.process_supervisor import torchrun_rank

from nex_n2_5_pro.sglang_prefill.weight_loading import load_model

_SGLANG_IMPORT_ERROR = None
try:
    import sglang.bench_one_batch as B
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.server_args import PortArgs, ServerArgs
except ImportError as exc:
    _SGLANG_IMPORT_ERROR = exc

MODEL = os.environ.get("NEXRT_MODEL_DIR", "/path/to/nex_n2_5_pro")
VOCAB = 248320
NLAYERS = 60
FULL_ATTN = [i for i in range(NLAYERS) if (i + 1) % 4 == 0]  # 3,7,...,59
GDN = [i for i in range(NLAYERS) if (i + 1) % 4 != 0]
ROTARY_DIM = 64
ROPE_THETA = 10000000.0
HEAD_DIM = 256
CHUNK_DEFAULT = 8192


def nexrt_partial_rope(x, positions):
    """x [L, H, 256] (post qk-norm k); positions [L] or [3,L] int -> neox partial-RoPE (rotary_dim 64, theta 1e7)."""
    d, half = ROTARY_DIM, ROTARY_DIM // 2
    dev = x.device
    pos = positions.float().to(dev)
    if pos.ndim > 1:  # SGLang mrope positions are [3, L]; text rows identical -> row 0
        pos = pos[0]
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, d, 2, device=dev).float() / d))  # [32]
    ang = pos[:, None] * inv[None, :]  # [L,32]
    cos = torch.cos(ang).repeat(1, 2)[:, None, :]  # [L,1,64]
    sin = torch.sin(ang).repeat(1, 2)[:, None, :]
    xr = x[..., :d].float()  # [L,H,64]
    rot = torch.cat([-xr[..., half:], xr[..., :half]], dim=-1)  # rotate_half (neox)
    out = xr * cos + rot * sin
    return torch.cat([out.to(x.dtype), x[..., d:]], dim=-1)  # [L,H,256]


# ------------------------- hook-based capture -------------------------
class Capture:
    def __init__(self, model, rope_flavor):
        self.rope_flavor = rope_flavor
        self.on = False
        self.pending = {}  # kpre + positions from the rotary_emb pre-hook
        self.kv = {
            lid: [] for lid in FULL_ATTN
        }  # lid -> list of (k_cache_chunk[Lc,nkv,256], v_chunk, pos)
        self.attn_calls = {lid: 0 for lid in FULL_ATTN}
        self.delta = None  # (relL2, maxabs, kabsmax) of nexrt_rope(kpre) vs sgl post-rope k
        self.hooks = []
        attn_mods = {
            m.layer_id: m
            for _n, m in model.named_modules()
            if isinstance(m, RadixAttention) and getattr(m, "layer_id", None) in FULL_ATTN
        }
        rope_mods = {
            id(m): m
            for _n, m in model.named_modules()
            if _n.split(".")[-1] == "rotary_emb" or "RotaryEmbedding" in type(m).__name__
        }
        self.n_attn, self.n_rope = len(attn_mods), len(rope_mods)
        for m in rope_mods.values():
            self.hooks.append(m.register_forward_pre_hook(self._rope_hook, with_kwargs=True))
        for lid, m in attn_mods.items():
            self.hooks.append(
                m.register_forward_pre_hook(self._mk_attn_hook(lid), with_kwargs=True)
            )

    def _rope_hook(self, mod, args, kwargs):
        if not self.on:
            return
        pos = kwargs.get("positions", args[0] if args else None)
        k = kwargs.get("key", args[2] if len(args) > 2 else None)
        # rotary_emb (forward_cuda) rotates k IN-PLACE -> must CLONE now to capture the true PRE-rope k
        self.pending["pos"] = None if pos is None else pos.detach().clone()
        self.pending["kpre"] = None if k is None else k.detach().clone()

    def _mk_attn_hook(self, lid):
        def hook(mod, args, kwargs):
            if not self.on:
                return
            kpost = kwargs.get("k", args[1] if len(args) > 1 else None)
            v = kwargs.get("v", args[2] if len(args) > 2 else None)
            pos = self.pending.get("pos")
            kpre = self.pending.get("kpre")
            if v is None or pos is None or kpre is None:
                return
            self.attn_calls[lid] += 1
            Lc = kpre.shape[0]
            kpre3 = kpre.detach().reshape(Lc, -1, HEAD_DIM)
            k_nr = nexrt_partial_rope(kpre3, pos)  # nexrt-rope k [Lc,nkv,256]
            if lid == FULL_ATTN[0] and self.delta is None and kpost is not None:
                kp = kpost.detach().reshape(Lc, -1, HEAD_DIM).float()
                dif = k_nr.float() - kp
                self.delta = (
                    float(dif.norm() / kp.norm().clamp_min(1e-9)),
                    float(dif.abs().max()),
                    float(kp.abs().max()),
                )
            kc = (
                (k_nr if self.rope_flavor == "nexrt" else kpost.detach().reshape(Lc, -1, HEAD_DIM))
                .to(torch.bfloat16)
                .cpu()
            )
            self.kv[lid].append(
                (
                    kc,
                    v.detach().reshape(Lc, -1, HEAD_DIM).to(torch.bfloat16).cpu(),
                    pos.detach().cpu(),
                )
            )

        return hook

    def reset(self):
        self.pending = {}
        self.kv = {lid: [] for lid in FULL_ATTN}
        self.attn_calls = {lid: 0 for lid in FULL_ATTN}

    def remove(self):
        for h in self.hooks:
            h.remove()

    def k_v_cache(self, lid, L):
        """Concatenate captured chunks in position order -> (k_cache[nkv,L,256], v_cache[nkv,L,256]) + pos check."""
        chunks = self.kv[lid]
        k = torch.cat([c[0] for c in chunks], dim=0)  # [L, nkv, 256]
        v = torch.cat([c[1] for c in chunks], dim=0)
        pos = torch.cat(
            [c[2].reshape(-1, c[2].shape[-1])[0] if c[2].ndim > 1 else c[2] for c in chunks], dim=0
        )
        assert k.shape[0] == L, f"L{lid}: rebuilt k len {k.shape[0]} != {L}"
        return k.permute(1, 0, 2).contiguous(), v.permute(1, 0, 2).contiguous(), pos


# ------------------------- prefill (single or chunked) -------------------------
def _sp():
    return SamplingParams(temperature=0, max_new_tokens=1)


def run_prefill(model_runner, tokens, chunk_size, cap):
    """(Possibly chunked) prefill of `tokens` as ONE continued request. Hooks fire per chunk and accumulate.
    Returns (next_token_ids, next_token_logits, batch)."""
    cap.reset()
    n = len(tokens)
    cap.on = True
    if n <= chunk_size:
        reqs = B.prepare_synthetic_inputs_for_latency_test(
            1, n, custom_inputs=[[int(x) for x in tokens]]
        )
        nti, ntl, batch = B.extend(reqs, model_runner)
        cap.on = False
        return nti, ntl, batch
    # chunked continued prefill (bounds per-forward memory; GDN pool + KV carry via prefix_indices)
    req = Req(
        rid=0,
        origin_input_text="",
        origin_input_ids=[int(x) for x in tokens[:chunk_size]],
        sampling_params=_sp(),
    )
    req.fill_ids = [int(x) for x in tokens[:chunk_size]]
    req.logprob_start_len = -1
    req.set_extend_input_len(
        len(req.fill_ids) - len(req.prefix_indices)
    )  # prefix_indices default (empty tensor)
    nti, ntl, batch = B.extend([req], model_runner)
    rp_idx = int(batch.req_pool_indices[0])
    consumed = chunk_size
    while consumed < n:
        e = min(consumed + chunk_size, n)
        req.fill_ids = [int(x) for x in tokens[:e]]
        req.prefix_indices = model_runner.req_to_token_pool.req_to_token[rp_idx, :consumed].to(
            req.prefix_indices.dtype
        )
        req.set_extend_input_len(e - consumed)
        nti, ntl, batch = B.extend([req], model_runner)
        rp_idx = int(batch.req_pool_indices[0])
        consumed = e
    cap.on = False
    return nti, ntl, batch


def read_gdn(model_runner, batch, transpose_ssm, flip_conv):
    slot = int(model_runner.req_to_token_pool.get_mamba_indices(batch.req_pool_indices)[0].item())
    out = {}
    for lid in GDN:
        lc = model_runner.req_to_token_pool.mamba2_layer_cache(lid)
        conv = lc.conv[0][slot].detach().to(torch.bfloat16).cpu()  # [1536,3]
        ssm = lc.temporal[slot].detach().float().cpu()  # [8, d, d]
        if flip_conv:
            conv = torch.flip(conv, dims=[-1]).contiguous()
        if transpose_ssm:
            ssm = ssm.transpose(1, 2).contiguous()  # FLA [V,K] -> nexrt [K,V]
        out[lid] = (conv, ssm)
    return slot, out


def assemble(cap, gdn_state, L):
    layers = {}
    for lid in range(NLAYERS):
        if lid in FULL_ATTN:
            k, v, _pos = cap.k_v_cache(lid, L)
            layers[lid] = {"kind": "full", "k_cache": k, "v_cache": v}
        else:
            conv, ssm = gdn_state[lid]
            layers[lid] = {"kind": "linear", "conv_state": conv, "recurrent_state": ssm}
    return layers


def save_state(outdir, tp_rank, tp, L, tokens, first_token, layers, extra=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dump = {
        "rank": tp_rank,
        "tp": tp,
        "L": L,
        "prompt_ids": torch.tensor(tokens, dtype=torch.long),
        "first_token": first_token,
        "layers": layers,
    }
    torch.save(dump, outdir / f"prefill_state_rank{tp_rank}.pt")
    if tp_rank == 0:
        meta = {
            "prompt_ids": torch.tensor(tokens, dtype=torch.long),
            "first_token": first_token,
            "L": L,
        }
        if extra:
            meta.update(extra)
        torch.save(meta, outdir / "prefill_meta.pt")


def _tokens_for(L, seed, tok, text, prompt_pt):
    if prompt_pt:  # reuse an existing prompt exactly
        pt = torch.load(prompt_pt, map_location="cpu", weights_only=False)
        return pt["prompt_ids"].tolist()[:L]
    from nex_n2_5_pro.real_prompts import build_real_prompt, _as_ids

    if text:  # chat-templated user text
        return _as_ids(
            tok.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
            )
        )
    return build_real_prompt(L, tok)  # default: deterministic real text


def _cmp_states(a_layers, b_layers, rp):
    """B-validation: element-wise relL2 between two states (single vs chunked). Should be ~0 (bit-ish)."""
    agg = {}
    for lid in a_layers:
        la, lb = a_layers[lid], b_layers[lid]
        comps = (
            [("k_cache",), ("v_cache",)]
            if la["kind"] == "full"
            else [("conv_state",), ("recurrent_state",)]
        )
        for (c,) in comps:
            x, y = la[c].float(), lb[c].float()
            n = (x - y).pow(2).sum().item()
            d = y.pow(2).sum().item()
            agg.setdefault(c, [0.0, 0.0])
            agg[c][0] += n
            agg[c][1] += d
    for c, (n, d) in agg.items():
        rp(f"@@ CHUNKCMP {c:16s} relL2 = {(n / max(d, 1e-9)) ** 0.5:.3e}")


def worker(server_args, port_args, gpu_id, tp_rank, cli):
    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)
    initialize_fp4_gemm_config(server_args)
    B.configure_logger(server_args, prefix=f" TP{tp_rank}")
    rp = print if tp_rank == 0 else (lambda *a, **k: None)
    # bench's minimal dummy tree cache lacks these (only hit on the continued-prefill / OOM message path)
    B.TreeCacheNamespace.available_and_evictable_str = lambda self: "(dummy tree cache)"
    B.TreeCacheNamespace.pretty_print = lambda self: None

    model_runner, tok = load_model(server_args, port_args, gpu_id, tp_rank)
    cap = Capture(model_runner.model, cli.rope)
    rp(
        f"[rank{tp_rank}] hooks: {cap.n_attn} full-attn RadixAttention, {cap.n_rope} rotary_emb (shared)"
    )

    Ls = [int(x) for x in cli.Ls.split(",")] if cli.Ls else [cli.L]

    for li, L in enumerate(Ls):
        tokens = _tokens_for(
            L, cli.seed + L, tok, cli.text if li == 0 else None, cli.prompt_pt if li == 0 else None
        )
        L = len(tokens)
        outdir = f"{cli.out}_{L}" if len(Ls) > 1 else cli.out

        torch.cuda.synchronize()
        t0 = time.time()
        nti, ntl, batch = run_prefill(model_runner, tokens, cli.chunk_size, cap)
        torch.cuda.synchronize()
        pref_t = time.time() - t0
        first_token = int(nti[0])
        nchunks = max(cap.attn_calls.values())
        if tp_rank == 0 and cap.delta is not None:
            rl2, mx, kmx = cap.delta
            rp(
                f"@@ ROPE_DELTA L={L}: relL2={rl2:.3e} maxabs={mx:.3e} (k absmax={kmx:.3f})  "
                f"[nexrt_rope(kpre) vs SGLang post-rope k]"
            )
        rp(
            f"@@ PREFILL L={L:7d} wall={pref_t:.3f}s first_token={first_token} chunks/layer={nchunks} "
            f"chunk_size={cli.chunk_size}"
        )

        te = time.time()
        slot, gdn_state = read_gdn(model_runner, batch, cli.transpose_ssm, cli.flip_conv)
        layers = assemble(cap, gdn_state, L)
        # reconstruction validation (B): every full-attn layer must rebuild to length L
        klens = {lid: layers[lid]["k_cache"].shape[1] for lid in FULL_ATTN}
        ok_len = all(v == L for v in klens.values())
        rp(f"@@ RECON  L={L}: all full-attn k_cache length=={L}: {ok_len}  (mamba_slot={slot})")

        extra = None
        if li == 0 and cli.ref_steps > 0:
            ref = [first_token]
            try:
                cur = nti
                for _ in range(cli.ref_steps):
                    cur, _ = B.decode(cur, batch, model_runner)
                    ref.append(int(cur[0]))
            except Exception as e:  # noqa: BLE001
                rp(f"[rank{tp_rank}] ref decode failed: {e!r}")
            extra = {"sgl_ref_tokens": ref, "logits_last": ntl.detach().float().cpu()}
            rp(f"[rank{tp_rank}] sgl greedy ref: {ref}")

        # B-validation: single vs chunked at this L (only if requested and L>chunk_size feasible both ways)
        if li == 0 and cli.validate_chunk and L > cli.chunk_size:
            layers_single = layers  # 'layers' above used cli.chunk_size (chunked)
            _n, _l, batch2 = run_prefill(model_runner, tokens, 10**9, cap)  # force single forward
            _slot2, gdn2 = read_gdn(model_runner, batch2, cli.transpose_ssm, cli.flip_conv)
            layers_full = assemble(cap, gdn2, L)
            rp(
                f"@@ CHUNKCMP L={L}: single-extend vs {-(-L // cli.chunk_size)}-chunk (relL2, expect ~0):"
            )
            _cmp_states(layers_full, layers_single, rp)

        if not cli.no_save:
            save_state(outdir, tp_rank, server_args.tp_size, L, tokens, first_token, layers, extra)
        if tp_rank == 0:
            nbytes = sum(
                t.numel() * t.element_size()
                for lid in layers
                for t in layers[lid].values()
                if torch.is_tensor(t)
            )
            rp(
                f"@@ EXTRACT L={L}: extract+save={time.time() - te:.2f}s  per-rank state={nbytes / 1e9:.3f} GB  "
                f"saved={not cli.no_save} -> {outdir if not cli.no_save else '(not saved)'}"
            )
        del layers
        print(f"@@ RANK{tp_rank}_L{L}_DONE", flush=True)

    cap.remove()
    print(f"@@ RANK{tp_rank}_ALL_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", type=str, default=MODEL, help="model directory (default: NEXRT_MODEL_DIR)"
    )
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument(
        "--Ls",
        type=str,
        default=None,
        help="comma list of lengths to sweep (each saved to <out>_<L>)",
    )
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_DEFAULT,
        help="max tokens per forward (bounds memory)",
    )
    ap.add_argument("--no-transpose-ssm", dest="transpose_ssm", action="store_false")
    ap.set_defaults(transpose_ssm=True)
    ap.add_argument("--flip-conv", action="store_true")
    ap.add_argument("--ref-steps", type=int, default=8)
    ap.add_argument("--rope", choices=["nexrt", "sgl"], default="nexrt")
    ap.add_argument("--prompt-pt", type=str, default=None)
    ap.add_argument("--text", type=str, default=None)
    ap.add_argument(
        "--validate-chunk",
        action="store_true",
        help="compare single-extend vs chunked state at first L",
    )
    ap.add_argument(
        "--no-save", action="store_true", help="measure only, do not write state to disk"
    )
    ap.add_argument("--mem-fraction", type=float, default=0.72)
    ap.add_argument(
        "--context-length", type=int, default=262144
    )  # model max_position_embeddings (256K); covers 262080
    ap.add_argument(
        "--max-prefill-tokens",
        type=int,
        default=16384,
        help="bounds the one-time DeepGEMM warmup token range; ModelRunner extend can still exceed it (JIT on demand)",
    )
    cli = ap.parse_args()
    rank = torchrun_rank(8)

    if _SGLANG_IMPORT_ERROR is not None:
        ap.error(
            "state generation requires SGLang in the active Python environment "
            f"(import failed: {_SGLANG_IMPORT_ERROR})"
        )

    server_args = ServerArgs(
        model_path=cli.model,
        tp_size=8,
        trust_remote_code=True,
        disable_cuda_graph=True,
        disable_radix_cache=True,
        mem_fraction_static=cli.mem_fraction,
        context_length=cli.context_length,
        attention_backend="fa3",
        chunked_prefill_size=min(cli.chunk_size, 16384),
        max_prefill_tokens=cli.max_prefill_tokens,
    )
    server_args.cuda_graph_max_bs = 1
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)

    try:
        worker(server_args, port_args, rank, rank, cli)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
