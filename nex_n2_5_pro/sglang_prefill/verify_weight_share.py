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

"""sglang-bridge verification: can nexrt alias sglang's loaded weights on 8xH100?

Loads the model with SGLang's REAL loader (TP=8, same path a prefill server takes), then per rank:
  1. INVENTORY  - runtime param name/shape/dtype/stride for one GDN / one attn / one MoE layer,
                  KV-pool + mamba-pool layout, memory split (weights vs pools vs free).
  2. BYTE-COMPARE - sglang's loaded per-rank tensors vs the exact checkpoint slices nexrt's
                  loaders take (LAYERS_DIR per-layer safetensors). PASS == bit-identical, i.e.
                  nexrt could alias the sglang tensor instead of loading its own copy.
                  (nexrt's gamma-FOLDED tensors are expected to differ; what we verify here is
                  the RAW-value hypothesis: sglang tensor == checkpoint slice == what nexrt
                  would read if it un-folds gamma.)
  3. TINY PREFILL - 256-token extend; verify KV pool gather + mamba slot state shapes nexrt's
                  handover kernel would consume.

Run: torchrun --standalone --nproc-per-node=8 verify_weight_share.py
"""

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
LAYERS_DIR = Path(os.environ.get("NEXRT_LAYERS_DIR", "/path/to/nexrt-layers"))
MODEL = os.environ.get("NEXRT_MODEL_DIR", "/path/to/nex_n2_5_pro")
if __package__ in (None, ""):
    sys.path.insert(0, str(HERE.parent.parent))


from safetensors import safe_open  # noqa: E402

from nex_n2_5_pro.sglang_prefill.weight_loading import load_model  # noqa: E402

import sglang.bench_one_batch as B  # noqa: E402
from sglang.srt.entrypoints.engine import _set_envs_and_config  # noqa: E402
from sglang.srt.layers.moe import initialize_moe_config  # noqa: E402
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config  # noqa: E402
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config  # noqa: E402
from sglang.srt.server_args import PortArgs, ServerArgs  # noqa: E402

from nex_n2_5_pro.sglang_prefill.process_supervisor import torchrun_rank  # noqa: E402

NLAYERS = 60
RINTER = 128  # moe_intermediate(1024) / TP(8)
MOE_LAYERS_FULL = [3]  # full 512-expert byte check
MOE_LAYERS_SAMPLE = [0, 31, 59]  # sampled experts
SAMPLE_EXPERTS = [0, 7, 255, 511]
ATTN_LAYERS = [3, 59]
GDN_LAYERS = [0, 58]


def _bytes_eq(a, b):
    """Bit-equality for any dtype pair of same shape (fp8-safe)."""
    if a.shape != b.shape:
        return False, f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    au = a.contiguous().view(torch.uint8) if a.dtype == torch.float8_e4m3fn else a.contiguous()
    bu = b.contiguous().view(torch.uint8) if b.dtype == torch.float8_e4m3fn else b.contiguous()
    if au.dtype != bu.dtype:
        return False, f"dtype {a.dtype} vs {b.dtype}"
    ok = torch.equal(au.cpu(), bu.cpu())
    return ok, "" if ok else "bytes differ"


class Report:
    def __init__(self, rank):
        self.rank = rank
        self.fail = 0

    def check(self, name, ok, detail=""):
        tag = "PASS" if ok else "FAIL"
        if not ok:
            self.fail += 1
        print(f"@@ R{self.rank} CHECK {name}: {tag} {detail}", flush=True)

    def info(self, name, detail):
        print(f"@@ R{self.rank} INFO {name}: {detail}", flush=True)


def _sf(layer):
    return safe_open(str(LAYERS_DIR / f"layer.{layer}.safetensors"), framework="pt", device="cpu")


def _pfx(layer):
    return f"model.language_model.layers.{layer}."


# ---------------- reference slices (the exact shard math nexrt's loaders use) ----------------
def ref_moe_expert(layer, e, rank):
    lo, hi = rank * RINTER, (rank + 1) * RINTER
    p = _pfx(layer) + "mlp."
    with _sf(layer) as f:
        g1 = f.get_slice(f"{p}experts.{e}.gate_proj.weight")[lo:hi]  # [128,4096] fp8
        g3 = f.get_slice(f"{p}experts.{e}.up_proj.weight")[lo:hi]  # [128,4096]
        g2 = f.get_slice(f"{p}experts.{e}.down_proj.weight")[:, lo:hi]  # [4096,128]
        s1 = f.get_slice(f"{p}experts.{e}.gate_proj.weight_scale")[rank : rank + 1]  # [1,32]
        s3 = f.get_slice(f"{p}experts.{e}.up_proj.weight_scale")[rank : rank + 1]  # [1,32]
        s2 = f.get_slice(f"{p}experts.{e}.down_proj.weight_scale")[:, rank : rank + 1]  # [32,1]
    return g1, g3, g2, s1.float(), s3.float(), s2.float()


def ref_moe_shared(layer, rank):
    lo, hi = rank * RINTER, (rank + 1) * RINTER
    p = _pfx(layer) + "mlp.shared_expert."
    with _sf(layer) as f:
        g1 = f.get_slice(p + "gate_proj.weight")[lo:hi]
        g3 = f.get_slice(p + "up_proj.weight")[lo:hi]
        g2 = f.get_slice(p + "down_proj.weight")[:, lo:hi]
        s1 = f.get_slice(p + "gate_proj.weight_scale")[rank : rank + 1]
        s3 = f.get_slice(p + "up_proj.weight_scale")[rank : rank + 1]
        s2 = f.get_slice(p + "down_proj.weight_scale")[:, rank : rank + 1]
    return g1, g3, g2, s1.float(), s3.float(), s2.float()


def ref_attn(layer, rank):
    """nexrt shard math (nex_n2_5_pro_full_attn_ref.shard_attn_weights): q rows rank*2048, kv head rank//4,
    o cols rank*1024; scales on the same 128-block boundaries."""
    h = rank // 4
    p = _pfx(layer) + "self_attn."
    with _sf(layer) as f:
        q = f.get_slice(p + "q_proj.weight")[rank * 2048 : (rank + 1) * 2048]
        qs = f.get_slice(p + "q_proj.weight_scale")[rank * 16 : (rank + 1) * 16]
        k = f.get_slice(p + "k_proj.weight")[h * 256 : (h + 1) * 256]
        ks = f.get_slice(p + "k_proj.weight_scale")[h * 2 : (h + 1) * 2]
        v = f.get_slice(p + "v_proj.weight")[h * 256 : (h + 1) * 256]
        vs = f.get_slice(p + "v_proj.weight_scale")[h * 2 : (h + 1) * 2]
        o = f.get_slice(p + "o_proj.weight")[:, rank * 1024 : (rank + 1) * 1024]
        os_ = f.get_slice(p + "o_proj.weight_scale")[:, rank * 8 : (rank + 1) * 8]
        qn = f.get_tensor(p + "q_norm.weight")
        kn = f.get_tensor(p + "k_norm.weight")
    return dict(
        q=q,
        qs=qs.float(),
        k=k,
        ks=ks.float(),
        v=v,
        vs=vs.float(),
        o=o,
        os=os_.float(),
        qn=qn,
        kn=kn,
    )


def _gather_qkv_rows(get, name, rank, kl, vl, kb, vb):
    q = get(name)[rank * kl : (rank + 1) * kl]
    k = get(name)[kb + rank * kl : kb + (rank + 1) * kl]
    v = get(name)[vb + rank * vl : vb + (rank + 1) * vl]
    return torch.cat([q, k, v], 0)


def ref_gdn(layer, rank):
    """nexrt shard math (nex_n2_5_pro_linear_attn_ref.shard_linear_attn_weights)."""
    p = _pfx(layer) + "linear_attn."
    with _sf(layer) as f:
        get = f.get_slice
        qkv = _gather_qkv_rows(
            get, p + "in_proj_qkv.weight", rank, 256, 1024, 2048, 4096
        )  # [1536,4096]
        qkvs = _gather_qkv_rows(get, p + "in_proj_qkv.weight_scale", rank, 2, 8, 16, 32)  # [12,32]
        z = get(p + "in_proj_z.weight")[rank * 1024 : (rank + 1) * 1024]
        zs = get(p + "in_proj_z.weight_scale")[rank * 8 : (rank + 1) * 8]
        b = get(p + "in_proj_b.weight")[rank * 8 : (rank + 1) * 8]
        a = get(p + "in_proj_a.weight")[rank * 8 : (rank + 1) * 8]
        conv = _gather_qkv_rows(get, p + "conv1d.weight", rank, 256, 1024, 2048, 4096)  # [1536,1,4]
        A_log = f.get_tensor(p + "A_log")[rank * 8 : (rank + 1) * 8]
        dt = f.get_tensor(p + "dt_bias")[rank * 8 : (rank + 1) * 8]
        o = get(p + "out_proj.weight")[:, rank * 1024 : (rank + 1) * 1024]
        os_ = get(p + "out_proj.weight_scale")[:, rank * 8 : (rank + 1) * 8]
        nw = f.get_tensor(p + "norm.weight")
    return dict(
        qkv=qkv,
        qkvs=qkvs.float(),
        z=z,
        zs=zs.float(),
        b=b,
        a=a,
        conv=conv,
        A_log=A_log,
        dt_bias=dt,
        o=o,
        os=os_.float(),
        norm=nw,
    )


# ---------------- runtime lookups ----------------
def _module(model, path):
    m = model
    for part in path.split("."):
        m = getattr(m, part) if not part.isdigit() else m[int(part)]
    return m


def _gdn_runtime_fields(gdn):
    """Return logical QKV/Z/B/A tensors from either supported SGLang layout."""
    if hasattr(gdn, "in_proj_qkv"):
        try:
            return {
                "qkv": gdn.in_proj_qkv.weight,
                "qkvs": gdn.in_proj_qkv.weight_scale_inv,
                "z": gdn.in_proj_z.weight,
                "zs": gdn.in_proj_z.weight_scale_inv,
                "b": gdn.in_proj_b.weight,
                "a": gdn.in_proj_a.weight,
            }
        except AttributeError as exc:
            raise RuntimeError("incomplete separate GDN projection representation") from exc

    if hasattr(gdn, "in_proj_qkvz"):
        try:
            qkvz = gdn.in_proj_qkvz.weight
            qkvz_scales = gdn.in_proj_qkvz.weight_scale_inv
            ba = gdn.in_proj_ba.weight
        except AttributeError as exc:
            raise RuntimeError("incomplete fused GDN projection representation") from exc
        return {
            "qkv": qkvz[:1536],
            "qkvs": qkvz_scales[:12],
            "z": qkvz[1536:],
            "zs": qkvz_scales[12:],
            "b": ba[:8],
            "a": ba[8:],
        }

    raise RuntimeError(
        "unsupported GDN projection representation: expected in_proj_qkv or in_proj_qkvz"
    )


def dump_params(rep, tag, mod, limit=64):
    for n, p in mod.named_parameters():
        rep.info(
            f"{tag}.{n}",
            f"shape={tuple(p.shape)} dtype={p.dtype} stride={tuple(p.stride())} "
            f"contig={p.is_contiguous()}",
        )
        limit -= 1
        if limit <= 0:
            break


def worker(server_args, port_args, gpu_id, tp_rank, only_moe_scales=False):
    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)
    initialize_fp4_gemm_config(server_args)
    B.configure_logger(server_args, prefix=f" TP{tp_rank}")
    B.TreeCacheNamespace.available_and_evictable_str = lambda self: "(dummy tree cache)"
    B.TreeCacheNamespace.pretty_print = lambda self: None

    model_runner, tok = load_model(server_args, port_args, gpu_id, tp_rank)
    rep = Report(tp_rank)
    model = model_runner.model
    lm = model.model  # language model (NexN25ProMoeModel)
    dev = next(model.parameters()).device

    if only_moe_scales:
        # Fast re-check after the expert-scale rename fix: routed scales + weight spot-check only.
        for L in MOE_LAYERS_FULL + MOE_LAYERS_SAMPLE:
            moe = _module(lm, f"layers.{L}.mlp.experts")
            s13, s2 = moe.w13_weight_scale_inv, moe.w2_weight_scale_inv
            rep.info(
                f"moe.L{L}.scale_sample",
                f"s13[0,0,:3]={s13[0, 0, :3].float().cpu().tolist()} "
                f"s2[0,:3,0]={s2[0, :3, 0].float().cpu().tolist()} "
                f"all_ones={bool((s13 == 1).all().item() and (s2 == 1).all().item())}",
            )
            ok_all, first_bad = True, None
            for e in range(512) if L in MOE_LAYERS_FULL else SAMPLE_EXPERTS:
                g1, g3, g2, rs1, rs3, rs2 = ref_moe_expert(L, e, tp_rank)
                os13, _ = _bytes_eq(s13[e].cpu(), torch.cat([rs1, rs3], 0))
                os2, _ = _bytes_eq(s2[e].cpu(), rs2)
                ow1, _ = (
                    _bytes_eq(moe.w13_weight[e, :RINTER].cpu(), g1)
                    if e in SAMPLE_EXPERTS
                    else (True, "")
                )
                ow2, _ = (
                    _bytes_eq(moe.w2_weight[e].cpu(), g2) if e in SAMPLE_EXPERTS else (True, "")
                )
                if not (os13 and os2 and ow1 and ow2):
                    ok_all = False
                    if first_bad is None:
                        first_bad = f"e{e} s13={os13} s2={os2} w1={ow1} w2={ow2}"
            n = 512 if L in MOE_LAYERS_FULL else len(SAMPLE_EXPERTS)
            rep.check(
                f"moe.L{L}.experts[{n}] SCALES(+w spot) == ckpt slice", ok_all, first_bad or ""
            )
        sh = _module(lm, "layers.3.mlp.shared_expert")
        g1, g3, g2, rs1, rs3, rs2 = ref_moe_shared(3, tp_rank)
        oks, _ = _bytes_eq(sh.gate_up_proj.weight_scale_inv.cpu(), torch.cat([rs1, rs3], 0))
        oks2, _ = _bytes_eq(sh.down_proj.weight_scale_inv.cpu(), rs2)
        rep.check("moe.L3.shared_expert scales == ckpt (rename regression check)", oks and oks2)
        # numeric sanity: with real scales a tiny prefill must give finite, non-degenerate logits
        reqs = B.prepare_synthetic_inputs_for_latency_test(1, 256)
        nti, ntl, batch = B.extend(reqs, model_runner)
        lf = ntl.float()
        rep.check(
            "prefill(256) logits finite + sane",
            bool(torch.isfinite(lf).all().item()) and float(lf.abs().max()) < 1e4,
            f"absmax={float(lf.abs().max()):.2f} next={int(nti[0])}",
        )
        # pool layouts + memory split (for the bridge handover design)
        pool = model_runner.token_to_kv_pool
        rep.info("kv_pool", f"{type(pool).__name__}")
        try:
            kb = pool.get_key_buffer(3)
            rep.info("kv_pool.k_buffer[L3]", f"shape={tuple(kb.shape)} dtype={kb.dtype}")
        except Exception as e:  # noqa: BLE001
            rep.info("kv_pool.buffers", f"accessor failed: {e!r}")
        r2t = model_runner.req_to_token_pool
        rp_idx = int(batch.req_pool_indices[0])
        idx = r2t.req_to_token[rp_idx, :256]
        rep.info(
            "req_to_token[:256]",
            f"first={int(idx[0])} last={int(idx[-1])} "
            f"contiguous={bool((idx[1:] - idx[:-1] == 1).all())}",
        )
        try:
            slot = int(r2t.get_mamba_indices(batch.req_pool_indices)[0].item())
            lc = r2t.mamba2_layer_cache(0)
            rep.info(
                "mamba[L0]",
                f"slot={slot} conv={tuple(lc.conv[0][slot].shape)} {lc.conv[0][slot].dtype} "
                f"temporal={tuple(lc.temporal[slot].shape)} {lc.temporal[slot].dtype} "
                f"pool_conv={tuple(lc.conv[0].shape)} pool_temporal={tuple(lc.temporal.shape)}",
            )
        except Exception as e:  # noqa: BLE001
            rep.info("mamba.pool", f"accessor failed: {e!r}")
        torch.cuda.synchronize()
        wbytes = sum(p.numel() * p.element_size() for p in model.parameters())
        free, total = torch.cuda.mem_get_info(dev)
        rep.info(
            "mem",
            f"weights={wbytes / 2**30:.2f}GiB allocated={torch.cuda.memory_allocated(dev) / 2**30:.2f}GiB "
            f"reserved={torch.cuda.memory_reserved(dev) / 2**30:.2f}GiB free={free / 2**30:.2f}GiB",
        )
        print(f"@@ RANK{tp_rank}_DONE fails={rep.fail}", flush=True)
        if rep.fail:
            raise SystemExit(1)
        return

    # ---------------- 1. inventory (rank 0 + rank 5 for kv-head cross-check) ----------------
    if tp_rank in (0, 5):
        dump_params(rep, "L0(gdn)", _module(lm, "layers.0.linear_attn"))
        dump_params(rep, "L3(attn)", _module(lm, "layers.3"))
        moe3 = _module(lm, "layers.3.mlp")
        for n in (
            "experts.w13_weight",
            "experts.w2_weight",
            "experts.w13_weight_scale_inv",
            "experts.w2_weight_scale_inv",
        ):
            try:
                p = _module(moe3, n)
                rep.info(
                    f"L3.mlp.{n}",
                    f"shape={tuple(p.shape)} dtype={p.dtype} stride={tuple(p.stride())}",
                )
            except AttributeError:
                rep.info(f"L3.mlp.{n}", "MISSING")
        dump_params(rep, "L3.mlp.gate", moe3.gate)
        dump_params(rep, "L3.mlp.shared", moe3.shared_expert)
        emb = _module(lm, "embed_tokens")
        rep.info("embed", f"{tuple(emb.weight.shape)} {emb.weight.dtype}")
        rep.info("lm_head", f"{tuple(model.lm_head.weight.shape)} {model.lm_head.weight.dtype}")
        rep.info("final_norm", f"{tuple(lm.norm.weight.shape)} {lm.norm.weight.dtype}")
        vis_bytes = sum(
            p.numel() * p.element_size() for n, p in model.named_parameters() if "visual" in n
        )
        rep.info("visual_tower_bytes", f"{vis_bytes / 1e9:.3f} GB")

    # ---------------- 2. byte-compare vs checkpoint slices ----------------
    # MoE routed experts
    for L in MOE_LAYERS_FULL + MOE_LAYERS_SAMPLE:
        moe = _module(lm, f"layers.{L}.mlp.experts")
        w13, w2 = moe.w13_weight, moe.w2_weight
        s13, s2 = moe.w13_weight_scale_inv, moe.w2_weight_scale_inv
        experts = range(512) if L in MOE_LAYERS_FULL else SAMPLE_EXPERTS
        ok_all, first_bad = True, None
        for e in experts:
            g1, g3, g2, rs1, rs3, rs2 = ref_moe_expert(L, e, tp_rank)
            o1, _ = _bytes_eq(w13[e, :RINTER].cpu(), g1)
            o3, _ = _bytes_eq(w13[e, RINTER:].cpu(), g3)
            o2, _ = _bytes_eq(w2[e].cpu(), g2)
            os13, _ = _bytes_eq(s13[e].cpu(), torch.cat([rs1, rs3], 0))
            os2, _ = _bytes_eq(s2[e].cpu(), rs2)
            if not (o1 and o3 and o2 and os13 and os2):
                ok_all = False
                if first_bad is None:
                    first_bad = f"e{e} w1={o1} w3={o3} w2={o2} s13={os13} s2={os2}"
        n = 512 if L in MOE_LAYERS_FULL else len(SAMPLE_EXPERTS)
        rep.check(f"moe.L{L}.experts[{n}] w13|w2|scales == ckpt slice", ok_all, first_bad or "")

    # MoE shared expert + gates (bf16)
    for L in [3]:
        moe = _module(lm, f"layers.{L}.mlp")
        g1, g3, g2, rs1, rs3, rs2 = ref_moe_shared(L, tp_rank)
        su = moe.shared_expert.gate_up_proj
        sd = moe.shared_expert.down_proj
        ok1, d1 = _bytes_eq(su.weight[:RINTER].cpu(), g1)
        ok3, d3 = _bytes_eq(su.weight[RINTER:].cpu(), g3)
        ok2, d2 = _bytes_eq(sd.weight.cpu(), g2)
        oks, _ = _bytes_eq(su.weight_scale_inv.cpu(), torch.cat([rs1, rs3], 0))
        oks2, _ = _bytes_eq(sd.weight_scale_inv.cpu(), rs2)
        rep.check(
            f"moe.L{L}.shared_expert == ckpt slice",
            ok1 and ok3 and ok2 and oks and oks2,
            f"{d1}{d3}{d2}",
        )
        with _sf(L) as f:
            gate_ref = f.get_tensor(_pfx(L) + "mlp.gate.weight")
            sgate_ref = f.get_tensor(_pfx(L) + "mlp.shared_expert_gate.weight")
        okg, dg = _bytes_eq(moe.gate.weight.cpu().to(gate_ref.dtype), gate_ref)
        oksg, dsg = _bytes_eq(moe.shared_expert_gate.weight.cpu().to(sgate_ref.dtype), sgate_ref)
        rep.check(
            f"moe.L{L}.gate + shared_expert_gate == ckpt (replicated)", okg and oksg, f"{dg}{dsg}"
        )

    # Full attention
    for L in ATTN_LAYERS:
        lay = _module(lm, f"layers.{L}")
        r = ref_attn(L, tp_rank)
        qkv_ref = torch.cat([r["q"], r["k"], r["v"]], 0)
        qkvs_ref = torch.cat([r["qs"], r["ks"], r["vs"]], 0)
        ok, d = _bytes_eq(lay.qkv_proj.weight.cpu(), qkv_ref)
        oks, ds = _bytes_eq(lay.qkv_proj.weight_scale_inv.cpu(), qkvs_ref)
        rep.check(
            f"attn.L{L}.qkv_proj == ckpt [q(r*2048)|k(h={tp_rank // 4})|v] slice",
            ok and oks,
            d + ds,
        )
        oko, do = _bytes_eq(lay.o_proj.weight.cpu(), r["o"])
        okos, _ = _bytes_eq(lay.o_proj.weight_scale_inv.cpu(), r["os"])
        rep.check(f"attn.L{L}.o_proj == ckpt col slice", oko and okos, do)
        okn, _ = _bytes_eq(lay.q_norm.weight.cpu().to(r["qn"].dtype), r["qn"])
        okk, _ = _bytes_eq(lay.k_norm.weight.cpu().to(r["kn"].dtype), r["kn"])
        with _sf(L) as f:
            iln = f.get_tensor(_pfx(L) + "input_layernorm.weight")
            pln = f.get_tensor(_pfx(L) + "post_attention_layernorm.weight")
        oki, _ = _bytes_eq(lay.input_layernorm.weight.cpu().to(iln.dtype), iln)
        okp, _ = _bytes_eq(lay.post_attention_layernorm.weight.cpu().to(pln.dtype), pln)
        rep.check(f"attn.L{L}.norms(q/k/input/post) == ckpt", okn and okk and oki and okp)

    # GDN
    for L in GDN_LAYERS:
        gdn = _module(lm, f"layers.{L}.linear_attn")
        r = ref_gdn(L, tp_rank)
        actual = _gdn_runtime_fields(gdn)
        ok, d = _bytes_eq(actual["qkv"].cpu(), r["qkv"])
        oks, ds = _bytes_eq(actual["qkvs"].cpu(), r["qkvs"])
        rep.check(f"gdn.L{L}.in_proj_qkv == ckpt gathered [q|k|v] slice", ok and oks, d + ds)
        okz, dz = _bytes_eq(actual["z"].cpu(), r["z"])
        okzs, dzs = _bytes_eq(actual["zs"].cpu(), r["zs"])
        rep.check(f"gdn.L{L}.in_proj_z == ckpt slice", okz and okzs, dz + dzs)
        okb, _ = _bytes_eq(actual["b"].cpu().to(r["b"].dtype), r["b"])
        oka, _ = _bytes_eq(actual["a"].cpu().to(r["a"].dtype), r["a"])
        rep.check(f"gdn.L{L}.in_proj_b/a == ckpt slice (bf16)", okb and oka)
        okc, dc = _bytes_eq(
            gdn.conv1d.weight.reshape(1536, -1).cpu().to(r["conv"].dtype),
            r["conv"].reshape(1536, -1),
        )
        rep.check(f"gdn.L{L}.conv1d == ckpt gathered slice", okc, dc)
        okA, _ = _bytes_eq(gdn.A_log.cpu().float(), r["A_log"].float())
        okd, _ = _bytes_eq(gdn.dt_bias.cpu().float(), r["dt_bias"].float())
        okn, _ = _bytes_eq(gdn.norm.weight.cpu().to(r["norm"].dtype), r["norm"])
        rep.check(f"gdn.L{L}.A_log/dt_bias/norm == ckpt slice", okA and okd and okn)
        oko, do = _bytes_eq(gdn.out_proj.weight.cpu(), r["o"])
        okos, _ = _bytes_eq(gdn.out_proj.weight_scale_inv.cpu(), r["os"])
        rep.check(f"gdn.L{L}.out_proj == ckpt col slice", oko and okos, do)

    # embed / lm_head / final norm
    with safe_open(str(LAYERS_DIR / "embed_tokens.safetensors"), framework="pt", device="cpu") as f:
        emb_ref = f.get_slice("model.language_model.embed_tokens.weight")[
            tp_rank * 31040 : (tp_rank + 1) * 31040
        ]
    with safe_open(str(LAYERS_DIR / "lm_head.safetensors"), framework="pt", device="cpu") as f:
        lmh_ref = f.get_slice("lm_head.weight")[tp_rank * 31040 : (tp_rank + 1) * 31040]
    with safe_open(str(LAYERS_DIR / "norm.safetensors"), framework="pt", device="cpu") as f:
        fn_ref = f.get_tensor("model.language_model.norm.weight")
    emb = _module(lm, "embed_tokens")
    oke, de = _bytes_eq(emb.weight.cpu()[:31040].to(emb_ref.dtype), emb_ref)
    okl, dl = _bytes_eq(model.lm_head.weight.cpu()[:31040].to(lmh_ref.dtype), lmh_ref)
    okf, _ = _bytes_eq(lm.norm.weight.cpu().to(fn_ref.dtype), fn_ref)
    rep.check("embed/lm_head vocab shard + final norm == ckpt slice", oke and okl and okf, de + dl)

    # ---------------- 3. tiny prefill -> pool layouts ----------------
    reqs = B.prepare_synthetic_inputs_for_latency_test(1, 256)
    nti, ntl, batch = B.extend(reqs, model_runner)
    pool = model_runner.token_to_kv_pool
    rep.info("kv_pool", f"{type(pool).__name__}")
    try:
        kb = pool.get_key_buffer(3)
        vb = pool.get_value_buffer(3)
        rep.info("kv_pool.k_buffer[L3]", f"shape={tuple(kb.shape)} dtype={kb.dtype}")
        rep.info("kv_pool.v_buffer[L3]", f"shape={tuple(vb.shape)} dtype={vb.dtype}")
    except Exception as e:  # noqa: BLE001
        rep.info("kv_pool.buffers", f"accessor failed: {e!r}")
    r2t = model_runner.req_to_token_pool
    rp_idx = int(batch.req_pool_indices[0])
    idx = r2t.req_to_token[rp_idx, :256]
    rep.info(
        "req_to_token[:256]",
        f"first={int(idx[0])} last={int(idx[-1])} "
        f"contiguous={bool((idx[1:] - idx[:-1] == 1).all())}",
    )
    try:
        slot = int(r2t.get_mamba_indices(batch.req_pool_indices)[0].item())
        lc = r2t.mamba2_layer_cache(0)
        rep.info(
            "mamba.conv[L0]",
            f"slot={slot} shape={tuple(lc.conv[0][slot].shape)} dtype={lc.conv[0][slot].dtype}",
        )
        rep.info(
            "mamba.temporal[L0]",
            f"shape={tuple(lc.temporal[slot].shape)} dtype={lc.temporal[slot].dtype}",
        )
        rep.info(
            "mamba.pool",
            f"conv_full={tuple(lc.conv[0].shape)} temporal_full={tuple(lc.temporal.shape)}",
        )
    except Exception as e:  # noqa: BLE001
        rep.info("mamba.pool", f"accessor failed: {e!r}")

    # ---------------- memory ----------------
    torch.cuda.synchronize()
    wbytes = sum(p.numel() * p.element_size() for p in model.parameters())
    alloc = torch.cuda.memory_allocated(dev)
    reserved = torch.cuda.memory_reserved(dev)
    free, total = torch.cuda.mem_get_info(dev)
    rep.info(
        "mem",
        f"weights={wbytes / 2**30:.2f}GiB allocated={alloc / 2**30:.2f}GiB "
        f"reserved={reserved / 2**30:.2f}GiB free={free / 2**30:.2f}GiB total={total / 2**30:.2f}GiB",
    )
    kv_bytes = 0
    try:
        kv_bytes = sum(
            pool.get_key_buffer(i).numel() * pool.get_key_buffer(i).element_size()
            + pool.get_value_buffer(i).numel() * pool.get_value_buffer(i).element_size()
            for i in range(60)
            if (i + 1) % 4 == 0
        )
        rep.info("kv_pool_bytes", f"{kv_bytes / 2**30:.2f}GiB")
    except Exception:  # noqa: BLE001
        pass

    print(f"@@ RANK{tp_rank}_DONE fails={rep.fail}", flush=True)
    if rep.fail:
        raise SystemExit(1)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only-moe-scales",
        action="store_true",
        help="fast mode: skip prefill/pools, re-check routed-expert scales only",
    )
    cli = ap.parse_args()
    rank = torchrun_rank(8)
    server_args = ServerArgs(
        model_path=MODEL,
        tp_size=8,
        trust_remote_code=True,
        disable_cuda_graph=True,
        disable_radix_cache=True,
        mem_fraction_static=0.72,
        context_length=262144,
        attention_backend="fa3",
        chunked_prefill_size=8192,
        max_prefill_tokens=512 if cli.only_moe_scales else 16384,
    )
    server_args.cuda_graph_max_bs = 1
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)

    try:
        worker(server_args, port_args, rank, rank, cli.only_moe_scales)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
