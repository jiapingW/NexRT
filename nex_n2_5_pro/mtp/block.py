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

"""The MTP draft block: weights, buffers, and the three launch phases of one round.

One round (uniform for every replay, incl. the first after hand-off):

    decode1: preprocess(chain_slot_a token, y_sel raw, mtp.norm) -> fc -> layer -> head -> D1 select
    decode2: preprocess(chain_slot_b token, x1 raw,   mtp.norm) -> fc -> layer -> head -> D2 select
    [ the M=4 verify runs here: m4.run() -- untouched ]
    prep   : extend base = C - L, verify_input_ids[0] <- pending, stop scan
    extend : preprocess(emitted_ids, x4 target raw, model.norm) -> fc(T=4) -> layer(T=4)
             -> row-select y_sel = x4[L-1] -> head -> EXT select (next round's d1)

Kernel families: every launch in the round is MTP/M4-family. The T=1 draft layer
runs the MTP-family clone kernels (kernels/mtp/nex_n2_5_pro_mtp_{qkv_proj,norm_rope,
attn_core,o_proj,gate_gemv,l1,l2_reduce,lmhead_gemv}_ctypes.cu -- same
device logic as the 1-batch sources, compiled under MTP-specific global/launch
symbols) on the MTP block's own weights, KV cache, and reduce workspaces:
    fc          = fp8-quantized mtp.fc through the MTP o_proj reduce (T=1) and the
                  dedicated FP32 FC reduction (T=4); gate input pinned at +30 => sigmoid==1 in bf16
    layer T=1   = MTP qkv/norm-rope/wgmma-core/o-proj + MTP gate/top-k/L1/fused-L2
                  (slot = control-block chain_slot views)
    layer T=4   = the M4 attention kernel sequence + M4 MoE front, base read from a staged
                  16-word ext_ctl whose word 1 = C - commit_count (written by MtpPrep)
    draft head  = the MTP lm_head GEMV into MTP-owned logits plus the MTP argmax
                  producer/exchange pairs on W_fused_mtp = lm_head x (1 + mtp.norm),
                  consuming RAW hiddens

Hand-off contract (block.seed): mtp KV lines [0..p-1], chain_slot_a = [d1, p] (seeded by
init_control_block from verify_input_ids[1]), and y_sel = the draft-prefill's last-row
RAW (pre-mtp.norm) hidden.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent

from nex_n2_5_pro import m4_control as ctlabi  # noqa: E402
from nex_n2_5_pro.attn import EPS, FullAttn, HID  # noqa: E402
from nex_n2_5_pro.config import LAYERS_DIR as SNAP  # noqa: E402
from nex_n2_5_pro.kernel_build import build_kernel  # noqa: E402
from nex_n2_5_pro.ctl_mirror import CtlMirror  # noqa: E402
from nex_n2_5_pro.moe import Moe, TOPK, _poff  # noqa: E402
from nex_n2_5_pro.mtp import weights as mtpw  # noqa: E402

KDIR = HERE.parent / "kernels" / "mtp"
FC_COLS = 1024


def _p(t):
    return ctypes.c_void_p(t.data_ptr())


def _sp():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def _eos_ids() -> list[int]:
    cfg = json.loads((SNAP / "generation_config.json").read_text())
    eos = cfg.get("eos_token_id", [])
    return [eos] if isinstance(eos, int) else list(eos)


class MtpDraft:
    def __init__(
        self,
        eng,
        m4,
        dev,
        eos_ids=None,
        *,
        sampling=False,
        temperature=0.6,
        top_p=0.95,
        share_weights_from=None,
    ):
        """share_weights_from: another MtpDraft whose draft-block weights, KV
        cache, and argmax banks this instance ALIASES (the non-CP and CP
        verification arms hold logically identical draft state, so a server
        building both arms keeps ONE copy). Only the small per-arm activation
        buffers and this arm's control-block views are private."""
        self.eng, self.m4, self.dev = eng, m4, dev
        rank, world = eng.rank, eng.world
        self.rank, self.world = rank, world
        self.sampling = bool(sampling)
        if self.sampling:
            # Initialize the M4-owned seed field from the engine request seed.
            # Later requests update that field through set_sampling_config().
            # Configuration lives in device buffers, so later request updates do
            # not invalidate a captured round graph.
            m4.enable_sampling(eng.head.seed_dev, temperature=temperature, top_p=top_p)

        import torch.distributed as dist

        if share_weights_from is not None:
            src = share_weights_from
            self.attn, self.moe = src.attn, src.moe
            self.fc_w, self.fc_scales = src.fc_w, src.fc_scales
            self.W_fused, self.norms = src.W_fused, src.norms
            self.pre, self.glue = src.pre, src.glue
            self.amd, self.ame = src.amd, src.ame
            self.am_part, self.am_bank = src.am_part, src.am_bank
            self.qkvl, self.nrl, self.wgl, self.opl = src.qkvl, src.nrl, src.wgl, src.opl
            self.fc4l = src.fc4l
            self.fc4_rep, self.fc4_rep_mc = src.fc4_rep, src.fc4_rep_mc
            self.fc4_bar, self.fc4_bar_mc = src.fc4_bar, src.fc4_bar_mc
            self.fc4_it = src.fc4_it
            self.gl, self.l1l, self.fll, self.lml = src.gl, src.l1l, src.fll, src.lml
            self.logits, self.inv_rms = src.logits, src.inv_rms
        else:
            # The draft blocks own their FULL kernel-family state: weights (ext
            # aliasing), draft KV, and their own NVLS reduce workspaces
            # (share_from=None) -- no 1-batch workspace is touched by the round.
            self.attn = FullAttn(
                "mtp",
                rank,
                world,
                dev,
                keep_ref=False,
                share_from=None,
                ext=mtpw.attn_ext(rank, dev),
            )
            self.moe = Moe(
                "mtp",
                rank,
                world,
                dev,
                keep_ref=False,
                share_from=None,
                ext=mtpw.moe_ext(rank, dev),
            )
            self.fc_w, self.fc_scales = mtpw.fc_shard(rank, dev)
            self.W_fused = mtpw.head_w_fused(eng.head.W_raw, dev)
            self.norms = mtpw.norm_weights(dev, eng.head.norm_w)

            vp, ci = ctypes.c_void_p, ctypes.c_int
            cf, cu, cll = ctypes.c_float, ctypes.c_ulonglong, ctypes.c_longlong
            # MTP-family T=1 layer kernels (clone TUs; 1-batch entry points are
            # never launched by the round).
            self.qkvl = ctypes.CDLL(str(build_kernel(KDIR, "nex_n2_5_pro_mtp_qkv_proj_ctypes")))
            self.qkvl.NexN25ProMtpQkvProjDynLaunch.argtypes = [vp] * 9
            self.nrl = ctypes.CDLL(str(build_kernel(KDIR, "nex_n2_5_pro_mtp_norm_rope_ctypes")))
            self.nrl.NexN25ProMtpNormRopeDynLaunch.argtypes = [vp] * 12 + [ci, cf, vp]
            self.wgl = ctypes.CDLL(
                str(
                    build_kernel(
                        KDIR, "nex_n2_5_pro_mtp_attn_core_ctypes", hopper=True, link_cuda=True
                    )
                )
            )
            self.wgl.NexN25ProMtpAttnWgmmaDynLaunch.argtypes = [vp] * 10
            self.wgl.NexN25ProMtpAttnWgmmaDynLaunch.restype = ci
            self.opl = ctypes.CDLL(str(build_kernel(KDIR, "nex_n2_5_pro_mtp_o_proj_ctypes")))
            self.opl.NexN25ProMtpOProjReduceLaunch.argtypes = (
                [vp] * 4 + [cu, vp, cu] + [vp] * 4 + [ci, ci, vp, vp, cll, vp]
            )
            self.fc4l = ctypes.CDLL(str(build_kernel(KDIR, "nex_n2_5_pro_mtp_fc_t4_ctypes")))
            self.fc4l.NexN25ProMtpFcT4Launch.argtypes = (
                [vp] * 4 + [cu, vp, cu] + [vp] * 4 + [ci, ci, vp]
            )
            # FC's FP32 payload/arrival protocol has its own cumulative state.
            # CP and non-CP draft objects share it, independently of the target LL wire.
            import torch.distributed._symmetric_memory as symm_mem

            self.fc4_rep = symm_mem.empty(2 * 8 * 4 * HID, dtype=torch.float32, device=dev)
            self.fc4_bar = symm_mem.empty(128, dtype=torch.int32, device=dev)
            rep_handle = symm_mem.rendezvous(self.fc4_rep, dist.group.WORLD.group_name)
            bar_handle = symm_mem.rendezvous(self.fc4_bar, dist.group.WORLD.group_name)
            self.fc4_rep_mc = int(getattr(rep_handle, "multicast_ptr", 0) or 0)
            self.fc4_bar_mc = int(getattr(bar_handle, "multicast_ptr", 0) or 0)
            if not self.fc4_rep_mc or not self.fc4_bar_mc:
                raise RuntimeError("multicast unavailable for MTP FC")
            self.fc4_rep.zero_()
            self.fc4_bar.zero_()
            self.fc4_it = torch.zeros(128, dtype=torch.int32, device=dev)
            torch.cuda.synchronize()
            dist.barrier()

            self.gl = ctypes.CDLL(str(build_kernel(KDIR, "nex_n2_5_pro_mtp_gate_gemv_ctypes")))
            self.gl.NexN25ProMtpGateGemvLaunch.argtypes = [vp, vp, vp, vp, cf, vp, cll, vp]
            self.gl.NexN25ProMtpGateTopKXnLaunch.argtypes = [vp] * 7
            self.l1l = ctypes.CDLL(str(build_kernel(KDIR, "nex_n2_5_pro_mtp_l1_ctypes")))
            self.l1l.NexN25ProMtpL1W13XnLaunch.argtypes = [vp] * 8
            self.fll = ctypes.CDLL(str(build_kernel(KDIR, "nex_n2_5_pro_mtp_l2_reduce_ctypes")))
            self.fll.NexN25ProMtpFusedL2ReduceResidLaunch.argtypes = (
                [vp] * 7 + [cu, vp, cu] + [vp] * 4 + [ci, ci, vp, vp, cll, vp]
            )
            self.lml = ctypes.CDLL(str(build_kernel(KDIR, "nex_n2_5_pro_mtp_lmhead_gemv_ctypes")))
            self.lml.NexN25ProMtpLmHeadGemvLaunch.argtypes = [vp, vp, vp, vp, cf, vp]
            # MTP-owned draft-head scratch (the engine head's buffers stay 1-batch).
            self.logits = torch.empty(31040, dtype=torch.float32, device=dev)
            self.inv_rms = torch.zeros(1, dtype=torch.float32, device=dev)

            self.pre = ctypes.CDLL(str(build_kernel(KDIR, "nex_n2_5_pro_mtp_preprocess_ctypes")))
            self.pre.NexN25ProMtpPreprocessLaunch.argtypes = [vp] * 7 + [ci, ci, vp]
            self.glue = ctypes.CDLL(str(build_kernel(KDIR, "nex_n2_5_pro_mtp_glue_ctypes")))
            # Draft selection: CUSTOM argmax head tails for the draft (d2/d3) and the
            # extend (d1) selects -- per-site producer+exchange pairs (sentinel-armed
            # NVLS push; every rank computes the same argmax redundantly, zero further
            # communication).  The lm_head GEMV launch is the engine's own; the bs=1
            # top-k/sampling machinery is untouched by the draft head.
            cu64 = ctypes.c_ulonglong
            self.amd = ctypes.CDLL(str(build_kernel(KDIR, "nex_n2_5_pro_mtp_draft_argmax_ctypes")))
            self.amd.NexN25ProMtpDraftArgmaxProducerLaunch.argtypes = [vp, vp, ci, vp]
            self.amd.NexN25ProMtpDraftArgmaxExchangeLaunch.argtypes = [vp, cu64, vp, vp, ci, ci, vp]
            self.ame = ctypes.CDLL(str(build_kernel(KDIR, "nex_n2_5_pro_mtp_ext_argmax_ctypes")))
            self.ame.NexN25ProMtpExtArgmaxProducerLaunch.argtypes = [vp, vp, ci, vp]
            self.ame.NexN25ProMtpExtArgmaxExchangeLaunch.argtypes = [vp, cu64, vp, vp, ci, vp]
            self.am_part = torch.zeros(132, dtype=torch.int64, device=dev)

            def _am_slots():
                x = symm_mem.empty(64, dtype=torch.int64, device=dev)
                x.zero_()  # 0 == sentinel (armed)
                h = symm_mem.rendezvous(x, dist.group.WORLD.group_name)
                mc = int(getattr(h, "multicast_ptr", 0) or 0)
                if mc == 0:
                    raise RuntimeError("argmax head: NVLS multicast unavailable")
                return x, mc

            self.am_bank = {m: _am_slots() for m in (0, 1, 2)}  # per-mode banks
            dist.barrier()  # all banks armed everywhere
            vp, ci = ctypes.c_void_p, ctypes.c_int
            self.glue.NexN25ProMtpPrepLaunch.argtypes = [vp, vp, vp, ci, vp]
            self.glue.NexN25ProMtpRowSelectLaunch.argtypes = [vp, vp, vp, vp]
            self.glue.NexN25ProMtpStatusFoldLaunch.argtypes = [vp, vp, vp, vp, vp]

        bf16 = torch.bfloat16
        self.cat1 = torch.zeros(FC_COLS, dtype=bf16, device=dev)  # T=1 local fc input
        self.cat4 = torch.zeros(4, FC_COLS, dtype=bf16, device=dev)  # T=4 local fc input
        self.x1 = torch.zeros(HID, dtype=bf16, device=dev)  # decode1 stream (-> y1 raw)
        self.x2 = torch.zeros(HID, dtype=bf16, device=dev)  # decode2 stream
        self.y_sel = torch.zeros(HID, dtype=bf16, device=dev)  # extend's selected RAW y
        self.gate_one = torch.full((4, FC_COLS), 30.0, dtype=bf16, device=dev)  # sigmoid==1 (bf16)
        self.ext_ctl = torch.zeros(ctlabi.WORDS, dtype=torch.int32, device=dev)
        eos = sorted(set(int(t) for t in eos_ids)) if eos_ids is not None else _eos_ids()
        self.eos_ids = torch.tensor(eos or [-1], dtype=torch.int32, device=dev)
        self.n_eos = len(eos)
        self.mirror = CtlMirror(m4.ctl, dev)  # host mirror of ctl, published by round()

        ctl = m4.ctl
        self.slot_a = ctl[ctlabi.CHAIN_SLOT_A : ctlabi.CHAIN_SLOT_A + 2]
        self.slot_b = ctl[ctlabi.CHAIN_SLOT_B : ctlabi.CHAIN_SLOT_B + 2]
        self.emitted_view = ctl[ctlabi.EMITTED_IDS : ctlabi.EMITTED_IDS + 4]

    # ---- hand-off / harness seeding -----------------------------------------------------
    def seed(self, mtp_k, mtp_v, length, y_sel_raw):
        """Seed the MTP KV prefix [0..length-1] and the extend-selected RAW hidden.
        mtp_k/mtp_v None => the prefix already lives in the resident cache (the bridge
        streams each prefill chunk's K/V straight into rows [s,e), and rounds only
        write positions >= length, so nothing needs restoring on re-seed).
        chain_slot_a/verify_input_ids are seeded separately via ctlabi.init_control_block."""
        if mtp_k is not None:
            self.attn.k_cache[:length].copy_(mtp_k)
            self.attn.v_cache[:length].copy_(mtp_v)
        self.y_sel.copy_(y_sel_raw)

    # ---- fc: the MTP o_proj kernel on the fp8-quantized mtp.fc --------------------------
    def _fc_t1(self, cat, x_out):
        a, cu, cll = self.attn, ctypes.c_ulonglong, ctypes.c_longlong
        x_out.zero_()  # the reduce folds x_out += fc
        self.opl.NexN25ProMtpOProjReduceLaunch(
            _p(cat),
            _p(self.gate_one),
            _p(self.fc_w),
            _p(self.fc_scales),
            cu(a.rep_mc),
            _p(a.rep),
            cu(a.bar_mc),
            _p(a.bar),
            _p(a.final_out),
            _p(a.status),
            _p(a.it_ctr),
            self.rank,
            self.world,
            _p(x_out),
            None,
            cll(0),
            _sp(),
        )

    # ---- the T=1 draft layer: MTP-family clone kernels on the draft state ---------------
    def _attn_t1(self, x):
        """Device-S T=1 full-attention layer on the draft KV (a.slot is bound to a
        control-block chain-slot view before this call); every launch is MTP-family."""
        a, w = self.attn, self.attn.w
        ci, cf, cu, cll = ctypes.c_int, ctypes.c_float, ctypes.c_ulonglong, ctypes.c_longlong
        self.qkvl.NexN25ProMtpQkvProjDynLaunch(
            _p(x),
            _p(w["qkv_w"]),
            _p(w["qkv_scales"]),
            _p(a.q_raw),
            _p(a.gate),
            _p(a.k_cache),
            _p(a.v_cache),
            _p(a.slot),
            _sp(),
        )
        self.nrl.NexN25ProMtpNormRopeDynLaunch(
            _p(a.q_raw),
            _p(a.k_cache),
            _p(w["qn"]),
            _p(w["kn"]),
            _p(a.cos_tab),
            _p(a.sin_tab),
            _p(a.q_out),
            _p(a.slot),
            _p(x),
            _p(a.v_cache),
            _p(a.gate),
            _p(a.gate),
            ci(HID),
            cf(EPS),
            _sp(),
        )
        rc = self.wgl.NexN25ProMtpAttnWgmmaDynLaunch(
            _p(a.q_out),
            _p(a.k_cache),
            _p(a.v_cache),
            _p(a.slot),
            _p(a.ws_m),
            _p(a.ws_l),
            _p(a.ws_acc),
            _p(a.ctx),
            _p(a.wgmma_grid_counter),
            _sp(),
        )
        if rc != 0:
            raise RuntimeError(f"NexN25ProMtpAttnWgmmaDynLaunch rc={rc}")
        self.opl.NexN25ProMtpOProjReduceLaunch(
            _p(a.ctx),
            _p(a.gate),
            _p(w["o_w"]),
            _p(w["o_scales"]),
            cu(a.rep_mc),
            _p(a.rep),
            cu(a.bar_mc),
            _p(a.bar),
            _p(a.final_out),
            _p(a.status),
            _p(a.it_ctr),
            self.rank,
            self.world,
            _p(x),
            None,
            cll(0),
            _sp(),
        )

    def _moe_t1(self, x):
        """T=1 MoE layer (raw weight-sharing layout) on the draft MoE state; every
        launch is MTP-family. Mirrors the raw-path math exactly."""
        m = self.moe
        cf, cu, cll = ctypes.c_float, ctypes.c_ulonglong, ctypes.c_longlong
        sg = m.k["shared_gate_up"]
        self.gl.NexN25ProMtpGateGemvLaunch(
            _p(x),
            _p(m.k["gate_w"]),
            _p(m.gate_logits),
            _p(m.rms_scale),
            cf(1e-6),
            _p(sg),
            cll(sg.numel() * sg.element_size()),
            _sp(),
        )
        self.gl.NexN25ProMtpGateTopKXnLaunch(
            _p(m.gate_logits), _p(m.rms_scale), _p(x), _p(m.gamma), _p(m.packed), _p(m.xn), _sp()
        )
        self.l1l.NexN25ProMtpL1W13XnLaunch(
            _p(m.packed),
            _p(m.xn),
            _p(m.k["w13"]),
            _p(m.k["w13_scales"]),
            _p(m.k["shared_gate_up"]),
            _p(m.k["shared_scales"]),
            _p(m.raw_h),
            _sp(),
        )
        self.fll.NexN25ProMtpFusedL2ReduceResidLaunch(
            _p(m.packed),
            _poff(m.packed, TOPK * 4),
            _p(m.raw_h),
            _p(m.k["W2"]),
            _p(m.k["w2_scales"]),
            _p(m.k["shared_W2"]),
            _p(m.k["shared_w2_scales"]),
            cu(m.rep_mc),
            _p(m.rep),
            cu(m.bar_mc),
            _p(m.bar),
            _p(m.out),
            _p(m.status),
            _p(m.it_ctr),
            m.rank,
            m.world,
            _p(x),
            None,
            cll(0),
            _sp(),
        )

    def _fc_t4(self):
        m4, cu, ci = self.m4, ctypes.c_ulonglong, ctypes.c_int
        m4.x4.zero_()
        m4._ck(
            self.fc4l.NexN25ProMtpFcT4Launch(
                _p(self.cat4),
                _p(self.gate_one),
                _p(self.fc_w),
                _p(self.fc_scales),
                cu(self.fc4_rep_mc),
                _p(self.fc4_rep),
                cu(self.fc4_bar_mc),
                _p(self.fc4_bar),
                _p(self.fc4_it),
                _p(m4.ws12_status),
                _p(m4.x4),
                ci(self.rank),
                ci(self.world),
                _sp(),
            ),
            "mtp fc t4",
        )

    # ---- the draft head: MTP lm_head GEMV + MTP argmax tails, RAW hidden in -------------
    def _head(self, y_raw, am):
        cu, ci = ctypes.c_ulonglong, ctypes.c_int
        self.lml.NexN25ProMtpLmHeadGemvLaunch(
            _p(y_raw),
            _p(self.W_fused),
            _p(self.logits),
            _p(self.inv_rms),
            ctypes.c_float(EPS),
            _sp(),
        )
        slots, mc = self.am_bank[am]
        if am == 0:
            self.m4._ck(
                self.ame.NexN25ProMtpExtArgmaxProducerLaunch(
                    _p(self.logits), _p(self.am_part), ci(self.rank), _sp()
                ),
                "mtp ext argmax producer",
            )
            self.m4._ck(
                self.ame.NexN25ProMtpExtArgmaxExchangeLaunch(
                    _p(self.am_part), cu(mc), _p(slots), _p(self.m4.ctl), ci(self.rank), _sp()
                ),
                "mtp ext argmax exchange",
            )
            return
        self.m4._ck(
            self.amd.NexN25ProMtpDraftArgmaxProducerLaunch(
                _p(self.logits), _p(self.am_part), ci(self.rank), _sp()
            ),
            "mtp draft argmax producer",
        )
        self.m4._ck(
            self.amd.NexN25ProMtpDraftArgmaxExchangeLaunch(
                _p(self.am_part), cu(mc), _p(slots), _p(self.m4.ctl), ci(self.rank), ci(am), _sp()
            ),
            "mtp draft argmax exchange",
        )

    def _preprocess(self, ids_view, hidden, final_norm_w, cat, rows):
        self.pre.NexN25ProMtpPreprocessLaunch(
            _p(self.eng.embed_w),
            _p(ids_view),
            _p(hidden),
            _p(final_norm_w),
            _p(self.norms["pre_fc_hidden"]),
            _p(self.norms["pre_fc_embed"]),
            _p(cat),
            self.rank,
            rows,
            _sp(),
        )

    # ---- one T=1 draft decode -----------------------------------------------------------
    def _decode(self, slot_view, hidden_in, x_stream, select):
        a = self.attn
        self._preprocess(slot_view, hidden_in, self.norms["mtp_norm"], self.cat1, 1)
        self._fc_t1(self.cat1, x_stream)
        a.slot = slot_view  # baked per launch: node-local pointer
        self._attn_t1(x_stream)
        self._moe_t1(x_stream)
        self._head(x_stream, am=1 if select == 1 else 2)

    def _fold_status(self):
        """OR the draft-side sticky statuses into ws12_status so the verify tail fails
        closed on a draft collective failure too (rank-skew on the draft path must veto
        the commit)."""
        self.glue.NexN25ProMtpStatusFoldLaunch(
            _p(self.attn.status),
            _p(self.moe.status),
            _p(self.eng.head.status),
            _p(self.m4.ws12_status),
            _sp(),
        )

    def draft_phase(self):
        """The 2 chain decodes that complete the window [anchor, d1, d2, d3]."""
        self._decode(self.slot_a, self.y_sel, self.x1, select=1)
        self._decode(self.slot_b, self.x1, self.x2, select=2)
        self._fold_status()

    # ---- the T=4 extend (runs right after m4.run()'s verify/commit) ---------------------
    def _attn_layer_m4_ext(self):
        m4, a, w = self.m4, self.attn, self.attn.w
        ci, cf = ctypes.c_int, ctypes.c_float
        m4._ck(
            m4.qkv4_launch(
                _p(m4.x4),
                _p(m4._qkv4_w(w)),
                _p(w["qkv_scales"]),
                _p(m4.q_raw4),
                _p(m4.gate4),
                _p(a.k_cache),
                _p(a.v_cache),
                _p(self.ext_ctl),
                _sp(),
            ),
            "mtp m4 attn qkv",
        )
        m4._ck(
            m4.nrl4.NexN25ProM4NormRopeLaunch(
                _p(m4.q_raw4),
                _p(a.k_cache),
                _p(w["qn"]),
                _p(w["kn"]),
                _p(a.cos_tab),
                _p(a.sin_tab),
                _p(m4.q_out4),
                _p(self.ext_ctl),
                _p(m4.x4),
                _p(a.v_cache),
                _p(m4.gate4),
                ci(HID),
                cf(EPS),
                _sp(),
            ),
            "mtp m4 norm rope",
        )
        m4._attn_core_m4(a, _p(self.ext_ctl))
        m4._attn_out_fused(a)

    def extend_phase(self):
        m4, ci = self.m4, ctypes.c_int
        self.glue.NexN25ProMtpPrepLaunch(
            _p(m4.ctl), _p(self.ext_ctl), _p(self.eos_ids), ci(self.n_eos), _sp()
        )
        # x4 still holds the verify's 4 RAW target hiddens; consume before overwriting.
        self._preprocess(self.emitted_view, m4.x4, self.norms["model_norm"], self.cat4, 4)
        self._fc_t4()
        self._attn_layer_m4_ext()
        m4._moe_front(self.moe, None, "mtp")
        self.glue.NexN25ProMtpRowSelectLaunch(_p(m4.x4), _p(m4.ctl), _p(self.y_sel), _sp())
        self._head(self.y_sel, am=0)
        self._fold_status()  # extend-side failures veto the NEXT commit

    # ---- one full round -----------------------------------------------------------------
    def round(self, route_sink=None):
        self.draft_phase()
        self.m4.run(route_sink=route_sink)
        self.extend_phase()
        self.mirror.launch()  # publish ctl to the host: the round's last node
