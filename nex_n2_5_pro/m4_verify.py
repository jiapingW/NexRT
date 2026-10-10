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

"""M=4 MTP verification engine: the production four-row verify step (TP=8).

One call to run() executes one captured verification unit over the four target
rows [t0, d1, d2, d3]: four-row embed gather, the 60 mixer+MoE layers against
the shared bs=1 weights and KV/state, the four-row lm_head, and the
verify/commit tail (sampled or greedy) writing the device-resident control
block.  Weights and KV are the bs=1 engine's own buffers -- nothing is
duplicated; the M=4 path owns only its private conv/state rings, workspaces,
and the 32-word control block.

Two arms, separate captured graphs:
  M4VerifyNonCP -- single-graph attention over the resident KV (short context).
  M4VerifyCp    -- context-parallel attention (KV aliased from the bs=1 CP
                   blocks, never duplicated), auto-dispatched at
                   M4_CP_DISPATCH_S tokens.

Greedy and sampled verification share the M=4 producer, reducer, and one-CTA
sampler/commit graph. Request parameters occupy fixed sampling-buffer fields.
"""

import ctypes
from pathlib import Path

import torch
import torch.distributed as dist

from nex_n2_5_pro import m4_control as ctlabi

HERE = Path(__file__).resolve().parent

M = 4  # target rows per verification unit
HID = 4096
ROWS = 31040  # lm_head vocab shard per rank
GATE_ROWS_PHYS = 528
PACKED_WORDS = 64
RAWH_ROWS = 11 * 128  # (routed top-10 + shared) x RINTER
CURSOR_LEN = 6
QKV_ROWS_GDN = 1536
Z_ROWS = 1024
HEADS = 8
QROW = 1024  # attn q / gate row width (4 heads x 256)
NHL = 4
HEAD_DIM = 256
FUSED_CTAS = 128
DIM = 4096
PROD_CTAS = 128
PROD_THREADS = 512
QKV_ROWS = 1536
GDN_HEADS = 8
GDN_DIM = 128

# Production MTP CP crossover: the CP verification graph is dispatched at and
# above this committed length.  Measured for the M=4 round (distinct from the
# bs=1 dispatch threshold, which belongs to the bs=1 graphs).
M4_CP_DISPATCH_S = 229376

# CP LL tail geometry (quad partial + 8-byte LL combine/merge wire).
NHCP_LOCAL = 16
OHEADS = 4
HD = 256
LL_PKT_PER_HR = HD // 4 + 2  # 64 data groups + padded LSE storage
NS_QUAD = 128


def pack_lmhead_bf16tc(Wf):
    """[31040, 4096] bf16 row-major -> the tc16 head kernel's lane-major prepack
    Wpack[tile 16r][k16][lane][frag a0|a1|a2|a3][2 bf16] (same elements, permuted).
    Each lane's uint4 per k16 step IS its four mma.m16n8k16 A fragments (16 REAL vocab
    rows per tile: a1/a3 = rows +8 are real, not padding). Built once at load; row-major
    W_fused stays resident for the bs=1 kernels."""
    dev = Wf.device
    tile = torch.arange(31040 // 16, device=dev).view(-1, 1, 1, 1, 1)
    k16 = torch.arange(4096 // 16, device=dev).view(1, -1, 1, 1, 1)
    lane = torch.arange(32, device=dev).view(1, 1, 32, 1, 1)
    f = torch.arange(4, device=dev).view(1, 1, 1, 4, 1)
    e = torch.arange(2, device=dev).view(1, 1, 1, 1, 2)
    gid, tig = lane >> 2, lane & 3
    row = tile * 16 + (f & 1) * 8 + gid
    k = k16 * 16 + (f >> 1) * 8 + tig * 2 + e
    src = (row * 4096 + k).reshape(-1)
    return Wf.reshape(-1)[src].contiguous()


def pack_gemm_bf16tc(W, K):
    """[rows, K] e4m3 row-major -> the bf16tc kernels' lane-major M4 prepack
    Wpack[tile n8][k32][lane][8 B] (same bytes, permuted). Each lane's 8 bytes are its
    mma.m16n8k16 A-fragment pairs for the two k16 steps of one k32 tile: one coalesced
    256 B warp load per two mmas. Built once per weight at load; the row-major weight
    stays resident for the bs=1 / draft kernels; both layouts remain available."""
    dev = W.device
    rows = W.numel() // K
    t = torch.arange(rows // 8, device=dev).view(-1, 1, 1, 1)
    c = torch.arange(K // 32, device=dev).view(1, -1, 1, 1)
    lane = torch.arange(32, device=dev).view(1, 1, 32, 1)
    b = torch.arange(8, device=dev).view(1, 1, 1, 8)
    gid, tig = lane >> 2, lane & 3
    step, half8, e = b >> 2, (b >> 1) & 1, b & 1
    row = t * 8 + gid
    k = c * 32 + step * 16 + half8 * 8 + tig * 2 + e
    src = (row * K + k).reshape(-1)
    return W.reshape(-1)[src].contiguous()


def pack_qkv_bf16tc(W):
    """[2560, 4096] qkvgate/qkvz specialization of pack_gemm_bf16tc."""
    return pack_gemm_bf16tc(W, 4096)


def pack_gate_tc8(W):
    """[513|528, 4096] bf16 row-major gate_w -> the tc8 gate kernel's lane-major
    prepack Wpack[tile 8r][k16][lane][k-half f][2 bf16] (rows zero-padded to 528;
    otherwise the same bytes, permuted -- pack_lmhead_bf16tc's pattern at 8-row
    tiles). Built once per weight at first eager use; row-major gate_w stays
    resident for the bs=1 kernels; both layouts remain available."""
    assert W.dtype == torch.bfloat16 and W.shape[-1] == 4096 and W.shape[0] <= 528
    dev = W.device
    wp = torch.zeros(528, 4096, dtype=torch.bfloat16, device=dev)
    wp[: W.shape[0]] = W
    tile = torch.arange(528 // 8, device=dev).view(-1, 1, 1, 1, 1)
    k16 = torch.arange(4096 // 16, device=dev).view(1, -1, 1, 1, 1)
    lane = torch.arange(32, device=dev).view(1, 1, 32, 1, 1)
    f = torch.arange(2, device=dev).view(1, 1, 1, 2, 1)
    e = torch.arange(2, device=dev).view(1, 1, 1, 1, 2)
    gid, tig = lane >> 2, lane & 3
    src = ((tile * 8 + gid) * 4096 + k16 * 16 + f * 8 + tig * 2 + e).reshape(-1)
    return wp.reshape(-1)[src].contiguous()


class M4Verify:
    """M=4 four-row MTP verification engine (shared core, TP=8).

    Production kernel selections, hardwired: bf16tc four-row QKV+gate
    projection, tc16 four-row lm_head, tc16 GDN front (ring), hoisted GDN
    recurrent core (ring), tc8 MoE gate GEMM, packed-key raw top-k with the
    fold-weight L1 and the LL-inbox MoE L2, quad attention core, and shared
    Tensor Core attention/GDN output projections with fixed-rank BF16 LL reduction,
    and the merge-fused C=32 x 512 sampled tail (greedy argmax/commit as the
    compatibility tail).
    """

    def __init__(self, eng, dev, *, use_cp_attention=True):
        from nex_n2_5_pro.decode_engine import _p, _sp
        from nex_n2_5_pro.head import EPS
        from nex_n2_5_pro.kernel_build import build_kernel
        from nex_n2_5_pro.moe import _poff

        self.eng, self.dev = eng, dev
        self._p, self._sp, self._eps, self._poff = _p, _sp, EPS, _poff
        self.use_cp_attention = bool(use_cp_attention)
        if self.use_cp_attention:
            missing = sorted(set(eng.attn) - set(eng.attn_cp))
            if missing:
                raise RuntimeError(
                    "the M4 CP arm requires DecodeEngine(cp=True); "
                    f"missing CP blocks for layers {missing}"
                )
        vp, ci, cf, cu = (ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_ulonglong)
        cll = ctypes.c_longlong

        # ---- four-row front + head libraries and row buffers ----
        self.embl4 = ctypes.CDLL(
            str(build_kernel(HERE / "kernels" / "embed", "nex_n2_5_pro_m4_embed_gather_ctypes"))
        )
        self.embl4.NexN25ProM4EmbedGatherLaunch.argtypes = [vp] * 5
        self.headl4 = ctypes.CDLL(
            str(build_kernel(HERE / "kernels" / "head", "nex_n2_5_pro_m4_lmhead_gemm_ctypes"))
        )
        # Transposed-mma verify head, W rows as A in 16-real-row
        # tiles, consuming the lane-major prepack built by _head_w.
        self.headl4.NexN25ProM4LmHeadGemmLaunch.argtypes = [vp, vp, vp, vp, cf, vp]
        self.head4_launch = self.headl4.NexN25ProM4LmHeadGemmLaunch
        # Dict container so multiple arms can SHARE one lazily-built head prepack
        # (the bridge points both arms' containers at one dict).
        self._head_pack = {}

        self.dbuf = torch.zeros(M, dtype=torch.int32, device=dev)
        self.cursor = torch.zeros(CURSOR_LEN, dtype=torch.int32, device=dev)
        self.x4 = torch.empty(M, HID, dtype=torch.bfloat16, device=dev)  # residual rows
        self.logits4 = torch.full((M, ROWS), float("nan"), dtype=torch.float32, device=dev)
        self.inv_rms4 = torch.zeros(M, dtype=torch.float32, device=dev)
        self.gate_logits4 = torch.empty(M, GATE_ROWS_PHYS, dtype=torch.bfloat16, device=dev)
        self.rms4 = torch.zeros(M, dtype=torch.float32, device=dev)
        self.packed4 = torch.zeros(M, PACKED_WORDS, dtype=torch.int32, device=dev)
        self.rawh4 = torch.empty(M, RAWH_ROWS, dtype=torch.float32, device=dev)
        self.S0 = None

        # ---- attention QKV (bf16tc lane-major prepack) + attn-side buffers ----
        self.attnl4 = ctypes.CDLL(
            str(build_kernel(HERE / "kernels" / "full_attn", "nex_n2_5_pro_m4_attn_qkv_ctypes"))
        )
        self.attnl4.NexN25ProM4AttnQkvLaunch.argtypes = [vp] * 9
        self.qkv4_launch = self.attnl4.NexN25ProM4AttnQkvLaunch
        # bf16tc consumes the lane-major prepack instead of row-major qkv_w; packs are
        # built lazily on the first (eager warmup) call per weight and cached, so graph
        # capture never allocates. Row-major stays resident for bs=1/draft kernels.
        self._bf16tc_packs = {}

        self.qkv4 = torch.empty(M, QKV_ROWS_GDN, dtype=torch.bfloat16, device=dev)
        self.z4 = torch.empty(M, Z_ROWS, dtype=torch.bfloat16, device=dev)
        self.beta4 = torch.zeros(M, HEADS, dtype=torch.float32, device=dev)
        self.g4 = torch.zeros(M, HEADS, dtype=torch.float32, device=dev)
        self.q_raw4 = torch.empty(M, QROW, dtype=torch.bfloat16, device=dev)
        self.gate4 = torch.empty(M, QROW, dtype=torch.bfloat16, device=dev)
        self.rec4 = torch.empty(
            M, 1024, dtype=torch.bfloat16, device=dev
        )  # retained recurrent rows
        self.ctx4 = torch.empty(M, 1024, dtype=torch.bfloat16, device=dev)  # retained context rows
        # Sticky status of the M=4 attention-side reduce family (cleared per request).
        self.ws_status = torch.zeros(1, dtype=torch.int32, device=dev)
        torch.cuda.synchronize()
        dist.barrier()

        # ---- M=4-only MoE LL-inbox control ----
        import torch.distributed._symmetric_memory as symm_mem

        self.ws_moe_cur_gen = torch.zeros(1, dtype=torch.int32, device=dev)
        self.ws_moe_status = torch.zeros(1, dtype=torch.int32, device=dev)
        torch.cuda.synchronize()
        dist.barrier()

        # ---- M=4 norm/RoPE + quad attention core ----
        self.nrl4 = ctypes.CDLL(
            str(build_kernel(HERE / "kernels" / "full_attn", "nex_n2_5_pro_m4_norm_rope_ctypes"))
        )
        self.nrl4.NexN25ProM4NormRopeLaunch.argtypes = [vp] * 11 + [ci, cf, vp]
        self.q_out4 = torch.empty(M, QROW, dtype=torch.bfloat16, device=dev)

        self.quadl = ctypes.CDLL(
            str(
                build_kernel(
                    HERE / "kernels" / "full_attn",
                    "nex_n2_5_pro_m4_attn_core_quad_ctypes",
                    hopper=True,
                    link_cuda=True,
                )
            )
        )
        # Quad core: one KV pass for all 16 q rows (declared tile-boundary reorder
        # class).  The Xeff epilogue additionally emits the hoisted x_eff4 the
        # tc16hx output projection consumes.
        self.quadl.NexN25ProM4AttnCoreQuadDynXeffLaunch.argtypes = (
            [vp] * 4 + [ci, cf] + [vp] * 6 + [vp]
        )
        nctas = next(iter(eng.attn.values())).nctas if eng.attn else 128
        self._core_nctas = nctas
        # M=4-owned row-sliced flash workspaces (the bs=1 engine's ws_m/ws_l/ws_acc
        # stay untouched).
        self.ws_attn_m4 = torch.empty(M * nctas * NHL, dtype=torch.float32, device=dev)
        self.ws_attn_l4 = torch.empty(M * nctas * NHL, dtype=torch.float32, device=dev)
        self.ws_attn_acc4 = torch.empty(
            M * nctas * NHL * HEAD_DIM, dtype=torch.bfloat16, device=dev
        )

        # ---- raw MoE front: packed-key top-k (+xn) and the fold-weight L1 ----
        self.topkraw4 = ctypes.CDLL(
            str(build_kernel(HERE / "kernels" / "moe", "nex_n2_5_pro_m4_gate_topk_ctypes"))
        )
        self.topkraw4.NexN25ProM4GateTopKLaunch.argtypes = [vp] * 7
        self.xn4 = torch.empty(M, HID, dtype=torch.bfloat16, device=dev)

        # ---- fused output-projection wire (shared by the attn and GDN mixers) ----
        self.aopf4 = ctypes.CDLL(
            str(
                build_kernel(
                    HERE / "kernels" / "full_attn", "nex_n2_5_pro_m4_attn_oproj_fused_ctypes"
                )
            )
        )
        self.gopf4 = ctypes.CDLL(
            str(
                build_kernel(
                    HERE / "kernels" / "linear_attn", "nex_n2_5_pro_m4_gdn_oproj_fused_ctypes"
                )
            )
        )
        self.gopf4.NexN25ProM4GdnOProjLaunch.argtypes = [vp] * 5 + [cu] + [vp] * 4 + [ci, ci, vp]
        self.oproj_xeff4 = torch.empty(4 * 1024, dtype=torch.bfloat16, device=dev)
        self.aopf4.NexN25ProM4AttnOProjLaunch.argtypes = [vp] * 3 + [cu] + [vp] * 4 + [ci, ci, vp]
        # CP and non-CP use the same packed Tensor Core projections and LL wire.
        self._oproj_packs = {}

        # 1 MiB LL wire: [2 parities][8 ranks][128 CTAs][32 slots][2 draft pairs]
        # of BF16x2/epoch packets. Both mixers advance the same per-CTA epochs.
        self.ws12_rep = symm_mem.empty(2 * 8 * M * HID, dtype=torch.float32, device=dev)
        h = symm_mem.rendezvous(self.ws12_rep, dist.group.WORLD.group_name)
        self.ws12_rep_mc = int(getattr(h, "multicast_ptr", 0) or 0)
        if self.ws12_rep_mc == 0:
            raise RuntimeError("multicast unavailable for the fused o-proj workspace")
        self.ws12_rep.zero_()
        self.ws12_it = torch.zeros(FUSED_CTAS, dtype=torch.int32, device=dev)
        self.ws12_status = torch.zeros(1, dtype=torch.int32, device=dev)
        torch.cuda.synchronize()
        dist.barrier()

        # ---- tensor-core gate GEMM (lane-major prepack, rms bit-identical) ----
        gate_library = ctypes.CDLL(
            str(build_kernel(HERE / "kernels" / "moe", "nex_n2_5_pro_m4_gate_gemm_ctypes"))
        )
        gate_library.NexN25ProM4GateGemmPdlLaunch.argtypes = [vp, vp, vp, vp, cf, vp, cll, vp]
        self._gate_pf_launch = gate_library.NexN25ProM4GateGemmPdlLaunch
        self._gate_library = gate_library
        self._gate_packs = {}

        # ---- MoE front prefetch wiring (read-only next-mixer-front warm) ----
        # The engine wires a per-graph (ptr, bytes) prefetch range on every MoE
        # block (front_pf_noncp / front_pf_cp); each M=4 graph bakes the tuple
        # matching its attention arm.  Fail closed on the frozen production
        # wiring: the range must point at the next layer's mixer-front weight.
        self._front_pf = {}
        for index, layer in enumerate(eng.layers[:-1]):
            next_layer = eng.layers[index + 1]
            moe = eng.moe[layer]
            if eng.is_lin[next_layer]:
                target = eng.gdn[next_layer].w["qkvz_w"]
                pf = getattr(moe, "front_pf_noncp", (0, 0))
            elif self.use_cp_attention:
                target = eng.attn_cp[next_layer].w["qgate_w"]
                pf = getattr(moe, "front_pf_cp", (0, 0))
            else:
                target = eng.attn[next_layer].w["qkv_w"]
                pf = getattr(moe, "front_pf_noncp", (0, 0))
            pf_ptr = int(pf[0] or 0)
            pf_bytes = int(pf[1] or 0)
            if pf_ptr != target.data_ptr():
                raise RuntimeError(
                    f"M4 front-pf wiring mismatch at L{layer}->L{next_layer}: "
                    f"{pf_ptr:#x} != consumed target {target.data_ptr():#x}"
                )
            target_bytes = target.numel() * target.element_size()
            if (
                pf_bytes <= 0
                or pf_bytes > target_bytes
                or (pf_bytes % 128) != 0
                or (pf_bytes >> 7) > PROD_CTAS * PROD_THREADS
            ):
                raise RuntimeError(
                    f"M4 invalid front-pf range at L{layer}->L{next_layer}: "
                    f"{pf_bytes} bytes for {target_bytes}-byte target"
                )
            self._front_pf[id(moe)] = {
                "ptr": pf_ptr,
                "bytes": pf_bytes,
                "layer": layer,
                "next_layer": next_layer,
                "target": target,
                "kind": "gdn" if eng.is_lin[next_layer] else "attn",
            }

        # ---- MoE L2: folded-input GEMV + LL-inbox TP reduction ----
        # ONE launch per MoE. Every producer CTA publishes tear-free 8-byte
        # [BF16x2|epoch] packets directly to each rank; CTAs 0..7 poll and fold.
        self.ws21_status = torch.zeros(1, dtype=torch.int32, device=dev)
        packet_count = 2 * 8 * (M * DIM // 2)
        self.ws42_inbox = symm_mem.empty(packet_count * 2, dtype=torch.int32, device=dev)
        h42 = symm_mem.rendezvous(self.ws42_inbox, dist.group.WORLD.group_name)
        self.ws42_inbox.zero_()
        self.ws42_inbox_ptrs = torch.tensor(list(h42.buffer_ptrs), dtype=torch.int64, device=dev)
        self.ws42_ll_ctr = torch.zeros(128, dtype=torch.int32, device=dev)
        self.moe_l2 = ctypes.CDLL(
            str(build_kernel(HERE / "kernels" / "moe", "nex_n2_5_pro_m4_moe_l2_ctypes"))
        )
        self.moe_l2.NexN25ProM4MoeL2Launch.argtypes = [vp] * 9 + [ci, ci, vp, cll] + [vp] * 3
        self.l1fold42 = ctypes.CDLL(
            str(build_kernel(HERE / "kernels" / "moe", "nex_n2_5_pro_m4_l1_foldw_ctypes"))
        )
        self.l1fold42.NexN25ProM4L1FoldWLaunch.argtypes = [vp] * 8
        torch.cuda.synchronize()
        dist.barrier()

        # ---- GDN ring front (tc16) + hoisted recurrent core ----
        self.frontring = ctypes.CDLL(
            str(
                build_kernel(
                    HERE / "kernels" / "linear_attn", "nex_n2_5_pro_m4_gdn_front_ring_ctypes"
                )
            )
        )
        self.frontring.NexN25ProM4LinearFrontLaunch.argtypes = [vp] * 14
        # tc16: transposed-mma front on the lane-major qkvz prepack (pack_qkv_bf16tc
        # reused -- same [2560,4096] fp8 shape as the attn qkvgate). Packs built lazily
        # per weight during eager warmups; row-major qkvz_w stays resident for
        # bs=1/draft.
        self._gdnfront_packs = {}
        self.gdncore2 = ctypes.CDLL(
            str(build_kernel(HERE / "kernels" / "linear_attn", "nex_n2_5_pro_m4_gdn_core_ctypes"))
        )
        # Hoisted recurrent core: all four tokens' norms computed in one parallel
        # entry phase (zero loop syncs), bit-exact vs the sequential ring recurrence.
        self.gdncore2.NexN25ProM4GdnCoreLaunch.argtypes = [vp] * 7

        # ---- the one M=4 control block; verify_input_ids is the execution source of truth ----
        self.ctl = ctlabi.new_control_block(dev)
        self.verify_ids = self.ctl[ctlabi.VERIFY_INPUT_IDS : ctlabi.VERIFY_INPUT_IDS + M]

        # ---- per-GDN-layer M=4-only rings (bs=1 conv_state4 / state_t are untouched) ----
        self.conv_ring, self.state_ring = {}, {}
        for layer, gdn in eng.gdn.items():
            self.conv_ring[layer] = torch.zeros(
                QKV_ROWS, ctlabi.CONV_RING_SLOTS, dtype=torch.bfloat16, device=dev
            )
            self.state_ring[layer] = torch.zeros(
                ctlabi.STATE_RING_SLOTS,
                GDN_HEADS,
                GDN_DIM,
                GDN_DIM,
                dtype=torch.float32,
                device=dev,
            )
        self._ring_of = {id(gdn): layer for layer, gdn in eng.gdn.items()}

        # cursor[5] is the embed gather's invalid-token status. A VIEW, not a
        # byte offset: moe._poff() adds bytes and would misalign the int32 load.
        self.cursor_status = self.cursor[5:6]
        # Compatibility mirror of [token_id, S] into DecodeEngine.slot, for harnesses
        # that inspect the engine slot. Never read back by any M=4 kernel.
        self.mirror_bs1_slot = True
        torch.cuda.synchronize()
        dist.barrier()

        self._init_sampling()

        # ---- CP attention arm wiring (buffers shared with the bs=1 CP blocks) ----
        self.inject_cp_status_rank = -1
        self.cp_slots4 = torch.zeros(M, 2, dtype=torch.int32, device=dev)
        self._cp_by_attn = {}
        self.cp_status_pointers = None
        self.cp_q_group4 = None
        self.cp_q_out4 = None
        self.cpc = None

        if not self.use_cp_attention:
            return
        self._cp_by_attn = {id(eng.attn[layer]): eng.attn_cp[layer] for layer in eng.attn}

        self.cpc = ctypes.CDLL(
            str(build_kernel(HERE / "kernels" / "full_attn", "nex_n2_5_pro_m4_cp_control_ctypes"))
        )
        self.cpc.NexN25ProM4CpStageSlotsLaunch.argtypes = [vp, vp, vp]
        self.cpc.NexN25ProM4CpQGate4Launch.argtypes = [vp] * 9
        self.cpc.NexN25ProM4CpNormRope4Launch.argtypes = [vp] * 11 + [ci, ctypes.c_float, vp]
        self.cpc.NexN25ProM4CpFoldStatusLaunch.argtypes = [vp, ci, vp, ci, ci, vp]

        # Four-row CP front buffers (row-private q group / normed q).
        from nex_n2_5_pro.attn_cp import NH_CP

        self.cp_q_group4 = torch.empty(M, NH_CP * 256, dtype=torch.bfloat16, device=dev)
        self.cp_q_out4 = torch.empty_like(self.cp_q_group4)
        torch.cuda.synchronize()
        dist.barrier()

        # cp_status is shared across CP blocks while tp_status belongs to the
        # corresponding unchanged full-attention block.  Deduplicate addresses
        # to make the table describe the actual local status allocations.
        addresses = []
        for layer in eng.attn:
            block = eng.attn_cp[layer]
            addresses.extend((block.cp_status.data_ptr(), block.tp_status.data_ptr()))
        unique_addresses = list(dict.fromkeys(addresses))
        self.cp_status_pointers = torch.tensor(unique_addresses, dtype=torch.int64, device=dev)

    # ---- M=4-specific prefill load: seed the rings and the control block ----

    def load(self, state_dir, draft_ids):
        """Base bs=1 prefill load, then seed the M=4 rings and control block.

        Clears only the M=4 sticky statuses.  The reduce-ring protocol state
        (bars, generations, per-CTA epochs) is cumulative and is never reset:
        resetting it would desynchronize generations across ranks and graph
        replays.  The ring values come from the engine buffers the unchanged
        bs=1 loader just filled, so the M=4 path provably starts from
        bit-identical history.
        """
        Lp, _first = self.eng.load_prefill(state_dir=Path(state_dir))
        assert len(draft_ids) == M
        self.dbuf.copy_(torch.tensor([int(t) for t in draft_ids], dtype=torch.int32))
        self.cursor.zero_()
        self.logits4.fill_(float("nan"))
        self.inv_rms4.zero_()
        self.S0 = Lp
        self.ws_status.zero_()
        self.ws_moe_status.zero_()
        self.ws12_status.zero_()
        self.ws21_status.zero_()
        length = Lp
        ctlabi.init_control_block(self.ctl, length, draft_ids)
        for layer, gdn in self.eng.gdn.items():
            ring = self.conv_ring[layer]
            ring.zero_()
            # conv_state4[:, 0:3] holds positions L-3, L-2, L-1 in order (oldest first).
            for k in range(3):
                ring[:, ctlabi.conv_slot(length - 3 + k)].copy_(gdn.conv_state4[:, k])
            state = self.state_ring[layer]
            state.zero_()
            state[ctlabi.state_slot(length - 1)].copy_(gdn.state_t)
        self.sampling_status.zero_()
        self.cp_slots4.zero_()
        self.inject_cp_status_rank = -1
        return length

    @staticmethod
    def _ck(rc, what):
        """Launch-status guard: a ctypes launcher returning nonzero means NOTHING was
        launched. Host int check only, no device synchronization."""
        if rc != 0:
            raise RuntimeError(f"{what} launch failed with CUDA error {rc}")

    # ---- MoE: raw front to the L1 boundary, then the LL-inbox L2 ----

    def moe_front_l1_boundary(self, m):
        """The M=4 MoE front to the packed/rms/rawh boundary: tc8 gate GEMM (warming
        the shared-expert payload from its drained CTAs), packed-key raw top-k with
        xn materialization, and the fold-weight L1 (route weights folded into rawh,
        as the L2 requires). Requires the RAW weight mode."""
        if not getattr(m, "raw", False):
            raise RuntimeError(
                "the M4 MoE front requires the RAW weight mode "
                "(NEXRT_MOE_RAW=1 or the sglang bridge)"
            )
        p, sp, cf = self._p, self._sp, ctypes.c_float
        sg = m.k["shared_gate_up"]  # routing- and row-independent
        self._ck(
            self._gate_pf_launch(
                p(self.x4),
                p(self._gate_w4(m)),
                p(self.gate_logits4),
                p(self.rms4),
                cf(1e-6),
                p(sg),
                ctypes.c_longlong(sg.numel() * sg.element_size()),
                sp(),
            ),
            "m4 gate gemm",
        )
        self._ck(
            self.topkraw4.NexN25ProM4GateTopKLaunch(
                p(self.gate_logits4),
                p(self.rms4),
                p(self.x4),
                p(m.gamma),
                p(self.packed4),
                p(self.xn4),
                sp(),
            ),
            "m4 gate topk raw",
        )
        self._ck(
            self.l1fold42.NexN25ProM4L1FoldWLaunch(
                p(self.packed4),
                p(self.xn4),
                p(m.k["w13"]),
                p(m.k["w13_scales"]),
                p(m.k["shared_gate_up"]),
                p(m.k["shared_scales"]),
                p(self.rawh4),
                sp(),
            ),
            "m4 l1 foldw",
        )

    def _moe_front(self, moe, route_sink, layer):
        """One MoE block: folded-input L1 followed by one LL-inbox L2 launch."""
        prefetch = self._front_pf.get(id(moe))
        prefetch_pointer = 0 if prefetch is None else prefetch["ptr"]
        prefetch_bytes = 0 if prefetch is None else prefetch["bytes"]

        pointer = self._p
        stream = self._sp
        self.moe_front_l1_boundary(moe)
        self._ck(
            self.moe_l2.NexN25ProM4MoeL2Launch(
                pointer(self.packed4),
                pointer(self.rawh4),
                pointer(moe.k["W2"]),
                pointer(moe.k["w2_scales"]),
                pointer(moe.k["shared_W2"]),
                pointer(moe.k["shared_w2_scales"]),
                pointer(self.ws_moe_cur_gen),
                pointer(self.ws42_inbox_ptrs),
                pointer(self.ws42_ll_ctr),
                ctypes.c_int(moe.rank),
                ctypes.c_int(moe.world),
                ctypes.c_void_p(prefetch_pointer),
                ctypes.c_longlong(prefetch_bytes),
                pointer(self.ws21_status),
                pointer(self.x4),
                stream(),
            ),
            "m4 moe l2 ll-inbox",
        )
        if route_sink is not None:  # TEST-ONLY (eager verification)
            route_sink.append((layer, self.packed4.clone(), self.rms4.clone()))

    # ---- the M=4 step: four ordered target rows, then one verify/commit ----

    def _run_chain(self, draft_dev=None, route_sink=None):
        """One M=4 verification unit: four ordered target rows, then ONE decision.

        Fully device-driven. Nothing writes DecodeEngine.slot or any host-visible
        state inside this call, and committed_len advances only through the
        verify/commit CTA.
        """
        eng, p, sp = self.eng, self._p, self._sp
        if draft_dev is not None:
            self.dbuf.copy_(draft_dev)
            self.verify_ids.copy_(draft_dev)
        self._ck(
            self.embl4.NexN25ProM4EmbedGatherLaunch(
                p(eng.embed_w), p(self.verify_ids), p(self.x4), p(self.cursor), sp()
            ),
            "m4 embed gather",
        )
        for layer in eng.layers:
            if eng.is_lin[layer]:
                self._gdn_layer_m4(eng.gdn[layer])
            else:
                self._attn_layer_m4(eng.attn[layer])
            self._moe_front(eng.moe[layer], route_sink, layer)
        self._ck(
            self.head4_launch(
                p(self.x4),
                p(self._head_w(eng.head)),
                p(self.logits4),
                p(self.inv_rms4),
                ctypes.c_float(self._eps),
                sp(),
            ),
            "m4 lm head",
        )
        self._verify_and_commit()

    def run(self, draft_dev=None, route_sink=None):
        if not self.use_cp_attention:
            return self._run_chain(draft_dev=draft_dev, route_sink=route_sink)
        if draft_dev is not None:
            self.dbuf.copy_(draft_dev)
            self.verify_ids.copy_(draft_dev)
        self._stage_cp_slots()
        return self._run_chain(draft_dev=None, route_sink=route_sink)

    def _head_w(self, head):
        """W for the M=4 verify head: the tc16 lane-major prepack (built once,
        lazily, during eager warmups -- never under graph capture)."""
        pk = self._head_pack.get("wf")
        if pk is None:
            pk = pack_lmhead_bf16tc(head.W_fused)
            self._head_pack["wf"] = pk
        return pk

    def outputs(self):
        """Post-run snapshot plus the M=4 handback into the bs=1-shaped debug views.

        Caller synchronizes first; host reads happen here, never in run().  The
        handback gathers the LOGICAL state selected by committed_len out of the
        rings and republishes it as conv_state4[:, :3] / state_t, which is what
        the external reference harness reads.  It runs only here -- outside every
        timed region.
        """
        eng, Lp = self.eng, self.S0
        out = {
            "logits4_true": self.logits4,
            "inv_rms4": self.inv_rms4,
            "x4": self.x4,
            "cursor": self.cursor,
            "slot": eng.slot,
            "kv_rows": {
                L: (eng.attn[L].k_cache[Lp : Lp + M], eng.attn[L].v_cache[Lp : Lp + M])
                for L in eng.attn
            },
            "gdn_state": {L: eng.gdn[L].state_t for L in eng.gdn},
            "conv_state": {L: eng.gdn[L].conv_state4 for L in eng.gdn},
            "status": eng.status_max(),
        }
        out["eng_status"] = out["status"]
        out["m4_reducer_status"] = int(self.ws_status.item())
        out["m4_moe_reducer_status"] = int(self.ws_moe_status.item())
        out["m4_fused_oproj_status"] = int(self.ws12_status.item())
        out["m4_moe_l2_status"] = int(self.ws21_status.item())
        out["status"] = max(
            out["eng_status"],
            out["m4_reducer_status"],
            out["m4_moe_reducer_status"],
            out["m4_fused_oproj_status"],
            out["m4_moe_l2_status"],
        )
        control = ctlabi.read_control_block(self.ctl)
        out["m4_control"] = control
        out["m4_verify_status"] = control["status"]
        out["status"] = max(out["status"], control["status"])
        self.handback_bs1_views(control["committed_len"])
        return out

    # ---- GDN layer: ring front (tc16) + hoisted ring recurrence + fused output ----

    def _gdn_layer_m4(self, g):
        p, sp = self._p, self._sp
        w = g.w
        layer = self._ring_of[id(g)]
        key = w["qkvz_w"].data_ptr()
        pk = self._gdnfront_packs.get(key)
        if pk is None:
            pk = pack_qkv_bf16tc(w["qkvz_w"])
            self._gdnfront_packs[key] = pk
        self._ck(
            self.frontring.NexN25ProM4LinearFrontLaunch(
                p(self.x4),
                p(pk),
                p(w["qkvz_scales"]),
                p(w["b_w"]),
                p(w["a_w"]),
                p(w["a_log_dt"]),
                p(w["conv_w"]),
                p(self.conv_ring[layer]),
                p(self.ctl),
                p(self.qkv4),
                p(self.z4),
                p(self.beta4),
                p(self.g4),
                sp(),
            ),
            "m4 gdn front ring tc16",
        )
        self._ck(
            self.gdncore2.NexN25ProM4GdnCoreLaunch(
                p(self.qkv4),
                p(self.beta4),
                p(self.g4),
                p(self.state_ring[layer]),
                p(self.ctl),
                p(self.rec4),
                sp(),
            ),
            "m4 gdn recurrent ring",
        )
        self._gdn_out_fused(g)

    # ---- attention layer (non-CP): ctl-driven QKV + norm/RoPE + quad core ----

    def _qkv4_w(self, w):
        """Weight for the M=4 qkvgate launch: the bf16tc lane-major prepack
        (built once per weight, cached by data_ptr)."""
        key = w["qkv_w"].data_ptr()
        pk = self._bf16tc_packs.get(key)
        if pk is None:
            pk = pack_qkv_bf16tc(w["qkv_w"])
            self._bf16tc_packs[key] = pk
        return pk

    def _attn_layer_m4(self, a):
        p, sp = self._p, self._sp
        from nex_n2_5_pro.attn import EPS, HID as AHID

        w = a.w
        ci, cf = ctypes.c_int, ctypes.c_float
        # No slot staging: the M=4 attention kernels read the unit base from word 1
        # of the control block, which is committed_len. Row j writes K/V at base + j.
        self._ck(
            self.qkv4_launch(
                p(self.x4),
                p(self._qkv4_w(w)),
                p(w["qkv_scales"]),
                p(self.q_raw4),
                p(self.gate4),
                p(a.k_cache),
                p(a.v_cache),
                p(self.ctl),
                sp(),
            ),
            "m4 attn qkv",
        )
        self._ck(
            self.nrl4.NexN25ProM4NormRopeLaunch(
                p(self.q_raw4),
                p(a.k_cache),
                p(w["qn"]),
                p(w["kn"]),
                p(a.cos_tab),
                p(a.sin_tab),
                p(self.q_out4),
                p(self.ctl),
                p(self.x4),
                p(a.v_cache),
                p(self.gate4),
                ci(AHID),
                cf(EPS),
                sp(),
            ),
            "m4 attn norm rope",
        )
        self._attn_core_m4(a, p(self.ctl))
        self._attn_out_fused(a)

    def _oproj_w(self, w):
        """Return the lane-major weight pack shared by the M=4 O-projections."""
        key = w["o_w"].data_ptr()
        pk = self._oproj_packs.get(key)
        if pk is None:
            pk = pack_gemm_bf16tc(w["o_w"], 1024)
            self._oproj_packs[key] = pk
        return pk

    def _attn_core_m4(self, a, slot_p):
        """Run quad attention and produce BF16 gated input for the O-projection.

        Used by both target verification and the MTP four-row extension.
        """
        p, sp, ci, cf = self._p, self._sp, ctypes.c_int, ctypes.c_float
        from nex_n2_5_pro.attn import SCALING

        self._ck(
            self.quadl.NexN25ProM4AttnCoreQuadDynXeffLaunch(
                p(self.q_out4),
                p(a.k_cache),
                p(a.v_cache),
                slot_p,
                ci(self._core_nctas),
                cf(SCALING),
                p(self.ws_attn_m4),
                p(self.ws_attn_l4),
                p(self.ws_attn_acc4),
                p(self.ctx4),
                p(self.gate4),
                p(self.oproj_xeff4),
                sp(),
            ),
            "m4 attn core quad xeff",
        )

    def _oproj_xeff_produce(self, w, rank, world, tag):
        """Project the precomputed BF16 gated context and reduce over TP ranks."""
        p, sp, cu, ci = self._p, self._sp, ctypes.c_ulonglong, ctypes.c_int
        self._ck(
            self.aopf4.NexN25ProM4AttnOProjLaunch(
                p(self.oproj_xeff4),
                p(self._oproj_w(w)),
                p(w["o_scales"]),
                cu(self.ws12_rep_mc),
                p(self.ws12_rep),
                p(self.ws12_it),
                p(self.ws12_status),
                p(self.x4),
                ci(rank),
                ci(world),
                sp(),
            ),
            f"{tag} oproj",
        )

    def _gdn_out_fused(self, g):
        p, sp, cu, ci = self._p, self._sp, ctypes.c_ulonglong, ctypes.c_int
        w = g.w
        self._ck(
            self.gopf4.NexN25ProM4GdnOProjLaunch(
                p(self.rec4),
                p(self.z4),
                p(w["norm_w"]),
                p(self._oproj_w(w)),
                p(w["o_scales"]),
                cu(self.ws12_rep_mc),
                p(self.ws12_rep),
                p(self.ws12_it),
                p(self.ws12_status),
                p(self.x4),
                ci(g.rank),
                ci(g.world),
                sp(),
            ),
            "m4 gdn oproj fused",
        )

    def _attn_out_fused(self, a):
        self._oproj_xeff_produce(a.w, a.rank, a.world, "attn")

    def _gate_w4(self, m):
        """Weight for the M=4 gate launch: the tc8 lane-major prepack (built once
        per weight at first eager use, cached by data_ptr)."""
        key = m.k["gate_w"].data_ptr()
        pk = self._gate_packs.get(key)
        if pk is None:
            pk = pack_gate_tc8(m.k["gate_w"])
            self._gate_packs[key] = pk
        return pk

    def handback_bs1_views(self, committed_len):
        """Gather the ring state selected by committed_len into the bs=1-shaped views."""
        for layer, gdn in self.eng.gdn.items():
            ring = self.conv_ring[layer]
            for k in range(3):
                gdn.conv_state4[:, k].copy_(ring[:, ctlabi.conv_slot(committed_len - 3 + k)])
            gdn.conv_state4[:, 3].zero_()
            gdn.state_t.copy_(self.state_ring[layer][ctlabi.state_slot(committed_len - 1)])

    # ---- shared greedy/sampled decision and commit tail ----

    def _verify_and_commit(self):
        if self.use_cp_attention:
            self._ck(
                self.cpc.NexN25ProM4CpFoldStatusLaunch(
                    self._p(self.cp_status_pointers),
                    ctypes.c_int(self.cp_status_pointers.numel()),
                    self._p(self.ws_status),
                    ctypes.c_int(self.eng.rank),
                    ctypes.c_int(self.inject_cp_status_rank),
                    self._sp(),
                ),
                "m4 fold CP status",
            )
        self._launch_sampling()

    def enable_sampling(self, seed_ptr, temperature=0.6, top_p=0.95):
        """Configure MTP sampling before the next replay; no graph replacement."""
        if seed_ptr.dtype != torch.int64 or seed_ptr.numel() != 1:
            raise ValueError("sampling seed must be one int64 device value")
        self.sampling_buffer[:1].copy_(seed_ptr)
        self.set_sampling_config(temperature, top_p)

    def _init_sampling(self):
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem
        from nex_n2_5_pro.kernel_build import build_kernel
        from nex_n2_5_pro.sampling import MAX_TOP_K, sampling_words

        vp, ci, cu = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
        self.sampling_lib = ctypes.CDLL(
            str(build_kernel(HERE / "kernels/head", "nex_n2_5_pro_m4_sampling_ctypes"))
        )
        self.sampling_lib.NexN25ProM4SamplingProducerLaunch.argtypes = (
            [vp] * 5 + [cu, vp, ci] + [vp] * 3
        )
        self.sampling_lib.NexN25ProM4SamplingReducerLaunch.argtypes = [vp] * 9
        self.sampling_lib.NexN25ProM4SamplerLaunch.argtypes = [vp] * 11
        self._sampling_handles = []

        def shared(n, dtype):
            x = symm_mem.empty(n, dtype=dtype, device=self.dev)
            handle = symm_mem.rendezvous(x, dist.group.WORLD.group_name)
            self._sampling_handles.append(handle)
            return (
                x,
                torch.tensor(list(handle.buffer_ptrs), dtype=torch.int64, device=self.dev),
                int(handle.multicast_ptr),
            )

        self.sampling_values, self.sampling_peer_values, _ = shared(
            2 * M * 24 * MAX_TOP_K, torch.float32
        )
        # Two trailing words carry status for the two candidate-bank parities.
        self.sampling_ids, self.sampling_peer_ids, _ = shared(
            2 * M * 24 * MAX_TOP_K + 2, torch.int32
        )
        self.sampling_barrier, _, self.sampling_multicast = shared(1, torch.int32)
        if not self.sampling_multicast:
            raise RuntimeError("M=4 sampling requires an NVLS multicast barrier")
        self.sampling_barrier.zero_()
        self.sampling_generation = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self.sampling_done = torch.zeros_like(self.sampling_generation)
        self.sampling_status = torch.zeros_like(self.sampling_generation)
        self.sampling_upstream = torch.tensor(
            [
                x.data_ptr()
                for x in (
                    self.ws_status,
                    self.ws_moe_status,
                    self.ws12_status,
                    self.ws21_status,
                    self.cursor_status,
                    self.sampling_status,
                )
            ],
            dtype=torch.int64,
            device=self.dev,
        )
        self.sampling_out_values = torch.empty(M * 8 * MAX_TOP_K, device=self.dev)
        self.sampling_out_ids = torch.empty(M * 8 * MAX_TOP_K, dtype=torch.int32, device=self.dev)
        self.sampling_buffer = torch.tensor(
            sampling_words(0, 0.0, 0.95, 20), dtype=torch.int64, device=self.dev
        )
        self.sv_top_ids = torch.full((M, MAX_TOP_K), -1, dtype=torch.int32, device=self.dev)
        self.sv_top_probs = torch.zeros(M, MAX_TOP_K, device=self.dev)
        self.sv_target_ids = torch.full((M,), -1, dtype=torch.int32, device=self.dev)
        self.forced_commit = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self.use_forced_commit = False
        torch.cuda.synchronize()
        dist.barrier()

    def set_sampling_config(self, temperature, top_p, top_k=20, seed=None):
        """Update fixed sampling-buffer fields without recapturing the graph."""
        from nex_n2_5_pro.sampling import sampling_words

        words = sampling_words(0 if seed is None else seed, temperature, top_p, top_k)
        start = 1 if seed is None else 0
        self.sampling_buffer[start:3].copy_(torch.tensor(words[start:3], dtype=torch.int64))

    def set_seed(self, seed):
        value = int(seed) & ((1 << 64) - 1)
        if value >= 1 << 63:
            value -= 1 << 64
        self.sampling_buffer[0].fill_(value)

    def _launch_sampling(self):
        p, sp = self._p, self._sp
        self._ck(
            self.sampling_lib.NexN25ProM4SamplingProducerLaunch(
                p(self.logits4),
                p(self.sampling_values),
                p(self.sampling_ids),
                p(self.sampling_barrier),
                p(self.sampling_generation),
                ctypes.c_uint64(self.sampling_multicast),
                p(self.sampling_done),
                self.eng.rank,
                p(self.sampling_buffer),
                p(self.sampling_upstream),
                sp(),
            ),
            "m4 sampling producer",
        )
        self._ck(
            self.sampling_lib.NexN25ProM4SamplingReducerLaunch(
                p(self.sampling_peer_values),
                p(self.sampling_peer_ids),
                p(self.sampling_barrier),
                p(self.sampling_out_values),
                p(self.sampling_out_ids),
                p(self.sampling_status),
                p(self.sampling_generation),
                p(self.sampling_buffer),
                sp(),
            ),
            "m4 sampling reducer",
        )
        self._ck(
            self.sampling_lib.NexN25ProM4SamplerLaunch(
                p(self.sampling_out_values),
                p(self.sampling_out_ids),
                p(self.sampling_buffer),
                p(self.sv_top_ids),
                p(self.sv_top_probs),
                p(self.sv_target_ids),
                p(self.ctl),
                p(self.eng.slot) if self.mirror_bs1_slot else None,
                p(self.sampling_status),
                p(self.forced_commit) if self.use_forced_commit else None,
                sp(),
            ),
            "m4 sampler/commit",
        )

    def _stage_cp_slots(self):
        self._ck(
            self.cpc.NexN25ProM4CpStageSlotsLaunch(
                self._p(self.ctl),
                self._p(self.cp_slots4),
                self._sp(),
            ),
            "m4 stage CP row slots",
        )


class M4VerifyNonCP(M4Verify):
    """Short-context arm: single-graph attention over the resident KV."""

    def __init__(self, eng, dev):
        super().__init__(eng, dev, use_cp_attention=False)


class M4VerifyCp(M4Verify):
    """Long-context arm: context-parallel attention (KV aliased from the bs=1 CP
    blocks, never duplicated); auto-dispatched at M4_CP_DISPATCH_S tokens with
    the quad CP partial and 8-byte LL combine/merge wire. Attention and GDN
    use the same Tensor Core O-projections as the non-CP arm."""

    def __init__(self, eng, dev, *, use_cp_attention=True):
        super().__init__(eng, dev, use_cp_attention=use_cp_attention)
        self.llt = None
        if not self.use_cp_attention:
            return

        from nex_n2_5_pro.attn_cp import _cp_group
        from nex_n2_5_pro.kernel_build import build_kernel

        vp, ci, cf = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
        self.qpl = ctypes.CDLL(
            str(
                build_kernel(
                    HERE / "kernels" / "full_attn",
                    "nex_n2_5_pro_m4_cp_quad_partial_ctypes",
                    hopper=True,
                    link_cuda=True,
                )
            )
        )
        self.qpl.NexN25ProM4CpQuadPartialLaunch.argtypes = [vp] * 4 + [ci, ci, ci, cf] + [vp] * 4
        self.llt = ctypes.CDLL(
            str(build_kernel(HERE / "kernels" / "full_attn", "nex_n2_5_pro_m4_cp_ll_tail_ctypes"))
        )
        self.llt.CpLLCombinePushEpoch8Launch.argtypes = [vp] * 4 + [vp, ci, ci, ci, cf, vp]
        self.llt.CpLLTailMergeEpoch8Launch.argtypes = [vp, vp, vp, ci, ci] + [vp] * 4 + [vp]

        # Quad-partial workspaces (rank-local): [64 units][ns] + [64][ns][HD].
        self.ll_ws_m = torch.zeros(M * NHCP_LOCAL, NS_QUAD, dtype=torch.float32, device=dev)
        self.ll_ws_l = torch.zeros_like(self.ll_ws_m)
        self.ll_ws_acc = torch.zeros(M * NHCP_LOCAL, NS_QUAD, HD, dtype=torch.bfloat16, device=dev)
        # LL inbox: [2 parity][4 src][4 rows][4 oheads][66 groups][2 packets]
        # of 8 bytes. Symmetric on the CP group; generations advance per call.
        cp_owner = next(iter(eng.attn_cp.values()))
        cp_group = _cp_group(eng.rank)
        words = 2 * 4 * M * OHEADS * LL_PKT_PER_HR * 4
        inbox, peer_ptrs, _mc = cp_owner._symm(words, torch.int32, cp_group.group_name)
        self.ll_inbox = inbox
        self.ll_inbox.fill_(0)
        self.ll_inbox_ptrs = peer_ptrs
        self.ll_epoch = torch.zeros(1, dtype=torch.int32, device=dev)
        self.ll_done = torch.zeros(1, dtype=torch.int32, device=dev)
        # The rendezvous is on the CP GROUP, so peer_ptrs is group-local
        # (4 entries) and cp_base is 0 for every rank; cp.cp_rank is the
        # group-local index the kernels expect.
        self._ll_cp_base = 0
        torch.cuda.synchronize()
        dist.barrier()

    def _attn_layer_m4(self, attention):
        if not self.use_cp_attention:
            return super()._attn_layer_m4(attention)
        cp = self._cp_by_attn[id(attention)]
        from nex_n2_5_pro.attn import EPS, HID, SCALING
        from nex_n2_5_pro.attn_cp import CP

        p, sp = self._p, self._sp
        ci, cf = ctypes.c_int, ctypes.c_float
        weights = cp.w
        self._ck(
            self.cpc.NexN25ProM4CpQGate4Launch(
                p(self.x4),
                p(weights["qgate_w"]),
                p(weights["qgate_scales"]),
                p(self.cp_q_group4),
                p(self.gate4),
                p(cp.k_cache),
                p(cp.v_cache),
                p(self.cp_slots4),
                sp(),
            ),
            "m4 batched CP qgate",
        )
        self._ck(
            self.cpc.NexN25ProM4CpNormRope4Launch(
                p(self.cp_q_group4),
                p(cp.k_cache),
                p(weights["qn"]),
                p(weights["kn"]),
                p(cp.cos_tab),
                p(cp.sin_tab),
                p(self.cp_q_out4),
                p(self.x4),
                p(cp.v_cache),
                p(self.gate4),
                p(self.cp_slots4),
                ci(HID),
                cf(EPS),
                sp(),
            ),
            "m4 batched CP norm rope",
        )
        # Quad partial: cp_q_out4 is [M, NHCP*HD] contiguous == [64, 256].
        self._ck(
            self.qpl.NexN25ProM4CpQuadPartialLaunch(
                p(self.cp_q_out4),
                p(cp.k_cache),
                p(cp.v_cache),
                p(self.cp_slots4),
                ci(cp.cp_rank),
                ci(CP),
                ci(NS_QUAD),
                cf(SCALING),
                p(self.ll_ws_m),
                p(self.ll_ws_l),
                p(self.ll_ws_acc),
                sp(),
            ),
            "m4 quad CP partial",
        )
        scale2 = SCALING * 1.4426950408889634
        self._ck(
            self.llt.CpLLCombinePushEpoch8Launch(
                p(self.ll_ws_m),
                p(self.ll_ws_l),
                p(self.ll_ws_acc),
                p(self.ll_inbox_ptrs),
                p(self.ll_epoch),
                ci(cp.cp_rank),
                ci(self._ll_cp_base),
                ci(NS_QUAD),
                cf(scale2),
                sp(),
            ),
            "m4 LL combine+push",
        )
        self._ck(
            self.llt.CpLLTailMergeEpoch8Launch(
                p(self.ll_inbox_ptrs),
                p(self.ll_epoch),
                p(self.ll_done),
                ci(cp.cp_rank),
                ci(self._ll_cp_base),
                p(self.ctx4),
                p(self.gate4),
                p(self.oproj_xeff4),
                None,
                sp(),
            ),
            "m4 LL merge",
        )
        self._attn_out_fused(attention)
