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

"""M=8 target-model verification runtime: eight ordered external target rows through
the complete backbone plus exact sampling/prefix-commit, as ONE device-driven unit.

THIRD, INDEPENDENT PRODUCTION KERNEL FAMILY. This class inherits from nothing and
launches only M8 kernels (kernels/*/nex_n2_5_pro_m8_*_ctypes.cu). BS=1 and M=4 production
kernels, launchers, runtime classes, mutable scratch, control words, barriers and
generations are never touched: the only sharing is the ENGINE's immutable model
tensors (weights, RoPE tables) and the shared KV-cache storage, aliased by pointer.
M=8-owned state: the m8 control block, per-GDN-layer conv ring [1536,16] bf16 and
recurrent state ring [9,8,128,128] fp32, every activation/logit/route buffer, its
own symmetric-memory reduce workspaces + per-CTA arrival counters + NVLS barriers
(counter families must never be shared across producer families), and its
small derived weight layouts, built during eager warmup. MoE W13/W2 tensors
remain in their original row-major layout. The raw LM-head pack is shared with DFlash.

The unit (one run() call == one captured graph):
  embed(ctl ids) -> per layer [GDN front/core/o_proj+reduce | attn
  qkv/norm-rope/core+combine-xeff/o_proj+reduce] + MoE [gate/topk/L1/L2+reduce]
  -> final-norm+lm_head (TRUE fp32 logits) -> tail producer -> TP reducer -> one-CTA sampler/commit. Fully device-driven: no .item(), no
  allocation, no host token materialization inside run().

Verification semantics: sampled row j is compared with input t(j+1) for j = 0..6
until the first mismatch; commit_count in 1..8; committed_len published LAST; any
status failure leaves committed_len unchanged (fail closed). Both rings hold the
whole live span of a unit (base + eight rows), so a failed unit does NOT damage
the state it would re-read: call reset_unit() to clear the sticky protocol
latches and the same committed_len can be re-verified without a reload.
"""

import ctypes
import weakref
from pathlib import Path

import torch

from nex_n2_5_pro import m8_control as ctlabi

HERE = Path(__file__).resolve().parent

M = ctlabi.M  # 8
HID = 4096
QKV_ROWS = 1536
Z_ROWS = 1024
LOCAL = 1024
GDN_HEADS = 8
GDN_DIM = 128
ROWS_LOCAL = 31040
GATE_ROWS_PHYS = 528
PACKED_WORDS = 64
RAWH_ROWS = 1408
RINTER_TP = 128  # one TP shard of the MoE intermediate dim
TOPK_EXPERTS = 10
CURSOR_LEN = M + 2


class _WgmmaWeightMap:
    """Own one WGMMA tensor map and keep its source allocation alive."""

    def __init__(self, library, weight):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare MoE weight maps before CUDA graph capture")
        if weight.dtype != torch.float8_e4m3fn or not weight.is_contiguous():
            raise ValueError("WGMMA W13 weights must be contiguous FP8 tensors")
        if tuple(weight.shape) != (512, 256, HID):
            raise ValueError(f"Unexpected W13 shape for WGMMA: {tuple(weight.shape)}")
        self.weight = weight
        self.handle = ctypes.c_void_p()
        rc = library.NexN25ProM8WmMapCreate(weight.data_ptr(), ctypes.byref(self.handle))
        if rc:
            raise RuntimeError(f"M8 WGMMA tensor-map creation failed: {rc}")
        self._finalizer = weakref.finalize(self, library.NexN25ProM8WmMapDestroy, self.handle)


# Every M8 translation unit, with the exact flags M8Verify.__init__ loads it
# under. Kept beside the loader so a new TU cannot be added to one and not the
# other: prebuild_m8_kernels() asserts it covers what __init__ actually builds.
M8_KERNEL_TUS = (
    ("embed", "nex_n2_5_pro_m8_embed_gather_ctypes", {}),
    ("linear_attn", "nex_n2_5_pro_m8_gdn_front_ctypes", {}),
    ("linear_attn", "nex_n2_5_pro_m8_gdn_core_ctypes", {}),
    ("linear_attn", "nex_n2_5_pro_m8_gdn_oproj_ctypes", {}),
    ("full_attn", "nex_n2_5_pro_m8_attn_qkv_ctypes", {}),
    ("full_attn", "nex_n2_5_pro_m8_norm_rope_ctypes", {}),
    ("full_attn", "nex_n2_5_pro_m8_attn_core_ctypes", {"hopper": True, "link_cuda": True}),
    ("full_attn", "nex_n2_5_pro_m8_attn_oproj_ctypes", {}),
    ("full_attn", "nex_n2_5_pro_m8_cp_attn_ctypes", {"hopper": True, "link_cuda": True}),
    ("moe", "nex_n2_5_pro_m8_gate_gemm_ctypes", {}),
    ("moe", "nex_n2_5_pro_m8_gate_topk_ctypes", {}),
    ("moe", "nex_n2_5_pro_m8_moe_wm16_ctypes", {"hopper": True, "link_cuda": True}),
    ("moe", "nex_n2_5_pro_m8_moe_l2tc_ctypes", {"hopper": True, "link_cuda": True}),
    ("head", "nex_n2_5_pro_m8_lmhead_gemm_ctypes", {}),
    ("head", "nex_n2_5_pro_m8_sampling_ctypes", {}),
)


def prebuild_m8_kernels(verbose=False):
    """Compile every M8 TU in ONE process; return [(name, so_path)].

    build_kernel is already flock-serialized, but a cold eight-rank start still
    has seven ranks blocking on locks while one compiles, and any nvcc failure
    then surfaces as seven confusing secondary errors. Callers run this on rank 0
    and dist.barrier() before constructing M8Verify, so the eight ranks only ever
    LOAD. Raises on the first TU that fails, naming that TU.
    """
    from nex_n2_5_pro.kernel_build import build_kernel

    built = []
    for subdir, name, flags in M8_KERNEL_TUS:
        so = build_kernel(HERE / "kernels" / subdir, name, **flags)
        built.append((name, so))
        if verbose:
            print(f"[m8-build] ok {name} -> {so}", flush=True)
    return built


def pack_gemm_bf16tc_m8(W, K):
    """[rows, K] e4m3 row-major -> the M8 bf16tc kernels' lane-major prepack
    Wpack[tile 8r][k32][lane][8 B] (same bytes, permuted). Each lane's 8 bytes are
    its mma.m16n8k16 A-fragment pairs for the two k16 steps of one k32 tile."""
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


def pack_qkv16_m8(W, K):
    """[rows, K] e4m3 row-major -> the ft16 GDN front's 16-real-row lane-major prepack
    Wpack[tile 16r][k32][lane][16 B] (same bytes, permuted): one uint4 per lane per k32
    IS its four m16n8k16 A fragments for the two k16 steps (the pack_lmhead_m8 full-M16
    law restated for fp8; byte fields rowh=(b>>1)&1 -> m0..7|m8..15, e=b&1)."""
    dev = W.device
    rows = W.numel() // K
    t = torch.arange(rows // 16, device=dev).view(-1, 1, 1, 1)
    c = torch.arange(K // 32, device=dev).view(1, -1, 1, 1)
    lane = torch.arange(32, device=dev).view(1, 1, 32, 1)
    b = torch.arange(16, device=dev).view(1, 1, 1, 16)
    gid, tig = lane >> 2, lane & 3
    step, half8, rowh, e = b >> 3, (b >> 2) & 1, (b >> 1) & 1, b & 1
    row = t * 16 + rowh * 8 + gid
    k = c * 32 + step * 16 + half8 * 8 + tig * 2 + e
    src = (row * K + k).reshape(-1)
    return W.reshape(-1)[src].contiguous()


def pack_gate_m8(W):
    """[513|528, 4096] bf16 gate_w -> the M8 gate kernel's lane-major prepack
    (rows zero-padded to 528)."""
    assert W.dtype == torch.bfloat16 and W.shape[-1] == HID and W.shape[0] <= GATE_ROWS_PHYS
    dev = W.device
    wp = torch.zeros(GATE_ROWS_PHYS, HID, dtype=torch.bfloat16, device=dev)
    wp[: W.shape[0]] = W
    tile = torch.arange(GATE_ROWS_PHYS // 8, device=dev).view(-1, 1, 1, 1, 1)
    k16 = torch.arange(HID // 16, device=dev).view(1, -1, 1, 1, 1)
    lane = torch.arange(32, device=dev).view(1, 1, 32, 1, 1)
    f = torch.arange(2, device=dev).view(1, 1, 1, 2, 1)
    e = torch.arange(2, device=dev).view(1, 1, 1, 1, 2)
    gid, tig = lane >> 2, lane & 3
    src = ((tile * 8 + gid) * HID + k16 * 16 + f * 8 + tig * 2 + e).reshape(-1)
    return wp.reshape(-1)[src].contiguous()


def pack_lmhead_m8(weight):
    """[31040, 4096] raw BF16 head shard -> the M8 tc16 head prepack
    Wpack[tile 16r][k16][lane][frag a0|a1|a2|a3][2 bf16] (16 REAL vocab rows/tile)."""
    dev = weight.device
    tile = torch.arange(ROWS_LOCAL // 16, device=dev).view(-1, 1, 1, 1, 1)
    k16 = torch.arange(HID // 16, device=dev).view(1, -1, 1, 1, 1)
    lane = torch.arange(32, device=dev).view(1, 1, 32, 1, 1)
    f = torch.arange(4, device=dev).view(1, 1, 1, 4, 1)
    e = torch.arange(2, device=dev).view(1, 1, 1, 1, 2)
    gid, tig = lane >> 2, lane & 3
    row = tile * 16 + (f & 1) * 8 + gid
    k = k16 * 16 + (f >> 1) * 8 + tig * 2 + e
    src = (row * HID + k).reshape(-1)
    return weight.reshape(-1)[src].contiguous()


class M8Verify:
    """The non-CP M=8 verification arm. Owns every mutable M8 state object;
    aliases only immutable engine weights and the shared KV caches."""

    def __init__(self, eng, dev, *, dflash_tap_layers=None):
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem

        from nex_n2_5_pro.decode_engine import _p, _sp
        from nex_n2_5_pro.head import EPS
        from nex_n2_5_pro.attn import SCALING
        from nex_n2_5_pro.kernel_build import build_kernel

        self.eng, self.dev = eng, dev
        self._p, self._sp, self._eps, self._scaling = _p, _sp, EPS, SCALING
        vp, cf, ci, cu, cll = (
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_ulonglong,
            ctypes.c_longlong,
        )
        kd = HERE / "kernels"

        # The M8 MoE kernels read the RAW (unfolded) weight layout and the per-layer
        # gamma that only the raw path materialises. Without it the failure is
        # `AttributeError: 'Moe' object has no attribute 'gamma'` at the first MoE
        # layer of the first run() -- long after construction, and unrecognisable.
        # Fail here instead, naming the knob.
        first = next(iter(eng.moe.values()), None)
        if first is not None and not getattr(first, "raw", False):
            raise RuntimeError(
                "M8Verify requires the RAW MoE weight layout: set NEXRT_MOE_RAW=1 "
                "before constructing the engine (the M8 MoE L1/top-k kernels read "
                "moe.gamma and the unfolded w13, which the folded path never builds)."
            )
        self._raw_head_weight = eng.head.retain_raw_weight()

        # ---- M8 kernel libraries (each an M8-only .so) ----
        self.embl = ctypes.CDLL(
            str(build_kernel(kd / "embed", "nex_n2_5_pro_m8_embed_gather_ctypes"))
        )
        self.embl.NexN25ProM8EmbedGatherLaunch.argtypes = [vp] * 5
        self.gfl = ctypes.CDLL(
            str(build_kernel(kd / "linear_attn", "nex_n2_5_pro_m8_gdn_front_ctypes"))
        )
        self.gfl.NexN25ProM8GdnFrontLaunch.argtypes = [vp] * 16
        self.gfl.NexN25ProM8GdnFrontSk2WsFloats.restype = ci
        self.gfl.NexN25ProM8GdnFrontSk2Tiles.restype = ci
        # Partial workspace + arrival tickets: zero-init ONCE, never reset (each
        # launch adds exactly 2 per tile, so the modular ticket elects exactly
        # one folder per tile per launch).
        self.front_sk2_ws = torch.zeros(
            self.gfl.NexN25ProM8GdnFrontSk2WsFloats(), dtype=torch.float32, device=dev
        )
        self.front_sk2_tickets = torch.zeros(
            self.gfl.NexN25ProM8GdnFrontSk2Tiles(), dtype=torch.int32, device=dev
        )
        self.gcl = ctypes.CDLL(
            str(build_kernel(kd / "linear_attn", "nex_n2_5_pro_m8_gdn_core_ctypes"))
        )
        self.gcl.NexN25ProM8GdnCoreLaunch.argtypes = [vp] * 7
        self.gol = ctypes.CDLL(
            str(build_kernel(kd / "linear_attn", "nex_n2_5_pro_m8_gdn_oproj_ctypes"))
        )
        self.gol.NexN25ProM8GdnOProjLaunch.argtypes = [vp] * 5 + [cu, vp] + [vp] * 3 + [ci, ci, vp]
        self.gol.NexN25ProM8OProjWorkspaceFloats.restype = ci
        self.gol.NexN25ProM8OProjFusedCTAs.restype = ci
        self.aql = ctypes.CDLL(
            str(build_kernel(kd / "full_attn", "nex_n2_5_pro_m8_attn_qkv_ctypes"))
        )
        self.aql.NexN25ProM8AttnQkvBf16TcLaunch.argtypes = [vp, vp, vp] + [vp] * 5 + [vp]
        self.nrl = ctypes.CDLL(
            str(build_kernel(kd / "full_attn", "nex_n2_5_pro_m8_norm_rope_ctypes"))
        )
        self.nrl.NexN25ProM8NormRopeLaunch.argtypes = [vp] * 11 + [ci, cf, vp]
        self.acl = ctypes.CDLL(
            str(
                build_kernel(
                    kd / "full_attn",
                    "nex_n2_5_pro_m8_attn_core_ctypes",
                    hopper=True,
                    link_cuda=True,
                )
            )
        )
        self.acl.NexN25ProM8AttnCoreLaunch.argtypes = [vp] * 4 + [ci, cf] + [vp] * 6 + [vp]
        # One full 132-CTA wave avoids the eight long-tail splits created by
        # the M=8 causal window at power-of-two contexts.
        # M=8's S7 = S + MROWS8 makes S7 % 128 == 8 at every power-of-two
        # context, so exactly 8 of 128 CTAs cross a 64-key tile boundary and set
        # a 2-tile critical path for an otherwise 1-tile grid. ns=132 removes
        # that and is exactly one full wave on 132 SMs.
        #
        # ns changes the attention split decomposition and FP reduction
        # order, so ns=132 belongs to the declared family tolerance rather than
        # a bitwise-equivalence contract. The correctness suites cover the exact
        # partition, empty splits, forbidden-key masking, and graph replay.
        self._attn_nctas = 132
        self.aol = ctypes.CDLL(
            str(build_kernel(kd / "full_attn", "nex_n2_5_pro_m8_attn_oproj_ctypes"))
        )
        self.aol.NexN25ProM8AttnOProjLaunch.argtypes = [vp] * 3 + [cu, vp] + [vp] * 3 + [ci, ci, vp]
        self.ggl = ctypes.CDLL(str(build_kernel(kd / "moe", "nex_n2_5_pro_m8_gate_gemm_ctypes")))
        self.ggl.NexN25ProM8GateGemmPdlLaunch.argtypes = [vp, vp, vp, vp, cf, vp, cll, vp]
        self.gtl = ctypes.CDLL(str(build_kernel(kd / "moe", "nex_n2_5_pro_m8_gate_topk_ctypes")))
        self.gtl.NexN25ProM8GateTopKPackedKeyLaunch.argtypes = [vp] * 9
        # WM16 weight-major L1 and tensor-core L2-local form one hardwired PDL
        # pair. The route-plan ABI is shared by both kernels.
        self.wml = ctypes.CDLL(
            str(
                build_kernel(
                    kd / "moe", "nex_n2_5_pro_m8_moe_wm16_ctypes", hopper=True, link_cuda=True
                )
            )
        )
        self.wml.NexN25ProM8WmPlanLaunch.argtypes = [vp, vp, vp]
        self.wml.NexN25ProM8WmPlanInts.restype = ci
        self.wm_plan = torch.zeros(self.wml.NexN25ProM8WmPlanInts(), dtype=torch.int32, device=dev)
        self.wml.NexN25ProM8WmMapCreate.argtypes = [vp, ctypes.POINTER(vp)]
        self.wml.NexN25ProM8WmMapDestroy.argtypes = [vp]
        self.wml.NexN25ProM8WmMapDestroy.restype = None
        self.wml.NexN25ProM8Wm4L1Launch.argtypes = [vp] * 12
        self._ck(self.wml.NexN25ProM8WmPrepare(), "m8 WGMMA prepare")
        for n in ("NexN25ProM8Wm4L1Ctas", "NexN25ProM8Wm4L1Smem", "NexN25ProM8Wm4L1BlocksPerSm"):
            getattr(self.wml, n).restype = ci
        self.l2 = ctypes.CDLL(
            str(
                build_kernel(
                    kd / "moe", "nex_n2_5_pro_m8_moe_l2tc_ctypes", hopper=True, link_cuda=True
                )
            )
        )
        self.l2.NexN25ProM8MoeL2LocalLaunch.argtypes = [vp] * 8 + [vp]
        # LL peer-inbox tail: the inbox multicast VA is the publish target.
        self.l2.NexN25ProM8MoeL2TailLlLaunch.argtypes = [
            vp,
            vp,
            vp,
            ctypes.c_ulonglong,
            vp,
            ci,
            ci,
            vp,
            vp,
            vp,
        ]
        self.l2.NexN25ProM8MoeL2InboxBytes.restype = cll
        for n in (
            "NexN25ProM8MoeL2PartialFloats",
            "NexN25ProM8MoeL2TailCtrInts",
            "NexN25ProM8MoeL2LocalCtas",
            "NexN25ProM8MoeL2LocalSmem",
            "NexN25ProM8MoeL2LocalBlocksPerSm",
        ):
            getattr(self.l2, n).restype = ci
        self.l2_p = torch.zeros(
            self.l2.NexN25ProM8MoeL2PartialFloats(), dtype=torch.float32, device=dev
        )
        self.hdl = ctypes.CDLL(str(build_kernel(kd / "head", "nex_n2_5_pro_m8_lmhead_gemm_ctypes")))
        self.hdl.NexN25ProM8FinalNormLaunch.argtypes = [vp, vp, vp, cf, vp]
        self.hdl.NexN25ProM8LmHeadGemmTc16Launch.argtypes = [vp, vp, vp, vp]
        # Optional DFlash boundary: snapshot the post-MoE residual after each
        # checkpoint-configured target layer. Ordinary M=8 users allocate and
        # launch none of this state.
        self.dflash_taps = None
        self._dflash_tap_launch = None
        if dflash_tap_layers is not None:
            tap_layers = tuple(int(layer) for layer in dflash_tap_layers)
            if not tap_layers or len(tap_layers) != len(set(tap_layers)):
                raise ValueError("dflash_tap_layers must be nonempty and unique")
            missing = sorted(set(tap_layers) - set(eng.layers))
            if missing:
                raise ValueError(
                    f"DFlash tap layers are outside the active target model: {missing}"
                )
            tap_library = ctypes.CDLL(
                str(build_kernel(kd / "embed", "nex_n2_5_pro_m8_feature_tap_ctypes"))
            )
            tap_library.NexN25ProM8FeatTapLaunch.argtypes = [vp, vp, vp]
            self._dflash_tap_library = tap_library
            self._dflash_tap_launch = tap_library.NexN25ProM8FeatTapLaunch
            self.dflash_taps = {
                layer: torch.empty(M, HID, dtype=torch.bfloat16, device=dev) for layer in tap_layers
            }

        # ---- the one M8 control block; verify_input_ids is the execution truth ----
        self.ctl = ctlabi.new_control_block(dev)
        self.verify_ids = self.ctl[ctlabi.VERIFY_INPUT_IDS : ctlabi.VERIFY_INPUT_IDS + M]
        self.cursor = torch.zeros(CURSOR_LEN, dtype=torch.int32, device=dev)
        self.cursor_status = self.cursor[M + 1 : M + 2]  # a VIEW, not a byte offset

        # ---- activation / route / logit buffers (all M8-owned, preallocated) ----
        self.x8 = torch.empty(M, HID, dtype=torch.bfloat16, device=dev)
        self.qkv8 = torch.empty(M, QKV_ROWS, dtype=torch.bfloat16, device=dev)
        self.z8 = torch.empty(M, Z_ROWS, dtype=torch.bfloat16, device=dev)
        self.beta8 = torch.zeros(M, GDN_HEADS, dtype=torch.float32, device=dev)
        self.g8 = torch.zeros(M, GDN_HEADS, dtype=torch.float32, device=dev)
        self.rec8 = torch.empty(M, LOCAL, dtype=torch.bfloat16, device=dev)
        self.q_raw8 = torch.empty(M, LOCAL, dtype=torch.bfloat16, device=dev)
        self.gate8 = torch.empty(M, LOCAL, dtype=torch.bfloat16, device=dev)
        self.q_out8 = torch.empty(M, LOCAL, dtype=torch.bfloat16, device=dev)
        self.ctx8 = torch.empty(M, LOCAL, dtype=torch.bfloat16, device=dev)
        self.xeff8 = torch.empty(M, LOCAL, dtype=torch.bfloat16, device=dev)
        self.gate_logits8 = torch.empty(M, GATE_ROWS_PHYS, dtype=torch.bfloat16, device=dev)
        self.rms8 = torch.zeros(M, dtype=torch.float32, device=dev)
        self.packed8 = torch.zeros(M, PACKED_WORDS, dtype=torch.int32, device=dev)
        self.xn8 = torch.empty(M, HID, dtype=torch.bfloat16, device=dev)
        self.moe_xq8 = torch.empty(M, HID, dtype=torch.float8_e4m3fn, device=dev)
        self.moe_xs8 = torch.empty(M, HID // 128, dtype=torch.float32, device=dev)
        self.rawh_rows = RAWH_ROWS
        self.rawh8 = torch.empty(M, self.rawh_rows, dtype=torch.float32, device=dev)
        self.logits8 = torch.full((M, ROWS_LOCAL), float("nan"), dtype=torch.float32, device=dev)

        # ---- M8-only per-GDN-layer rings (BS1 conv_state4/state_t untouched) ----
        self.conv_ring, self.state_ring = {}, {}
        for layer in eng.gdn:
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

        # ---- attention core workspaces ----
        # M=8-owned split count; it is not inherited from BS=1 geometry.
        # bs=1 engine geometry. Sizing the workspaces from the same value that is
        # passed to the launcher keeps them in bounds: at ns=132 an eng.attn-
        # derived 128 would under-allocate m/l/acc by 4 splits.
        self.nctas = self._attn_nctas
        NHL = 4
        HEAD_DIM = 256
        self.ws_attn_m8 = torch.empty(M * self.nctas * NHL, dtype=torch.float32, device=dev)
        self.ws_attn_l8 = torch.empty(M * self.nctas * NHL, dtype=torch.float32, device=dev)
        self.ws_attn_acc8 = torch.empty(
            M * self.nctas * NHL * HEAD_DIM, dtype=torch.bfloat16, device=dev
        )

        # ---- M8 LL o_proj inbox, shared sequentially by the two mixer types ----
        ws_floats = int(self.gol.NexN25ProM8OProjWorkspaceFloats())
        op_ctas = int(self.gol.NexN25ProM8OProjFusedCTAs())
        self.op_rep = symm_mem.empty(ws_floats, dtype=torch.float32, device=dev)
        rep_handle = symm_mem.rendezvous(self.op_rep, dist.group.WORLD.group_name)
        self.op_rep_mc = int(getattr(rep_handle, "multicast_ptr", 0) or 0)
        if self.op_rep_mc == 0:
            raise RuntimeError("M8 o_proj inbox requires NVLS multicast")
        self.op_rep.zero_()
        self.op_it = torch.zeros(op_ctas, dtype=torch.int32, device=dev)
        assert ws_floats == 2 * 8 * op_ctas * 128 * 2, (
            "LL inbox sizing must match the allocated workspace"
        )
        self.op_status = torch.zeros(1, dtype=torch.int32, device=dev)

        self.moe_gen = torch.zeros(1, dtype=torch.int32, device=dev)
        self.moe_status = torch.zeros(1, dtype=torch.int32, device=dev)
        # M8-owned symmetric inbox (2 parity halves x ranks x 16384 8 B
        # packets = 2 MiB/rank) + 128 per-CTA private epoch counters.
        # Zero-initialized once: epoch 0 never equals a real gen (>= 1).
        ib_bytes = int(self.l2.NexN25ProM8MoeL2InboxBytes())
        self.moe_ll_inbox = symm_mem.empty(ib_bytes // 8, dtype=torch.int64, device=dev)
        ll_handle = symm_mem.rendezvous(self.moe_ll_inbox, dist.group.WORLD.group_name)
        self.moe_ll_peers = torch.tensor(list(ll_handle.buffer_ptrs), dtype=torch.int64, device=dev)
        # NVLS multicast alias of the same inbox: the LL tail's single
        # publication target. Required at TP > 1; at world == 1 there is
        # no multicast team (0) and the LL kernel publishes nothing.
        self.moe_ll_mc = int(getattr(ll_handle, "multicast_ptr", 0) or 0)
        if dist.get_world_size() > 1 and self.moe_ll_mc == 0:
            raise RuntimeError(
                "M8 LL tail requires NVLS multicast at TP > 1 (inbox multicast_ptr == 0)"
            )
        self.moe_ll_inbox.zero_()
        self.moe_ll_ctr = torch.zeros(
            int(self.l2.NexN25ProM8MoeL2TailCtrInts()), dtype=torch.int32, device=dev
        )

        self._init_sampling()

        # ---- M8-owned lane-major prepacks (immutable derived weights; lazy) ----
        self._qkvz16_packs = {}  # gdn qkvz_w  ptr -> ft16 16-row pack (fp8, K=4096)
        self._qkv_packs = {}  # attn qkv_w  ptr -> pack (fp8, K=4096)
        self._oproj_packs = {}  # o_w         ptr -> pack (fp8, K=1024)
        self._gate_packs = {}  # gate_w      ptr -> pack (bf16)
        self._wm_maps = {}
        self._head_pack = None

        # The qualified production path does not issue speculative next-layer
        # prefetches from the gate kernel.
        self._front_pf = {}

        torch.cuda.synchronize()
        dist.barrier()

    # ---- configuration -------------------------------------------------------

    def _init_sampling(self):
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem
        from nex_n2_5_pro.kernel_build import build_kernel
        from nex_n2_5_pro.sampling import MAX_TOP_K, sampling_words

        vp, ci, cu = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
        self.sampling_lib = ctypes.CDLL(
            str(build_kernel(HERE / "kernels/head", "nex_n2_5_pro_m8_sampling_ctypes"))
        )
        self.sampling_lib.NexN25ProM8SamplingProducerLaunch.argtypes = (
            [vp] * 5 + [cu, vp, ci] + [vp] * 3
        )
        self.sampling_lib.NexN25ProM8SamplingReducerLaunch.argtypes = [vp] * 9
        self.sampling_lib.NexN25ProM8SamplerLaunch.argtypes = [vp] * 11
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
            raise RuntimeError("M=8 sampling requires an NVLS multicast barrier")
        self.sampling_barrier.zero_()
        self.sampling_generation = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self.sampling_done = torch.zeros_like(self.sampling_generation)
        self.sampling_status = torch.zeros_like(self.sampling_generation)
        self.sampling_upstream = torch.tensor(
            [
                x.data_ptr()
                for x in (
                    self.op_status,
                    self.moe_status,
                    self.cursor_status,
                    self.sampling_status,
                    self.sampling_status,
                )
            ],
            dtype=torch.int64,
            device=self.dev,
        )
        self.sampling_out_values = torch.empty(M * 8 * MAX_TOP_K, device=self.dev)
        self.sampling_out_ids = torch.empty(M * 8 * MAX_TOP_K, dtype=torch.int32, device=self.dev)
        self.sampling_buffer = torch.tensor(
            sampling_words(0, 0.6, 0.95, 20), dtype=torch.int64, device=self.dev
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

    def _launch_sampling(self, upstream=None):
        p, sp = self._p, self._sp
        self._ck(
            self.sampling_lib.NexN25ProM8SamplingProducerLaunch(
                p(self.logits8),
                p(self.sampling_values),
                p(self.sampling_ids),
                p(self.sampling_barrier),
                p(self.sampling_generation),
                ctypes.c_uint64(self.sampling_multicast),
                p(self.sampling_done),
                self.eng.rank,
                p(self.sampling_buffer),
                p(self.sampling_upstream if upstream is None else upstream),
                sp(),
            ),
            "m8 sampling producer",
        )
        self._ck(
            self.sampling_lib.NexN25ProM8SamplingReducerLaunch(
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
            "m8 sampling reducer",
        )
        self._ck(
            self.sampling_lib.NexN25ProM8SamplerLaunch(
                p(self.sampling_out_values),
                p(self.sampling_out_ids),
                p(self.sampling_buffer),
                p(self.sv_top_ids),
                p(self.sv_top_probs),
                p(self.sv_target_ids),
                p(self.ctl),
                None,
                p(self.sampling_status),
                p(self.forced_commit) if self.use_forced_commit else None,
                sp(),
            ),
            "m8 sampler/commit",
        )

    # ---- prepack access (lazy; NEVER under graph capture) --------------------
    def _pack_of(self, cache, W, K):
        key = W.data_ptr()
        pk = cache.get(key)
        if pk is None:
            pk = pack_gemm_bf16tc_m8(W, K)
            cache[key] = pk
        return pk

    def _pack16_of(self, W, K):
        key = W.data_ptr()
        pk = self._qkvz16_packs.get(key)
        if pk is None:
            pk = pack_qkv16_m8(W, K)
            self._qkvz16_packs[key] = pk
        return pk

    def _gate_pack(self, W):
        key = W.data_ptr()
        pk = self._gate_packs.get(key)
        if pk is None:
            pk = pack_gate_m8(W)
            self._gate_packs[key] = pk
        return pk

    def lm_head_pack(self):
        """Return the one raw-shard tc16 pack shared by M=8 and DFlash."""
        if self._head_pack is None:
            self._head_pack = pack_lmhead_m8(self._raw_head_weight)
        return self._head_pack

    def build_packs(self):
        """Materialize every prepack up front (call once before capture)."""
        eng = self.eng
        for gdn in eng.gdn.values():
            self._pack16_of(gdn.w["qkvz_w"], HID)
            self._pack_of(self._oproj_packs, gdn.w["o_w"], LOCAL)
        for attn in eng.attn.values():
            self._pack_of(self._qkv_packs, attn.w["qkv_w"], HID)
            self._pack_of(self._oproj_packs, attn.w["o_w"], LOCAL)
        for moe in eng.moe.values():
            self._gate_pack(moe.k["gate_w"])
            self._wm_map(moe.k["w13"])
        self.lm_head_pack()

    def _wm_of(self, layer):
        """Alias the engine's row-major routed and shared W13 tensors."""
        weights = self.eng.moe[layer].k
        return {"w13": weights["w13"], "shared_w13": weights["shared_gate_up"]}

    def _wm_map(self, weight):
        key = weight.data_ptr()
        owned = self._wm_maps.get(key)
        if owned is None:
            owned = _WgmmaWeightMap(self.wml, weight)
            self._wm_maps[key] = owned
        return owned

    def _l2_of(self, layer):
        """Alias the engine's row-major routed and shared W2 tensors."""
        weights = self.eng.moe[layer].k
        return {"w2": weights["W2"], "shared_w2": weights["shared_W2"]}

    def load(self, state_dir, verify_input_ids):
        """Seed the engine's BS1 state from the prefill dir, then scatter the SAME
        history into the M8 rings and initialize the M8 control block. The BS1
        loader itself is untouched; M8 provably starts from bit-identical history."""
        length, _first = self.eng.load_prefill(state_dir=Path(state_dir))
        self.seed(length, verify_input_ids)
        return length

    def seed(self, length, verify_input_ids):
        """Initialize M8 rings from the engine's current prefill boundary."""
        ctlabi.init_control_block(self.ctl, length, verify_input_ids)
        for layer, gdn in self.eng.gdn.items():
            ring = self.conv_ring[layer]
            ring.zero_()
            # conv_state4[:, 0:3] holds positions L-3, L-2, L-1 (oldest first).
            for k in range(3):
                ring[:, ctlabi.conv_slot(length - 3 + k)].copy_(gdn.conv_state4[:, k])
            state = self.state_ring[layer]
            state.zero_()
            state[ctlabi.state_slot(length - 1)].copy_(gdn.state_t)
        self.cursor.zero_()
        self.logits8.fill_(float("nan"))
        self.xn8.zero_()
        self.op_status.zero_()
        self.moe_status.zero_()
        self.sampling_status.zero_()
        return length

    def reset_unit(self):
        """Clear the sticky protocol latches so a FAILED unit can be retried at the
        same committed_len, without reloading the request.

        This is sound only because both rings hold a unit's entire live span (see
        m8_control_block.cuh): the failed unit left the base recurrent snapshot and
        the three convolution history taps untouched, and its provisional KV rows
        at C..C+7 are simply rewritten by the retry. Nothing else needs undoing --
        committed_len never moved, and the control block's own status word is
        recomputed by the tail of every unit.

        Deliberately does NOT touch the rings, committed_len, the protocol
        generations, or cursor[M] (the monotonic per-replay consumption count).
        Host-side, outside run(): calling it under capture would bake a status
        clear into the graph and defeat fail-closed propagation across replays.
        """
        self.op_status.zero_()
        self.moe_status.zero_()
        self.cursor_status.zero_()
        self.sampling_status.zero_()

    @staticmethod
    def _ck(rc, what):
        """Launch-status guard: nonzero means NOTHING was launched. Host int
        check only, no device synchronization."""
        if rc != 0:
            raise RuntimeError(f"{what} launch failed with CUDA error {rc}")

    # ---- per-layer pieces ----------------------------------------------------
    def _gdn_layer(self, layer, gdn):
        p, sp = self._p, self._sp
        w = gdn.w
        self._ck(
            self.gfl.NexN25ProM8GdnFrontLaunch(
                p(self.x8),
                p(self._pack16_of(w["qkvz_w"], HID)),
                p(w["qkvz_scales"]),
                p(w["b_w"]),
                p(w["a_w"]),
                p(w["a_log_dt"]),
                p(w["conv_w"]),
                p(self.conv_ring[layer]),
                p(self.ctl),
                p(self.qkv8),
                p(self.z8),
                p(self.beta8),
                p(self.g8),
                p(self.front_sk2_ws),
                p(self.front_sk2_tickets),
                sp(),
            ),
            "m8 gdn front",
        )
        self._ck(
            self.gcl.NexN25ProM8GdnCoreLaunch(
                p(self.qkv8),
                p(self.beta8),
                p(self.g8),
                p(self.state_ring[layer]),
                p(self.ctl),
                p(self.rec8),
                sp(),
            ),
            "m8 gdn core",
        )
        self._ck(
            self.gol.NexN25ProM8GdnOProjLaunch(
                p(self.rec8),
                p(self.z8),
                p(w["norm_w"]),
                p(self._pack_of(self._oproj_packs, w["o_w"], LOCAL)),
                p(w["o_scales"]),
                ctypes.c_ulonglong(self.op_rep_mc),
                p(self.op_rep),
                p(self.op_it),
                p(self.op_status),
                p(self.x8),
                self.eng.rank,
                self.eng.world,
                sp(),
            ),
            "m8 gdn oproj",
        )

    def _attn_layer(self, attn):
        p, sp = self._p, self._sp
        w = attn.w
        ci, cf = ctypes.c_int, ctypes.c_float
        self._ck(
            self.aql.NexN25ProM8AttnQkvBf16TcLaunch(
                p(self.x8),
                p(self._pack_of(self._qkv_packs, w["qkv_w"], HID)),
                p(w["qkv_scales"]),
                p(self.q_raw8),
                p(self.gate8),
                p(attn.k_cache),
                p(attn.v_cache),
                p(self.ctl),
                sp(),
            ),
            "m8 attn qkv",
        )
        self._ck(
            self.nrl.NexN25ProM8NormRopeLaunch(
                p(self.q_raw8),
                p(attn.k_cache),
                p(w["qn"]),
                p(w["kn"]),
                p(attn.cos_tab),
                p(attn.sin_tab),
                p(self.q_out8),
                p(self.ctl),
                p(self.x8),
                p(attn.v_cache),
                p(self.gate8),
                ci(HID),
                cf(self._eps),
                sp(),
            ),
            "m8 attn norm rope",
        )
        self._ck(
            self.acl.NexN25ProM8AttnCoreLaunch(
                p(self.q_out8),
                p(attn.k_cache),
                p(attn.v_cache),
                p(self.ctl),
                ci(self.nctas),
                cf(self._scaling),
                p(self.ws_attn_m8),
                p(self.ws_attn_l8),
                p(self.ws_attn_acc8),
                p(self.ctx8),
                p(self.gate8),
                p(self.xeff8),
                sp(),
            ),
            "m8 attn core",
        )
        self._ck(
            self.aol.NexN25ProM8AttnOProjLaunch(
                p(self.xeff8),
                p(self._pack_of(self._oproj_packs, w["o_w"], LOCAL)),
                p(w["o_scales"]),
                ctypes.c_ulonglong(self.op_rep_mc),
                p(self.op_rep),
                p(self.op_it),
                p(self.op_status),
                p(self.x8),
                self.eng.rank,
                self.eng.world,
                sp(),
            ),
            "m8 attn oproj",
        )

    def _moe_l2(self, layer, moe):
        """Run the M8 TP=8 TC local projection and LL peer-inbox tail."""
        p, sp = self._p, self._sp
        w = self._l2_of(layer)
        self._ck(
            self.l2.NexN25ProM8MoeL2LocalLaunch(
                p(self.wm_plan),
                p(self.packed8),
                p(self.rawh8),
                p(w["w2"]),
                p(moe.k["w2_scales"]),
                p(w["shared_w2"]),
                p(moe.k["shared_w2_scales"]),
                p(self.l2_p),
                sp(),
            ),
            "m8 moe l2 local",
        )
        self._ck(
            self.l2.NexN25ProM8MoeL2TailLlLaunch(
                p(self.l2_p),
                p(self.moe_gen),
                p(self.moe_ll_peers),
                ctypes.c_ulonglong(self.moe_ll_mc),
                p(self.moe_ll_ctr),
                self.eng.rank,
                self.eng.world,
                p(self.moe_status),
                p(self.x8),
                sp(),
            ),
            "m8 moe l2 tail (LL peer-inbox)",
        )

    def _moe_layer(self, layer, moe):
        """Gate, top-k, route plan, PDL WM16 L1, tensor-core L2 and LL tail."""
        p, sp = self._p, self._sp
        pf_ptr, pf_bytes = self._front_pf.get(layer, (0, 0))
        self._ck(
            self.ggl.NexN25ProM8GateGemmPdlLaunch(
                p(self.x8),
                p(self._gate_pack(moe.k["gate_w"])),
                p(self.gate_logits8),
                p(self.rms8),
                ctypes.c_float(1e-6),
                ctypes.c_void_p(pf_ptr) if pf_ptr else None,
                ctypes.c_longlong(pf_bytes),
                sp(),
            ),
            "m8 gate gemm",
        )
        self._ck(
            self.gtl.NexN25ProM8GateTopKPackedKeyLaunch(
                p(self.gate_logits8),
                p(self.rms8),
                p(self.x8),
                p(moe.gamma),
                p(self.packed8),
                p(self.xn8),
                p(self.moe_xq8),
                p(self.moe_xs8),
                sp(),
            ),
            "m8 gate topk",
        )
        self._ck(
            self.wml.NexN25ProM8WmPlanLaunch(p(self.packed8), p(self.wm_plan), sp()),
            "m8 moe route plan",
        )
        w = self._wm_of(layer)
        wm_map = self._wm_map(w["w13"])
        self._ck(
            self.wml.NexN25ProM8Wm4L1Launch(
                p(self.wm_plan),
                p(self.packed8),
                p(self.xn8),
                p(w["w13"]),
                p(moe.k["w13_scales"]),
                p(w["shared_w13"]),
                p(moe.k["shared_scales"]),
                p(self.rawh8),
                p(self.moe_xq8),
                p(self.moe_xs8),
                wm_map.handle,
                sp(),
            ),
            "m8 moe wm l1",
        )
        self._moe_l2(layer, moe)

    def _tail(self):
        self._launch_sampling()

    # ---- the ONE unit --------------------------------------------------------
    def run(self, verify_dev=None, layer_probe=None):
        """One M=8 verification unit: eight ordered target rows, one decision.
        Fully device-driven; nothing host-visible is written inside this call and
        committed_len advances only through the fused tail CTA.

        `layer_probe(layer)`, EAGER-ONLY, is a host callback invoked after each
        layer's MoE launches so instrumentation observes the exact complete
        forward instead of re-deriving it. Never pass it under capture: a host
        callback would run at capture time, not replay time."""
        eng, p, sp = self.eng, self._p, self._sp
        if layer_probe is not None and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("layer_probe is eager-only; it cannot be captured")
        if verify_dev is not None:
            self.verify_ids.copy_(verify_dev)
        self._ck(
            self.embl.NexN25ProM8EmbedGatherLaunch(
                p(eng.embed_w), p(self.ctl), p(self.x8), p(self.cursor), sp()
            ),
            "m8 embed gather",
        )
        for layer in eng.layers:
            if eng.is_lin[layer]:
                self._gdn_layer(layer, eng.gdn[layer])
            else:
                self._attn_layer(eng.attn[layer])
            self._moe_layer(layer, eng.moe[layer])
            if self.dflash_taps is not None and layer in self.dflash_taps:
                self._ck(
                    self._dflash_tap_launch(p(self.x8), p(self.dflash_taps[layer]), sp()),
                    "m8 dflash feature tap",
                )
            if layer_probe is not None:
                layer_probe(layer)
        self._ck(
            self.hdl.NexN25ProM8FinalNormLaunch(
                p(self.x8), p(self.eng.head.norm_w), p(self.xn8), ctypes.c_float(self._eps), sp()
            ),
            "m8 final norm",
        )
        self._ck(
            self.hdl.NexN25ProM8LmHeadGemmTc16Launch(
                p(self.xn8), p(self.lm_head_pack()), p(self.logits8), sp()
            ),
            "m8 lm head",
        )
        self._tail()

    # ---- capture -------------------------------------------------------------
    def capture(self, warmup=3):
        """Warm up run() eagerly (stabilizes the allocator, builds every prepack,
        advances protocol generations), then capture ONE graph of the whole unit.
        Capture only records launches; the caller must re-seed (load) before
        replaying from a known state."""
        import torch.distributed as dist

        self.build_packs()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                self.run()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        dist.barrier()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.run()
        torch.cuda.synchronize()
        dist.barrier()
        return g

    # ---- inspection / handback ----------------------------------------------
    def status_max(self):
        import torch.distributed as dist

        vals = torch.tensor(
            [
                int(self.sampling_status.item()),
                int(self.op_status.item()),
                int(self.moe_status.item()),
                int(self.cursor_status.item()),
            ],
            device=self.dev,
        )
        dist.all_reduce(vals, op=dist.ReduceOp.MAX)
        return int(vals.max().item())

    def outputs(self):
        control = ctlabi.read_control_block(self.ctl)
        out = {
            "m8_control": control,
            "logits8_true": self.logits8,
            "xn8": self.xn8,
            "x8": self.x8,
            "cursor": self.cursor,
            "top_ids": self.sv_top_ids,
            "top_probs": self.sv_top_probs,
            "target_ids": self.sv_target_ids,
            "status": max(self.status_max(), control["status"]),
        }
        self.handback_bs1_views(control["committed_len"])
        return out

    def handback_bs1_views(self, committed_len):
        """Gather the ring state selected by committed_len into the BS1-shaped
        views (conv_state4[:, :3] / state_t) for external-oracle comparison.
        Runs only outside run(); never in a timed region."""
        for layer, gdn in self.eng.gdn.items():
            ring = self.conv_ring[layer]
            for k in range(3):
                gdn.conv_state4[:, k].copy_(ring[:, ctlabi.conv_slot(committed_len - 3 + k)])
            gdn.conv_state4[:, 3].zero_()
            gdn.state_t.copy_(self.state_ring[layer][ctlabi.state_slot(committed_len - 1)])


def build_m8(rank, world, dev, eng=None, *, dflash_tap_layers=None):
    if eng is None:
        from nex_n2_5_pro.decode_engine import DecodeEngine

        eng = DecodeEngine(rank, world, dev)
    return M8Verify(eng, dev, dflash_tap_layers=dflash_tap_layers)
