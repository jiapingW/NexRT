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

"""Three-way fail-closed graph-family isolation: BS=1 / M=4 / M=8.

Captures, on ONE shared engine: the 1-batch step graph, the production M=4
verify-only graph, and the M=8 unit graph. Each
graph's actual profiled kernel symbols are audited:

  * the BS=1 graph must contain its positive sentinels and ZERO M4/MTP/M8
    kernels;
  * the M=4 verify graph must contain its positive sentinels and ZERO 1-batch
    production kernels and ZERO M8 kernels;
  * the M=8 graph must contain its positive M8 sentinels and ZERO 1-batch and
    ZERO M4/MTP kernels;
  * an EMPTY or unrecognized profile FAILS -- nothing passes by absence.

Also audits pointer disjointness: no M8 mutable buffer may alias any engine or
M4 mutable buffer (weights/KV are intentionally shared storage).

Run:  NEXRT_NLAYERS=4 torchrun --standalone --nproc_per_node=8 \
          tests/test_m8_graph_arms.py --state-dir states/prefill_state_60_512
"""

import argparse
import os
import re
import sys
from pathlib import Path

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

# 1-batch production kernels (word-boundary anchored; the M4/M8 prefixes remove
# the boundary, so clones can never match).
BS1_KERNELS = [
    r"\bQkvGroupSplitK",
    r"\bNormRopeFusedKernel",
    r"\bNormRopeFusedCpKernel",
    r"\bwgmma_fd4_mega",
    r"\bOProjFusedPush",
    r"\bOProjPartialSlot",
    r"\bWholeVectorReduceKernel",
    r"\bGateGemvRowRawAndRmsCta0Kernel",
    r"\bGateTopKSingleCtaNoLogit",
    r"\bL1K2PairPacked64",
    r"\bFusedL2Reduce",
    r"\bLmHeadGemvRow",
    r"\bEmbedGatherKernel",
    r"\bResidualAddKernel",
    r"\bLinearFrontSplitCta",
    r"\bRecurrentCore",
    r"\bLinearFinalOProj",
    r"\bDupQOnlyGroupSplitKKernel",
    r"\bProducerKernel",
    r"\bReducerKernel",
    r"\bSamplerKernel",
]
# M4/MTP family markers (spelling markers plus marker-free family names).
M4_MTP_KERNELS = [
    r"M4",
    r"Mtp",
    r"m4_",
    r"mtp_",
    r"\bStageCpSlotsKernel",
    r"\bFoldCpStatusKernel",
    r"\bcp_ll_",
    r"\bMultiRow",
    r"\bquad",
    r"\bQuad",
]
# M8 family markers.
M8_KERNELS = [r"M8", r"m8_"]


BS1_SENTINELS = {
    r"\bEmbedGatherKernel": 1,
    r"\bFusedL2ReducePushKernel": 1,
    r"\bLinearFrontSplitCtaSk4Kernel": 1,
}
M4_SENTINELS = {
    r"\bM4AttnQkvKernel": 1,
    r"\bM4LinearFrontKernel": 1,
    r"\bM4ProducerKernel": 1,
    r"\bM4ReducerKernel": 1,
    r"\bM4SamplerKernel": 1,
    r"\bM4MoeL2ProducerH32XKernel": 1,
}
M8_SENTINELS = {
    r"\bM8EmbedGatherKernel": 1,
    r"\bM8GdnFrontFt16Sk2Kernel": 1,
    r"\bm8_attn_partial_splitkv_dyn": 1,
    r"\bM8MoeWm4L1Kernel": 1,
    r"\bM8MoeL2LocalKernel": 1,
    r"\bM8MoeL2TailLlKernel": 1,
    r"\bM8ProducerKernel": 1,
    r"\bM8ReducerKernel": 1,
    r"\bM8SamplerKernel": 1,
    r"\bM8LmHeadBf16TcKernel": 1,
}


def capture_graph(fn, warmup=3):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    return g


def kernel_agg(g, rank, reps=3):
    for _ in range(2):
        g.replay()
    torch.cuda.synchronize()
    dist.barrier()
    if rank != 0:
        for _ in range(reps):
            g.replay()
        torch.cuda.synchronize()
        return {}
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            g.replay()
        torch.cuda.synchronize()
    agg = {}
    for evt in prof.key_averages():
        t = float(evt.self_device_time_total)
        if t > 0.0:
            agg[evt.key] = evt.count / reps
    return agg


def count(agg, pattern):
    return sum(c for k, c in agg.items() if re.search(pattern, k))


def check_family(name, agg, positives, absents_lists):
    bad = []
    if not agg:
        bad.append(f"{name}: empty kernel profile (no kernels captured)")
        return bad
    for pat, lo in positives.items():
        c = count(agg, pat)
        if c < lo:
            bad.append(f"{name}: missing sentinel {pat} (saw {c:.1f})")
    for label, pats in absents_lists:
        for pat in pats:
            c = count(agg, pat)
            if c > 0:
                bad.append(f"{name}: forbidden {label} kernel {pat} (saw {c:.1f})")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default="states/prefill_state_60_512")
    cli = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl")
    assert world == 8

    from decode_engine import DecodeEngine
    from m8_verify import M8Verify

    eng = DecodeEngine(rank, world, dev)
    state_dir = HERE / cli.state_dir

    from m4_verify import M4VerifyNonCP

    m4 = M4VerifyNonCP(eng, dev)
    m4.enable_sampling(eng.head.seed_dev)

    m8 = M8Verify(eng, dev)
    m8.set_seed(11)

    # ---- pointer disjointness: M8 mutable state vs engine/M4 mutable state ----
    m8_ptrs = {}
    for nm in (
        "ctl",
        "x8",
        "logits8",
        "front_sk2_ws",
        "front_sk2_tickets",
        "wm_plan",
        "l2_p",
        "op_rep",
        "op_it",
        "op_status",
        "moe_ll_inbox",
        "moe_ll_ctr",
        "moe_gen",
        "moe_status",
        "sampling_values",
        "sampling_barrier",
        "sampling_ids",
        "cursor",
        "rms8",
        "packed8",
    ):
        m8_ptrs[nm] = int(getattr(m8, nm).data_ptr())
    for L, r in m8.conv_ring.items():
        m8_ptrs[f"conv_ring{L}"] = int(r.data_ptr())
    for L, r in m8.state_ring.items():
        m8_ptrs[f"state_ring{L}"] = int(r.data_ptr())
    other_ptrs = {
        "eng.slot": int(eng.slot.data_ptr()),
        "eng.x": int(eng.x.data_ptr()),
        "m4.ctl": int(m4.ctl.data_ptr()),
        "m4.x4": int(m4.x4.data_ptr()),
        "m4.logits4": int(m4.logits4.data_ptr()),
        "m4.sampling_values": int(m4.sampling_values.data_ptr()),
        "m4.ws12_rep": int(m4.ws12_rep.data_ptr()),
    }
    collisions = [(a, b) for a, pa in m8_ptrs.items() for b, pb in other_ptrs.items() if pa == pb]
    alias_bad = [] if not collisions else [f"pointer aliasing: {collisions}"]

    # ---- captures on the shared engine ----
    eng.load_prefill(state_dir=state_dir)
    g_bs1 = eng.capture(warmup=3)

    m4.load(state_dir, [3, 4, 5, 6])
    g_m4 = capture_graph(m4.run)

    m8.load(state_dir, [3, 4, 5, 6, 7, 8, 9, 10])
    g_m8 = m8.capture(warmup=3)

    # ---- profile + audit ----
    bad = list(alias_bad)
    agg = kernel_agg(g_bs1, rank)
    if rank == 0:
        bad += check_family(
            "bs1", agg, BS1_SENTINELS, [("m4/mtp", M4_MTP_KERNELS), ("m8", M8_KERNELS)]
        )
    agg = kernel_agg(g_m4, rank)
    if rank == 0:
        bad += check_family(
            "m4-verify", agg, M4_SENTINELS, [("bs1", BS1_KERNELS), ("m8", M8_KERNELS)]
        )
    agg = kernel_agg(g_m8, rank)
    if rank == 0:
        bad += check_family(
            "m8", agg, M8_SENTINELS, [("bs1", BS1_KERNELS), ("m4/mtp", M4_MTP_KERNELS)]
        )

    fail = torch.tensor([1 if bad else 0], device=dev)
    dist.all_reduce(fail, op=dist.ReduceOp.MAX)
    if rank == 0:
        for b in bad:
            print(f"   [SYMBOL-FAIL] {b}", flush=True)
        print(
            f"== M8 three-way graph-family isolation "
            f"({len(eng.layers)}L): {'PASS' if fail.item() == 0 else 'FAIL'} ==",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    if fail.item() != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
