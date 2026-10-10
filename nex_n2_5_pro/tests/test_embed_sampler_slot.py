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

"""Unit tests for the two decode-loop bookend pieces:
  (1) embedding gather   -- token_id from slot[0] -> x[4096] == embed_w[token] (pure index-select).
  (2) sampler slot form  -- NexN25ProLmHeadSamplerSlotLaunch reads position=slot[1], writes slot[0]=sampled id,
                            bumps slot[1]=S+1; the sampled id must equal the by-value sampler at position=S.

Run with one GPU:  python3 nex_n2_5_pro/tests/test_embed_sampler_slot.py
"""

import ctypes
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from kernel_build import build_kernel  # noqa: E402
from sampling import sampling_words  # noqa: E402

KH = HERE / "kernels" / "head"
KE = HERE / "kernels" / "embed"
HID, NCAND = 4096, 8 * 28


def build(kdir, name):
    return build_kernel(kdir, name)


def main():
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    vp, ci = ctypes.c_void_p, ctypes.c_int
    p = lambda t: ctypes.c_void_p(t.data_ptr())
    sp = lambda: ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    bad = 0

    # ---- (1) embedding gather ----
    emb = ctypes.CDLL(str(build(KE, "nex_n2_5_pro_embed_gather_ctypes")))
    emb.NexN25ProEmbedGatherLaunch.argtypes = [vp] * 4
    VOCAB = 2000
    embed_w = (torch.randn(VOCAB, HID, dtype=torch.bfloat16) * 0.1).contiguous()
    x = torch.empty(HID, dtype=torch.bfloat16)
    for token in (-1, 0, 1, 777, 1999):
        slot = torch.tensor([token, 12345], dtype=torch.int32)  # [token_id, S]
        x.zero_()
        emb.NexN25ProEmbedGatherLaunch(p(embed_w), p(slot), p(x), sp())
        torch.cuda.synchronize()
        d = (x.float() - embed_w[max(token, 0)].float()).abs().max().item()
        ok = d == 0.0
        bad += not ok
        print(f"embed  token={token:5d}  max_abs_vs_row={d:.3e}  {'OK' if ok else 'MISMATCH'}")

    # ---- (2) sampler slot form vs by-value ----
    smp = ctypes.CDLL(str(build(KH, "nex_n2_5_pro_lmhead_sampler_ctypes")))
    smp.NexN25ProLmHeadSamplerLaunch.argtypes = [vp, vp, vp, vp, vp, ci, vp, vp]
    smp.NexN25ProLmHeadSamplerSlotLaunch.argtypes = [vp] * 6
    cand_val = (torch.randn(NCAND, dtype=torch.float32) * 3.0).contiguous()
    cand_id = torch.randperm(200000)[:NCAND].to(torch.int32).contiguous()
    config = torch.tensor(sampling_words(42, 0.6, 0.95, 20), dtype=torch.int64)
    status = torch.zeros(1, dtype=torch.int32)
    for S in (100, 1024, 16000):
        out_bv = torch.zeros(1, dtype=torch.int32)
        pos = torch.tensor([S], dtype=torch.int32)  # by-value position buffer
        smp.NexN25ProLmHeadSamplerLaunch(
            p(cand_val), p(cand_id), p(config), p(pos), p(out_bv), ci(1), p(status), sp()
        )
        slot = torch.tensor([0, S], dtype=torch.int32)  # [token_id, S]
        smp.NexN25ProLmHeadSamplerSlotLaunch(
            p(cand_val), p(cand_id), p(config), p(slot), p(status), sp()
        )
        torch.cuda.synchronize()
        tok_ok = int(slot[0].item()) == int(out_bv[0].item())
        seq_ok = int(slot[1].item()) == S + 1
        ok = tok_ok and seq_ok and int(status[0]) == 0
        bad += not ok
        print(
            f"sampler S={S:6d}  slot=[tok {int(slot[0])}, S {int(slot[1])}]  by-value tok {int(out_bv[0])}  "
            f"token{'=' if tok_ok else '!='}  S+1{'=' if seq_ok else '!='}  {'OK' if ok else 'MISMATCH'}"
        )

    print("ALL OK" if bad == 0 else f"{bad} FAILURE(S)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
