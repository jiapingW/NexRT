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

"""nexrt nex_n2_5_pro decode head (TP=8): final hidden state -> next token, CUDA-graph-safe.

Composes the three head kernel stages into one graph-capturable module:
    h_raw[4096] --GEMV(+RMS fold)--> logits_local[31040] fp32 + inv_rms
              --producer--> per-rank top-k candidates into a symmetric buffer (+arrive)
              --reducer(xinv_rms)--> up to 224 TRUE-logit candidates containing the global top-k
              --sampler(seed,position)--> token_id
lm_head is vocab-parallel (h replicated, vocab sharded 31040/rank) => NO tp-reduce on the GEMV; the only
cross-rank comm is the candidate gather inside the reducer (symm-mem barrier, graph-replay-safe: gen/parity
double-buffer, signed-spin, reset-free).

The 1-batch/M=4 paths lazily fold final norm.weight into
W_fused = (1+norm_w) (x) lm_head_w. M=8 keeps the raw shard for its explicit
final-norm kernel and shares that raw-weight tc16 pack with DFlash.

Run (parity vs torch head + perf):  torchrun --standalone --nproc_per_node=8 head.py
"""

import ctypes
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

HERE = Path(__file__).resolve().parent
KDIR = HERE / "kernels" / "head"
if __package__ in (None, ""):
    sys.path.insert(0, str(HERE.parent))

from nex_n2_5_pro.kernel_build import build_kernel  # noqa: E402
from nex_n2_5_pro.sampling import resolve_sampling_config, sampling_defaults, sampling_words  # noqa: E402
from nex_n2_5_pro.weights import load_final_norm, load_lm_head  # noqa: E402

DIM, ROWS, VOCAB = 4096, 31040, 248320
K, T, TOP_P, SEED, EPS = 20, 0.6, 0.95, 42, 1e-6


def build(name):
    return build_kernel(KDIR, name)


def _ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def _sp():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


class Head:
    def __init__(self, rank, dev, ext=None):
        self.rank, self.dev = rank, dev
        self.ext = ext
        vp, cf, ci, cu = ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_ulonglong
        g = ctypes.CDLL(str(build("nex_n2_5_pro_lmhead_gemv_ctypes")))
        g.NexN25ProLmHeadGemvLaunch.argtypes = [vp, vp, vp, vp, cf, vp]
        t = ctypes.CDLL(str(build("nex_n2_5_pro_lmhead_topk_ctypes")))
        t.NexN25ProLmHeadTopkProducerLaunch.argtypes = [vp] * 5 + [cu, vp, ci, vp, vp]
        t.NexN25ProLmHeadTopkReducerLaunch.argtypes = [vp] * 10
        for n in ("NexN25ProLmHeadTopkNC", "NexN25ProLmHeadTopkM", "NexN25ProLmHeadTopkK"):
            getattr(t, n).restype = ci
        s = ctypes.CDLL(str(build("nex_n2_5_pro_lmhead_sampler_ctypes")))
        s.NexN25ProLmHeadSamplerLaunch.argtypes = [vp, vp, vp, vp, vp, ci, vp, vp]
        s.NexN25ProLmHeadSamplerSlotLaunch.argtypes = [vp] * 6
        self.g, self.t, self.s = g, t, s
        self.NC, self.M, self.KK = (
            t.NexN25ProLmHeadTopkNC(),
            t.NexN25ProLmHeadTopkM(),
            t.NexN25ProLmHeadTopkK(),
        )
        self.n_cand = self.NC * self.KK  # candidates/rank
        self.out_n = 8 * self.M * self.KK  # reducer output (contains global top-k)

        # Raw row-major shard. In bridge mode this aliases sglang's verified
        # rank-local ParallelLMHead tensor, avoiding a second checkpoint read for that shard.
        # Explicit-norm consumers retain it; 1-batch/M=4 fold it lazily.
        if ext is not None:
            lw = ext["lm_head"]  # [31040,4096] bf16
            self.norm_w = ext["norm_w"]  # [4096]
        else:
            lw = load_lm_head(dev)[rank * ROWS : (rank + 1) * ROWS].contiguous()  # [31040,4096]
            self.norm_w = load_final_norm(dev)  # [4096]
        self.W_raw = lw if lw.is_contiguous() else lw.contiguous()
        self._raw_users = 0
        self._W_fused = None

        # --- fixed buffers (pre-allocated once; stable addresses for graph capture) ---
        self.logits = torch.empty(ROWS, dtype=torch.float32, device=dev)
        self.inv_rms = torch.zeros(1, dtype=torch.float32, device=dev)
        self.sampling_defaults = sampling_defaults()
        self.sampling_buffer = torch.tensor(
            sampling_words(SEED, *self.sampling_defaults), dtype=torch.int64, device=dev
        )
        # MTP also consumes this fixed seed field; this view owns no separate storage.
        self.seed_dev = self.sampling_buffer[:1]
        self.cand_val, self.peer_val, _ = self._symm(2 * self.n_cand, torch.float32)
        self.cand_id, self.peer_id, _ = self._symm(2 * self.n_cand, torch.int32)
        self.bar, _, self.bar_mc = self._symm(1, torch.int32)
        self.bar.zero_()
        if self.bar_mc == 0:
            raise RuntimeError("multicast barrier unavailable (NVLS required)")
        self.cur_gen = torch.zeros(1, dtype=torch.int32, device=dev)
        self.done_ctr = torch.zeros(1, dtype=torch.int32, device=dev)
        self.status = torch.zeros(1, dtype=torch.int32, device=dev)
        self.out_val = torch.zeros(self.out_n, dtype=torch.float32, device=dev)
        self.out_id = torch.zeros(self.out_n, dtype=torch.int32, device=dev)
        self.out_token = torch.zeros(1, dtype=torch.int32, device=dev)
        torch.cuda.synchronize()
        dist.barrier()  # all ranks' symm counters zeroed before any multicast arrival

    def retain_raw_weight(self):
        """Register an explicit-norm consumer and return the raw head shard."""
        if self.W_raw is None:
            raise RuntimeError(
                "Head.W_raw was released before an explicit-norm consumer registered"
            )
        self._raw_users += 1
        return self.W_raw

    @property
    def W_fused(self):
        """Return the lazily built final-norm-folded 1-batch/M=4 shard."""
        if self._W_fused is None:
            if self.W_raw is None:
                raise RuntimeError("Head.W_raw was released before W_fused was built")
            self._W_fused = (
                (self.W_raw.float() * (1.0 + self.norm_w.float())).to(torch.bfloat16).contiguous()
            )
            if self._raw_users == 0:
                self.W_raw = None
        return self._W_fused

    def _symm(self, numel, dtype):
        x = symm_mem.empty(numel, dtype=dtype, device=self.dev)
        h = symm_mem.rendezvous(x, dist.group.WORLD.group_name)
        peers = torch.tensor(
            list(h.buffer_ptrs), dtype=torch.int64, device=self.dev
        )  # 8 peer buffer ptrs
        return x, peers, int(getattr(h, "multicast_ptr", 0) or 0)

    def set_sampling_config(self, temperature=None, top_p=None, top_k=None, seed=None):
        """Update the next request's parameters without changing graph addresses."""
        values = resolve_sampling_config(self.sampling_defaults, temperature, top_p, top_k)
        words = sampling_words(0 if seed is None else seed, *values)
        start = 1 if seed is None else 0
        self.sampling_buffer[start:3].copy_(torch.tensor(words[start:3], dtype=torch.int64))

    def _logits_and_candidates(self, h_raw):
        self.g.NexN25ProLmHeadGemvLaunch(
            _ptr(h_raw),
            _ptr(self.W_fused),
            _ptr(self.logits),
            _ptr(self.inv_rms),
            ctypes.c_float(EPS),
            _sp(),
        )
        self._select_candidates()

    def _select_candidates(self):
        cu = ctypes.c_ulonglong
        self.t.NexN25ProLmHeadTopkProducerLaunch(
            _ptr(self.logits),
            _ptr(self.cand_val),
            _ptr(self.cand_id),
            _ptr(self.bar),
            _ptr(self.cur_gen),
            cu(self.bar_mc),
            _ptr(self.done_ctr),
            self.rank,
            _ptr(self.sampling_buffer),
            _sp(),
        )
        self.t.NexN25ProLmHeadTopkReducerLaunch(
            _ptr(self.peer_val),
            _ptr(self.peer_id),
            _ptr(self.bar),
            _ptr(self.out_val),
            _ptr(self.out_id),
            _ptr(self.status),
            _ptr(self.cur_gen),
            _ptr(self.inv_rms),
            _ptr(self.sampling_buffer),
            _sp(),
        )

    @torch.inference_mode()
    def forward(self, h_raw, position):
        """Write the sampled token for the supplied device position."""
        self._logits_and_candidates(h_raw)
        self.s.NexN25ProLmHeadSamplerLaunch(
            _ptr(self.out_val),
            _ptr(self.out_id),
            _ptr(self.sampling_buffer),
            _ptr(position),
            _ptr(self.out_token),
            1,
            _ptr(self.status),
            _sp(),
        )
        return self.out_token

    @torch.inference_mode()
    def forward_slot(self, h_raw, slot):
        """Select one token and advance slot=[token, S] exactly once in graph."""
        self._logits_and_candidates(h_raw)
        self._sample_slot(slot)

    def _sample_slot(self, slot):
        self.s.NexN25ProLmHeadSamplerSlotLaunch(
            _ptr(self.out_val),
            _ptr(self.out_id),
            _ptr(self.sampling_buffer),
            _ptr(slot),
            _ptr(self.status),
            _sp(),
        )

    @torch.inference_mode()
    def sample_prefill(self, logits, slot, prompt_length):
        """Sample the first output from retained prefill logits, including cache hits."""
        if logits.shape != (1, VOCAB):
            raise ValueError(f"expected prefill logits [1, {VOCAB}], got {tuple(logits.shape)}")
        self.logits.copy_(logits[0, self.rank * ROWS : (self.rank + 1) * ROWS])
        self.inv_rms.fill_(1)
        slot[1] = prompt_length - 1
        self._select_candidates()
        self._sample_slot(slot)
        # The retained session must not alias the slot modified during decode.
        token = slot[:1].to(torch.int64)
        if int(token[0]) < 0:
            raise RuntimeError("sampling failed: prefill logits have no finite winner")
        return token


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl")
    assert world == 8

    head = Head(rank, dev)
    head.set_sampling_config(T, TOP_P, K, seed=SEED)
    gen = torch.Generator(device=dev).manual_seed(2024)  # SAME on all ranks -> replicated h
    h_raw = torch.randn(DIM, dtype=torch.bfloat16, device=dev, generator=gen)
    position = torch.zeros(1, dtype=torch.int32, device=dev)

    # ---- torch reference head (this rank's vocab shard) ----
    xf = h_raw.float()
    h_normed = (xf * torch.rsqrt(xf.pow(2).mean() + EPS) * (1.0 + head.norm_w.float())).to(
        torch.bfloat16
    )
    lw_full = load_lm_head(dev)
    logits_local_ref = (
        h_normed.float() @ lw_full[rank * ROWS : (rank + 1) * ROWS].float().T
    )  # [31040] true
    gathered = [torch.empty(ROWS, dtype=torch.float32, device=dev) for _ in range(world)]
    dist.all_gather(gathered, logits_local_ref)
    logits_full = torch.cat(gathered)  # [248320] true logits
    ref_val, ref_id = torch.topk(logits_full, K)

    head.forward(h_raw, position)
    torch.cuda.synchronize()
    dist.barrier()

    # Reducer candidates -> top-20
    kv, kpos = torch.topk(head.out_val, K)
    kid = head.out_id[kpos]
    id_match = set(kid.tolist()) == set(ref_id.tolist())
    val_close = torch.allclose(kv.sort().values, ref_val.sort().values, atol=1e-2, rtol=1e-3)
    inv_ref = torch.rsqrt(xf.pow(2).mean() + EPS)
    inv_ok = abs(head.inv_rms.item() - inv_ref.item()) < 1e-4
    tok = int(head.out_token.item())
    in_top = tok in set(ref_id.tolist())

    # ---- perf: captured-graph latency via the local benchmark helper (hot back-to-back, TP-max) ----
    from nex_n2_5_pro.bench import bench_graph

    us = (
        bench_graph(
            lambda: head.forward(h_raw, position),
            warmup=50,
            rep=2000,
            flush_l2=False,
            tp_reduce=True,
            device=str(dev),
        ).median_ms
        * 1e3
    )

    st = torch.tensor([int(head.status.item())], device=dev)
    dist.all_reduce(st, op=dist.ReduceOp.MAX)
    if rank == 0:
        okid = id_match and val_close and inv_ok and in_top and st.item() == 0
        print("== nexrt head (TP=8): GEMV->topk->sampler vs torch head ==", flush=True)
        print(f"   top-20 ids match torch : {id_match}", flush=True)
        print(f"   top-20 values close    : {val_close}", flush=True)
        print(
            f"   inv_rms match          : {inv_ok}  (kernel {head.inv_rms.item():.5f})", flush=True
        )
        print(
            f"   sampled token in top-20: {in_top}  (token={tok})  status={st.item()}", flush=True
        )
        print(
            f"   [{'OK' if okid else 'CHECK'}]  head latency (captured graph): {us:.2f} us/token",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
