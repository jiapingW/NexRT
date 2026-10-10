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

"""TP8 M4/M8 sampling, prefix commit, and replay checks against a CPU oracle.

Run: torchrun --standalone --nproc_per_node=8 -m nex_n2_5_pro.tests.test_speculative_sampling_gpu
"""

import os
import struct
from types import SimpleNamespace

import torch
import torch.distributed as dist

from nex_n2_5_pro.decode_engine import _p, _sp
from nex_n2_5_pro.m4_verify import M4Verify
from nex_n2_5_pro.m8_verify import M8Verify
from nex_n2_5_pro.tests.test_sampling_gpu import splitmix64

VOCAB, LOCAL = 248320, 31040


def row_reference(values, ids, k, t, p, seed, pos):
    if t == 0 or k == 1:
        return int(ids[0])
    v = values[:k].double()
    probabilities = ((v - v[0]) / t).softmax(0)
    keep = probabilities.flip(0).cumsum(0).flip(0) > 1 - p
    keep[0] = True
    probabilities *= keep
    uniform = (splitmix64(seed ^ splitmix64(pos & 0xFFFFFFFF)) >> 40) / (1 << 24)
    return int(ids[(probabilities.cumsum(0) >= uniform * probabilities.sum()).nonzero()[0, 0]])


def make_tail(rank, m):
    cls = M4Verify if m == 4 else M8Verify
    unit = cls.__new__(cls)
    unit.dev = torch.device("cuda", rank)
    unit.eng = SimpleNamespace(rank=rank, slot=torch.zeros(2, dtype=torch.int32, device=unit.dev))
    unit._p, unit._sp = _p, _sp
    unit.ctl = torch.zeros(32, dtype=torch.int32, device=unit.dev)
    unit.mirror_bs1_slot = True
    names = (
        ("ws_status", "ws_moe_status", "ws12_status", "ws21_status", "cursor_status")
        if m == 4
        else ("op_status", "moe_status", "cursor_status")
    )
    for name in names:
        setattr(unit, name, torch.zeros(1, dtype=torch.int32, device=unit.dev))
    setattr(unit, f"logits{m}", torch.zeros(m, LOCAL, device=unit.dev))
    unit._init_sampling()
    return unit


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    generator = torch.Generator().manual_seed(1729)
    summary = []
    for m in (4, 8):
        unit = make_tail(rank, m)
        unit._launch_sampling()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            unit._launch_sampling()
        address = unit.sampling_buffer.data_ptr()
        unit.sampling_buffer[3:].fill_(123456)
        cases = 0
        for pattern in ("random", "ties", "concentrated", "sparse", "top_p_boundary"):
            logits = torch.randn(m, VOCAB, generator=generator)
            if pattern == "ties":
                logits.zero_()
                logits[:, 1::2] = -0.0
            elif pattern == "concentrated":
                logits[:, :60] = torch.arange(60, 0, -1) * 0.01 + 20
            elif pattern == "sparse":
                logits.fill_(-float("inf"))
                logits[:, [1, LOCAL + 7, VOCAB - 1]] = torch.tensor([3.0, 2.0, 1.0])
            elif pattern == "top_p_boundary":
                logits.fill_(-float("inf"))
                logits[:, :16] = 1.0
            ids = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :28]
            values = logits.gather(1, ids)
            getattr(unit, f"logits{m}").copy_(logits[:, rank * LOCAL : (rank + 1) * LOCAL])
            policies = [
                (k, t, p)
                for k in (28, 1, 20, 2, *range(3, 20), *range(21, 28))
                for t, p in ((0.6, 0.95), (0.0, 1.0))
            ]
            policies += [(28, 0.6, 0.001), (28, 0.6, 1.0), (20, 1e-30, 0.95)]
            if pattern == "top_p_boundary":
                policies = [(28, 0.6, 15 / 16)]
            for k, t, p in policies:
                seed, pos = (1 << 63) + cases * 197, 8192 + cases
                targets = [
                    row_reference(values[r], ids[r], k, t, p, seed, pos + r) for r in range(m)
                ]
                depth = cases % m
                verify = [123] + targets[:-1]
                if depth < m - 1:
                    verify[depth + 1] = (targets[depth] + 1) % VOCAB
                control = torch.zeros(32, dtype=torch.int32)
                control[0], control[1] = verify[0], pos
                control[4 : 4 + m] = torch.tensor(verify)
                unit.ctl.copy_(control)
                unit.set_sampling_config(t, p, k, seed=seed)
                graph.replay()
                actual, ctl = unit.sv_target_ids.cpu().tolist(), unit.ctl.cpu().tolist()
                assert actual == targets, (m, pattern, k, t, targets, actual)
                assert ctl[1] == pos + depth + 1 and ctl[2:4] == [depth + 1] * 2, (m, ctl)
                assert ctl[0] == targets[depth]
                assert ctl[4 + m : 4 + m + depth + 1] == targets[: depth + 1]
                assert ctl[4 + 2 * m] == depth and ctl[6 + 2 * m] == 0
                exported = unit.sv_top_ids.cpu()
                effective_k = 1 if t == 0 else k
                assert torch.equal(exported[:, :effective_k], ids[:, :effective_k].int())
                probs = unit.sv_top_probs.cpu()
                assert bool(torch.isfinite(probs).all())
                torch.testing.assert_close(probs.sum(-1), torch.ones(m), atol=2e-6, rtol=0)
                if pattern == "top_p_boundary":
                    assert bool(((probs > 0).sum(-1) == 15).all())
                    torch.testing.assert_close(
                        probs[:, :15], torch.full((m, 15), 1 / 15), atol=2e-7, rtol=0
                    )
                if m == 4:
                    from nex_n2_5_pro.mtp.sampling_ref import verify_round_from_export

                    resolution = verify_round_from_export(
                        exported, probs, actual, verify, seed=seed, committed_len=pos, temperature=t
                    )
                    assert resolution.commit_count == depth + 1
                gathered = [None] * 8
                dist.all_gather_object(gathered, (actual, ctl))
                assert all(x == gathered[0] for x in gathered)
                assert unit.sampling_buffer.data_ptr() == address
                assert bool((unit.sampling_buffer[3:] == 123456).all())
                cases += 1
        from nex_n2_5_pro import m4_control, m8_control

        abi = m4_control if m == 4 else m8_control
        invalid_configs = {
            "bad_k": (0.6, 0.95, 100),
            "zero_k": (0.0, 1.0, 0),
            "nan_t": (float("nan"), 0.95, 20),
            "negative_t": (-0.6, 0.95, 20),
            "zero_p": (0.6, 0.0, 20),
            "large_p": (0.6, 1.1, 20),
            "nan_p": (0.6, float("nan"), 20),
        }
        poisons = (
            "upstream",
            "nan",
            "negative_nan",
            "inf",
            "empty",
            "sticky",
            "combined",
            *invalid_configs,
        )
        for poison, temperature in ((name, t) for t in (0.0, 0.6) for name in poisons):
            unit.ctl.copy_(control)
            unit.sampling_status.zero_()
            unit.cursor_status.zero_()
            local = getattr(unit, f"logits{m}")
            local.zero_()
            unit.set_sampling_config(temperature, 0.95, 20)
            expected_status = abi.STATUS_BAD_RESOLUTION
            if poison == "upstream":
                expected_status = abi.STATUS_UPSTREAM_FAILED
            elif poison == "sticky" or poison in invalid_configs:
                expected_status = abi.STATUS_BAD_SAMPLING_CONFIG
            elif poison == "combined":
                expected_status = abi.STATUS_UPSTREAM_FAILED | abi.STATUS_BAD_SAMPLING_CONFIG
            if rank == 3:
                if poison in ("upstream", "combined"):
                    unit.cursor_status.fill_(1)
                elif poison == "sticky":
                    unit.sampling_status.fill_(abi.STATUS_BAD_SAMPLING_CONFIG)
                elif poison in invalid_configs:
                    pass  # Configuration is replicated below.
                else:
                    local[0, 0] = {
                        "nan": float("nan"),
                        "negative_nan": -float("nan"),
                        "inf": float("inf"),
                        "empty": 0,
                    }[poison]
            if poison == "empty":
                local.fill_(-float("inf"))
            if poison in invalid_configs or poison == "combined":
                t, p, k = invalid_configs.get(poison, invalid_configs["bad_k"])
                unit.sampling_buffer[1] = struct.unpack("<q", struct.pack("<ff", t, p))[0]
                unit.sampling_buffer[2] = k
            before_slot = unit.eng.slot.clone()
            graph.replay()
            ctl = unit.ctl.cpu().tolist()
            assert ctl[1] == pos and ctl[2:4] == [0, 0], (m, poison, ctl)
            assert ctl[6 + 2 * m] == expected_status, (m, poison, ctl)
            assert unit.sampling_status.item() == expected_status
            assert bool((unit.sv_target_ids == -1).all())
            assert torch.equal(unit.eng.slot, before_slot)
            gathered = [None] * 8
            dist.all_gather_object(gathered, ctl)
            assert all(x == gathered[0] for x in gathered)
        if m == 8:
            from nex_n2_5_pro.m8_cp import M8CpVerify

            # CP errors participate only in the CP arm's collective.
            cp = M8CpVerify.__new__(M8CpVerify)
            cp.__dict__.update(unit.__dict__)
            cp_status = torch.zeros(1, dtype=torch.int32, device=unit.dev)
            cp.cp_sampling_upstream = unit.sampling_upstream.clone()
            cp.cp_sampling_upstream[4] = cp_status.data_ptr()
            unit.sampling_status.zero_()
            unit.cursor_status.zero_()
            unit.set_sampling_config(0.0, 1.0, 28)
            getattr(unit, f"logits{m}").zero_()
            unit.ctl.copy_(control)
            if rank == 3:
                cp_status.fill_(1)
            cp.use_cp = False
            non_cp_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(non_cp_graph):
                cp._tail()
            non_cp_graph.replay()
            assert unit.ctl[6 + 2 * m].item() == 0
            unit.ctl.copy_(control)
            cp.use_cp = True
            cp_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cp_graph):
                cp._tail()
            cp_graph.replay()
            ctl = unit.ctl.cpu().tolist()
            assert ctl[1] == pos and ctl[2] == 0
            assert ctl[6 + 2 * m] == abi.STATUS_UPSTREAM_FAILED
            cp_status.zero_()
        # A new request clears sticky failure state without resetting epochs.
        unit.sampling_status.zero_()
        unit.cursor_status.zero_()
        unit.set_sampling_config(0.0, 1.0, 28)
        getattr(unit, f"logits{m}").zero_()
        unit.ctl.copy_(control)
        if rank == 3:
            torch.cuda._sleep(2000000)
        graph.replay()
        assert unit.ctl[6 + 2 * m].item() == 0
        assert bool((unit.sv_target_ids == 0).all())
        # Preserve the existing state-validation hook for every commit depth.
        unit.use_forced_commit = True
        unit.forced_commit.fill_(1)
        forced_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(forced_graph):
            unit._launch_sampling()
        for count in (*range(1, m + 1), 0, m + 1):
            unit.forced_commit.fill_(count)
            unit.ctl.copy_(control)
            forced_graph.replay()
            ctl = unit.ctl.cpu().tolist()
            if 1 <= count <= m:
                assert ctl[1] == pos + count and ctl[2] == count and ctl[6 + 2 * m] == 0
            else:
                assert ctl[1] == pos and ctl[2] == 0
                assert ctl[6 + 2 * m] == abi.STATUS_BAD_COMMIT_COUNT
        summary.append(
            dict(
                m=m,
                cases=cases,
                poison_cases=2 * len(poisons) + (m == 8),
                policy_graph_capture_count=1,
                forced_counts=m + 2,
                delayed_rank="pass",
                status="pass",
            )
        )
        if rank == 0:
            print(summary[-1], flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
