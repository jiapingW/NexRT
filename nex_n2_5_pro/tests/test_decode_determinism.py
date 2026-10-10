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

"""Repeat a saved prefill handoff through one complete TP8 decode mode.

Run from the repository root on the GPU host (repeat for each mode and --cp):
    NEXRT_NLAYERS=60 NEXRT_MOE_RAW=1 torchrun --standalone --nproc_per_node=8 \
        -m nex_n2_5_pro.tests.test_decode_determinism \
        --mode dflash --state-dir /path/to/prefill_state

Uses the existing offline weight/state loaders. MTP also needs mtp_handoff.pt;
DFlash needs NEXRT_DFLASH_DIR. This tests decode repeatability, not SGLang
prefill repeatability or agreement with another decode implementation.
"""

import argparse
import hashlib
import os
from pathlib import Path

import torch
import torch.distributed as dist

from nex_n2_5_pro.decode_engine import DecodeEngine


def fingerprint(tensor):
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("bs1", "mtp", "dflash"), required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--cp", action="store_true")
    parser.add_argument("--rounds", type=int, default=32)
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error("--rounds must be at least 2")

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    assert world == 8, "requires TP8"
    engine = DecodeEngine(rank, world, device, cp=args.cp)
    assert len(engine.layers) == 60, "set NEXRT_NLAYERS=60 for a complete model"
    verifier = block = None

    if args.mode == "mtp":
        from nex_n2_5_pro import m4_control as control
        from nex_n2_5_pro.m4_verify import M4VerifyCp, M4VerifyNonCP
        from nex_n2_5_pro.mtp.block import MtpDraft

        verifier = (M4VerifyCp if args.cp else M4VerifyNonCP)(engine, device)
        block = MtpDraft(engine, verifier, device, eos_ids=[], sampling=True)
        hand = torch.load(args.state_dir / "mtp_handoff.pt", map_location="cpu", weights_only=False)
        first_token = int(hand["t0"])
        step = block.round
        logits = verifier.logits4
    elif args.mode == "dflash":
        from nex_n2_5_pro import m8_control as control
        from nex_n2_5_pro.dflash.block import DFlashBlock
        from nex_n2_5_pro.dflash.config import DFlashConfig
        from nex_n2_5_pro.m8_cp import M8CpVerify

        config = DFlashConfig.load()
        verifier = M8CpVerify(
            engine, device, use_cp=args.cp, dflash_tap_layers=config.target_layer_ids
        )
        block = DFlashBlock.from_checkpoint(engine, verifier, device, config=config)
        _, first_token = engine.load_prefill(args.state_dir)
        step = block.round
        logits = verifier.logits8
    else:
        step = engine.step_cp if args.cp else engine.step
        logits = engine.head.logits

    def restore(temperature, seed):
        engine.head.set_sampling_config(temperature, 0.95, 20, seed=seed)
        if args.mode == "bs1":
            length, _ = engine.load_prefill(args.state_dir)
        elif args.mode == "mtp":
            length = verifier.load(str(args.state_dir), [first_token, int(hand["d1"]), 0, 0])
            assert length == hand["L"]
            block.seed(
                hand["k"][rank // 4].to(device),
                hand["v"][rank // 4].to(device),
                length,
                hand["y_sel_raw"].to(device),
            )
        else:
            length = block.load_request(args.state_dir, first_token)
        if verifier is not None:
            verifier.set_sampling_config(temperature, 0.95, 20, seed=seed)
        # Leave enough space even when every speculative row is accepted.
        width = 1 if verifier is None else control.M
        capacity = next(iter(engine.attn.values())).k_cache.shape[0]
        if block is not None and args.mode == "dflash":
            capacity = min(capacity, block.cache_capacity, config.max_position_embeddings)
        assert length + (args.rounds + 7) * width <= capacity, "insufficient KV headroom"
        torch.cuda.synchronize()
        dist.barrier()

    def trajectory(graph):
        records = []
        good = True
        for _ in range(args.rounds):
            graph.replay()
            if verifier is None:
                token, length = engine.slot.tolist()
                event = (token, length)
                good &= token >= 0 and engine.status_max() == 0
            else:
                snapshot = control.read_control_block(verifier.ctl)
                count = int(snapshot["commit_count"])
                good &= (
                    snapshot["status"] == 0
                    and 1 <= count <= control.M
                    and snapshot["output_count"] == count
                )
                event = (snapshot["committed_len"], count, tuple(snapshot["emitted_ids"][:count]))
            good &= bool(torch.isfinite(logits).all().item())
            records.append((event, fingerprint(logits)))

        # Compare each rank with its own prior run. Unused KV capacity and live
        # communication epochs are not model state and must not be compared/reset.
        if verifier is None:
            length = int(engine.slot[1].item())
            states = [engine.x]
            for layer in engine.gdn.values():
                states.extend((layer.conv_state4, layer.state_t))
        else:
            length = int(verifier.ctl[control.COMMITTED_LEN].item())
            states = [verifier.ctl]
            for layer in engine.gdn:
                states.extend((verifier.conv_ring[layer], verifier.state_ring[layer]))
            if args.mode == "mtp":
                states.extend(
                    (block.y_sel, block.attn.k_cache[:length], block.attn.v_cache[:length])
                )
            else:
                good &= block.status() == 0
                states.extend(verifier.dflash_taps.values())
                from nex_n2_5_pro.dflash.config import SLIDING_LAYERS

                # The last accepted target features enter draft KV on the next
                # round. Compare live context, excluding provisional noise rows
                # and inactive physical slots left by earlier requests.
                context_end = length - int(verifier.ctl[control.COMMIT_COUNT].item())
                for layer, (key, value) in enumerate(
                    zip(block.state.key_cache, block.state.value_cache)
                ):
                    start = block.base_position
                    if layer < SLIDING_LAYERS:
                        start = max(start, context_end - config.sliding_window)
                    indices = torch.arange(start, context_end, device=device) % key.shape[0]
                    states.extend((key.index_select(0, indices), value.index_select(0, indices)))
        for layer in engine.attn.values():
            states.extend((layer.k_cache[:length], layer.v_cache[:length]))
        return (records, [fingerprint(state) for state in states]), good

    references = {}
    failures = 0
    # Different capture warmups advance the live transport epochs. Interleave
    # seeds, then repeat them: a request must not inherit another request's RNG.
    cases = ((0.0, 0), (0.6, 20261008), (0.6, (1 << 63) + 123))
    for warmup in (1, 7):
        restore(*cases[0])
        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        for repeat in range(2):
            for case in cases:
                restore(*case)
                result, good = trajectory(graph)
                if case not in references:
                    references[case] = result
                failures += int(not good or result != references[case])
                # Check emitted decisions across ranks separately from floating
                # intermediates, which may use a rank-specific fixed sum order.
                events = [record[0] for record in result[0]]
                all_events = [None] * world
                dist.all_gather_object(all_events, events)
                failures += int(any(other != events for other in all_events))
                if rank == 0:
                    print(
                        f"{args.mode} cp={args.cp} warmup={warmup} repeat={repeat} "
                        f"temperature={case[0]} seed={case[1]} checked",
                        flush=True,
                    )

    result = torch.tensor([failures], device=device)
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    failures = int(result.item())
    if rank == 0:
        print(f"decode determinism: 12 trajectories/rank, failures={failures}", flush=True)
    dist.destroy_process_group()
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
