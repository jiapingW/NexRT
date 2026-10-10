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

"""TP8 greedy/top-k sampling against a CPU reference, without model weights.

Run: torchrun --standalone --nproc_per_node=8 -m nex_n2_5_pro.tests.test_sampling_gpu
"""

import os

import torch
import torch.distributed as dist

from nex_n2_5_pro.head import Head, ROWS, VOCAB


def splitmix64(value):
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def reference(logits, top_k, temperature, top_p, seed, position):
    if temperature == 0 or top_k == 1:
        return int(logits.argmax())
    ids = torch.argsort(logits, descending=True, stable=True)[:top_k]
    values = logits[ids].double()
    probabilities = ((values - values.max()) / temperature).softmax(0)
    tail = probabilities.flip(0).cumsum(0).flip(0)
    keep = tail > 1 - top_p
    keep[0] = True
    probabilities *= keep
    uniform = (splitmix64(seed ^ splitmix64(position & 0xFFFFFFFF)) >> 40) / (1 << 24)
    cdf = probabilities.cumsum(0)
    return int(ids[(cdf >= uniform * probabilities.sum()).nonzero()[0, 0]])


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device)
    assert dist.get_world_size() == 8
    # Tail-only correctness: the weight shard is never read.
    head = Head(
        rank,
        device,
        ext={
            "lm_head": torch.empty((ROWS, 4096), dtype=torch.bfloat16, device=device),
            "norm_w": torch.zeros(4096, dtype=torch.bfloat16, device=device),
        },
    )
    slot = torch.tensor([0, 100], dtype=torch.int32, device=device)
    gathered = torch.empty(16, dtype=torch.int32, device=device)
    head.logits.zero_()
    head.inv_rms.fill_(1)
    head._select_candidates()
    head._sample_slot(slot)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        head._select_candidates()
        head._sample_slot(slot)
    buffer_address = head.sampling_buffer.data_ptr()
    head.sampling_buffer[3:].fill_(12345)
    generator = torch.Generator().manual_seed(20260914)
    cases = 0
    configurations = [
        (k, t, p)
        for k in (28, 1, 20, 2, *range(3, 20), *range(21, 28))
        for t, p in ((0.6, 0.95), (0.0, 1.0))
    ]
    configurations += [(28, 0.6, 0.001), (28, 0.6, 1), (20, 1e-30, 0.95), (28, 10, 0.95)]
    for pattern in ("random", "ties", "concentrated", "sparse"):
        logits = torch.randn(VOCAB, generator=generator)
        if pattern == "ties":
            logits.zero_()
            logits[1::2] = -0.0
        elif pattern == "concentrated":
            logits[:60] = torch.arange(60, 0, -1) * 0.01 + 20
        elif pattern == "sparse":
            logits.fill_(-float("inf"))
            logits[[1, ROWS + 7, VOCAB - 1]] = torch.tensor([3.0, 2.0, 1.0])
        head.logits.copy_(logits[rank * ROWS : (rank + 1) * ROWS])
        for k, temperature, top_p in configurations:
            seed, position = (1 << 63) + cases * 197, 8192 + cases
            head.set_sampling_config(temperature, top_p, k, seed=seed)
            slot[1] = position
            graph.replay()
            dist.all_gather_into_tensor(gathered, slot)
            expected = reference(logits, k, temperature, top_p, seed, position)
            assert gathered.reshape(8, 2).tolist() == [[expected, position + 1]] * 8, (
                pattern,
                k,
                temperature,
                top_p,
                expected,
                gathered.tolist(),
            )
            assert int(head.status[0]) == 0
            assert head.sampling_buffer.data_ptr() == buffer_address
            assert head.seed_dev.data_ptr() == buffer_address
            assert head.sampling_buffer[3:].tolist() == [12345] * 5
            cases += 1

    # Reuse the same prefill logits with different request parameters and seeds.
    full = torch.zeros((1, VOCAB), device=device)
    for temperature, k, seed in ((0.0, 28, 42), (0.6, 28, 43), (0.6, 20, (1 << 63) + 7)):
        head.set_sampling_config(temperature, 0.95, k, seed=seed)
        token = head.sample_prefill(full, slot, 100)
        expected = reference(full[0].cpu(), k, temperature, 0.95, seed, 99)
        assert int(token[0]) == expected
        assert slot.tolist() == [expected, 100]
        slot[0] = 17
        assert int(token[0]) == expected  # retained session must not alias the decode slot
    head.set_sampling_config()
    from nex_n2_5_pro.sampling import sampling_words

    assert head.sampling_buffer[1:3].tolist() == sampling_words(0, *head.sampling_defaults)[1:3]

    # A nonfinite winning logit must fail, never index before the candidate array.
    for value in (float("inf"), float("nan"), -float("inf")):
        head.status.zero_()
        head.logits.fill_(value)
        slot[1] = 100
        graph.replay()
        dist.all_gather_into_tensor(gathered, slot)
        assert gathered.reshape(8, 2).tolist() == [[-1, 101]] * 8
        assert int(head.status[0]) == 2
    head.status.zero_()
    head.logits.zero_()
    head.sampling_buffer[2] = 29
    slot[1] = 100
    graph.replay()
    dist.all_gather_into_tensor(gathered, slot)
    assert gathered.reshape(8, 2).tolist() == [[-1, 101]] * 8
    assert int(head.status[0]) == 3
    if rank == 0:
        print(f"PASS: {cases} TP8 dynamic cases, prefill reuse, invalid logits and buffer checks")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
