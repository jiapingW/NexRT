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

"""CUDA-graph timing helpers used by the standalone diagnostics."""

import statistics
from dataclasses import dataclass
from typing import Callable, List

import torch

# Number of int32 elements in the L2 eviction buffer.
_L2_FLUSH_SIZE = 256 * 1024 * 1024 // 4


@dataclass(frozen=True)
class BenchResult:
    """Timing statistics in milliseconds."""

    median_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float
    stddev_ms: float
    reps: int


def _maybe_barrier(tp_reduce: bool) -> Callable[[], None]:
    """Return a dist.barrier callable if tp_reduce and a group is initialized, else a no-op."""
    if tp_reduce:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return lambda: dist.barrier()
    return lambda: None


def _tp_max(value: float, tp_reduce: bool, device: torch.device) -> float:
    """MAX-reduce a scalar across the torch.distributed group (if tp_reduce and a group is
    initialized), else return unchanged -- so a TP=N measurement reports the slowest rank."""
    if not tp_reduce:
        return value
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _capture(fn: Callable) -> "torch.cuda.CUDAGraph":
    """Capture fn() (already warmed up) into a CUDA graph."""
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def bench_graph(
    fn: Callable,
    *,
    warmup: int = 10,
    rep: int = 100,
    flush_l2: bool = False,
    tp_reduce: bool = False,
    device: str = "cuda",
) -> BenchResult:
    """Time a captured graph with CUDA events.

    Without eviction, report the average across back-to-back replays. With
    eviction, collect per-replay statistics. tp_reduce takes the maximum of
    each reported timing statistic across ranks. Isolated block timings do
    not establish complete-model performance.
    """
    if rep < 1:
        raise ValueError(f"rep must be >= 1, got {rep}")
    dev = torch.device(device)
    barrier = _maybe_barrier(tp_reduce)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = _capture(fn)
    torch.cuda.synchronize()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if not flush_l2:  # HOT back-to-back
        barrier()
        start.record()
        for _ in range(rep):
            g.replay()
        end.record()
        end.synchronize()
        t = _tp_max(start.elapsed_time(end) / rep, tp_reduce, dev)
        return BenchResult(t, t, t, t, 0.0, rep)
    cache = torch.empty(_L2_FLUSH_SIZE, dtype=torch.int32, device=dev)  # COLD per-replay
    times: List[float] = []
    for _ in range(rep):
        cache.random_()
        torch.cuda.synchronize()
        barrier()
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return BenchResult(
        median_ms=_tp_max(times[len(times) // 2], tp_reduce, dev),
        min_ms=_tp_max(times[0], tp_reduce, dev),
        max_ms=_tp_max(times[-1], tp_reduce, dev),
        mean_ms=_tp_max(statistics.mean(times), tp_reduce, dev),
        stddev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
        reps=rep,
    )


@dataclass(frozen=True)
class PipelineResult:
    """Differences between successive prefix-graph timings.

    The differences sum to the measured full-prefix time; they are not
    individual kernel durations.
    """

    names: List[str]
    per_ms: List[float]
    e2e_ms: float
    cold: bool
    reps: int


def profile_pipeline(
    stages: List,
    *,
    warmup: int = 15,
    rep: int = 100,
    flush_l2: bool = False,
    tp_reduce: bool = False,
    device: str = "cuda",
) -> PipelineResult:
    """Time growing prefixes of a pipeline and return successive differences.

    Each prefix is captured and timed separately, so differences include
    changes in cache state, scheduling, and measurement noise. Use a profiler
    on the complete graph when individual kernel durations are required.
    """
    names = [n for n, _ in stages]
    thunks = [t for _, t in stages]
    n = len(thunks)
    dev = torch.device(device)
    barrier = _maybe_barrier(tp_reduce)
    cache = torch.empty(_L2_FLUSH_SIZE, dtype=torch.int32, device=dev) if flush_l2 else None

    def time_first(k: int) -> float:
        for _ in range(warmup):
            for t in thunks[:k]:
                t()
        torch.cuda.synchronize()
        barrier()
        g = _capture(lambda: [t() for t in thunks[:k]])
        torch.cuda.synchronize()
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if cache is None:
            barrier()
            start.record()
            for _ in range(rep):
                g.replay()
            end.record()
            end.synchronize()
            return _tp_max(start.elapsed_time(end) / rep, tp_reduce, dev)
        ts: List[float] = []
        for _ in range(min(rep, 60)):
            cache.random_()
            torch.cuda.synchronize()
            barrier()
            start.record()
            g.replay()
            end.record()
            end.synchronize()
            ts.append(start.elapsed_time(end))
        ts.sort()
        return _tp_max(ts[len(ts) // 2], tp_reduce, dev)

    cum = [0.0] + [time_first(k) for k in range(1, n + 1)]  # cumulative first-k-stage latency
    per = [cum[k + 1] - cum[k] for k in range(n)]  # each stage's in-context increment
    return PipelineResult(names, per, cum[n], flush_l2, rep)
