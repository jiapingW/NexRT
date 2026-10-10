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

"""Load SGLang weights with FP8 scale-name adaptation and mmap prefetch."""

import importlib
import mmap
import os

# Compile DeepGEMM only for prefill shapes used by the running engine.
os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")


def _patch_fp8_scale_names(model_path, architectures=None):
    """Map checkpoint block-FP8 scale names to SGLang's registered parameters.

    ``architectures`` overrides the checkpoint's own list, e.g. for the MTP draft
    model SGLang selects from the target checkpoint config."""
    # Resolve SGLang's implementation from the checkpoint architecture instead
    # of baking an upstream module name into NexRT's public model identity.
    from sglang.srt.models.registry import ModelRegistry
    from transformers import AutoConfig

    if architectures is None:
        architectures = AutoConfig.from_pretrained(model_path, trust_remote_code=True).architectures
    model_cls, _ = ModelRegistry.resolve_model_cls(architectures or [])
    model_module = importlib.import_module(model_cls.__module__)

    def _rename(weights):
        for name, w in weights:
            if name.endswith(".weight_scale"):
                name = name + "_inv"
            yield name, w

    def _wrap(orig):
        def patched(self, weights, *a, **k):
            return orig(self, _rename(weights), *a, **k)

        patched._nexrt_scale_rename = True
        return patched

    n = 0
    for cname in dir(model_module):
        cls = getattr(model_module, cname)
        if isinstance(cls, type) and "load_weights" in cls.__dict__:
            if getattr(cls.__dict__["load_weights"], "_nexrt_scale_rename", False):
                continue  # already wrapped through another module
            cls.load_weights = _wrap(cls.__dict__["load_weights"])
            n += 1
    return n


def _prefault_checkpoint_file(path: str) -> None:
    # Buffered reads and mmap faults can take different paths on network file
    # systems. Touch the mapping that the safetensors loader will subsequently
    # read. Small chunks allow the loading thread to run between prefetch steps.
    chunk_bytes = 16 << 20
    with open(path, "rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as pages:
            for offset in range(0, len(pages), chunk_bytes):
                pages[offset : min(offset + chunk_bytes, len(pages)) : mmap.PAGESIZE]


def load_model(server_args, port_args, gpu_id, tp_rank):
    """Load through SGLang, using its rank-distributed prefetch scheduling."""
    import sglang.bench_one_batch as bench
    from sglang.srt.model_loader import weight_utils

    server_args.weight_loader_prefetch_checkpoints = True
    print(
        f"[rank{tp_rank}] fp8 scale-name remap on "
        f"{_patch_fp8_scale_names(server_args.model_path)} classes",
        flush=True,
    )
    original = weight_utils._prefetch_checkpoint_file
    weight_utils._prefetch_checkpoint_file = _prefault_checkpoint_file
    try:
        runner, tokenizer = bench.load_model(server_args, port_args, gpu_id, tp_rank)
        return getattr(runner, "torch_runner", runner), tokenizer
    finally:
        weight_utils._prefetch_checkpoint_file = original
