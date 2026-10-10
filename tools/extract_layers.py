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

"""Extract the auxiliary weights used by co-located NexRT serving.

By default, only full-attention projections and layer norms are exported for CP.
``--full`` additionally prepares fixtures for standalone decode diagnostics.
An optional MTP checkpoint supplies ``mtp.safetensors``. Scale aliases are
normalized to ``weight_scale`` without changing their multiplier values.

Run on a CPU host:
    python tools/extract_layers.py --src /path/to/model --dst /path/to/nexrt-layers
"""

import argparse
import json
import os
import time

import torch  # noqa: F401  (safetensors framework="pt" needs torch for fp8 dtypes)
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/path/to/nex_n2_5_pro"
DST = "/path/to/nexrt-layers"
N_LAYERS = 60

GLOBAL_TARGETS = {
    "embed_tokens.safetensors": "model.language_model.embed_tokens.weight",
    "norm.safetensors": "model.language_model.norm.weight",
    "lm_head.safetensors": "lm_head.weight",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="layers to extract (default: 15 CP layers, or all 60 with --full)",
    )
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--no-cp", action="store_true", help="extract only optional MTP weights")
    modes.add_argument(
        "--full", action="store_true", help="export full fixtures for standalone diagnostics"
    )
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument(
        "--mtp-src", default=None, help="optional sharded checkpoint containing mtp.* tensors"
    )
    cli = ap.parse_args()

    allowed = range(N_LAYERS) if cli.full else range(3, N_LAYERS, 4)
    layers = cli.layers if cli.layers is not None else allowed
    invalid = sorted({layer for layer in layers if layer not in allowed})
    if invalid:
        raise SystemExit(f"invalid layers {invalid}; expected {allowed}")
    os.makedirs(cli.dst, exist_ok=True)
    idx = {}
    if not cli.no_cp:
        with open(os.path.join(cli.src, "model.safetensors.index.json"), encoding="utf-8") as f:
            idx = json.load(f)["weight_map"]

    # outfile -> [keys]
    targets = {}
    for L in [] if cli.no_cp else layers:
        pref = f"model.language_model.layers.{L}."
        keys = [
            k
            for k in idx
            if k.startswith(pref)
            and (
                cli.full
                or k.startswith(pref + "self_attn.")
                or k in (pref + "input_layernorm.weight", pref + "post_attention_layernorm.weight")
            )
        ]
        if not keys:
            raise SystemExit(f"no tensors for layer {L} in source index")
        targets[f"layer.{L}.safetensors"] = keys
    if cli.full:
        missing = [key for key in GLOBAL_TARGETS.values() if key not in idx]
        if missing:
            raise SystemExit(f"required tensors missing from source index: {missing}")
        targets.update({out: [key] for out, key in GLOBAL_TARGETS.items()})

    t0 = time.time()
    for out, keys in targets.items():
        byshard = {}
        for k in keys:
            byshard.setdefault(idx[k], []).append(k)
        d = {}
        for shard, ks in sorted(byshard.items()):
            with safe_open(os.path.join(cli.src, shard), framework="pt", device="cpu") as f:
                for k in ks:
                    out_key = k.removesuffix("_inv") if k.endswith(".weight_scale_inv") else k
                    if out_key in d and k.endswith(".weight_scale_inv"):
                        continue
                    tensor = f.get_tensor(k)
                    d[out_key] = tensor.float() if out_key.endswith(".weight_scale") else tensor
        gb = sum(v.numel() * v.element_size() for v in d.values()) / 1e9
        save_file(d, os.path.join(cli.dst, out))
        print(
            f"  wrote {out}: {len(d)} tensors, {gb:.2f} GB  (t={time.time() - t0:.1f}s)", flush=True
        )
        del d

    if cli.mtp_src:
        with open(os.path.join(cli.mtp_src, "model.safetensors.index.json"), encoding="utf-8") as f:
            mtp_idx = json.load(f)["weight_map"]
        mtp_keys = sorted(key for key in mtp_idx if key.startswith("mtp."))
        if not mtp_keys:
            raise SystemExit(f"no mtp.* tensors in {cli.mtp_src}")
        byshard = {}
        for key in mtp_keys:
            byshard.setdefault(mtp_idx[key], []).append(key)
        mtp = {}
        for shard, keys in sorted(byshard.items()):
            with safe_open(os.path.join(cli.mtp_src, shard), framework="pt", device="cpu") as f:
                for key in keys:
                    out_key = (
                        key[: -len("weight_scale_inv")] + "weight_scale"
                        if key.endswith("weight_scale_inv")
                        else key
                    )
                    if out_key in mtp and key.endswith(".weight_scale_inv"):
                        continue
                    tensor = f.get_tensor(key)
                    mtp[out_key] = tensor.float() if out_key.endswith(".weight_scale") else tensor
        gb = sum(value.numel() * value.element_size() for value in mtp.values()) / 1e9
        save_file(mtp, os.path.join(cli.dst, "mtp.safetensors"))
        print(
            f"  wrote mtp.safetensors: {len(mtp)} tensors, {gb:.2f} GB  "
            f"(t={time.time() - t0:.1f}s)",
            flush=True,
        )
        del mtp

    print(f"DONE in {time.time() - t0:.1f}s -> {cli.dst}", flush=True)


if __name__ == "__main__":
    main()
