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

"""Validate the single-node tensor-parallel workers launched by torchrun."""

from __future__ import annotations

import os


def torchrun_rank(tp_size: int) -> int:
    """Use torchrun's existing single-node TP workers and rendezvous store."""
    if "LOCAL_RANK" not in os.environ:
        raise ValueError(f"launch with torchrun --standalone --nproc-per-node={tp_size}")
    rank = int(os.environ["LOCAL_RANK"])
    if (
        int(os.environ["WORLD_SIZE"]) != tp_size
        or int(os.environ["LOCAL_WORLD_SIZE"]) != tp_size
        or int(os.environ["RANK"]) != rank
        or not 0 <= rank < tp_size
    ):
        raise ValueError(f"the bridge requires one node with {tp_size} torchrun workers")
    os.environ["SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE"] = "env://"
    return rank
