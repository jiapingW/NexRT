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

"""CPU-only regression checks for MTP graph-routing profile validation."""

import pytest

from nex_n2_5_pro.tests.test_mtp_graph_arms import check_symbols

ATTENTION = ("mtp_wgmma_fd4_mega", "wgmma_fd4_partial", "wgmma_fd4_combine")
COMMON = {
    "M4AttnOProjKernel": 16,
    "M4GdnOProjKernel": 45,
}
DRAFT = {
    "MtpQkvGroupSplitKKernel": 2,
    "MtpNormRopeFusedKernel": 2,
    "MtpOProjFusedPushKernel": 4,
    "MtpGateGemvRowRawAndRmsCta0Kernel": 2,
    "MtpGateTopKSingleCtaNoLogitRmsScaleXnKernel": 2,
    "MtpL1K2PairPacked64UnweightedXbf16V4SplitKW13XnKernel": 2,
    "MtpFusedL2ReducePushKernel": 2,
    "MtpLmHeadGemvRowRmsCta0Kernel": 3,
    "MtpFcT4Kernel": 1,
}
CORE = {
    "noncp": {"m4_attn_partial_quad_dyn": 15, "M4AttnQkvKernel": 15},
    "cp8": {
        "m4_cp_quad_partial": 15,
        "cp_ll_combine_push_ep8": 15,
        "cp_ll_tail_merge_ep8": 15,
    },
}


def symbol(name, family="M4"):
    namespace = f"nexrt::sm90::nex_n2_5_pro::{family}::" if family else ""
    template = "<1>" if name in ATTENTION[:2] else ""
    return f"void {namespace}{name}{template}()"


def profile(arm, counts=(2, 0, 0), *, round_graph=True, family="M4"):
    kernels = COMMON | CORE[arm]
    if round_graph:
        kernels |= DRAFT
        if arm == "cp8":
            kernels["m4_attn_partial_quad_dyn"] = 1
    agg = {symbol(name): (count, 1.0) for name, count in kernels.items()}
    for name, count in zip(ATTENTION, counts):
        if count:
            agg[symbol(name, family)] = (count, 1.0)
    return agg


@pytest.mark.parametrize("arm", ("noncp", "cp8"))
@pytest.mark.parametrize("counts", ((2, 0, 0), (0, 2, 2)))
def test_valid_attention_paths(arm, counts):
    assert check_symbols(arm, profile(arm, counts)) == []


@pytest.mark.parametrize("arm", ("noncp", "cp8"))
@pytest.mark.parametrize(
    "counts",
    (
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 1, 1),
        (0, 2, 1),
        (0, 3, 3),
        (0, 2.5, 2.5),
        (1, 0, 0),
        (3, 0, 0),
        (1, 1, 1),
        (2, 2, 2),
    ),
)
def test_invalid_attention_counts(arm, counts):
    errors = check_symbols(arm, profile(arm, counts))
    assert any("draft attention counts" in error for error in errors)


@pytest.mark.parametrize("arm", ("noncp", "cp8"))
@pytest.mark.parametrize("counts", ((2, 0, 0), (0, 2, 2)))
@pytest.mark.parametrize("family", ("bs1", "M8", ""))
def test_attention_requires_mtp_family(arm, counts, family):
    errors = check_symbols(arm, profile(arm, counts, family=family))
    assert any("outside MTP family" in error for error in errors)


@pytest.mark.parametrize("arm", ("noncp", "cp8"))
@pytest.mark.parametrize("name", ATTENTION)
def test_foreign_attention_cannot_hide_alongside_valid_path(arm, name):
    agg = profile(arm)
    agg[symbol(name, "bs1")] = (2, 1.0)
    assert any("outside MTP family" in error for error in check_symbols(arm, agg))


@pytest.mark.parametrize("arm", ("noncp", "cp8"))
def test_clean_verify_graph(arm):
    assert check_symbols(arm, profile(arm, (0, 0, 0), round_graph=False), round_graph=False) == []


@pytest.mark.parametrize("arm", ("noncp", "cp8"))
@pytest.mark.parametrize("name", ATTENTION)
@pytest.mark.parametrize("family", ("M4", "bs1"))
def test_verify_graph_excludes_each_draft_attention_kernel(arm, name, family):
    agg = profile(arm, (0, 0, 0), round_graph=False)
    agg[symbol(name, family)] = (1, 1.0)
    errors = check_symbols(arm, agg, round_graph=False)
    assert any("draft clone in verify-only graph" in error for error in errors)


@pytest.mark.parametrize("arm", ("noncp", "cp8"))
def test_other_routing_checks_remain_active(arm):
    assert check_symbols(arm, {})  # An empty capture must still fail.
    agg = profile(arm)
    agg[symbol("MtpQkvGroupSplitKKernel")] = (1, 1.0)
    assert any("draft clone" in error for error in check_symbols(arm, agg))
    agg = profile(arm)
    agg[symbol("QkvGroupSplitKKernel", "bs1")] = (1, 1.0)
    assert any("1-batch production kernel" in error for error in check_symbols(arm, agg))
