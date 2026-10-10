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

"""Host-only checks for sampling at the streamed prefill boundary."""

import ast
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

BRIDGE = Path(__file__).parents[1] / "sglang_prefill" / "bridge.py"


class FakeReq:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.prefix_indices = SimpleNamespace(dtype="int64")

    def set_extend_input_len(self, length):
        self.extend_input_len = length


class FakePool:
    dtype = "int64"

    def __getitem__(self, index):
        return self

    def to(self, dtype):
        return self


class ChunkedPrefillTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(BRIDGE.read_text())
        nodes = [
            node
            for node in tree.body
            if getattr(node, "name", None) in ("_stage_extend", "PrefillSession", "chunked_prefill")
        ]
        self.events = []
        self.batch = SimpleNamespace(req_pool_indices=[0])
        self.runner = SimpleNamespace(req_to_token_pool=SimpleNamespace(req_to_token=FakePool()))

        def extend(requests, runner):
            req = requests[0]
            length = len(req.fill_ids)
            logits = ("logits", length)
            self.events.append(("extend", length))
            return [100 + length], logits, self.batch

        namespace = {"dataclass": dataclass, "B": SimpleNamespace(extend=extend)}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(BRIDGE), "exec"), namespace)
        self.prefill = namespace["chunked_prefill"]
        modules = {}
        for name, symbol, value in (
            ("sglang.srt.managers.schedule_batch", "Req", FakeReq),
            ("sglang.srt.sampling.sampling_params", "SamplingParams", SimpleNamespace),
        ):
            module = ModuleType(name)
            setattr(module, symbol, value)
            modules[name] = module
        module_patch = patch.dict("sys.modules", modules)
        module_patch.start()
        self.addCleanup(module_patch.stop)

    def test_selection_precedes_final_callbacks_and_is_retained(self):
        for chunk in (2, 8):
            with self.subTest(chunk=chunk):
                self.events.clear()

                def select(logits):
                    self.events.append(("sample", logits[1]))
                    return [7]

                def consume(start, end, token):
                    self.events.append(("consume", end, token))

                def handoff(start, end, token, batch):
                    self.events.append(("handoff", end, token))

                ids, logits, batch, session = self.prefill(
                    self.runner,
                    [1, 2, 3, 4, 5],
                    chunk,
                    on_chunk=consume,
                    on_chunk_batch=handoff,
                    return_session=True,
                    sample_next_token=select,
                )
                self.assertEqual(ids, [7])
                self.assertEqual(session.next_token_ids, [7])
                self.assertEqual(session.next_token_logits, ("logits", 5))
                self.assertEqual(
                    self.events[-4:],
                    [("extend", 5), ("sample", 5), ("consume", 5, 7), ("handoff", 5, 7)],
                )
                self.assertEqual(sum(e[0] == "sample" for e in self.events), 1)
                if chunk == 2:
                    self.assertEqual(
                        self.events[:3], [("extend", 2), ("consume", 2, 102), ("handoff", 2, 102)]
                    )

    def test_default_remains_greedy(self):
        ids, _, _ = self.prefill(self.runner, [1, 2, 3], 2)
        self.assertEqual(ids, [103])

    def test_resume_and_cache_hit_sample_the_final_logits(self):
        _, _, _, session = self.prefill(self.runner, [1, 2, 3], 2, return_session=True)
        ids, _, _, extended = self.prefill(
            self.runner,
            [1, 2, 3, 4, 5],
            2,
            resume=session,
            return_session=True,
            sample_next_token=lambda logits: [logits[1] + 10],
        )
        self.assertEqual(ids, [15])
        self.events.clear()
        ids, _, _, hit = self.prefill(
            self.runner,
            [1, 2, 3, 4, 5],
            2,
            resume=extended,
            return_session=True,
            sample_next_token=lambda logits: [logits[1] + 20],
        )
        self.assertEqual(self.events, [])
        self.assertEqual(ids, [25])
        self.assertEqual(hit.next_token_ids, [25])
        self.assertEqual(extended.next_token_ids, [15])

    def test_selection_failure_does_not_consume_final_draft_row(self):
        def fail(logits):
            raise RuntimeError("sampling failed")

        consumed = []
        with self.assertRaisesRegex(RuntimeError, "sampling failed"):
            self.prefill(
                self.runner,
                [1, 2, 3],
                2,
                on_chunk=lambda start, end, token: consumed.append(end),
                sample_next_token=fail,
            )
        self.assertEqual(consumed, [2])


if __name__ == "__main__":
    unittest.main()
