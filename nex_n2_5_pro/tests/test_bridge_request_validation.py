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

"""Host-only request-boundary checks for the bridge server."""

import ast
import json
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from pathlib import Path
import unittest


BRIDGE = Path(__file__).parents[1] / "sglang_prefill" / "bridge_serve.py"


def _load_validation_helper(name="_validated_prompt_ids"):
    tree = ast.parse(BRIDGE.read_text(encoding="utf-8"), filename=str(BRIDGE))
    helper = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"Path": Path, "json": json, "STOP_TOKEN_STRS": ("<tool_call>", "<unknown>")}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(BRIDGE), "exec"), namespace)
    return namespace[name]


class BridgeRequestValidationTests(unittest.TestCase):
    def test_model_and_tokenizer_eos_are_both_stops(self):
        stop_ids = _load_validation_helper("_stop_ids")
        tokenizer = SimpleNamespace(
            eos_token_id=248046,
            unk_token_id=0,
            convert_tokens_to_ids=lambda text: 248058 if text == "<tool_call>" else 0,
        )
        with TemporaryDirectory() as directory:
            config = Path(directory) / "generation_config.json"
            self.assertEqual(stop_ids(tokenizer, directory), {248046, 248058})
            for eos in (248044, [248046, 248044, 248044]):
                with self.subTest(eos=eos):
                    config.write_text(json.dumps({"eos_token_id": eos}))
                    self.assertEqual(stop_ids(tokenizer, directory), {248044, 248046, 248058})
            tokenizer.eos_token_id = [248046, 248044]
            for eos in (None, []):
                config.write_text(json.dumps({"eos_token_id": eos}))
                self.assertEqual(stop_ids(tokenizer, directory), {248044, 248046, 248058})
            tokenizer.eos_token_id = None
            self.assertEqual(stop_ids(tokenizer, directory), {248058})

    def test_socket_rejects_server_file_paths(self):
        request_text = _load_validation_helper("_request_text")
        for request in ({"file": "prompt.txt"}, {"file": "prompt.txt", "text": "hello"}):
            with self.subTest(request=request):
                with self.assertRaisesRegex(ValueError, "reads files locally"):
                    request_text(request)
        self.assertEqual(request_text({"text": "next prompt"}), "next prompt")

    def test_socket_requires_string_text(self):
        request_text = _load_validation_helper("_request_text")
        for value in (None, 7, ["prompt"], {"file": "prompt.txt"}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "text must be a string"):
                    request_text({"text": value})

    def test_oversized_rejection_does_not_change_next_request(self):
        validate = _load_validation_helper()
        with self.assertRaisesRegex(ValueError, "input was not truncated"):
            validate(range(5), 4)
        assert validate([7, 8, 9], 4) == [7, 8, 9]

    def test_empty_prompt_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty prompt"):
            _load_validation_helper()([], 4)

    def test_speculative_minimum_is_checked_before_dispatch(self):
        validate = _load_validation_helper()
        with self.assertRaisesRegex(ValueError, "at least 3 tokens"):
            validate([7], 4, min_length=3)
        assert validate([7], 4) == [7]
        assert validate([7, 8, 9], 4, min_length=3) == [7, 8, 9]

    def test_socket_and_file_paths_share_validation(self):
        source = BRIDGE.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(BRIDGE))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_validated_prompt_ids"
        ]
        assert len(calls) == 2
        assert "[: cli.max_prompt]" not in source
        assert "REJECT {error}" in source


if __name__ == "__main__":
    unittest.main()
