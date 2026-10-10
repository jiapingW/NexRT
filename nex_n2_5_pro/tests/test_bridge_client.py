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

"""Client prompt files are read locally and sent as inline text."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


CLIENT = Path(__file__).parents[1] / "sglang_prefill" / "bridge_client.py"
SPEC = importlib.util.spec_from_file_location("bridge_client", CLIENT)
client = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(client)


class BridgeClientTests(unittest.TestCase):
    def run_client(self, arguments):
        connection = Mock()
        connection.recv.return_value = b'{"ok": true}\n'
        with (
            patch.object(sys, "argv", [str(CLIENT), "--port", "9411", *arguments]),
            patch.object(client.socket, "create_connection", return_value=connection),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit) as result,
        ):
            client.main()
        self.assertEqual(result.exception.code, 0)
        return json.loads(connection.sendall.call_args.args[0])

    def test_file_sends_utf8_text_and_request_options(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt.txt"
            text = "你好，NexRT\nSecond line."
            path.write_text(text, encoding="utf-8")
            request = self.run_client(
                ["--file", str(path), "--chat", "--max-new", "16", "--seed", "7"]
            )
        self.assertEqual(request, {"text": text, "chat": True, "max_new": 16, "seed": 7})

    def test_sampling_parameters(self):
        request = self.run_client(
            ["--text", "hello", "--temperature", "0", "--top-p", "0.95", "--top-k", "28"]
        )
        self.assertEqual(request, dict(text="hello", temperature=0.0, top_p=0.95, top_k=28))

    def test_text_and_shutdown_requests(self):
        self.assertEqual(self.run_client(["--text", "hello"]), {"text": "hello"})
        self.assertEqual(self.run_client(["--shutdown"]), {"cmd": "shutdown"})

    def test_unreadable_file_fails_before_connecting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.txt"
            with (
                patch.object(sys, "argv", [str(CLIENT), "--port", "9411", "--file", str(path)]),
                patch.object(client.socket, "create_connection") as connect,
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as result,
            ):
                client.main()
            self.assertEqual(result.exception.code, 2)
            connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
