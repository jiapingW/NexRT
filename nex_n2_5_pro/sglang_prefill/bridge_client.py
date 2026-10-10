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

"""Minimal client for bridge_serve --listen: one JSON line request per connection.

python3 bridge_client.py --port 9411 --chat --text "hello"     # chat-template prompt
python3 bridge_client.py --port 9411 --file doc.txt --seed 7   # raw file prompt, fixed seed
python3 bridge_client.py --port 9411 --cache-key chat-7 --file growing.txt
python3 bridge_client.py --port 9411 --shutdown
"""

import argparse
import json
import socket
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--text", default=None)
    ap.add_argument("--file", default=None, help="read a UTF-8 prompt file on the client")
    ap.add_argument(
        "--chat",
        action="store_true",
        help="server wraps the prompt in the model's chat template (user turn)",
    )
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--cache-key",
        default=None,
        help="stable owner/session key for exact-hit and suffix-append reuse",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="per-request sampling temperature (1-batch or sampled MTP)",
    )
    ap.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="per-request nucleus threshold (1-batch or sampled MTP)",
    )
    ap.add_argument("--top-k", type=int, default=None, help="1-batch candidate limit (1 to 28)")
    ap.add_argument("--shutdown", action="store_true")
    cli = ap.parse_args()

    if cli.shutdown:
        req = {"cmd": "shutdown"}
    else:
        try:
            text = Path(cli.file).read_text(encoding="utf-8") if cli.file else cli.text or ""
        except (OSError, UnicodeError) as error:
            ap.error(f"cannot read prompt file: {error}")
        req = {"text": text}
        if cli.chat:
            req["chat"] = True
        if cli.max_new is not None:
            req["max_new"] = cli.max_new
        if cli.seed is not None:
            req["seed"] = cli.seed
        if cli.cache_key is not None:
            req["cache_key"] = cli.cache_key
        if cli.temperature is not None:
            req["temperature"] = cli.temperature
        if cli.top_p is not None:
            req["top_p"] = cli.top_p
        if cli.top_k is not None:
            req["top_k"] = cli.top_k

    s = socket.create_connection(
        (cli.host, cli.port), timeout=1800
    )  # first request may pay JIT warmup
    s.sendall((json.dumps(req) + "\n").encode())
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
    s.close()
    print(buf.decode().strip())
    rc = 0
    try:  # human-readable footer for eyeballing timings
        r = json.loads(buf.decode())
        if "error" in r:
            rc = 1
        if "mtp_us_per_tok" in r:
            print(
                f"-- prefill {r['prefill_s']} s | handoff {r['gather_ms']} ms | "
                f"MTP {r['mtp_us_per_tok']} us/token x {r['generated_tokens']} | "
                f"AC={r['accept_mean']} rounds={r['rounds']} status={r['status']}"
            )
        elif "prefill_s" in r:
            print(
                f"-- prefill {r['prefill_s']} s | handoff {r['gather_ms']} ms | "
                f"decode {r['decode_us_per_tok']} us/tok x {r['generated_tokens'] - 1} | "
                f"stop={r['stop']} status={r['status']}"
            )
    except Exception:  # noqa: BLE001
        pass
    sys.exit(rc)


if __name__ == "__main__":
    main()
