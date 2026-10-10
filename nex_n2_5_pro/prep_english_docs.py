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

"""Build the plain-JSONL English document pool used by real_prompts.py.

Source: ShareGPT gpt responses = properly-cased natural English prose. Strips ```code``` blocks, keeps
convos that are >=97% ASCII (English) and reasonably long, writes {"content": <text>} lines until we have
enough characters for 128K+ tokens. Deterministic order (source order) -> reproducible prompts.

Run with Python 3:  python3 prep_english_docs.py --src /path/to/sharegpt.json
"""

import argparse
import json
import re
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parent / "data_en" / "english_docs.jsonl"
DEFAULT_TARGET_CHARS = 3_000_000  # ~128K tokens * ~4 chars + margin
CODE = re.compile(r"```.*?```", re.DOTALL)


def ascii_frac(s):
    return sum(c.isascii() for c in s) / max(1, len(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="ShareGPT JSON input")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--target-chars", type=int, default=DEFAULT_TARGET_CHARS)
    cli = ap.parse_args()

    cli.out.parent.mkdir(parents=True, exist_ok=True)
    with open(cli.src, encoding="utf-8") as source:
        data = json.load(source)
    total, kept = 0, 0
    with open(cli.out, "w", encoding="utf-8") as w:
        for convo in data:
            parts = [
                t.get("value", "") for t in convo.get("conversations", []) if t.get("from") == "gpt"
            ]
            text = CODE.sub(" ", "\n\n".join(parts)).strip()  # drop code blocks -> keep prose
            if len(text) < 400 or ascii_frac(text) < 0.97:  # skip short / non-English
                continue
            w.write(json.dumps({"content": text}, ensure_ascii=False) + "\n")
            kept += 1
            total += len(text)
            if total >= cli.target_chars:
                break
    print(f"wrote {kept} English docs, {total} chars -> {cli.out}")


if __name__ == "__main__":
    main()
