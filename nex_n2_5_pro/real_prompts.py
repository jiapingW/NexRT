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

r"""Deterministic real-text prompts for prefill-state generation and decode validation.

build_real_prompt(L, tok) -> list[int] of EXACTLY L token ids:
  a real long-form document (English prose in plain JSONL, with no `datasets` dependency) wrapped in the
  nex_n2_5_pro chat template
  (`<|im_start|>user\n{doc}<|im_end|>\n<|im_start|>assistant\n<think>\n`). Deterministic in L, so the
  independent validation paths prefill identical tokens (bit-for-bit comparable).
"""

import glob
import json
import os

# Long-form English text (ShareGPT assistant prose, code stripped; built by prep_english_docs.py -> data_en/).
# Plain jsonl {"content": ...}: no additional data-loading libraries and scales past 128K tokens.
_DOC_GLOB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_en", "*.jsonl")

_pool = {"text": "", "chars": 0}


def _text_pool(min_chars):
    """Deterministic concatenation of real article text with >= min_chars characters (cached, grows)."""
    if _pool["chars"] >= min_chars:
        return _pool["text"]
    parts, total = [], 0
    for jf in sorted(glob.glob(_DOC_GLOB)):
        if os.path.getsize(jf) == 0:
            continue
        with open(jf, encoding="utf-8") as f:
            for line in f:
                try:
                    c = json.loads(line).get("content", "")
                except Exception:
                    continue
                c = c.strip()
                if c:
                    parts.append(c)
                    total += len(c) + 2
                    if total >= min_chars:
                        break
        if total >= min_chars:
            break
    if total < min_chars:  # absurdly large L: tile (still real tokens)
        base = "\n\n".join(parts) or "The quick brown fox jumps over the lazy dog. "
        parts = [base] * ((min_chars // max(1, len(base))) + 1)
    _pool["text"] = "\n\n".join(parts)
    _pool["chars"] = len(_pool["text"])
    return _pool["text"]


def _as_ids(enc):
    ids = enc if isinstance(enc, list) else enc["input_ids"]
    if ids and isinstance(ids[0], list):  # batched form -> first row
        ids = ids[0]
    return [int(x) for x in ids]


def _fixed_suffix_len(tok):
    r"""Token count of the FIXED tail shared by every user prompt:
    `<|im_end|>\n<|im_start|>assistant\n<think>\n` (the user-turn close AND the generation prompt).
    Computed as the common token suffix of two DIFFERENT-content templatings, so the truncation below
    never drops the `<|im_end|>` that closes the user turn (a malformed prompt otherwise)."""
    a = _as_ids(
        tok.apply_chat_template(
            [{"role": "user", "content": "a"}], add_generation_prompt=True, tokenize=True
        )
    )
    b = _as_ids(
        tok.apply_chat_template(
            [{"role": "user", "content": "bb bb"}], add_generation_prompt=True, tokenize=True
        )
    )
    i = 0
    while i < min(len(a), len(b)) and a[-1 - i] == b[-1 - i]:
        i += 1
    return max(1, i)


def build_real_prompt(L, tok):
    """EXACTLY L real-text token ids under the NexN25Pro chat template, generation-prompt tail kept intact.

    Deterministic: same (L, tok) -> same ids, so torch-prefill and SGLang-prefill process identical
    tokens. The document is truncated in the MIDDLE (content), never the tail, so the trailing
    `<think>\n` generation prompt is always present (a chat/thinking model needs it)."""
    assert L >= 8, "L too small for a chat-templated prompt"
    s_len = _fixed_suffix_len(tok)
    chars = max(64, L * 5)  # English ~4 chars/tok; 5x = safe over-estimate
    ids = []
    for _ in range(12):
        doc = _text_pool(chars)
        ids = _as_ids(
            tok.apply_chat_template(
                [{"role": "user", "content": doc}],
                add_generation_prompt=True,
                tokenize=True,
                # This intermediate corpus is cropped below, keeping the chat suffix intact.
                tokenizer_kwargs={"verbose": False},
            )
        )
        if len(ids) >= L:
            break
        chars *= 2
    assert len(ids) >= L, f"could not reach L={L} (got {len(ids)})"
    ids = ids[: L - s_len] + ids[-s_len:]  # keep prefix+content, then intact tail
    assert len(ids) == L, (len(ids), L)
    return ids


if __name__ == "__main__":
    import sys
    from transformers import AutoTokenizer

    M = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get("NEXRT_MODEL_DIR", "/path/to/nex_n2_5_pro")
    )
    tk = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
    for L in (32, 512, 4096):
        ids = build_real_prompt(L, tk)
        assert len(ids) == L
        head = tk.decode(ids[:24], skip_special_tokens=False)
        tail = tk.decode(ids[-10:], skip_special_tokens=False)
        print(f"L={L:6d}  ok  head={head!r}\n           tail={tail!r}")
