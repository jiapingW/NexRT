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

"""Interactive plain-text chat client for the NexRT OpenAI SSE endpoint."""

import argparse
import json
import os
import sys
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _completion_url(base_url):
    url = base_url.rstrip("/")
    if url.endswith("/completions") and not url.endswith("/chat/completions"):
        return url
    if not url.endswith("/v1"):
        url += "/v1"
    return url + "/completions"


def _conversation_prefix(system):
    if not system:
        return ""
    return f"<|im_start|>system\n{system}<|im_end|>\n"


def _user_turn(content):
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _error_message(error):
    if isinstance(error, HTTPError):
        try:
            body = json.loads(error.read())
            return str(body.get("error", {}).get("message") or error)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return str(error)
    return str(error)


def _stream_payload(model, prompt, cache_key, max_tokens, seed=None):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if cache_key is not None:
        payload["cache_key"] = cache_key
    if seed is not None:
        payload["seed"] = seed
    return payload


def _reset_cache_key(cache_key):
    return None if cache_key is None else f"nexrt-chat-{uuid.uuid4().hex}"


def stream_reply(url, model, prompt, cache_key, max_tokens, api_key, seed=None):
    payload = _stream_payload(model, prompt, cache_key, max_tokens, seed)
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers=headers,
    )

    pieces = []
    done = False
    with urlopen(request, timeout=1800) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            if "error" in event:
                error = event["error"]
                message = error.get("message", error) if isinstance(error, dict) else error
                raise RuntimeError(str(message))
            choices = event.get("choices") or []
            if not choices:
                continue
            content = choices[0].get("text")
            if content:
                sys.stdout.write(content)
                sys.stdout.flush()
                pieces.append(content)
    if not done:
        raise RuntimeError("stream ended before [DONE]")
    return "".join(pieces)


def main():
    parser = argparse.ArgumentParser(description="Interactive streaming chat with NexRT")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Nex-N2.5-Pro")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--system", default=None, help="optional system message")
    parser.add_argument(
        "--cache-key",
        default=None,
        help="enable target-only prompt reuse with this session cache key",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be >= 1")

    initial_transcript = _conversation_prefix(args.system)
    transcript = initial_transcript
    cache_key = args.cache_key
    url = _completion_url(args.url)

    print(f"Connected to {args.model}. Use /clear to reset or /exit to quit.")
    while True:
        try:
            user_text = input("\nYou> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break
        if not user_text:
            continue
        if user_text in ("/exit", "/quit"):
            break
        if user_text == "/clear":
            transcript = initial_transcript
            cache_key = _reset_cache_key(cache_key)
            print("Conversation cleared.")
            continue

        prompt = transcript + _user_turn(user_text)
        sys.stdout.write("Assistant> ")
        sys.stdout.flush()
        try:
            assistant_text = stream_reply(
                url,
                args.model,
                prompt,
                cache_key,
                args.max_tokens,
                args.api_key,
                seed=args.seed,
            )
        except KeyboardInterrupt:
            print("\nRequest interrupted.")
            continue
        except (HTTPError, URLError, OSError, RuntimeError, json.JSONDecodeError) as error:
            print(f"\nRequest failed: {_error_message(error)}", file=sys.stderr)
            continue
        transcript = prompt + assistant_text + "<|im_end|>\n"
        print()


if __name__ == "__main__":
    main()
