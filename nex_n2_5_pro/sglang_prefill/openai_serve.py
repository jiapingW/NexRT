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

"""OpenAI-compatible HTTP facade for ``bridge_serve.py --listen``.

The bridge remains the sole owner of tokenization, cache state, and GPU execution. This process
validates OpenAI request envelopes, serializes calls into the single-slot bridge, and maps either
completed responses or incremental bridge events to OpenAI response objects and SSE chunks.
"""

import argparse
import json
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit


MAX_BODY_BYTES = 16 * 1024 * 1024

_UNSUPPORTED_CONTROL_DEFAULTS = {
    "response_format": (None,),
    "tools": (None, []),
    "tool_choice": (None, "none"),
    "parallel_tool_calls": (None, False),
    "functions": (None, []),
    "function_call": (None, "none"),
    "logit_bias": (None, {}),
    "presence_penalty": (None, 0, 0.0),
    "frequency_penalty": (None, 0, 0.0),
    "logprobs": (None, False),
    "top_logprobs": (None, 0),
    "echo": (None, False),
    "best_of": (None, 1),
    "suffix": (None,),
}


class RequestError(Exception):
    def __init__(self, message, *, param=None, code=None, status=HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.param = param
        self.code = code
        self.status = status


@dataclass
class ServerState:
    backend_host: str
    backend_port: int
    backend_timeout: float
    model_name: str
    forward_sampling: bool = False
    http_write_timeout: float = 1.0

    def __post_init__(self):
        if self.backend_timeout <= 0.0 or self.http_write_timeout <= 0.0:
            raise ValueError("server timeouts must be > 0")
        self.backend_lock = threading.Lock()


def _require_int(value, name, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestError(f"{name} must be an integer", param=name)
    if minimum is not None and value < minimum:
        raise RequestError(f"{name} must be >= {minimum}", param=name)
    return value


def _reject_unsupported_controls(body):
    for param, defaults in _UNSUPPORTED_CONTROL_DEFAULTS.items():
        if param not in body or any(body[param] == default for default in defaults):
            continue
        raise RequestError(
            f"{param} is not supported by the NexRT backend",
            param=param,
        )


def _stream_include_usage(body):
    options = body.get("stream_options")
    if options is None:
        return False
    if not body.get("stream", False):
        raise RequestError("stream_options requires stream=true", param="stream_options")
    if not isinstance(options, dict):
        raise RequestError("stream_options must be an object", param="stream_options")
    unsupported = sorted(set(options) - {"include_usage"})
    if unsupported:
        param = f"stream_options.{unsupported[0]}"
        raise RequestError(f"{param} is not supported", param=param)
    include_usage = options.get("include_usage", False)
    if not isinstance(include_usage, bool):
        raise RequestError(
            "stream_options.include_usage must be a boolean",
            param="stream_options.include_usage",
        )
    return include_usage


def _bridge_payload(body, endpoint, *, forward_sampling=False):
    if not isinstance(body, dict):
        raise RequestError("request body must be a JSON object")
    _reject_unsupported_controls(body)
    if body.get("n", 1) != 1:
        raise RequestError("only n=1 is supported", param="n")
    if body.get("stop") is not None:
        raise RequestError(
            "custom stop sequences are not supported; model stop tokens remain enabled",
            param="stop",
        )
    if not isinstance(body.get("stream", False), bool):
        raise RequestError("stream must be a boolean", param="stream")

    _stream_include_usage(body)
    max_tokens = body.get("max_completion_tokens", body.get("max_tokens", 256))
    max_tokens = _require_int(max_tokens, "max_tokens", minimum=1)
    payload = {"max_new": max_tokens}
    if body.get("stream", False):
        payload["stream"] = True

    if endpoint == "chat":
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise RequestError("messages must be a non-empty array", param="messages")
        payload["messages"] = messages
    else:
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise RequestError("prompt must be a string", param="prompt")
        if not prompt:
            raise RequestError("prompt must not be empty", param="prompt")
        payload["text"] = prompt

    if body.get("seed") is not None:
        payload["seed"] = _require_int(body["seed"], "seed")
    if body.get("cache_key") is not None:
        cache_key = body["cache_key"]
        if not isinstance(cache_key, str) or not cache_key:
            raise RequestError("cache_key must be a non-empty string", param="cache_key")
        payload["cache_key"] = cache_key

    for param in ("temperature", "top_p", "top_k"):
        if body.get(param) is not None:
            if not forward_sampling:
                raise RequestError(
                    "per-request sampling requires --forward-sampling and a compatible backend",
                    param=param,
                )
            payload[param] = body[param]
    return payload


def _call_bridge(state, payload):
    encoded = (json.dumps(payload, ensure_ascii=False) + "\n").encode()
    try:
        with state.backend_lock:
            with socket.create_connection(
                (state.backend_host, state.backend_port), timeout=state.backend_timeout
            ) as conn:
                conn.settimeout(state.backend_timeout)
                conn.sendall(encoded)
                response = bytearray()
                while not response.endswith(b"\n"):
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    response.extend(chunk)
                    if len(response) > MAX_BODY_BYTES:
                        raise RuntimeError("bridge response exceeds 16 MiB")
    except (OSError, TimeoutError) as error:
        raise RequestError(
            f"NexRT backend unavailable: {error}",
            code="backend_unavailable",
            status=HTTPStatus.SERVICE_UNAVAILABLE,
        ) from error
    if not response:
        raise RequestError(
            "NexRT backend closed without a response",
            code="backend_empty_response",
            status=HTTPStatus.BAD_GATEWAY,
        )
    try:
        result = json.loads(response)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RequestError(
            f"invalid response from NexRT backend: {error}",
            code="backend_invalid_response",
            status=HTTPStatus.BAD_GATEWAY,
        ) from error
    return _validate_bridge_result(result)


def _validate_bridge_result(result):
    if not isinstance(result, dict):
        raise RequestError(
            "bridge result must be a JSON object",
            code="backend_invalid_response",
            status=HTTPStatus.BAD_GATEWAY,
        )
    if "error" in result:
        raise RequestError(
            str(result["error"]),
            code="backend_error",
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
    try:
        status = int(result.get("status", 0))
    except (TypeError, ValueError) as error:
        raise RequestError(
            "NexRT backend returned an invalid status",
            code="backend_invalid_response",
            status=HTTPStatus.BAD_GATEWAY,
        ) from error
    if status != 0:
        raise RequestError(
            f"NexRT decode failed with status {status}",
            code="decode_error",
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
    return result


def _iter_bridge_events(state, payload):
    """Yield one JSON object per bridge NDJSON event while holding the single-slot lock."""
    encoded = (json.dumps(payload, ensure_ascii=False) + "\n").encode()
    try:
        with state.backend_lock:
            with socket.create_connection(
                (state.backend_host, state.backend_port), timeout=state.backend_timeout
            ) as conn:
                conn.settimeout(state.backend_timeout)
                conn.sendall(encoded)
                with conn.makefile("rb") as reader:
                    while True:
                        line = reader.readline(MAX_BODY_BYTES + 1)
                        if not line:
                            break
                        if len(line) > MAX_BODY_BYTES:
                            raise RuntimeError("bridge event exceeds 16 MiB")
                        try:
                            event = json.loads(line)
                        except (UnicodeDecodeError, json.JSONDecodeError) as error:
                            raise RequestError(
                                f"invalid event from NexRT backend: {error}",
                                code="backend_invalid_response",
                                status=HTTPStatus.BAD_GATEWAY,
                            ) from error
                        if not isinstance(event, dict):
                            raise RequestError(
                                "bridge event must be a JSON object",
                                code="backend_invalid_response",
                                status=HTTPStatus.BAD_GATEWAY,
                            )
                        yield event
    except (OSError, TimeoutError) as error:
        raise RequestError(
            f"NexRT backend unavailable: {error}",
            code="backend_unavailable",
            status=HTTPStatus.SERVICE_UNAVAILABLE,
        ) from error


def _finish_reason(result, requested_tokens):
    stop = result.get("stop")
    if stop == "max-tokens" or (
        stop is None and int(result.get("generated_tokens", 0)) >= requested_tokens
    ):
        return "length"
    return "stop"


def _usage(result):
    prompt = int(result.get("prompt_tokens", 0))
    completion = int(result.get("generated_tokens", 0))
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _metrics(result):
    names = (
        "ttft_s",
        "prefill_s",
        "gather_ms",
        "decode_us_per_tok",
        "mtp_us_per_tok",
        "accept_mean",
        "handoff_mode",
        "cache_decision",
        "cache_base_tokens",
        "cache_retained",
    )
    return {name: result[name] for name in names if name in result}


def _openai_response(endpoint, result, model_name, requested_tokens, *, request_id=None):
    created = int(time.time())
    request_id = request_id or uuid.uuid4().hex
    finish_reason = _finish_reason(result, requested_tokens)
    text = result.get("generated_text", "")
    if endpoint == "chat":
        return {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": _usage(result),
            "system_fingerprint": "nexrt",
            "x_nexrt": _metrics(result),
        }
    return {
        "id": f"cmpl-{request_id}",
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": _usage(result),
        "system_fingerprint": "nexrt",
        "x_nexrt": _metrics(result),
    }


def _stream_base(endpoint, model_name, *, request_id=None, created=None):
    request_id = request_id or uuid.uuid4().hex
    return {
        "id": f"{'chatcmpl' if endpoint == 'chat' else 'cmpl'}-{request_id}",
        "object": "chat.completion.chunk" if endpoint == "chat" else "text_completion",
        "created": int(time.time()) if created is None else int(created),
        "model": model_name,
        "system_fingerprint": "nexrt",
    }


def _stream_choice(endpoint, base, *, text=None, role=False, finish_reason=None):
    event = dict(base)
    if endpoint == "chat":
        delta = {}
        if role:
            delta["role"] = "assistant"
            delta["content"] = ""
        if text is not None:
            delta["content"] = text
        event["choices"] = [
            {
                "index": 0,
                "delta": delta,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ]
    else:
        event["choices"] = [
            {
                "text": text or "",
                "index": 0,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ]
    return event


def _error_body(error):
    return {
        "error": {
            "message": str(error),
            "type": "invalid_request_error" if error.status < 500 else "server_error",
            "param": error.param,
            "code": error.code,
        }
    }


class OpenAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "NexRTOpenAI/0.1"

    @property
    def state(self):
        return self.server.state

    def log_message(self, fmt, *args):
        print(f"@@ OPENAI {self.client_address[0]} {fmt % args}", flush=True)

    def _send_json(self, status, body):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _send_sse(self, body):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.wfile.write(b"data: " + data + b"\n\n")
        self.wfile.flush()

    def _serve_stream(self, endpoint, bridge_request, include_usage):
        events = _iter_bridge_events(self.state, bridge_request)
        started = False
        completed = False
        try:
            try:
                first = next(events)
            except StopIteration as error:
                raise RequestError(
                    "NexRT backend closed without a stream start event",
                    code="backend_empty_response",
                    status=HTTPStatus.BAD_GATEWAY,
                ) from error
            if first.get("type") == "error" or ("error" in first and first.get("type") is None):
                raise RequestError(
                    str(first.get("error")),
                    code="backend_error",
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            if first.get("type") != "start":
                raise RequestError(
                    f"expected bridge stream start, got {first.get('type')!r}",
                    code="backend_invalid_response",
                    status=HTTPStatus.BAD_GATEWAY,
                )

            base = _stream_base(endpoint, self.state.model_name)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "close")
            self.end_headers()
            started = True
            if endpoint == "chat":
                self._send_sse(_stream_choice(endpoint, base, role=True))

            for event in events:
                event_type = event.get("type")
                if event_type == "delta":
                    text = event.get("text", "")
                    if text:
                        self._send_sse(_stream_choice(endpoint, base, text=text))
                elif event_type == "done":
                    _validate_bridge_result(event)
                    finish = _stream_choice(
                        endpoint,
                        base,
                        finish_reason=_finish_reason(event, bridge_request["max_new"]),
                    )
                    metrics = _metrics(event)
                    if metrics:
                        finish["x_nexrt"] = metrics
                    self._send_sse(finish)
                    if include_usage:
                        usage = dict(base, choices=[], usage=_usage(event))
                        self._send_sse(usage)
                    completed = True
                    break
                elif event_type == "error":
                    self._send_sse(
                        _error_body(
                            RequestError(
                                str(event.get("error", "NexRT stream failed")),
                                code="backend_error",
                                status=HTTPStatus.INTERNAL_SERVER_ERROR,
                            )
                        )
                    )
                    completed = True
                    break
                else:
                    raise RequestError(
                        f"unknown bridge stream event: {event_type!r}",
                        code="backend_invalid_response",
                        status=HTTPStatus.BAD_GATEWAY,
                    )
            if not completed:
                self._send_sse(
                    _error_body(
                        RequestError(
                            "NexRT stream ended before a done event",
                            code="backend_empty_response",
                            status=HTTPStatus.BAD_GATEWAY,
                        )
                    )
                )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except OSError:
            pass
        except RequestError as error:
            if not started:
                raise
            try:
                self._send_sse(_error_body(error))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except OSError:
                pass
        finally:
            events.close()

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if path in ("/health", "/healthz"):
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "model": self.state.model_name,
                    "backend": f"{self.state.backend_host}:{self.state.backend_port}",
                },
            )
            return
        model = {
            "id": self.state.model_name,
            "object": "model",
            "created": 0,
            "owned_by": "nexrt",
        }
        if path == "/v1/models":
            self._send_json(HTTPStatus.OK, {"object": "list", "data": [model]})
            return
        if path.startswith("/v1/models/"):
            requested = unquote(path[len("/v1/models/") :])
            if requested == self.state.model_name:
                self._send_json(HTTPStatus.OK, model)
            else:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    _error_body(
                        RequestError(
                            f"model '{requested}' not found",
                            param="model",
                            code="model_not_found",
                            status=HTTPStatus.NOT_FOUND,
                        )
                    ),
                )
            return
        self._send_json(
            HTTPStatus.NOT_FOUND,
            _error_body(
                RequestError(
                    f"unknown endpoint: {path}", code="not_found", status=HTTPStatus.NOT_FOUND
                )
            ),
        )

    def do_POST(self):  # noqa: N802
        path = urlsplit(self.path).path.rstrip("/")
        endpoint = (
            "chat"
            if path == "/v1/chat/completions"
            else "completion"
            if path == "/v1/completions"
            else None
        )
        if endpoint is None:
            self._send_json(
                HTTPStatus.NOT_FOUND,
                _error_body(
                    RequestError(
                        f"unknown endpoint: {path}", code="not_found", status=HTTPStatus.NOT_FOUND
                    )
                ),
            )
            return
        try:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise RequestError("Content-Length must be an integer") from error
            if length <= 0 or length > MAX_BODY_BYTES:
                raise RequestError("request body size must be between 1 byte and 16 MiB")
            try:
                body = json.loads(self.rfile.read(length))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RequestError(f"invalid JSON body: {error}") from error
            if not isinstance(body, dict):
                raise RequestError("request body must be a JSON object")
            requested_model = body.get("model", self.state.model_name)
            if requested_model != self.state.model_name:
                raise RequestError(
                    f"model '{requested_model}' not found",
                    param="model",
                    code="model_not_found",
                    status=HTTPStatus.NOT_FOUND,
                )
            bridge_request = _bridge_payload(
                body, endpoint, forward_sampling=self.state.forward_sampling
            )
            if body.get("stream", False):
                self._serve_stream(
                    endpoint,
                    bridge_request,
                    include_usage=_stream_include_usage(body),
                )
                return
            result = _call_bridge(self.state, bridge_request)
            response = _openai_response(
                endpoint, result, self.state.model_name, bridge_request["max_new"]
            )
            self._send_json(HTTPStatus.OK, response)
        except RequestError as error:
            self._send_json(error.status, _error_body(error))
        except Exception as error:  # noqa: BLE001
            wrapped = RequestError(
                f"internal server error: {error}",
                code="internal_error",
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            self._send_json(wrapped.status, _error_body(wrapped))


class OpenAIServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, state):
        self.state = state
        super().__init__(address, OpenAIHandler)

    def get_request(self):
        conn, address = super().get_request()
        conn.settimeout(self.state.http_write_timeout)
        return conn, address


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend-host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=9411)
    parser.add_argument("--backend-timeout", type=float, default=1800.0)
    parser.add_argument("--http-write-timeout", type=float, default=1.0)
    parser.add_argument("--model-name", default="Nex-N2.5-Pro")
    parser.add_argument(
        "--forward-sampling",
        action="store_true",
        help="forward sampling parameters to a 1-batch or sampled MTP backend",
    )
    args = parser.parse_args()
    state = ServerState(
        backend_host=args.backend_host,
        backend_port=args.backend_port,
        backend_timeout=args.backend_timeout,
        model_name=args.model_name,
        forward_sampling=args.forward_sampling,
        http_write_timeout=args.http_write_timeout,
    )
    server = OpenAIServer((args.host, args.port), state)
    print(
        f"@@ OPENAI LISTENING http://{args.host}:{args.port}/v1 "
        f"model={args.model_name} backend={args.backend_host}:{args.backend_port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
