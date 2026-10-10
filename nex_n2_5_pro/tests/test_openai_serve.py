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

"""Host-only tests for the bridge's OpenAI protocol facade."""

import http.client
import json
import socket
import threading
import time

import pytest


from nex_n2_5_pro.sglang_prefill.openai_serve import (
    OpenAIServer,
    RequestError,
    ServerState,
    _bridge_payload,
    _openai_response,
)
from nex_n2_5_pro.sglang_prefill.serve_control import dispatch_port_request
from nex_n2_5_pro.sglang_prefill.stream_protocol import (
    BridgeStreamWriter,
    IncrementalDetokenizer,
    decode_visible_text,
    visible_token_ids,
)
from tools.stream_chat import _reset_cache_key, _stream_payload


class FakeTokenizer:
    pieces = {1: "Hello", 2: " ", 3: "world", 4: "!", 9: "<eos>"}

    def decode(self, token_ids, **_kwargs):
        return "".join(self.pieces[int(token_id)] for token_id in token_ids)


class CharacterTokenizer:
    def decode(self, token_ids, **_kwargs):
        return "".join(chr(int(token_id)) for token_id in token_ids)


class ByteFallbackTokenizer:
    def decode(self, token_ids, **_kwargs):
        return bytes(int(token_id) for token_id in token_ids).decode("utf-8", errors="replace")


class PausedConnection:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.writes = []

    def settimeout(self, _timeout):
        pass

    def sendall(self, data):
        self.started.set()
        if not self.release.wait(1.0):
            raise TimeoutError("test writer was not released")
        self.writes.append(data)

    def shutdown(self, _how):
        self.release.set()


def test_chat_request_and_response():
    body = {
        "model": "Nex-N2.5-Pro",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 7,
        "seed": 42,
        "cache_key": "session-1",
    }
    payload = _bridge_payload(body, "chat")
    assert payload == {
        "messages": body["messages"],
        "max_new": 7,
        "seed": 42,
        "cache_key": "session-1",
    }

    result = {
        "prompt_tokens": 11,
        "generated_tokens": 7,
        "generated_text": "world",
        "stop": "max-tokens",
        "status": 0,
        "ttft_s": 0.12,
        "prefill_s": 0.1,
        "gather_ms": 0.2,
        "handoff_mode": "append",
    }
    response = _openai_response(
        "chat", result, body["model"], payload["max_new"], request_id="fixed"
    )
    assert response["id"] == "chatcmpl-fixed"
    assert response["choices"][0]["message"] == {"role": "assistant", "content": "world"}
    assert response["choices"][0]["finish_reason"] == "length"
    assert response["usage"] == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    assert response["x_nexrt"]["ttft_s"] == 0.12
    assert response["x_nexrt"]["handoff_mode"] == "append"


def test_incremental_detokenizer_holds_partial_words_and_drops_stop_token():
    decoder = IncrementalDetokenizer(FakeTokenizer(), stop_ids={9})
    assert decoder.push([1]) == ""
    assert decoder.push([2]) == "Hello "
    assert decoder.push([3, 4, 9]) == ""
    assert decoder.finish() == "world!"


@pytest.mark.parametrize(
    "text,group_size",
    [
        ("alpha beta gamma\nnext line", 1),
        ("def f(x):return{x:i*i for i in range(x)}", 3),
        ("你好世界，这是一段连续中文文本", 4),
    ],
    ids=("whitespace", "code-no-space", "cjk"),
)
def test_incremental_detokenizer_matches_one_shot_with_bounded_state(text, group_size):
    tokenizer = CharacterTokenizer()
    stop_id = 0
    tokens = [ord(character) for character in text] + [stop_id, ord("X")]
    decoder = IncrementalDetokenizer(
        tokenizer,
        stop_ids={stop_id},
        max_tail_tokens=4,
    )
    fragments = []
    for offset in range(0, len(tokens), group_size):
        fragments.append(decoder.push(tokens[offset : offset + group_size]))
        assert decoder.retained_tokens <= 4
    fragments.append(decoder.finish())

    assert visible_token_ids(tokens, {stop_id}) == tokens[: len(text)]
    assert "".join(fragments) == decode_visible_text(tokenizer, tokens, {stop_id}) == text
    assert any(fragments[:-1]), "long non-whitespace text must stream before finish"


def test_incremental_detokenizer_keeps_byte_fallback_sequences_intact():
    tokenizer = ByteFallbackTokenizer()
    text = "你好, NexRT!" * 4
    tokens = list(text.encode("utf-8")) + [0]
    decoder = IncrementalDetokenizer(tokenizer, stop_ids={0}, max_tail_tokens=6)
    fragments = []
    for offset in range(0, len(tokens), 4):
        fragments.append(decoder.push(tokens[offset : offset + 4]))
        assert decoder.retained_tokens <= 6
    fragments.append(decoder.finish())

    assert "".join(fragments) == decode_visible_text(tokenizer, tokens, {0}) == text


def test_bridge_stream_writer_orders_events_and_flushes_text():
    sender, receiver = socket.socketpair()
    try:
        writer = BridgeStreamWriter(sender, FakeTokenizer(), stop_ids={9})
        writer.event({"type": "start", "prompt_tokens": 3})
        writer.tokens([1, 2])
        writer.tokens([3, 4, 9])
        writer.finish(
            {
                "generated_text": "Hello world!<eos>",
                "generated_tokens": 5,
                "status": 0,
            }
        )
        writer.close()
        sender.shutdown(socket.SHUT_WR)
        events = [json.loads(line) for line in receiver.makefile("r") if line.strip()]
    finally:
        sender.close()
        receiver.close()

    assert [event["type"] for event in events] == ["start", "delta", "delta", "done"]
    assert "".join(event.get("text", "") for event in events) == "Hello world!"
    assert "generated_text" not in events[-1]


def test_stream_writer_terminal_event_uses_one_backlog_slot():
    conn = PausedConnection()
    writer = BridgeStreamWriter(conn, CharacterTokenizer(), max_backlog=2, max_tail_tokens=1)
    assert writer.event({"type": "start"})
    assert conn.started.wait(1.0)
    assert writer.tokens([ord("x")])
    assert writer.finish({"generated_text": "x", "generated_tokens": 1, "status": 0})
    conn.release.set()

    assert writer.close()
    events = [json.loads(line) for write in conn.writes for line in write.decode().splitlines()]
    assert [event["type"] for event in events] == ["start", "delta", "done"]
    assert not writer.disconnected.is_set()


def test_stream_writer_receiver_stall_never_blocks_callback_or_cleanup():
    sender, receiver = socket.socketpair()
    sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    writer = BridgeStreamWriter(
        sender,
        CharacterTokenizer(),
        max_backlog=2,
        shutdown_timeout=0.05,
        write_timeout=0.05,
        max_tail_tokens=1,
    )
    lifecycle = []
    report = []
    try:
        assert writer.event({"type": "start", "padding": "x" * (2 * 1024 * 1024)})
        time.sleep(0.02)

        callback_elapsed = []
        accepted = []

        def complete_request_lifecycle():
            started = time.monotonic()
            accepted.extend(writer.tokens([ord("x")]) for _ in range(1000))
            callback_elapsed.append(time.monotonic() - started)
            lifecycle.append("cache-finished")
            return {"generated_text": "x", "generated_tokens": 1, "status": 0}

        def must_not_terminate(*_args):
            raise AssertionError("slow stream incorrectly terminated the serving group")

        cleanup_started = time.monotonic()
        result = dispatch_port_request(
            complete_request_lifecycle,
            tp_rank=0,
            conn=sender,
            event_stream=writer,
            report=report,
            terminate_fatally=must_not_terminate,
        )
        cleanup_elapsed = time.monotonic() - cleanup_started

        assert callback_elapsed[0] < 0.25
        assert cleanup_elapsed < 0.5
        assert lifecycle == ["cache-finished"]
        assert result is report[0]
        assert not all(accepted)
        assert writer.disconnected.is_set()
        assert writer.stopped.is_set()
        assert not writer.thread.is_alive()
    finally:
        sender.close()
        receiver.close()


def test_openai_server_forwards_bridge_events_before_done():
    bridge = socket.socket()
    bridge.bind(("127.0.0.1", 0))
    bridge.listen(1)
    bridge_port = bridge.getsockname()[1]
    received = []

    def serve_bridge():
        with bridge:
            conn, _ = bridge.accept()
            with conn:
                request = b""
                while not request.endswith(b"\n"):
                    request += conn.recv(4096)
                received.append(json.loads(request))
                conn.sendall(b'{"type":"start","prompt_tokens":1}\n')
                time.sleep(0.02)
                conn.sendall(b'{"type":"delta","text":"A"}\n')
                time.sleep(0.02)
                conn.sendall(b'{"type":"delta","text":" B"}\n')
                conn.sendall(
                    b'{"type":"done","prompt_tokens":1,"generated_tokens":2,'
                    b'"stop":"max-tokens","status":0,"decode_us_per_tok":10.0}\n'
                )

    bridge_thread = threading.Thread(target=serve_bridge)
    bridge_thread.start()
    state = ServerState("127.0.0.1", bridge_port, 5.0, "model")
    server = OpenAIServer(("127.0.0.1", 0), state)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        body = json.dumps(
            {
                "model": "model",
                "prompt": "x",
                "max_tokens": 2,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        )
        connection.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 200
        chunks = []
        while True:
            raw_line = response.readline()
            assert raw_line, "SSE response ended before [DONE]"
            line = raw_line.decode().strip()
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            chunks.append(json.loads(data))
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        server_thread.join()
        bridge_thread.join()

    assert received[0]["stream"] is True
    assert [chunk["choices"][0]["text"] for chunk in chunks[:3]] == ["A", " B", ""]
    assert chunks[2]["choices"][0]["finish_reason"] == "length"
    assert chunks[2]["x_nexrt"]["decode_us_per_tok"] == 10.0
    assert chunks[3]["usage"] == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}


def _start_openai_server(state):
    server = OpenAIServer(("127.0.0.1", 0), state)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    return server, thread


@pytest.mark.parametrize("body", ("null", "[]", '"prompt"', "7"))
def test_non_object_json_body_returns_400_without_calling_backend(body):
    state = ServerState("127.0.0.1", 1, 1.0, "model")
    server, thread = _start_openai_server(state)
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    try:
        connection.request(
            "POST",
            "/v1/completions",
            body,
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join()

    assert response.status == 400
    assert payload["error"]["message"] == "request body must be a JSON object"


def test_json_and_sse_surface_nonzero_backend_status_as_errors():
    bridge = socket.socket()
    bridge.bind(("127.0.0.1", 0))
    bridge.listen(2)

    def serve_bridge():
        with bridge:
            for streaming in (False, True):
                conn, _ = bridge.accept()
                with conn:
                    request = b""
                    while not request.endswith(b"\n"):
                        request += conn.recv(4096)
                    if streaming:
                        conn.sendall(b'{"type":"start","prompt_tokens":1}\n')
                        conn.sendall(b'{"type":"delta","text":"partial"}\n')
                        conn.sendall(
                            b'{"type":"done","prompt_tokens":1,"generated_tokens":1,"status":8}\n'
                        )
                    else:
                        conn.sendall(b'{"prompt_tokens":1,"generated_tokens":1,"status":8}\n')

    bridge_thread = threading.Thread(target=serve_bridge)
    bridge_thread.start()
    state = ServerState("127.0.0.1", bridge.getsockname()[1], 2.0, "model")
    server, server_thread = _start_openai_server(state)
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        connection.request(
            "POST",
            "/v1/completions",
            json.dumps({"model": "model", "prompt": "x", "max_tokens": 2}),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        json_error = json.loads(response.read())
        connection.close()

        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        connection.request(
            "POST",
            "/v1/completions",
            json.dumps({"model": "model", "prompt": "x", "max_tokens": 2, "stream": True}),
            {"Content-Type": "application/json"},
        )
        stream_response = connection.getresponse()
        events = []
        for raw_line in stream_response:
            line = raw_line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            events.append(json.loads(data))
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()
        bridge_thread.join()

    assert response.status == 500
    assert json_error["error"]["code"] == "decode_error"
    assert stream_response.status == 200
    assert events[0]["choices"][0]["text"] == "partial"
    assert events[1]["error"]["code"] == "decode_error"
    assert not any(
        event.get("choices") and event["choices"][0].get("finish_reason") is not None
        for event in events
    )


def test_stalled_sse_reader_releases_backend_lock_for_next_request():
    bridge = socket.socket()
    bridge.bind(("127.0.0.1", 0))
    bridge.listen(2)
    failures = []
    first_connected = threading.Event()

    def serve_bridge():
        try:
            with bridge:
                first, _ = bridge.accept()
                with first:
                    request = b""
                    while not request.endswith(b"\n"):
                        request += first.recv(4096)
                    first_connected.set()
                    first.sendall(b'{"type":"start","prompt_tokens":1}\n')
                    payload = json.dumps({"type": "delta", "text": "x" * (8 * 1024 * 1024)})
                    first.sendall(payload.encode() + b"\n")

                second, _ = bridge.accept()
                with second:
                    request = b""
                    while not request.endswith(b"\n"):
                        request += second.recv(4096)
                    second.sendall(
                        b'{"prompt_tokens":1,"generated_tokens":1,'
                        b'"generated_text":"ok","status":0}\n'
                    )
        except Exception as error:  # noqa: BLE001
            failures.append(error)

    bridge_thread = threading.Thread(target=serve_bridge)
    bridge_thread.start()
    state = ServerState(
        "127.0.0.1",
        bridge.getsockname()[1],
        2.0,
        "model",
        http_write_timeout=0.05,
    )
    server, server_thread = _start_openai_server(state)
    stalled = socket.create_connection(("127.0.0.1", server.server_port), timeout=2)
    stalled.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    body = json.dumps({"model": "model", "prompt": "x", "max_tokens": 2, "stream": True}).encode()
    stalled.sendall(
        b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
        + f"Content-Length: {len(body)}\r\n".encode()
        + b"Content-Type: application/json\r\n\r\n"
        + body
    )
    assert first_connected.wait(1.0)

    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    try:
        connection.request(
            "POST",
            "/v1/completions",
            json.dumps({"model": "model", "prompt": "y", "max_tokens": 1}),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
    finally:
        connection.close()
        stalled.close()
        server.shutdown()
        server.server_close()
        server_thread.join()
        bridge_thread.join()

    assert not failures
    assert response.status == 200
    assert payload["choices"][0]["text"] == "ok"


def test_stream_chat_cache_key_is_explicitly_optional():
    disabled = _stream_payload("model", "prompt", None, 8)
    enabled = _stream_payload("model", "prompt", "session-1", 8)

    assert "cache_key" not in disabled
    assert enabled["cache_key"] == "session-1"
    assert _reset_cache_key(None) is None
    assert _reset_cache_key("session-1").startswith("nexrt-chat-")


@pytest.mark.parametrize(
    "body,param",
    [
        ({"messages": []}, "messages"),
        ({"messages": [{"role": "user", "content": "x"}], "n": 2}, "n"),
        ({"prompt": ["a", "b"]}, "prompt"),
        ({"prompt": "x", "stop": "END"}, "stop"),
        ({"prompt": "x", "max_tokens": 0}, "max_tokens"),
        ({"prompt": "x", "temperature": 0.6}, "temperature"),
        ({"prompt": "x", "response_format": {"type": "json_object"}}, "response_format"),
        ({"prompt": "x", "tools": [{"type": "function"}]}, "tools"),
        ({"prompt": "x", "tool_choice": "auto"}, "tool_choice"),
        ({"prompt": "x", "logit_bias": {"42": 1}}, "logit_bias"),
        ({"prompt": "x", "presence_penalty": 0.5}, "presence_penalty"),
        ({"prompt": "x", "frequency_penalty": -0.5}, "frequency_penalty"),
        ({"prompt": "x", "logprobs": True}, "logprobs"),
        ({"prompt": "x", "stream": "yes"}, "stream"),
        (
            {"prompt": "x", "stream": True, "stream_options": {"include_usage": "yes"}},
            "stream_options.include_usage",
        ),
    ],
)
def test_unsupported_or_invalid_requests_are_explicit(body, param):
    endpoint = "chat" if "messages" in body else "completion"
    with pytest.raises(RequestError) as caught:
        _bridge_payload(body, endpoint)
    assert caught.value.param == param
    assert caught.value.status == 400


@pytest.mark.parametrize(
    "endpoint,body",
    [
        ("chat", {"messages": [{"role": "user", "content": "hello"}]}),
        ("completions", {"prompt": "hello"}),
    ],
)
def test_sampling_fields_are_forwarded_explicitly(endpoint, body):
    body = dict(body, temperature=0, top_p=0.95, top_k=28, seed=(1 << 63) + 9)
    payload = _bridge_payload(body, endpoint, forward_sampling=True)
    for key in ("temperature", "top_p", "top_k", "seed"):
        assert payload[key] == body[key]
    with pytest.raises(RequestError, match="forward-sampling"):
        _bridge_payload(body, endpoint)
    body.pop("temperature")
    body.pop("top_p")
    with pytest.raises(RequestError, match="forward-sampling"):
        _bridge_payload(body, endpoint)
