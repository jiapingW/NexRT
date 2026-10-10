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

"""Host-side callback contracts for standard and MTP generation loops."""

import json
import socket
import pytest
from nex_n2_5_pro import m4_control as ctlabi
from nex_n2_5_pro.decode_engine import DecodeEngine
from nex_n2_5_pro.mtp.serve import generate_mtp
from nex_n2_5_pro.sglang_prefill.stream_protocol import BridgeStreamWriter, decode_visible_text


class FakeGraph:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


class FakeSlot:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


class FakeMirror:
    def __init__(self, values):
        self.values = iter(values)

    def last_gen(self):
        return 0

    def wait(self, _generation):
        return next(self.values)


class FakeTokenizer:
    pieces = {1: "Hello", 2: " ", 3: "ignored", 9: "<eos>"}

    def decode(self, token_ids, **_kwargs):
        return "".join(self.pieces[int(token_id)] for token_id in token_ids)


def finish_stream(writer, sender, receiver, generated, stop_ids):
    result = {
        "generated_text": decode_visible_text(FakeTokenizer(), generated, stop_ids),
        "generated_tokens": len(generated),
        "stop": "stop-token",
        "status": 0,
    }
    writer.finish(result)
    assert writer.close()
    sender.shutdown(socket.SHUT_WR)
    events = [json.loads(line) for line in receiver.makefile("r") if line.strip()]
    streamed = "".join(event.get("text", "") for event in events)
    assert streamed == result["generated_text"]
    assert "<eos>" not in streamed
    return events


def test_standard_generate_reports_each_post_prefill_token():
    engine = DecodeEngine.__new__(DecodeEngine)
    engine.slot = FakeSlot((10, 5))
    engine.slot_mirror = FakeMirror(((11, 6), (12, 7)))
    emitted = []
    graph = FakeGraph()

    tokens = DecodeEngine.generate(
        engine,
        graph,
        max_new_tokens=3,
        on_tokens=lambda chunk: emitted.append(list(chunk)),
    )

    assert tokens == [10, 11, 12]
    assert emitted == [[11], [12]]
    assert graph.replays == 2


def test_mtp_generate_reports_each_committed_round_chunk():
    words = [0] * ctlabi.EXT_WORDS
    words[ctlabi.COMMIT_COUNT] = 2
    words[ctlabi.OUTPUT_COUNT] = 2
    words[ctlabi.EMITTED_IDS : ctlabi.EMITTED_IDS + 2] = [21, 22]
    words[ctlabi.FLAGS] = ctlabi.FLAG_STOP_HIT
    words[ctlabi.STATUS] = ctlabi.STATUS_OK
    block = type("FakeBlock", (), {"mirror": FakeMirror((words,))})()
    emitted = []

    tokens, accepts, status = generate_mtp(
        block,
        FakeGraph(),
        max_new_tokens=4,
        on_tokens=lambda chunk: emitted.append(list(chunk)),
    )

    assert tokens == [21, 22]
    assert accepts == [2]
    assert status == 0
    assert emitted == [[21, 22]]


def test_bs1_stop_token_stream_deltas_equal_json_text(monkeypatch):
    stream = type("FakeCudaStream", (), {"synchronize": lambda self: None})()
    monkeypatch.setattr("nex_n2_5_pro.decode_engine.torch.cuda.current_stream", lambda: stream)
    sender, receiver = socket.socketpair()
    try:
        writer = BridgeStreamWriter(sender, FakeTokenizer(), stop_ids={9})
        writer.event({"type": "start", "prompt_tokens": 1})
        writer.tokens([1])
        engine = DecodeEngine.__new__(DecodeEngine)
        engine.slot = FakeSlot((1, 5))
        engine.slot_mirror = FakeMirror(((2, 6), (9, 7), (3, 8)))
        generated = DecodeEngine.generate(
            engine,
            FakeGraph(),
            max_new_tokens=4,
            eos_id={9},
            on_tokens=writer.tokens,
        )
        events = finish_stream(writer, sender, receiver, generated, {9})
    finally:
        sender.close()
        receiver.close()

    assert generated == [1, 2, 9]
    assert [event["type"] for event in events] == ["start", "delta", "done"]


def test_mtp_stop_token_stream_deltas_equal_json_text():
    words = [0] * ctlabi.EXT_WORDS
    words[ctlabi.COMMIT_COUNT] = 2
    words[ctlabi.OUTPUT_COUNT] = 2
    words[ctlabi.EMITTED_IDS : ctlabi.EMITTED_IDS + 2] = [2, 9]
    words[ctlabi.FLAGS] = ctlabi.FLAG_STOP_HIT
    words[ctlabi.STATUS] = ctlabi.STATUS_OK
    block = type("FakeBlock", (), {"mirror": FakeMirror((words,))})()

    sender, receiver = socket.socketpair()
    try:
        writer = BridgeStreamWriter(sender, FakeTokenizer(), stop_ids={9})
        writer.event({"type": "start", "prompt_tokens": 1})
        writer.tokens([1])
        tokens, accepts, status = generate_mtp(
            block,
            FakeGraph(),
            max_new_tokens=4,
            on_tokens=writer.tokens,
        )
        generated = [1] + tokens
        events = finish_stream(writer, sender, receiver, generated, {9})
    finally:
        sender.close()
        receiver.close()

    assert generated == [1, 2, 9]
    assert accepts == [2]
    assert status == 0
    assert [event["type"] for event in events] == ["start", "delta", "done"]


def test_standard_generate_rejects_sampler_error_and_drains_queued_step(monkeypatch):
    drained = []
    stream = type("FakeCudaStream", (), {"synchronize": lambda self: drained.append(True)})()
    monkeypatch.setattr("nex_n2_5_pro.decode_engine.torch.cuda.current_stream", lambda: stream)
    engine = DecodeEngine.__new__(DecodeEngine)
    engine.slot = FakeSlot((10, 5))
    engine.slot_mirror = FakeMirror(((-1, 6),))
    emitted = []
    graph = FakeGraph()
    with pytest.raises(RuntimeError, match="sampling failed"):
        engine.generate(graph, max_new_tokens=4, on_tokens=emitted.extend)
    assert emitted == []
    assert graph.replays == 2
    assert drained == [True]
