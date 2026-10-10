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

"""Bounded incremental NDJSON transport for the rank-0 bridge socket."""

import json
import queue
import socket
import threading


_END = object()


def visible_token_ids(token_ids, stop_ids=()):
    """Return generated IDs before the first model terminal token."""
    stops = {int(token_id) for token_id in stop_ids}
    visible = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id in stops:
            break
        visible.append(token_id)
    return visible


def _decode(tokenizer, token_ids):
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def decode_visible_text(tokenizer, token_ids, stop_ids=()):
    """Decode the same stop-truncated token view used by incremental streams."""
    return _decode(tokenizer, visible_token_ids(token_ids, stop_ids))


class IncrementalDetokenizer:
    """Emit stable text while retaining only a bounded, locally unstable suffix.

    A split is accepted only when decoding the prefix and suffix independently reproduces the
    current one-shot decode exactly. This keeps byte-fallback sequences together and bounds work
    per token without relying on whitespace or language-specific flush rules.
    """

    DEFAULT_MAX_TAIL_TOKENS = 32

    def __init__(self, tokenizer, stop_ids=(), *, max_tail_tokens=None):
        self.tokenizer = tokenizer
        self.stop_ids = {int(token_id) for token_id in stop_ids}
        self.max_tail_tokens = (
            self.DEFAULT_MAX_TAIL_TOKENS if max_tail_tokens is None else int(max_tail_tokens)
        )
        if self.max_tail_tokens < 1:
            raise ValueError("max_tail_tokens must be >= 1")
        self.token_cache = []
        self.stop_seen = False

    @property
    def retained_tokens(self):
        return len(self.token_cache)

    def _decode(self, token_ids=None):
        return _decode(
            self.tokenizer,
            self.token_cache if token_ids is None else token_ids,
        )

    def _emit_stable_prefix(self):
        if len(self.token_cache) <= self.max_tail_tokens:
            return ""

        full_text = self._decode()
        max_keep = min(self.max_tail_tokens, len(self.token_cache) - 1)
        for keep in range(max_keep, 0, -1):
            split = len(self.token_cache) - keep
            prefix_text = self._decode(self.token_cache[:split])
            suffix_text = self._decode(self.token_cache[split:])
            if prefix_text + suffix_text != full_text:
                continue
            if prefix_text.endswith("\ufffd"):
                continue
            del self.token_cache[:split]
            return prefix_text

        raise RuntimeError(
            "tokenizer has no exact incremental split within "
            f"{self.max_tail_tokens} retained tokens"
        )

    def push(self, token_ids):
        if self.stop_seen:
            return ""
        fragments = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id in self.stop_ids:
                self.stop_seen = True
                break
            self.token_cache.append(token_id)
            text = self._decode()
            if text and text[-1].isspace() and not text.endswith("\ufffd"):
                fragments.append(text)
                self.token_cache.clear()
                continue
            fragment = self._emit_stable_prefix()
            if fragment:
                fragments.append(fragment)
        return "".join(fragments)

    def finish(self):
        text = self._decode() if self.token_cache else ""
        self.token_cache.clear()
        return text


class BridgeStreamWriter:
    """Drain generation events without blocking the TP decode callback.

    The writer exclusively owns socket writes and may call ``shutdown`` to release a stuck write.
    The production dispatcher remains responsible for the socket's final ``close``.
    """

    DEFAULT_MAX_BACKLOG = 256
    DEFAULT_SHUTDOWN_TIMEOUT = 0.25
    DEFAULT_WRITE_TIMEOUT = 1.0

    def __init__(
        self,
        conn,
        tokenizer,
        stop_ids=(),
        *,
        max_backlog=None,
        shutdown_timeout=None,
        write_timeout=None,
        max_tail_tokens=None,
    ):
        self.conn = conn
        self.detokenizer = IncrementalDetokenizer(
            tokenizer,
            stop_ids,
            max_tail_tokens=max_tail_tokens,
        )
        backlog = self.DEFAULT_MAX_BACKLOG if max_backlog is None else int(max_backlog)
        if backlog < 2:
            raise ValueError("max_backlog must be >= 2")
        self.shutdown_timeout = (
            self.DEFAULT_SHUTDOWN_TIMEOUT if shutdown_timeout is None else float(shutdown_timeout)
        )
        write_timeout = (
            self.DEFAULT_WRITE_TIMEOUT if write_timeout is None else float(write_timeout)
        )
        if self.shutdown_timeout <= 0.0 or write_timeout <= 0.0:
            raise ValueError("writer timeouts must be > 0")
        self.events = queue.Queue(maxsize=backlog)
        self.state_lock = threading.Lock()
        self.disconnected = threading.Event()
        self.stopped = threading.Event()
        self.finished = False
        self.error = None
        try:
            self.conn.settimeout(write_timeout)
        except (AttributeError, OSError):
            pass
        self.thread = threading.Thread(
            target=self._run,
            name="bridge-stream-writer",
            daemon=True,
        )
        self.thread.start()

    def _enqueue(self, item):
        with self.state_lock:
            if self.finished or self.disconnected.is_set():
                return False
            try:
                self.events.put_nowait(item)
            except queue.Full:
                self.disconnected.set()
                return False
            return True

    def event(self, payload):
        return self._enqueue(("event", dict(payload)))

    def tokens(self, token_ids):
        return self._enqueue(("tokens", tuple(int(token_id) for token_id in token_ids)))

    def _finish_with(self, item):
        with self.state_lock:
            if self.finished or self.disconnected.is_set():
                return False
            try:
                self.events.put_nowait(item)
            except queue.Full:
                self.disconnected.set()
                self.finished = True
                return False
            self.finished = True
            return True

    def finish(self, result):
        return self._finish_with(("finish", dict(result)))

    def fail(self, error):
        return self._finish_with(("error", str(error)))

    def _shutdown_socket(self):
        try:
            self.conn.shutdown(socket.SHUT_RDWR)
        except (AttributeError, OSError):
            pass

    def cancel(self):
        """Discard queued output and release any blocked writer syscall."""
        with self.state_lock:
            self.finished = True
            self.disconnected.set()
            try:
                self.events.put_nowait(_END)
            except queue.Full:
                pass
        self._shutdown_socket()

    def close(self):
        """Stop the writer within a fixed deadline; never wait behind a slow client."""
        with self.state_lock:
            if not self.finished:
                self.finished = True
                try:
                    self.events.put_nowait(_END)
                except queue.Full:
                    self.disconnected.set()
        self.thread.join(self.shutdown_timeout)
        if self.thread.is_alive():
            self.disconnected.set()
            self._shutdown_socket()
            self.thread.join(self.shutdown_timeout)
        return not self.thread.is_alive()

    def _send(self, payload):
        data = (json.dumps(payload, ensure_ascii=False) + "\n").encode()
        self.conn.sendall(data)

    def _run(self):
        try:
            while not self.disconnected.is_set():
                try:
                    item = self.events.get(timeout=self.shutdown_timeout)
                except queue.Empty:
                    if self.finished:
                        break
                    continue
                if item is _END:
                    break
                kind, payload = item
                if kind == "tokens":
                    text = self.detokenizer.push(payload)
                    if text:
                        self._send({"type": "delta", "text": text})
                elif kind == "finish":
                    text = self.detokenizer.finish()
                    if text:
                        self._send({"type": "delta", "text": text})
                    payload.pop("generated_text", None)
                    payload["type"] = "done"
                    self._send(payload)
                    break
                elif kind == "error":
                    self._send({"type": "error", "error": payload})
                    break
                else:
                    self._send(payload)
        except Exception as error:  # noqa: BLE001
            self.error = error
            self.disconnected.set()
        finally:
            self.stopped.set()
