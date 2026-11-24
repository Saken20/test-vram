"""Stdlib mock of an OpenAI-compatible server (vLLM-like) for tests and dry runs without a GPU.

Endpoints: /v1/models, /v1/chat/completions (SSE + usage), /v1/completions, /v1/audio/transcriptions
(multipart WAV), /v1/audio/speech (chunked 16-bit PCM, 24 kHz), /v1/embeddings, /metrics (vLLM format,
KV usage grows with in-flight requests).

    mock = OpenAIMock(limit=4).start()     # port 0 = a free port; mock.url
    ...
    mock.stop()

Run manually: python -m tests.mocks.openai_mock [port]
"""
from __future__ import annotations

import json
import sys
import threading
import time
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WORDS = ["Sure", ",", " here", " is", " a", " short", " answer", "."]


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 512  # the default backlog (5) delays bursts of connections by ~1 s SYN retries


class OpenAIMock:
    def __init__(self, port: int = 0, limit: int | None = None, limit_status: int = 429, token_delay: float = 0.01,
                 audio_chunks: int = 3, chunk_delay: float = 0.01, pool_blocks: int = 1000, block_size: int = 16):
        self.port = port
        self.limit = limit                 # concurrent requests above this get `limit_status`
        self.limit_status = limit_status
        self.force_status: int | None = None   # every /v1 request fails with this HTTP status
        self.token_delay = token_delay
        self.audio_chunks = audio_chunks
        self.chunk_delay = chunk_delay
        self.pool_blocks, self.block_size = pool_blocks, block_size
        self.inflight = 0
        self.peak = 0
        self.requests: list[dict] = []     # (path, parsed body/fields) of every /v1 POST
        self._lock = threading.Lock()
        self.httpd = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_port}"

    def start(self):
        self.httpd = _Server(("127.0.0.1", self.port), self._handler())
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        return self

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()

    def metrics_text(self) -> str:
        usage = min(1.0, self.inflight * 0.02)
        return (
            "# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.\n"
            "# TYPE vllm:kv_cache_usage_perc gauge\n"
            f'vllm:kv_cache_usage_perc{{engine="0",model_name="mock-model"}} {usage}\n'
            f'vllm:num_requests_running{{engine="0",model_name="mock-model"}} {float(self.inflight)}\n'
            'vllm:num_requests_waiting{engine="0",model_name="mock-model"} 0.0\n'
            'vllm:num_preemptions_total{engine="0",model_name="mock-model"} 0.0\n'
            f'vllm:cache_config_info{{block_size="{self.block_size}",cache_dtype="auto",'
            f'num_gpu_blocks="{self.pool_blocks}"}} 1.0\n'
        )

    def _handler(self):
        mock = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            # ---------------------------------------------------- helpers
            def _send(self, status, body: bytes, ctype="application/json"):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _json(self, status, obj):
                self._send(status, json.dumps(obj).encode())

            def _start_chunked(self, ctype):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

            def _chunk(self, data: bytes):
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            def _end_chunked(self):
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

            def _sse(self, obj):
                self._chunk(b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n")

            # ---------------------------------------------------- routes
            def do_GET(self):
                if self.path == "/metrics":
                    self._send(200, mock.metrics_text().encode(), "text/plain; version=0.0.4")
                elif self.path == "/v1/models":
                    self._json(200, {"object": "list", "data": [{"id": "mock-model", "object": "model"}]})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                with mock._lock:
                    mock.inflight += 1
                    mock.peak = max(mock.peak, mock.inflight)
                    over = mock.limit is not None and mock.inflight > mock.limit
                try:
                    if mock.force_status:
                        return self._json(mock.force_status, {"error": {"message": "forced failure"}})
                    if over:
                        return self._json(mock.limit_status, {"error": {"message": "too many requests"}})
                    route = {
                        "/v1/chat/completions": self.chat,
                        "/v1/completions": self.completions,
                        "/v1/audio/transcriptions": self.transcriptions,
                        "/v1/audio/speech": self.speech,
                        "/v1/embeddings": self.embeddings,
                    }.get(self.path)
                    if route is None:
                        return self._json(404, {"error": "not found"})
                    route(raw)
                finally:
                    with mock._lock:
                        mock.inflight -= 1

            def _body(self, raw):
                body = json.loads(raw or b"{}")
                mock.requests.append({"path": self.path, "body": body})
                return body

            def _generate(self, body, chat):
                prompt_tokens = 10 + len(json.dumps(body.get("messages") or body.get("prompt") or "")) // 4
                words = WORDS[: max(1, min(len(WORDS), int(body.get("max_tokens") or len(WORDS))))]
                obj = "chat.completion" if chat else "text_completion"
                if not body.get("stream"):
                    time.sleep(mock.token_delay * len(words))
                    text = "".join(words)
                    choice = {"index": 0, "message": {"role": "assistant", "content": text}} if chat else \
                        {"index": 0, "text": text}
                    return self._json(200, {"object": obj, "choices": [choice], "usage": {
                        "prompt_tokens": prompt_tokens, "completion_tokens": len(words),
                        "total_tokens": prompt_tokens + len(words)}})
                self._start_chunked("text/event-stream")
                if chat:
                    self._sse({"object": f"{obj}.chunk", "choices": [{"index": 0, "delta": {"role": "assistant"}}]})
                for w in words:
                    time.sleep(mock.token_delay)
                    choice = {"index": 0, "delta": {"content": w}} if chat else {"index": 0, "text": w}
                    self._sse({"object": f"{obj}.chunk", "choices": [choice]})
                if (body.get("stream_options") or {}).get("include_usage"):
                    self._sse({"object": f"{obj}.chunk", "choices": [], "usage": {
                        "prompt_tokens": prompt_tokens, "completion_tokens": len(words),
                        "total_tokens": prompt_tokens + len(words)}})
                self._sse(b"[DONE]")
                self._end_chunked()

            def chat(self, raw):
                self._generate(self._body(raw), chat=True)

            def completions(self, raw):
                self._generate(self._body(raw), chat=False)

            def transcriptions(self, raw):
                msg = BytesParser(policy=HTTP).parsebytes(
                    b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + raw)
                fields, audio = {}, b""
                for part in msg.iter_parts():
                    name = part.get_param("name", header="content-disposition")
                    data = part.get_payload(decode=True) or b""
                    if part.get_filename():
                        audio = data
                    else:
                        fields[name] = data.decode()
                mock.requests.append({"path": self.path, "body": fields, "audio_bytes": len(audio)})
                if audio[:4] != b"RIFF":
                    return self._json(400, {"error": {"message": "expected a WAV file"}})
                time.sleep(mock.token_delay * 3)
                self._json(200, {"text": f"transcribed {len(audio)} bytes"})

            def speech(self, raw):
                body = self._body(raw)
                if not body.get("input"):
                    return self._json(400, {"error": {"message": "empty input"}})
                self._start_chunked("audio/pcm")
                for _ in range(mock.audio_chunks):  # 0.1 s of 16-bit PCM at 24 kHz per chunk
                    time.sleep(mock.chunk_delay)
                    self._chunk(b"\x00\x10" * 2400)
                self._end_chunked()

            def embeddings(self, raw):
                body = self._body(raw)
                n = max(1, len(str(body.get("input", ""))) // 4)
                self._json(200, {"object": "list", "data": [{"index": 0, "embedding": [0.1] * 8}],
                                 "usage": {"prompt_tokens": n, "total_tokens": n}})

        return Handler


if __name__ == "__main__":
    m = OpenAIMock(port=int(sys.argv[1]) if len(sys.argv) > 1 else 8000).start()
    print(f"mock OpenAI-compatible server on {m.url} (metrics: {m.url}/metrics)")
    threading.Event().wait()
