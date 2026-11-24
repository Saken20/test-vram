"""OpenAI-compatible HTTP API (vLLM, SGLang, TGI, llama.cpp, Kokoro-FastAPI, faster-whisper servers, ...).

kinds:
  chat           POST /v1/chat/completions  (SSE stream + stream_options.include_usage; TTFB = first token)
  completions    POST /v1/completions       (same streaming rules)
  transcription  POST /v1/audio/transcriptions  (multipart WAV upload; JSON {"text": ...})
  speech         POST /v1/audio/speech      (streamed audio body; TTFB = first chunk)
  embeddings     POST /v1/embeddings

``request:`` fields are merged into the body (templates allowed). Set ``request: {stream: false}`` to
disable streaming for chat/completions.
"""
from __future__ import annotations

import asyncio
import json

import httpx

from .. import audio as audio_util
from ..templating import drop_none, render
from .base import Adapter, CallState, Payload, Result, ServiceError

ENDPOINTS = {
    "chat": "/v1/chat/completions",
    "completions": "/v1/completions",
    "transcription": "/v1/audio/transcriptions",
    "speech": "/v1/audio/speech",
    "embeddings": "/v1/embeddings",
}


def base_url(url: str) -> str:
    """Server root: strips a trailing slash and ``/v1``."""
    url = url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def make_client(timeout_s: float, base: str = "") -> httpx.AsyncClient:
    # trust_env=False: local services must not go through HTTP(S)_PROXY from the environment.
    # No connection limit: the load level must be decided by the scenario, not by the client pool.
    return httpx.AsyncClient(
        base_url=base, trust_env=False, timeout=httpx.Timeout(timeout_s, connect=min(10.0, timeout_s)),
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=256),
    )


def http_error_code(exc: BaseException) -> tuple[str, str] | None:
    if isinstance(exc, httpx.TimeoutException):
        return "TIMEOUT", f"{type(exc).__name__}: {exc}"[:300]
    if isinstance(exc, httpx.ConnectError):
        return "CONNECT_ERROR", str(exc)[:300]
    return None


async def raise_for_status(resp: httpx.Response) -> None:
    if resp.status_code >= 400:
        body = (await resp.aread()).decode("utf-8", "replace")
        raise ServiceError(f"HTTP_{resp.status_code}", body[:300])


def check_stream_error(event) -> None:
    """An SSE event carrying ``error`` (vLLM reports failures inside a 200 stream) fails the call."""
    if isinstance(event, dict) and event.get("error"):
        err = event["error"]
        raise ServiceError("STREAM_ERROR", err if isinstance(err, str) else json.dumps(err)[:300])


class OpenAIAdapter(Adapter):
    def __init__(self, name, cfg, recorder=None, tokens=None, **kw):
        super().__init__(name, cfg, recorder, tokens, **kw)
        self.kind = cfg.get("kind", "chat")
        self.base = base_url(cfg["url"])
        self.metrics_url = f"{self.base}/metrics"
        self.model = cfg.get("model") or ""
        self._client: httpx.AsyncClient | None = None
        self._model_checked = bool(self.model)
        self._model_lock: asyncio.Lock | None = None

    @property
    def default_input(self) -> str:
        return {"chat": "messages", "transcription": "audio"}.get(self.kind, "text")

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:  # created lazily inside the running event loop
            self._client = make_client(self.timeout_s, self.base)
        return self._client

    async def close(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def error_code(self, exc):
        return http_error_code(exc) or super().error_code(exc)

    async def resolve_model(self) -> str:
        """Model name: from config, else the first id of GET /v1/models (once)."""
        if self._model_checked:
            return self.model
        if self._model_lock is None:
            self._model_lock = asyncio.Lock()
        async with self._model_lock:  # concurrent first calls wait for a single lookup
            if self._model_checked:
                return self.model
            self._model_checked = True
            try:
                resp = await self.client.get("/v1/models", timeout=5.0)
                data = resp.json().get("data") or []
                if resp.status_code == 200 and data:
                    self.model = data[0].get("id", "")
            except Exception:  # noqa: BLE001 — server may not implement /v1/models
                pass
        return self.model

    async def _body(self, payload: Payload, base: dict) -> dict:
        model = await self.resolve_model()
        extra = drop_none(render(self.cfg.get("request") or {}, self.variables(payload)))
        body = {"model": model} if model else {}
        body.update(base)
        body.update(extra)
        return body

    async def _call(self, payload: Payload, st: CallState) -> Result:
        return await getattr(self, f"_{self.kind}")(payload, st)

    # ------------------------------------------------------------ text generation
    async def _chat(self, payload, st):
        return await self._generate(payload, st, {"messages": self.messages(payload)}, chat=True)

    async def _completions(self, payload, st):
        return await self._generate(payload, st, {"prompt": payload.text or ""}, chat=False)

    async def _generate(self, payload, st, base, chat):
        body = await self._body(payload, {**base, "stream": True})
        if not body.get("stream"):
            st.busy()
            resp = await self.client.post(ENDPOINTS[self.kind], json=body)
            await raise_for_status(resp)
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content") if chat else choice.get("text")
            self._usage(st, data.get("usage"))
            return Result(ok=True, text=text or "")
        body.setdefault("stream_options", {"include_usage": True})
        parts, chunks = [], 0
        st.busy()
        async with self.client.stream("POST", ENDPOINTS[self.kind], json=body) as resp:
            await raise_for_status(resp)
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except ValueError:
                    continue
                check_stream_error(event)
                for choice in event.get("choices") or []:
                    if chat:
                        delta = choice.get("delta") or {}
                        piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                    else:
                        piece = choice.get("text") or ""
                    if piece:
                        st.first()
                        parts.append(piece)
                        chunks += 1
                if event.get("usage"):
                    self._usage(st, event["usage"])
        st.extra["chunks"] = chunks
        return Result(ok=True, text="".join(parts))

    def _usage(self, st, usage):
        if not usage:
            return
        prompt, out = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if prompt is None and out is None:
            return
        st.extra["prompt_tokens"], st.extra["output_tokens"] = prompt or 0, out or 0
        st.extra["usage"] = True
        st.tokens = (prompt or 0) + (out or 0)

    # ------------------------------------------------------------ audio
    async def _transcription(self, payload, st):
        wav = self.wav(payload)
        if wav is None:
            raise ServiceError("NO_INPUT", "transcription needs audio input")
        fields = await self._body(payload, {})
        data = {k: v if isinstance(v, str) else json.dumps(v) for k, v in fields.items()}
        files = {"file": ("audio.wav", wav, "audio/wav")}
        if fields.get("stream") is True:  # vLLM streaming transcription (SSE, like chat)
            parts = []
            st.busy()
            async with self.client.stream("POST", ENDPOINTS["transcription"], data=data, files=files) as resp:
                await raise_for_status(resp)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                        continue
                    try:
                        event = json.loads(line[5:])
                    except ValueError:
                        continue
                    check_stream_error(event)
                    for choice in event.get("choices") or []:
                        piece = (choice.get("delta") or {}).get("content") or ""
                        if piece:
                            st.first()
                            parts.append(piece)
                    if event.get("usage"):
                        self._usage(st, event["usage"])
            return Result(ok=True, text="".join(parts))
        st.busy()
        resp = await self.client.post(ENDPOINTS["transcription"], data=data, files=files)
        await raise_for_status(resp)
        ctype = resp.headers.get("content-type", "")
        if "json" in ctype:
            js = resp.json()
            text = js.get("text", "")
            self._usage(st, js.get("usage") if isinstance(js.get("usage"), dict) else None)
        else:
            text = resp.text
        return Result(ok=True, text=text)

    async def _speech(self, payload, st):
        body = await self._body(payload, {"input": payload.text or ""})
        buf = bytearray()
        st.busy()
        async with self.client.stream("POST", ENDPOINTS["speech"], json=body) as resp:
            await raise_for_status(resp)
            async for chunk in resp.aiter_bytes():
                if chunk:
                    st.first()
                    buf.extend(chunk)
        if not buf:
            raise ServiceError("EMPTY_AUDIO", "the service returned no audio")
        audio = bytes(buf)
        st.extra["chars"] = len(payload.text or "")
        return Result(ok=True, audio=audio, audio_seconds=audio_util.output_seconds(audio, self.audio_out))

    # ------------------------------------------------------------ embeddings
    async def _embeddings(self, payload, st):
        body = await self._body(payload, {"input": payload.text or ""})
        st.busy()
        resp = await self.client.post(ENDPOINTS["embeddings"], json=body)
        await raise_for_status(resp)
        data = resp.json()
        usage = data.get("usage") or {}
        if usage.get("prompt_tokens") is not None:
            st.tokens = usage["prompt_tokens"]
            st.extra.update(prompt_tokens=usage["prompt_tokens"], output_tokens=0, usage=True)
        st.extra["vectors"] = len(data.get("data") or [])
        return Result(ok=True)
