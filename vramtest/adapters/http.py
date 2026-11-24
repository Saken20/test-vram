"""Any HTTP endpoint: JSON or multipart request from a template, JSON / raw / streamed response.

    adapter: http
    url: http://localhost:9000/generate
    method: POST
    json: {prompt: "{text}"}            # or multipart: {file: "{audio_wav}", lang: en}
    headers: {Authorization: "Bearer ..."}
    stream: false                        # true: read the body in chunks, TTFB = first chunk
    response: {text: "output.text"}      # dotted JSON path ("choices.0.text"); "$body" = raw body
                                         # audio: "$body" (raw bytes) or a base64 JSON field; ok: "status"

With ``stream: true`` a Server-Sent Events response (``text/event-stream``, e.g. TGI ``/generate_stream``) is
parsed per event: the ``response`` paths are applied to every ``data:`` event (``"$body"`` = the event's data)
and text / audio are concatenated; TTFB = the first event with text or audio.
"""
from __future__ import annotations

import base64
import json

from .. import audio as audio_util
from ..templating import drop_none, jsonable, placeholders, render
from .base import Adapter, CallState, Payload, Result, ServiceError
from .openai import http_error_code, make_client, raise_for_status

_MISSING = object()


def json_path(data, path: str):
    """``"a.b.0.c"`` -> data["a"]["b"][0]["c"]; returns _MISSING when absent."""
    cur = data
    for part in str(path).split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.lstrip("-").isdigit() and -len(cur) <= int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return _MISSING
    return cur


class HttpAdapter(Adapter):
    def __init__(self, name, cfg, recorder=None, tokens=None, **kw):
        super().__init__(name, cfg, recorder, tokens, **kw)
        self.url = cfg["url"]
        self.method = str(cfg.get("method") or "POST").upper()
        self.response = cfg.get("response") or {}
        self._client = None

    def templates(self):
        return {"json": self.cfg.get("json"), "multipart": self.cfg.get("multipart"),
                "headers": self.cfg.get("headers"), "url": self.url}

    @property
    def default_input(self) -> str:
        used = placeholders(self.templates())
        if used & {"audio", "audio_wav"}:
            return "audio"
        return "messages" if "messages" in used else "text"

    @property
    def client(self):
        if self._client is None:
            self._client = make_client(self.timeout_s)
        return self._client

    async def close(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def error_code(self, exc):
        return http_error_code(exc) or super().error_code(exc)

    def _request_kwargs(self, payload: Payload) -> dict:
        v = self.variables(payload)
        kw = {"headers": {k: str(x) for k, x in drop_none(render(self.cfg.get("headers") or {}, v)).items()}}
        url = render(self.url, v)
        if self.cfg.get("multipart") is not None:
            data, files = {}, {}
            for key, value in drop_none(render(self.cfg["multipart"], v)).items():
                if isinstance(value, (bytes, bytearray)):
                    name = "audio.wav" if bytes(value[:4]) == b"RIFF" else "file.bin"
                    files[key] = (name, bytes(value), "audio/wav" if name.endswith(".wav") else
                                  "application/octet-stream")
                else:
                    data[key] = value if isinstance(value, str) else json.dumps(value)
            kw.update(data=data, files=files)
        elif self.cfg.get("json") is not None:
            kw["json"] = jsonable(drop_none(render(self.cfg["json"], v)))
        return {"url": url, **kw}

    async def _call(self, payload: Payload, st: CallState) -> Result:
        kw = self._request_kwargs(payload)
        url = kw.pop("url")
        st.busy()
        if self.cfg.get("stream"):
            buf = bytearray()
            async with self.client.stream(self.method, url, **kw) as resp:
                await raise_for_status(resp)
                ctype = resp.headers.get("content-type", "")
                if "text/event-stream" in ctype:
                    return await self._sse(resp, st)
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        st.first()
                        buf.extend(chunk)
            body = bytes(buf)
        else:
            resp = await self.client.request(self.method, url, **kw)
            await raise_for_status(resp)
            ctype = resp.headers.get("content-type", "")
            body = resp.content
        return self._parse(body, ctype, st)

    def _needs_json(self) -> bool:
        return any(p and p != "$body" for p in self.response.values())

    def _parse(self, body: bytes, ctype: str, st: CallState) -> Result:
        data = _MISSING
        if self._needs_json():
            try:
                data = json.loads(body.decode("utf-8"))
            except ValueError:
                raise ServiceError("BAD_RESPONSE", f"expected JSON ({ctype}): {body[:200]!r}") from None
        text, audio = self._values(data, body)
        audio_s = audio_util.output_seconds(audio, self.audio_out) if audio else None
        st.extra["bytes"] = len(body)
        return Result(ok=True, text=text, audio=bytes(audio) if audio else None, audio_seconds=audio_s)

    async def _sse(self, resp, st: CallState) -> Result:
        """Server-Sent Events: the response paths are applied to every event; text/audio are concatenated."""
        texts, audio, events, size = [], bytearray(), 0, 0
        has_output = bool(self.response.get("text") or self.response.get("audio"))
        lines: list[str] = []

        def dispatch():
            nonlocal events
            raw = "\n".join(lines)
            lines.clear()
            if raw.strip() == "[DONE]":
                return True
            data = _MISSING
            if self._needs_json():
                try:
                    data = json.loads(raw)
                except ValueError:
                    return False  # keep-alive or non-JSON event
            if isinstance(data, dict) and data.get("error"):
                err = data["error"]
                raise ServiceError("STREAM_ERROR", err if isinstance(err, str) else json.dumps(err)[:300])
            text, chunk = self._values(data, raw.encode("utf-8"))
            events += 1
            if text or chunk or not has_output:
                st.first()
            if text:
                texts.append(text)
            if chunk:
                audio.extend(chunk)
            return False

        async for line in resp.aiter_lines():
            size += len(line) + 1
            if not line.strip():  # a blank line ends an event
                if lines and dispatch():
                    break
            elif line.startswith("data:"):
                lines.append(line[5:].removeprefix(" "))
        else:
            if lines:
                dispatch()
        st.extra["bytes"] = size
        st.extra["events"] = events
        audio_b = bytes(audio) if audio else None
        return Result(ok=True, text="".join(texts) if self.response.get("text") else None, audio=audio_b,
                      audio_seconds=audio_util.output_seconds(audio_b, self.audio_out) if audio_b else None)

    def _values(self, data, body: bytes) -> tuple:
        """(text, audio bytes) from a parsed JSON document (``"$body"`` = ``body``); checks ``response.ok``."""

        def get(key):
            path = self.response.get(key)
            if not path:
                return None
            if path == "$body":
                return body
            value = json_path(data, path)
            return None if value is _MISSING else value

        ok_value = get("ok")
        if self.response.get("ok") and not ok_value:
            raise ServiceError("STATUS_FALSE", f"response field {self.response['ok']!r} is {ok_value!r}")
        text = get("text")
        if isinstance(text, (bytes, bytearray)):
            text = bytes(text).decode("utf-8", "replace")
        elif text is not None and not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False)
        audio = get("audio")
        if isinstance(audio, str):
            audio = base64.b64decode(audio)
        return text, audio
