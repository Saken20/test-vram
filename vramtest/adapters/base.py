"""Adapter contract: one ``call(payload)`` = one request to a service, timed into a CallRecord.

Subclasses implement ``_call(payload, st)``; the base class handles the timeout, error capture,
in-flight accounting and the CallRecord. ``call`` never raises for service errors.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field

from .. import audio as audio_util
from ..kv_calc import TokenCounter
from ..recorder import CallRecord, Recorder
from ..templating import placeholders

DEFAULT_AUDIO_IN = {"format": "f32le", "sample_rate": 16000}


@dataclass
class Payload:
    """What a step receives."""
    text: str | None = None
    audio: bytes | None = None          # raw PCM encoded per dataset.audio_in
    audio_seconds: float | None = None
    messages: list[dict] | None = None  # chat: [{role, content}, ...]
    session: int = 0


@dataclass
class Result:
    ok: bool
    text: str | None = None
    audio: bytes | None = None          # as returned by the service (format: the service's audio_out)
    audio_seconds: float | None = None
    record: CallRecord | None = None


class ServiceError(Exception):
    """The service answered with an error (HTTP status, error event, status=false, ...)."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class CallState:
    """Mutable per-call state filled by ``_call``."""
    recorder: Recorder
    service: str
    t_start: float = field(default_factory=time.monotonic)
    t_busy: float | None = None
    t_first: float | None = None
    tokens: int | None = None           # exact KV tokens when the response reports usage
    extra: dict = field(default_factory=dict)

    def busy(self) -> None:
        """Processing starts now (call once the request is fully sent; idempotent)."""
        if self.t_busy is None:
            self.t_busy = self.recorder.busy(self.service)

    def first(self) -> None:
        """First byte/token of the response arrived (idempotent)."""
        if self.t_first is None:
            self.t_first = time.monotonic()


class Adapter:
    """Base class. ``name`` is the service name, ``cfg`` its (defaulted) config dict."""

    kind = "generic"

    def __init__(self, name: str, cfg: dict, recorder: Recorder | None = None, tokens: TokenCounter | None = None,
                 *, audio_in: dict | None = None, system_prompt: str = ""):
        self.name = name
        self.cfg = cfg
        self.recorder = recorder if recorder is not None else Recorder()
        self.tokens = tokens or TokenCounter("", cfg.get("chars_per_token", 4.0))
        self.audio_in = audio_in or dict(DEFAULT_AUDIO_IN)
        self.audio_out = cfg.get("audio_out") or {"format": "s16le", "sample_rate": 24000}
        self.system_prompt = system_prompt or ""
        self.timeout_s = float(cfg.get("timeout_s") or 120)

    # ------------------------------------------------------------ to override
    async def _call(self, payload: Payload, st: CallState) -> Result:
        raise NotImplementedError

    def templates(self):
        """Templates of this adapter (used to decide which placeholders to compute)."""
        return self.cfg.get("request") or {}

    @property
    def default_input(self) -> str:
        """Natural input of this service when used alone: audio | text | messages."""
        used = placeholders(self.templates())
        if self.cfg.get("stream_audio") or used & {"audio", "audio_wav"}:
            return "audio"
        if "messages" in used:
            return "messages"
        return "text"

    async def close(self) -> None:
        pass

    def error_code(self, exc: BaseException) -> tuple[str, str]:
        if isinstance(exc, asyncio.TimeoutError):
            return "TIMEOUT", f"no complete response within {self.timeout_s:g} s"
        return type(exc).__name__, str(exc)[:300]

    # ------------------------------------------------------------ helpers
    def messages(self, payload: Payload) -> list[dict]:
        """Chat messages: payload.messages, else [system prompt] + user text."""
        if payload.messages:
            return payload.messages
        msgs = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        return msgs + [{"role": "user", "content": payload.text or ""}]

    def wav(self, payload: Payload) -> bytes | None:
        if payload.audio is None:
            return None
        x = audio_util.decode(payload.audio, self.audio_in["format"])
        return audio_util.wav_bytes(x, self.audio_in["sample_rate"])

    @property
    def uses_audio(self) -> bool:
        """True when this service consumes Payload.audio (audio seconds / RTF are then input-based)."""
        return self.default_input == "audio"

    def audio_in_seconds(self, payload: Payload) -> float | None:
        if payload.audio is None or not self.uses_audio:
            return None
        if payload.audio_seconds:
            return payload.audio_seconds
        return audio_util.seconds(payload.audio, self.audio_in["format"], self.audio_in["sample_rate"])

    def variables(self, payload: Payload) -> dict:
        """Placeholder values for templates; ``{audio_wav}`` is only encoded when a template uses it."""
        v = {
            "text": payload.text,
            "audio": payload.audio,
            "messages": self.messages(payload) if (payload.messages or payload.text is not None) else None,
            "system_prompt": self.system_prompt,
            "uuid": uuid.uuid4().hex,
            "session": payload.session,
            "model": self.cfg.get("model") or None,
        }
        if "audio_wav" in placeholders(self.templates()):
            v["audio_wav"] = self.wav(payload)
        return v

    def estimate_tokens(self, payload: Payload, output_text: str | None) -> tuple[int, int]:
        """(prompt_tokens, output_tokens) estimate when the service does not report usage."""
        if payload.messages:
            prompt = self.tokens.chat(payload.messages)
        else:
            prompt = self.tokens.count(payload.text or "")
        audio_s = self.audio_in_seconds(payload) or 0.0
        prompt += int(audio_s * float(self.cfg.get("audio_tokens_per_s") or 0))
        return prompt, self.tokens.count(output_text or "")

    # ------------------------------------------------------------ the call
    async def call(self, payload: Payload) -> Result:
        st = CallState(self.recorder, self.name)
        result, code, error = None, "OK", ""
        try:
            result = await asyncio.wait_for(self._call(payload, st), timeout=self.timeout_s)
        except ServiceError as exc:
            code, error = exc.code, exc.message[:300]
        except asyncio.CancelledError:
            # the caller cancelled us (e.g. end of a scenario): keep in-flight counters balanced, then re-raise
            if st.t_busy is not None:
                self.recorder.done(CallRecord(self.name, st.t_start, st.t_busy, time.monotonic(), False,
                                              "CANCELLED", "cancelled by the caller"))
            else:  # a late busy() (e.g. from gRPC's request-stream task) must not leave a call "in flight"
                st.t_busy = time.monotonic()
            raise
        except Exception as exc:  # noqa: BLE001 — every failure becomes a record, never an exception
            code, error = self.error_code(exc)
        st.busy()  # failed before the request was fully sent
        t_end = time.monotonic()
        ok = result is not None and result.ok
        if result is None:
            result = Result(ok=False)

        extra = dict(st.extra)
        if st.tokens is not None:
            tokens = st.tokens
        else:
            prompt, out = self.estimate_tokens(payload, result.text)
            extra.setdefault("prompt_tokens", prompt)
            extra.setdefault("output_tokens", out)
            tokens = prompt + out
        audio_in_s = self.audio_in_seconds(payload)
        latency = t_end - st.t_busy
        if audio_in_s:
            extra["audio_in_s"] = round(audio_in_s, 3)
        if result.audio_seconds:
            extra["audio_out_s"] = round(result.audio_seconds, 3)
        audio_s = result.audio_seconds or audio_in_s
        if audio_s:
            extra["audio_s"] = round(audio_s, 3)
            if ok:
                extra["rtf"] = round(latency / audio_s, 4)
        if result.text:
            extra.setdefault("text", result.text[:200])
        record = CallRecord(
            service=self.name, t_start=st.t_start, t_busy=st.t_busy, t_end=t_end, ok=ok, code=code if not ok else "OK",
            error=error, ttfb_s=max(0.0, st.t_first - st.t_busy) if st.t_first is not None else None,
            tokens=int(tokens), extra=extra,
        )
        if not ok and code == "OK":
            record.code = "FAILED"
        self.recorder.done(record)
        result.record = record
        return result
