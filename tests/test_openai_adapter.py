import json

import numpy as np

from tests.conftest import run
from vramtest import config
from vramtest.adapters import Payload, make_adapter
from vramtest.recorder import Recorder


def _adapter(url, kind, recorder=None, **extra):
    cfg = config.from_dict({"services": {"svc": {"adapter": "openai", "url": url, "kind": kind, **extra}}})
    return make_adapter("svc", cfg["services"]["svc"], recorder, system_prompt="You are helpful.")


async def _call(adapter, payload):
    try:
        return await adapter.call(payload)
    finally:
        await adapter.close()


def test_chat_streaming_usage_and_ttft(openai_mock):
    rec = Recorder()
    a = _adapter(openai_mock.url + "/v1", "chat", rec, request={"max_tokens": 5, "temperature": 0.1})
    r = run(_call(a, Payload(text="hello")))
    assert r.ok and r.text == "Sure, here is a"
    body = openai_mock.requests[-1]["body"]
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["model"] == "mock-model"   # resolved from /v1/models
    assert body["max_tokens"] == 5 and body["temperature"] == 0.1
    assert body["messages"][0] == {"role": "system", "content": "You are helpful."}
    rec_ = r.record
    assert rec_.service == "svc" and rec_.code == "OK"
    assert 0 < rec_.ttfb_s < rec_.latency_s
    assert rec_.extra["usage"] and rec_.tokens == rec_.extra["prompt_tokens"] + 5
    assert rec.records == [rec_] and rec.snapshot() == {"svc": 0}


def test_chat_explicit_messages_and_no_stream(openai_mock):
    a = _adapter(openai_mock.url, "chat", model="m1", request={"stream": False})
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]
    r = run(_call(a, Payload(text="c", messages=msgs)))
    assert r.ok and r.text.startswith("Sure") and r.record.ttfb_s is None
    body = openai_mock.requests[-1]["body"]
    assert body["messages"] == msgs and body["model"] == "m1" and body["stream"] is False


def test_completions(openai_mock):
    r = run(_call(_adapter(openai_mock.url, "completions"), Payload(text="Once upon a time")))
    assert r.ok and r.text == "Sure, here is a short answer." and r.record.ttfb_s is not None
    assert openai_mock.requests[-1]["body"]["prompt"] == "Once upon a time"


def test_transcription_uploads_wav(openai_mock):
    audio = np.zeros(8000, dtype="<f4").tobytes()  # 0.5 s at 16 kHz, f32le
    a = _adapter(openai_mock.url, "transcription", model="whisper", request={"language": "en"})
    r = run(_call(a, Payload(audio=audio)))
    assert r.ok and r.text == f"transcribed {44 + 8000 * 2} bytes"
    req = openai_mock.requests[-1]
    assert req["body"] == {"model": "whisper", "language": "en"} and req["audio_bytes"] == 44 + 16000
    assert r.record.extra["audio_in_s"] == 0.5 and r.record.extra["rtf"] > 0


def test_speech_streamed_audio(openai_mock):
    openai_mock.chunk_delay = 0.05
    a = _adapter(openai_mock.url, "speech", request={"voice": "v1", "response_format": "pcm"},
                 audio_out={"format": "s16le", "sample_rate": 24000})
    r = run(_call(a, Payload(text="Hello there")))
    assert r.ok and len(r.audio) == 3 * 4800 and abs(r.audio_seconds - 0.3) < 1e-9
    assert r.record.ttfb_s < r.record.latency_s - 0.05
    assert openai_mock.requests[-1]["body"]["voice"] == "v1"
    assert r.record.extra["audio_out_s"] == 0.3


def test_embeddings(openai_mock):
    r = run(_call(_adapter(openai_mock.url, "embeddings"), Payload(text="x" * 40)))
    assert r.ok and r.record.tokens == 10 and r.record.extra["vectors"] == 1


def test_http_errors_are_recorded(openai_mock):
    openai_mock.force_status = 503
    rec = Recorder()
    r = run(_call(_adapter(openai_mock.url, "chat", rec), Payload(text="hi")))
    assert not r.ok and r.record.code == "HTTP_503" and "forced" in r.record.error
    assert rec.snapshot() == {"svc": 0}
    openai_mock.force_status = None
    r = run(_call(_adapter(openai_mock.url, "speech"), Payload(text="")))
    assert not r.ok and r.record.code == "HTTP_400"


def test_concurrency_limit_429(openai_mock):
    import asyncio
    openai_mock.limit = 2
    openai_mock.token_delay = 0.05
    a = _adapter(openai_mock.url, "chat", model="m")

    async def burst():
        try:
            return await asyncio.gather(*(a.call(Payload(text="hi")) for _ in range(6)))
        finally:
            await a.close()
    codes = sorted(r.record.code for r in run(burst()))
    assert codes.count("HTTP_429") >= 1 and codes.count("OK") >= 2


def test_connection_error_and_timeout():
    r = run(_call(_adapter("http://127.0.0.1:9", "chat", model="m"), Payload(text="hi")))
    assert not r.ok and r.record.code == "CONNECT_ERROR"


def test_timeout(openai_mock):
    openai_mock.token_delay = 0.2
    r = run(_call(_adapter(openai_mock.url, "chat", model="m", timeout_s=0.3), Payload(text="hi")))
    assert not r.ok and r.record.code == "TIMEOUT" and r.record.ttfb_s is not None


def test_streamed_transcription_and_error_event():
    import httpx
    events = {"ok": [{"choices": [{"delta": {"content": "hel"}}]}, {"choices": [{"delta": {"content": "lo"}}]},
                     {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2}}],
              "bad": [{"choices": [{"delta": {"content": "hel"}}]}, {"error": {"message": "engine dead"}}]}

    def handler(request):
        kind = "bad" if b"bad" in request.content else "ok"
        body = "".join(f"data: {json.dumps(e)}\n\n" for e in events[kind]) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    async def go(language):
        a = _adapter("http://asr.test", "transcription", model="w", request={"stream": True, "language": language})
        a._client = httpx.AsyncClient(base_url=a.base, transport=httpx.MockTransport(handler))
        return await _call(a, Payload(audio=np.zeros(1600, dtype="<f4").tobytes()))
    r = run(go("en"))
    assert r.ok and r.text == "hello" and r.record.tokens == 9 and r.record.ttfb_s is not None
    r = run(go("bad"))
    assert not r.ok and r.record.code == "STREAM_ERROR" and "engine dead" in r.record.error
