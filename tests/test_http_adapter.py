import numpy as np

from tests.conftest import run
from vramtest import config
from vramtest.adapters import Payload, make_adapter
from vramtest.adapters.http import json_path


def _adapter(svc):
    cfg = config.from_dict({"services": {"h": {"adapter": "http", **svc}}})
    return make_adapter("h", cfg["services"]["h"])


async def _call(adapter, payload):
    try:
        return await adapter.call(payload)
    finally:
        await adapter.close()


def test_json_path():
    data = {"a": {"b": [{"c": 1}, {"c": 2}]}}
    assert json_path(data, "a.b.1.c") == 2 and json_path(data, "a.b.-1.c") == 2
    assert json_path(data, "a.x") is not None and json_path(data, "a.b.5") is json_path(data, "a.x")


def test_json_request_and_dotted_response(openai_mock):
    a = _adapter({"url": openai_mock.url + "/v1/completions", "json": {"prompt": "{text}", "max_tokens": 3},
                  "response": {"text": "choices.0.text"}})
    r = run(_call(a, Payload(text="hey")))
    assert r.ok and r.text == "Sure, here" and r.record.ttfb_s is None
    assert openai_mock.requests[-1]["body"] == {"prompt": "hey", "max_tokens": 3}


def test_streamed_raw_body_audio(openai_mock):
    a = _adapter({"url": openai_mock.url + "/v1/audio/speech", "json": {"input": "{text}"}, "stream": True,
                  "response": {"audio": "$body"}, "audio_out": {"format": "s16le", "sample_rate": 24000}})
    r = run(_call(a, Payload(text="hey")))
    assert r.ok and len(r.audio) == 14400 and abs(r.audio_seconds - 0.3) < 1e-9 and r.record.ttfb_s is not None


def test_multipart_wav(openai_mock):
    a = _adapter({"url": openai_mock.url + "/v1/audio/transcriptions",
                  "multipart": {"file": "{audio_wav}", "model": "w"}, "response": {"text": "text"}})
    assert a.default_input == "audio"
    r = run(_call(a, Payload(audio=np.zeros(1600, dtype="<f4").tobytes())))
    assert r.ok and r.text == f"transcribed {44 + 3200} bytes"
    assert openai_mock.requests[-1]["body"] == {"model": "w"}


def test_errors(openai_mock):
    a = _adapter({"url": openai_mock.url + "/nope", "json": {}, "response": {"text": "x"}})
    r = run(_call(a, Payload(text="hey")))
    assert not r.ok and r.record.code == "HTTP_404"
    a = _adapter({"url": openai_mock.url + "/metrics", "method": "GET", "response": {"text": "x"}})
    r = run(_call(a, Payload(text="hey")))
    assert not r.ok and r.record.code == "BAD_RESPONSE"
    a = _adapter({"url": openai_mock.url + "/metrics", "method": "GET", "response": {"text": "$body"}})
    r = run(_call(a, Payload(text="hey")))
    assert r.ok and "vllm:kv_cache_usage_perc" in r.text


def test_sse_stream_openai_events(openai_mock):
    # the mock's chat endpoint answers with text/event-stream: the text path is applied per event
    a = _adapter({"url": openai_mock.url + "/v1/chat/completions", "stream": True,
                  "json": {"messages": [{"role": "user", "content": "{text}"}], "stream": True},
                  "response": {"text": "choices.0.delta.content"}})
    r = run(_call(a, Payload(text="hey")))
    assert r.ok and r.text == "".join(["Sure", ",", " here", " is", " a", " short", " answer", "."])
    assert r.record.ttfb_s is not None and r.record.extra["events"] >= 8


class _FakeSSE:
    def __init__(self, lines):
        self.lines = lines

    async def aiter_lines(self):
        for line in self.lines:
            yield line


def test_sse_tgi_style_events_and_errors():
    from vramtest.adapters.base import CallState, ServiceError
    from vramtest.recorder import Recorder
    a = _adapter({"url": "http://127.0.0.1:9/generate_stream", "stream": True, "json": {"inputs": "{text}"},
                  "response": {"text": "token.text"}})
    lines = [": keep-alive", "", 'data: {"token": {"text": "Hel"}, "generated_text": null}', "",
             "event: message", 'data:{"token": {"text": "lo"},', 'data: "generated_text": "Hello"}', ""]
    st = CallState(Recorder(), "h")
    st.busy()
    r = run(a._sse(_FakeSSE(lines), st))
    assert r.ok and r.text == "Hello" and st.t_first is not None and st.extra["events"] == 2
    st = CallState(Recorder(), "h")
    try:
        run(a._sse(_FakeSSE(['data: {"token": {"text": "x"}}', "", 'data: {"error": "overloaded"}', ""]), st))
        raise AssertionError("expected a ServiceError")
    except ServiceError as exc:
        assert exc.code == "STREAM_ERROR" and exc.message == "overloaded"
    raw = _adapter({"url": "http://127.0.0.1:9/s", "stream": True, "response": {"text": "$body"}})
    r = run(raw._sse(_FakeSSE(["data: a", "", "data: b", "", "data: [DONE]", "", "data: c"]), CallState(Recorder(), "h")))
    assert r.text == "ab"
