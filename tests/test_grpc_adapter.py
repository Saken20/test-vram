import asyncio

import pytest

from tests.conftest import ECHO_PROTO, run
from tests.mocks.grpc_mock import start_echo
from vramtest import config
from vramtest.adapters import Payload, make_adapter
from vramtest.recorder import Recorder


def _adapter(port, method, recorder=None, **extra):
    svc = {"adapter": "grpc", "address": f"127.0.0.1:{port}", "proto": ECHO_PROTO, "method": f"echo.Echo/{method}",
           "request": {"text": "{text}", "meta": {"id": "{uuid}", "session": "{session}"}},
           "response": {"text": "reply", "audio": "audio", "ok": "ok"},
           "audio_out": {"format": "s16le", "sample_rate": 24000}, **extra}
    cfg = config.from_dict({"services": {"echo": svc}})
    return make_adapter("echo", cfg["services"]["echo"], recorder)


async def _with_server(fn, state=None):
    server, port = await start_echo(state)
    try:
        return await fn(port, server.state)
    finally:
        await server.stop(None)


def test_unary():
    async def body(port, state):
        a = _adapter(port, "Unary", metadata={"x-key": "abc"})
        r = await a.call(Payload(text="hello", session=7))
        await a.close()
        return r, state
    r, state = run(_with_server(body))
    assert r.ok and r.text == "echo: hello" and abs(r.audio_seconds - 0.1) < 1e-9
    assert r.record.ttfb_s is None and r.record.extra["messages_in"] == 1
    assert state.metadata[-1]["x-key"] == "abc"


def test_server_stream_concatenates_and_measures_ttfb():
    async def body(port, state):
        state.delay = 0.05
        a = _adapter(port, "ServerStream")
        r = await a.call(Payload(text="one two three"))
        await a.close()
        return r
    r = run(_with_server(body))
    assert r.ok and r.text == "one two three" and len(r.audio) == 3 * 4800
    assert r.record.extra["messages_in"] == 3
    assert 0 < r.record.ttfb_s < r.record.latency_s - 0.05


@pytest.mark.parametrize("realtime", [False, True])
def test_client_stream_audio_chunks(realtime):
    audio = b"\x00" * 40000   # f32le 16 kHz: 10000 samples = 0.625 s; 16000-byte chunks = 0.25 s each

    async def body(port, state):
        a = _adapter(port, "ClientStream", stream_audio={"field": "chunk", "chunk_bytes": 16000, "realtime": realtime})
        assert a.default_input == "audio"
        r = await a.call(Payload(audio=audio))
        await a.close()
        return r
    r = run(_with_server(body))
    assert r.ok and r.text == "received 40000 bytes in 3 messages"
    rec = r.record
    assert rec.extra["messages_out"] == 3 and rec.extra["audio_in_s"] == 0.625
    upload = rec.t_busy - rec.t_start
    if realtime:   # two pauses of one chunk duration (0.25 s) between three chunks
        assert upload >= 0.45
    else:
        assert upload < 0.3


def test_bidi_stream():
    async def body(port, state):
        a = _adapter(port, "BidiStream", stream_audio={"field": "chunk", "chunk_bytes": 1000})
        r = await a.call(Payload(audio=b"\x01" * 2500))
        await a.close()
        return r
    r = run(_with_server(body))
    assert r.ok and r.text == "[1000][1000][500]" and r.record.extra["messages_in"] == 3


def test_request_list_sends_several_messages():
    async def body(port, state):
        a = _adapter(port, "BidiStream", request=[{"text": "a"}, {"text": "{text}"}])
        r = await a.call(Payload(text="b"))
        await a.close()
        return r
    r = run(_with_server(body))
    assert r.ok and r.text == "[a][b]"


def test_status_false_and_resource_exhausted():
    async def body(port, state):
        a = _adapter(port, "Unary", timeout_s=5)
        bad = await a.call(Payload(text="fail"))
        state.limit, state.delay = 1, 0.1
        results = await asyncio.gather(*(a.call(Payload(text="x")) for _ in range(4)))
        await a.close()
        return bad, results
    bad, results = run(_with_server(body))
    assert not bad.ok and bad.record.code == "STATUS_FALSE"
    codes = [r.record.code for r in results]
    assert "RESOURCE_EXHAUSTED" in codes and "OK" in codes


def test_unavailable_and_bad_template():
    rec = Recorder()
    a = _adapter(1, "Unary", rec, timeout_s=2)
    r = run(a.call(Payload(text="x")))
    assert not r.ok and r.record.code in ("UNAVAILABLE", "DEADLINE_EXCEEDED")
    b = _adapter(1, "Unary", rec, request={"nonexistent_field": "{text}"})
    r = run(b.call(Payload(text="x")))
    assert not r.ok and r.record.code == "BAD_REQUEST_TEMPLATE"
    assert rec.snapshot() == {"echo": 0} and len(rec.records) == 2


def test_unknown_method_fails_fast():
    from vramtest.protos import ProtoError
    with pytest.raises(ProtoError, match="not found"):
        _adapter(1, "Nope")
