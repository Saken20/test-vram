import asyncio

from tests.conftest import run
from vramtest import config
from vramtest.adapters import Payload, make_adapter
from vramtest.adapters.base import Adapter, Result
from vramtest.pipeline import run_single, run_turn, single_steps, wait_first_output
from vramtest.recorder import CallRecord, Recorder


class FakeAdapter(Adapter):
    """Sleeps `delay`, first output after `ttfb` (None = not streamed); echoes its input."""

    def __init__(self, name, recorder, delay, ttfb=None, natural="text", fail=False):
        super().__init__(name, {"timeout_s": 5}, recorder)
        self.delay, self.ttfb, self.natural, self.fail = delay, ttfb, natural, fail
        self.seen = []

    @property
    def default_input(self):
        return self.natural

    async def _call(self, payload, st):
        self.seen.append(payload)
        st.busy()
        if self.ttfb is not None:
            await asyncio.sleep(self.ttfb)
            st.first()
            await asyncio.sleep(self.delay - self.ttfb)
        else:
            await asyncio.sleep(self.delay)
        return Result(ok=not self.fail, text=f"{self.name}({payload.text})")


def _rec(latency, ttfb=None):
    return CallRecord("s", 0.0, 1.0, 1.0 + latency, True, ttfb_s=ttfb)


def test_wait_first_output_formula():
    assert wait_first_output([]) is None
    assert wait_first_output([_rec(2.0)]) == 2.0
    assert wait_first_output([_rec(2.0, 0.5)]) == 0.5
    assert wait_first_output([_rec(1.0), _rec(3.0, 0.2), _rec(2.0, 0.4)]) == 1.0 + 3.0 + 0.4
    assert wait_first_output([_rec(1.0), _rec(3.0, 0.2), _rec(2.0)]) == 1.0 + 3.0 + 2.0


def test_run_turn_timing_and_prev_chaining():
    rec = Recorder()
    adapters = {"asr": FakeAdapter("asr", rec, 0.1, natural="audio"),
                "llm": FakeAdapter("llm", rec, 0.2, ttfb=0.05, natural="messages"),
                "tts": FakeAdapter("tts", rec, 0.2, ttfb=0.05)}
    steps = [{"service": "asr", "input": "audio"}, {"service": "llm", "input": "prev"},
             {"service": "tts", "input": "prev"}]
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "old q"},
            {"role": "assistant", "content": "a"}, {"role": "user", "content": "ref text"}]
    payload = Payload(text="ref text", audio=b"\0" * 64, audio_seconds=1.0, messages=msgs, session=4)
    turn = run(run_turn(adapters, steps, payload))
    assert turn.ok and [s.service for s in turn.steps] == ["asr", "llm", "tts"]
    asr_in, llm_in, tts_in = (adapters[n].seen[0] for n in ("asr", "llm", "tts"))
    assert asr_in.audio == payload.audio and asr_in.text is None and asr_in.session == 4
    # chat step: previous text replaces the last user message, system prompt + history kept
    assert llm_in.text == "asr(None)" and llm_in.messages[:-1] == msgs[:-1]
    assert llm_in.messages[-1] == {"role": "user", "content": "asr(None)"}
    assert tts_in.text == "llm(asr(None))" and tts_in.messages is None
    r = turn.records
    assert abs(turn.wait_first_output_s - (r[0].latency_s + r[1].latency_s + r[2].ttfb_s)) < 1e-9
    assert 0.33 < turn.wait_first_output_s < 0.6
    d = turn.to_dict()
    assert d["ok"] and d["session"] == 4 and d["errors"] == [] and len(d["steps"]) == 3
    assert d["steps"][1]["ttfb_s"] is not None and d["steps"][0]["ttfb_s"] is None
    assert rec.snapshot() == {"asr": 0, "llm": 0, "tts": 0}


def test_failed_step_falls_back_and_marks_turn():
    rec = Recorder()
    adapters = {"a": FakeAdapter("a", rec, 0.01, fail=True), "b": FakeAdapter("b", rec, 0.01)}
    steps = [{"service": "a", "input": "text"}, {"service": "b", "input": "prev"}]
    turn = run(run_turn(adapters, steps, Payload(text="orig")))
    assert not turn.ok and turn.errors == ["a:FAILED"] and adapters["b"].seen[0].text == "orig"
    assert turn.wait_first_output_s is not None
    turn = run(run_turn(adapters, steps, Payload(text="orig"), stop_on_error=True))
    assert len(turn.steps) == 1 and turn.wait_first_output_s is None


def test_single_service_mode(openai_mock):
    cfg = config.from_dict({"services": {"tts": {"adapter": "openai", "url": openai_mock.url, "kind": "speech"},
                                         "asr": {"adapter": "openai", "url": openai_mock.url,
                                                 "kind": "transcription"}}})
    tts = make_adapter("tts", cfg["services"]["tts"])
    asr = make_adapter("asr", cfg["services"]["asr"])
    assert single_steps(tts) == [{"service": "tts", "input": "text"}]
    assert single_steps(asr) == [{"service": "asr", "input": "audio"}]

    async def go():
        try:
            return await run_single(tts, Payload(text="hello"))
        finally:
            await tts.close()
    turn = run(go())
    assert turn.ok and turn.wait_first_output_s == turn.records[0].ttfb_s


def test_voice_pipeline_against_mock(openai_mock):
    import numpy as np
    cfg = config.from_dict({
        "services": {"asr": {"adapter": "openai", "url": openai_mock.url, "kind": "transcription"},
                     "llm": {"adapter": "openai", "url": openai_mock.url, "kind": "chat"},
                     "tts": {"adapter": "openai", "url": openai_mock.url, "kind": "speech"}},
        "pipeline": [{"service": "asr", "input": "audio"}, {"service": "llm", "input": "prev"},
                     {"service": "tts", "input": "prev"}]})
    rec = Recorder(cfg["services"])
    adapters = {n: make_adapter(n, s, rec) for n, s in cfg["services"].items()}

    async def go():
        try:
            audio = np.zeros(16000, dtype="<f4").tobytes()
            return await asyncio.gather(*(run_turn(adapters, cfg["pipeline"], Payload(audio=audio, session=i))
                                          for i in range(3)))
        finally:
            for a in adapters.values():
                await a.close()
    turns = run(go())
    assert all(t.ok for t in turns) and len(rec.records) == 9
    assert all(t.steps[-1].result.audio for t in turns)


def test_cancelled_call_keeps_inflight_balanced():
    rec = Recorder()
    a = FakeAdapter("slow", rec, 1.0)

    async def go():
        task = asyncio.ensure_future(a.call(Payload(text="x")))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return True
    assert run(go())
    assert rec.snapshot() == {"slow": 0} and rec.records[0].code == "CANCELLED"


def test_cancelled_before_busy_ignores_a_late_busy():
    """gRPC consumes client-streamed requests in its own task: a busy() after the cancellation must not leak."""
    rec = Recorder()
    states = []

    class LateBusy(Adapter):
        async def _call(self, payload, st):
            states.append(st)
            await asyncio.sleep(1.0)  # still sending the request stream

    a = LateBusy("late", {"timeout_s": 5}, rec)

    async def go():
        task = asyncio.ensure_future(a.call(Payload(text="x")))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        states[0].busy()  # the request-stream task finishing late
    run(go())
    assert rec.snapshot().get("late", 0) == 0 and rec.records == []
