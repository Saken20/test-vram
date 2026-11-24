"""Scenario behaviour against the mocks: burst, concurrency, sessions, ramp limits, dead-process detection."""
import json
import os
import subprocess
import sys

from tests.conftest import run
from tests.fake_nvml import FakeNvml
from tests.mocks.openai_mock import OpenAIMock
from vramtest import config, scenarios
from vramtest.context import Context


def _ctx(tmp_path, services, nvml=None, **sc):
    cfg = config.from_dict({
        "gpu": {"sample_hz": 50, "metrics_poll_s": 0.05},
        "services": services,
        "dataset": {"audio_dir": str(tmp_path / "audio"), "system_prompt": "Be brief.", "history_turns": 0},
        "scenarios": sc,
    })
    return Context(cfg, str(tmp_path / "run"), nvml=nvml or FakeNvml({}), kv=False)


def test_burst_single_service_levels(openai_mock, tmp_path):
    ctx = _ctx(tmp_path, {"llm": {"adapter": "openai", "url": openai_mock.url}}, burst={"pause_s": 0.05})
    res = run(scenarios.burst(ctx, target="llm", levels=[1, 3], rounds=2))
    assert res["kind"] == "burst" and res["target"] == "llm"
    assert res["steps"] == [{"service": "llm", "input": "messages"}]
    assert [lv["level"] for lv in res["levels"]] == [1, 3]
    lv = res["levels"][1]
    assert lv["stats"]["turns"] == 6 and lv["stats"]["failed"] == 0
    assert lv["summary"]["services"]["llm"]["inflight_peak"] == 3
    assert lv["stats"]["steps"][0]["ttfb_s"]["p50"] is not None  # streamed chat: time to first token
    assert len(res["records"]) == 8 and openai_mock.peak == 3
    saved = json.load(open(tmp_path / "run" / "burst_llm.json"))
    assert saved["scenario"] == "burst_llm" and saved["stats"]["turns"] == 8 and saved["samples"]


def test_concurrency_keeps_n_in_flight(openai_mock, tmp_path):
    ctx = _ctx(tmp_path, {"emb": {"adapter": "openai", "url": openai_mock.url, "kind": "embeddings"},
                          "gen": {"adapter": "openai", "url": openai_mock.url, "kind": "completions"}})
    res = run(scenarios.concurrency(ctx, target="gen", users=4, duration=0.5))
    s = res["summary"]["services"]
    assert res["scenario"] == "concurrency_gen_4" and s["gen"]["inflight_peak"] == 4
    assert s["gen"]["calls"] >= 8 and s["emb"]["calls"] == 0 and res["stats"]["failed"] == 0


def test_sessions_spread_turns_and_keep_history(openai_mock, tmp_path):
    ctx = _ctx(tmp_path, {"llm": {"adapter": "openai", "url": openai_mock.url}},
               sessions={"turns_per_session": 2, "session_duration_s": 0.5, "duration_s": 1.0,
                         "profiles": {"normal": 1, "peak": 3}})
    res = run(scenarios.sessions(ctx, profile="peak"))
    assert res["scenario"] == "sessions_peak" and res["target"] == "all"  # no pipeline: every service per turn
    assert res["params"]["sessions"] == 3 and res["summary"]["sessions"] == 3
    # 3 sessions x 2 session cycles x 2 turns, spaced by think time (not back-to-back)
    assert 9 <= res["stats"]["turns"] <= 12
    assert res["summary"]["services"]["llm"]["inflight_mean"] < 1.5
    # second turn of a session carries the session's own history (system + user + assistant + user)
    chats = [r["body"]["messages"] for r in openai_mock.requests if r["path"] == "/v1/chat/completions"]
    assert max(len(m) for m in chats) == 4 and min(len(m) for m in chats) == 2
    try:
        run(scenarios.sessions(ctx, profile="nope"))
    except ValueError as exc:
        assert "unknown profile" in str(exc)


def test_ramp_finds_mock_limit(tmp_path):
    mock = OpenAIMock(limit=6).start()
    try:
        ctx = _ctx(tmp_path, {"llm": {"adapter": "openai", "url": mock.url, "request": {"max_tokens": 8}}},
                   ramp={"levels": [2, 4, 8], "step_duration_s": 0.6, "max_error_rate": 0.0, "bisect_steps": 2,
                         "cooldown_s": 0.05})
        res = run(scenarios.ramp(ctx, target="llm"))
    finally:
        mock.stop()
    r = res["ramp"]
    assert r["last_ok"] == 6 and r["fail"] == 7, [(s["level"], s["reason"]) for s in r["steps"]]
    assert [s["level"] for s in r["steps"]] == [2, 4, 8, 6, 7]   # bisection 4..8 -> 6 OK -> 7 fails
    assert "HTTP_429" in r["fail_reason"] and r["unit"] == "in flight"


def test_ramp_stops_on_latency_limit(openai_mock, tmp_path):
    ctx = _ctx(tmp_path, {"llm": {"adapter": "openai", "url": openai_mock.url,
                                  "limits": {"max_p95_latency_s": 0.001, "max_p95_ttfb_s": None}}},
               ramp={"levels": [1, 2, 4], "step_duration_s": 0.3, "bisect_steps": 2, "cooldown_s": 0})
    r = run(scenarios.ramp(ctx, target="llm"))["ramp"]
    assert r["last_ok"] is None and r["fail"] == 1 and "p95 latency" in r["fail_reason"]
    assert len(r["steps"]) == 1


def test_dead_process_stops_burst_and_fails_ramp(openai_mock, tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    child.kill()
    child.wait()  # reaped: /proc/<pid> is gone
    ctx = _ctx(tmp_path, {"llm": {"adapter": "openai", "url": openai_mock.url, "pids": [child.pid]}},
               nvml=FakeNvml({child.pid: 1000.0}), burst={"pause_s": 0},
               ramp={"levels": [1, 2], "step_duration_s": 0.3, "bisect_steps": 0, "cooldown_s": 0})
    res = run(scenarios.burst(ctx, target="llm", levels=[1, 2, 3, 4], rounds=1))
    assert "llm" in res["dead"] and len(res["levels"]) < 4
    r = run(scenarios.ramp(ctx, target="llm"))["ramp"]
    assert r["fail"] == 1 and "exited" in r["fail_reason"]


def test_idle_writes_samples(tmp_path):
    ctx = _ctx(tmp_path, {"llm": {"adapter": "openai", "url": "http://127.0.0.1:9", "pids": [os.getpid()]}},
               nvml=FakeNvml({os.getpid(): 2048.0}), idle={"duration_s": 0.3})
    res = run(scenarios.idle(ctx))
    assert res["kind"] == "idle" and res["records"] == [] and len(res["samples"]) >= 5
    assert res["summary"]["services"]["llm"]["mib_median"] == 2048.0
    assert os.path.exists(tmp_path / "run" / "idle.json")


def test_sessions_stagger_start(openai_mock, tmp_path):
    """stagger_start false: every session's first turn starts at t0; true: at random moments within
    the first session_duration_s."""
    starts = {}
    for stagger in (False, True):
        ctx = _ctx(tmp_path, {"emb": {"adapter": "openai", "url": openai_mock.url, "kind": "embeddings"}},
                   sessions={"turns_per_session": 1, "session_duration_s": 1.0, "duration_s": 1.0,
                             "profiles": {"many": 8}, "stagger_start": stagger})
        res = run(scenarios.sessions(ctx, profile="many"))
        assert res["params"]["stagger_start"] is stagger and res["stats"]["turns"] == 8
        starts[stagger] = sorted(r["t_busy"] - res["t0"] for r in res["records"])
    assert starts[False][-1] < 0.2
    assert starts[True][-1] - starts[True][0] > 0.2 and starts[True][-1] < 1.0
