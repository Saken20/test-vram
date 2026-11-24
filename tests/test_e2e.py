"""End-to-end: every scenario against the mocks (OpenAI-compatible + gRPC echo) with a fake NVML, then the
markdown/JSON report and the PDF."""
import json
import os

from tests.conftest import ECHO_PROTO, run
from tests.fake_nvml import FakeNvml
from tests.mocks.grpc_mock import start_echo
from tests.mocks.openai_mock import OpenAIMock
from tests.test_report import check_tables
from vramtest import config, pdf_report, report, scenarios
from vramtest.context import Context

KV_BPT = 64 * 1024


def _config(voice_url, llm_url, echo_port, audio_dir):
    return config.from_dict({
        "gpu": {"sample_hz": 50, "metrics_poll_s": 0.05},
        "services": {
            "asr": {"adapter": "openai", "url": voice_url, "kind": "transcription", "model": "whisper-mock",
                    "pids": [os.getppid()], "kv_bytes_per_token": 16 * 1024},
            "llm": {"adapter": "openai", "url": llm_url, "kind": "chat", "model": "chat-mock",
                    "pids": [os.getpid()], "kv_bytes_per_token": KV_BPT, "request": {"max_tokens": 8}},
            "tts": {"adapter": "openai", "url": voice_url, "kind": "speech", "model": "tts-mock",
                    "cmdline_match": "no-such-process-on-this-gpu"},
            "echo": {"adapter": "grpc", "address": f"127.0.0.1:{echo_port}", "proto": ECHO_PROTO,
                     "method": "echo.Echo/ServerStream", "request": {"text": "{text}"},
                     "response": {"text": "reply"}, "cmdline_match": "no-such-process-on-this-gpu"},
        },
        "pipeline": [{"service": "asr", "input": "audio"}, {"service": "llm", "input": "prev"},
                     {"service": "tts", "input": "prev"}],
        "dataset": {"audio_dir": audio_dir, "system_prompt": "You are a helpful assistant. Answer briefly."},
        "scenarios": {
            "idle": {"duration_s": 0.4},
            "burst": {"levels": [1, 4], "rounds": 2, "pause_s": 0},
            "concurrency": {"users": 3, "duration_s": 0.6},
            "sessions": {"turns_per_session": 2, "session_duration_s": 0.4, "duration_s": 0.8,
                         "profiles": {"normal": 1, "peak": 3}},
            "ramp": {"levels": [2, 4, 6, 8], "step_duration_s": 0.8, "max_error_rate": 0.0, "bisect_steps": 1,
                     "cooldown_s": 0.05},
        },
        "capacity": {"unit": "session", "safety_margin": 0.1,
                     "targets": [{"label": "1 session", "sessions": 1}, {"label": "peak", "sessions": 3},
                                 {"label": "large fleet", "sessions": 200}]},
    })


def test_all_scenarios_report_and_pdf(tmp_path):
    voice = OpenAIMock().start()
    llm = OpenAIMock(limit=6).start()
    run_dir = str(tmp_path / "results" / "e2e")

    def procs():
        return {os.getpid(): 20000.0 + 40 * llm.inflight, os.getppid(): 3000.0 + 10 * voice.inflight}

    async def go():
        server, port = await start_echo()
        try:
            cfg = _config(voice.url, llm.url, port, str(tmp_path / "audio"))
            ctx = Context(cfg, run_dir, nvml=FakeNvml(procs))
            out = {"idle": await scenarios.idle(ctx)}
            out["burst"] = await scenarios.burst(ctx, target="pipeline")
            out["burst_echo"] = await scenarios.burst(ctx, target="echo", levels=[1, 3], rounds=1)
            out["conc"] = await scenarios.concurrency(ctx, target="pipeline")
            out["normal"] = await scenarios.sessions(ctx, profile="normal")
            out["peak"] = await scenarios.sessions(ctx, profile="peak")
            out["ramp"] = await scenarios.ramp(ctx, target="llm")
            return cfg, out
        finally:
            await server.stop(None)
            voice.stop()
            llm.stop()

    cfg, out = run(go())
    names = sorted(os.listdir(run_dir))
    for name in ("idle", "burst_pipeline", "burst_echo", "concurrency_pipeline_3", "sessions_normal",
                 "sessions_peak", "ramp_llm"):
        assert f"{name}.json" in names, names

    # burst: per level per-step seconds, wait to first output, VRAM per service
    levels = out["burst"]["levels"]
    assert [lv["level"] for lv in levels] == [1, 4]
    assert levels[1]["stats"]["turns"] == 8 and levels[1]["stats"]["failed"] == 0
    assert [s["service"] for s in levels[1]["stats"]["steps"]] == ["asr", "llm", "tts"]
    assert levels[1]["stats"]["wait_first_output_s"]["p50"] > 0
    assert levels[1]["summary"]["services"]["llm"]["inflight_peak"] == 4
    assert levels[1]["summary"]["services"]["llm"]["mib_peak"] >= 20000
    assert out["burst_echo"]["levels"][1]["stats"]["steps"][0]["service"] == "echo"
    assert out["conc"]["summary"]["services"]["llm"]["inflight_peak"] == 3
    assert out["peak"]["summary"]["sessions"] == 3 and out["peak"]["stats"]["turns"] >= 6

    ramp = out["ramp"]["ramp"]
    assert ramp["last_ok"] == 6 and ramp["fail"] == 7, ramp  # 8 fails, bisection: 7 fails too
    assert "errors" in ramp["fail_reason"]

    md = report.build(cfg, run_dir)
    for section in ("# GPU inference load report", "## 1. Summary", "Pipeline: asr → llm → tts", "Service: echo",
                    "## 2. Idle memory", "## 3. Measured scenarios", "## 4. Limits (ramp)",
                    "## 5. Capacity: sessions", "## 6. Recommended vLLM settings", "**first output**",
                    "4 requests at once", "3 sessions (peak)", "idle VRAM - KV pool"):
        assert section in md, section
    check_tables(md)
    rep = json.load(open(os.path.join(run_dir, "report.json")))
    svc = rep["services"]
    # pool from /metrics: 1000 blocks x 16 tokens; base = idle - pool
    assert svc["llm"]["kv_pool_tokens"] == 16000
    assert abs(svc["llm"]["base_gib"] - (20000 / 1024 - 16000 * KV_BPT / 1024 ** 3)) < 0.05
    assert svc["llm"]["counted"] and svc["asr"]["counted"]
    assert not svc["tts"]["counted"] and not svc["echo"]["counted"]   # not seen on the GPU: not counted
    assert svc["llm"]["limit_inflight"] == 6 and svc["llm"]["duty"] > 0
    assert svc["llm"]["calibration"] is not None
    assert set(rep["summary"][0]["rows"][0]["need_gib"]) == {"asr", "llm"}
    assert [g["target"] for g in rep["summary"]] == ["pipeline", "echo"]
    assert rep["limits"][0]["last_ok"] == 6
    cap = rep["capacity"]
    assert cap["compute_cap"] == int(report.solve_sessions(6, svc["llm"]["duty"]))
    big = cap["targets"][-1]
    assert big["servers"] >= 2 and big["servers_by_compute"] >= 2
    assert cap["targets"][0]["vllm"]["llm"]["fits"]
    assert 0 < cap["targets"][0]["vllm"]["llm"]["gpu_memory_utilization"] < 1

    pdf = pdf_report.build(cfg, run_dir)
    assert pdf.endswith("summary.pdf") and os.path.getsize(pdf) > 3000
    with open(pdf, "rb") as f:
        assert f.read(5) == b"%PDF-"
