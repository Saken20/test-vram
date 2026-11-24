import asyncio
import os

import pytest
import yaml

from tests.conftest import ECHO_PROTO, run
from tests.fake_nvml import FakeNvml
from tests.mocks.grpc_mock import start_echo
from vramtest import config, context
from vramtest.__main__ import main, parse_args
from vramtest.gpu_monitor import GpuMonitor
from vramtest.recorder import Recorder


def test_cli_common_flags_before_or_after_subcommand():
    before = parse_args(["--quick", "--run", "x", "all", "--skip-ramp"])
    after = parse_args(["all", "--quick", "--run", "x", "--skip-ramp"])
    assert (before.quick, before.run, before.cmd) == (after.quick, after.run, after.cmd) == (True, "x", "all")
    plain = parse_args(["report"])
    assert (plain.quick, plain.run, plain.config) == (False, "default", None)
    b = parse_args(["burst", "--levels", "1,5", "--service", "llm", "--config", "c.yaml"])
    assert b.levels == [1, 5] and b.service == "llm" and b.config == "c.yaml"
    assert parse_args(["ramp"]).service == "all" and parse_args(["sessions"]).profile == "all"


def test_cli_rejects_unknown_service(tmp_path, capsys):
    path = _write(tmp_path, {"services": {"llm": {"adapter": "openai", "url": "http://127.0.0.1:9"}}})
    for argv in (["burst", "--service", "nope"], ["ramp", "--service", "pipeline"], ["gen-audio", "--service", "all"]):
        with pytest.raises(SystemExit) as exc:
            main(["--config", path, *argv])
        assert "unknown --service" in str(exc.value.code)


def _write(tmp_path, data):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def test_validate_command(tmp_path, capsys):
    path = _write(tmp_path, {
        "services": {"llm": {"adapter": "openai", "url": "http://localhost:8000", "model": "org/model"},
                     "echo": {"adapter": "grpc", "address": "localhost:50051", "proto": ECHO_PROTO,
                              "method": "echo.Echo/ServerStream"}},
        "pipeline": [{"service": "llm", "input": "messages"}, {"service": "echo", "input": "prev"}],
        "dataset": {"audio_dir": str(tmp_path / "audio")}})
    with pytest.raises(SystemExit) as exc:
        main(["--config", path, "validate"])
    out = capsys.readouterr().out
    assert exc.value.code == 0 and "/echo.Echo/ServerStream (unary_stream)" in out and "llm(messages) -> echo(prev)" in out

    bad = _write(tmp_path, {"services": {"x": {"adapter": "grpc", "address": "h:1", "proto": ECHO_PROTO,
                                               "method": "echo.Echo/Missing"}}})
    with pytest.raises(SystemExit) as exc:
        main(["--config", bad, "validate"])
    assert exc.value.code == 1 and "not found" in capsys.readouterr().err

    invalid = _write(tmp_path, {"services": {"x": {"adapter": "nope"}}})
    with pytest.raises(SystemExit) as exc:
        main(["--config", invalid, "validate"])
    assert exc.value.code == 2 and "unknown adapter 'nope'" in capsys.readouterr().err


def test_context_and_probe_with_mocks(openai_mock, tmp_path, capsys):
    pid = os.getpid()

    async def go():
        server, port = await start_echo()
        try:
            cfg = config.from_dict({
                "services": {
                    "asr": {"adapter": "openai", "url": openai_mock.url, "kind": "transcription"},
                    "chat-llm": {"adapter": "openai", "url": openai_mock.url, "pids": [pid],
                                 "kv_bytes_per_token": 1000},
                    "voice": {"adapter": "openai", "url": openai_mock.url, "kind": "speech"},
                    "echo": {"adapter": "grpc", "address": f"127.0.0.1:{port}", "proto": ECHO_PROTO,
                             "method": "echo.Echo/Unary", "request": {"text": "{text}"},
                             "response": {"text": "reply"}},
                },
                "pipeline": [{"service": "asr", "input": "audio"}, {"service": "chat-llm", "input": "prev"},
                             {"service": "voice", "input": "prev"}],
                "dataset": {"audio_dir": str(tmp_path / "audio")},
            })
            ctx = context.Context(cfg, str(tmp_path / "run"), nvml=FakeNvml({pid: 1234.0}))
            assert ctx.service_pids["chat-llm"] == [pid] and ctx.pid_notes["chat-llm"] == "config.pids"
            assert ctx.metrics_urls["chat-llm"] == openai_mock.url + "/metrics"
            assert "voice" not in ctx.metrics_urls  # speech defaults to engine: generic
            assert ctx.kv["chat-llm"]["bytes_per_token"] == 1000
            assert ctx.default_targets() == ["pipeline"] and ctx.steps("voice") == [{"service": "voice", "input": "text"}]
            assert ctx.speech_services() == ["voice"] and ctx.needs_audio(ctx.pipeline)
            info = ctx.service_info()
            assert info["gpu_name"] == "FakeGPU 48GB" and info["kv"]["chat-llm"]["bytes_per_token"] == 1000
            result = await context.probe(ctx)
            await ctx.ensure_audio(ctx.pipeline)
            return ctx, result
        finally:
            await server.stop(None)
    ctx, result = run(go())
    out = capsys.readouterr().out
    assert all(result["services"][n]["call"]["ok"] for n in ("asr", "chat-llm", "voice", "echo"))
    assert result["pipeline"]["ok"] and result["pipeline"]["wait_first_output_s"] > 0
    assert "synthetic 1 s tone" in out and "1000 bytes/token" in out
    assert os.path.exists(tmp_path / "run" / "probe.json")
    assert len(ctx.dataset.clips) == len(ctx.dataset.texts)  # ensure_audio synthesized the prompts


def test_steps_closes_the_adapter_it_creates(openai_mock, monkeypatch):
    from vramtest.adapters.openai import OpenAIAdapter
    closed = []
    orig = OpenAIAdapter.close

    async def close(self):
        closed.append(self.name)
        await orig(self)
    monkeypatch.setattr(OpenAIAdapter, "close", close)
    cfg = config.from_dict({"services": {"voice": {"adapter": "openai", "url": openai_mock.url, "kind": "speech"}}})
    ctx = context.Context(cfg, nvml=FakeNvml({}), kv=False)
    assert ctx.steps("voice") == [{"service": "voice", "input": "text"}] and closed == ["voice"]

    async def inside_loop():
        steps = ctx.steps("voice")
        await asyncio.sleep(0)  # the close is scheduled on the running loop
        return steps
    assert run(inside_loop()) == [{"service": "voice", "input": "text"}] and closed == ["voice", "voice"]


def test_gpu_monitor_dynamic_services(openai_mock):
    rec = Recorder(["a-svc", "b.svc"])
    rec.busy("a-svc")
    mon = GpuMonitor(FakeNvml({1: 100.0, 2: 50.0}), {"a-svc": [1, 2], "b.svc": [999999999]},
                     {"a-svc": openai_mock.url + "/metrics"}, rec, sample_hz=100, metrics_poll_s=0)
    s = mon.sample()
    assert s["svc_mib"] == {"a-svc": 150.0, "b.svc": None} and s["inflight"] == {"a-svc": 1, "b.svc": 0}
    assert s["vllm"]["a-svc"]["pool_tokens"] == 16000 and s["used_mib"] == 650.0
    assert "b.svc" in mon.dead  # that pid does not exist
