import pytest

from vramtest import config


def _errors(raw):
    with pytest.raises(config.ConfigError) as exc:
        config.from_dict(raw)
    return "\n".join(exc.value.errors)


def test_defaults_and_derived_values():
    cfg = config.from_dict({
        "services": {
            "llm": {"adapter": "openai", "url": "http://localhost:8000/v1"},
            "tts": {"adapter": "openai", "url": "http://localhost:8880", "kind": "speech"},
            "g": {"adapter": "grpc", "address": "localhost:50051", "proto": "examples/protos/echo.proto",
                  "method": "echo.Echo/Unary", "stream_audio": {"field": "chunk"}},
            "h": {"adapter": "http", "url": "http://127.0.0.1:9000/generate"},
        },
        "pipeline": [{"service": "llm", "input": "text"}, {"service": "tts", "input": "prev"}],
    })
    llm, tts, g, h = (cfg["services"][n] for n in ("llm", "tts", "g", "h"))
    assert llm["kind"] == "chat" and llm["engine"] == "vllm" and llm["port"] == 8000 and llm["name"] == "llm"
    assert tts["engine"] == "generic" and tts["port"] == 8880
    assert g["port"] == 50051 and g["stream_audio"] == {"field": "chunk", "chunk_bytes": 15360, "realtime": False}
    assert h["port"] == 9000 and h["method"] == "POST"
    assert llm["limits"]["max_p95_latency_s"] == 20
    assert cfg["dataset"]["audio_in"] == {"format": "f32le", "sample_rate": 16000}
    assert cfg["scenarios"]["sessions"]["profiles"] == {"normal": 4, "peak": 16}
    assert cfg["_warnings"] == []


def test_partial_overrides_merge_with_defaults():
    cfg = config.from_dict({"services": {"x": {"adapter": "openai", "url": "http://h:1", "limits": {"max_p95_ttfb_s": 1}}},
                            "scenarios": {"ramp": {"levels": [1, 2]}}})
    assert cfg["services"]["x"]["limits"] == {"max_p95_latency_s": 20, "max_p95_ttfb_s": 1}
    assert cfg["scenarios"]["ramp"]["levels"] == [1, 2] and cfg["scenarios"]["ramp"]["cooldown_s"] == 5


def test_unknown_adapter():
    assert "services.x.adapter: unknown adapter 'rest'" in _errors({"services": {"x": {"adapter": "rest"}}})


def test_missing_required_fields():
    err = _errors({"services": {"a": {"adapter": "openai"}, "g": {"adapter": "grpc"}, "h": {"adapter": "http"}}})
    assert "services.a.url: required" in err
    assert "services.g.address: required" in err
    assert "services.g.proto: required" in err
    assert "services.g.method: required" in err
    assert "services.h.url: required" in err


def test_bad_kind_method_and_proto():
    err = _errors({"services": {
        "a": {"adapter": "openai", "url": "http://h:1", "kind": "vision"},
        "g": {"adapter": "grpc", "address": "h:1", "proto": "nope.proto", "method": "NoSlash"},
    }})
    assert "services.a.kind: 'vision' is not one of" in err
    assert "services.g.proto: file not found" in err
    assert "services.g.method: 'NoSlash' must look like" in err


def test_pipeline_errors():
    err = _errors({"services": {"a": {"adapter": "openai", "url": "http://h:1"}},
                   "pipeline": [{"service": "a", "input": "prev"}, {"service": "zzz", "input": "text"},
                                {"service": "a", "input": "video"}]})
    assert "pipeline[0].input: the first step has no previous step" in err
    assert "pipeline[1].service: unknown service 'zzz'" in err
    assert "pipeline[2].input: 'video' is not one of" in err


def test_no_services_and_bad_values():
    assert "define at least one service" in _errors({})
    err = _errors({"services": {"a": {"adapter": "openai", "url": "http://h:1", "engine": "trt", "timeout_s": 0}},
                   "dataset": {"audio_in": {"format": "mp3", "sample_rate": 16000}}})
    assert "services.a.engine" in err and "services.a.timeout_s" in err and "dataset.audio_in.format" in err


def test_unknown_keys_are_warnings():
    cfg = config.from_dict({"services": {"a": {"adapter": "openai", "url": "http://h:1", "modle": "x"}}, "extra": 1})
    assert any("modle" in w for w in cfg["_warnings"]) and any("extra" in w for w in cfg["_warnings"])


def test_load_file(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("services:\n  llm: {adapter: openai, url: 'http://localhost:8000'}\n")
    cfg = config.load(str(path))
    assert cfg["_path"] == str(path) and cfg["services"]["llm"]["port"] == 8000
    with pytest.raises(config.ConfigError, match="not found"):
        config.load(str(tmp_path / "missing.yaml"))
    path.write_text("services: [\n")
    with pytest.raises(config.ConfigError, match="YAML syntax"):
        config.load(str(path))


def test_apply_quick():
    cfg = config.apply_quick(config.from_dict({"services": {"a": {"adapter": "openai", "url": "http://h:1"}}}))
    assert cfg["scenarios"]["idle"]["duration_s"] == 10 and cfg["scenarios"]["ramp"]["bisect_steps"] == 0


def test_history_turns_range():
    svc = {"a": {"adapter": "openai", "url": "http://h:1"}}
    for bad in ([3, 1], -1, [-1, 2]):
        assert "dataset.history_turns" in _errors({"services": svc, "dataset": {"history_turns": bad}})
    for good in (0, 2, [0, 0], [1, 3]):
        config.from_dict({"services": svc, "dataset": {"history_turns": good}})
