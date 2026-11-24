"""Every examples/*.yaml must load and validate (no network, no running services)."""
import glob
import os
import re

import pytest

from vramtest import __main__ as cli
from vramtest import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = sorted(glob.glob(os.path.join(ROOT, "examples", "*.yaml")))


def test_examples_exist():
    names = {os.path.basename(p) for p in EXAMPLES}
    assert {"vllm-chat.yaml", "voice-pipeline.yaml", "grpc-service.yaml", "http-service.yaml"} <= names


@pytest.mark.parametrize("path", EXAMPLES, ids=os.path.basename)
def test_example_is_valid(path, capsys):
    cfg = config.load(path)
    assert cfg["_warnings"] == []
    assert cfg["services"]
    for step in cfg["pipeline"]:
        assert step["service"] in cfg["services"]
    for target in cfg["capacity"]["targets"]:
        assert target["label"] and target["sessions"] > 0
    # the CLI `validate` command also compiles .proto files and loads the dataset
    assert cli.validate(cfg) == 0, capsys.readouterr().err


@pytest.mark.parametrize("path", EXAMPLES, ids=os.path.basename)
def test_example_is_portable(path):
    """Examples must not depend on one machine: no absolute paths, no e-mail addresses."""
    text = open(path, encoding="utf-8").read()
    assert not re.search(r"(?<![\w:/])/(home|root|Users|workspace|mnt)/", text)
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)


def test_voice_pipeline_shape():
    cfg = config.load(os.path.join(ROOT, "examples", "voice-pipeline.yaml"))
    svc = cfg["services"]
    assert [s["input"] for s in cfg["pipeline"]] == ["audio", "prev", "prev"]
    assert svc["asr"]["kind"] == "transcription" and svc["tts"]["kind"] == "speech"
    assert svc["llm"]["engine"] == "vllm" and svc["tts"]["engine"] == "generic"
    assert set(cfg["scenarios"]["sessions"]["profiles"]) == {"normal", "peak"}


def test_grpc_example_shapes():
    from vramtest import protos
    cfg = config.load(os.path.join(ROOT, "examples", "grpc-service.yaml"))
    shapes = {}
    for name, s in cfg["services"].items():
        m = protos.load(config.resolve(s["proto"])).method(s["method"])
        shapes[name] = (m.client_streaming, m.server_streaming)
    assert shapes == {"echo_text": (False, False), "echo_stream": (False, True), "echo_audio": (True, False)}
    assert cfg["services"]["echo_audio"]["stream_audio"]["field"] == "chunk"
