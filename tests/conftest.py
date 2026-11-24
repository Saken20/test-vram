import asyncio
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.mocks.openai_mock import OpenAIMock  # noqa: E402

ECHO_PROTO = os.path.join(ROOT, "examples", "protos", "echo.proto")


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def openai_mock():
    mock = OpenAIMock().start()
    yield mock
    mock.stop()


@pytest.fixture(autouse=True)
def _no_proxy_env(monkeypatch):
    # adapters must ignore proxies for local services; make sure a proxy in the environment would break them
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
