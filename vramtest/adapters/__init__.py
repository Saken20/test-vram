"""Service adapters: ``make_adapter(name, svc_cfg, recorder, tokens)`` -> Adapter.

Adapters are imported lazily so that, e.g., an OpenAI-only setup does not need gRPC to be importable.
"""
from __future__ import annotations

import importlib

from .base import Adapter, CallState, Payload, Result, ServiceError

ADAPTERS = {
    "openai": ("vramtest.adapters.openai", "OpenAIAdapter"),
    "grpc": ("vramtest.adapters.grpc", "GrpcAdapter"),
    "http": ("vramtest.adapters.http", "HttpAdapter"),
}


def adapter_class(adapter: str) -> type[Adapter]:
    if adapter not in ADAPTERS:
        raise ValueError(f"unknown adapter {adapter!r} (expected one of {', '.join(ADAPTERS)})")
    module, cls = ADAPTERS[adapter]
    return getattr(importlib.import_module(module), cls)


def make_adapter(name: str, svc_cfg: dict, recorder=None, tokens=None, *, audio_in: dict | None = None,
                 system_prompt: str = "") -> Adapter:
    """Build the adapter for one service.

    recorder: a :class:`vramtest.recorder.Recorder` (a private one is created when None);
    tokens: a :class:`vramtest.kv_calc.TokenCounter` for KV-token estimates;
    audio_in: ``dataset.audio_in`` ({format, sample_rate}) describing ``Payload.audio``;
    system_prompt: used when a chat payload has only text.
    """
    return adapter_class(svc_cfg["adapter"])(name, svc_cfg, recorder, tokens, audio_in=audio_in,
                                             system_prompt=system_prompt)


__all__ = ["ADAPTERS", "Adapter", "CallState", "Payload", "Result", "ServiceError", "adapter_class", "make_adapter"]
