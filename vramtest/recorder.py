"""Call records and per-service "in flight" counters.

Every adapter call produces one :class:`CallRecord`. The :class:`Recorder` collects them and keeps a
live count of requests currently being processed by each service (sampled by the GPU monitor).
Service names are free-form: whatever keys the user defined under ``services:``.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field


@dataclass
class CallRecord:
    service: str
    t_start: float               # call started (time.monotonic)
    t_busy: float                # service starts processing (client-streamed audio: after the last chunk)
    t_end: float
    ok: bool
    code: str = "OK"             # "OK", "HTTP_503", gRPC code name, "TIMEOUT", exception type, ...
    error: str = ""
    ttfb_s: float | None = None  # time to first byte/token of a streamed response, from t_busy
    tokens: int = 0              # KV-cache tokens held during the call (prompt + output; usage or estimate)
    extra: dict = field(default_factory=dict)   # prompt_tokens, output_tokens, audio_s, rtf, ...

    @property
    def latency_s(self) -> float:
        return self.t_end - self.t_busy

    def to_dict(self) -> dict:
        d = asdict(self)
        d["latency_s"] = self.latency_s
        return d


class Recorder:
    """Collects CallRecords; ``inflight[service]`` = calls between ``busy()`` and ``done()``."""

    def __init__(self, services=()):
        self.records: list[CallRecord] = []
        self.inflight: dict[str, int] = defaultdict(int)
        for name in services:
            self.inflight[name] = 0

    def busy(self, service: str) -> float:
        """Mark the start of processing; returns the timestamp to use as ``t_busy``."""
        self.inflight[service] += 1
        return time.monotonic()

    def done(self, record: CallRecord) -> CallRecord:
        """Store a finished call (``busy`` must have been called for it)."""
        self.inflight[record.service] -= 1
        self.records.append(record)
        return record

    def snapshot(self) -> dict[str, int]:
        return dict(self.inflight)
