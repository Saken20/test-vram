"""One "turn" = the configured pipeline steps in order (e.g. ASR -> LLM -> TTS), with per-step timings.

Step inputs:
  audio     the turn's audio clip (Payload.audio)
  text      the turn's text
  messages  chat messages (system prompt + random history + text), built by the dataset
  prev      the previous step's output (text and/or audio). For a chat step the previous text replaces
            the last user message, so the system prompt and history are kept.

When a step fails, the next ``prev`` step falls back to the turn's own text/audio so that the
remaining services still get load (the turn is reported as failed).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .adapters.base import Adapter, Payload, Result
from .recorder import CallRecord


@dataclass
class StepResult:
    service: str
    input: str
    result: Result

    @property
    def record(self) -> CallRecord:
        return self.result.record

    def to_dict(self) -> dict:
        r = self.record
        return {
            "service": self.service,
            "input": self.input,
            "ok": r.ok,
            "code": r.code,
            "latency_s": r.latency_s,
            "ttfb_s": r.ttfb_s,
            "tokens": r.tokens,
            "audio_in_s": r.extra.get("audio_in_s"),
            "audio_out_s": r.extra.get("audio_out_s"),
        }


@dataclass
class TurnResult:
    steps: list[StepResult] = field(default_factory=list)
    wait_first_output_s: float | None = None
    session: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.steps) and all(s.record.ok for s in self.steps)

    @property
    def records(self) -> list[CallRecord]:
        return [s.record for s in self.steps]

    @property
    def errors(self) -> list[str]:
        return [f"{s.service}:{s.record.code}" for s in self.steps if not s.record.ok]

    @property
    def t(self) -> float | None:
        return min((r.t_busy for r in self.records), default=None)

    @property
    def output(self) -> Result | None:
        return self.steps[-1].result if self.steps else None

    def to_dict(self) -> dict:
        return {
            "t": self.t,
            "session": self.session,
            "ok": self.ok,
            "errors": self.errors,
            "steps": [s.to_dict() for s in self.steps],
            "wait_first_output_s": self.wait_first_output_s,
        }


def wait_first_output(records: list[CallRecord]) -> float | None:
    """Sum of each step's latency, except the last step contributes its TTFB (or latency without TTFB).

    = time from "the input is complete" (user stopped speaking / request sent) to the first output.
    """
    if not records:
        return None
    last = records[-1]
    tail = last.ttfb_s if last.ttfb_s is not None else last.latency_s
    return sum(r.latency_s for r in records[:-1]) + tail


def step_payload(kind: str, base: Payload, prev: Result | None, adapter: Adapter) -> Payload:
    """The payload a step with input ``kind`` receives, from the turn payload and the previous result."""
    if kind == "audio":
        return Payload(audio=base.audio, audio_seconds=base.audio_seconds, session=base.session)
    if kind == "text":
        return Payload(text=base.text, session=base.session)
    if kind == "messages":
        return Payload(text=base.text, messages=base.messages, session=base.session)
    # prev
    if prev is None or not prev.ok or (prev.text is None and prev.audio is None):
        return Payload(text=base.text, audio=base.audio, audio_seconds=base.audio_seconds,
                       messages=base.messages, session=base.session)
    messages = None
    if prev.text is not None and base.messages and adapter.default_input == "messages":
        messages = [*base.messages[:-1], {"role": "user", "content": prev.text}]
    return Payload(text=prev.text, audio=prev.audio, audio_seconds=prev.audio_seconds if prev.audio else None,
                   messages=messages, session=base.session)


async def run_turn(adapters: dict[str, Adapter], steps: list[dict], payload: Payload,
                   *, stop_on_error: bool = False) -> TurnResult:
    """Run one turn through ``steps`` ([{service, input}, ...]) using ``adapters`` ({name: Adapter})."""
    turn = TurnResult(session=payload.session)
    prev = None
    for step in steps:
        adapter = adapters[step["service"]]
        kind = step.get("input") or adapter.default_input
        result = await adapter.call(step_payload(kind, payload, prev, adapter))
        turn.steps.append(StepResult(step["service"], kind, result))
        prev = result
        if stop_on_error and not result.ok:
            break
    if len(turn.steps) == len(steps):
        turn.wait_first_output_s = wait_first_output(turn.records)
    return turn


def single_steps(adapter: Adapter) -> list[dict]:
    """Steps for loading one service alone, with its natural input (audio | text | messages)."""
    return [{"service": adapter.name, "input": adapter.default_input}]


async def run_single(adapter: Adapter, payload: Payload) -> TurnResult:
    """Single-service mode: one call with the service's natural input."""
    return await run_turn({adapter.name: adapter}, single_steps(adapter), payload)
