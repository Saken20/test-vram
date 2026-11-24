"""Run context shared by all commands, and the ``probe`` command.

:class:`Context` resolves, once per run, everything that does not change during the run:

- ``cfg``            the validated config dict (see vramtest.config)
- ``run_dir``        results/<run>/ (created) or None
- ``nvml``           :class:`vramtest.gpu_monitor.Nvml` (``available`` is False without a GPU)
- ``gpu_procs``      {pid: MiB} of GPU processes at start
- ``service_pids``   {service: [pid, ...]} GPU processes of each service; ``pid_notes`` explains how found
- ``metrics_urls``   {service: vLLM /metrics URL} for ``engine: vllm`` services where one was found
- ``kv``             {service: kv_calc.kv_info(...)} (bytes/token, decoder params, weights) for vllm services
- ``tokens``         {service: TokenCounter}
- ``dataset``        :class:`vramtest.dataset.Dataset`
- ``pipeline``       cfg["pipeline"] (list of {service, input})

Adapters are created per scenario (each with the scenario's Recorder) via :meth:`Context.make_adapters`.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
from urllib.parse import urlsplit

import numpy as np

from . import audio as audio_util
from . import kv_calc, procinfo, vllm_metrics
from .adapters import Adapter, Payload, make_adapter
from .adapters.openai import base_url
from .dataset import Dataset, gen_audio
from .gpu_monitor import GpuMonitor, Nvml
from .pipeline import run_turn, single_steps, step_payload
from .recorder import Recorder


def discover_metrics_url(pids, first=()) -> str:
    """Find a vLLM /metrics endpoint: candidate URLs ``first``, then ports the service processes
    (and their parents) listen on."""
    candidates = list(first)
    ports = set()
    for pid in pids:
        for p in [pid, *procinfo.ancestors(pid)]:
            ports.update(procinfo.listening_ports(p))
    candidates += [f"http://127.0.0.1:{port}/metrics" for port in sorted(ports)]
    for url in candidates:
        try:
            if vllm_metrics.looks_like_vllm(vllm_metrics.fetch(url, timeout=1.0)):
                return url
        except Exception:  # noqa: BLE001 — the port may be gRPC or something else
            continue
    return ""


def find_metrics_url(svc: dict, pids) -> str:
    """Auto-discover a service's vLLM /metrics: {url}/metrics first (openai/http), then listening ports."""
    first = []
    if svc["adapter"] in ("openai", "http") and svc.get("url"):
        parts = urlsplit(svc["url"])
        first = [f"{base_url(svc['url'])}/metrics", f"{parts.scheme}://{parts.netloc}/metrics"]
    port_pids = set(pids)
    if svc.get("port"):
        port_pids |= procinfo.pids_listening_on(svc["port"])
    return discover_metrics_url(port_pids, dict.fromkeys(first))


def _close_now(adapter: Adapter) -> None:
    """Close an adapter from synchronous code (inside a running event loop the close is scheduled)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(adapter.close())
        return
    task = loop.create_task(adapter.close())
    _CLOSING.add(task)
    task.add_done_callback(_CLOSING.discard)


_CLOSING: set = set()  # keeps scheduled close tasks referenced until they finish


class Context:
    def __init__(self, cfg: dict, run_dir: str | None = None, *, nvml=None, kv: bool = True):
        self.cfg = cfg
        self.run_dir = run_dir
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)
        self.services: dict[str, dict] = cfg["services"]
        self.pipeline: list[dict] = cfg.get("pipeline") or []
        self.nvml = nvml if nvml is not None else Nvml(cfg["gpu"]["index"])
        if not self.nvml.available:
            print(f"[warn] {self.nvml.error} — VRAM will not be measured, only load and latency")
        self.gpu_procs = self.nvml.processes()
        self.service_pids: dict[str, list[int]] = {}
        self.pid_notes: dict[str, str] = {}
        self.metrics_urls: dict[str, str] = {}
        self.kv: dict[str, dict] = {}
        self.tokens = {name: kv_calc.TokenCounter(s.get("tokenizer") or "", s.get("chars_per_token") or 4.0)
                       for name, s in self.services.items()}
        for name, svc in self.services.items():
            pids, note = procinfo.resolve_service_pids(svc, self.gpu_procs.keys())
            self.service_pids[name] = sorted(pids)
            self.pid_notes[name] = note
            if svc["engine"] != "vllm":
                continue
            url = svc.get("metrics_url") or find_metrics_url(svc, pids)
            if url:
                self.metrics_urls[name] = url
            if kv:
                self.kv[name] = kv_calc.kv_info(svc)
        self.dataset = Dataset(cfg)

    # ------------------------------------------------------------ adapters / steps
    def make_adapter(self, name: str, recorder: Recorder | None = None) -> Adapter:
        return make_adapter(name, self.services[name], recorder, self.tokens[name],
                            audio_in=self.dataset.audio_in, system_prompt=self.dataset.system_prompt)

    def make_adapters(self, recorder: Recorder | None = None, names=None) -> dict[str, Adapter]:
        """{name: Adapter} sharing one recorder (default: all services)."""
        recorder = recorder if recorder is not None else Recorder(self.services)
        return {n: self.make_adapter(n, recorder) for n in (names or self.services)}

    def steps(self, target: str | None = None) -> list[dict]:
        """Turn steps: ``"pipeline"`` (or None with a pipeline defined) -> the pipeline; a service name ->
        that service alone with its natural input."""
        if target in (None, "pipeline"):
            if not self.pipeline:
                raise ValueError("no pipeline defined in the config; pass --service NAME")
            return self.pipeline
        if target not in self.services:
            raise ValueError(f"unknown service {target!r} (defined: {', '.join(self.services)})")
        adapter = self.make_adapter(target)  # only to read its natural input
        try:
            return single_steps(adapter)
        finally:
            _close_now(adapter)

    def default_targets(self) -> list[str]:
        """What scenarios load by default: the pipeline if defined, else every service separately."""
        return ["pipeline"] if self.pipeline else list(self.services)

    def step_services(self, steps: list[dict]) -> list[str]:
        return list(dict.fromkeys(s["service"] for s in steps))

    def needs_audio(self, steps: list[dict]) -> bool:
        return any(s.get("input") == "audio" for s in steps)

    def speech_services(self) -> list[str]:
        """Services that can synthesize audio from text (candidates for gen-audio)."""
        out = []
        for name, s in self.services.items():
            if (s["adapter"] == "openai" and s.get("kind") == "speech") or \
                    (s["adapter"] in ("grpc", "http") and (s.get("response") or {}).get("audio")):
                out.append(name)
        return out

    def monitor(self, recorder: Recorder | None = None) -> GpuMonitor:
        """A (not yet started) GPU monitor for this run's services."""
        g = self.cfg["gpu"]
        return GpuMonitor(self.nvml, self.service_pids, self.metrics_urls, recorder, g["sample_hz"],
                          g["metrics_poll_s"])

    def service_info(self) -> dict:
        _, total = self.nvml.memory()
        return {
            "gpu_name": getattr(self.nvml, "name", None),
            "gpu_total_mib": total,
            "services": {n: {k: s.get(k) for k in ("adapter", "kind", "model", "engine", "url", "address",
                                                   "method")} for n, s in self.services.items()},
            "pipeline": self.pipeline,
            "pids": self.service_pids,
            "pid_notes": self.pid_notes,
            "metrics_urls": self.metrics_urls,
            "kv": self.kv,
            "tokens_exact": {n: t.exact for n, t in self.tokens.items()},
        }

    def payload(self, rng: random.Random, session: int = 0, history=None) -> Payload:
        return self.dataset.payload(rng, session, history)

    async def gen_audio(self, service: str | None = None) -> int:
        """Synthesize the dataset prompts into dataset.audio_dir with a speech-capable service."""
        service = service or next(iter(self.speech_services()), None)
        if not service:
            raise ValueError("no speech-capable service (openai kind: speech, or response.audio) — "
                             "pass --service NAME or put your own 16-bit WAV files into dataset.audio_dir")
        adapter = self.make_adapter(service)
        try:
            made = await gen_audio(adapter, self.dataset.texts, self.dataset.audio_dir,
                                   self.dataset.audio_in["sample_rate"])
        finally:
            await adapter.close()
        self.dataset.reload_clips()
        return made

    async def ensure_audio(self, steps: list[dict] | None = None) -> None:
        """Make sure audio clips exist when ``steps`` (default: all known steps) take audio input."""
        if self.dataset.clips:
            return
        if steps is not None and not self.needs_audio(steps):
            return
        print(f"[data] no WAV files in {self.dataset.audio_dir} — synthesizing the prompts ...")
        await self.gen_audio()
        if not self.dataset.clips:
            raise RuntimeError(f"no audio input: put 16-bit PCM WAV files into {self.dataset.audio_dir}")


def _probe_payload(ctx: Context, adapter: Adapter, rng) -> tuple[Payload, str]:
    """Turn payload for a probe (``adapter`` = first service called); a 1 s synthetic tone when it needs
    audio and no clips exist."""
    p = ctx.payload(rng)
    note = ""
    if adapter.default_input == "audio" and p.audio is None:
        sr = ctx.dataset.audio_in["sample_rate"]
        t = np.arange(sr, dtype=np.float32) / sr
        tone = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        p.audio = audio_util.encode(tone, ctx.dataset.audio_in["format"])
        p.audio_seconds = 1.0
        note = " (synthetic 1 s tone: no WAV clips yet)"
    return p, note


def _fmt(v, digits=3):
    return "-" if v is None else f"{v:.{digits}f}"


async def probe(ctx: Context) -> dict:
    """One call per service (and one pipeline turn), GPU processes, vLLM metrics, KV bytes/token.

    Prints a readable summary and returns it as a dict (also saved to <run_dir>/probe.json).
    """
    rng = random.Random(0)
    out = {"gpu": {"available": ctx.nvml.available, "error": ctx.nvml.error, "name": getattr(ctx.nvml, "name", None)},
           "services": {}, "pipeline": None}
    used, total = ctx.nvml.memory()
    if ctx.nvml.available:
        print(f"GPU: {out['gpu']['name']}  used {used:.0f} / {total:.0f} MiB, processes: "
              f"{', '.join(f'{p}={m and round(m)}MiB' for p, m in ctx.gpu_procs.items()) or 'none'}")
    else:
        print(f"GPU: not available ({ctx.nvml.error})")
    recorder = Recorder(ctx.services)
    adapters = {}
    try:
        for name in ctx.services:
            svc = ctx.services[name]
            info = {"pids": ctx.service_pids[name], "pid_note": ctx.pid_notes[name],
                    "metrics_url": ctx.metrics_urls.get(name), "kv": ctx.kv.get(name)}
            print(f"\n[{name}] adapter={svc['adapter']}" + (f" kind={svc['kind']}" if svc["adapter"] == "openai" else "")
                  + f" engine={svc['engine']}")
            try:
                adapters[name] = adapter = ctx.make_adapter(name, recorder)
            except Exception as exc:  # noqa: BLE001 — e.g. proto does not compile
                info["error"] = f"{type(exc).__name__}: {exc}"
                print(f"  adapter: FAILED {info['error']}")
                out["services"][name] = info
                continue
            payload, note = _probe_payload(ctx, adapter, rng)
            result = await adapter.call(step_payload(adapter.default_input, payload, None, adapter))
            r = result.record
            info["call"] = r.to_dict()
            status = "OK" if r.ok else f"FAILED {r.code} {r.error[:200]}"
            print(f"  call ({adapter.default_input}{note}): {status}  latency={_fmt(r.latency_s)}s "
                  f"ttfb={_fmt(r.ttfb_s)}s tokens={r.tokens}")
            if result.text:
                print(f"  output text: {result.text[:120]!r}")
            if result.audio:
                print(f"  output audio: {len(result.audio)} bytes, {_fmt(result.audio_seconds, 2)} s")
            mib = [ctx.gpu_procs.get(p) for p in ctx.service_pids[name]]
            print(f"  GPU processes: {ctx.service_pids[name] or 'none'} ({ctx.pid_notes[name]})"
                  + (f", {sum(m for m in mib if m):.0f} MiB" if any(mib) else ""))
            if svc["engine"] == "vllm":
                url = ctx.metrics_urls.get(name)
                if url:
                    try:
                        m = vllm_metrics.summarize(vllm_metrics.fetch(url))
                    except Exception as exc:  # noqa: BLE001
                        m = {"error": str(exc)}
                    info["metrics"] = m
                    print(f"  vLLM metrics: {url}  kv_usage={m.get('kv_usage')} pool_tokens={m.get('pool_tokens')}")
                else:
                    print("  vLLM metrics: not found (set metrics_url)")
                kv = ctx.kv.get(name) or {}
                if kv.get("bytes_per_token"):
                    print(f"  KV cache: {kv['bytes_per_token']} bytes/token ({kv.get('source')})"
                          + (f", weights {kv['weights_gib']:.2f} GiB ({kv['weights_source']})"
                             if kv.get("weights_gib") else "")
                          + (f"; {kv['note']}" if kv.get("note") and kv.get("source") != "config.kv_bytes_per_token"
                             else ""))
                elif kv.get("error"):
                    print(f"  KV cache: unknown ({kv['error']})")
                else:
                    print("  KV cache: unknown (set model_config or kv_bytes_per_token)")
            out["services"][name] = info
        if ctx.pipeline and all(s["service"] in adapters for s in ctx.pipeline):
            payload, note = _probe_payload(ctx, adapters[ctx.pipeline[0]["service"]], rng)
            turn = await run_turn(adapters, ctx.pipeline, payload)
            out["pipeline"] = turn.to_dict()
            steps = " -> ".join(f"{s.service}:{s.record.code} {s.record.latency_s:.2f}s" for s in turn.steps)
            print(f"\n[pipeline] {steps}  wait for first output: {_fmt(turn.wait_first_output_s, 2)} s{note}")
    finally:
        for a in adapters.values():
            await a.close()
    if ctx.run_dir:
        with open(os.path.join(ctx.run_dir, "probe.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, default=str, indent=1)
    return out
