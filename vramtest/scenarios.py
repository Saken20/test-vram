"""Load scenarios: idle, burst, concurrency, sessions, ramp.

Every scenario writes ``<run_dir>/<name>.json`` with:

    scenario      file name without .json (e.g. "burst_pipeline", "sessions_peak", "ramp_llm")
    kind          idle | burst | concurrency | sessions | ramp
    target        "pipeline", a service name, "all" (sessions without a pipeline) or None (idle)
    steps         the turn steps [{service, input}] used by the scenario
    params        scenario parameters
    t0, t1        monotonic start / end
    service_info  GPU name/total, service PIDs, vLLM /metrics URLs, KV bytes/token (Context.service_info)
    dead          {service: seconds since t0} for services whose GPU process exited during the scenario
    summary       analysis.summarize_window over the whole scenario (every configured service)
    records       every call (CallRecord.to_dict)
    samples       GPU monitor samples
    turns, stats  turns (TurnResult.to_dict) and analysis.turns_stats (not for idle / ramp)
    levels        burst only: [{level, t_from, t_to, summary, stats, turns}]
    ramp          ramp only: {target, steps: [{level, ok, reason, summary, stats, ...}], last_ok, fail, fail_reason}

A "turn" is one pass through the steps (the pipeline, or a single service with its natural input).
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import random
import time

from . import analysis
from .pipeline import run_turn, single_steps
from .recorder import Recorder

FAIL_BACKOFF_S = 0.2   # a worker whose turn failed waits this long (do not hammer a dead service)


# ------------------------------------------------------------------ helpers
async def _sleep_until(t: float) -> None:
    delay = t - time.monotonic()
    if delay > 0:
        await asyncio.sleep(delay)


def _target_services(ctx, target) -> list[str]:
    """Services a target loads: the pipeline's, one service, or every service ("all")."""
    if target in (None, "pipeline"):
        if not ctx.pipeline:
            raise ValueError("no pipeline defined in the config; pass --service NAME")
        return ctx.step_services(ctx.pipeline)
    if target == "all":
        return list(ctx.services)
    if target not in ctx.services:
        raise ValueError(f"unknown target {target!r} (services: {', '.join(ctx.services)}"
                         f"{', pipeline' if ctx.pipeline else ''})")
    return [target]


def _steps_for(ctx, target, adapters) -> list[dict]:
    """Turn steps for a target, using already created adapters (for the services' natural inputs)."""
    if target in (None, "pipeline"):
        return list(ctx.pipeline)
    if target == "all":  # no pipeline: one call to every service per turn, each with its natural input
        return [single_steps(adapters[name])[0] for name in ctx.services]
    return single_steps(adapters[target])


def _fmt_stat(stat, unit=" s") -> str:
    """"0.81 s (p95 1.40)" from {p50, p95}."""
    if not stat or stat.get("p50") is None:
        return "-"
    return f"{stat['p50']:.2f}{unit} (p95 {stat['p95']:.2f})"


def _mib(v) -> str:
    return "-" if v is None else f"{v:.0f} MiB"


def _turn_line(stats, summary, services) -> str:
    """One concise line: seconds per step, wait until first output, errors, VRAM."""
    parts = []
    for i, st in enumerate(stats["steps"]):
        label = st["service"]
        if sum(1 for s in stats["steps"] if s["service"] == label) > 1:
            label = f"{label}#{i + 1}"
        parts.append(f"{label} {_fmt_stat(st['latency_s'])}")
    if len(stats["steps"]) > 1 or any(st["ttfb_s"]["p50"] is not None for st in stats["steps"]):
        parts.append(f"first output {_fmt_stat(stats['wait_first_output_s'])}")
    parts.append(f"errors {stats['failed']}/{stats['turns']}")
    used = summary["gpu"]["used_mib_peak"]
    if used is not None:
        mem = ", ".join(f"{s} {_mib(summary['services'][s]['mib_peak'])}" for s in services
                        if summary["services"].get(s, {}).get("mib_peak") is not None)
        parts.append(f"GPU {_mib(used)}" + (f" ({mem})" if mem else ""))
    return " · ".join(parts)


def print_summary(title, summary, services) -> None:
    gpu = summary["gpu"]
    print(f"\n=== {title}  ({summary['duration_s']} s)  GPU peak: {_mib(gpu['used_mib_peak'])}")
    for svc in services:
        s = summary["services"].get(svc)
        if not s:
            continue
        lat = s["latency_p95_s"]
        errors = f" {s['error_codes']}" if s["error_codes"] else ""
        kv = f" kv_usage_peak={s['vllm_kv_usage_peak']:.2f}" if s["vllm_kv_usage_peak"] is not None else ""
        print(f"  {svc}: calls={s['calls']} errors={s['errors']}{errors} "
              f"p95={'-' if lat is None else f'{lat:.2f}'}s inflight_peak={s['inflight_peak']} "
              f"kv_tokens_peak={s['kv_tokens_peak']} vram_peak={_mib(s['mib_peak'])}{kv}")


def _dead(monitor, services) -> list[str]:
    return [s for s in services if s in monitor.dead]


async def _run(ctx, name, kind, target, params, body, services):
    """Common frame: recorder, adapters, GPU monitor; writes <run_dir>/<name>.json.

    ``body(adapters, steps, recorder, monitor)`` runs the load and returns extra result keys.
    """
    recorder = Recorder(ctx.services)
    adapters = ctx.make_adapters(recorder, services) if services else {}
    steps = _steps_for(ctx, target, adapters) if services else []
    if steps:
        await ctx.ensure_audio(steps)
    monitor = ctx.monitor(recorder).start()
    started = dt.datetime.now().isoformat(timespec="seconds")
    t0 = time.monotonic()
    extra = {}
    try:
        extra = await body(adapters, steps, recorder, monitor) or {}
    finally:
        t1 = time.monotonic()
        samples = monitor.stop()
        for adapter in adapters.values():
            await adapter.close()
    records = [r.to_dict() for r in recorder.records]
    result = {
        "scenario": name,
        "kind": kind,
        "target": target,
        "steps": steps,
        "params": params,
        "started_at": started,
        "t0": t0,
        "t1": t1,
        "service_info": ctx.service_info(),
        "dead": {k: v - t0 for k, v in monitor.dead.items()},
        "summary": analysis.summarize_window(records, samples, t0, t1, params.get("sessions"), list(ctx.services)),
        "records": records,
        "samples": samples,
        **extra,
    }
    if "turns" in result and "stats" not in result:
        result["stats"] = analysis.turns_stats(result["turns"])
    if ctx.run_dir:
        path = os.path.join(ctx.run_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, default=str)
    print_summary(name, result["summary"], services or list(ctx.services))
    if monitor.dead:
        print(f"[!!] GPU process exited: {', '.join(monitor.dead)}")
    if ctx.run_dir:
        print(f"[saved] {path}")
    return result


async def _worker(ctx, adapters, steps, slot, deadline, turns, seed):
    """Turns back-to-back without pauses until ``deadline``: exactly one turn in flight per worker."""
    rng = random.Random(seed + slot)
    while time.monotonic() < deadline:
        turn = await run_turn(adapters, steps, ctx.payload(rng, session=slot))
        turns.append(turn.to_dict())
        if not turn.ok:
            await asyncio.sleep(FAIL_BACKOFF_S)


def _reply_text(turn) -> str | None:
    """Text the turn produced (last step with text output) — goes into the session's chat history."""
    for step in reversed(turn.steps):
        if step.result.ok and step.result.text:
            return step.result.text
    return None


async def _session(ctx, adapters, steps, slot, deadline, turns, scfg, seed=1000):
    """One session: ``turns_per_session`` turns spread evenly over ``session_duration_s``, repeated until
    the deadline. Chat history grows with the session's own previous turns."""
    n_turns, length = max(1, int(scfg["turns_per_session"])), float(scfg["session_duration_s"])
    rng = random.Random(seed + slot)
    if scfg.get("stagger_start"):  # sessions start at random moments instead of all at once
        await _sleep_until(min(time.monotonic() + rng.uniform(0, length), deadline))
    while time.monotonic() < deadline:
        t_session, history = time.monotonic(), []
        for i in range(n_turns):
            if time.monotonic() >= deadline:
                break
            payload = ctx.payload(rng, session=slot, history=list(history))
            turn = await run_turn(adapters, steps, payload)
            turns.append(turn.to_dict())
            history += [{"role": "user", "content": payload.text or ""},
                        {"role": "assistant", "content": _reply_text(turn) or ""}]
            await _sleep_until(min(t_session + (i + 1) * length / n_turns, deadline))
        await _sleep_until(min(t_session + length, deadline))


def _window(recorder, monitor, t_from, t_to, ctx, sessions=None):
    records = [r.to_dict() for r in recorder.records]
    return analysis.summarize_window(records, list(monitor.samples), t_from, t_to, sessions, list(ctx.services))


# ------------------------------------------------------------------ scenarios
async def idle(ctx):
    """VRAM with no requests."""
    duration = ctx.cfg["scenarios"]["idle"]["duration_s"]
    print(f"[idle] {duration} s without requests ...")

    async def body(adapters, steps, recorder, monitor):
        await asyncio.sleep(duration)

    return await _run(ctx, "idle", "idle", None, {"duration_s": duration, "sessions": 0}, body, [])


async def burst(ctx, target=None, levels=None, rounds=None):
    """N turns started at the same instant, for each N in ``levels`` (``rounds`` bursts per level).

    Per level: seconds of every step, the wait until first output, errors and VRAM per service.
    """
    bcfg = ctx.cfg["scenarios"]["burst"]
    target = target or ctx.default_targets()[0]
    levels = list(levels or bcfg["levels"])
    rounds = rounds or bcfg["rounds"]
    pause = bcfg.get("pause_s", 0)
    services = _target_services(ctx, target)
    print(f"[burst:{target}] simultaneous turns: {levels}, bursts per level: {rounds}")

    async def body(adapters, steps, recorder, monitor):
        out, all_turns = [], []
        for n in levels:
            dead = _dead(monitor, services)
            if dead:
                print(f"[burst:{target}] GPU process of {', '.join(dead)} exited — stopping")
                break
            t_from, turns = time.monotonic(), []
            for r in range(rounds):
                payloads = [ctx.payload(random.Random(3000 + 101 * r + i), session=i) for i in range(n)]
                results = await asyncio.gather(*(run_turn(adapters, steps, p) for p in payloads))
                turns += [t.to_dict() for t in results]
                if pause and (r < rounds - 1 or n != levels[-1]):
                    await asyncio.sleep(pause)
            t_to = time.monotonic()
            win = _window(recorder, monitor, t_from, t_to, ctx)
            stats = analysis.turns_stats(turns)
            out.append({"level": n, "t_from": t_from, "t_to": t_to, "summary": win, "stats": stats, "turns": turns})
            all_turns += turns
            print(f"  N={n}: {_turn_line(stats, win, services)}")
        return {"levels": out, "turns": all_turns}

    return await _run(ctx, f"burst_{target}", "burst", target,
                      {"levels": levels, "rounds": rounds, "pause_s": pause}, body, services)


async def concurrency(ctx, target=None, users=None, duration=None):
    """``users`` workers send turns back-to-back for ``duration`` seconds."""
    ccfg = ctx.cfg["scenarios"]["concurrency"]
    target = target or ctx.default_targets()[0]
    users = users or ccfg["users"]
    duration = duration or ccfg["duration_s"]
    services = _target_services(ctx, target)
    print(f"[concurrency:{target}] {users} users back-to-back, {duration} s ...")

    async def body(adapters, steps, recorder, monitor):
        deadline = time.monotonic() + duration
        turns = []
        await asyncio.gather(*(_worker(ctx, adapters, steps, i, deadline, turns, 2000) for i in range(users)))
        stats = analysis.turns_stats(turns)
        print(f"  {users} users: {_turn_line(stats, _window(recorder, monitor, 0, time.monotonic(), ctx), services)}")
        return {"turns": turns, "stats": stats}

    return await _run(ctx, f"concurrency_{target}_{users}", "concurrency", target,
                      {"users": users, "duration_s": duration}, body, services)


async def sessions(ctx, profile="normal", duration=None, target=None):
    """K concurrent sessions (K = ``scenarios.sessions.profiles[profile]``); each session does
    ``turns_per_session`` turns spread over ``session_duration_s`` (think time between turns), then starts
    the next one, until ``duration`` seconds pass.

    target: default the pipeline; without one, every turn calls each service once ("all").
    """
    scfg = ctx.cfg["scenarios"]["sessions"]
    profiles = scfg["profiles"]
    if profile not in profiles:
        raise ValueError(f"unknown profile {profile!r} (defined: {', '.join(profiles)})")
    k = int(profiles[profile])
    duration = duration or scfg["duration_s"]
    target = target or ("pipeline" if ctx.pipeline else "all")
    services = _target_services(ctx, target)
    print(f"[sessions:{profile}] {k} concurrent sessions, {scfg['turns_per_session']} turns per "
          f"{scfg['session_duration_s']} s session, {duration} s ...")

    async def body(adapters, steps, recorder, monitor):
        deadline = time.monotonic() + duration
        turns = []
        await asyncio.gather(*(_session(ctx, adapters, steps, i, deadline, turns, scfg) for i in range(k)))
        stats = analysis.turns_stats(turns)
        print(f"  {k} sessions: {_turn_line(stats, _window(recorder, monitor, 0, time.monotonic(), ctx), services)}")
        return {"turns": turns, "stats": stats}

    params = {"profile": profile, "sessions": k, "duration_s": duration,
              "turns_per_session": scfg["turns_per_session"], "session_duration_s": scfg["session_duration_s"],
              "stagger_start": bool(scfg.get("stagger_start"))}
    return await _run(ctx, f"sessions_{profile}", "sessions", target, params, body, services)


async def ramp(ctx, target=None, levels=None, step=None):
    """Raise concurrency level by level until errors, a crashed GPU process or the service ``limits``
    (max_p95_latency_s / max_p95_ttfb_s) are exceeded; then refine the limit by bisection.

    A service target uses N back-to-back workers (N = requests in flight). The pipeline target uses N
    concurrent sessions (as in ``sessions``), so its limit is in sessions.
    """
    rcfg = ctx.cfg["scenarios"]["ramp"]
    target = target or ctx.default_targets()[0]
    levels = sorted(set(levels or rcfg["levels"]))
    step_s = step or rcfg["step_duration_s"]
    max_error_rate = rcfg.get("max_error_rate", 0.05)
    bisect_steps = rcfg.get("bisect_steps", 0)
    cooldown = rcfg.get("cooldown_s", 0)
    services = _target_services(ctx, target)
    sessions_mode = target == "pipeline"
    unit = "sessions" if sessions_mode else "in flight"
    print(f"[ramp:{target}] levels {levels} ({unit}), {step_s} s each ...")

    async def run_level(adapters, steps, recorder, monitor, n):
        t_from = time.monotonic()
        deadline = t_from + step_s
        turns = []
        if sessions_mode:
            scfg = ctx.cfg["scenarios"]["sessions"]
            tasks = [_session(ctx, adapters, steps, i, deadline, turns, scfg, seed=5000) for i in range(n)]
        else:
            tasks = [_worker(ctx, adapters, steps, i, deadline, turns, 4000) for i in range(n)]
        await asyncio.gather(*tasks)
        t_to = time.monotonic()
        win = _window(recorder, monitor, t_from, t_to, ctx, n if sessions_mode else None)
        ok, reason = analysis.verdict(win, services, ctx.services, max_error_rate, monitor.dead)
        stats = analysis.turns_stats(turns)
        print(f"  N={n}: {'OK' if ok else 'FAIL (' + reason + ')'} · {_turn_line(stats, win, services)}")
        return {"level": n, "t_from": t_from, "t_to": t_to, "ok": ok, "reason": reason, "summary": win,
                "stats": stats}

    async def body(adapters, steps, recorder, monitor):
        out, last_ok, fail = [], None, None
        for i, n in enumerate(levels):
            if i and cooldown:
                await asyncio.sleep(cooldown)
            st = await run_level(adapters, steps, recorder, monitor, n)
            out.append(st)
            if not st["ok"]:
                fail = n
                break
            last_ok = n
        # refine the limit between the last OK level and the failed one (if the service is still alive)
        lo, hi = last_ok or 0, fail
        for _ in range(bisect_steps):
            if hi is None or hi - lo <= 1 or _dead(monitor, services):
                break
            if cooldown:
                await asyncio.sleep(cooldown)
            mid = (lo + hi) // 2
            st = await run_level(adapters, steps, recorder, monitor, mid)
            out.append(st)
            if st["ok"]:
                lo = last_ok = mid
            else:
                hi = fail = mid
        reason = next((s["reason"] for s in out if s["level"] == fail and not s["ok"]), None)
        return {"ramp": {"target": target, "unit": unit, "services": services, "steps": out,
                         "last_ok": last_ok, "fail": fail, "fail_reason": reason}}

    result = await _run(ctx, f"ramp_{target}", "ramp", target,
                        {"levels": levels, "step_s": step_s, "max_error_rate": max_error_rate,
                         "bisect_steps": bisect_steps}, body, services)
    r = result["ramp"]
    if r["fail"] is None:
        print(f"[ramp:{target}] no failure: sustained N={r['last_ok']} ({unit}) — add higher levels "
              "to scenarios.ramp.levels to find the limit")
    else:
        print(f"[ramp:{target}] limit: OK up to N={r['last_ok']}, failed at N={r['fail']} ({r['fail_reason']})")
    return result
