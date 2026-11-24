"""Window summaries: calls, latency percentiles, peaks of in-flight calls / KV tokens / VRAM.

Works with any service names. ``records`` are ``CallRecord.to_dict()`` dicts, ``samples`` come from
:class:`vramtest.gpu_monitor.GpuMonitor`, ``turns`` are ``TurnResult.to_dict()`` dicts (vramtest.pipeline).
"""
from __future__ import annotations

import math
from collections import Counter


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def sweep(intervals, t_from, t_to):
    """Peak and time-weighted mean of the sum of weights of overlapping intervals [(start, end, weight)]."""
    events = []
    for start, end, weight in intervals:
        start, end = max(start, t_from), min(end, t_to)
        if end > start:
            events.append((start, weight))
            events.append((end, -weight))
    if not events or t_to <= t_from:
        return 0, 0.0
    events.sort(key=lambda e: (e[0], e[1]))  # closings before openings at the same instant
    level = peak = 0
    area, last_t = 0.0, t_from
    for t, delta in events:
        area += level * (t - last_t)
        last_t = t
        level += delta
        peak = max(peak, level)
    return peak, area / (t_to - t_from)


def _vals(samples, getter):
    out = []
    for s in samples:
        try:
            v = getter(s)
        except (KeyError, TypeError):
            continue
        if v is not None:
            out.append(v)
    return out


def services_in(records, samples=()) -> list[str]:
    """Service names seen in records or monitor samples (sorted, stable)."""
    names = {r["service"] for r in records}
    for s in samples:
        for key in ("svc_mib", "inflight", "vllm"):
            names.update((s.get(key) or {}).keys())
    return sorted(names)


def summarize_window(records, samples, t_from, t_to, sessions=None, services=None):
    """Per-service summary of calls whose processing started in [t_from, t_to] + GPU memory in the window.

    services: names to include (default: every service seen in records/samples).
    """
    recs = [r for r in records if t_from <= r["t_busy"] <= t_to]
    smp = [s for s in samples if t_from <= s.get("t", -1) <= t_to and "error" not in s]
    out = {"duration_s": round(t_to - t_from, 2), "sessions": sessions, "services": {}}
    for svc in (services if services is not None else services_in(records, samples)):
        rs = [r for r in recs if r["service"] == svc]
        ok = [r for r in rs if r["ok"]]
        lat = [r["latency_s"] for r in ok]
        ttfb = [r["ttfb_s"] for r in ok if r.get("ttfb_s") is not None]
        intervals = [(r["t_busy"], r["t_end"], 1) for r in records if r["service"] == svc]
        token_intervals = [(r["t_busy"], r["t_end"], r["tokens"]) for r in records if r["service"] == svc]
        inflight_peak, inflight_mean = sweep(intervals, t_from, t_to)
        kv_peak, kv_mean = sweep(token_intervals, t_from, t_to)
        mib = _vals(smp, lambda s, svc=svc: s["svc_mib"][svc])
        kv_usage = _vals(smp, lambda s, svc=svc: s["vllm"][svc]["kv_usage"])
        running = _vals(smp, lambda s, svc=svc: s["vllm"][svc]["running"])
        waiting = _vals(smp, lambda s, svc=svc: s["vllm"][svc]["waiting"])
        preempt = _vals(smp, lambda s, svc=svc: s["vllm"][svc]["preemptions"])
        pool = _vals(smp, lambda s, svc=svc: s["vllm"][svc]["pool_tokens"])
        extra = [r.get("extra") or {} for r in ok]
        rtf = [e["rtf"] for e in extra if e.get("rtf")]
        out["services"][svc] = {
            "calls": len(rs),
            "ok": len(ok),
            "errors": len(rs) - len(ok),
            "error_rate": round((len(rs) - len(ok)) / len(rs), 4) if rs else 0.0,
            "error_codes": dict(Counter(r["code"] for r in rs if not r["ok"])),
            "error_sample": next((r["error"] for r in rs if not r["ok"] and r["error"]), ""),
            "latency_p50_s": percentile(lat, 0.5),
            "latency_p95_s": percentile(lat, 0.95),
            "latency_max_s": max(lat) if lat else None,
            "ttfb_p50_s": percentile(ttfb, 0.5),
            "ttfb_p95_s": percentile(ttfb, 0.95),
            "rtf_p50": percentile(rtf, 0.5),
            "rtf_p95": percentile(rtf, 0.95),
            "tokens_mean": (sum(r["tokens"] for r in rs) / len(rs)) if rs else None,
            "tokens_max": max((r["tokens"] for r in rs), default=None),
            "inflight_peak": inflight_peak,
            "inflight_mean": round(inflight_mean, 4),
            "kv_tokens_peak": kv_peak,
            "kv_tokens_mean": round(kv_mean, 1),
            "mib_min": min(mib) if mib else None,
            "mib_median": percentile(mib, 0.5),
            "mib_peak": max(mib) if mib else None,
            "vllm_kv_usage_peak": max(kv_usage) if kv_usage else None,
            "vllm_running_peak": max(running) if running else None,
            "vllm_waiting_peak": max(waiting) if waiting else None,
            "vllm_preemptions": (max(preempt) - min(preempt)) if preempt else None,
            "vllm_pool_tokens": max(pool) if pool else None,
        }
    used = _vals(smp, lambda s: s["used_mib"])
    total = _vals(smp, lambda s: s["total_mib"])
    out["gpu"] = {
        "used_mib_min": min(used) if used else None,
        "used_mib_peak": max(used) if used else None,
        "total_mib": total[0] if total else None,
    }
    return out


def verdict(win_summary, services, svc_cfg, max_error_rate, dead):
    """Did a load level pass? -> (ok, reason).

    svc_cfg: {service: service config}; limits are read from ``svc_cfg[svc]["limits"]``.
    """
    for svc in services:
        if svc in dead:
            return False, f"{svc}: its GPU process exited (crashed)"
        s = win_summary["services"].get(svc)
        if not s or s["calls"] == 0:
            return False, f"{svc}: no completed calls"
        if s["error_rate"] > max_error_rate:
            return False, f"{svc}: errors {s['error_rate']:.0%} {s['error_codes']} {s['error_sample'][:120]}"
        limits = (svc_cfg.get(svc) or {}).get("limits") or {}
        p95, limit = s["latency_p95_s"], limits.get("max_p95_latency_s")
        if p95 is not None and limit and p95 > limit:
            return False, f"{svc}: p95 latency {p95:.1f} s > {limit} s"
        ttfb, limit = s["ttfb_p95_s"], limits.get("max_p95_ttfb_s")
        if ttfb is not None and limit and ttfb > limit:
            return False, f"{svc}: p95 time to first output {ttfb:.1f} s > {limit} s"
    return True, "ok"


def _p(values):
    return {"p50": percentile(values, 0.5), "p95": percentile(values, 0.95)}


def turns_stats(turns):
    """Median and p95 per pipeline step and for the wait until first output, + error counts.

    Returns {turns, failed, errors: {"svc:CODE": n}, wait_first_output_s: {p50, p95},
             steps: [{service, latency_s: {p50, p95}, ttfb_s: {p50, p95}}, ...]} (steps in pipeline order).
    """
    out = {"turns": len(turns), "failed": sum(1 for t in turns if not t["ok"])}
    ok_turns = [t for t in turns if t["ok"]]
    out["wait_first_output_s"] = _p([t["wait_first_output_s"] for t in ok_turns
                                     if t.get("wait_first_output_s") is not None])
    n_steps = max((len(t["steps"]) for t in turns), default=0)
    steps = []
    for i in range(n_steps):
        rows = [t["steps"][i] for t in ok_turns if len(t["steps"]) > i]
        name = next((t["steps"][i]["service"] for t in turns if len(t["steps"]) > i), None)
        steps.append({
            "service": name,
            "latency_s": _p([r["latency_s"] for r in rows if r.get("latency_s") is not None]),
            "ttfb_s": _p([r["ttfb_s"] for r in rows if r.get("ttfb_s") is not None]),
        })
    out["steps"] = steps
    errors = {}
    for t in turns:
        for e in t["errors"]:
            errors[e] = errors.get(e, 0) + 1
    out["errors"] = errors
    return out
