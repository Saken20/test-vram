"""Run report: summary, idle memory, measured scenarios, limits, capacity modelling, vLLM settings.

``build(cfg, run_dir)`` reads every scenario JSON in ``run_dir`` (whatever subset was run), writes
``report.md`` + ``report.json`` (the PDF is built from the latter) and returns the markdown.

Memory model per service (``ServiceModel``):
- vLLM reserves the whole KV pool at start, so its idle VRAM is not what it needs. "Base" memory =
  idle VRAM - KV pool (pool from /metrics, config ``kv_pool_tokens``), else ``weights_gib`` + ``overhead_gib``.
- Need under load = base + KV in use (vLLM /metrics ``kv_cache_usage_perc`` x pool, else analytic: tokens held
  by in-flight requests x KV bytes/token, calibrated by /metrics when available) x (1 + safety margin).
  With prefix caching the shared system prompt is stored once, not per request.
- Other services: need = measured VRAM of their processes (+ growth per in-flight request for extrapolation).
- A service whose GPU process was never seen does not count (it is not on this GPU).
Capacity: a session keeps on average ``d`` requests in flight in each service (measured by ``sessions``);
peak in flight for N sessions = N·d + 3·sqrt(N·d) (Poisson +3σ, at most N). Servers needed = max(by memory,
by throughput from the ramp limits).
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import math
import os
import statistics

from .kv_calc import GIB, TokenCounter

GPU_HEADROOM = 0.95  # never plan to use more than 95% of the GPU memory
KIND_ORDER = ("idle", "burst", "concurrency", "sessions", "ramp")


# ------------------------------------------------------------------ loading / formatting
def load_results(run_dir):
    """{scenario name: result dict} of every scenario JSON in ``run_dir``."""
    results = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        if os.path.basename(path).startswith(("report", "probe")):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as exc:
            print(f"[report] skipping {path}: {exc}")
            continue
        if isinstance(data, dict) and "scenario" in data and "summary" in data:
            results[data["scenario"]] = data
    return results


def _gib(mib):
    return None if mib is None else mib / 1024


def _fmt(v, digits=1):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _sec(v):
    return f"{v:.2f}" if v < 1 else f"{v:.1f}"


def fmt_seconds(stat):
    """"0.8 s (1.4)": median and, in parentheses, the time 95% of requests stay under."""
    if not stat or stat.get("p50") is None:
        return "-"
    return f"{_sec(stat['p50'])} s ({_sec(stat['p95'])})"


def plural(n, word):
    return word if n == 1 else word + "s"


def first(values):
    return next((v for v in values if v is not None), None)


def scenario_order(name, res):
    kind = res.get("kind") or name.split("_")[0]
    idx = KIND_ORDER.index(kind) if kind in KIND_ORDER else len(KIND_ORDER)
    return idx, (res.get("params") or {}).get("sessions") or 0, name


def all_windows(results):
    """Every measured window: whole scenarios, and each level of burst / ramp."""
    for name, res in results.items():
        if res.get("kind") == "burst" and res.get("levels"):
            for st in res["levels"]:
                yield f"{name}@{st['level']}", st["summary"]
        elif res.get("ramp"):
            for st in res["ramp"]["steps"]:
                yield f"{name}@{st['level']}", st["summary"]
        else:
            yield name, res["summary"]


def solve_sessions(limit_inflight, duty):
    """Max N sessions with N·d + 3·sqrt(N·d) <= L (Poisson peak ~ +3σ)."""
    if limit_inflight is None or not duty:
        return None
    if limit_inflight <= 0:
        return 0.0
    x = (-3 + math.sqrt(9 + 4 * limit_inflight)) / 2
    return max(x * x / duty, 0.0)


def describe_service(svc: dict) -> str:
    """Plain-language description of what a service does."""
    if svc.get("adapter") == "openai":
        return {
            "transcription": "speech is transcribed to text",
            "chat": "the language model writes a reply",
            "completions": "the language model completes the text",
            "speech": "text is synthesized to speech",
            "embeddings": "text is turned into an embedding vector",
        }.get(svc.get("kind"), "OpenAI-compatible request")
    if svc.get("adapter") == "grpc":
        return f"gRPC call {svc.get('method') or ''}".strip()
    return f"HTTP {svc.get('method') or 'POST'} request"


# ------------------------------------------------------------------ memory model
class ServiceModel:
    """Memory/throughput model of one service from the measurements."""

    def __init__(self, svc, cfg, results, margin, info):
        self.svc = svc
        self.scfg = cfg["services"][svc]
        self.margin = margin
        self.is_vllm = self.scfg.get("engine") == "vllm"
        self.kv = (info.get("kv") or {}).get(svc) or {}
        self.bpt = self.kv.get("bytes_per_token")
        summaries = [s["services"][svc] for _, s in all_windows(results) if svc in s.get("services", {})]
        self.loaded = any(s["calls"] for s in summaries)
        gpu_data = any(s.get("gpu", {}).get("used_mib_peak") is not None for _, s in all_windows(results))
        self.gpu_data = gpu_data

        idle = results.get("idle")
        idle_s = (idle["summary"]["services"].get(svc) if idle else None) or {}
        self.idle_mib = idle_s.get("mib_median")
        if self.idle_mib is None:
            self.idle_mib = min((s["mib_min"] for s in summaries if s["mib_min"] is not None), default=None)
        self.seen = self.idle_mib is not None or any(s["mib_peak"] is not None for s in summaries)

        # vLLM KV pool: /metrics -> config kv_pool_tokens (vLLM log) -> estimate below
        self.pool_tokens = first(s["vllm_pool_tokens"] for s in summaries)
        self.pool_source = "vLLM /metrics" if self.pool_tokens else None
        if not self.pool_tokens and self.scfg.get("kv_pool_tokens"):
            self.pool_tokens, self.pool_source = int(self.scfg["kv_pool_tokens"]), "config kv_pool_tokens"
        self.pool_gib = self.pool_tokens * self.bpt / GIB if (self.pool_tokens and self.bpt) else None

        # memory without the KV cache (weights + CUDA graphs + activations + context)
        self.base_gib, self.base_source = None, None
        if self.is_vllm:
            weights = self.kv.get("weights_gib") or self.scfg.get("weights_gib")
            wsrc = self.kv.get("weights_source") or "config weights_gib"
            if self.idle_mib is not None and self.pool_gib is not None and _gib(self.idle_mib) > self.pool_gib:
                self.base_gib, self.base_source = _gib(self.idle_mib) - self.pool_gib, "idle VRAM - KV pool"
            elif weights and (self.seen or (not gpu_data and self.scfg.get("weights_gib"))):
                # only when the service's process was seen on the GPU (or there is no GPU data at all and
                # the weights were given explicitly): a service on another GPU/host must not count
                self.base_gib = weights + (self.scfg.get("overhead_gib") or 0.0)
                self.base_source = f"weights ({wsrc}) + overhead_gib"
                if self.idle_mib is not None and self.bpt and not self.pool_gib:
                    self.pool_gib = max(0.0, _gib(self.idle_mib) - self.base_gib)
                    self.pool_tokens = int(self.pool_gib * GIB / self.bpt)
                    self.pool_source = "estimate: idle - weights - overhead"
        elif self.idle_mib is not None:
            self.base_gib, self.base_source = _gib(self.idle_mib), "idle VRAM of the process"
        self.counted = self.base_gib is not None

        # calibration of the analytic KV estimate by /metrics (prefix caching, block granularity, ...)
        ratios = []
        for s in summaries:
            if s["vllm_kv_usage_peak"] is not None and self.pool_tokens and s["kv_tokens_peak"]:
                ratios.append(s["vllm_kv_usage_peak"] * self.pool_tokens / s["kv_tokens_peak"])
        self.calibration = statistics.median(ratios) if ratios else None

        # non-vLLM services: VRAM growth per request in flight (least squares through the origin)
        self.mib_per_request = None
        if not self.is_vllm and self.idle_mib is not None:
            pts = [(s["inflight_peak"], s["mib_peak"] - self.idle_mib) for s in summaries
                   if s["mib_peak"] is not None and s["inflight_peak"]]
            sxx = sum(x * x for x, _ in pts)
            if sxx:
                self.mib_per_request = max(0.0, sum(x * y for x, y in pts) / sxx)

        # load profile of one session: from `sessions` runs and the OK levels of a pipeline ramp
        windows = [(r["summary"]["sessions"], r["summary"]["duration_s"], r["summary"]["services"].get(svc))
                   for r in results.values() if r.get("kind") == "sessions" and r["summary"].get("sessions")]
        for r in results.values():
            if r.get("ramp") and r["ramp"].get("target") == "pipeline":
                windows += [(st["level"], st["summary"]["duration_s"], st["summary"]["services"].get(svc))
                            for st in r["ramp"]["steps"] if st["ok"]]
        windows = [(n, d, s) for n, d, s in windows if s is not None and d]
        weight = sum(n * d for n, d, _ in windows)
        self.duty = sum(s["inflight_mean"] * d for _, d, s in windows) / weight if weight else None
        calls = [(s["calls"], s["tokens_mean"]) for _, _, s in windows if s["tokens_mean"]]
        if not calls:
            calls = [(s["calls"], s["tokens_mean"]) for s in summaries if s["tokens_mean"]]
        total_calls = sum(c for c, _ in calls)
        self.tokens_per_request = sum(c * t for c, t in calls) / total_calls if total_calls else None

        # vLLM prefix caching: the shared system prompt is held in the KV cache once for all requests
        self.shared_tokens = 0
        if self.is_vllm and self.scfg.get("prefix_caching") and self.calibration is None and self._uses_messages(cfg):
            from .dataset import load_system_prompt
            prompt = load_system_prompt((cfg.get("dataset") or {}).get("system_prompt") or "")
            if prompt:
                counter = TokenCounter(self.scfg.get("tokenizer") or "", self.scfg.get("chars_per_token") or 4.0)
                self.shared_tokens = counter.count(prompt) + 4

        # limit from ramp (requests in flight); without a failure it is only a lower bound
        self.ramp = (results.get(f"ramp_{svc}") or {}).get("ramp")
        reached = bool(self.ramp) and self.ramp["fail"] is not None
        self.limit_inflight = (self.ramp["last_ok"] or 0) if reached else None
        self.limit_lower_bound = self.ramp["last_ok"] if (self.ramp and not reached) else None

    def _uses_messages(self, cfg):
        if self.scfg.get("adapter") == "openai" and self.scfg.get("kind") == "chat":
            return True
        return any(st.get("service") == self.svc and st.get("input") == "messages" for st in cfg.get("pipeline") or [])

    # ---------------------------------------------------------- measured windows
    def kv_measured_gib(self, s):
        """(from /metrics, analytic) KV GiB at the window's peak."""
        if not self.bpt or not s:
            return None, None
        metric = (s["vllm_kv_usage_peak"] * self.pool_gib
                  if (s["vllm_kv_usage_peak"] is not None and self.pool_gib) else None)
        tokens = s["kv_tokens_peak"] or 0
        if self.shared_tokens and s["inflight_peak"]:
            # each of the in-flight requests counted the shared prompt; the cache holds one copy
            tokens = max(self.shared_tokens, tokens - (s["inflight_peak"] - 1) * self.shared_tokens)
        return metric, tokens * self.bpt / GIB

    def need_measured_gib(self, s):
        if not self.counted or not s:
            return None
        if not self.is_vllm:
            return _gib(s["mib_peak"]) if s["mib_peak"] is not None else self.base_gib
        metric, analytic = self.kv_measured_gib(s)
        if metric is None and analytic is None:
            return self.base_gib
        kv = metric if metric is not None else analytic * (self.calibration or 1.0)
        return self.base_gib + self._kv_floor(kv) * (1 + self.margin)

    def _kv_floor(self, kv_gib):
        mml = self.scfg.get("max_model_len")
        return max(kv_gib, mml * self.bpt / GIB) if (mml and self.bpt) else kv_gib

    # ---------------------------------------------------------- extrapolation
    def peak_inflight(self, n):
        if n is None or self.duty is None:
            return None
        mean = n * self.duty
        return min(math.ceil(n), math.ceil(mean + 3 * math.sqrt(mean)))

    def need_gib(self, n):
        k = self.peak_inflight(n)
        if k is None or self.base_gib is None:
            return None
        if self.is_vllm:
            if not self.bpt:
                return self.base_gib
            if not self.tokens_per_request:
                return None
            unique = max(0.0, self.tokens_per_request - self.shared_tokens)
            kv_tokens = (k * unique + self.shared_tokens) if k else 0
            kv = kv_tokens * (self.calibration or 1.0) * self.bpt / GIB
            return self.base_gib + self._kv_floor(kv) * (1 + self.margin)
        return self.base_gib + (self.mib_per_request or 0.0) * k / 1024 * (1 + self.margin)

    def sessions_capacity(self):
        return solve_sessions(self.limit_inflight, self.duty)

    def to_dict(self):
        return {
            "adapter": self.scfg.get("adapter"), "kind": self.scfg.get("kind") if self.scfg.get("adapter") == "openai"
            else None, "model": self.scfg.get("model") or None, "engine": self.scfg.get("engine"),
            "description": describe_service(self.scfg),
            "seen_on_gpu": self.seen, "counted": self.counted, "loaded": self.loaded, "gpu_data": self.gpu_data,
            "idle_nvml_gib": _gib(self.idle_mib), "kv_pool_tokens": self.pool_tokens, "kv_pool_gib": self.pool_gib,
            "kv_pool_source": self.pool_source, "base_gib": self.base_gib, "base_source": self.base_source,
            "bytes_per_token": self.bpt, "kv_source": self.kv.get("source"), "calibration": self.calibration,
            "duty": self.duty, "tokens_per_request": self.tokens_per_request, "shared_tokens": self.shared_tokens,
            "mib_per_request": self.mib_per_request, "limit_inflight": self.limit_inflight,
            "limit_lower_bound": self.limit_lower_bound, "sessions_capacity": self.sessions_capacity(),
        }


def servers_needed(models, n, gpu_total_gib, compute_cap):
    """Minimum identical servers so that load ``n``, split evenly, fits by memory and by throughput."""
    counted = [m for m in models.values() if m.counted]
    if not gpu_total_gib or not counted:
        if compute_cap and n > 0:  # throughput is known even without memory data: a lower bound
            k = math.ceil(math.ceil(n) / compute_cap)
            return {"servers": None, "servers_by_memory": None, "servers_by_compute": k,
                    "text": f"≥ {k} (throughput; no VRAM data)"}
        return {"servers": None, "text": "no VRAM data"}
    budget = gpu_total_gib * GPU_HEADROOM

    def total(sessions):
        needs = [m.need_gib(sessions) for m in counted]
        return None if any(x is None for x in needs) else sum(needs)

    if total(1) is None:
        return {"servers": None, "text": "not enough data"}
    if total(1) > budget:
        return {"servers": None, "text": "even 1 does not fit"}
    n = math.ceil(n)
    if n <= 0:
        return {"servers": 0, "text": "0"}
    if compute_cap == 0:
        return {"servers": None, "text": "cannot sustain even 1"}
    k_mem = next(k for k in range(1, n + 1) if total(math.ceil(n / k)) <= budget)
    k_cpu = math.ceil(n / compute_cap) if compute_cap else 1
    k = max(k_mem, k_cpu)
    why = "" if k == 1 else (" (memory)" if k_mem >= k_cpu else " (throughput)")
    return {"servers": k, "servers_by_memory": k_mem, "servers_by_compute": k_cpu if compute_cap else None,
            "text": f"{k}{why}"}


# ------------------------------------------------------------------ summary rows
def _group_key(res):
    return res.get("target") or "pipeline"


def summary_groups(cfg, results, models, unit):
    """Rows for the plain summary, grouped by what was loaded (the pipeline, or a single service):
    burst levels ("N requests at once") and session profiles."""
    groups = {}

    def group(res):
        key = _group_key(res)
        if key not in groups:
            steps = [s["service"] for s in res.get("steps") or []]
            if not steps:
                stats = res.get("stats") or next((lv.get("stats") for lv in res.get("levels") or []), None) or {}
                steps = [s["service"] for s in stats.get("steps") or []]
            groups[key] = {"target": key, "steps": steps,
                           "chain": key != "all", "rows": []}
        return groups[key]

    def row(label, short, kind, n, stats, summary):
        need = {svc: m.need_measured_gib(summary["services"].get(svc)) for svc, m in models.items() if m.counted}
        present = [x for x in need.values() if x is not None]
        return {"label": label, "short": short, "kind": kind, "n": n, "stats": stats, "need_gib": need,
                "need_total_gib": sum(present) if present else None,
                "nvml_gib": {svc: _gib((summary["services"].get(svc) or {}).get("mib_peak")) for svc in models},
                "gpu_used_gib": _gib(summary["gpu"]["used_mib_peak"])}

    bursts = sorted((r for r in results.values() if r.get("kind") == "burst"), key=lambda r: r["scenario"])
    for res in bursts:
        g = group(res)
        for st in res.get("levels") or []:
            n = st["level"]
            short = f"{n} {plural(n, 'request')}"
            label = short if n == 1 else f"{short} at once"
            g["rows"].append(row(label, short, "burst", n, st["stats"], st["summary"]))
    sess = sorted((r for r in results.values() if r.get("kind") == "sessions" and r.get("turns") is not None),
                  key=lambda r: (r["params"]["sessions"], r["scenario"]))
    for res in sess:
        g = group(res)
        k, profile = res["params"]["sessions"], res["params"].get("profile", "")
        short = f"{k} {plural(k, unit)}"
        label = f"{short} ({profile})" if profile else short
        g["rows"].append(row(label, short, "sessions", k, res.get("stats") or {}, res["summary"]))
    order = ["pipeline", *cfg["services"], "all"]
    return [groups[k] for k in sorted(groups, key=lambda k: order.index(k) if k in order else len(order))]


def step_headers(steps):
    out = []
    for i, svc in enumerate(steps):
        out.append(f"{svc} #{i + 1}" if steps.count(svc) > 1 else svc)
    return out


# ------------------------------------------------------------------ build
def build(cfg, run_dir):
    results = load_results(run_dir)
    if not results:
        raise SystemExit(f"no results in {run_dir} — run some scenarios first (e.g. `python -m vramtest burst`)")
    cap = cfg.get("capacity") or {}
    margin = cap.get("safety_margin", 0.10)
    unit = cap.get("unit") or "session"
    info = first(r.get("service_info") for r in sorted(results.values(), key=lambda r: r.get("started_at", ""),
                                                        reverse=True)) or {}
    models = {svc: ServiceModel(svc, cfg, results, margin, info) for svc in cfg["services"]}
    gpu_total_gib = _gib(info.get("gpu_total_mib"))
    run = os.path.basename(os.path.normpath(run_dir))
    started = min((r.get("started_at") for r in results.values() if r.get("started_at")), default=None)
    lines = []
    w = lines.append
    out = {
        "run": run, "generated_at": dt.datetime.now().isoformat(timespec="seconds"), "started_at": started,
        "gpu": {"name": info.get("gpu_name"), "total_gib": gpu_total_gib},
        "gpu_total_gib": gpu_total_gib, "unit": unit, "safety_margin": margin,
        "pipeline": cfg.get("pipeline") or [], "scenarios_run": sorted(results),
        "services": {}, "summary": [], "idle": {}, "scenarios": {}, "limits": [], "capacity": {}, "errors": [],
    }

    # ---- header
    w(f"# GPU inference load report — run `{run}`\n")
    if gpu_total_gib:
        w(f"GPU: **{info.get('gpu_name')}**, {gpu_total_gib:.1f} GiB total. Safety margin in the estimates: "
          f"{margin:.0%}. Memory in GiB (1024³ bytes; nvidia-smi shows MiB).\n")
    else:
        w("GPU: **not available** (NVML) — VRAM was not measured; only load and latency are reported.\n")
    if cfg.get("pipeline"):
        w("Pipeline (one turn): " + " → ".join(f"{s['service']} ({s['input']})" for s in cfg["pipeline"]) + "\n")
    w("| service | adapter | model | GPU processes | vLLM /metrics | KV per token | KV pool |")
    w("|---|---|---|---|---|---|---|")
    for svc, m in models.items():
        s = m.scfg
        adapter = s.get("adapter") + (f" ({s.get('kind')})" if s.get("adapter") == "openai" else "")
        kv = f"{m.bpt / 1024:.0f} KiB ({m.kv.get('source')})" if m.bpt else "-"
        if not m.is_vllm:
            pool = "- (not vLLM)"
        elif m.pool_tokens and m.pool_gib is not None:
            pool = f"{m.pool_tokens:,} tokens = {m.pool_gib:.2f} GiB ({m.pool_source})"
        elif m.pool_tokens:
            pool = f"{m.pool_tokens:,} tokens ({m.pool_source}; GiB unknown without KV bytes/token)"
        else:
            pool = "unknown"
        pids = (info.get("pids") or {}).get(svc)
        w(f"| {svc} | {adapter} | {s.get('model') or '-'} | {pids or 'not found'} | "
          f"{(info.get('metrics_urls') or {}).get(svc) or '-'} | {kv} | {pool} |")
        out["services"][svc] = m.to_dict()

    # ---- 1. summary
    groups = summary_groups(cfg, results, models, unit)
    out["summary"] = groups
    counted = [svc for svc, m in models.items() if m.counted]
    if groups:
        w("\n## 1. Summary\n")
        w("\"N requests at once\" = N turns started at the same instant (burst). "
          f"\"K {plural(2, unit)}\" = K concurrent {plural(2, unit)} with pauses between turns. "
          "Times: \"0.8 s (1.4)\" = median 0.8 s, 95% of requests within 1.4 s. "
          "\"First output\" = from the moment the input is complete until the first output of the last step "
          "(sum of the previous steps + time to first byte/token of the last one).\n")
        for g in groups:
            title = ("Pipeline: " + " → ".join(g["steps"])) if g["target"] == "pipeline" else \
                ("Every service once per turn: " + ", ".join(g["steps"])) if g["target"] == "all" else \
                f"Service: {g['target']}"
            w(f"**{title}** — seconds per step\n")
            heads = step_headers(g["steps"])
            head = ["load", *heads] + (["**first output**"] if g["chain"] else []) + ["errors"]
            w("| " + " | ".join(head) + " |")
            w("|" + "---|" * len(head))
            for r in g["rows"]:
                st = r["stats"]
                cells = [fmt_seconds(x["latency_s"]) for x in st.get("steps", [])][:len(heads)]
                cells += ["-"] * (len(heads) - len(cells))
                if g["chain"]:
                    cells.append(f"**{fmt_seconds(st.get('wait_first_output_s'))}**")
                cells.append(f"{st.get('failed', 0)} of {st.get('turns', 0)}")
                w(f"| {r['label']} | " + " | ".join(cells) + " |")
                if st.get("failed"):
                    out["errors"].append({"load": r["label"], "target": g["target"], "failed": st["failed"],
                                          "turns": st["turns"], "errors": st.get("errors") or {}})
            w("")
        if counted:
            w("GPU memory, GiB: \"needed\" = what the services really use (without vLLM's empty KV reserve, "
              f"+{margin:.0%} margin on the KV cache); \"GPU used\" = what nvidia-smi shows.\n")
            w("| load | " + " | ".join(counted) + " | **total needed** | GPU used |")
            w("|---|" + "---|" * (len(counted) + 2))
            for g in groups:
                for r in g["rows"]:
                    label = r["label"] if len(groups) == 1 else f"{r['label']} — {g['target']}"
                    w(f"| {label} | " + " | ".join(_fmt(r["need_gib"].get(s), 2) for s in counted)
                      + f" | **{_fmt(r['need_total_gib'], 2)}** | {_fmt(r['gpu_used_gib'], 2)} |")
            w("")
    not_seen = [svc for svc, m in models.items() if not m.counted]
    if not_seen and gpu_total_gib:
        w(f"Not counted in memory totals (GPU process not found or not enough data): {', '.join(not_seen)}. "
          "Set `pids` / `cmdline_match` if they do run on this GPU.\n")

    # ---- 2. idle memory
    w("## 2. Idle memory: reserved vs actually needed\n")
    w("vLLM reserves its whole KV-cache pool at start, so nvidia-smi shows almost the same at idle as under load. "
      "\"Actually needed\" = weights + CUDA graphs + activations, without the empty KV cache.\n")
    w("| service | used now (NVML) | of which KV pool | actually needed at idle | how |")
    w("|---|---|---|---|---|")
    for svc, m in models.items():
        how = m.base_source or ("VRAM not measured (no GPU data)" if not m.gpu_data else
                                "GPU process not seen" if not m.seen else
                                "needs /metrics, kv_pool_tokens or weights_gib")
        w(f"| {svc} | {_fmt(_gib(m.idle_mib), 2)} | {_fmt(m.pool_gib, 2) if m.is_vllm else '-'} | "
          f"{_fmt(m.base_gib, 2)} | {how} |")
    total_base = sum(m.base_gib for m in models.values() if m.base_gib is not None)
    idle_used = _gib(results["idle"]["summary"]["gpu"]["used_mib_min"]) if "idle" in results else None
    out["idle"] = {"needed_gib": total_base if counted else None, "gpu_used_gib": idle_used,
                   "services": {svc: {"nvml_gib": _gib(m.idle_mib), "kv_pool_gib": m.pool_gib, "base_gib": m.base_gib,
                                      "source": m.base_source} for svc, m in models.items()}}
    if counted:
        w(f"\nTotal actually needed at idle: **{total_base:.2f} GiB** (GPU used at idle: {_fmt(idle_used, 2)} GiB).\n")
    else:
        w("\nNo service memory measured.\n")

    # ---- 3. measured scenarios
    plain = sorted(((n, r) for n, r in results.items() if r.get("kind") in ("idle", "concurrency", "sessions")),
                   key=lambda x: scenario_order(*x))
    if plain:
        w("## 3. Measured scenarios\n")
        w("Per service: peak requests in flight / KV cache at peak (vLLM) / memory needed. Needed = base + KV at "
          "peak (+margin). KV comes from vLLM /metrics, else analytic: tokens of in-flight requests × bytes/token.\n")
        cols = [svc for svc, m in models.items() if m.counted or m.loaded]
        w("| scenario | sessions / users | " + " | ".join(cols) + " | total needed | GPU used | errors | p95 latency, s |")
        w("|---|---|" + "---|" * (len(cols) + 4))
        for name, res in plain:
            s = res["summary"]
            cells, total, errs, p95 = [], 0.0, [], []
            for svc in cols:
                m, ss = models[svc], s["services"].get(svc)
                if not ss:
                    cells.append("-")
                    continue
                need = m.need_measured_gib(ss)
                if need is not None:
                    total += need
                if ss["errors"]:
                    errs.append(f"{svc}: {ss['errors']}/{ss['calls']}")
                if ss["latency_p95_s"] is not None:
                    p95.append(f"{svc} {ss['latency_p95_s']:.2f}")
                if not m.counted:
                    cells.append(f"{ss['inflight_peak']} / -")
                elif m.is_vllm and m.bpt:
                    metric, analytic = m.kv_measured_gib(ss)
                    kv = metric if metric is not None else analytic
                    cells.append(f"{ss['inflight_peak']} / {_fmt(kv, 2)} / **{_fmt(need, 2)}**")
                else:
                    cells.append(f"{ss['inflight_peak']} / **{_fmt(need, 2)}**")
            params = res.get("params") or {}
            who = params.get("sessions") if res.get("kind") != "concurrency" else params.get("users")
            total = total if counted else None
            w(f"| {name} | {_fmt(who)} | " + " | ".join(cells) + f" | **{_fmt(total, 2)}** | "
              f"{_fmt(_gib(s['gpu']['used_mib_peak']), 2)} | {', '.join(errs) or '0'} | {', '.join(p95) or '-'} |")
            out["scenarios"][name] = {"kind": res.get("kind"), "target": res.get("target"), "need_total_gib": total,
                                      "gpu_used_gib": _gib(s["gpu"]["used_mib_peak"]),
                                      "dead": res.get("dead") or {}}
        w("")

    # ---- 4. limits
    ramps = sorted(((n, r) for n, r in results.items() if r.get("ramp")), key=lambda x: x[0])
    if ramps:
        w("## 4. Limits (ramp)\n")
        w("Concurrency is raised until errors exceed `max_error_rate`, a GPU process dies, or p95 latency / time "
          "to first output exceed the service `limits`. Service targets: N = requests in flight; "
          "pipeline: N = concurrent sessions.\n")
        w("| target | OK up to N | failed at N | reason | VRAM at last OK, GiB | vLLM KV usage | p95 latency, s | "
          "p95 first output, s |")
        w("|---|---|---|---|---|---|---|---|")
        for name, res in ramps:
            r = res["ramp"]
            svcs = r.get("services") or [s["service"] for s in res.get("steps") or []]
            ok_step = next((st for st in reversed(r["steps"]) if st["level"] == r["last_ok"] and st["ok"]), None)
            if ok_step:
                ss = ok_step["summary"]["services"]

                def col(key, conv=lambda v: v, digits=2):
                    return ", ".join(f"{x} {_fmt(conv((ss.get(x) or {}).get(key)), digits)}" if len(svcs) > 1
                                     else _fmt(conv((ss.get(x) or {}).get(key)), digits) for x in svcs)
                mib, kvu, p95, ttfb = col("mib_peak", _gib), col("vllm_kv_usage_peak"), col("latency_p95_s"), \
                    col("ttfb_p95_s")
            else:
                mib = kvu = p95 = ttfb = "-"
            fail = r["fail"] if r["fail"] is not None else "not reached"
            w(f"| {r.get('target', name)} ({r.get('unit') or 'in flight'}) | {_fmt(r['last_ok'])} | {fail} | {r.get('fail_reason') or '-'} | {mib} | "
              f"{kvu} | {p95} | {ttfb} |")
            out["limits"].append({"scenario": name, "target": r.get("target"), "unit": r.get("unit"),
                                  "last_ok": r["last_ok"], "fail": r["fail"], "reason": r.get("fail_reason")})
        w("")

    # ---- 5. capacity
    targets = [t for t in cap.get("targets") or [] if isinstance(t, dict) and t.get("sessions") is not None]
    caps = [m.sessions_capacity() for m in models.values() if m.sessions_capacity() is not None]
    pipe_ramp = (results.get("ramp_pipeline") or {}).get("ramp")
    if pipe_ramp and pipe_ramp["fail"] is not None:
        caps.append(pipe_ramp["last_ok"] or 0)
    compute_cap = math.floor(min(caps)) if caps else None
    have_duty = any(m.duty is not None for m in models.values())
    w(f"## 5. Capacity: {plural(2, unit)}\n")
    out["capacity"] = {"unit": unit, "safety_margin": margin, "compute_cap": compute_cap, "targets": []}
    if not have_duty:
        w(f"Not enough data: run `python -m vramtest sessions` to measure how much load one {unit} creates.\n")
    else:
        w(f"One {unit} = the load measured by the `sessions` scenario "
          f"({cfg['scenarios']['sessions']['turns_per_session']} turns per "
          f"{cfg['scenarios']['sessions']['session_duration_s']} s). From the measurements: the average number of "
          f"requests in flight per {unit} in each service (d) and the average KV tokens per request; "
          f"peak in flight for N {plural(2, unit)} = N·d + 3·√(N·d) (at most N).\n")
        for svc, m in models.items():
            if not m.duty:  # not loaded by the sessions
                continue
            extra = ""
            if m.is_vllm and m.tokens_per_request:
                extra = f", ≈{m.tokens_per_request:.0f} KV tokens per request"
                if m.shared_tokens:
                    extra += f" (of which {m.shared_tokens} = shared system prompt, cached once)"
                if m.calibration:
                    extra += f" × calibration {m.calibration:.2f} from /metrics"
            elif m.mib_per_request is not None:
                extra = f", +{m.mib_per_request:.0f} MiB per request in flight"
            if m.sessions_capacity() is not None:
                extra += (f"; limit ≈ {math.floor(m.sessions_capacity())} {plural(2, unit)} per GPU "
                          f"(ramp: {m.limit_inflight} in flight)")
            elif m.limit_lower_bound:
                extra += f"; ramp did not fail (sustained {m.limit_lower_bound} in flight) — limit not counted"
            if not m.counted and m.gpu_data:
                extra += "; not seen on this GPU — memory not counted"
            w(f"- {svc}: d = {m.duty:.3f} requests in flight per {unit}{extra}")
        if pipe_ramp:
            w(f"- pipeline ramp: OK up to {pipe_ramp['last_ok']} concurrent {plural(2, unit)}"
              + ("" if pipe_ramp["fail"] is not None else " (no failure reached)"))
        if compute_cap is not None:
            w(f"- **Throughput of one server: ≈ {compute_cap} concurrent {plural(2, unit)}** "
              "(minimum over the ramp limits).")
        w("")
        busy = [svc for svc, m in models.items() if m.duty]
        if targets:
            w(f"| target | {plural(2, unit)} | peak in flight (" + " / ".join(busy) + ") | "
              + "".join(f"{s} | " for s in counted) + "total on 1 GPU | servers (same GPU) |")
            w("|---|---|---|" + "---|" * (len(counted) + 2))
        for t in targets:
            n = t["sessions"]
            label = t.get("label") or f"{n} {plural(n, unit)}"
            needs = {svc: models[svc].need_gib(n) for svc in counted}
            total = sum(needs.values()) if counted and all(x is not None for x in needs.values()) else None
            peaks = {svc: m.peak_inflight(n) for svc, m in models.items()}
            servers = servers_needed(models, n, gpu_total_gib, compute_cap)
            w(f"| {label} | {n} | " + " / ".join(_fmt(peaks[s]) for s in busy) + " | "
              + "".join(f"{_fmt(needs[s], 2)} | " for s in counted) + f"**{_fmt(total, 2)}** | {servers['text']} |")
            out["capacity"]["targets"].append({"label": label, "sessions": n, "peak_inflight": peaks,
                                               "need_gib": needs, "total_gib": total, **servers})
        w("")

    # ---- 6. vLLM settings
    vllm = [svc for svc in counted if models[svc].is_vllm]
    if vllm and out["capacity"]["targets"] and gpu_total_gib:
        w("## 6. Recommended vLLM settings (per server)\n")
        w("Load split evenly over the servers from section 5. `--gpu-memory-utilization` = memory the service "
          "needs / total GPU memory (weights + graphs + KV for the peak); `--max-num-seqs` = peak requests in "
          "flight.\n")
        w(f"| target | {plural(2, unit)} per server | " + " | ".join(
            f"{s}: gpu_memory_utilization / max_num_seqs" for s in vllm) + " |")
        w("|---|---|" + "---|" * len(vllm))
        for t in out["capacity"]["targets"]:
            per = math.ceil(t["sessions"] / t["servers"]) if t.get("servers") else t["sessions"]
            cells, rec = [], {}
            for svc in vllm:
                need, k = models[svc].need_gib(per), models[svc].peak_inflight(per)
                if need is None:
                    cells.append("-")
                    rec[svc] = None
                elif need > gpu_total_gib * GPU_HEADROOM:
                    cells.append("does not fit on 1 GPU")
                    rec[svc] = {"fits": False}
                else:
                    util = math.ceil(need / gpu_total_gib * 100) / 100
                    cells.append(f"{util:.2f} / {k}")
                    rec[svc] = {"fits": True, "gpu_memory_utilization": util, "max_num_seqs": k}
            t["per_server"] = per
            t["vllm"] = rec
            w(f"| {t['label']} | {per} | " + " | ".join(cells) + " |")
        w("")

    # ---- notes
    w("## Notes\n")
    w("- Analytic KV: 2 × layers × KV heads × head_dim × bytes per element, per token (from the model's "
      "config.json, or `kv_bytes_per_token`). Tokens = prompt + output (from the response `usage` when present). "
      "With vLLM /metrics the estimate is calibrated by the real `kv_cache_usage_perc`.")
    w("- Without /metrics the KV pool is estimated as idle VRAM − weights − overhead. More precise: copy from the "
      "vLLM start-up log `GPU KV cache size: N tokens` → `kv_pool_tokens`, `Model loading took X GiB` → "
      "`weights_gib`.")
    w("- Prefix caching: a shared system prompt is held in the KV cache once (`prefix_caching: false` if disabled).")
    w("- Servers needed accounts for both memory (each service's base memory repeats on every GPU) and throughput "
      "(ramp limits).")
    w("- vLLM refuses to start when the KV pool is smaller than `--max-model-len`. When lowering "
      "`--gpu-memory-utilization`, set `--max-model-len` explicitly and put it into `max_model_len` so the "
      "recommendations respect that minimum.")
    w("- Several vLLM instances on one GPU: the sum of their `--gpu-memory-utilization` plus the memory of the other "
      "services should stay ≤ ~0.95; start them one by one (vLLM checks free memory at start).")
    report = "\n".join(lines) + "\n"
    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(run_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    return report
