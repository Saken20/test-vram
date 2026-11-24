"""Plain-language PDF summary of a run (results/<run>/summary.pdf).

Built from results/<run>/report.json (written by ``report.build``). Command: python -m vramtest --run <name> pdf
"""
from __future__ import annotations

import datetime as dt
import json
import os
from xml.sax.saxutils import escape

from .report import _sec, fmt_seconds, plural, step_headers

INPUT_TEXT = {
    "audio": "an audio clip",
    "text": "a text prompt",
    "messages": "chat messages",
    "prev": "the previous step's output",
}


def _gb(v, digits=1):
    return "-" if v is None else f"{v:.{digits}f} GiB"  # values are GiB (1024³ bytes), as in report.md


def _short(stat):
    return "-" if not stat or stat.get("p50") is None else f"{_sec(stat['p50'])} s"


def _e(text):
    return escape(str(text))


def _headline(group, row):
    """The number a reader cares about for a row: wait until first output (pipeline / single service)."""
    st = row["stats"]
    if group["chain"]:
        return st.get("wait_first_output_s")
    steps = st.get("steps") or []
    return steps[0]["latency_s"] if steps else None


def _title(groups):
    groups = groups[:1]  # the main group: the pipeline if there is one
    bursts = sorted({r["n"] for g in groups for r in g["rows"] if r["kind"] == "burst"})
    if bursts:
        nums = ", ".join(map(str, bursts[:-1])) + (" and " if len(bursts) > 1 else "") + str(bursts[-1])
        return f"GPU load test: {nums} {plural(bursts[-1], 'request')} at once"
    sess = [r for g in groups for r in g["rows"] if r["kind"] == "sessions"]
    if sess:
        return f"GPU load test: {sess[-1]['short']} at once"
    return "GPU inference load test"


def build(cfg, run_dir):
    from .pdf_style import KeepTogether, P, Spacer, build as build_pdf, callout, kpis, table

    path = os.path.join(run_dir, "report.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found — run `python -m vramtest report` first")
    with open(path, encoding="utf-8") as f:
        rep = json.load(f)
    groups = rep.get("summary") or []
    services = rep.get("services") or {}
    unit = rep.get("unit") or "session"
    gpu = rep.get("gpu") or {}
    gpu_name = gpu.get("name") or "GPU memory not measured"
    total = gpu.get("total_gib")
    try:
        date = dt.datetime.fromisoformat(rep.get("started_at") or rep["generated_at"]).strftime("%Y-%m-%d")
    except (KeyError, TypeError, ValueError):
        date = ""
    run = rep.get("run") or os.path.basename(os.path.normpath(run_dir))
    title = _title(groups)
    counted = [s for s, m in services.items() if m.get("counted")]

    story = [
        P(_e(title), "title"),
        Spacer(1, 4),
        P(_e(" · ".join(x for x in (f"{gpu_name}" + (f", {total:.0f} GiB memory" if total else ""), date,
                                    f"run \"{run}\"") if x)), "subtitle"),
        Spacer(1, 12),
    ]

    # ---- KPI tiles
    tiles = []
    main = groups[0] if groups else None
    if main:
        burst = [r for r in main["rows"] if r["kind"] == "burst"]
        sess = [r for r in main["rows"] if r["kind"] == "sessions"]
        picks = burst[:1] + burst[-1:] * (len(burst) > 1) + sess[-1:]
        what = "wait for first output" if main["chain"] else f"{main['target']} response time"
        for r in picks:
            stat = _headline(main, r)
            sub = f"95% within {_sec(stat['p95'])} s" if stat and stat.get("p95") is not None else \
                "no successful requests"
            tiles.append((_short(stat), _e(f"{what}: {r['short']}"), sub))
    peak_need = max((r["need_total_gib"] for g in groups for r in g["rows"] if r.get("need_total_gib") is not None),
                    default=None)
    if peak_need is not None and len(tiles) < 4:
        tiles.append((_gb(peak_need), "GPU memory needed at the highest load", "without vLLM's empty reserve"))
    if tiles:
        story += [kpis(tiles), Spacer(1, 12)]

    # ---- how a request flows
    pipeline = rep.get("pipeline") or []
    story.append(P("How a request flows", "h1"))
    if pipeline:
        rows = [["Step", "What happens"]]
        for i, st in enumerate(pipeline, 1):
            svc = services.get(st["service"]) or {}
            model = f" ({svc['model']})" if svc.get("model") else ""
            rows.append([_e(f"{i}. {st['service']}"),
                         _e(f"Takes {INPUT_TEXT.get(st['input'], st['input'])}: {svc.get('description', '')}{model}")])
        rows.append(["Wait for first output", "From the moment the input is complete until the last step produces "
                     "its first output: the previous steps in full + the first byte/token of the last one"])
        story += [P("Each request passes these steps one after another."), Spacer(1, 6),
                  table(rows, [0.30, 0.70], bold_last=True)]
    else:
        rows = [["Service", "What happens"]]
        for name, svc in services.items():
            model = f" ({svc['model']})" if svc.get("model") else ""
            rows.append([_e(name), _e(f"{svc.get('description', '')}{model}")])
        story += [P("No pipeline is configured: each service was tested on its own."), Spacer(1, 6),
                  table(rows, [0.30, 0.70])]
    story += [Spacer(1, 6),
              P(f"\"N requests at once\" means N requests started at the same instant. \"K {plural(2, unit)}\" means "
                f"K {plural(2, unit)} running at the same time, each sending requests with pauses in between.",
                "small")]

    # ---- seconds per step
    if groups:
        story.append(P("1. How many seconds each step takes", "h1"))
        for g in groups:
            heads = step_headers(g["steps"])
            if len(groups) > 1:
                story.append(P(_e("Pipeline" if g["target"] == "pipeline" else
                                  "All services" if g["target"] == "all" else f"Service {g['target']}"), "body"))
            head = ["Load", *map(_e, heads)] + (["Wait for first output"] if g["chain"] else [])
            body = []
            for r in g["rows"]:
                st = r["stats"]
                cells = [fmt_seconds(x["latency_s"]) for x in st.get("steps", [])]
                cells += ["-"] * (len(heads) - len(cells))
                if g["chain"]:
                    cells.append(f"<b>{fmt_seconds(st.get('wait_first_output_s'))}</b>")
                body.append([_e(r["label"]), *cells])
            n = len(head)
            widths = [0.28] + [0.72 / (n - 1)] * (n - 1) if n > 1 else [1.0]
            story += [table([head, *body], widths), Spacer(1, 5)]
        story.append(P("\"0.8 s (1.4)\" = usually 0.8 seconds; 95% of requests take no longer than 1.4 seconds.",
                       "small"))
    else:
        story.append(P("Run <i>burst</i> or <i>sessions</i> to get per-step timings.", "small"))

    # ---- memory
    if groups and counted:
        rows = [["Load", *map(_e, counted), "Total needed", "GPU used"]]
        for g in groups:
            for r in g["rows"]:
                label = r["label"] if len(groups) == 1 else f"{r['label']} ({g['target']})"
                rows.append([_e(label), *(_gb(r["need_gib"].get(s)) for s in counted),
                             f"<b>{_gb(r['need_total_gib'])}</b>", _gb(r["gpu_used_gib"])])
        n = len(rows[0])
        widths = [0.28] + [0.72 / (n - 1)] * (n - 1)
        story.append(KeepTogether([
            P("2. How much GPU memory is needed", "h1"),
            P("\"Needed\" is the memory the services really use. Some servers (e.g. vLLM) reserve memory in advance, "
              "so \"GPU used\" is larger."),
            Spacer(1, 6),
            table(rows, widths),
            Spacer(1, 5),
            P(f"Total GPU memory: {_gb(total)}." if total else "", "small"),
        ]))
    elif not total:
        story += [Spacer(1, 8), callout("GPU memory was not measured (NVML not available on the test machine).")]

    # ---- errors
    errors = rep.get("errors") or []
    if errors:
        text = "; ".join(f"{e['load']}: {e['failed']} of {e['turns']} failed"
                         + (f" ({', '.join(f'{k} ×{v}' for k, v in e['errors'].items())})" if e.get("errors") else "")
                         for e in errors)
        story += [Spacer(1, 10), callout(f"<b>There were errors.</b> {_e(text)}. Times in the tables only count "
                                         "successful requests.")]

    # ---- capacity
    targets = (rep.get("capacity") or {}).get("targets") or []
    if targets and any(t.get("total_gib") is not None or t.get("servers") or t.get("servers_by_compute")
                       for t in targets):
        rows = [["Target", plural(2, unit).capitalize(), "Memory on one GPU", "Servers needed"]]
        for t in targets:
            rows.append([_e(t["label"]), str(t["sessions"]), _gb(t.get("total_gib")), _e(t.get("text") or "-")])
        cap = rep["capacity"].get("compute_cap")
        story.append(KeepTogether([
            P("3. How many servers are needed", "h1"),
            table(rows, [0.40, 0.15, 0.22, 0.23]),
            Spacer(1, 5),
            P((f"One server handles about {cap} concurrent {plural(cap, unit)} (from the ramp test). " if cap is not None
               else "") + f"Includes a {rep.get('safety_margin', 0.1):.0%} safety margin on the KV cache.", "small"),
        ]))

    # ---- technical details
    tech = [["Load", "Step", "First byte/token", "Full response"]]
    for g in groups:
        heads = step_headers(g["steps"])
        for r in g["rows"]:
            for i, st in enumerate(r["stats"].get("steps", [])):
                label = r["label"] if len(groups) == 1 else f"{r['label']} ({g['target']})"
                tech.append([_e(label) if i == 0 else "", _e(heads[i] if i < len(heads) else st["service"]),
                             fmt_seconds(st.get("ttfb_s")), fmt_seconds(st.get("latency_s"))])
    limits = rep.get("limits") or []
    block = [P("Technical details", "h1")]
    if len(tech) > 1:
        block += [table(tech, [0.34, 0.22, 0.22, 0.22]), Spacer(1, 5),
                  P("If \"first byte/token\" is close to \"full response\", the service does not stream its output, "
                    "or requests wait in a queue. In the test each step starts after the previous one has finished; "
                    "if your system streams between steps, the real wait is shorter.", "small")]
    if limits:
        rows = [["Ramp target", "OK up to", "Failed at", "Reason"]]
        for lim in limits:
            rows.append([_e(lim.get("target")), _e(f"{lim['last_ok']} {lim.get('unit') or ''}".strip())
                         if lim["last_ok"] is not None else "-",
                         _e(lim["fail"]) if lim["fail"] is not None else "not reached", _e(lim.get("reason") or "-")])
        block += [Spacer(1, 8), table(rows, [0.20, 0.18, 0.14, 0.48])]
    rows = [["Service", "Idle (nvidia-smi)", "KV-cache pool", "Needed at idle"]]
    for name, m in services.items():
        rows.append([_e(name), _gb(m.get("idle_nvml_gib"), 2), _gb(m.get("kv_pool_gib"), 2),
                     _gb(m.get("base_gib"), 2) if m.get("counted") else
                     "not on this GPU" if m.get("gpu_data") else "-"])
    block += [Spacer(1, 8), table(rows, [0.28, 0.24, 0.24, 0.24])]
    story.append(KeepTogether(block) if len(tech) < 25 else block[0])
    if len(tech) >= 25:
        story += block[1:]

    out = os.path.join(run_dir, "summary.pdf")
    build_pdf(out, story, title, f"GPU load test · {gpu_name} · {date}")
    return out
