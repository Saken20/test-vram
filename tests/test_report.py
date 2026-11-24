"""Report modelling on synthetic results (no services needed) and partial runs."""
import json
import math
import os

import pytest

from vramtest import analysis, config, pdf_report, report

GIB_MIB = 1024


def _cfg(services, **extra):
    return config.from_dict({"services": services, **extra})


def _rec(svc, t0, t1, tokens=0, ok=True):
    return {"service": svc, "t_start": t0, "t_busy": t0, "t_end": t1, "ok": ok, "code": "OK" if ok else "HTTP_500",
            "error": "" if ok else "boom", "ttfb_s": None, "tokens": tokens, "extra": {}, "latency_s": t1 - t0}


def _samples(svc_mib, t0, t1, vllm=None, used_extra=500.0, total=24 * GIB_MIB):
    out = []
    t = t0
    while t <= t1:
        known = [v for v in svc_mib.values() if v is not None]
        out.append({"t": t, "used_mib": (sum(known) + used_extra) if known else None,
                    "total_mib": total if known else None, "svc_mib": dict(svc_mib), "inflight": {},
                    "vllm": vllm or {}})
        t += 0.5
    return out


def _write(run_dir, name, kind, records, samples, t0, t1, sessions=None, info=None, **extra):
    services = sorted({*(r["service"] for r in records), *(k for s in samples for k in s["svc_mib"])})
    res = {"scenario": name, "kind": kind, "target": extra.pop("target", "pipeline"),
           "steps": extra.pop("steps", []), "params": {"sessions": sessions, **extra.pop("params", {})},
           "started_at": "2026-01-02T03:04:05", "t0": t0, "t1": t1,
           "service_info": info or {"gpu_name": "TestGPU", "gpu_total_mib": 24 * GIB_MIB, "pids": {}, "kv": {}},
           "dead": {}, "records": records, "samples": samples,
           "summary": analysis.summarize_window(records, samples, t0, t1, sessions, services), **extra}
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, f"{name}.json"), "w") as f:
        json.dump(res, f)
    return res


def _info(**kv):
    return {"gpu_name": "TestGPU", "gpu_total_mib": 24 * GIB_MIB, "pids": {}, "kv": kv}


def test_solve_sessions():
    assert report.solve_sessions(None, 0.5) is None and report.solve_sessions(10, None) is None
    assert report.solve_sessions(0, 0.5) == 0.0
    n = report.solve_sessions(20, 0.5)
    assert abs(n * 0.5 + 3 * math.sqrt(n * 0.5) - 20) < 1e-9


def test_vllm_base_from_config_pool_and_not_seen_service_excluded(tmp_path):
    run_dir = str(tmp_path)
    cfg = _cfg({"llm": {"adapter": "openai", "url": "http://h:1", "kv_pool_tokens": 32768},
                "other": {"adapter": "openai", "url": "http://h:2", "weights_gib": 5.0},
                "tts": {"adapter": "openai", "url": "http://h:3", "kind": "speech"}})
    info = _info(llm={"bytes_per_token": 128 * 1024, "source": "test"}, other={"bytes_per_token": 1024})
    _write(run_dir, "idle", "idle", [], _samples({"llm": 10240.0, "other": None, "tts": 2048.0}, 0, 5), 0, 5,
           sessions=0, info=info)
    models = {s: report.ServiceModel(s, cfg, report.load_results(run_dir), 0.1, info) for s in cfg["services"]}
    llm = models["llm"]
    assert llm.pool_gib == 4.0 and llm.pool_source == "config kv_pool_tokens"
    assert llm.base_gib == pytest.approx(6.0) and llm.base_source == "idle VRAM - KV pool"
    # weights_gib is set, but the process was never seen while other services were: not on this GPU
    assert not models["other"].counted and models["other"].base_gib is None
    assert models["tts"].counted and models["tts"].base_gib == 2.0
    md = report.build(cfg, run_dir)
    assert "| other | - | - | - | GPU process not seen |" in md
    assert "Total actually needed at idle: **8.00 GiB**" in md


def test_weights_estimate_and_pool_remainder(tmp_path):
    cfg = _cfg({"llm": {"adapter": "openai", "url": "http://h:1", "weights_gib": 4.0, "overhead_gib": 1.0}})
    info = _info(llm={"bytes_per_token": 1024 * 1024, "weights_gib": 4.0, "weights_source": "config.weights_gib"})
    _write(str(tmp_path), "idle", "idle", [], _samples({"llm": 8 * 1024.0}, 0, 2), 0, 2, sessions=0, info=info)
    m = report.ServiceModel("llm", cfg, report.load_results(str(tmp_path)), 0.1, info)
    assert m.base_gib == 5.0 and m.pool_gib == pytest.approx(3.0) and m.pool_tokens == 3072
    assert m.pool_source.startswith("estimate")


def _sessions_run(run_dir, info, k=2, duration=10.0, tokens=1000, svc="llm", mib=None, extra_svcs=()):
    # every session holds one request for half of the time -> d = 0.5
    records = []
    for s in range(k):
        for i in range(2):
            t = 1 + i * 5
            records.append(_rec(svc, t, t + 2.5, tokens))
    samples = _samples(mib or {svc: 10240.0}, 0, duration)
    return _write(run_dir, f"sessions_k{k}", "sessions", records, samples, 0, duration, sessions=k, info=info,
                  params={"profile": f"k{k}"}, turns=[], stats=analysis.turns_stats([]))


def test_capacity_duty_prefix_caching_and_servers(tmp_path):
    run_dir = str(tmp_path)
    prompt = "x" * 400  # 100 tokens at 4 chars per token
    cfg = _cfg({"llm": {"adapter": "openai", "url": "http://h:1", "kv_pool_tokens": 16384}},
               dataset={"system_prompt": prompt},
               capacity={"unit": "user", "safety_margin": 0.0,
                         "targets": [{"label": "one", "sessions": 1}, {"label": "many", "sessions": 10000}]})
    info = _info(llm={"bytes_per_token": 1024 * 1024, "source": "test"})  # 1 MiB per token
    _write(run_dir, "idle", "idle", [], _samples({"llm": 20 * 1024.0}, 0, 2), 0, 2, sessions=0, info=info)
    _sessions_run(run_dir, info)
    ramp_rec = [_rec("llm", 0, 1, 1000)]
    _write(run_dir, "ramp_llm", "ramp", ramp_rec, _samples({"llm": 20 * 1024.0}, 0, 1), 0, 1, info=info,
           target="llm", ramp={"target": "llm", "unit": "in flight", "services": ["llm"], "last_ok": 20, "fail": 30,
                               "fail_reason": "llm: errors", "steps": []})
    results = report.load_results(run_dir)
    m = report.ServiceModel("llm", cfg, results, 0.0, info)
    assert m.duty == pytest.approx(0.5)  # 2 sessions keep 1 request in flight on average
    assert m.tokens_per_request == 1000 and m.shared_tokens == 104 and m.calibration is None
    assert m.base_gib == pytest.approx(4.0)  # 20 GiB idle - 16384 tokens x 1 MiB
    assert m.peak_inflight(10) == 10  # capped at N
    k = m.peak_inflight(40)
    assert k == math.ceil(20 + 3 * math.sqrt(20))
    # KV = k unique parts + one shared system prompt
    assert m.need_gib(40) == pytest.approx(4.0 + (k * (1000 - 104) + 104) / 1024)
    assert m.limit_inflight == 20 and m.sessions_capacity() == pytest.approx(report.solve_sessions(20, m.duty))

    md = report.build(cfg, run_dir)
    assert "## 5. Capacity: users" in md and "shared system prompt" in md
    rep = json.load(open(os.path.join(run_dir, "report.json")))
    one, many = rep["capacity"]["targets"]
    assert one["servers"] == 1 and one["vllm"]["llm"]["fits"]
    assert many["servers"] == max(many["servers_by_memory"], many["servers_by_compute"]) > 100
    assert rep["capacity"]["compute_cap"] == math.floor(m.sessions_capacity())
    pdf = pdf_report.build(cfg, run_dir)
    assert os.path.getsize(pdf) > 2000


def test_generic_service_growth_per_request(tmp_path):
    cfg = _cfg({"tts": {"adapter": "openai", "url": "http://h:1", "kind": "speech"}})
    info = _info()
    _write(str(tmp_path), "idle", "idle", [], _samples({"tts": 1000.0}, 0, 2), 0, 2, sessions=0, info=info)
    recs = [_rec("tts", 0.1, 1.9) for _ in range(4)]
    _write(str(tmp_path), "concurrency_tts_4", "concurrency", recs, _samples({"tts": 1400.0}, 0, 2), 0, 2,
           info=info, target="tts", params={"users": 4})
    m = report.ServiceModel("tts", cfg, report.load_results(str(tmp_path)), 0.0, info)
    assert m.mib_per_request == pytest.approx(100.0) and not m.is_vllm
    md = report.build(cfg, str(tmp_path))
    assert "concurrency_tts_4 | 4 |" in md


def test_partial_runs_without_gpu(tmp_path):
    """Only a burst, NVML unavailable: latency tables still render, memory sections degrade gracefully."""
    cfg = _cfg({"llm": {"adapter": "openai", "url": "http://h:1"}})
    info = {"gpu_name": None, "gpu_total_mib": None, "pids": {}, "kv": {}}
    turns = [{"t": 0, "session": 0, "ok": i != 2, "errors": [] if i != 2 else ["llm:HTTP_500"],
              "steps": [{"service": "llm", "input": "messages", "ok": True, "code": "OK", "latency_s": 0.5 + i / 10,
                         "ttfb_s": 0.1, "tokens": 10}], "wait_first_output_s": 0.1} for i in range(4)]
    recs = [_rec("llm", 0, 0.5 + i / 10, 10, ok=i != 2) for i in range(4)]
    samples = _samples({"llm": None}, 0, 1)
    summ = analysis.summarize_window(recs, samples, 0, 1, None, ["llm"])
    _write(str(tmp_path), "burst_llm", "burst", recs, samples, 0, 1, info=info, target="llm",
           steps=[{"service": "llm", "input": "messages"}],
           levels=[{"level": 4, "t_from": 0, "t_to": 1, "summary": summ, "stats": analysis.turns_stats(turns),
                    "turns": turns}], turns=turns)
    md = report.build(cfg, str(tmp_path))
    assert "GPU: **not available**" in md and "Service: llm" in md and "4 requests at once" in md
    assert "1 of 4" in md and "run `python -m vramtest sessions`" in md
    assert "## 4. Limits" not in md and "## 6." not in md
    rep = json.load(open(tmp_path / "report.json"))
    assert rep["errors"][0]["errors"] == {"llm:HTTP_500": 1}
    pdf = pdf_report.build(cfg, str(tmp_path))
    assert os.path.getsize(pdf) > 2000


def test_empty_run_dir(tmp_path):
    with pytest.raises(SystemExit):
        report.build(_cfg({"llm": {"adapter": "openai", "url": "http://h:1"}}), str(tmp_path))


def check_tables(md):
    """Every markdown table: header, separator and rows have the same number of cells."""
    rows = [line for line in md.splitlines() if line.startswith("|")]
    blocks, cur = [], []
    for line in md.splitlines():
        if line.startswith("|"):
            cur.append(line)
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    assert rows
    for block in blocks:
        counts = {len(line.strip().strip("|").split(" | ")) if not set(line) <= set("|-") else line.count("|") - 1
                  for line in block}
        assert len(counts) == 1, block


def test_tables_are_consistent(tmp_path):
    test_capacity_duty_prefix_caching_and_servers(tmp_path)
    check_tables(open(tmp_path / "report.md").read())


def test_pdf_memory_unit_is_gib():
    # report.json holds GiB (1024^3 bytes); the PDF must not label them as decimal GB
    assert pdf_report._gb(1.5) == "1.5 GiB" and pdf_report._gb(None) == "-"
