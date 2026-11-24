from vramtest import analysis, vllm_metrics


def test_vllm_metrics_summary():
    text = (
        "# HELP x\n"
        'vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.25\n'
        'vllm:num_requests_running{engine="0"} 3.0\n'
        'vllm:num_requests_waiting{engine="0"} 1.0\n'
        'vllm:num_preemptions_total{engine="0"} 2.0\n'
        'vllm:cache_config_info{block_size="16",num_gpu_blocks="100",cache_dtype="auto"} 1.0\n'
    )
    s = vllm_metrics.summarize(text)
    assert s == {"kv_usage": 0.25, "running": 3, "waiting": 1, "preemptions": 2, "pool_tokens": 1600}
    # older metric name (vLLM V0)
    assert vllm_metrics.summarize("vllm:gpu_cache_usage_perc 0.5\n")["kv_usage"] == 0.5
    assert vllm_metrics.looks_like_vllm(text) and not vllm_metrics.looks_like_vllm("process_cpu_seconds 1\n")


def test_metrics_from_mock(openai_mock):
    s = vllm_metrics.summarize(vllm_metrics.fetch(openai_mock.url + "/metrics"))
    assert s["kv_usage"] == 0.0 and s["pool_tokens"] == 16000


def test_sweep_peak_and_mean():
    peak, mean = analysis.sweep([(0, 2, 1), (1, 3, 1), (2, 4, 1)], 0, 4)
    assert peak == 2 and abs(mean - 1.5) < 1e-9
    # touching intervals do not overlap
    assert analysis.sweep([(0, 1, 1), (1, 2, 1)], 0, 2)[0] == 1
    # weights and clipping to the window
    assert analysis.sweep([(0, 10, 100), (5, 6, 50)], 4, 8) == (150, (4 * 100 + 50) / 4)
    assert analysis.sweep([], 0, 1) == (0, 0.0)


def test_percentile():
    assert analysis.percentile([], 0.5) is None
    assert analysis.percentile([1, 2, 3, 4], 0.5) == 2.5
    assert analysis.percentile([5], 0.95) == 5


def _record(svc, t, lat, ok=True, tokens=10, ttfb=None, code="OK", rtf=None):
    return {"service": svc, "t_start": t, "t_busy": t, "t_end": t + lat, "latency_s": lat, "ok": ok, "code": code,
            "error": "" if ok else "boom", "ttfb_s": ttfb, "tokens": tokens, "extra": {"rtf": rtf} if rtf else {}}


def test_summarize_window_dynamic_services_and_verdict():
    records = [_record("my-llm", 0, 1.0, ttfb=0.2), _record("my-llm", 0.5, 2.0, ttfb=0.3),
               _record("voice", 1, 0.5, rtf=0.1), _record("voice", 2, 0.5, ok=False, code="HTTP_503")]
    samples = [{"t": 0.5, "used_mib": 1000, "total_mib": 8000, "svc_mib": {"my-llm": 600, "voice": None},
                "inflight": {"my-llm": 2}, "vllm": {"my-llm": {"kv_usage": 0.3, "running": 2, "waiting": 0,
                                                               "preemptions": 0, "pool_tokens": 1600}}}]
    s = analysis.summarize_window(records, samples, 0, 4)
    assert set(s["services"]) == {"my-llm", "voice"}
    llm, voice = s["services"]["my-llm"], s["services"]["voice"]
    assert llm["calls"] == 2 and llm["inflight_peak"] == 2 and llm["kv_tokens_peak"] == 20
    assert llm["mib_peak"] == 600 and llm["vllm_kv_usage_peak"] == 0.3 and llm["ttfb_p50_s"] == 0.25
    assert voice["errors"] == 1 and voice["error_codes"] == {"HTTP_503": 1} and voice["rtf_p50"] == 0.1
    assert s["gpu"]["used_mib_peak"] == 1000
    only = analysis.summarize_window(records, samples, 0, 4, services=["voice"])
    assert list(only["services"]) == ["voice"]
    cfgs = {"my-llm": {"limits": {"max_p95_latency_s": 1.5, "max_p95_ttfb_s": 3}}, "voice": {"limits": {}}}
    ok, why = analysis.verdict(s, ["my-llm"], cfgs, 0.05, {})
    assert not ok and "p95 latency" in why
    assert not analysis.verdict(s, ["voice"], cfgs, 0.05, {})[0]
    assert analysis.verdict(s, ["voice"], cfgs, 0.6, {})[0]
    assert "crashed" in analysis.verdict(s, ["voice"], cfgs, 0.6, {"voice": 1.0})[1]


def test_turns_stats():
    def turn(ok, lat, wait):
        return {"ok": ok, "errors": [] if ok else ["b:HTTP_500"], "wait_first_output_s": wait,
                "steps": [{"service": "a", "latency_s": lat, "ttfb_s": None},
                          {"service": "b", "latency_s": 1.0, "ttfb_s": 0.1}]}
    st = analysis.turns_stats([turn(True, 1.0, 1.1), turn(True, 3.0, 3.1), turn(False, 9.0, 9.0)])
    assert st["turns"] == 3 and st["failed"] == 1 and st["errors"] == {"b:HTTP_500": 1}
    assert st["wait_first_output_s"]["p50"] == 2.1
    assert st["steps"][0] == {"service": "a", "latency_s": {"p50": 2.0, "p95": 2.9}, "ttfb_s": {"p50": None, "p95": None}}
