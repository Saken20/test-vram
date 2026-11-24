"""Parse vLLM Prometheus metrics (/metrics)."""
import re
import urllib.request

_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(.*)\})?\s+([-+0-9.eEnaNinf]+)')
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')

KV_USAGE_NAMES = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")


def parse(text):
    """List of (name, labels, value)."""
    out = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        labels = dict(_LABEL.findall(m.group(3) or ""))
        try:
            value = float(m.group(4))
        except ValueError:
            continue
        out.append((m.group(1), labels, value))
    return out


def _sum(metrics, name):
    values = [v for n, _, v in metrics if n == name]
    return sum(values) if values else None


def summarize(text):
    """What the KV-cache model needs: usage fraction, running/waiting requests, preemptions, pool size."""
    metrics = parse(text)
    kv_usage = None
    for name in KV_USAGE_NAMES:
        values = [v for n, _, v in metrics if n == name]
        if values:
            kv_usage = max(values)
            break
    pool_tokens = None
    for name, labels, _ in metrics:
        if name == "vllm:cache_config_info" and "num_gpu_blocks" in labels:
            try:
                pool_tokens = int(labels["num_gpu_blocks"]) * int(labels.get("block_size", 16))
            except ValueError:
                pass
            break
    return {
        "kv_usage": kv_usage,
        "running": _sum(metrics, "vllm:num_requests_running"),
        "waiting": _sum(metrics, "vllm:num_requests_waiting"),
        "preemptions": _sum(metrics, "vllm:num_preemptions_total"),
        "pool_tokens": pool_tokens,
    }


# Metrics are local: the HTTP(S)_PROXY of the environment must not apply.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch(url, timeout=0.8):
    with _OPENER.open(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def looks_like_vllm(text):
    return "vllm:" in text
