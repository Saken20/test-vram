"""Load, default and validate the YAML configuration.

The configuration is kept as plain nested dicts (easy to dump into result JSON files):

- ``load(path)`` reads YAML (default ``./config.yaml``), merges it over :data:`DEFAULTS`, fills per-service
  defaults (:data:`SERVICE_DEFAULTS` + adapter-specific ones) and validates. On errors it raises
  :class:`ConfigError` listing every problem. Non-fatal remarks (unknown keys, ...) go to ``cfg["_warnings"]``.
- ``cfg["services"][name]`` is a complete dict for every service; ``cfg["services"][name]["name"] == name``.
- ``cfg["pipeline"]`` is a list of ``{"service": str, "input": "audio"|"text"|"messages"|"prev"}`` (maybe empty).
- ``cfg["_path"]`` / ``cfg["_dir"]``: the config file and its directory (used by :func:`resolve`).

Derived per-service values filled by :func:`load`:

- ``port``: from ``url`` / ``address`` when not given (used to find the service's GPU processes);
- ``engine``: ``vllm`` for the openai adapter (except ``kind: speech``), else ``generic``.
"""
from __future__ import annotations

import copy
import os
import re
from urllib.parse import urlsplit

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = "config.yaml"

ADAPTERS = ("openai", "grpc", "http")
OPENAI_KINDS = ("chat", "completions", "transcription", "speech", "embeddings")
ENGINES = ("vllm", "generic")
INPUTS = ("audio", "text", "messages", "prev")
AUDIO_FORMATS = ("f32le", "s16le")
OUTPUT_AUDIO_FORMATS = ("s16le", "f32le", "pcm", "wav", "mp3", "opus", "aac", "flac")
HTTP_METHODS = ("GET", "POST", "PUT", "PATCH")

DEFAULTS = {
    "gpu": {"index": 0, "sample_hz": 10, "metrics_poll_s": 1.0},
    "services": {},
    "pipeline": [],
    "dataset": {
        "texts": "data/prompts.txt",
        "audio_dir": "data/audio",
        "audio_in": {"format": "f32le", "sample_rate": 16000},
        "system_prompt": "",
        "history_turns": [0, 2],
        "seed": 0,
    },
    "scenarios": {
        "idle": {"duration_s": 60},
        "burst": {"levels": [1, 10, 100], "rounds": 3, "pause_s": 3},
        "concurrency": {"users": 10, "duration_s": 120},
        "sessions": {
            "turns_per_session": 3,
            "session_duration_s": 30,
            "duration_s": 180,
            "profiles": {"normal": 4, "peak": 16},
            "stagger_start": False,     # true = sessions start at random moments within the first session_duration_s
        },
        "ramp": {
            "levels": [1, 2, 5, 10, 20, 40, 80, 160],
            "step_duration_s": 60,
            "max_error_rate": 0.05,
            "bisect_steps": 2,
            "cooldown_s": 5,
        },
    },
    "capacity": {
        "unit": "session",
        "safety_margin": 0.10,
        "targets": [{"label": "1 session", "sessions": 1}],
    },
}

# Keys every service gets (adapter-specific keys are added by ADAPTER_DEFAULTS).
SERVICE_DEFAULTS = {
    "adapter": None,
    "model": "",
    "request": {},
    "timeout_s": 120,
    # process discovery
    "port": None,
    "pids": [],
    "cmdline_match": "",
    # memory model
    "engine": None,              # derived: vllm for openai (except speech), else generic
    "metrics_url": "",
    "model_config": "",
    "kv_bytes_per_token": None,
    "kv_cache_dtype": None,      # e.g. fp8 when vLLM runs with --kv-cache-dtype fp8
    "weights_gib": None,
    "kv_pool_tokens": None,
    "overhead_gib": 1.0,
    "max_model_len": None,
    "prefix_caching": True,
    # token accounting (KV estimate when the response has no `usage`)
    "tokenizer": "",             # optional HF tokenizer id/path (needs `transformers`)
    "chars_per_token": 4.0,
    "audio_tokens_per_s": 0.0,   # KV tokens per second of input audio (audio-LLM style ASR)
    "audio_out": {"format": "s16le", "sample_rate": 24000},
    "limits": {"max_p95_latency_s": 20, "max_p95_ttfb_s": 3},
}

ADAPTER_DEFAULTS = {
    "openai": {"url": "", "kind": "chat"},
    "grpc": {"address": "", "proto": "", "include_dirs": [], "method": "", "stream_audio": None,
             "response": {}, "metadata": {}},
    "http": {"url": "", "method": "POST", "json": None, "multipart": None, "headers": {}, "stream": False,
             "response": {}},
}

STREAM_AUDIO_DEFAULTS = {"field": "", "chunk_bytes": 15360, "realtime": False}


class ConfigError(ValueError):
    """Invalid configuration; ``errors`` holds one message per problem."""

    def __init__(self, errors, path=None):
        self.errors = list(errors)
        where = f" in {path}" if path else ""
        super().__init__(f"invalid config{where}:\n  - " + "\n  - ".join(self.errors))


def _merge(base, override):
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def port_from(svc: dict) -> int | None:
    """Port of a service from ``url`` (http/openai) or ``address`` (grpc host:port)."""
    if svc.get("url"):
        try:
            parts = urlsplit(svc["url"])
            return parts.port or {"http": 80, "https": 443}.get(parts.scheme)
        except ValueError:
            return None
    address = svc.get("address") or ""
    m = re.search(r":(\d+)$", address)
    return int(m.group(1)) if m else None


def _service_defaults(name: str, raw) -> dict:
    svc = copy.deepcopy(SERVICE_DEFAULTS)
    adapter = raw.get("adapter") if isinstance(raw, dict) else None
    _merge(svc, copy.deepcopy(ADAPTER_DEFAULTS.get(adapter, {})))
    _merge(svc, copy.deepcopy(raw) if isinstance(raw, dict) else {})
    svc["name"] = name
    if isinstance(svc.get("stream_audio"), dict):
        svc["stream_audio"] = {**STREAM_AUDIO_DEFAULTS, **svc["stream_audio"]}
    if svc.get("port") is None:
        svc["port"] = port_from(svc)
    if svc.get("engine") is None:
        svc["engine"] = "vllm" if adapter == "openai" and svc.get("kind") != "speech" else "generic"
    return svc


def _known_keys(adapter):
    return set(SERVICE_DEFAULTS) | set(ADAPTER_DEFAULTS.get(adapter, {})) | {"name"}


def _check_audio_fmt(errors, where, fmt, allowed):
    if not isinstance(fmt, dict):
        errors.append(f"{where}: expected a mapping {{format, sample_rate}}")
        return
    if fmt.get("format") not in allowed:
        errors.append(f"{where}.format: {fmt.get('format')!r} is not one of {', '.join(allowed)}")
    if not isinstance(fmt.get("sample_rate"), int) or fmt["sample_rate"] <= 0:
        errors.append(f"{where}.sample_rate: expected a positive integer")


def _validate_service(name, svc, errors, warnings):
    where = f"services.{name}"
    adapter = svc["adapter"]
    if adapter not in ADAPTERS:
        errors.append(f"{where}.adapter: unknown adapter {adapter!r} (expected one of {', '.join(ADAPTERS)})")
        return
    if adapter == "openai":
        if not svc.get("url"):
            errors.append(f"{where}.url: required for the openai adapter (e.g. http://localhost:8000)")
        if svc["kind"] not in OPENAI_KINDS:
            errors.append(f"{where}.kind: {svc['kind']!r} is not one of {', '.join(OPENAI_KINDS)}")
    elif adapter == "grpc":
        if not svc.get("address"):
            errors.append(f"{where}.address: required for the grpc adapter (host:port)")
        if not svc.get("proto"):
            errors.append(f"{where}.proto: required for the grpc adapter (path to a .proto file)")
        elif not os.path.exists(resolve(svc["proto"])):
            errors.append(f"{where}.proto: file not found: {svc['proto']}")
        if not svc.get("method"):
            errors.append(f"{where}.method: required for the grpc adapter (\"<package>.<Service>/<Method>\")")
        elif not re.fullmatch(r"(?:\w+\.)*\w+/\w+", str(svc["method"])):
            errors.append(f"{where}.method: {svc['method']!r} must look like \"<package>.<Service>/<Method>\"")
        sa = svc.get("stream_audio")
        if sa is not None and (not isinstance(sa, dict) or not sa.get("field")):
            errors.append(f"{where}.stream_audio: expected a mapping with `field` (message field for audio chunks)")
        elif sa and (not isinstance(sa["chunk_bytes"], int) or sa["chunk_bytes"] <= 0):
            errors.append(f"{where}.stream_audio.chunk_bytes: expected a positive integer")
        if not isinstance(svc.get("request", {}), (dict, list)):
            errors.append(f"{where}.request: expected a mapping (or a list of mappings = several messages)")
    elif adapter == "http":
        if not svc.get("url"):
            errors.append(f"{where}.url: required for the http adapter (full endpoint URL)")
        if str(svc.get("method", "")).upper() not in HTTP_METHODS:
            errors.append(f"{where}.method: {svc.get('method')!r} is not one of {', '.join(HTTP_METHODS)}")
        if svc.get("json") is not None and svc.get("multipart") is not None:
            errors.append(f"{where}: use either `json` or `multipart`, not both")
        if svc.get("multipart") is not None and not isinstance(svc["multipart"], dict):
            errors.append(f"{where}.multipart: expected a mapping of form fields")
    if adapter != "openai" and "kind" in svc and svc["kind"] is not None:
        warnings.append(f"{where}.kind: only used by the openai adapter (ignored)")
    if svc["engine"] not in ENGINES:
        errors.append(f"{where}.engine: {svc['engine']!r} is not one of {', '.join(ENGINES)}")
    if adapter in ("grpc", "http") and not isinstance(svc.get("response") or {}, dict):
        errors.append(f"{where}.response: expected a mapping like {{text: \"path.to.field\"}}")
    if not isinstance(svc.get("request") or {}, (dict, list)):
        errors.append(f"{where}.request: expected a mapping")
    if not isinstance(svc.get("timeout_s"), (int, float)) or svc["timeout_s"] <= 0:
        errors.append(f"{where}.timeout_s: expected a positive number")
    if not isinstance(svc.get("pids") or [], list):
        errors.append(f"{where}.pids: expected a list of process ids")
    _check_audio_fmt(errors, f"{where}.audio_out", svc.get("audio_out"), OUTPUT_AUDIO_FORMATS)
    if not isinstance(svc.get("limits"), dict):
        errors.append(f"{where}.limits: expected a mapping")
    unknown = set(svc) - _known_keys(adapter)
    if unknown:
        warnings.append(f"{where}: unknown keys ignored: {', '.join(sorted(unknown))}")


def validate(cfg: dict) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for a config that already has defaults applied."""
    errors, warnings = [], []
    unknown = set(cfg) - set(DEFAULTS) - {"_path", "_dir", "_warnings"}
    if unknown:
        warnings.append(f"unknown top-level keys ignored: {', '.join(sorted(unknown))}")
    services = cfg.get("services")
    if not isinstance(services, dict) or not services:
        errors.append("services: define at least one service (a mapping name -> settings)")
        services = {}
    for name, svc in services.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            errors.append(f"services.{name}: service names may use letters, digits, '_', '-', '.'")
        _validate_service(name, svc, errors, warnings)

    pipeline = cfg.get("pipeline") or []
    if not isinstance(pipeline, list):
        errors.append("pipeline: expected a list of steps like {service: NAME, input: text}")
        pipeline = []
    for i, step in enumerate(pipeline):
        where = f"pipeline[{i}]"
        if not isinstance(step, dict):
            errors.append(f"{where}: expected a mapping {{service, input}}")
            continue
        if step.get("service") not in services:
            errors.append(f"{where}.service: unknown service {step.get('service')!r}"
                          f" (defined: {', '.join(services) or 'none'})")
        if step.get("input") not in INPUTS:
            errors.append(f"{where}.input: {step.get('input')!r} is not one of {', '.join(INPUTS)}")
        elif i == 0 and step["input"] == "prev":
            errors.append(f"{where}.input: the first step has no previous step (use audio, text or messages)")

    ds = cfg["dataset"]
    _check_audio_fmt(errors, "dataset.audio_in", ds.get("audio_in"), AUDIO_FORMATS)
    ht = ds.get("history_turns")
    if not (isinstance(ht, int) or (isinstance(ht, list) and len(ht) == 2 and all(isinstance(x, int) for x in ht))):
        errors.append("dataset.history_turns: expected an integer or [min, max]")
    elif min(ht if isinstance(ht, list) else [ht]) < 0 or (isinstance(ht, list) and ht[0] > ht[1]):
        errors.append("dataset.history_turns: expected a non-negative integer or [min, max] with min <= max")
    if not isinstance(ds.get("texts"), (str, list)):
        errors.append("dataset.texts: expected a file path or a list of prompts")

    sc = cfg["scenarios"]
    for key in ("levels",):
        for scen in ("burst", "ramp"):
            levels = sc[scen].get(key)
            if not isinstance(levels, list) or not levels or not all(isinstance(x, int) and x > 0 for x in levels):
                errors.append(f"scenarios.{scen}.levels: expected a non-empty list of positive integers")
    profiles = sc["sessions"].get("profiles")
    if not isinstance(profiles, dict) or not all(isinstance(v, int) and v > 0 for v in profiles.values()):
        errors.append("scenarios.sessions.profiles: expected a mapping name -> number of concurrent sessions")
    for i, t in enumerate(cfg["capacity"].get("targets") or []):
        if not isinstance(t, dict) or not isinstance(t.get("sessions"), (int, float)):
            errors.append(f"capacity.targets[{i}]: expected {{label, sessions}}")
    return errors, warnings


def apply_defaults(raw: dict) -> dict:
    """Merge a raw (parsed YAML) config over the defaults and fill per-service defaults (no validation)."""
    if raw is not None and not isinstance(raw, dict):
        raise ConfigError(["the file must contain a YAML mapping at the top level"])
    raw = copy.deepcopy(raw or {})
    services_raw = raw.pop("services", None)
    cfg = _merge(copy.deepcopy(DEFAULTS), raw)
    if services_raw is not None and not isinstance(services_raw, dict):
        cfg["services"] = services_raw  # reported by validate()
    else:
        cfg["services"] = {name: _service_defaults(name, svc) for name, svc in (services_raw or {}).items()}
    if cfg.get("pipeline") is None:
        cfg["pipeline"] = []
    return cfg


def load(path: str | None = None, *, check: bool = True) -> dict:
    """Read the YAML config (default ``./config.yaml``), apply defaults and validate."""
    path = path or DEFAULT_PATH
    if not os.path.exists(path):
        raise ConfigError([f"config file not found: {path} (copy one of examples/*.yaml to ./config.yaml "
                           "or pass --config PATH)"])
    with open(path, encoding="utf-8") as f:
        try:
            raw = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ConfigError([f"YAML syntax error: {exc}"], path) from None
    return from_dict(raw, path=path, check=check)


def from_dict(raw: dict, path: str | None = None, *, check: bool = True) -> dict:
    """Same as :func:`load` for an already parsed mapping (handy in tests)."""
    cfg = apply_defaults(raw)
    cfg["_path"] = os.path.abspath(path) if path else None
    cfg["_dir"] = os.path.dirname(cfg["_path"]) if path else os.getcwd()
    _CONFIG_DIRS[:] = [cfg["_dir"]]
    errors, warnings = validate(cfg)
    cfg["_warnings"] = warnings
    if check and errors:
        raise ConfigError(errors, path)
    return cfg


# directory of the most recently loaded config file: second place where relative paths are searched
_CONFIG_DIRS: list[str] = []


def resolve(path: str | None) -> str | None:
    """Resolve a path from the config: absolute as is; else relative to the current directory,
    then to the config file's directory, then to the repository root (first existing wins)."""
    if not path:
        return path
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    for base in (os.getcwd(), *_CONFIG_DIRS, ROOT):
        candidate = os.path.join(base, path)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(os.getcwd(), path)


def apply_quick(cfg: dict) -> dict:
    """Short durations for a smoke run (``--quick``)."""
    sc = cfg["scenarios"]
    sc["idle"]["duration_s"] = 10
    sc["burst"]["rounds"] = 1
    sc["concurrency"]["duration_s"] = 40
    sc["sessions"]["duration_s"] = 40
    sc["ramp"]["step_duration_s"] = 15
    sc["ramp"]["bisect_steps"] = 0
    sc["ramp"]["cooldown_s"] = 1
    return cfg


def public(cfg: dict) -> dict:
    """Config without private ``_`` keys (for result files)."""
    return {k: v for k, v in cfg.items() if not k.startswith("_")}
