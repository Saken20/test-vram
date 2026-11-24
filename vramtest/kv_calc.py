"""Analytic KV-cache size per token from a model's config.json, weights size, token counting.

``model_config`` may be a local config.json, a local model directory, or a Hugging Face repo id
("org/name"; config.json is downloaded over HTTPS and cached in ~/.cache/vramtest/). Offline or on
errors the result simply carries an ``error`` instead of numbers.
"""
from __future__ import annotations

import glob
import json
import os
import re
import urllib.request

GIB = 1024 ** 3
MIB = 1024 ** 2

DTYPE_BYTES = {
    "bfloat16": 2, "float16": 2, "half": 2, "bf16": 2, "fp16": 2, "float32": 4, "float": 4, "fp32": 4,
    "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1, "int8": 1,
}

HF_URL = "https://huggingface.co/{repo}/resolve/main/config.json"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "vramtest", "hf")
_REPO_ID = re.compile(r"^[\w.-]+/[\w.-]+$")

# keys under which multimodal models nest their text decoder config
_NESTED_KEYS = ("text_config", "thinker_config", "llm_config", "language_config", "decoder")


def _find_text_config(node):
    """Find the dict with the text decoder parameters inside a (possibly nested) config.json."""
    if isinstance(node, dict):
        if "decoder_layers" in node and "decoder_attention_heads" in node:  # encoder-decoder (Whisper, BART, ...)
            return node
        if "num_hidden_layers" in node and ("num_key_value_heads" in node or "num_attention_heads" in node
                                            or "kv_lora_rank" in node):
            return node
        for key in _NESTED_KEYS:
            if key in node:
                found = _find_text_config(node[key])
                if found:
                    return found
        for value in node.values():
            found = _find_text_config(value)
            if found:
                return found
    return None


def load_model_config(spec: str, timeout: float = 5.0) -> tuple[dict, str]:
    """``spec`` = local config.json / model dir / HF repo id -> (parsed config.json, source)."""
    path = os.path.expanduser(spec)
    if os.path.isdir(path):
        path = os.path.join(path, "config.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f), path
    if not _REPO_ID.match(spec):
        raise FileNotFoundError(f"model_config {spec!r}: not a local file/dir and not a Hugging Face repo id")
    cached = os.path.join(CACHE_DIR, spec.replace("/", "--"), "config.json")
    if os.path.isfile(cached):
        with open(cached, encoding="utf-8") as f:
            return json.load(f), f"hf:{spec}"
    req = urllib.request.Request(HF_URL.format(repo=spec), headers={"User-Agent": "vramtest"})
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:  # gated models (e.g. meta-llama/*)
        req.add_header("Authorization", f"Bearer {token}")
    # Uses the environment's proxy settings (this is an internet download, unlike the local services).
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    cfg = json.loads(data)
    try:
        os.makedirs(os.path.dirname(cached), exist_ok=True)
        with open(cached, "wb") as f:
            f.write(data)
    except OSError:
        pass
    return cfg, f"hf:{spec}"


def params_from_config(root: dict, source: str = "", kv_dtype: str | None = None) -> dict:
    """Decoder parameters relevant to the KV cache: dict(layers, kv_heads, head_dim, dtype_bytes, source, ...).

    Understands decoder-only configs (also nested in multimodal ones), MLA, and encoder-decoder configs
    (``decoder_layers`` / ``decoder_attention_heads`` / ``d_model``, e.g. Whisper: decoder self-attention only).
    """
    cfg = _find_text_config(root)
    if cfg is None:
        raise ValueError(f"{source or 'config'}: no text decoder parameters (num_hidden_layers, ...) found")
    dtype = kv_dtype if kv_dtype and kv_dtype != "auto" else (
        cfg.get("torch_dtype") or cfg.get("dtype") or root.get("torch_dtype") or root.get("dtype") or "bfloat16")
    dtype_bytes = DTYPE_BYTES.get(str(dtype).lower().replace("torch.", ""), 2)
    if cfg.get("kv_lora_rank"):  # MLA (DeepSeek-V2/V3 style): compressed latent + rope key per layer
        return {
            "layers": cfg["num_hidden_layers"], "mla": True,
            "kv_lora_rank": cfg["kv_lora_rank"], "qk_rope_head_dim": cfg.get("qk_rope_head_dim", 64),
            "dtype_bytes": dtype_bytes, "source": source,
        }
    if cfg.get("decoder_layers") and cfg.get("decoder_attention_heads"):
        # encoder-decoder: the decoder's self-attention KV grows per generated token. The cross-attention KV
        # (decoder layers x encoder length, per request) is extra and not included here.
        heads = cfg["decoder_attention_heads"]
        return {
            "layers": cfg["decoder_layers"],
            "kv_heads": heads,
            "head_dim": cfg.get("head_dim") or (cfg.get("d_model") or cfg["hidden_size"]) // heads,
            "dtype_bytes": dtype_bytes,
            "source": source,
            "encoder_decoder": True,
            "note": "encoder-decoder: decoder self-attention only; cross-attention KV is extra",
        }
    heads = cfg["num_attention_heads"]
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // heads
    return {
        "layers": cfg["num_hidden_layers"],
        "kv_heads": cfg.get("num_key_value_heads") or heads,
        "head_dim": head_dim,
        "dtype_bytes": dtype_bytes,
        "source": source,
    }


def bytes_per_token(params: dict) -> int:
    if params.get("mla"):
        return params["layers"] * (params["kv_lora_rank"] + params["qk_rope_head_dim"]) * params["dtype_bytes"]
    # K and V on every layer: kv_heads * head_dim elements each
    return 2 * params["layers"] * params["kv_heads"] * params["head_dim"] * params["dtype_bytes"]


def weights_gib_from_dir(path: str) -> float | None:
    """Total size of *.safetensors (else *.bin) in a local model directory, GiB."""
    if not path or not os.path.isdir(os.path.expanduser(path)):
        return None
    path = os.path.expanduser(path)
    files = glob.glob(os.path.join(path, "*.safetensors")) or glob.glob(os.path.join(path, "pytorch_model*.bin"))
    return sum(os.path.getsize(f) for f in files) / GIB if files else None


def kv_info(svc_cfg: dict, timeout: float = 5.0) -> dict:
    """Everything the memory model needs for one service, never raises.

    Returns dict(bytes_per_token, source, weights_gib, weights_source, error, + decoder params when known).
    """
    out = {"bytes_per_token": None, "source": None, "weights_gib": svc_cfg.get("weights_gib"),
           "weights_source": "config.weights_gib" if svc_cfg.get("weights_gib") else None, "error": None}
    spec = svc_cfg.get("model_config") or ""
    if spec:
        from .config import resolve
        local = resolve(spec)
        spec_or_local = local if os.path.exists(local) else spec
        try:
            root, source = load_model_config(spec_or_local, timeout)
            params = params_from_config(root, source, svc_cfg.get("kv_cache_dtype"))
            out.update(params)
            out["bytes_per_token"] = bytes_per_token(params)
        except Exception as exc:  # noqa: BLE001 — offline, missing file, unusual config
            out["error"] = f"{type(exc).__name__}: {exc}"[:300]
        if out["weights_gib"] is None:
            w = weights_gib_from_dir(spec_or_local)
            if w:
                out["weights_gib"], out["weights_source"] = w, "safetensors"
    if svc_cfg.get("kv_bytes_per_token"):
        out["bytes_per_token"] = int(svc_cfg["kv_bytes_per_token"])
        out["source"] = "config.kv_bytes_per_token"
        out["error"] = None
    return out


class TokenCounter:
    """Token counts: exact with a HF tokenizer (optional ``transformers``), else a chars-per-token estimate."""

    def __init__(self, tokenizer_path: str = "", chars_per_token: float = 4.0):
        self.chars_per_token = chars_per_token or 4.0
        self.tokenizer = None
        if tokenizer_path:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            except Exception as exc:  # noqa: BLE001 — the tokenizer is optional
                print(f"[tokens] tokenizer {tokenizer_path!r} not loaded ({exc}); using an estimate")

    @property
    def exact(self) -> bool:
        return self.tokenizer is not None

    def count(self, text: str | None) -> int:
        if not text:
            return 0
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return max(1, round(len(text) / self.chars_per_token))

    def chat(self, messages: list[dict]) -> int:
        """Prompt tokens of a chat (+ ~4 template tokens per message)."""
        return sum(self.count(m.get("content") if isinstance(m.get("content"), str) else "") + 4
                   for m in messages if m.get("content"))
