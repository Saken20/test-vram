import json

from vramtest import kv_calc


def test_nested_multimodal_config(tmp_path):
    cfg = {
        "torch_dtype": "bfloat16",
        "thinker_config": {
            "audio_config": {"encoder_layers": 24, "num_hidden_layers": 24},
            "text_config": {"num_hidden_layers": 28, "num_key_value_heads": 8,
                            "hidden_size": 2048, "num_attention_heads": 16},
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    for spec in (str(tmp_path), str(tmp_path / "config.json")):
        root, source = kv_calc.load_model_config(spec)
        params = kv_calc.params_from_config(root, source)
        assert params["layers"] == 28 and params["head_dim"] == 128 and params["dtype_bytes"] == 2
        assert kv_calc.bytes_per_token(params) == 2 * 28 * 8 * 128 * 2


def test_plain_config_and_kv_dtype():
    root = {"num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 4096,
            "torch_dtype": "float16"}
    assert kv_calc.bytes_per_token(kv_calc.params_from_config(root)) == 2 * 32 * 8 * 128 * 2
    assert kv_calc.bytes_per_token(kv_calc.params_from_config(root, kv_dtype="fp8")) == 2 * 32 * 8 * 128
    mla = {"num_hidden_layers": 4, "num_attention_heads": 8, "kv_lora_rank": 512, "qk_rope_head_dim": 64}
    assert kv_calc.bytes_per_token(kv_calc.params_from_config(mla)) == 4 * 576 * 2


def test_kv_info_local_dir_weights_and_override(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(
        {"num_hidden_layers": 2, "num_attention_heads": 4, "hidden_size": 256, "torch_dtype": "bfloat16"}))
    (tmp_path / "model-00001.safetensors").write_bytes(b"\0" * (1024 ** 2))
    info = kv_calc.kv_info({"model_config": str(tmp_path)})
    assert info["bytes_per_token"] == 2 * 2 * 4 * 64 * 2 and info["error"] is None
    assert abs(info["weights_gib"] - 1 / 1024) < 1e-9 and info["weights_source"] == "safetensors"
    info = kv_calc.kv_info({"model_config": str(tmp_path), "kv_bytes_per_token": 1000, "weights_gib": 5})
    assert info["bytes_per_token"] == 1000 and info["source"] == "config.kv_bytes_per_token"
    assert info["weights_gib"] == 5


def test_kv_info_offline_is_graceful(monkeypatch, tmp_path):
    monkeypatch.setattr(kv_calc, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(kv_calc, "HF_URL", "http://127.0.0.1:9/{repo}/config.json")
    info = kv_calc.kv_info({"model_config": "org/model"}, timeout=1)
    assert info["bytes_per_token"] is None and info["error"]
    assert kv_calc.kv_info({"model_config": "no such path"})["error"]
    assert kv_calc.kv_info({})["bytes_per_token"] is None


def test_hf_config_is_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(kv_calc, "CACHE_DIR", str(tmp_path))
    d = tmp_path / "org--model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"num_hidden_layers": 1, "num_attention_heads": 1, "hidden_size": 64}))
    root, source = kv_calc.load_model_config("org/model")
    assert source == "hf:org/model" and root["num_hidden_layers"] == 1


def test_token_counter_estimate():
    tc = kv_calc.TokenCounter("", chars_per_token=2.0)
    assert tc.count("abcd") == 2 and tc.count("") == 0
    assert tc.chat([{"role": "system", "content": "aaaa"}, {"role": "user", "content": "bb"}]) == (2 + 4) + (1 + 4)


def test_encoder_decoder_config_whisper_style(tmp_path):
    # Whisper large-v3 config.json shape: no num_attention_heads / hidden_size, decoder_* + d_model instead
    root = {"model_type": "whisper", "num_hidden_layers": 32, "encoder_layers": 32, "decoder_layers": 32,
            "encoder_attention_heads": 20, "decoder_attention_heads": 20, "d_model": 1280, "torch_dtype": "float16"}
    params = kv_calc.params_from_config(root)
    assert params["layers"] == 32 and params["kv_heads"] == 20 and params["head_dim"] == 64
    assert params["encoder_decoder"] and "cross-attention" in params["note"]
    assert kv_calc.bytes_per_token(params) == 2 * 32 * 20 * 64 * 2 == 163840
    (tmp_path / "config.json").write_text(json.dumps(root))
    info = kv_calc.kv_info({"model_config": str(tmp_path)})
    assert info["bytes_per_token"] == 163840 and info["error"] is None
