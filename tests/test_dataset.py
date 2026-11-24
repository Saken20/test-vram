import os
import random

import numpy as np

from tests.conftest import run
from vramtest import audio, config
from vramtest.adapters import make_adapter
from vramtest.dataset import Dataset, gen_audio, load_system_prompt


def _cfg(tmp_path, **ds):
    return config.from_dict({"services": {"a": {"adapter": "openai", "url": "http://h:1"}},
                             "dataset": {"audio_dir": str(tmp_path / "audio"), **ds}})


def test_default_prompts_are_neutral_and_loaded(tmp_path):
    ds = Dataset(_cfg(tmp_path))
    assert 15 <= len(ds.texts) <= 30 and ds.clips == [] and ds.system_prompt == ""


def test_inline_texts_messages_and_history(tmp_path):
    ds = Dataset(_cfg(tmp_path, texts=["one", "two"], system_prompt="Be brief.", history_turns=[1, 1]))
    rng = random.Random(1)
    p = ds.payload(rng, session=5)
    assert p.text in ("one", "two") and p.audio is None and p.session == 5
    assert p.messages[0] == {"role": "system", "content": "Be brief."}
    assert [m["role"] for m in p.messages] == ["system", "user", "assistant", "user"]
    assert p.messages[-1] == {"role": "user", "content": p.text}
    p = ds.payload(rng, history=[])
    assert len(p.messages) == 2


def test_system_prompt_from_file(tmp_path):
    f = tmp_path / "sys.txt"
    f.write_text("From a file.\n")
    assert load_system_prompt(str(f)) == "From a file."
    assert load_system_prompt("Inline prompt.") == "Inline prompt."


def test_clips_are_resampled_and_encoded(tmp_path):
    d = tmp_path / "audio"
    d.mkdir()
    x = 0.5 * np.sin(np.linspace(0, 100, 8000)).astype(np.float32)
    audio.write_wav(str(d / "c1.wav"), x, 8000)          # 1 s at 8 kHz
    (d / "c1.txt").write_text("the transcript")
    ds = Dataset(_cfg(tmp_path, audio_in={"format": "s16le", "sample_rate": 16000}))
    (clip,) = ds.clips
    assert abs(clip.seconds - 1.0) < 1e-6 and len(clip.data) == 16000 * 2 and clip.text == "the transcript"
    p = ds.payload(random.Random(0))
    assert p.text == "the transcript" and p.audio == clip.data and p.audio_seconds == clip.seconds


def test_gen_audio_with_speech_service(openai_mock, tmp_path):
    cfg = config.from_dict({"services": {"tts": {"adapter": "openai", "url": openai_mock.url, "kind": "speech",
                                                 "audio_out": {"format": "s16le", "sample_rate": 24000}}}})
    adapter = make_adapter("tts", cfg["services"]["tts"])
    out = tmp_path / "gen"

    async def go():
        try:
            return await gen_audio(adapter, ["first", "", "third"], str(out), 16000, log=lambda *_: None)
        finally:
            await adapter.close()
    assert run(go()) == 2
    files = sorted(os.listdir(out))
    assert files == ["manifest.json", "prompt_00.txt", "prompt_00.wav", "prompt_02.txt", "prompt_02.wav"]
    x, sr = audio.read_wav(str(out / "prompt_00.wav"))
    assert sr == 16000 and abs(len(x) / sr - 0.3) < 0.01
