"""Test data: prompts, audio clips, chat messages, and ``gen_audio`` (synthesize clips with a speech service).

    dataset:
      texts: data/prompts.txt        # one prompt per line ('#' comments) or an inline list
      audio_dir: data/audio          # *.wav, 16-bit PCM; an optional <name>.txt next to it = transcript
      audio_in: {format: f32le, sample_rate: 16000}   # encoding of Payload.audio / {audio}
      system_prompt: ""              # path or inline text
      history_turns: [0, 2]          # random previous (user, assistant) turns in chat messages
"""
from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass

from . import audio as audio_util
from .adapters.base import Adapter, Payload
from .config import resolve

# Generic assistant turns used to build chat history (only their length matters for the KV cache).
HISTORY_REPLIES = [
    "Sure. In short: it depends on the context, but the most common answer is the simplest one.",
    "Here is a brief overview with the key points and one practical example.",
    "Good question. There are three main factors to consider, and the first one matters most.",
    "I would start with the basics, then compare the options side by side before deciding.",
    "That is correct. The main reason is efficiency, although there are a few exceptions.",
    "Let me summarize the previous answer and add a couple of details you may find useful.",
]


@dataclass
class AudioClip:
    path: str
    data: bytes          # encoded per dataset.audio_in
    seconds: float
    text: str            # transcript (empty when unknown)


def load_texts(spec) -> list[str]:
    if isinstance(spec, list):
        return [str(t).strip() for t in spec if str(t).strip()]
    path = resolve(spec)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"dataset.texts: file not found: {spec}")
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]


def load_system_prompt(spec: str) -> str:
    """``dataset.system_prompt``: a path to a text file, or the prompt itself."""
    if not spec:
        return ""
    path = resolve(spec)
    if len(spec) < 4096 and "\n" not in spec and path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    return spec


def load_clips(audio_dir: str, audio_in: dict) -> list[AudioClip]:
    """All ``audio_dir/*.wav`` resampled and encoded to ``audio_in``."""
    clips = []
    fmt, sr = audio_in["format"], audio_in["sample_rate"]
    for path in sorted(glob.glob(os.path.join(audio_dir, "*.wav"))):
        x, rate = audio_util.read_wav(path)
        x = audio_util.resample(x, rate, sr)
        txt = os.path.splitext(path)[0] + ".txt"
        text = ""
        if os.path.exists(txt):
            with open(txt, encoding="utf-8") as f:
                text = f.read().strip()
        clips.append(AudioClip(path, audio_util.encode(x, fmt), len(x) / sr, text))
    return clips


class Dataset:
    """Prompts, audio clips and the chat message builder for one config."""

    def __init__(self, cfg: dict):
        ds = cfg["dataset"]
        self.audio_in = dict(ds["audio_in"])
        self.texts = load_texts(ds["texts"])
        if not self.texts:
            raise ValueError(f"dataset.texts: no prompts in {ds['texts']!r}")
        self.system_prompt = load_system_prompt(ds.get("system_prompt") or "")
        ht = ds.get("history_turns", 0)
        self.history_turns = (ht, ht) if isinstance(ht, int) else (int(ht[0]), int(ht[1]))
        self.audio_dir = resolve(ds["audio_dir"])
        self.clips = load_clips(self.audio_dir, self.audio_in)

    def reload_clips(self) -> list[AudioClip]:
        self.clips = load_clips(self.audio_dir, self.audio_in)
        return self.clips

    def random_history(self, rng: random.Random) -> list[dict]:
        history = []
        for _ in range(rng.randint(*self.history_turns)):
            history.append({"role": "user", "content": rng.choice(self.texts)})
            history.append({"role": "assistant", "content": rng.choice(HISTORY_REPLIES)})
        return history

    def messages(self, text: str, history=()) -> list[dict]:
        """System prompt (if any) + history + the user text."""
        msgs = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        return [*msgs, *history, {"role": "user", "content": text}]

    def payload(self, rng: random.Random, session: int = 0, history: list[dict] | None = None) -> Payload:
        """A turn input: a random clip (if any), its transcript or a random prompt, chat messages.

        history: explicit chat history (e.g. a session's previous turns); default: random history.
        """
        clip = rng.choice(self.clips) if self.clips else None
        text = clip.text if clip and clip.text else rng.choice(self.texts)
        hist = self.random_history(rng) if history is None else history
        return Payload(text=text, audio=clip.data if clip else None, audio_seconds=clip.seconds if clip else None,
                       messages=self.messages(text, hist), session=session)


async def gen_audio(adapter: Adapter, texts: list[str], out_dir: str, sample_rate: int = 16000,
                    prefix: str = "prompt", log=print) -> int:
    """Synthesize ``texts`` with any adapter whose result carries audio; writes ``out_dir/<prefix>_NN.wav``
    (16-bit mono at ``sample_rate``) + ``.txt`` transcripts. Returns the number of clips written."""
    os.makedirs(out_dir, exist_ok=True)
    made = 0
    for i, text in enumerate(texts):
        result = await adapter.call(Payload(text=text, session=0))
        if not result.ok or not result.audio:
            rec = result.record
            log(f"[gen-audio] no audio for #{i} ({rec.code if rec else '?'} {rec.error[:120] if rec else ''})")
            continue
        decoded = audio_util.to_float(result.audio, adapter.audio_out)
        if decoded is None:
            log(f"[gen-audio] {adapter.name}: audio format {adapter.audio_out.get('format')!r} cannot be decoded; "
                "use pcm/s16le/f32le/wav (e.g. request: {response_format: pcm})")
            return made
        x, rate = decoded
        x = audio_util.resample(x, rate, sample_rate)
        base = os.path.join(out_dir, f"{prefix}_{i:02d}")
        audio_util.write_wav(base + ".wav", x, sample_rate)
        with open(base + ".txt", "w", encoding="utf-8") as f:
            f.write(text)
        made += 1
        log(f"[gen-audio] {base}.wav  {len(x) / sample_rate:.1f} s")
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"generated": made, "sample_rate": sample_rate, "service": adapter.name}, f)
    return made
