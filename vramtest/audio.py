"""Small PCM / WAV helpers shared by the dataset and the adapters.

Raw audio formats: ``f32le`` (float32 little-endian) and ``s16le`` (int16 little-endian), mono.
"""
from __future__ import annotations

import io
import wave

import numpy as np

FORMATS = ("f32le", "s16le")


def bytes_per_sample(fmt: str) -> int:
    return 4 if fmt == "f32le" else 2


def seconds(raw: bytes, fmt: str, sample_rate: int) -> float:
    return len(raw) / bytes_per_sample(fmt) / sample_rate if sample_rate else 0.0


def decode(raw: bytes, fmt: str) -> np.ndarray:
    """Raw PCM bytes -> float32 samples in [-1, 1]."""
    if fmt == "f32le":
        return np.frombuffer(raw[: len(raw) // 4 * 4], dtype="<f4").astype(np.float32)
    return np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0


def encode(x: np.ndarray, fmt: str) -> bytes:
    if fmt == "f32le":
        return np.asarray(x, dtype="<f4").tobytes()
    return (np.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    from math import gcd

    from scipy.signal import resample_poly
    g = gcd(sr_from, sr_to)
    return resample_poly(x, sr_to // g, sr_from // g).astype(np.float32)


def wav_bytes(x: np.ndarray, sample_rate: int) -> bytes:
    """Float samples -> 16-bit mono WAV file bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(encode(x, "s16le"))
    return buf.getvalue()


def read_wav(src) -> tuple[np.ndarray, int]:
    """16-bit PCM WAV (path or bytes) -> (float mono samples, sample_rate)."""
    f = io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src
    with wave.open(f, "rb") as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{src if isinstance(src, str) else 'WAV'}: only 16-bit PCM WAV is supported")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def write_wav(path: str, x: np.ndarray, sample_rate: int) -> None:
    with open(path, "wb") as f:
        f.write(wav_bytes(x, sample_rate))


def to_float(raw: bytes, audio_fmt: dict) -> tuple[np.ndarray, int] | None:
    """Decode service output audio described by ``{format, sample_rate}``; None for compressed formats."""
    fmt, sr = audio_fmt.get("format", "s16le"), int(audio_fmt.get("sample_rate", 24000))
    if fmt == "wav" or raw[:4] == b"RIFF":
        try:
            return read_wav(bytes(raw))
        except Exception:  # noqa: BLE001 — streamed WAV may carry a bogus header length
            return decode(bytes(raw[44:]), "s16le"), sr
    if fmt in FORMATS or fmt == "pcm":
        return decode(raw, "f32le" if fmt == "f32le" else "s16le"), sr
    return None


def output_seconds(raw: bytes, audio_fmt: dict) -> float | None:
    """Duration of service output audio, or None when the format is not decodable (mp3, opus, ...)."""
    if not raw:
        return 0.0
    fmt, sr = audio_fmt.get("format", "s16le"), int(audio_fmt.get("sample_rate", 24000))
    if fmt in FORMATS or fmt == "pcm":
        return seconds(raw, "f32le" if fmt == "f32le" else "s16le", sr)
    if fmt == "wav" or raw[:4] == b"RIFF":
        try:
            x, rate = read_wav(bytes(raw))
            return len(x) / rate
        except Exception:  # noqa: BLE001 — streamed WAV may carry a bogus header length
            return max(0, len(raw) - 44) / 2 / sr
    return None
