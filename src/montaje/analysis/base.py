"""Analyzer protocol and frame/audio decoding helpers (§9).

Analyzers are pure functions over the proxy/audio artifacts: same inputs and
params → same events. The runner handles caching and persistence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from montaje import ffmpeg
from montaje.models.asset import Asset
from montaje.models.events import Event


class Analyzer(Protocol):
    name: str
    version: int

    def params(self) -> dict: ...

    def run(self, asset: Asset, ws) -> list[Event]: ...


def proxy_height(video: Path, width: int) -> int:
    """Even height that preserves aspect ratio at `width`.

    Computed up front so raw decoded buffers can be reshaped deterministically —
    letting ffmpeg pick with `-2` would leave the height unknown to the caller.
    """
    out = ffmpeg.probe(["-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height", "-of", "csv=p=0", str(video)])
    w0, h0 = (int(x) for x in out.stdout.strip().split(",")[:2])
    return max(2, round(width * h0 / w0 / 2) * 2)


def decode_gray_frames(video: Path, fps: float, width: int = 96) -> tuple[np.ndarray, float]:
    """Decode to a (n, h, w) uint8 grayscale array at a fixed sample rate.

    Small frames are enough for luma/detail/motion statistics and keep a
    100-clip project in memory-friendly territory.
    """
    height = proxy_height(video, width)
    out = ffmpeg.run(["-v", "error", "-i", str(video),
                      "-vf", f"fps={fps},scale={width}:{height}",
                      "-pix_fmt", "gray", "-f", "rawvideo", "-"])
    n = len(out.stdout) // (width * height)
    frames = np.frombuffer(out.stdout[: n * width * height], dtype=np.uint8).reshape(n, height, width)
    return frames, fps


def decode_audio_mono(wav_16k: Path) -> tuple[np.ndarray, int]:
    """Load 16 kHz mono PCM as float32 in [-1, 1]."""
    out = ffmpeg.run(["-v", "error", "-i", str(wav_16k), "-f", "f32le",
                      "-ac", "1", "-ar", "16000", "-"])
    return np.frombuffer(out.stdout, dtype=np.float32), 16000


def spans_from_mask(mask: np.ndarray, rate: float, min_len_s: float = 0.0) -> list[tuple[float, float]]:
    """Convert a boolean per-sample mask into (t0, t1) spans."""
    spans: list[tuple[float, float]] = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            spans.append((start / rate, i / rate))
            start = None
    if start is not None:
        spans.append((start / rate, len(mask) / rate))
    return [(a, b) for a, b in spans if b - a >= min_len_s]


def detail_energy(frames: np.ndarray) -> np.ndarray:
    """Per-frame high-frequency energy (gradient magnitude mean) — blur/occlusion proxy."""
    gx = np.abs(np.diff(frames.astype(np.float32), axis=2)).mean(axis=(1, 2))
    gy = np.abs(np.diff(frames.astype(np.float32), axis=1)).mean(axis=(1, 2))
    return gx + gy
