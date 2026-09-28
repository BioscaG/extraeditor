"""Generate the baseline SFX library procedurally (§14.1).

Every file in `library/sfx/` needs a known license. Rather than depending on
downloads that may vanish or carry unclear terms, the baseline set is **synthesized
here**: the code is the license, so `sfx.yaml` can honestly say CC0 and the origin
is auditable. Curated packs can be added alongside with their own per-file terms.

Each sound records its **peak offset** — the instant of the hit — because that is
what the mixer aligns to the anchor frame, not the file start (§14.1).

    python library/sfx/make_sfx.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import yaml

SR = 48000
HERE = Path(__file__).parent


def _write(path: Path, x: np.ndarray) -> None:
    """Write mono float audio as a 24-bit WAV via ffmpeg."""
    peak = float(np.max(np.abs(x))) or 1.0
    normalized = (x / peak * 0.89).astype(np.float32)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1",
         "-i", "pipe:0", "-c:a", "pcm_s24le", str(path)],
        input=normalized.tobytes(), check=True,
    )


def _t(duration: float) -> np.ndarray:
    return np.arange(int(duration * SR)) / SR


def _noise(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


def _onepole_lowpass(x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """Single-pole lowpass. Enough to shape noise into air rather than hiss."""
    alpha = 1.0 - np.exp(-2 * np.pi * cutoff_hz / SR)
    out = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc += alpha * (v - acc)
        out[i] = acc
    return out


def _onepole_highpass(x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    return x - _onepole_lowpass(x, cutoff_hz)


def whoosh(duration: float = 0.45, seed: int = 1, bright: bool = True) -> tuple[np.ndarray, float]:
    """Filtered noise sweeping up then down — a movement sound, peaking in the middle."""
    t = _t(duration)
    n = len(t)
    noise = _noise(n, seed)
    # A sweeping band, approximated by crossfading a lowpassed and highpassed copy.
    sweep = np.sin(np.pi * t / duration) ** 2
    low = _onepole_lowpass(noise, 900)
    high = _onepole_highpass(noise, 2500 if bright else 1400)
    body = low * (1 - sweep) + high * sweep
    envelope = np.sin(np.pi * t / duration) ** 1.4
    return body * envelope, duration / 2


def impact(duration: float = 0.7, seed: int = 2, deep: bool = True) -> tuple[np.ndarray, float]:
    """A transient hit: click plus a pitched-down body. Peak is at the very start."""
    t = _t(duration)
    n = len(t)
    base = 55.0 if deep else 110.0
    # Downward pitch sweep gives the weight; exponential decay gives the transient.
    phase = 2 * np.pi * base * (duration / 2.5) * (1 - np.exp(-2.5 * t / duration))
    body = np.sin(phase) * np.exp(-6 * t / duration)
    click = _onepole_highpass(_noise(n, seed), 3000) * np.exp(-90 * t)
    return (body * 0.9 + click * 0.35).astype(np.float32), 0.004


def riser(duration: float = 2.0, seed: int = 3) -> tuple[np.ndarray, float]:
    """Noise and pitch rising to a peak at the very end, so it lands on the downbeat."""
    t = _t(duration)
    n = len(t)
    progress = t / duration
    tone = np.sin(2 * np.pi * (200 + 1400 * progress**2) * t)
    air = _onepole_highpass(_noise(n, seed), 1200) * progress
    envelope = progress**1.8
    return ((tone * 0.35 + air * 0.65) * envelope).astype(np.float32), duration


def downlifter(duration: float = 1.2, seed: int = 4) -> tuple[np.ndarray, float]:
    """The inverse of a riser: falls away. Peak at the start, tail to nothing."""
    t = _t(duration)
    n = len(t)
    progress = t / duration
    tone = np.sin(2 * np.pi * (900 - 780 * progress) * t)
    air = _onepole_lowpass(_noise(n, seed), 2500) * (1 - progress)
    envelope = np.exp(-3 * progress)
    return ((tone * 0.4 + air * 0.6) * envelope).astype(np.float32), 0.01


def tick(duration: float = 0.09, seed: int = 5) -> tuple[np.ndarray, float]:
    """A soft click for text pops. Quiet and short enough to sit under everything."""
    t = _t(duration)
    body = np.sin(2 * np.pi * 2200 * t) * np.exp(-60 * t)
    noise = _onepole_highpass(_noise(len(t), seed), 4000) * np.exp(-140 * t)
    return (body * 0.6 + noise * 0.4).astype(np.float32), 0.002


def pop(duration: float = 0.14, seed: int = 6) -> tuple[np.ndarray, float]:
    """A pitched blip: warmer than a tick, for word emphasis."""
    t = _t(duration)
    phase = 2 * np.pi * 620 * t * (1 + 0.6 * np.exp(-30 * t))
    return (np.sin(phase) * np.exp(-28 * t)).astype(np.float32), 0.006


def sub_drop(duration: float = 1.6, seed: int = 7) -> tuple[np.ndarray, float]:
    """A sine falling from 90 Hz to nearly DC. The drop's weight."""
    t = _t(duration)
    progress = t / duration
    freq = 90 * np.exp(-2.6 * progress)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    return (np.sin(phase) * np.exp(-1.6 * progress)).astype(np.float32), 0.02


def crowd_bed(duration: float = 6.0, seed: int = 8) -> tuple[np.ndarray, float]:
    """A steady band-limited murmur, to glue dry clip audio into an ambience."""
    n = int(duration * SR)
    noise = _noise(n, seed)
    band = _onepole_lowpass(_onepole_highpass(noise, 250), 3200)
    # Slow amplitude drift so it does not read as flat hiss.
    t = _t(duration)
    drift = 1 + 0.12 * np.sin(2 * np.pi * 0.23 * t) + 0.08 * np.sin(2 * np.pi * 0.07 * t)
    fade = np.minimum(1.0, np.minimum(t, duration - t) / 0.4)
    return (band * drift * fade).astype(np.float32), duration / 2


def tape_stop(duration: float = 0.8, seed: int = 9) -> tuple[np.ndarray, float]:
    """A pitch-and-amplitude collapse, for freeze frames."""
    t = _t(duration)
    progress = t / duration
    freq = 440 * (1 - progress) ** 1.4 + 20
    phase = 2 * np.pi * np.cumsum(freq) / SR
    wobble = 1 + 0.05 * np.sin(2 * np.pi * 7 * t)
    return (np.sin(phase * wobble) * (1 - progress) ** 0.8).astype(np.float32), 0.01


SPECS = [
    ("whoosh.fast", "whoosh", lambda: whoosh(0.32, 11, True), "high"),
    ("whoosh.slow", "whoosh", lambda: whoosh(0.7, 12, False), "medium"),
    ("whoosh.soft", "whoosh", lambda: whoosh(0.5, 13, False), "low"),
    ("impact.deep", "impact", lambda: impact(0.8, 21, True), "high"),
    ("impact.soft", "impact", lambda: impact(0.45, 22, False), "medium"),
    ("riser.short", "riser", lambda: riser(1.0, 31), "medium"),
    ("riser.long", "riser", lambda: riser(2.4, 32), "high"),
    ("downlifter.fall", "downlifter", lambda: downlifter(1.2, 41), "medium"),
    ("tick.soft", "tick", lambda: tick(0.09, 51), "low"),
    ("pop.warm", "pop", lambda: pop(0.14, 61), "low"),
    ("sub_drop.deep", "sub_drop", lambda: sub_drop(1.6, 71), "high"),
    ("crowd_bed.murmur", "crowd_bed", lambda: crowd_bed(6.0, 81), "low"),
    ("tape_stop.collapse", "tape_stop", lambda: tape_stop(0.8, 91), "medium"),
]


def build_all(out_dir: Path = HERE) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for sfx_id, category, generator, energy in SPECS:
        audio, peak_offset = generator()
        filename = f"{sfx_id.replace('.', '_')}.wav"
        _write(out_dir / filename, audio)
        # Measured loudness of the normalized file, so the mixer can set sane gains.
        rms = float(np.sqrt(np.mean((audio / (np.max(np.abs(audio)) or 1) * 0.89) ** 2)))
        entries.append({
            "id": sfx_id,
            "file": filename,
            "category": category,
            "duration_s": round(len(audio) / SR, 4),
            "peak_offset_s": round(peak_offset, 4),
            "loudness_lufs": round(20 * float(np.log10(max(rms, 1e-9))), 2),
            "energy": energy,
            "license": "CC0-1.0",
            "source": "synthesized by library/sfx/make_sfx.py",
        })
    manifest = {
        "note": (
            "Synthesized procedurally so the license is unambiguous (§14.1). "
            "peak_offset_s is the instant of the hit, which the mixer aligns to the "
            "anchor frame. Regenerate with `python library/sfx/make_sfx.py`."
        ),
        "sfx": entries,
    }
    (out_dir / "sfx.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest


if __name__ == "__main__":
    manifest = build_all()
    print(f"Wrote {len(manifest['sfx'])} SFX + sfx.yaml to {HERE}")
