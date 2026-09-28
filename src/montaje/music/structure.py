"""Music structure: sections and energy curve (§9, §14.2).

Sections are found by novelty detection on a self-similarity matrix of MFCC-like
features, with boundaries snapped to downbeats — a section change that is not on a
bar line is musically wrong and would drag the whole edit off the grid.

`allin1` replaces this behind `MusicStructure` (DECISIONS.md).
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from montaje.analysis.base import decode_audio_mono
from montaje.music.beats import BeatGrid, analyze_beats, transient_envelope

# Section roles, ordered by typical energy. The agent maps story beats onto these.
ROLE_ORDER = ("intro", "build", "drop", "break", "outro")


class Section(BaseModel):
    id: str
    t0: float
    t1: float
    role: str = "build"
    energy: float = 0.5  # 0–1, normalized within the track
    bars: int = 0

    @property
    def duration_s(self) -> float:
        return self.t1 - self.t0


class MusicStructure(BaseModel):
    grid: BeatGrid
    sections: list[Section] = Field(default_factory=list)
    energy_curve: list[float] = Field(default_factory=list)  # per-second, 0–1
    duration_s: float = 0.0

    def section_at(self, t: float) -> Section | None:
        for s in self.sections:
            if s.t0 <= t < s.t1:
                return s
        return self.sections[-1] if self.sections and t >= self.sections[-1].t1 else None

    def phrase_boundaries(self, bars: int = 4) -> list[float]:
        """Downbeat times every `bars` bars — the only musically safe cut points (§14.2)."""
        return self.grid.downbeats[::bars]


def mel_features(x: np.ndarray, sr: int, hop_s: float = 0.05, n_bands: int = 24) -> np.ndarray:
    """Log-energy in log-spaced frequency bands per frame — a cheap timbre descriptor."""
    hop = max(1, int(hop_s * sr))
    win = hop * 4
    n = max(0, (len(x) - win) // hop)
    if n < 2:
        return np.zeros((0, n_bands), dtype=np.float32)
    window = np.hanning(win).astype(np.float32)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * window, axis=1))
    freqs = np.fft.rfftfreq(win, 1 / sr)
    edges = np.geomspace(60.0, min(sr / 2 - 1, 7000.0), n_bands + 1)
    out = np.zeros((n, n_bands), dtype=np.float32)
    for b in range(n_bands):
        band = (freqs >= edges[b]) & (freqs < edges[b + 1])
        if band.any():
            out[:, b] = np.log1p(spec[:, band].mean(axis=1) * 100)
    # Per-frame normalization makes the comparison about timbre, not loudness.
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norm, 1e-6)


def novelty_curve(features: np.ndarray, kernel_bars_frames: int = 16) -> np.ndarray:
    """Checkerboard-kernel novelty over the self-similarity matrix."""
    n = len(features)
    if n < 2 * kernel_bars_frames + 1:
        return np.zeros(n)
    sim = features @ features.T
    k = kernel_bars_frames
    novelty = np.zeros(n)
    for i in range(k, n - k):
        past = sim[i - k : i, i - k : i].mean()
        future = sim[i : i + k, i : i + k].mean()
        across = sim[i - k : i, i : i + k].mean()
        # High self-similarity on each side, low across the boundary → a change.
        novelty[i] = (past + future) / 2 - across
    return novelty


def energy_curve(x: np.ndarray, sr: int, hop_s: float = 1.0) -> list[float]:
    """Per-second RMS energy, normalized to 0–1 across the track."""
    hop = max(1, int(hop_s * sr))
    n = len(x) // hop
    if n == 0:
        return []
    blocks = x[: n * hop].reshape(n, hop)
    rms = np.sqrt((blocks.astype(np.float64) ** 2).mean(axis=1))
    if rms.max() <= 0:
        return [0.0] * n
    return [round(float(v), 4) for v in rms / rms.max()]


def _assign_roles(sections: list[Section]) -> None:
    """Label sections by position and relative energy.

    Position matters as much as energy: the first section is the intro and the last
    is the outro regardless of level, and the loudest middle section is the drop.
    """
    if not sections:
        return
    energies = [s.energy for s in sections]
    peak = int(np.argmax(energies))
    for i, s in enumerate(sections):
        if i == 0:
            s.role = "intro"
        elif i == len(sections) - 1:
            s.role = "outro"
        elif i == peak:
            s.role = "drop"
        elif s.energy < np.median(energies) * 0.8:
            s.role = "break"
        else:
            s.role = "build"


def analyze_structure(wav, grid: BeatGrid | None = None, min_section_s: float = 8.0,
                      phrase_bars: int = 4, novelty_z: float = 4.0) -> MusicStructure:
    """Sections snapped to phrase boundaries, plus the energy curve."""
    x, sr = decode_audio_mono(wav)
    grid = grid or analyze_beats(wav)
    duration = len(x) / sr if sr else 0.0
    if duration <= 0:
        return MusicStructure(grid=grid, duration_s=0.0)

    hop_s = 0.05
    feats = mel_features(x, sr, hop_s=hop_s)
    novelty = novelty_curve(feats)
    curve = energy_curve(x, sr)

    # Candidate boundaries: phrase starts (every `phrase_bars` bars). A section
    # change anywhere else would put the following cuts off the bar line (§14.2).
    candidates = [t for t in grid.downbeats[::phrase_bars] if 0 < t < duration - min_section_s]
    scored: list[tuple[float, float]] = []
    for t in candidates:
        i = int(t / hop_s)
        if 0 <= i < len(novelty):
            scored.append((float(novelty[i]), t))

    # Only accept boundaries whose novelty genuinely stands out. Taking every
    # candidate in rank order splits homogeneous sections at arbitrary phrase
    # boundaries (4 true sections became 6 on the fixture), but a mean+sigma
    # threshold fails the other way: novelty spans orders of magnitude, so one
    # strong boundary inflates the standard deviation enough to hide the others.
    # A median + MAD threshold is insensitive to exactly those outliers — the same
    # robust statistic the shot detector uses.
    boundaries: list[float] = []
    if scored:
        values = np.array([v for v, _ in scored])
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med))) or 1e-9
        threshold = med + novelty_z * 1.4826 * mad
        for value, t in sorted(scored, reverse=True):
            if value < threshold:
                break
            if all(abs(t - b) >= min_section_s for b in boundaries):
                boundaries.append(t)
    boundaries = sorted(boundaries)

    edges = [0.0, *boundaries, duration]
    sections: list[Section] = []
    for i, (a, b) in enumerate(zip(edges, edges[1:], strict=False)):
        if b - a < min_section_s / 2:
            continue
        lo, hi = int(a), max(int(a) + 1, int(b))
        seg = curve[lo:hi] or [0.0]
        sections.append(Section(
            id=f"sec{i:02d}", t0=round(a, 3), t1=round(b, 3),
            energy=round(float(np.mean(seg)), 4),
            bars=int(round((b - a) / (grid.beat_period_s * grid.beats_per_bar))),
        ))
    _assign_roles(sections)
    return MusicStructure(grid=grid, sections=sections, energy_curve=curve, duration_s=duration)


def transient_density(x: np.ndarray, sr: int, hop_s: float = 1.0) -> list[float]:
    """Per-second count of amplitude transients — a percussive-activity proxy."""
    amp, rate = transient_envelope(x, sr)
    if len(amp) == 0:
        return []
    thresh = float(np.percentile(amp, 90))
    per = max(1, int(hop_s * rate))
    n = len(amp) // per
    return [float((amp[i * per : (i + 1) * per] > thresh).sum()) for i in range(n)]
