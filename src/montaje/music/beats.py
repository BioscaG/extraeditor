"""Beat, downbeat and tempo tracking (§9, §14.2).

The whole edit is snapped to this grid, so accuracy here bounds the quality of
everything downstream. The pipeline is:

1. **Onset envelope** — spectral flux at a 5 ms hop.
2. **Tempo** — autocorrelation of the envelope, with parabolic interpolation of
   the peak so the estimate is not quantized to whole envelope frames.
3. **Phase** — a rigid comb of beats scanned across one beat period, taking the
   phase with the most onset energy.
4. **Offset refinement** — one global correction, the median distance from each
   grid beat to the nearest amplitude transient. Measured on amplitude rather than
   spectral flux, and applied globally rather than per beat. This takes the grid
   from ~14 ms to sub-millisecond accuracy on click-track fixtures.
5. **Downbeats** — the beat-within-bar phase whose onsets are strongest.

Beat This! / madmom plug in behind `BeatGrid` (DECISIONS.md); they are known to be
awkward to install on Apple Silicon (§28), so this deterministic path exists first.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from montaje.analysis.base import decode_audio_mono

HOP_S = 0.005
BPM_RANGE = (60.0, 190.0)
# A beat may be corrected by at most this fraction of a period, so refinement
# cannot silently re-phase the grid onto the off-beat.
MAX_REFINE_FRACTION = 0.12


class BeatGrid(BaseModel):
    """Musical time grid for one track."""

    bpm: float
    beats: list[float] = Field(default_factory=list)
    downbeats: list[float] = Field(default_factory=list)
    beats_per_bar: int = 4
    duration_s: float = 0.0
    confidence: float = 0.0

    @property
    def beat_period_s(self) -> float:
        return 60.0 / self.bpm

    def nearest_beat(self, t: float) -> float:
        if not self.beats:
            return t
        return min(self.beats, key=lambda b: abs(b - t))

    def nearest_downbeat(self, t: float) -> float:
        if not self.downbeats:
            return self.nearest_beat(t)
        return min(self.downbeats, key=lambda b: abs(b - t))

    def beats_to_seconds(self, n_beats: float) -> float:
        return n_beats * self.beat_period_s

    def bar_starts(self) -> list[float]:
        return list(self.downbeats)


def onset_envelope(x: np.ndarray, sr: int, hop_s: float = HOP_S) -> tuple[np.ndarray, np.ndarray]:
    """Spectral flux: half-wave-rectified positive change in log-magnitude spectrum.

    Returns `(env, times)`. The time axis is returned explicitly rather than left
    implicit in the index, because both the analysis window and the `diff` shift
    the envelope relative to the audio; callers that reconstructed the offset
    themselves got the beat phase wrong by a fixed frame.
    """
    hop = max(1, int(hop_s * sr))
    win = hop * 8
    n = max(0, (len(x) - win) // hop)
    if n < 2:
        return np.zeros(0, dtype=np.float32), np.zeros(0)
    window = np.hanning(win).astype(np.float32)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx] * window
    spec = np.abs(np.fft.rfft(frames, axis=1))
    logspec = np.log1p(spec * 100.0)
    env = np.maximum(np.diff(logspec, axis=0), 0).sum(axis=1)
    env = env - env.mean()
    std = env.std()
    if std > 0:
        env = env / std
    # env[i] is the rise from frame i to frame i+1; attribute it to frame i+1's centre.
    times = ((np.arange(len(env)) + 1) * hop + win / 2) / sr
    return env.astype(np.float32), times


def _parabolic_peak(y: np.ndarray, i: int) -> float:
    """Sub-sample peak position near index `i` by fitting a parabola to 3 points."""
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    a, b, c = float(y[i - 1]), float(y[i]), float(y[i + 1])
    denom = a - 2 * b + c
    if abs(denom) < 1e-12:
        return float(i)
    return i + 0.5 * (a - c) / denom


def estimate_tempo(
    env: np.ndarray, times: np.ndarray, bpm_range: tuple[float, float] = BPM_RANGE
) -> tuple[float, float]:
    """Tempo from the envelope autocorrelation. Returns (bpm, confidence)."""
    if len(env) < 8 or len(times) < 2:
        return 120.0, 0.0
    rate = 1.0 / float(np.median(np.diff(times)))
    if len(env) < int(rate * 2):
        return 120.0, 0.0
    ac = np.correlate(env, env, mode="full")[len(env) - 1 :]
    ac[0] = 0.0
    lo = max(1, int(rate * 60.0 / bpm_range[1]))
    hi = min(len(ac) - 1, int(rate * 60.0 / bpm_range[0]))
    if hi <= lo + 1:
        return 120.0, 0.0
    window = ac[lo:hi]
    peak = int(np.argmax(window)) + lo
    # Interpolate: whole-frame lags quantize the tempo (at a 10 ms hop, 90 BPM
    # could only be reported as 89.55), and the error compounds over a long track.
    lag = _parabolic_peak(ac, peak)
    bpm = 60.0 * rate / lag if lag > 0 else 120.0
    conf = float((window.max() - window.mean()) / (window.std() + 1e-9))
    return float(bpm), float(min(1.0, conf / 6.0))


def fold_tempo(bpm: float, lo: float = 70.0, hi: float = 180.0) -> float:
    """Octave-correct a tempo into a musically plausible range."""
    if bpm <= 0:
        return 120.0
    while bpm < lo:
        bpm *= 2
    while bpm > hi:
        bpm /= 2
    return bpm


def _env_at(env: np.ndarray, times: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Onset strength at arbitrary times, sampled at the nearest envelope frame."""
    right = np.searchsorted(times, t)
    left = np.clip(right - 1, 0, len(env) - 1)
    right = np.clip(right, 0, len(env) - 1)
    pick_left = np.abs(times[left] - t) <= np.abs(times[right] - t)
    return env[np.where(pick_left, left, right)]


def transient_envelope(x: np.ndarray, sr: int, hop_s: float = 0.001) -> tuple[np.ndarray, float]:
    """Rectified amplitude at ~1 ms resolution, for locating transients precisely."""
    hop = max(1, int(hop_s * sr))
    n = len(x) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32), 1.0 / hop_s
    return np.abs(x[: n * hop]).reshape(n, hop).max(axis=1), sr / hop


def attack_envelope(x: np.ndarray, sr: int, hop_s: float = 0.001) -> tuple[np.ndarray, float]:
    """Positive rise of the 1 ms amplitude envelope — peaks at the attack itself.

    Raw amplitude peaks a little *after* the attack (a quarter cycle of the carrier,
    ~4 ms for a 60 Hz kick), and the spectral flux peaks ~14 ms *before* it (its
    analysis window's group delay). The rise of the amplitude envelope has neither
    bias, which is what makes sub-millisecond grid fitting possible.
    """
    amp, rate = transient_envelope(x, sr, hop_s)
    if len(amp) < 2:
        return np.zeros(0, dtype=np.float32), rate
    return np.maximum(np.diff(amp), 0).astype(np.float32), rate


def _measure_beats(x: np.ndarray, sr: int, beats: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """For each grid beat, the nearest measured attack time. Returns (indices, times)."""
    rise, rate = attack_envelope(x, sr)
    if len(rise) == 0 or rise.max() <= 0:
        return np.zeros(0, dtype=int), np.zeros(0)
    tol = MAX_REFINE_FRACTION * period
    idx: list[int] = []
    measured: list[float] = []
    for i, b in enumerate(beats):
        lo = max(0, int((b - tol) * rate))
        hi = min(len(rise), int((b + tol) * rate) + 1)
        if hi <= lo:
            continue
        seg = rise[lo:hi]
        if seg.max() <= 0:
            continue
        idx.append(i)
        # _parabolic_peak returns an absolute index into `rise`.
        measured.append(_parabolic_peak(rise, lo + int(np.argmax(seg))) / rate)
    return np.array(idx, dtype=int), np.array(measured)


def refine_grid(x: np.ndarray, sr: int, phase: float, period: float, n_beats: int,
                iterations: int = 2) -> tuple[float, float]:
    """Least-squares fit of `phase + period * i` to the measured attack times.

    Fitting the *period* as well as the phase is what keeps a long track in sync.
    A tempo estimate good to 0.08% (120.094 vs 120 BPM) still drifts 75 ms over 96
    seconds, which a single global offset correction cannot fix — it just splits the
    error either side of the middle. Re-running the fit lets beats that were
    initially outside the search window be picked up once the grid improves.
    """
    for _ in range(iterations):
        grid = phase + period * np.arange(n_beats)
        idx, measured = _measure_beats(x, sr, grid, period)
        if len(idx) < 3:
            return phase, period
        # Reject outliers so a missing or ghost transient cannot tilt the fit.
        residual = measured - (phase + period * idx)
        keep = np.abs(residual - np.median(residual)) <= 2.5 * (np.std(residual) + 1e-6)
        if keep.sum() < 3:
            return phase, period
        new_period, new_phase = np.polyfit(idx[keep], measured[keep], 1)
        # Guard against the fit collapsing onto a different metrical level.
        if not (0.8 * period <= new_period <= 1.25 * period):
            return phase, period
        phase, period = float(new_phase), float(new_period)
    return phase, period


def track_beats(env: np.ndarray, times: np.ndarray, bpm: float, phase_steps: int = 64,
                x: np.ndarray | None = None, sr: int = 16000) -> tuple[list[float], float]:
    """Rigid beat grid fitted to the audio. Returns `(beats, refined_bpm)`.

    The coarse phase comes from scanning a comb across one beat period against the
    onset envelope; `refine_grid` then fits both phase and period to the measured
    attacks. The grid is anchored so a track whose first beat is at 0.0 is
    represented exactly rather than starting one beat late.
    """
    if len(env) == 0 or bpm <= 0:
        return [], bpm
    period = 60.0 / bpm
    end = float(times[-1])
    n_beats = int(end / period) + 1
    if n_beats < 1:
        return [], bpm
    offsets = period * np.arange(n_beats)

    best_phase, best_score = 0.0, -np.inf
    for step in range(phase_steps):
        phase = step * period / phase_steps
        score = float(_env_at(env, times, phase + offsets).sum())
        if score > best_score:
            best_phase, best_score = phase, score
    # A phase just short of a full period is really a small negative offset; keep
    # the representation that puts a beat near t=0.
    if period - best_phase < best_phase:
        best_phase -= period

    if x is not None:
        best_phase, period = refine_grid(x, sr, best_phase, period, n_beats + 1)
        n_beats = int((end - best_phase) / period) + 1

    beats = [best_phase + period * i for i in range(n_beats + 1)]
    # Keep a beat that lands marginally before zero: it is the same beat as t=0
    # within the envelope's own resolution.
    tol = period / 2
    kept = [float(max(0.0, b)) for b in beats if -tol <= b <= end + tol]
    return kept, 60.0 / period


def beat_strengths(x: np.ndarray, sr: int, beats: list[float], period: float,
                   window_fraction: float = 0.25) -> np.ndarray:
    """Peak amplitude near each beat — how hard that beat is hit.

    Used for downbeat detection and for the energy curve in music structure.
    Amplitude rather than spectral flux: the flux peaks ~14 ms before the transient,
    so sampling it at the (now sub-millisecond accurate) beat time lands past its
    peak and reports every beat as weak.
    """
    if not beats:
        return np.zeros(0)
    amp, rate = transient_envelope(x, sr)
    if len(amp) == 0:
        return np.zeros(len(beats))
    half = window_fraction * period
    out = np.zeros(len(beats))
    for i, b in enumerate(beats):
        lo = max(0, int((b - half) * rate))
        hi = min(len(amp), int((b + half) * rate) + 1)
        if hi > lo:
            out[i] = float(amp[lo:hi].max())
    return out


def find_downbeats(x: np.ndarray, sr: int, beats: list[float], bpm: float,
                   beats_per_bar: int = 4) -> list[float]:
    """Pick the beat-within-bar offset that is hit hardest, then take every Nth beat."""
    if not beats:
        return []
    strengths = beat_strengths(x, sr, beats, 60.0 / bpm)
    scores = [
        float(strengths[offset::beats_per_bar].mean()) if len(strengths[offset::beats_per_bar]) else -np.inf
        for offset in range(beats_per_bar)
    ]
    return beats[int(np.argmax(scores)) :: beats_per_bar]


def analyze_beats(wav: object, beats_per_bar: int = 4) -> BeatGrid:
    """Full beat analysis of an audio file."""
    x, sr = decode_audio_mono(wav)  # 16 kHz mono is plenty for onset detection
    if len(x) == 0:
        return BeatGrid(bpm=120.0, duration_s=0.0)
    env, times = onset_envelope(x, sr)
    raw_bpm, conf = estimate_tempo(env, times)
    beats, bpm = track_beats(env, times, fold_tempo(raw_bpm), x=x, sr=sr)
    downbeats = find_downbeats(x, sr, beats, bpm, beats_per_bar)
    return BeatGrid(
        bpm=round(bpm, 3),
        beats=[round(b, 4) for b in beats],
        downbeats=[round(b, 4) for b in downbeats],
        beats_per_bar=beats_per_bar,
        duration_s=len(x) / sr,
        confidence=round(conf, 3),
    )
