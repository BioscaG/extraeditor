"""Audio event tagging without a model download (§9).

M1 classifies windows as silence / speech / music / crowd / other from cheap
spectral features. CLAP replaces this behind the same interface (DECISIONS.md);
downstream code only ever consumes the label spans, not the classifier.

The features that actually separate these classes on real footage:

- **level** separates silence from everything else.
- **syllabic modulation** (envelope std / mean, 2–8 Hz) is what separates speech
  from crowd noise. Both are broadband with high spectral flatness — measured on
  the festival fixtures, speech sits at flatness 0.33–0.58, so a "speech is
  tonal" rule would reject it. Speech pulses at syllable rate; crowd noise and
  sustained music do not.
- **low-frequency share** plus low flatness marks music (bass + tonal content).
- **spectral centroid** separates bright broadband crowd noise from the rest.
"""

from __future__ import annotations

import numpy as np

from montaje.analysis.base import decode_audio_mono
from montaje.models.asset import Asset
from montaje.models.events import Event

SILENCE_DBFS = -45.0
SPEECH_BAND_HZ = (300.0, 3400.0)
SPEECH_BAND_MIN = 0.45  # share of energy in the speech band
SPEECH_MODULATION_MIN = 0.7  # envelope std/mean; crowd noise stays well below this
MUSIC_LOW_MIN = 0.30  # share of energy below 250 Hz
MUSIC_FLATNESS_MAX = 0.25
CROWD_FLATNESS_MIN = 0.25
CROWD_CENTROID_MIN = 1200.0


def window_features(block: np.ndarray, sr: int) -> dict[str, float]:
    """Spectral and envelope features for one analysis window."""
    rms = float(np.sqrt(np.mean(block**2)) + 1e-12)
    spec = np.abs(np.fft.rfft(block * np.hanning(len(block))))
    freqs = np.fft.rfftfreq(len(block), 1 / sr)
    total = float(spec.sum()) + 1e-9
    lo, hi = SPEECH_BAND_HZ
    # 10 ms envelope hops resolve the 2–8 Hz syllabic rate.
    hop = max(1, sr // 100)
    env = np.abs(block[: len(block) // hop * hop]).reshape(-1, hop).mean(axis=1)
    return {
        "dbfs": 20 * float(np.log10(rms)),
        "centroid": float((spec * freqs).sum() / total),
        "flatness": float(np.exp(np.mean(np.log(spec + 1e-9))) / (np.mean(spec) + 1e-9)),
        "speech_ratio": float(spec[(freqs >= lo) & (freqs <= hi)].sum() / total),
        "low_ratio": float(spec[freqs < 250].sum() / total),
        "modulation": float(np.std(env) / (np.mean(env) + 1e-9)) if len(env) else 0.0,
    }


def classify_window(f: dict[str, float]) -> tuple[str, float]:
    """Map features to (label, confidence). Order encodes precedence."""
    if f["dbfs"] < SILENCE_DBFS:
        return "silence", 1.0
    if f["speech_ratio"] >= SPEECH_BAND_MIN and f["modulation"] >= SPEECH_MODULATION_MIN:
        # Confidence grows with how far past both gates the window sits.
        margin = (f["speech_ratio"] - SPEECH_BAND_MIN) + (f["modulation"] - SPEECH_MODULATION_MIN) / 2
        return "speech", float(min(1.0, 0.6 + margin))
    if f["low_ratio"] >= MUSIC_LOW_MIN and f["flatness"] <= MUSIC_FLATNESS_MAX:
        return "music", float(min(1.0, 0.5 + f["low_ratio"]))
    if f["flatness"] >= CROWD_FLATNESS_MIN and f["centroid"] >= CROWD_CENTROID_MIN:
        return "crowd", float(min(1.0, 0.4 + f["flatness"]))
    return "other", 0.4


class AudioEventsAnalyzer:
    name = "audio_events"
    version = 1

    def __init__(self, window_s: float = 1.0):
        self.window_s = window_s

    def params(self) -> dict:
        return {
            "window_s": self.window_s,
            "silence_dbfs": SILENCE_DBFS,
            "speech_band_min": SPEECH_BAND_MIN,
            "speech_modulation_min": SPEECH_MODULATION_MIN,
            "music_low_min": MUSIC_LOW_MIN,
        }

    def run(self, asset: Asset, ws) -> list[Event]:
        wav = ws.audio_16k_path(asset.asset_id)
        if not wav.exists():
            return []
        x, sr = decode_audio_mono(wav)
        win = int(self.window_s * sr)
        if win == 0 or len(x) < win // 2:
            return []

        analyzer = f"{self.name}@{self.version}"
        events: list[Event] = []
        for i in range(max(1, len(x) // win)):
            block = x[i * win : (i + 1) * win]
            if len(block) < win // 2:
                break
            f = window_features(block, sr)
            label, score = classify_window(f)
            t0 = i * self.window_s
            events.append(Event(
                asset_id=asset.asset_id, analyzer=analyzer, type=label,
                t0=t0, t1=t0 + self.window_s, score=round(score, 3),
                data={k: round(v, 3) for k, v in f.items()},
            ))
        return events
