"""Voice activity detection (§9).

Speech segments gate ASR, because Whisper hallucinates confidently on music and
silence (§28). This uses the same features as `audio_events` on short windows,
with hysteresis: a lower "continue" threshold than "start", so a brief pause
inside a sentence does not split it into two segments.

Silero VAD replaces the inner detector behind the same event contract.
"""

from __future__ import annotations

import numpy as np

from montaje.analysis.base import decode_audio_mono
from montaje.analysis.local.audio_events import SILENCE_DBFS, window_features
from montaje.models.asset import Asset
from montaje.models.events import Event


class VadAnalyzer:
    name = "vad"
    version = 1

    def __init__(self, window_s: float = 0.2, start_band_min: float = 0.45,
                 continue_band_min: float = 0.30, min_speech_s: float = 0.4,
                 merge_gap_s: float = 0.35):
        self.window_s = window_s
        self.start_band_min = start_band_min
        self.continue_band_min = continue_band_min
        self.min_speech_s = min_speech_s
        self.merge_gap_s = merge_gap_s

    def params(self) -> dict:
        return {"window_s": self.window_s, "start_band_min": self.start_band_min,
                "continue_band_min": self.continue_band_min,
                "min_speech_s": self.min_speech_s, "merge_gap_s": self.merge_gap_s}

    def run(self, asset: Asset, ws) -> list[Event]:
        wav = ws.audio_16k_path(asset.asset_id)
        if not wav.exists():
            return []
        x, sr = decode_audio_mono(wav)
        win = int(self.window_s * sr)
        if win == 0 or len(x) < win:
            return []

        n = len(x) // win
        band = np.zeros(n)
        loud = np.zeros(n, dtype=bool)
        for i in range(n):
            f = window_features(x[i * win : (i + 1) * win], sr)
            band[i] = f["speech_ratio"]
            loud[i] = f["dbfs"] >= SILENCE_DBFS

        # Hysteresis: open on start_band_min, stay open down to continue_band_min.
        active = np.zeros(n, dtype=bool)
        on = False
        for i in range(n):
            if not loud[i]:
                on = False
            elif on:
                on = band[i] >= self.continue_band_min
            else:
                on = band[i] >= self.start_band_min
            active[i] = on

        segments: list[list[float]] = []
        for i, v in enumerate(active):
            if not v:
                continue
            t0, t1 = i * self.window_s, (i + 1) * self.window_s
            if segments and t0 - segments[-1][1] <= self.merge_gap_s:
                segments[-1][1] = t1
            else:
                segments.append([t0, t1])

        analyzer = f"{self.name}@{self.version}"
        return [
            Event(asset_id=asset.asset_id, analyzer=analyzer, type="speech",
                  t0=round(a, 3), t1=round(b, 3), score=1.0, data={})
            for a, b in segments
            if b - a >= self.min_speech_s
        ]
