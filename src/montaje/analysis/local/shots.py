"""Shot boundary detection (§9).

M1 uses a content-difference detector on the proxy (histogram + luma deltas with
an adaptive threshold): no model download, deterministic, good enough to give the
rails snap targets. TransNetV2 slots in behind the same interface later.
"""

from __future__ import annotations

import numpy as np

from montaje.analysis.base import decode_gray_frames
from montaje.models.asset import Asset, AssetKind
from montaje.models.events import Event


class ShotsAnalyzer:
    name = "shots"
    version = 1

    def __init__(self, fps: float = 8.0, threshold_sigma: float = 4.0, min_shot_s: float = 0.4):
        self.fps = fps
        self.threshold_sigma = threshold_sigma
        self.min_shot_s = min_shot_s

    def params(self) -> dict:
        return {"fps": self.fps, "threshold_sigma": self.threshold_sigma, "min_shot_s": self.min_shot_s}

    def run(self, asset: Asset, ws) -> list[Event]:
        if asset.kind == AssetKind.PHOTO:
            return []
        frames, rate = decode_gray_frames(ws.proxy_path(asset.asset_id), self.fps)
        if len(frames) < 3:
            return [Event(asset_id=asset.asset_id, analyzer=f"{self.name}@{self.version}",
                          type="shot", t0=0.0, t1=asset.duration_s, score=1.0, data={})]

        f32 = frames.astype(np.float32)
        # Histogram correlation distance is robust to global motion; frame diff catches hard cuts.
        hists = np.stack([np.histogram(f, bins=32, range=(0, 255))[0] for f in frames]).astype(np.float32)
        hists /= hists.sum(axis=1, keepdims=True) + 1e-9
        hist_d = np.abs(np.diff(hists, axis=0)).sum(axis=1)
        frame_d = np.abs(np.diff(f32, axis=0)).mean(axis=(1, 2)) / 255.0
        signal = hist_d + frame_d

        med = float(np.median(signal))
        mad = float(np.median(np.abs(signal - med))) or 1e-6
        thresh = med + self.threshold_sigma * 1.4826 * mad

        cuts = [0.0]
        for i, v in enumerate(signal):
            t = (i + 1) / rate
            if v > thresh and t - cuts[-1] >= self.min_shot_s:
                cuts.append(t)

        analyzer = f"{self.name}@{self.version}"
        duration = asset.duration_s
        bounds = cuts + [duration]
        events = []
        for a, b in zip(bounds, bounds[1:], strict=False):
            if b - a <= 0:
                continue
            events.append(Event(asset_id=asset.asset_id, analyzer=analyzer, type="shot",
                                t0=round(a, 3), t1=round(min(b, duration), 3), score=1.0, data={}))
        return events
