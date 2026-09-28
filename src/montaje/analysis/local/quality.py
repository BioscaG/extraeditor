"""Quality analyzer: per-second blur, exposure clipping, shake, `usable` spans (§9)."""

from __future__ import annotations

import numpy as np

from montaje.analysis.base import decode_gray_frames, detail_energy, spans_from_mask
from montaje.models.asset import Asset, AssetKind
from montaje.models.events import Event


class QualityAnalyzer:
    name = "quality"
    version = 1

    def __init__(self, fps: float = 4.0, blur_min_detail: float = 4.0,
                 clip_frac_max: float = 0.25, shake_max: float = 18.0):
        self.fps = fps
        self.blur_min_detail = blur_min_detail
        self.clip_frac_max = clip_frac_max
        self.shake_max = shake_max

    def params(self) -> dict:
        return {"fps": self.fps, "blur_min_detail": self.blur_min_detail,
                "clip_frac_max": self.clip_frac_max, "shake_max": self.shake_max}

    def run(self, asset: Asset, ws) -> list[Event]:
        if asset.kind == AssetKind.PHOTO:
            return []
        frames, rate = decode_gray_frames(ws.proxy_path(asset.asset_id), self.fps)
        if len(frames) < 2:
            return []
        f32 = frames.astype(np.float32)
        detail = detail_energy(frames)
        clip_hi = (frames >= 250).mean(axis=(1, 2))
        clip_lo = (frames <= 5).mean(axis=(1, 2))
        # Shake proxy: mean absolute frame difference; fast global shifts dominate it.
        motion = np.abs(np.diff(f32, axis=0)).mean(axis=(1, 2))
        motion = np.concatenate([[motion[0]], motion])

        analyzer = f"{self.name}@{self.version}"
        events: list[Event] = []
        # Per-second metrics
        per_sec = max(1, int(rate))
        n_secs = int(np.ceil(len(frames) / per_sec))
        ok = np.zeros(len(frames), dtype=bool)
        for s in range(n_secs):
            sl = slice(s * per_sec, (s + 1) * per_sec)
            d, ch, cl, m = (float(detail[sl].mean()), float(clip_hi[sl].mean()),
                            float(clip_lo[sl].mean()), float(motion[sl].mean()))
            events.append(Event(
                asset_id=asset.asset_id, analyzer=analyzer, type="metrics",
                t0=float(s), t1=float(min(s + 1, len(frames) / rate)),
                score=1.0,
                data={"detail": round(d, 2), "clip_hi": round(ch, 3),
                      "clip_lo": round(cl, 3), "shake": round(m, 2)},
            ))
            usable = d >= self.blur_min_detail and ch <= self.clip_frac_max and m <= self.shake_max
            ok[sl] = usable
        for t0, t1 in spans_from_mask(ok, rate, min_len_s=0.5):
            events.append(Event(asset_id=asset.asset_id, analyzer=analyzer, type="usable",
                                t0=t0, t1=t1, score=1.0, data={}))
        return events
