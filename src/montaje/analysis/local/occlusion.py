"""Occlusion analyzer: dark + low-detail spans and reveal/cover edges (§9).

Emits meaning-free events. Whether a `reveal` means "hand uncovers the lens"
is discovered later by pattern mining + the agent (§11–12) — never coded here.
"""

from __future__ import annotations

from montaje.analysis.base import decode_gray_frames, detail_energy, spans_from_mask
from montaje.models.asset import Asset, AssetKind
from montaje.models.events import Event


class OcclusionAnalyzer:
    name = "occlusion"
    version = 1

    def __init__(self, fps: float = 6.0, luma_max: float = 40.0, detail_max: float = 6.0,
                 min_len_s: float = 0.25):
        self.fps = fps
        self.luma_max = luma_max  # 0–255 mean luma below which a frame counts as dark
        self.detail_max = detail_max  # gradient energy below which a frame counts as featureless
        self.min_len_s = min_len_s

    def params(self) -> dict:
        return {"fps": self.fps, "luma_max": self.luma_max, "detail_max": self.detail_max,
                "min_len_s": self.min_len_s}

    def run(self, asset: Asset, ws) -> list[Event]:
        if asset.kind == AssetKind.PHOTO:
            return []
        frames, rate = decode_gray_frames(ws.proxy_path(asset.asset_id), self.fps)
        if len(frames) == 0:
            return []
        luma = frames.mean(axis=(1, 2))
        detail = detail_energy(frames)
        occluded = (luma < self.luma_max) & (detail < self.detail_max)

        analyzer = f"{self.name}@{self.version}"
        events: list[Event] = []
        duration = len(frames) / rate
        for t0, t1 in spans_from_mask(occluded, rate, self.min_len_s):
            i0, i1 = int(t0 * rate), max(int(t0 * rate) + 1, int(t1 * rate))
            data = {
                "mean_luma": round(float(luma[i0:i1].mean()) / 255.0, 3),
                "detail": round(float(detail[i0:i1].mean()), 2),
            }
            events.append(Event(asset_id=asset.asset_id, analyzer=analyzer, type="occluded",
                                t0=t0, t1=t1, score=1.0, data=data))
            # Edge events: a cover starts an occlusion mid-clip; a reveal ends one.
            if t0 > 0.05:
                events.append(Event(asset_id=asset.asset_id, analyzer=analyzer, type="cover",
                                    t0=max(0.0, t0 - 0.2), t1=t0 + 0.2, score=0.9, data=data))
            if t1 < duration - 0.05:
                events.append(Event(asset_id=asset.asset_id, analyzer=analyzer, type="reveal",
                                    t0=max(0.0, t1 - 0.2), t1=t1 + 0.2, score=0.9, data=data))
        return events
