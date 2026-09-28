"""Per-shot color statistics for normalization and matching (§9, §15)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from montaje import ffmpeg
from montaje.analysis.base import proxy_height
from montaje.index.store import Store
from montaje.models.asset import Asset, AssetKind
from montaje.models.events import Event


def _decode_rgb(video: Path, fps: float, width: int = 64) -> np.ndarray:
    height = proxy_height(video, width)
    out = ffmpeg.run(["-v", "error", "-i", str(video),
                      "-vf", f"fps={fps},scale={width}:{height}",
                      "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
    buf = np.frombuffer(out.stdout, dtype=np.uint8)
    frame_bytes = width * height * 3
    if frame_bytes == 0 or len(buf) < frame_bytes:
        return np.zeros((0, max(1, height), width, 3), dtype=np.uint8)
    n = len(buf) // frame_bytes
    return buf[: n * frame_bytes].reshape(n, height, width, 3)


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb float in [0,1], shape (..., 3) → CIELAB (D65)."""
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]], dtype=np.float32)
    xyz = lin @ m.T
    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    t = xyz / white
    eps = 216 / 24389
    kappa = 24389 / 27
    f = np.where(t > eps, np.cbrt(t), (kappa * t + 16) / 116)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    """Coarse skin-tone mask in normalized RGB; protects faces from gray-world shifts."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    s = r + g + b + 1e-6
    rn, gn = r / s, g / s
    return (rn > 0.35) & (rn < 0.47) & (gn > 0.28) & (gn < 0.36) & (r > g) & (g > b)


class ColorStatsAnalyzer:
    name = "color_stats"
    version = 1

    def __init__(self, fps: float = 2.0):
        self.fps = fps

    def params(self) -> dict:
        return {"fps": self.fps}

    def run(self, asset: Asset, ws) -> list[Event]:
        if asset.kind == AssetKind.PHOTO:
            return []
        frames = _decode_rgb(ws.proxy_path(asset.asset_id), self.fps)
        if len(frames) == 0:
            return []
        with Store(ws.db_path) as store:
            shots = store.get_events(asset.asset_id, "shots", "shot")
        if not shots:
            shots = [Event(asset_id=asset.asset_id, analyzer="shots@1", type="shot",
                           t0=0.0, t1=asset.duration_s)]

        analyzer = f"{self.name}@{self.version}"
        events: list[Event] = []
        for idx, shot in enumerate(shots):
            i0 = int(shot.t0 * self.fps)
            i1 = max(i0 + 1, int(shot.t1 * self.fps))
            chunk = frames[i0:i1]
            if len(chunk) == 0:
                continue
            rgb = chunk.astype(np.float32) / 255.0
            lab = _srgb_to_lab(rgb)
            skin = _skin_mask(rgb)
            mean_rgb = rgb.reshape(-1, 3).mean(axis=0)
            # Gray-world white balance estimate: per-channel gains toward the mean.
            gains = float(mean_rgb.mean()) / np.maximum(mean_rgb, 1e-4)
            events.append(Event(
                asset_id=asset.asset_id, analyzer=analyzer, type="shot_color",
                t0=shot.t0, t1=shot.t1, score=1.0,
                data={
                    "shot_index": idx,
                    "lab_mean": [round(float(v), 2) for v in lab.reshape(-1, 3).mean(axis=0)],
                    "lab_p05": [round(float(v), 2) for v in np.percentile(lab.reshape(-1, 3), 5, axis=0)],
                    "lab_p95": [round(float(v), 2) for v in np.percentile(lab.reshape(-1, 3), 95, axis=0)],
                    "lab_std": [round(float(v), 2) for v in lab.reshape(-1, 3).std(axis=0)],
                    "wb_gains": [round(float(v), 4) for v in gains],
                    "exposure": round(float(lab[..., 0].mean() / 100.0), 4),
                    "skin_fraction": round(float(skin.mean()), 4),
                },
            ))
        return events
