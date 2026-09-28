"""Motion analyzer: camera motion magnitude/direction and subject energy (§9).

Direction feeds motion-matched transitions: a whip pan left should follow a shot
already moving left.
"""

from __future__ import annotations

import numpy as np

from montaje.analysis.base import decode_gray_frames
from montaje.models.asset import Asset, AssetKind
from montaje.models.events import Event


def _dominant_shift(a: np.ndarray, b: np.ndarray, max_shift: int = 8) -> tuple[int, int, float]:
    """Brute-force integer translation search (phase-correlation lite) on tiny frames."""
    best = (0, 0, float("inf"))
    h, w = a.shape
    for dy in range(-max_shift, max_shift + 1, 2):
        for dx in range(-max_shift, max_shift + 1, 2):
            ys = slice(max(0, dy), h + min(0, dy))
            xs = slice(max(0, dx), w + min(0, dx))
            ys2 = slice(max(0, -dy), h + min(0, -dy))
            xs2 = slice(max(0, -dx), w + min(0, -dx))
            err = float(np.abs(a[ys, xs] - b[ys2, xs2]).mean())
            if err < best[2]:
                best = (dx, dy, err)
    return best


def _label(dx: int, dy: int) -> str:
    if abs(dx) <= 1 and abs(dy) <= 1:
        return "static"
    if abs(dx) >= abs(dy):
        return "left" if dx < 0 else "right"
    return "up" if dy < 0 else "down"


class MotionAnalyzer:
    name = "motion"
    version = 1

    def __init__(self, fps: float = 4.0):
        self.fps = fps

    def params(self) -> dict:
        return {"fps": self.fps}

    def run(self, asset: Asset, ws) -> list[Event]:
        if asset.kind == AssetKind.PHOTO:
            return []
        frames, rate = decode_gray_frames(ws.proxy_path(asset.asset_id), self.fps, width=64)
        if len(frames) < 2:
            return []
        f32 = frames.astype(np.float32)
        analyzer = f"{self.name}@{self.version}"
        events: list[Event] = []
        per_sec = max(1, int(rate))
        for s in range(int(np.ceil((len(frames) - 1) / per_sec))):
            i0 = s * per_sec
            i1 = min(len(frames) - 1, i0 + per_sec)
            if i1 <= i0:
                break
            dxs, dys, errs, resid = [], [], [], []
            for i in range(i0, i1):
                dx, dy, err = _dominant_shift(f32[i], f32[i + 1])
                dxs.append(dx)
                dys.append(dy)
                errs.append(err)
                # Residual after compensating camera motion ≈ subject motion energy.
                resid.append(err)
            dx_m, dy_m = int(round(float(np.mean(dxs)))), int(round(float(np.mean(dys))))
            events.append(Event(
                asset_id=asset.asset_id, analyzer=analyzer, type="motion",
                t0=float(s), t1=float(s + 1), score=1.0,
                data={
                    "camera_magnitude": round(float(np.hypot(dx_m, dy_m)), 2),
                    "direction": _label(dx_m, dy_m),
                    "dx": dx_m, "dy": dy_m,
                    "subject_energy": round(float(np.mean(resid)), 2),
                },
            ))
        return events
