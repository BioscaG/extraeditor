"""Subject location per shot, for reframing (§9 `subjects`, §15).

What this exists for: cropping 16:9 footage to a 9:16 frame throws away two thirds of
the picture. Doing it from the centre cuts people out of shot; letterboxing instead
wastes half the screen. Either way it is the most visible quality problem in a vertical
edit, and it is fixable without a model.

**Interest** is the sum of three signals, each chosen because it is individually simple
to reason about and to verify:

1. **Detail** — local gradient energy. Subjects carry detail; sky, walls, grass and
   out-of-focus backgrounds do not.
2. **Skin tone** — in footage of people the subject is a person, and a skin-coloured
   region is the strongest cue available without a face detector.
3. **Motion** — a region that changes while its surroundings do not is the subject.
   Computed after subtracting the frame-wide mean change, so a camera pan (where
   *everything* moves) contributes nothing.

Spectral-residual saliency was tried first and rejected: it assumes natural image
statistics, and on the flat-background footage available here it locked onto FFT ringing
rather than the subject — on a test plate with a patch at x=0.85 it reported x=0.50. A
method whose domain assumption cannot be checked is not one to build reframing on.

Apple Vision via the Swift helper (§7.2) replaces the detector behind these events.
"""

from __future__ import annotations

import numpy as np

from montaje import ffmpeg
from montaje.analysis.base import proxy_height
from montaje.analysis.local.color_stats import _skin_mask
from montaje.index.store import Store
from montaje.models.asset import Asset, AssetKind
from montaje.models.events import Event

# Small enough that the arithmetic is instant, large enough to localize a face in frame.
GRID_W = 64
# Relative weights of the three signals. Skin outranks detail because a detailed
# background (foliage, a crowd, a brick wall) is not the subject, while a face is.
DETAIL_WEIGHT = 1.0
SKIN_WEIGHT = 2.5
MOTION_WEIGHT = 1.5
# Blur radius in grid cells. Interest is a region property, not a pixel property.
SMOOTH_CELLS = 2
# Interest below this fraction of the map's peak is discarded before taking a centroid.
# Without it the background contributes at its baseline level across thousands of cells
# and outweighs the subject, so every centroid lands in the middle of the frame — which
# looks exactly like reframing working while doing nothing at all.
KEEP_ABOVE_PEAK = 0.45
# Weighted spread (as a fraction of the frame diagonal) at which confidence reaches zero.
# Interest scattered this widely is a texture, not a subject.
MAX_SPREAD = 0.28


def _decode_rgb(video, fps: float, width: int = GRID_W) -> tuple[np.ndarray, int]:
    height = proxy_height(video, width)
    out = ffmpeg.run(["-v", "error", "-i", str(video),
                      "-vf", f"fps={fps},scale={width}:{height}",
                      "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
    buf = np.frombuffer(out.stdout, dtype=np.uint8)
    frame_bytes = width * height * 3
    if frame_bytes == 0 or len(buf) < frame_bytes:
        return np.zeros((0, max(1, height), width, 3), dtype=np.uint8), height
    n = len(buf) // frame_bytes
    return buf[: n * frame_bytes].reshape(n, height, width, 3), height


def box_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur via cumulative sums. Cheap, and adequate for an interest map."""
    if radius <= 0:
        return image
    padded = np.pad(image, radius, mode="edge")
    window = 2 * radius + 1
    cumulative = np.cumsum(padded, axis=0)
    rows = (cumulative[window - 1 :] - np.pad(cumulative[:-window], ((1, 0), (0, 0)))) / window
    cumulative = np.cumsum(rows, axis=1)
    cols = (cumulative[:, window - 1 :] - np.pad(cumulative[:, :-window], ((0, 0), (1, 0)))) / window
    return cols[: image.shape[0], : image.shape[1]]


def _normalize(field: np.ndarray) -> np.ndarray:
    peak = field.max()
    return field / peak if peak > 0 else np.zeros_like(field)


def detail_map(gray: np.ndarray) -> np.ndarray:
    """Local gradient energy per cell, normalized to 0–1.

    Subjects carry detail; sky, walls and defocused backgrounds do not.
    """
    gy, gx = np.gradient(gray.astype(np.float64))
    return _normalize(box_blur(np.hypot(gx, gy), SMOOTH_CELLS))


def motion_map(gray_frames: np.ndarray) -> np.ndarray:
    """Per-cell change over time, with the frame-wide mean change removed.

    Subtracting the global mean is what makes this a *subject* signal rather than a
    camera-movement signal: during a pan every cell changes, so nothing stands out.
    """
    if len(gray_frames) < 2:
        return np.zeros(gray_frames.shape[1:], dtype=np.float64)
    diffs = np.abs(np.diff(gray_frames.astype(np.float64), axis=0)).mean(axis=0)
    relative = np.maximum(0.0, diffs - diffs.mean())
    return _normalize(box_blur(relative, SMOOTH_CELLS))


def interest_map(frames: np.ndarray) -> np.ndarray:
    """Combined interest over a run of frames, normalized to 0–1."""
    rgb = frames.astype(np.float32) / 255.0
    gray = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    detail = np.mean([detail_map(frame) for frame in gray], axis=0)
    skin = _normalize(box_blur(_skin_mask(rgb).mean(axis=0).astype(np.float64), SMOOTH_CELLS))
    motion = motion_map(gray)

    combined = DETAIL_WEIGHT * detail + SKIN_WEIGHT * skin + MOTION_WEIGHT * motion
    return _normalize(combined)


def subject_center(frames: np.ndarray) -> tuple[float, float, float]:
    """Centre of interest across frames, as `(x, y, confidence)` in 0–1.

    The maps are averaged before a centroid is taken, rather than averaging per-frame
    centroids: a single moving subject would otherwise be averaged into the middle of
    the frame alongside a static distraction.
    """
    if len(frames) == 0:
        return 0.5, 0.5, 0.0
    combined = interest_map(frames)
    peak = combined.max()
    if peak <= 0:
        return 0.5, 0.5, 0.0

    # Keep only the top of the map, then subtract the cut so the remaining weights
    # reflect how far above it each cell is.
    cut = KEEP_ABOVE_PEAK * peak
    weight = np.where(combined >= cut, combined - cut, 0.0)
    mass = weight.sum()
    if mass <= 0:
        return 0.5, 0.5, 0.0

    height, width = weight.shape
    ys, xs = np.mgrid[0:height, 0:width]
    nx = xs / max(1, width - 1)
    ny = ys / max(1, height - 1)
    cx = float((weight * nx).sum() / mass)
    cy = float((weight * ny).sum() / mass)

    # Confidence is how tightly the interest clusters around that centroid, measured as
    # its weighted spread. Counting surviving *cells* instead is fooled by grain: random
    # noise leaves few cells above the cut but scatters them across the whole frame, and
    # reported higher confidence than a clean shot of an actual subject.
    variance = float((weight * ((nx - cx) ** 2 + (ny - cy) ** 2)).sum() / mass)
    spread = variance ** 0.5
    confidence = max(0.0, min(1.0, (MAX_SPREAD - spread) / MAX_SPREAD))
    return round(cx, 4), round(cy, 4), round(confidence, 4)


class SubjectsAnalyzer:
    """Per-shot subject centre and confidence, for `reframe: auto_subject`."""

    name = "subjects"
    version = 1

    def __init__(self, fps: float = 2.0, grid_width: int = GRID_W):
        self.fps = fps
        self.grid_width = grid_width

    def params(self) -> dict:
        return {
            "fps": self.fps,
            "grid_width": self.grid_width,
            "weights": [DETAIL_WEIGHT, SKIN_WEIGHT, MOTION_WEIGHT],
            "keep_above_peak": KEEP_ABOVE_PEAK,
            "smooth_cells": SMOOTH_CELLS,
        }

    def run(self, asset: Asset, ws) -> list[Event]:
        if asset.kind == AssetKind.PHOTO:
            return []
        frames, _ = _decode_rgb(ws.proxy_path(asset.asset_id), self.fps, self.grid_width)
        if len(frames) == 0:
            return []

        with Store(ws.db_path) as store:
            shots = store.get_events(asset.asset_id, "shots", "shot")
        if not shots:
            shots = [Event(asset_id=asset.asset_id, analyzer="shots@1", type="shot",
                           t0=0.0, t1=asset.duration_s)]

        analyzer = f"{self.name}@{self.version}"
        events: list[Event] = []
        for index, shot in enumerate(shots):
            i0 = int(shot.t0 * self.fps)
            i1 = max(i0 + 1, int(shot.t1 * self.fps))
            chunk = frames[i0:i1]
            if len(chunk) == 0:
                continue
            cx, cy, confidence = subject_center(chunk)
            events.append(Event(
                asset_id=asset.asset_id, analyzer=analyzer, type="subject",
                t0=shot.t0, t1=shot.t1, score=confidence,
                data={"shot_index": index, "center_x": cx, "center_y": cy,
                      "confidence": confidence},
            ))
        return events


def center_for_range(
    events: list[Event], t0: float, t1: float, min_confidence: float = 0.12
) -> tuple[float, float] | None:
    """Subject centre covering a source range, or None if nothing is confident enough.

    Returning None rather than a default is deliberate: `auto_subject` must fall back to
    a centre crop when there is no subject, and a low-confidence guess is worse than the
    centre because it moves the crop somewhere arbitrary.
    """
    overlapping = [
        e for e in events
        if e.type == "subject" and e.t1 > t0 and e.t0 < t1
        and float(e.data.get("confidence", 0.0)) >= min_confidence
    ]
    if not overlapping:
        return None
    weights = [float(e.data["confidence"]) for e in overlapping]
    total = sum(weights) or 1.0
    cx = sum(float(e.data["center_x"]) * w for e, w in zip(overlapping, weights, strict=True)) / total
    cy = sum(float(e.data["center_y"]) * w for e, w in zip(overlapping, weights, strict=True)) / total
    return round(cx, 4), round(cy, 4)
