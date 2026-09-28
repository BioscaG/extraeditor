"""Subject-aware reframing (§13.3 `subject-tracked reframe`, §17.2 `reframe`).

Cropping 16:9 footage into a 9:16 frame discards two thirds of the picture. There are
three ways to choose what survives, and only one is acceptable:

- **letterbox** — keeps everything, wastes half the screen. What the conform did before.
- **centre crop** — uses the whole screen, cuts people out of frame whenever they are
  not dead centre, which on handheld footage is most of the time.
- **subject crop** — puts the crop window over whatever the `subjects` analyzer found.

The crop is computed once per shot and held still. A crop that follows the subject frame
by frame looks like a security camera; the eye reads a locked frame as a deliberate
composition. Slow eased movement across a shot is a separate effect and belongs in a
component, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

# Faces sit in the upper half of a portrait frame, so a subject found high in the source
# is biased a little higher still — cropping a head to the exact centre leaves too much
# headroom and cuts the chin.
HEADROOM_BIAS = 0.06
# How far the crop may sit from centre, as a fraction of the available travel. Allowing
# the full range puts the subject hard against the frame edge, which reads as a mistake
# rather than as composition.
MAX_OFFSET = 0.82


@dataclass(frozen=True)
class Crop:
    """A crop rectangle in source pixels, ready for ffmpeg's `crop` filter."""

    width: int
    height: int
    x: int
    y: int
    reason: str = ""

    def to_filter(self) -> str:
        return f"crop={self.width}:{self.height}:{self.x}:{self.y}"

    @property
    def is_full_frame(self) -> bool:
        return self.x == 0 and self.y == 0


def _even(value: int) -> int:
    """Chroma-subsampled formats need even dimensions and offsets."""
    return value - (value % 2)


def plan_crop(
    source_w: int,
    source_h: int,
    target_w: int,
    target_h: int,
    center: tuple[float, float] | None = None,
) -> Crop | None:
    """The crop that fills a `target` frame from a `source`, centred on `center`.

    `center` is normalized (0–1) in source coordinates; None means centre the crop.
    Returns None when the aspect ratios already match, so the caller can skip the filter
    entirely rather than emitting a no-op.
    """
    if source_w <= 0 or source_h <= 0 or target_w <= 0 or target_h <= 0:
        return None
    source_aspect = source_w / source_h
    target_aspect = target_w / target_h
    if abs(source_aspect - target_aspect) < 0.01:
        return None

    if source_aspect > target_aspect:
        # Source is wider: keep full height, crop horizontally.
        crop_h = source_h
        crop_w = _even(int(round(source_h * target_aspect)))
    else:
        crop_w = source_w
        crop_h = _even(int(round(source_w / target_aspect)))
    crop_w = min(crop_w, _even(source_w))
    crop_h = min(crop_h, _even(source_h))

    travel_x = source_w - crop_w
    travel_y = source_h - crop_h

    if center is None:
        return Crop(crop_w, crop_h, _even(travel_x // 2), _even(travel_y // 2),
                    reason="centre (no confident subject)")

    cx, cy = center
    # Bias a high subject a little higher: exact centring on a face leaves too much
    # headroom and crops the chin.
    if travel_y > 0 and cy < 0.5:
        cy = max(0.0, cy - HEADROOM_BIAS)

    x = _offset(cx, travel_x, crop_w)
    y = _offset(cy, travel_y, crop_h)
    return Crop(crop_w, crop_h, x, y,
                reason=f"subject at ({center[0]:.2f}, {center[1]:.2f})")


def _offset(normalized_center: float, travel: int, crop_size: int) -> int:
    """Crop offset that puts `normalized_center` in the middle of the crop window.

    The deviation from centre is *clamped* to `MAX_OFFSET` of the available travel, not
    scaled by it: scaling weakens every reframe in order to restrain the extreme ones,
    which is the wrong trade — a subject two thirds of the way across should be followed
    fully, and only one hard against the frame edge needs holding back.
    """
    if travel <= 0:
        return 0
    source_size = crop_size + travel
    ideal = normalized_center * source_size - crop_size / 2
    middle = travel / 2
    limit = middle * MAX_OFFSET
    limited = middle + max(-limit, min(limit, ideal - middle))
    return _even(int(round(min(max(limited, 0.0), float(travel)))))


def crop_for_shot(
    shot,
    source_w: int,
    source_h: int,
    target_w: int,
    target_h: int,
    center: tuple[float, float] | None,
) -> Crop | None:
    """Resolve a shot's `reframe` policy into a crop.

    - `manual` uses the shot's explicit rect;
    - `auto_subject` uses the analyzer's centre, falling back to the middle;
    - anything else centres.
    """
    mode = shot.reframe.mode
    if mode == "manual" and shot.reframe.rect and len(shot.reframe.rect) == 4:
        rx, ry, rw, rh = shot.reframe.rect
        return Crop(
            width=_even(max(2, int(round(rw * source_w)))),
            height=_even(max(2, int(round(rh * source_h)))),
            x=_even(max(0, int(round(rx * source_w)))),
            y=_even(max(0, int(round(ry * source_h)))),
            reason="manual rect",
        )
    if mode == "auto_subject":
        return plan_crop(source_w, source_h, target_w, target_h, center)
    return plan_crop(source_w, source_h, target_w, target_h, None)
