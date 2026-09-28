"""Per-shot exposure and white-balance normalization (§15 step 2).

Footage from several phones at the same event does not match: different auto white
balance, different exposure. Inconsistent color between phones is one of the
listed amateur tells (§28), and it is the cheapest one to fix.

The estimate is robust gray-world **with skin-tone protection**: warm skin would
otherwise read as a warm cast, and correcting it drains the life out of faces. The
correction strength is clamped so a genuinely warm scene (sunset, red stage
lighting) stays warm rather than being neutralized into grey.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from montaje.models.events import Event

# Maximum per-channel gain the normalizer may apply, before `strength` scaling.
MAX_GAIN = 1.35
MIN_GAIN = 1.0 / MAX_GAIN
# Above this share of skin pixels, gray-world is not trustworthy: the frame is
# mostly a face and its warmth is the subject, not a cast.
SKIN_HEAVY_FRACTION = 0.25
MAX_EXPOSURE_STOPS = 0.75


@dataclass(frozen=True)
class ShotCorrection:
    """Per-shot correction, in a form ffmpeg can apply directly."""

    shot_index: int
    gain_r: float
    gain_g: float
    gain_b: float
    exposure_gain: float
    notes: str = ""

    @property
    def is_identity(self) -> bool:
        return all(abs(v - 1.0) < 1e-3 for v in
                   (self.gain_r, self.gain_g, self.gain_b, self.exposure_gain))

    def to_filter(self) -> str | None:
        """An ffmpeg filter chain applying this correction, or None if it is a no-op.

        `colorchannelmixer` multiplies each channel independently, which is exactly a
        white-balance gain; exposure rides on the same filter as a common factor.
        """
        if self.is_identity:
            return None
        r = self.gain_r * self.exposure_gain
        g = self.gain_g * self.exposure_gain
        b = self.gain_b * self.exposure_gain
        return f"colorchannelmixer=rr={r:.4f}:gg={g:.4f}:bb={b:.4f}"


def _clamp_gain(gain: float, strength: float) -> float:
    """Blend toward the correction by `strength`, then clamp to the allowed range."""
    blended = 1.0 + (gain - 1.0) * strength
    return float(np.clip(blended, MIN_GAIN, MAX_GAIN))


def correction_for_shot(
    stats: dict,
    strength: float = 0.7,
    target_exposure: float | None = None,
) -> ShotCorrection:
    """Derive a correction from one `color_stats` shot_color event's data."""
    gains = stats.get("wb_gains") or [1.0, 1.0, 1.0]
    skin = float(stats.get("skin_fraction", 0.0))
    notes: list[str] = []

    effective = strength
    if skin > SKIN_HEAVY_FRACTION:
        # Scale the correction down rather than skipping it: a face-heavy shot still
        # benefits from a small nudge, just not a full gray-world pull.
        effective = strength * 0.35
        notes.append(f"skin-heavy ({skin * 100:.0f}%), correction reduced")

    # Normalize the gains so green is the anchor: only the R/B *ratio* is a cast.
    g = float(gains[1]) or 1.0
    gain_r = _clamp_gain(float(gains[0]) / g, effective)
    gain_g = 1.0
    gain_b = _clamp_gain(float(gains[2]) / g, effective)

    exposure_gain = 1.0
    if target_exposure is not None:
        current = float(stats.get("exposure", target_exposure)) or target_exposure
        if current > 1e-4:
            ratio = target_exposure / current
            limit = 2.0 ** MAX_EXPOSURE_STOPS
            exposure_gain = float(np.clip(1.0 + (ratio - 1.0) * effective, 1 / limit, limit))
            if abs(exposure_gain - ratio) > 0.01:
                notes.append("exposure correction clamped")

    return ShotCorrection(
        shot_index=int(stats.get("shot_index", 0)),
        gain_r=round(gain_r, 4),
        gain_g=round(gain_g, 4),
        gain_b=round(gain_b, 4),
        exposure_gain=round(exposure_gain, 4),
        notes="; ".join(notes),
    )


def target_exposure_for(events: list[Event]) -> float:
    """Median exposure across shots — the level everything is pulled toward.

    Median, not mean: one blown-out or near-black shot should not drag the whole
    edit's brightness with it.
    """
    values = [e.data.get("exposure") for e in events if e.data.get("exposure") is not None]
    return float(np.median(values)) if values else 0.5


def corrections_for_asset(
    events: list[Event],
    strength: float = 0.7,
    match_exposure: bool = True,
) -> dict[int, ShotCorrection]:
    """Corrections for every shot of one asset, keyed by shot index."""
    shot_events = [e for e in events if e.type == "shot_color"]
    target = target_exposure_for(shot_events) if match_exposure else None
    return {
        int(e.data.get("shot_index", i)): correction_for_shot(e.data, strength, target)
        for i, e in enumerate(shot_events)
    }


def match_to_anchor(
    stats: dict,
    anchor_stats: dict,
    strength: float = 0.6,
) -> ShotCorrection:
    """Match one shot to an anchor shot in Lab (§15 step 3).

    Used within a sync group (the same moment from two phones) and within a section.
    Operating on the Lab means rather than raw RGB keeps the match perceptual: two
    shots can have very different RGB averages and still look matched.
    """
    lab = stats.get("lab_mean") or [50.0, 0.0, 0.0]
    anchor = anchor_stats.get("lab_mean") or [50.0, 0.0, 0.0]

    # a* is roughly green↔red and b* is blue↔yellow, so a Lab difference maps onto
    # R and B gains directly enough for a clamped correction.
    da = (anchor[1] - lab[1]) / 128.0
    db = (anchor[2] - lab[2]) / 128.0
    gain_r = _clamp_gain(1.0 + da, strength)
    gain_b = _clamp_gain(1.0 - db, strength)

    exposure_gain = 1.0
    if lab[0] > 1e-3:
        exposure_gain = _clamp_gain(anchor[0] / lab[0], strength)

    return ShotCorrection(
        shot_index=int(stats.get("shot_index", 0)),
        gain_r=round(gain_r, 4), gain_g=1.0, gain_b=round(gain_b, 4),
        exposure_gain=round(exposure_gain, 4),
        notes=f"matched to shot {anchor_stats.get('shot_index')}",
    )


def pick_anchor(events: list[Event]) -> Event | None:
    """The shot a group should be matched to: the one closest to the group median.

    Matching to the median shot rather than the first means one badly-exposed opener
    cannot drag the whole group with it.
    """
    shots = [e for e in events if e.type == "shot_color"]
    if not shots:
        return None
    labs = np.array([e.data.get("lab_mean", [50.0, 0.0, 0.0]) for e in shots], dtype=float)
    median = np.median(labs, axis=0)
    distances = np.linalg.norm(labs - median, axis=1)
    return shots[int(np.argmin(distances))]
