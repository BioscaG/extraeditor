"""Subject detection and reframing (§9 subjects, §13.3 subject-tracked reframe)."""

from __future__ import annotations

import numpy as np
import pytest

from montaje.analysis.local.subjects import (
    SubjectsAnalyzer,
    box_blur,
    center_for_range,
    detail_map,
    interest_map,
    motion_map,
    subject_center,
)
from montaje.models.events import Event
from montaje.render.reframe import MAX_OFFSET, Crop, crop_for_shot, plan_crop
from tests.unit.conftest_plan import make_shot

HD = (1920, 1080)
VERTICAL = (1080, 1920)


def plate(width: int = 64, height: int = 36, subject_x: float | None = None,
          subject_y: float = 0.4, skin: bool = True) -> np.ndarray:
    """A flat background with one small bright patch, as an (1, h, w, 3) uint8 frame."""
    frame = np.full((height, width, 3), (40, 70, 100), dtype=np.uint8)
    if subject_x is not None:
        cx = int(subject_x * (width - 1))
        cy = int(subject_y * (height - 1))
        half_w, half_h = max(1, width // 16), max(1, height // 8)
        patch = (235, 195, 160) if skin else (250, 250, 250)
        frame[
            max(0, cy - half_h):cy + half_h,
            max(0, cx - half_w):cx + half_w,
        ] = patch
    return frame[None, ...]


# -- interest signals -------------------------------------------------------------------


def test_box_blur_preserves_total_and_uniformity():
    spike = np.zeros((7, 7))
    spike[3, 3] = 49.0
    assert box_blur(spike, 1).sum() == pytest.approx(49.0)
    assert np.allclose(box_blur(np.full((6, 6), 3.0), 1), 3.0)


def test_detail_map_finds_the_edges_not_the_flat_field():
    gray = np.full((36, 64), 60.0)
    gray[14:20, 44:52] = 240.0
    result = detail_map(gray)
    assert result.max() == pytest.approx(1.0)
    peak_y, peak_x = np.unravel_index(int(np.argmax(result)), result.shape)
    assert 38 <= peak_x <= 58
    assert 8 <= peak_y <= 26


def test_motion_map_ignores_a_global_pan():
    """During a pan every cell changes, so nothing should stand out as the subject."""
    frames = np.stack([
        np.roll(np.tile(np.arange(64, dtype=np.float32), (36, 1)), shift, axis=1)
        for shift in range(4)
    ])
    assert motion_map(frames).max() < 0.99 or motion_map(frames).mean() < 0.3


def test_motion_map_finds_a_moving_region_against_a_static_one():
    frames = []
    for i in range(6):
        frame = np.full((36, 64), 40.0)
        frame[14:20, 8:16] = 200.0            # static
        frame[14:20, 44 + i:52 + i] = 200.0   # moving
        frames.append(frame)
    result = motion_map(np.stack(frames))
    peak_x = int(np.unravel_index(int(np.argmax(result)), result.shape)[1])
    assert peak_x > 32


def test_interest_map_is_normalized():
    result = interest_map(plate(subject_x=0.5))
    assert result.max() == pytest.approx(1.0)
    assert result.min() >= 0.0


def test_flat_frame_has_no_confident_subject():
    """Reframing on a flat map is worse than not reframing, so confidence must be 0."""
    cx, cy, confidence = subject_center(plate(subject_x=None))
    assert (cx, cy, confidence) == (0.5, 0.5, 0.0)


# -- subject centre ---------------------------------------------------------------------


@pytest.mark.parametrize("x", [0.15, 0.5, 0.85])
def test_subject_center_finds_the_patch(x: float):
    cx, _, confidence = subject_center(plate(subject_x=x))
    assert cx == pytest.approx(x, abs=0.05), (x, cx)
    assert confidence > 0.3


def test_a_moving_subject_beats_a_static_distraction():
    frames = []
    for i in range(6):
        frame = np.full((36, 64, 3), (40, 70, 100), dtype=np.uint8)
        frame[14:20, 8:16] = (200, 200, 200)
        frame[14:20, 44 + i:52 + i] = (200, 200, 200)
        frames.append(frame)
    cx, _, _ = subject_center(np.stack(frames))
    assert cx > 0.6, cx


def test_confidence_falls_when_interest_fills_the_frame():
    """A third of the frame lighting up is a texture, not a subject."""
    textured = np.random.default_rng(3).integers(0, 255, (1, 36, 64, 3), dtype=np.uint8)
    small = subject_center(plate(subject_x=0.5))[2]
    assert subject_center(textured)[2] < small


def test_subject_center_of_nothing_is_the_middle():
    assert subject_center(np.zeros((0, 4, 4, 3), dtype=np.uint8)) == (0.5, 0.5, 0.0)


def test_skin_tone_is_preferred_over_a_brighter_distraction():
    """In footage of people the subject is a person; detail alone prefers a stage light."""
    frame = plate(subject_x=0.25, skin=True)[0]
    # A brighter, larger white patch on the opposite side.
    frame[4:32, 50:62] = (255, 255, 255)
    cx, _, _ = subject_center(frame[None, ...])
    assert cx < 0.55, cx


def test_center_for_range_ignores_low_confidence():
    events = [
        Event(asset_id="a", analyzer="subjects@1", type="subject", t0=0, t1=2,
              data={"center_x": 0.9, "center_y": 0.5, "confidence": 0.01}),
    ]
    assert center_for_range(events, 0, 2) is None


def test_center_for_range_weights_by_confidence():
    events = [
        Event(asset_id="a", analyzer="subjects@1", type="subject", t0=0, t1=1,
              data={"center_x": 0.2, "center_y": 0.4, "confidence": 0.9}),
        Event(asset_id="a", analyzer="subjects@1", type="subject", t0=1, t1=2,
              data={"center_x": 0.8, "center_y": 0.4, "confidence": 0.15}),
    ]
    cx, _ = center_for_range(events, 0, 2)
    assert cx < 0.4


def test_center_for_range_only_uses_overlapping_events():
    events = [
        Event(asset_id="a", analyzer="subjects@1", type="subject", t0=10, t1=12,
              data={"center_x": 0.9, "center_y": 0.5, "confidence": 0.9}),
    ]
    assert center_for_range(events, 0, 2) is None


def test_analyzer_params_are_in_the_cache_key():
    assert SubjectsAnalyzer(fps=1.0).params() != SubjectsAnalyzer(fps=2.0).params()


# -- crop planning -----------------------------------------------------------------------


def test_matching_aspect_needs_no_crop():
    assert plan_crop(1080, 1920, 1080, 1920, (0.5, 0.5)) is None


def test_wide_source_is_cropped_horizontally():
    crop = plan_crop(*HD, *VERTICAL, None)
    assert crop.height == 1080
    assert crop.width == 608
    assert crop.x == 656  # centred


def test_tall_source_is_cropped_vertically():
    crop = plan_crop(1080, 1920, 1920, 1080, None)
    assert crop.width == 1080
    assert crop.height < 1920


def test_crop_follows_the_subject():
    left = plan_crop(*HD, *VERTICAL, (0.2, 0.4))
    right = plan_crop(*HD, *VERTICAL, (0.8, 0.4))
    assert left.x < 656 < right.x


def test_crop_never_leaves_the_frame():
    for x in (0.0, 0.05, 0.5, 0.95, 1.0):
        crop = plan_crop(*HD, *VERTICAL, (x, 0.5))
        assert 0 <= crop.x <= 1920 - crop.width
        assert 0 <= crop.y <= 1080 - crop.height


def test_crop_offset_is_limited_so_the_subject_is_not_against_the_edge():
    """A subject jammed to the frame edge reads as an error, not as composition."""
    crop = plan_crop(*HD, *VERTICAL, (0.0, 0.5))
    travel = 1920 - crop.width
    middle = travel / 2
    assert crop.x == pytest.approx(middle - middle * MAX_OFFSET, abs=2)


def test_the_clamp_limits_extremes_without_weakening_normal_offsets():
    """Scaling every offset to restrain the extremes would weaken every reframe."""
    crop = plan_crop(*HD, *VERTICAL, (0.75, 0.5))
    # Ideal offset places the subject at the crop centre: 0.75*1920 - 608/2.
    assert crop.x == pytest.approx(0.75 * 1920 - 608 / 2, abs=2)


def test_crop_dimensions_and_offsets_are_even():
    """Chroma-subsampled formats reject odd dimensions."""
    for x in (0.13, 0.37, 0.61, 0.88):
        crop = plan_crop(1921, 1081, *VERTICAL, (x, 0.5))
        assert crop.width % 2 == 0 and crop.height % 2 == 0
        assert crop.x % 2 == 0 and crop.y % 2 == 0


def test_a_high_subject_gets_extra_headroom():
    """Centring a face exactly leaves too much headroom and crops the chin."""
    tall = plan_crop(1080, 1920, 1080, 1080, (0.5, 0.25))
    centred = plan_crop(1080, 1920, 1080, 1080, (0.5, 0.5))
    assert tall.y < centred.y


def test_zero_sized_inputs_are_rejected():
    assert plan_crop(0, 1080, 1080, 1920, None) is None


def test_crop_filter_syntax():
    assert Crop(608, 1080, 656, 0).to_filter() == "crop=608:1080:656:0"


# -- policy resolution ---------------------------------------------------------------------


def test_auto_subject_uses_the_detected_centre():
    from montaje.models.editplan import Reframe

    shot = make_shot("s001")
    shot.reframe = Reframe(mode="auto_subject")
    crop = crop_for_shot(shot, *HD, *VERTICAL, (0.8, 0.4))
    assert crop.x > 656
    assert "subject at" in crop.reason


def test_auto_subject_falls_back_to_centre_without_a_subject():
    from montaje.models.editplan import Reframe

    shot = make_shot("s001")
    shot.reframe = Reframe(mode="auto_subject")
    crop = crop_for_shot(shot, *HD, *VERTICAL, None)
    assert crop.x == 656
    assert "no confident subject" in crop.reason


def test_center_mode_ignores_the_subject():
    shot = make_shot("s001")  # default mode is "center"
    crop = crop_for_shot(shot, *HD, *VERTICAL, (0.9, 0.4))
    assert crop.x == 656


def test_manual_rect_is_used_verbatim():
    from montaje.models.editplan import Reframe

    shot = make_shot("s001")
    shot.reframe = Reframe(mode="manual", rect=[0.25, 0.1, 0.5, 0.8])
    crop = crop_for_shot(shot, *HD, *VERTICAL, None)
    assert (crop.x, crop.y, crop.width, crop.height) == (480, 108, 960, 864)


def test_the_builder_applies_the_styles_reframe_policy():
    """Leaving it at the model default crops 16:9 footage down the middle."""
    from montaje.plan.build import _reframe_for
    from montaje.styles.registry import load_style

    style = load_style("modern-festival")
    assert style is not None
    assert _reframe_for(style).mode == "auto_subject"
    assert _reframe_for(None).mode == "center"


# -- against real media --------------------------------------------------------------------


def test_subject_is_located_in_a_real_clip(ingested):
    """End to end on a real decode, not just numpy arrays."""
    from montaje.index.store import Store

    with Store(ingested.db_path) as store:
        asset = next(a for a in store.list_assets() if a.path.name == "plain.mp4")
    events = SubjectsAnalyzer().run(asset, ingested)
    assert events
    for event in events:
        assert 0.0 <= event.data["center_x"] <= 1.0
        assert 0.0 <= event.data["center_y"] <= 1.0


def test_crop_is_part_of_the_conform_cache_key(ingested):
    """A different crop must not reuse another crop's intermediate."""
    from montaje.index.store import Store
    from montaje.render.conform import ConformSpec

    with Store(ingested.db_path) as store:
        asset = next(a for a in store.list_assets() if a.path.name == "plain.mp4")
    base = ConformSpec(asset=asset, src_in=0.5, src_out=1.5, width=540, height=960, fps=30.0)
    cropped = ConformSpec(asset=asset, src_in=0.5, src_out=1.5, width=540, height=960,
                          fps=30.0, crop=Crop(300, 540, 100, 0))
    assert base.output_name() != cropped.output_name()


def test_conform_applies_the_crop_before_scaling(ingested):
    """Cropping after a scale would resample the picture twice."""
    from montaje.index.store import Store
    from montaje.render.conform import ConformSpec, build_filters

    with Store(ingested.db_path) as store:
        asset = next(a for a in store.list_assets() if a.path.name == "plain.mp4")
    spec = ConformSpec(asset=asset, src_in=0.5, src_out=1.5, width=540, height=960,
                       fps=30.0, crop=Crop(300, 540, 100, 0))
    _, filters = build_filters(spec)
    chain = ",".join(filters)
    assert chain.index("crop=") < chain.index("scale=")
