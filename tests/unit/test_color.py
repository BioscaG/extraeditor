"""Color normalization, matching and grading (§15)."""

from __future__ import annotations

from pathlib import Path

import pytest

from montaje.color.grade import Grade, grade_from_style
from montaje.color.normalize import (
    MAX_GAIN,
    MIN_GAIN,
    ShotCorrection,
    correction_for_shot,
    corrections_for_asset,
    match_to_anchor,
    pick_anchor,
    target_exposure_for,
)
from montaje.models.events import Event
from montaje.models.style import ColorPolicy


def stats(wb=(1.0, 1.0, 1.0), skin=0.05, exposure=0.5, lab=(50.0, 0.0, 0.0), index=0) -> dict:
    return {"shot_index": index, "wb_gains": list(wb), "skin_fraction": skin,
            "exposure": exposure, "lab_mean": list(lab)}


def color_event(index: int = 0, **kw) -> Event:
    return Event(asset_id="a_1", analyzer="color_stats@1", type="shot_color",
                 t0=float(index), t1=float(index + 1), data=stats(index=index, **kw))


# -- white balance ------------------------------------------------------------------


def test_neutral_frame_needs_no_correction():
    assert correction_for_shot(stats()).is_identity


def test_warm_frame_gets_its_red_pulled_down():
    """A warm cast means red is over-represented, so the corrective gain is below 1."""
    correction = correction_for_shot(stats(wb=(0.8, 1.0, 1.25)))
    assert correction.gain_r < 1.0
    assert correction.gain_b > 1.0


def test_green_is_the_anchor_channel():
    """Only the R/B ratio is a colour cast; green carries the exposure."""
    assert correction_for_shot(stats(wb=(0.8, 1.0, 1.25))).gain_g == 1.0


def test_correction_strength_scales_the_pull():
    weak = correction_for_shot(stats(wb=(0.7, 1.0, 1.4)), strength=0.2)
    strong = correction_for_shot(stats(wb=(0.7, 1.0, 1.4)), strength=1.0)
    assert abs(strong.gain_r - 1.0) > abs(weak.gain_r - 1.0)


def test_zero_strength_is_a_no_op():
    assert correction_for_shot(stats(wb=(0.5, 1.0, 2.0)), strength=0.0).is_identity


def test_gains_are_clamped():
    """A genuinely warm scene must stay warm rather than be neutralised to grey."""
    correction = correction_for_shot(stats(wb=(0.2, 1.0, 5.0)), strength=1.0)
    # Gains are stored rounded to 4dp, so allow that quantum at the clamp edges.
    for gain in (correction.gain_r, correction.gain_b):
        assert MIN_GAIN - 1e-4 <= gain <= MAX_GAIN + 1e-4


def test_skin_heavy_frames_get_a_reduced_correction():
    """Warm skin is the subject, not a cast; correcting it fully drains faces."""
    normal = correction_for_shot(stats(wb=(0.8, 1.0, 1.25), skin=0.05))
    face = correction_for_shot(stats(wb=(0.8, 1.0, 1.25), skin=0.5))
    assert abs(face.gain_r - 1.0) < abs(normal.gain_r - 1.0)
    assert "skin-heavy" in face.notes


# -- exposure --------------------------------------------------------------------------


def test_exposure_is_pulled_toward_the_target():
    dark = correction_for_shot(stats(exposure=0.25), strength=1.0, target_exposure=0.5)
    bright = correction_for_shot(stats(exposure=0.75), strength=1.0, target_exposure=0.5)
    assert dark.exposure_gain > 1.0
    assert bright.exposure_gain < 1.0


def test_exposure_correction_is_clamped_to_a_stop_range():
    correction = correction_for_shot(stats(exposure=0.01), strength=1.0, target_exposure=0.5)
    assert correction.exposure_gain <= 2.0 ** 0.75 + 1e-4
    assert "clamped" in correction.notes


def test_no_target_means_no_exposure_change():
    assert correction_for_shot(stats(exposure=0.2)).exposure_gain == 1.0


def test_target_exposure_uses_the_median():
    """One blown-out shot must not drag the whole edit's brightness."""
    events = [color_event(0, exposure=0.4), color_event(1, exposure=0.45),
              color_event(2, exposure=0.99)]
    assert target_exposure_for(events) == pytest.approx(0.45)


def test_target_exposure_of_nothing_is_a_safe_default():
    assert target_exposure_for([]) == 0.5


# -- filters ----------------------------------------------------------------------------


def test_identity_correction_emits_no_filter():
    assert ShotCorrection(0, 1.0, 1.0, 1.0, 1.0).to_filter() is None


def test_filter_multiplies_exposure_into_every_channel():
    f = ShotCorrection(0, 0.9, 1.0, 1.1, 2.0).to_filter()
    assert "rr=1.8000" in f and "gg=2.0000" in f and "bb=2.2000" in f


def test_corrections_for_asset_are_keyed_by_shot_index():
    events = [color_event(0, wb=(0.8, 1.0, 1.2)), color_event(1, wb=(1.2, 1.0, 0.8))]
    corrections = corrections_for_asset(events)
    assert set(corrections) == {0, 1}
    assert corrections[0].gain_r < 1.0
    assert corrections[1].gain_r > 1.0


# -- matching ------------------------------------------------------------------------------


def test_match_moves_a_shot_toward_the_anchor():
    """A shot bluer than its anchor needs more red and less blue."""
    shot = stats(lab=(50.0, -10.0, -15.0))
    anchor = stats(lab=(50.0, 5.0, 10.0))
    correction = match_to_anchor(shot, anchor, strength=1.0)
    assert correction.gain_r > 1.0
    assert correction.gain_b < 1.0


def test_match_to_itself_is_identity():
    s = stats(lab=(50.0, 3.0, -2.0))
    assert match_to_anchor(s, s, strength=1.0).is_identity


def test_match_records_the_anchor_in_notes():
    assert "shot 7" in match_to_anchor(stats(), stats(index=7)).notes


def test_match_gains_are_clamped():
    correction = match_to_anchor(stats(lab=(10.0, -120.0, 120.0)),
                                stats(lab=(90.0, 120.0, -120.0)), strength=1.0)
    for gain in (correction.gain_r, correction.gain_b, correction.exposure_gain):
        assert MIN_GAIN - 1e-4 <= gain <= MAX_GAIN + 1e-4


def test_anchor_is_the_median_shot_not_the_first():
    """One badly-exposed opener must not drag the whole group."""
    events = [
        color_event(0, lab=(10.0, -30.0, 0.0)),  # outlier
        color_event(1, lab=(50.0, 0.0, 0.0)),
        color_event(2, lab=(52.0, 1.0, 1.0)),
        color_event(3, lab=(48.0, -1.0, 0.0)),
    ]
    anchor = pick_anchor(events)
    assert anchor is not None
    assert anchor.data["shot_index"] != 0


def test_pick_anchor_of_nothing_is_none():
    assert pick_anchor([]) is None


# -- grade ---------------------------------------------------------------------------------


def test_empty_grade_emits_no_filter():
    assert Grade().to_filter() is None


def test_parametric_grade_emits_eq_and_colorbalance():
    f = Grade(contrast=1.1, saturation=0.9, shadow_tint=(-0.05, 0.0, 0.08)).to_filter()
    assert "eq=contrast=1.1000" in f
    assert "colorbalance" in f


def test_lut_grade_skips_parametric_adjustments():
    """A LUT already encodes contrast and saturation; applying both doubles the look."""
    f = Grade(lut=Path("/luts/warm.cube"), contrast=1.5, saturation=2.0).to_filter()
    assert "lut3d" in f
    assert "eq=contrast" not in f


def test_finishing_is_applied_after_the_look():
    chain = Grade(contrast=1.1, vignette=0.4, sharpen=0.3, grain=0.2).filters()
    assert chain.index("eq=contrast=1.1000:saturation=1.0000") < next(
        i for i, f in enumerate(chain) if f.startswith("vignette")
    )
    # Grain must be last: it sits on top of the graded image.
    assert chain[-1].startswith("noise")


def test_grade_from_style_ignores_a_missing_lut(tmp_path):
    """A style is data and may reference a LUT this install does not have."""
    grade = grade_from_style(ColorPolicy(lut="nonexistent.cube", grain=0.1), lut_dir=tmp_path)
    assert grade.lut is None


def test_grade_from_style_leaves_finishing_to_the_composition():
    """Grain per shot *and* over the composition would double it, and grain under the
    captions but not over them reads as two images composited together."""
    grade = grade_from_style(ColorPolicy(grain=0.2, vignette=0.3))
    assert grade.grain == 0.0
    assert grade.vignette == 0.0


def test_grade_from_style_resolves_an_existing_lut(tmp_path):
    (tmp_path / "warm.cube").write_text("LUT_3D_SIZE 2\n")
    grade = grade_from_style(ColorPolicy(lut="warm.cube"), lut_dir=tmp_path)
    assert grade.lut == tmp_path / "warm.cube"
