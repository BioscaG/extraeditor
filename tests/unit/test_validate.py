"""Plan validation: hard errors block a render, warnings are advisory (§18.2)."""

from __future__ import annotations

import pytest

from montaje.models.editplan import AudioMode, Section, Sfx
from montaje.models.events import Event
from montaje.plan.validate import Severity, ValidationContext, errors, validate, warnings
from tests.unit.conftest_plan import (
    make_asset,
    make_plan,
    make_shot,
    transition_ref,
    usable_events,
)


@pytest.fixture()
def ctx():
    return ValidationContext(
        assets={"a_1": make_asset("a_1", 30.0)},
        usable_spans={"a_1": [(0.0, 30.0)]},
        min_shot_s=0.3,
    )


def codes(problems, severity=None) -> set[str]:
    return {p.code for p in problems if severity is None or p.severity == severity}


def test_a_clean_plan_has_no_errors(ctx):
    problems = validate(make_plan(), ctx)
    assert errors(problems) == [], [str(p) for p in errors(problems)]


def test_errors_are_sorted_before_warnings(ctx):
    plan = make_plan([make_shot("s001", src_out=100.0, intent="")])
    problems = validate(plan, ctx)
    assert problems[0].severity == Severity.ERROR


# -- source ranges -----------------------------------------------------------------


def test_source_beyond_asset_duration_is_an_error(ctx):
    plan = make_plan([make_shot("s001", src_in=25.0, src_out=40.0)])
    assert "source_out_of_range" in codes(errors(validate(plan, ctx)))


def test_unknown_asset_is_an_error(ctx):
    plan = make_plan([make_shot("s001", asset="a_missing")])
    assert "unknown_asset" in codes(errors(validate(plan, ctx)))


def test_shot_below_minimum_length_is_an_error(ctx):
    plan = make_plan([make_shot("s001", src_in=1.0, src_out=1.1)])
    assert "shot_too_short" in codes(errors(validate(plan, ctx)))


def test_duplicate_shot_ids_are_an_error(ctx):
    plan = make_plan([
        make_shot("s001", src_in=0.0, src_out=2.0, timeline_in=0),
        make_shot("s001", src_in=5.0, src_out=7.0, timeline_in=60),
    ])
    assert "duplicate_shot_id" in codes(errors(validate(plan, ctx)))


def test_unusable_material_without_intent_is_an_error(ctx):
    ctx.usable_spans["a_1"] = [(10.0, 30.0)]
    plan = make_plan([make_shot("s001", src_in=0.0, src_out=2.0, intent="")])
    assert "outside_usable" in codes(errors(validate(plan, ctx)))


def test_unusable_material_with_intent_is_only_a_warning(ctx):
    """The agent may knowingly use a weak shot; §18.2 requires it to say why."""
    ctx.usable_spans["a_1"] = [(10.0, 30.0)]
    plan = make_plan([make_shot("s001", src_in=0.0, src_out=2.0,
                                intent="deliberate blurry opener")])
    problems = validate(plan, ctx)
    assert "outside_usable" not in codes(errors(problems))
    assert "outside_usable" in codes(warnings(problems))


def test_missing_intent_is_a_warning(ctx):
    plan = make_plan([make_shot("s001", intent="")])
    assert "missing_intent" in codes(warnings(validate(plan, ctx)))


def test_reused_source_range_is_a_warning(ctx):
    plan = make_plan([
        make_shot("s001", src_in=1.0, src_out=3.0, timeline_in=0),
        make_shot("s002", src_in=1.5, src_out=3.5, timeline_in=60),
    ])
    assert "reused_source" in codes(warnings(validate(plan, ctx)))


def test_degraded_asset_is_a_warning(ctx):
    ctx.assets["a_deg"] = make_asset("a_deg", 30.0, degraded=True)
    ctx.usable_spans["a_deg"] = [(0.0, 30.0)]
    plan = make_plan([make_shot("s001", asset="a_deg")])
    assert "degraded_asset" in codes(warnings(validate(plan, ctx)))


# -- timeline -----------------------------------------------------------------------


def test_timeline_gap_is_an_error(ctx):
    plan = make_plan([
        make_shot("s001", src_in=0.0, src_out=2.0, timeline_in=0),
        make_shot("s002", src_in=5.0, src_out=7.0, timeline_in=90),  # 30 frame gap
    ])
    assert "timeline_gap" in codes(errors(validate(plan, ctx)))


def test_timeline_not_starting_at_zero_is_an_error(ctx):
    plan = make_plan([make_shot("s001", timeline_in=15)])
    assert "timeline_gap" in codes(errors(validate(plan, ctx)))


def test_unintended_overlap_is_an_error(ctx):
    plan = make_plan([
        make_shot("s001", src_in=0.0, src_out=2.0, timeline_in=0),
        make_shot("s002", src_in=5.0, src_out=7.0, timeline_in=30),
    ])
    assert "timeline_overlap" in codes(errors(validate(plan, ctx)))


def test_transition_overlap_is_allowed(ctx):
    """A transition on the incoming shot legitimately overlaps the outgoing one."""
    plan = make_plan([
        make_shot("s001", src_in=0.0, src_out=2.0, timeline_in=0),
        make_shot("s002", src_in=5.0, src_out=7.0, timeline_in=52,
                  transition=transition_ref(frames=8)),
    ])
    assert "timeline_overlap" not in codes(errors(validate(plan, ctx)))


def test_empty_plan_is_an_error(ctx):
    assert "empty_timeline" in codes(errors(validate(make_plan([]), ctx)))


# -- duration ------------------------------------------------------------------------


def test_duration_outside_tolerance_is_an_error(ctx):
    ctx.duration_target_s = 75.0
    ctx.duration_tolerance_s = 15.0
    assert "duration_out_of_tolerance" in codes(errors(validate(make_plan(), ctx)))


def test_duration_within_tolerance_passes(ctx):
    ctx.duration_target_s = 4.0
    ctx.duration_tolerance_s = 2.0
    assert "duration_out_of_tolerance" not in codes(errors(validate(make_plan(), ctx)))


# -- components and sfx ----------------------------------------------------------------


def test_unapproved_component_is_an_error(ctx):
    ctx.stable_components = {"transition.zoom_punch@1.0"}
    plan = make_plan([make_shot("s001", transition=transition_ref("transition.made_up@9.9"))])
    assert "unapproved_component" in codes(errors(validate(plan, ctx)))


def test_stable_component_passes(ctx):
    ctx.stable_components = {"transition.whip_pan@1.0"}
    plan = make_plan([make_shot("s001", transition=transition_ref("transition.whip_pan@1.0"))])
    assert "unapproved_component" not in codes(errors(validate(plan, ctx)))


def test_accepted_draft_component_passes(ctx):
    from montaje.models.editplan import Draft

    ctx.stable_components = {"transition.whip_pan@1.0"}
    plan = make_plan([make_shot("s001", transition=transition_ref("draft.neon@0.1"))])
    plan.drafts = [Draft(id="draft.neon", path="drafts/neon.tsx")]
    assert "unapproved_component" not in codes(errors(validate(plan, ctx)))


def test_sfx_without_license_is_an_error(ctx):
    ctx.licensed_sfx = {"impact.deep_03"}
    plan = make_plan()
    plan.sfx = [Sfx(id="fx1", sfx="mystery.whoosh", anchor_frame=10)]
    assert "sfx_without_license" in codes(errors(validate(plan, ctx)))


def test_licensed_sfx_passes(ctx):
    ctx.licensed_sfx = {"impact.deep_03"}
    plan = make_plan()
    plan.sfx = [Sfx(id="fx1", sfx="impact.deep_03", anchor_frame=10)]
    assert "sfx_without_license" not in codes(errors(validate(plan, ctx)))


def test_sfx_anchored_off_the_timeline_is_an_error(ctx):
    plan = make_plan()
    plan.sfx = [Sfx(id="fx1", sfx="impact.deep_03", anchor_frame=9999)]
    assert "sfx_off_timeline" in codes(errors(validate(plan, ctx)))


def test_sfx_requires_an_anchor():
    with pytest.raises(ValueError):
        Sfx(id="fx1", sfx="impact.deep_03")


# -- audio ----------------------------------------------------------------------------


def test_speech_over_music_without_ducking_is_an_error(ctx):
    ctx.speech_spans = {"a_1": [(1.0, 3.0)]}
    plan = make_plan([make_shot("s001", src_in=1.0, src_out=3.0,
                                audio_mode=AudioMode.ORIGINAL)])
    assert "speech_without_ducking" in codes(errors(validate(plan, ctx)))


def test_speech_with_ducking_passes(ctx):
    ctx.speech_spans = {"a_1": [(1.0, 3.0)]}
    plan = make_plan([make_shot("s001", src_in=1.0, src_out=3.0,
                                audio_mode=AudioMode.ORIGINAL, duck_db=-14.0)])
    assert "speech_without_ducking" not in codes(errors(validate(plan, ctx)))


def test_music_only_shot_needs_no_ducking(ctx):
    ctx.speech_spans = {"a_1": [(1.0, 3.0)]}
    plan = make_plan([make_shot("s001", src_in=1.0, src_out=3.0,
                                audio_mode=AudioMode.MUSIC_ONLY)])
    assert "speech_without_ducking" not in codes(errors(validate(plan, ctx)))


def test_overlapping_speech_is_an_error(ctx):
    """Two people talking at once is unlistenable, and a J-cut can cause it."""
    ctx.speech_spans = {"a_1": [(0.0, 10.0)]}
    plan = make_plan([
        make_shot("s001", src_in=0.0, src_out=2.0, timeline_in=0,
                  audio_mode=AudioMode.ORIGINAL, duck_db=-14.0),
        make_shot("s002", src_in=5.0, src_out=7.0, timeline_in=60,
                  audio_mode=AudioMode.ORIGINAL, duck_db=-14.0),
    ])
    plan.shots[1].audio.j_cut_frames = 20  # reaches back into the previous shot's speech
    assert "overlapping_speech" in codes(errors(validate(plan, ctx)))


# -- density and sections ---------------------------------------------------------------


def test_transition_density_warning(ctx):
    ctx.max_transitions_per_10s = 1.0
    plan = make_plan([
        make_shot("s001", src_in=0.0, src_out=2.0, timeline_in=0,
                  transition=transition_ref()),
        make_shot("s002", src_in=5.0, src_out=7.0, timeline_in=60,
                  transition=transition_ref()),
    ])
    assert "transition_density" in codes(warnings(validate(plan, ctx)))


def test_sfx_density_warning(ctx):
    ctx.max_sfx_per_10s = 1.0
    plan = make_plan()
    plan.sfx = [Sfx(id=f"fx{i}", sfx="impact.deep_03", anchor_frame=i * 10) for i in range(5)]
    assert "sfx_density" in codes(warnings(validate(plan, ctx)))


def test_unknown_section_reference_is_a_warning(ctx):
    plan = make_plan([make_shot("s001", section="nonexistent")])
    assert "unknown_section" in codes(warnings(validate(plan, ctx)))


def test_declared_section_passes(ctx):
    plan = make_plan(
        [make_shot("s001", section="intro")],
        sections=[Section(id="intro", from_frame=0, to_frame=60)],
    )
    assert "unknown_section" not in codes(warnings(validate(plan, ctx)))


def test_context_from_events_extracts_usable_and_speech():
    events = {
        "a_1": usable_events("a_1", 0.0, 10.0)
        + [Event(asset_id="a_1", analyzer="vad@1", type="speech", t0=2.0, t1=4.0)]
    }
    ctx = ValidationContext.from_events({"a_1": make_asset("a_1")}, events)
    assert ctx.usable_spans["a_1"] == [(0.0, 10.0)]
    assert ctx.speech_spans["a_1"] == [(2.0, 4.0)]


def test_problem_str_includes_shot_and_code(ctx):
    plan = make_plan([make_shot("s001", src_out=100.0)])
    problem = errors(validate(plan, ctx))[0]
    assert "s001" in str(problem) and "source_out_of_range" in str(problem)
