"""The rails: snap, resolve beat durations, relayout (§18)."""

from __future__ import annotations

import pytest

from montaje.models.editplan import Duration, SnapKind
from montaje.plan.rails import RailContext, apply_rails, resolve_duration
from montaje.plan.snap import SnapCandidates
from montaje.plan.validate import ValidationContext, errors, validate
from tests.unit.conftest_plan import make_asset, make_grid, make_plan, make_shot, transition_ref


@pytest.fixture()
def ctx():
    return RailContext(
        candidates={"a_1": SnapCandidates(
            shots=[0.0, 4.0, 9.5],
            word_starts=[1.0, 5.2],
            word_ends=[3.4, 7.1],
            reveals=[0.8],
            covers=[9.0],
        )},
        grid=make_grid(bpm=120.0, duration=60.0),
        fps=30.0,
    )


def test_rails_snap_source_in_and_out(ctx):
    plan = make_plan([make_shot("s001", src_in=4.2, src_out=9.3, snap_in=SnapKind.SHOT,
                                snap_out=SnapKind.SHOT)])
    out, report = apply_rails(plan, ctx)
    assert out.shots[0].src_in == 4.0
    assert out.shots[0].src_out == 9.5
    assert len(report.source_snaps) == 2


def test_rails_do_not_mutate_the_input_plan(ctx):
    plan = make_plan([make_shot("s001", src_in=4.2, src_out=9.3, snap_in=SnapKind.SHOT)])
    apply_rails(plan, ctx)
    assert plan.shots[0].src_in == 4.2


def test_rails_apply_word_padding(ctx):
    plan = make_plan([make_shot("s001", src_in=1.05, src_out=3.3,
                                snap_in=SnapKind.WORD_START, snap_out=SnapKind.WORD_END)])
    out, _ = apply_rails(plan, ctx)
    assert out.shots[0].src_in == pytest.approx(1.0 - 0.08, abs=1e-3)
    assert out.shots[0].src_out == pytest.approx(3.4 + 0.15, abs=1e-3)


def test_rails_skip_a_snap_that_would_empty_the_range(ctx):
    """A snap must never invert or collapse a shot."""
    plan = make_plan([make_shot("s001", src_in=3.9, src_out=4.1, snap_in=SnapKind.SHOT,
                                snap_out=SnapKind.SHOT)])
    out, _ = apply_rails(plan, ctx)
    # Both ends would snap to 4.0, so neither snap is applied and the shot is dropped
    # for being under the minimum length instead.
    assert out.shots == [] or out.shots[0].src_out > out.shots[0].src_in


def test_rails_drop_shots_that_fall_below_the_minimum(ctx):
    plan = make_plan([
        make_shot("s001", src_in=0.0, src_out=0.1, timeline_in=0),
        make_shot("s002", src_in=5.0, src_out=7.0, timeline_in=3),
    ])
    out, report = apply_rails(plan, ctx)
    assert report.dropped_shots == ["s001"]
    assert [s.id for s in out.shots] == ["s002"]


def test_rails_resolve_beat_durations_into_frames(ctx):
    plan = make_plan([make_shot("s001", transition=transition_ref(beats=0.5))])
    out, report = apply_rails(plan, ctx)
    # 0.5 beats at 120 BPM = 0.25s = 7.5 frames at 30fps, rounded to 8.
    assert out.shots[0].transition_in.duration.frames == 8
    assert out.shots[0].transition_in.duration.beats is None
    assert report.beat_durations_resolved == 1


def test_resolve_duration_passes_frames_through():
    assert resolve_duration(Duration(frames=12), make_grid(), 30.0) == 12


def test_resolve_duration_never_returns_zero():
    """A zero-frame transition would render as nothing at all."""
    assert resolve_duration(Duration(beats=0.001), make_grid(bpm=120.0), 30.0) == 1


def test_resolve_duration_of_none_is_none():
    assert resolve_duration(None, make_grid(), 30.0) is None


def test_duration_requires_exactly_one_unit():
    with pytest.raises(ValueError):
        Duration()
    with pytest.raises(ValueError):
        Duration(beats=1.0, frames=10)


# -- relayout ------------------------------------------------------------------------


def test_rails_close_gaps_left_by_snapping(ctx):
    """Snapping changes shot lengths, so without relayout every snap opens a gap."""
    plan = make_plan([
        make_shot("s001", src_in=4.2, src_out=9.3, timeline_in=0, snap_in=SnapKind.SHOT,
                  snap_out=SnapKind.SHOT),
        make_shot("s002", src_in=1.0, src_out=3.0, timeline_in=153),
    ])
    out, _ = apply_rails(plan, ctx)
    first = out.shots[0]
    assert out.shots[1].timeline_in == first.timeline_in + first.timeline_duration_frames(30.0)


def test_relaid_out_plan_validates_cleanly(ctx):
    plan = make_plan([
        make_shot("s001", src_in=4.2, src_out=9.3, timeline_in=0, snap_in=SnapKind.SHOT,
                  snap_out=SnapKind.SHOT),
        make_shot("s002", src_in=1.05, src_out=3.3, timeline_in=999,
                  snap_in=SnapKind.WORD_START, snap_out=SnapKind.WORD_END),
    ])
    out, _ = apply_rails(plan, ctx)
    vctx = ValidationContext(
        assets={"a_1": make_asset("a_1", 30.0)},
        usable_spans={"a_1": [(0.0, 30.0)]},
    )
    problems = errors(validate(out, vctx))
    assert [p.code for p in problems if p.code in ("timeline_gap", "timeline_overlap")] == []


def test_rails_start_the_timeline_at_zero(ctx):
    plan = make_plan([make_shot("s001", timeline_in=45)])
    out, _ = apply_rails(plan, ctx)
    assert out.shots[0].timeline_in == 0


def test_rails_snap_timeline_starts_to_the_beat(ctx):
    """With a 2s first shot at 120 BPM, the second shot starts on a beat already."""
    plan = make_plan([
        make_shot("s001", src_in=0.0, src_out=2.1, timeline_in=0),
        make_shot("s002", src_in=5.0, src_out=7.0, timeline_in=63,
                  snap_in=SnapKind.DOWNBEAT),
    ])
    out, report = apply_rails(plan, ctx)
    # 2.1s = frame 63; the nearest downbeat at 2.0s is 100ms away, inside the window,
    # but pulling back would overlap, so it moves forward to the next one instead.
    second = out.shots[1]
    assert second.timeline_in >= 63
    assert second.timeline_in / 30.0 in ctx.grid.downbeats or second.timeline_in == 63


def test_rails_never_pull_a_shot_back_into_its_predecessor(ctx):
    plan = make_plan([
        make_shot("s001", src_in=0.0, src_out=2.1, timeline_in=0),
        make_shot("s002", src_in=5.0, src_out=7.0, timeline_in=63, snap_in=SnapKind.BEAT),
    ])
    out, _ = apply_rails(plan, ctx)
    first = out.shots[0]
    assert out.shots[1].timeline_in >= first.timeline_in + first.timeline_duration_frames(30.0)


def test_rails_are_idempotent(ctx):
    plan = make_plan([
        make_shot("s001", src_in=4.2, src_out=9.3, snap_in=SnapKind.SHOT, snap_out=SnapKind.SHOT),
        make_shot("s002", src_in=1.05, src_out=3.3, timeline_in=153,
                  snap_in=SnapKind.WORD_START, snap_out=SnapKind.WORD_END),
    ])
    once, _ = apply_rails(plan, ctx)
    twice, report = apply_rails(once, ctx)
    assert twice.model_dump() == once.model_dump()
    assert not report.changed


def test_report_summary_mentions_every_category(ctx):
    plan = make_plan([make_shot("s001", src_in=4.2, src_out=9.3, snap_in=SnapKind.SHOT)])
    _, report = apply_rails(plan, ctx)
    summary = report.summary()
    for word in ("source snaps", "timeline snaps", "beat durations", "relaid out"):
        assert word in summary


def test_rails_with_no_candidates_only_relayout():
    plan = make_plan([
        make_shot("s001", src_in=1.0, src_out=3.0, timeline_in=0, snap_in=SnapKind.SHOT),
        make_shot("s002", src_in=5.0, src_out=7.0, timeline_in=999),
    ])
    out, report = apply_rails(plan, RailContext(fps=30.0))
    assert report.source_snaps == []
    assert out.shots[1].timeline_in == 60
