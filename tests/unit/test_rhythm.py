"""Rhythm report: where rhythm is guaranteed, because the critic cannot see it (§18.3)."""

from __future__ import annotations

import pytest

from montaje.models.editplan import Section, Sfx
from montaje.models.events import Event
from montaje.models.style import Pacing, SfxPolicy, Style, TransitionPolicy
from montaje.music.structure import MusicStructure
from montaje.music.structure import Section as MusicSection
from montaje.plan.rhythm import build_rhythm_report
from tests.unit.conftest_plan import make_grid, make_plan, make_shot, transition_ref

FPS = 30.0


def on_grid_plan(n: int = 8, shot_frames: int = 15) -> object:
    """n shots of exactly half a beat at 120 BPM, so every cut is on the grid."""
    shots = [
        make_shot(f"s{i:03d}", src_in=i * 2.0, src_out=i * 2.0 + shot_frames / FPS,
                  timeline_in=i * shot_frames, section="intro")
        for i in range(n)
    ]
    return make_plan(shots, sections=[Section(id="intro", from_frame=0, to_frame=n * shot_frames,
                                              music_section="intro")])


def structure_for(duration: float = 60.0, energy: list[float] | None = None) -> MusicStructure:
    grid = make_grid(bpm=120.0, duration=duration)
    return MusicStructure(
        grid=grid,
        sections=[MusicSection(id="sec00", t0=0.0, t1=duration, role="intro", energy=0.5)],
        energy_curve=energy or [0.5] * int(duration),
        duration_s=duration,
    )


def test_on_grid_ratio_is_one_when_every_cut_is_on_a_beat():
    report = build_rhythm_report(on_grid_plan(), structure=structure_for())
    assert report.on_grid_ratio == 1.0


def test_off_grid_cuts_lower_the_ratio():
    shots = [
        make_shot("s001", src_in=0.0, src_out=0.6, timeline_in=0),
        make_shot("s002", src_in=2.0, src_out=2.6, timeline_in=18),
        make_shot("s003", src_in=4.0, src_out=4.6, timeline_in=36),
    ]
    report = build_rhythm_report(make_plan(shots), structure=structure_for())
    assert report.on_grid_ratio < 1.0


def test_beat_offsets_are_recorded_for_every_cut_plus_the_final_out():
    plan = on_grid_plan(n=4)
    report = build_rhythm_report(plan, structure=structure_for())
    assert len(report.beat_offsets_ms) == 5


def test_offset_histogram_buckets_by_magnitude():
    report = build_rhythm_report(on_grid_plan(), structure=structure_for())
    hist = report.offset_histogram()
    assert hist["≤20ms"] == len(report.beat_offsets_ms)
    assert sum(hist.values()) == len(report.beat_offsets_ms)


def test_no_grid_means_no_offsets():
    report = build_rhythm_report(on_grid_plan())
    assert report.beat_offsets_ms == []
    assert report.on_grid_ratio == 0.0


# -- density ------------------------------------------------------------------------


def test_transition_and_sfx_density_are_per_10s():
    shots = [
        make_shot("s001", src_in=0.0, src_out=5.0, timeline_in=0),
        make_shot("s002", src_in=6.0, src_out=11.0, timeline_in=150,
                  transition=transition_ref()),
    ]
    plan = make_plan(shots)
    plan.sfx = [Sfx(id="fx1", sfx="impact.a", anchor_frame=150)]
    report = build_rhythm_report(plan, structure=structure_for())
    # 10s total, 1 transition, 1 sfx.
    assert report.transitions_per_10s == pytest.approx(1.0, abs=0.01)
    assert report.sfx_per_10s == pytest.approx(1.0, abs=0.01)


def test_transition_overuse_is_flagged():
    style = Style(name="s", transitions=TransitionPolicy(max_per_10s=1.0))
    shots = [
        make_shot(f"s{i:03d}", src_in=i, src_out=i + 0.5, timeline_in=i * 15,
                  transition=transition_ref())
        for i in range(6)
    ]
    report = build_rhythm_report(make_plan(shots), structure=structure_for(), style=style)
    assert any("transitions per 10s" in p for p in report.problems)


def test_sfx_overuse_is_flagged():
    style = Style(name="s", sfx=SfxPolicy(density_max_per_10s=1.0))
    plan = on_grid_plan()
    plan.sfx = [Sfx(id=f"fx{i}", sfx="impact.a", anchor_frame=i * 10) for i in range(10)]
    report = build_rhythm_report(plan, structure=structure_for(), style=style)
    assert any("SFX per 10s" in p for p in report.problems)


# -- pacing against the style -----------------------------------------------------------


def test_section_pacing_is_measured_in_beats():
    style = Style(name="s", pacing=Pacing(avg_shot_beats={"intro": 0.5}))
    report = build_rhythm_report(on_grid_plan(), structure=structure_for(), style=style)
    section = report.sections[0]
    # 15 frames at 30fps = 0.5s = 1 beat at 120 BPM.
    assert section.avg_shot_beats == pytest.approx(1.0, abs=0.01)
    assert section.target_shot_beats == 0.5


def test_pacing_off_target_is_flagged_with_a_direction():
    style = Style(name="s", pacing=Pacing(avg_shot_beats={"intro": 0.25}))
    report = build_rhythm_report(on_grid_plan(), structure=structure_for(), style=style)
    assert any("cut faster" in p for p in report.problems)


def test_pacing_on_target_is_not_flagged():
    style = Style(name="s", pacing=Pacing(avg_shot_beats={"intro": 1.0}))
    report = build_rhythm_report(on_grid_plan(), structure=structure_for(), style=style)
    assert report.sections[0].on_target is True
    assert not any("beats per shot" in p for p in report.problems)


def test_a_fully_on_grid_edit_is_flagged_as_mechanical():
    """§16.4: break the beat grid on purpose."""
    style = Style(name="s", pacing=Pacing(on_grid_ratio=0.75))
    report = build_rhythm_report(on_grid_plan(), structure=structure_for(), style=style)
    assert any("mechanical" in p for p in report.problems)


def test_a_loose_edit_is_flagged():
    style = Style(name="s", pacing=Pacing(on_grid_ratio=0.9))
    shots = [
        make_shot(f"s{i:03d}", src_in=i, src_out=i + 0.6, timeline_in=i * 19)
        for i in range(6)
    ]
    report = build_rhythm_report(make_plan(shots), structure=structure_for(), style=style)
    assert any("feel loose" in p for p in report.problems)


# -- energy and people --------------------------------------------------------------------


def test_visual_energy_comes_from_motion_events():
    events = {"a_1": [
        Event(asset_id="a_1", analyzer="motion@1", type="motion", t0=float(i), t1=float(i + 1),
              data={"camera_magnitude": 5.0, "subject_energy": 10.0})
        for i in range(20)
    ]}
    report = build_rhythm_report(on_grid_plan(), structure=structure_for(), events=events)
    assert report.sections[0].visual_energy == pytest.approx(6.0, abs=0.01)


def test_energy_correlation_needs_at_least_three_sections():
    report = build_rhythm_report(on_grid_plan(), structure=structure_for())
    assert report.energy_correlation is None


def test_us_share_is_reported_and_flagged_when_low():
    plan = on_grid_plan(n=5)
    report = build_rhythm_report(plan, structure=structure_for(),
                                 clip_people={"a_1": False})
    assert report.us_share == 0.0
    assert any("us in frame" in p for p in report.problems)


def test_us_share_high_is_not_flagged():
    report = build_rhythm_report(on_grid_plan(n=5), structure=structure_for(),
                                 clip_people={"a_1": True})
    assert report.us_share == 1.0
    assert not any("us in frame" in p for p in report.problems)


# -- rendering ------------------------------------------------------------------------------


def test_markdown_contains_the_headline_numbers():
    style = Style(name="s", pacing=Pacing(avg_shot_beats={"intro": 1.0}, on_grid_ratio=0.75))
    report = build_rhythm_report(on_grid_plan(), structure=structure_for(), style=style)
    text = report.to_markdown()
    assert "# Rhythm report" in text
    assert "On-grid cuts" in text
    assert "Cut-to-beat offsets" in text
    assert "| section |" in text


def test_markdown_lists_problems_when_present():
    style = Style(name="s", pacing=Pacing(avg_shot_beats={"intro": 0.25}))
    report = build_rhythm_report(on_grid_plan(), structure=structure_for(), style=style)
    assert "## Problems" in report.to_markdown()


def test_report_handles_an_empty_plan():
    report = build_rhythm_report(make_plan([]), structure=structure_for())
    assert report.shots == 0
    assert "Rhythm report" in report.to_markdown()
