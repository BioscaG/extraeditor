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
    """Frame 4 is 0.133s: 117ms from the beat and 117ms from the half-beat either side,
    which is as far from the grid as a 30fps timeline can get at 120 BPM."""
    shots = [
        make_shot("s001", src_in=0.0, src_out=0.6, timeline_in=0),
        make_shot("s002", src_in=2.0, src_out=2.6, timeline_in=19),
        make_shot("s003", src_in=4.0, src_out=4.6, timeline_in=34),
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


def test_identical_shot_lengths_are_flagged():
    """§16.4, measured where it actually shows: every shot the same length reads as generated.

    This replaced a check on a *fully on-grid* edit being mechanical. Cutting on the grid is
    what makes an edit feel cut to the music; it is the constant cut rate that gives it away.
    """
    report = build_rhythm_report(on_grid_plan(n=8), structure=structure_for(), style=Style(name="s"))
    assert report.length_variety == 0.0
    assert any("shot lengths barely vary" in p for p in report.problems)


def test_varied_shot_lengths_are_not_flagged():
    shots, cursor = [], 0
    for i, frames in enumerate((15, 8, 30, 8, 22, 15, 8, 30)):
        shots.append(make_shot(f"s{i:03d}", src_in=i * 2.0, src_out=i * 2.0 + frames / 30,
                               timeline_in=cursor))
        cursor += frames
    report = build_rhythm_report(make_plan(shots), structure=structure_for(), style=Style(name="s"))
    assert report.length_variety > 0.2
    assert not any("shot lengths barely vary" in p for p in report.problems)


def test_a_loose_edit_is_flagged():
    """Cuts far from any half-beat position. At 120 BPM those sit 250ms apart, so the
    window is ~62ms and a cut must miss by more than that to count as loose."""
    style = Style(name="s", pacing=Pacing(on_grid_ratio=0.9))
    # 0.125s past every beat: the maximum possible distance from the half-beat grid.
    shots = [
        make_shot(f"s{i:03d}", src_in=i, src_out=i + 0.6, timeline_in=4 + i * 15)
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


# -- shot lengths are whole half-beats ---------------------------------------------------


def test_the_variation_pattern_is_whole_half_beats():
    """A shot lasting a fraction of a grid step puts the cut after it off the grid by
    construction, however well the rest of the edit is built."""
    from montaje.plan.build import _variation_half_beats

    for target in (1, 2, 3, 4, 6, 8):
        pattern = _variation_half_beats(target)
        assert all(isinstance(n, int) and n >= 1 for n in pattern)


def test_the_variation_pattern_keeps_the_target_mean_exactly():
    """Rounding to a musical unit biases the mean up, and the one-half-beat floor can only
    push it up further: uncorrected, a drop targeting 1.0 beats measured 1.3."""
    from montaje.plan.build import _variation_half_beats

    for target in (1, 2, 3, 4, 5, 8, 16):
        pattern = _variation_half_beats(target)
        assert sum(pattern) == target * len(pattern)


def test_the_variation_pattern_actually_varies():
    from montaje.plan.build import _variation_half_beats

    assert len(set(_variation_half_beats(4))) > 2


def test_the_shortest_target_still_produces_some_variation():
    """At one half-beat the floor bites hardest; a constant pattern would read as a metronome."""
    from montaje.plan.build import _variation_half_beats

    assert len(set(_variation_half_beats(2))) > 1


def test_the_minimum_shot_length_is_itself_a_whole_half_beat():
    """Clamping to 0.35s afterwards put shots off a 0.25s grid — the exact unmusical length
    the quantization exists to prevent."""
    from montaje.plan.build import _variation_half_beats

    pattern = _variation_half_beats(2, min_half_beats=2)
    assert min(pattern) == 2


def test_a_floor_longer_than_the_target_does_not_hang():
    """Every shot is already as short as allowed; the mean cannot be met and that is honest."""
    from montaje.plan.build import _variation_half_beats

    pattern = _variation_half_beats(1, min_half_beats=4)
    assert min(pattern) == 4


def test_shot_length_is_a_whole_number_of_half_beats():
    from montaje.plan.build import _shot_length_s

    style = Style(name="s", pacing=Pacing(avg_shot_beats={"drop": 1.0}))
    half_beat_s = 30.0 / 120.0
    for index in range(10):
        length = _shot_length_s("drop", style, 120.0, index)
        assert (length / half_beat_s) == pytest.approx(round(length / half_beat_s))


# -- footage coverage --------------------------------------------------------------------


def _many_candidates(n: int = 40):
    from montaje.plan.build import Candidate

    # All equally good, so nothing but the decay can separate them.
    return [Candidate(f"a_{i}", 0.0, 20.0, motion=3.0) for i in range(n)]


def test_repeated_use_of_one_asset_decays_its_score():
    """One clip supplied 15 of 108 shots in a real edit while 52 usable clips went unused."""
    from montaje.plan.build import _pick

    candidates = _many_candidates(4)
    used: dict[str, list[tuple[float, float]]] = {}
    picked = []
    for _ in range(8):
        chosen, t0, t1 = _pick(candidates, used, want_energy=0.5, length_s=1.0)
        used.setdefault(chosen.asset_id, []).append((t0, t1))
        picked.append(chosen.asset_id)
    # Four clips, eight shots: every clip should carry two, not one clip carrying eight.
    assert len(set(picked)) == 4
    assert max(picked.count(a) for a in set(picked)) <= 3


def test_an_unused_clip_beats_an_already_used_better_one():
    from montaje.plan.build import Candidate, _pick

    better = Candidate("a_good", 0.0, 20.0, motion=6.0)
    fresh = Candidate("a_fresh", 0.0, 20.0, motion=3.0)
    used = {"a_good": [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]}
    chosen, _, _ = _pick([better, fresh], used, want_energy=0.9, length_s=1.0)
    assert chosen.asset_id == "a_fresh"


def test_an_outstanding_clip_can_still_carry_a_second_shot():
    """Not a hard cap: sometimes one clip really is the best thing in the shoot.

    What buys a repeat is a *logged moment*, which outweighs the decay several times over.
    Technical merit alone does not, and should not: a fresh ordinary clip tells the viewer
    something new, where a second look at a sharp clip of nothing does not.
    """
    from montaje.plan.build import Candidate, _pick

    hero = Candidate("a_hero", 0.0, 20.0, motion=3.0, moment_score=0.9, moment_kind="hero")
    ordinary = Candidate("a_dull", 0.0, 20.0, motion=3.0)
    used = {"a_hero": [(0.0, 1.0)]}
    chosen, _, _ = _pick([hero, ordinary], used, want_energy=0.5, length_s=1.0)
    assert chosen.asset_id == "a_hero"


def test_technical_merit_alone_does_not_buy_a_repeat():
    from montaje.plan.build import Candidate, _pick

    sharp = Candidate("a_sharp", 0.0, 20.0, motion=3.0, quality=1.0)
    fresh = Candidate("a_fresh", 0.0, 20.0, motion=3.0, quality=0.6)
    chosen, _, _ = _pick([sharp, fresh], {"a_sharp": [(0.0, 1.0)]},
                         want_energy=0.5, length_s=1.0)
    assert chosen.asset_id == "a_fresh"


# -- unusable footage is not offered -----------------------------------------------------


def _inputs_for(events: list) -> object:
    from montaje.models.events import Event  # noqa: F401
    from montaje.plan.build import BuildInputs
    from tests.unit.conftest_plan import make_asset

    return BuildInputs(assets={"a_1": make_asset("a_1", 30.0)}, events={"a_1": events})


def test_a_clip_the_analyzer_found_unusable_is_not_offered():
    """A night clip measured detail 1.4 against a threshold of 4 and 70% of pixels near
    black, had no usable span, and still supplied a shot — half a second of black in a
    finished render, which only the auto-checks noticed."""
    from montaje.models.events import Event
    from montaje.plan.build import collect_candidates

    metrics = [
        Event(asset_id="a_1", analyzer="quality@1", type="metrics", t0=float(i), t1=float(i + 1),
              data={"detail": 1.4, "clip_hi": 0.0, "clip_lo": 0.74, "shake": 4.3})
        for i in range(30)
    ]
    assert collect_candidates(_inputs_for(metrics)) == []


def test_an_unanalyzed_clip_is_still_offered_whole():
    """No analysis is not the same as analysis finding nothing: an un-analyzed project must
    still produce an edit."""
    from montaje.plan.build import collect_candidates

    assert collect_candidates(_inputs_for([])) != []


def test_a_clip_with_some_usable_footage_offers_only_that():
    from montaje.models.events import Event
    from montaje.plan.build import collect_candidates

    events = [
        Event(asset_id="a_1", analyzer="quality@1", type="metrics", t0=float(i), t1=float(i + 1),
              data={"detail": 8.0, "clip_hi": 0.0, "clip_lo": 0.0, "shake": 2.0})
        for i in range(30)
    ] + [
        Event(asset_id="a_1", analyzer="quality@1", type="usable", t0=10.0, t1=20.0),
    ]
    spans = [(c.t0, c.t1) for c in collect_candidates(_inputs_for(events))]
    assert spans == [(10.0, 20.0)]
