"""Structure detection and fit-to-duration against a track with a known bar grid."""

from __future__ import annotations

import pytest

from montaje.music.beats import analyze_beats
from montaje.music.edit import PHRASE_BAR_OPTIONS, fit_to_duration, validate_joins
from montaje.music.structure import analyze_structure


@pytest.fixture(scope="module")
def track(fixtures):
    return fixtures / "track.wav"


@pytest.fixture(scope="module")
def structure(track):
    return analyze_structure(track)


@pytest.fixture(scope="module")
def track_truth(truth):
    return truth["track.wav"]


def test_tempo_and_grid_on_a_real_arrangement(track, track_truth):
    grid = analyze_beats(track)
    assert grid.bpm == pytest.approx(track_truth["bpm"], rel=0.002)
    # Fitting the period (not just the phase) is what keeps a 96 s track in sync.
    for t in track_truth["beats"]:
        assert min(abs(b - t) for b in grid.beats) < 0.010, t


def test_downbeats_land_on_bar_lines(track, track_truth):
    grid = analyze_beats(track)
    for t in track_truth["downbeats"]:
        assert min(abs(d - t) for d in grid.downbeats) < 0.010, t


def test_section_count_matches(structure, track_truth):
    assert len(structure.sections) == len(track_truth["section_bars"])


def test_section_boundaries_match_the_arrangement(structure, track_truth):
    want = track_truth["section_boundaries"]
    got = [s.t0 for s in structure.sections] + [structure.sections[-1].t1]
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        assert abs(g - w) < 0.1, (got, want)


def test_section_bar_counts_match(structure, track_truth):
    assert [s.bars for s in structure.sections] == track_truth["section_bars"]


def test_roles_follow_position_and_energy(structure):
    roles = [s.role for s in structure.sections]
    assert roles[0] == "intro"
    assert roles[-1] == "outro"
    assert "drop" in roles
    # The drop must be the loudest section.
    drop = next(s for s in structure.sections if s.role == "drop")
    assert drop.energy == max(s.energy for s in structure.sections)


def test_energy_curve_is_normalized_per_second(structure, track_truth):
    assert len(structure.energy_curve) == pytest.approx(track_truth["duration_s"], abs=1)
    assert max(structure.energy_curve) == pytest.approx(1.0, abs=0.01)
    assert min(structure.energy_curve) >= 0.0


def test_section_at_finds_the_containing_section(structure):
    assert structure.section_at(5.0).role == "intro"
    assert structure.section_at(60.0).role == "drop"


def test_phrase_boundaries_are_every_n_bars(structure, track_truth):
    phrases = structure.phrase_boundaries(4)
    bar_s = track_truth["bar_s"]
    for a, b in zip(phrases, phrases[1:], strict=False):
        assert b - a == pytest.approx(4 * bar_s, abs=0.05)


# -- fit to duration -------------------------------------------------------------


@pytest.mark.parametrize("target", [40.0, 64.0, 80.0, 120.0])
def test_fit_hits_the_target_within_tolerance(structure, target):
    fit = fit_to_duration(structure, target, tolerance_s=4.0)
    assert abs(fit.total_s - target) <= 4.0, (target, fit.total_s, fit.notes)


def test_shortening_removes_whole_phrases(structure, track_truth):
    fit = fit_to_duration(structure, 40.0)
    phrase_s = fit.phrase_bars * track_truth["bar_s"]
    assert fit.removed_s > 0
    # A whole number of phrases removed: cutting mid-phrase breaks the listener's
    # bar count and is instantly audible (§14.2).
    n = fit.removed_s / phrase_s
    assert n == pytest.approx(round(n), abs=0.02), (fit.removed_s, phrase_s)


def test_lengthening_repeats_whole_phrases(structure):
    fit = fit_to_duration(structure, 130.0)
    assert fit.repeated_s > 0
    assert fit.removed_s == 0


def test_every_join_lands_on_a_downbeat(structure):
    """§18.2 hard error: music joins off the downbeat."""
    for target in (40.0, 60.0, 130.0):
        fit = fit_to_duration(structure, target)
        assert validate_joins(fit, structure) == [], (target, fit.notes)


def test_edits_are_in_timeline_order_and_contiguous(structure):
    fit = fit_to_duration(structure, 50.0)
    frames = [e.timeline_in for e in fit.edits]
    assert frames == sorted(frames)
    # Each edit starts exactly where the previous one ended on the timeline.
    fps = 30.0
    expected = 0
    for e in fit.edits:
        assert e.timeline_in == expected
        expected += round((e.src_out - e.src_in) * fps)


def test_source_ranges_stay_inside_the_track(structure, track_truth):
    fit = fit_to_duration(structure, 50.0)
    for e in fit.edits:
        assert 0 <= e.src_in < e.src_out <= track_truth["duration_s"] + 0.1


def test_first_edit_has_no_join(structure):
    fit = fit_to_duration(structure, 40.0)
    assert fit.edits[0].join is None


def test_no_edit_needed_when_target_matches(structure, track_truth):
    fit = fit_to_duration(structure, track_truth["duration_s"])
    assert fit.joins == 0
    assert fit.removed_s == 0 and fit.repeated_s == 0


def test_drop_is_preserved_when_shortening(structure):
    """The drop is the payoff the edit is built around; it must survive a cut."""
    fit = fit_to_duration(structure, 48.0)
    drop = next(s for s in structure.sections if s.role == "drop")
    kept = [(e.src_in, e.src_out) for e in fit.edits]
    drop_mid = (drop.t0 + drop.t1) / 2
    assert any(a <= drop_mid < b for a, b in kept), (drop_mid, kept)


def test_longest_phrase_grid_is_preferred(structure):
    """Removing one 16-bar phrase is less audible than four 4-bar ones."""
    fit = fit_to_duration(structure, 64.0, tolerance_s=4.0)
    assert fit.phrase_bars == max(PHRASE_BAR_OPTIONS)


def test_track_shorter_than_a_phrase_passes_through(tmp_path):
    from montaje import ffmpeg
    from montaje.music.structure import analyze_structure as an

    short = tmp_path / "short.wav"
    ffmpeg.run(["-y", "-v", "error", "-f", "lavfi", "-i",
                "sine=frequency=220:sample_rate=16000:duration=1", str(short)])
    fit = fit_to_duration(an(short), 30.0)
    assert len(fit.edits) == 1
    assert fit.edits[0].join is None
    assert fit.removed_s == 0 and fit.repeated_s == 0


def test_zero_length_track_yields_no_edits():
    from montaje.music.beats import BeatGrid
    from montaje.music.structure import MusicStructure

    empty = MusicStructure(grid=BeatGrid(bpm=120.0), duration_s=0.0)
    assert fit_to_duration(empty, 30.0).edits == []


def test_phrase_spans_cover_the_track_without_fragments(structure, track_truth):
    """A millisecond-long leading phrase rounds to zero frames and corrupts timing."""
    from montaje.music.edit import phrase_spans

    for phrase_bars in PHRASE_BAR_OPTIONS:
        spans = phrase_spans(structure, phrase_bars)
        assert spans[0][0] == 0.0
        assert spans[-1][1] == pytest.approx(structure.duration_s)
        for a, b in zip(spans, spans[1:], strict=False):
            assert a[1] == b[0]
        min_len = 0.5 * phrase_bars * track_truth["bar_s"]
        assert all(b - a >= min_len for a, b in spans), (phrase_bars, spans)
