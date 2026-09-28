"""Beat tracking against click tracks whose beat positions are sample-exact."""

from __future__ import annotations

import pytest

from montaje.music.beats import BeatGrid, analyze_beats


@pytest.mark.parametrize("name,expected_bpm", [("click_120.wav", 120.0), ("click_90.wav", 90.0)])
def test_tempo_is_recovered(fixtures, name, expected_bpm):
    grid = analyze_beats(fixtures / name)
    # Parabolic interpolation of the autocorrelation peak removes frame quantization.
    assert grid.bpm == pytest.approx(expected_bpm, rel=0.005), grid.bpm


@pytest.mark.parametrize("name", ["click_120.wav", "click_90.wav"])
def test_beats_land_on_the_true_beats(fixtures, truth, name):
    grid = analyze_beats(fixtures / name)
    want = truth[name]["beats"]
    # Amplitude-based offset refinement gets this to sub-millisecond on click
    # tracks; 5 ms leaves headroom without letting the ~14 ms group-delay bias back in.
    for t in want:
        assert min(abs(b - t) for b in grid.beats) < 0.005, (t, grid.beats[:8])


def test_beat_count_matches_the_click_count(fixtures, truth):
    grid = analyze_beats(fixtures / "click_120.wav")
    assert abs(len(grid.beats) - len(truth["click_120.wav"]["beats"])) <= 1


def test_downbeats_are_every_fourth_beat(fixtures):
    grid = analyze_beats(fixtures / "click_120.wav")
    assert len(grid.downbeats) == pytest.approx(len(grid.beats) / 4, abs=1)
    for d in grid.downbeats:
        assert d in grid.beats


def test_downbeats_align_with_the_accented_clicks(fixtures, truth):
    """The fixture accents every 4th click; the tracker must find that phase."""
    grid = analyze_beats(fixtures / "click_120.wav")
    want = truth["click_120.wav"]["downbeats"]
    for t in want:
        assert min(abs(d - t) for d in grid.downbeats) < 0.005, (t, grid.downbeats[:4])


def test_confidence_is_high_on_a_click_track(fixtures):
    assert analyze_beats(fixtures / "click_120.wav").confidence > 0.3


def test_nearest_beat_snaps_both_directions():
    grid = BeatGrid(bpm=120.0, beats=[0.0, 0.5, 1.0, 1.5], downbeats=[0.0, 1.0])
    assert grid.nearest_beat(0.52) == 0.5
    assert grid.nearest_beat(0.74) == 0.5
    assert grid.nearest_beat(0.76) == 1.0


def test_nearest_downbeat_uses_downbeats_only():
    grid = BeatGrid(bpm=120.0, beats=[0.0, 0.5, 1.0, 1.5], downbeats=[0.0, 1.0])
    assert grid.nearest_downbeat(0.6) == 1.0


def test_nearest_downbeat_falls_back_to_beats_when_absent():
    grid = BeatGrid(bpm=120.0, beats=[0.0, 0.5], downbeats=[])
    assert grid.nearest_downbeat(0.4) == 0.5


def test_beat_period_and_beats_to_seconds():
    grid = BeatGrid(bpm=120.0)
    assert grid.beat_period_s == pytest.approx(0.5)
    assert grid.beats_to_seconds(0.25) == pytest.approx(0.125)


def test_empty_audio_yields_a_safe_default(tmp_path):
    silent = tmp_path / "silent.wav"
    from montaje import ffmpeg

    ffmpeg.run(["-y", "-v", "error", "-f", "lavfi", "-i",
                "anullsrc=r=16000:cl=mono:d=0.05", str(silent)])
    grid = analyze_beats(silent)
    assert grid.bpm > 0


# -- the half-beat grid ------------------------------------------------------------------


def test_half_beats_interleave_the_off_beats():
    grid = BeatGrid(bpm=120.0, beats=[0.0, 0.5, 1.0, 1.5])
    assert grid.half_beats[:6] == [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]


def test_half_beats_are_interpolated_from_measured_beats_not_the_nominal_period():
    """The off-beats must inherit the fitted grid's accuracy. A track whose real tempo is
    120.094 BPM drifts 75ms from nominal over 96s, and so would its off-beats."""
    grid = BeatGrid(bpm=120.0, beats=[0.0, 0.52, 1.06])
    assert grid.half_beats[1] == pytest.approx(0.26)
    assert grid.half_beats[3] == pytest.approx(0.79)


def test_half_beats_extend_past_the_final_beat():
    """A cut just past the last beat still needs a position to land on."""
    grid = BeatGrid(bpm=120.0, beats=[0.0, 0.5, 1.0])
    assert grid.half_beats[-1] == pytest.approx(1.25)


def test_half_beats_of_a_degenerate_grid_are_the_beats():
    assert BeatGrid(bpm=120.0, beats=[0.4]).half_beats == [0.4]
    assert BeatGrid(bpm=120.0, beats=[]).half_beats == []


def test_half_beat_period_is_half_the_beat_period():
    grid = BeatGrid(bpm=128.0)
    assert grid.half_beat_period_s == pytest.approx(grid.beat_period_s / 2)
