"""Snapping rails: the LLM picks moments, code picks frames (§18.1)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from montaje.models.editplan import SnapKind
from montaje.models.events import Event
from montaje.plan.snap import SnapCandidates, nearest, snap_source_time, snap_timeline_frame

CANDS = SnapCandidates(
    shots=[0.0, 4.0, 9.5],
    word_starts=[1.0, 3.0],
    word_ends=[2.5, 6.0],
    silences=[7.0],
    reveals=[0.8],
    covers=[9.0],
)


def test_nearest_picks_the_closest_value():
    assert nearest([0.0, 1.0, 2.0], 1.4) == 1.0
    assert nearest([0.0, 1.0, 2.0], 1.6) == 2.0


def test_nearest_respects_max_delta():
    assert nearest([0.0, 5.0], 2.4, max_delta=1.0) is None
    assert nearest([0.0, 5.0], 0.5, max_delta=1.0) == 0.0


def test_nearest_on_empty_is_none():
    assert nearest([], 1.0) is None


def test_snap_to_shot_boundary():
    assert snap_source_time(4.2, SnapKind.SHOT, CANDS) == 4.0


def test_snap_to_word_start_applies_preroll():
    """A cut exactly on the word start clips its attack."""
    got = snap_source_time(1.1, SnapKind.WORD_START, CANDS, word_preroll_s=0.08)
    assert got == pytest.approx(1.0 - 0.08)


def test_snap_to_word_end_applies_postroll():
    got = snap_source_time(2.4, SnapKind.WORD_END, CANDS, word_postroll_s=0.15)
    assert got == pytest.approx(2.5 + 0.15)


def test_word_preroll_never_goes_negative():
    cands = SnapCandidates(word_starts=[0.02])
    assert snap_source_time(0.03, SnapKind.WORD_START, cands, word_preroll_s=0.08) == 0.0


def test_snap_to_occlusion_reveal():
    assert snap_source_time(0.9, SnapKind.OCCLUSION_REVEAL, CANDS) == 0.8


def test_snap_to_occlusion_cover():
    assert snap_source_time(9.2, SnapKind.OCCLUSION_COVER, CANDS) == 9.0


def test_snap_none_is_identity():
    assert snap_source_time(3.7, SnapKind.NONE, CANDS) == 3.7


def test_snap_beyond_max_delta_is_left_alone():
    """A boundary too far away is not a boundary the agent meant."""
    assert snap_source_time(20.0, SnapKind.SHOT, CANDS, max_delta_s=0.5) == 20.0


def test_timeline_snap_kinds_are_identity_in_source_domain():
    for kind in (SnapKind.BEAT, SnapKind.DOWNBEAT, SnapKind.BAR):
        assert snap_source_time(3.7, kind, CANDS) == 3.7


def test_snap_with_no_candidates_is_identity():
    assert snap_source_time(3.7, SnapKind.WORD_START, SnapCandidates()) == 3.7


# -- timeline snapping -------------------------------------------------------------

BEATS = [0.0, 0.5, 1.0, 1.5, 2.0]
DOWNBEATS = [0.0, 2.0]


def test_snap_frame_to_beat():
    # frame 17 at 30fps = 0.567s, nearest beat 0.5s = frame 15
    assert snap_timeline_frame(17, SnapKind.BEAT, 30.0, BEATS, DOWNBEATS) == 15


def test_snap_frame_to_downbeat():
    # frame 62 = 2.067s, nearest downbeat 2.0s = frame 60
    assert snap_timeline_frame(62, SnapKind.DOWNBEAT, 30.0, BEATS, DOWNBEATS) == 60


def test_snap_outside_window_is_left_alone():
    """The agent may deliberately cut off the grid (§16.4); the rail must not override that."""
    # frame 22 = 0.733s is 233ms from the nearest beat, beyond the 120ms window.
    assert snap_timeline_frame(22, SnapKind.BEAT, 30.0, BEATS, DOWNBEATS, window_ms=120.0) == 22


def test_snap_window_is_configurable():
    # frame 22 = 0.733s: 233ms from beat 0.5s, 267ms from beat 1.0s. A 300ms window
    # admits both, so it snaps to the closer one (0.5s = frame 15).
    assert snap_timeline_frame(22, SnapKind.BEAT, 30.0, BEATS, DOWNBEATS, window_ms=300.0) == 15


def test_bar_falls_back_to_downbeats():
    assert snap_timeline_frame(62, SnapKind.BAR, 30.0, BEATS, DOWNBEATS) == 60


def test_snap_frame_without_grid_is_identity():
    assert snap_timeline_frame(17, SnapKind.BEAT, 30.0, [], []) == 17


# -- candidate extraction ------------------------------------------------------------


def test_candidates_from_events_uses_occlusion_edges_not_centres():
    """The usable frame is where the occlusion ends, not the middle of the event."""
    events = [
        Event(asset_id="a", analyzer="occlusion@1", type="reveal", t0=0.6, t1=1.0),
        Event(asset_id="a", analyzer="occlusion@1", type="cover", t0=9.0, t1=9.4),
    ]
    c = SnapCandidates.from_events(events)
    assert c.reveals == [1.0]
    assert c.covers == [9.0]


def test_candidates_from_events_collects_shot_edges():
    events = [
        Event(asset_id="a", analyzer="shots@1", type="shot", t0=0.0, t1=4.0),
        Event(asset_id="a", analyzer="shots@1", type="shot", t0=4.0, t1=9.0),
    ]
    assert SnapCandidates.from_events(events).shots == [0.0, 4.0, 9.0]


def test_vad_provides_speech_boundaries_when_asr_is_absent():
    events = [Event(asset_id="a", analyzer="vad@1", type="speech", t0=2.0, t1=5.0)]
    c = SnapCandidates.from_events(events)
    assert c.word_starts == [2.0]
    assert c.word_ends == [5.0]


def test_asr_words_take_precedence_over_vad():
    events = [
        Event(asset_id="a", analyzer="asr@1", type="word", t0=2.1, t1=2.4),
        Event(asset_id="a", analyzer="vad@1", type="speech", t0=2.0, t1=5.0),
    ]
    c = SnapCandidates.from_events(events)
    assert 2.1 in c.word_starts
    assert 2.0 not in c.word_starts


def test_candidates_are_sorted_and_deduplicated():
    events = [
        Event(asset_id="a", analyzer="shots@1", type="shot", t0=4.0, t1=9.0),
        Event(asset_id="a", analyzer="shots@1", type="shot", t0=0.0, t1=4.0),
    ]
    shots = SnapCandidates.from_events(events).shots
    assert shots == sorted(shots) == [0.0, 4.0, 9.0]


# -- properties ----------------------------------------------------------------------


@given(t=st.floats(min_value=0.0, max_value=100.0, allow_nan=False))
def test_snapping_is_idempotent(t: float):
    """Snapping a snapped time must not move it again."""
    once = snap_source_time(t, SnapKind.SHOT, CANDS)
    twice = snap_source_time(once, SnapKind.SHOT, CANDS)
    assert once == pytest.approx(twice)


@given(t=st.floats(min_value=0.0, max_value=100.0, allow_nan=False))
def test_snapping_never_moves_further_than_max_delta(t: float):
    got = snap_source_time(t, SnapKind.SHOT, CANDS, max_delta_s=0.5)
    assert abs(got - t) <= 0.5 + 1e-9


@given(frame=st.integers(min_value=0, max_value=10_000))
def test_timeline_snapping_stays_within_the_window(frame: int):
    got = snap_timeline_frame(frame, SnapKind.BEAT, 30.0, BEATS, DOWNBEATS, window_ms=120.0)
    assert abs(got - frame) <= round(0.120 * 30.0) + 1


@given(frame=st.integers(min_value=0, max_value=10_000))
def test_timeline_snapping_is_idempotent(frame: int):
    once = snap_timeline_frame(frame, SnapKind.BEAT, 30.0, BEATS, DOWNBEATS)
    assert snap_timeline_frame(once, SnapKind.BEAT, 30.0, BEATS, DOWNBEATS) == once
