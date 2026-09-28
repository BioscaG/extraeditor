"""Analyzer assertions against the fixtures' known ground truth (§27)."""

from __future__ import annotations

import pytest

from montaje.analysis.local.audio_events import AudioEventsAnalyzer
from montaje.analysis.local.color_stats import ColorStatsAnalyzer
from montaje.analysis.local.loudness import LoudnessAnalyzer
from montaje.analysis.local.occlusion import OcclusionAnalyzer
from montaje.analysis.local.quality import QualityAnalyzer
from montaje.analysis.local.shots import ShotsAnalyzer
from montaje.index.store import Store


def _asset_by_name(ws, name: str):
    with Store(ws.db_path) as store:
        for a in store.list_assets():
            if a.path.name == name:
                return a
    raise AssertionError(f"{name} not ingested")


def test_occlusion_finds_both_dark_spans(ingested, truth):
    asset = _asset_by_name(ingested, "occlusion.mp4")
    events = OcclusionAnalyzer().run(asset, ingested)
    occluded = [e for e in events if e.type == "occluded"]
    expected = truth["occlusion.mp4"]["occluded_spans"]
    assert len(occluded) == len(expected), [(e.t0, e.t1) for e in occluded]
    for got, (want_t0, want_t1) in zip(occluded, expected, strict=True):
        assert abs(got.t0 - want_t0) < 0.4, (got.t0, want_t0)
        assert abs(got.t1 - want_t1) < 0.4, (got.t1, want_t1)


def test_occlusion_emits_reveal_at_the_head_span_end(ingested, truth):
    asset = _asset_by_name(ingested, "occlusion.mp4")
    events = OcclusionAnalyzer().run(asset, ingested)
    reveals = [e for e in events if e.type == "reveal"]
    assert reveals, "expected a reveal where the opening dark span ends"
    want = truth["occlusion.mp4"]["reveal_near_s"]
    assert min(abs((r.t0 + r.t1) / 2 - want) for r in reveals) < 0.5


def test_occlusion_emits_cover_before_the_tail_span(ingested, truth):
    asset = _asset_by_name(ingested, "occlusion.mp4")
    covers = [e for e in OcclusionAnalyzer().run(asset, ingested) if e.type == "cover"]
    assert covers
    want = truth["occlusion.mp4"]["cover_near_s"]
    assert min(abs((c.t0 + c.t1) / 2 - want) for c in covers) < 0.5


def test_occlusion_finds_nothing_in_bright_footage(ingested):
    asset = _asset_by_name(ingested, "plain.mp4")
    assert [e for e in OcclusionAnalyzer().run(asset, ingested) if e.type == "occluded"] == []


def test_occlusion_events_are_meaning_free(ingested):
    """Event types must stay semantic-free: no 'hand', no 'vlog_start' (§2.5)."""
    asset = _asset_by_name(ingested, "occlusion.mp4")
    types = {e.type for e in OcclusionAnalyzer().run(asset, ingested)}
    assert types <= {"occluded", "reveal", "cover"}


def test_quality_marks_bright_detailed_footage_usable(ingested):
    asset = _asset_by_name(ingested, "plain.mp4")
    events = QualityAnalyzer().run(asset, ingested)
    usable = [e for e in events if e.type == "usable"]
    assert usable, "testsrc2 is sharp and well exposed; expected a usable span"
    assert sum(e.t1 - e.t0 for e in usable) > asset.duration_s * 0.5


def test_quality_emits_per_second_metrics(ingested):
    asset = _asset_by_name(ingested, "plain.mp4")
    metrics = [e for e in QualityAnalyzer().run(asset, ingested) if e.type == "metrics"]
    assert len(metrics) >= 3
    assert set(metrics[0].data) == {"detail", "clip_hi", "clip_lo", "shake"}


def test_quality_excludes_occluded_spans_from_usable(ingested, truth):
    asset = _asset_by_name(ingested, "occlusion.mp4")
    usable = [e for e in QualityAnalyzer().run(asset, ingested) if e.type == "usable"]
    # The opening 0–1 s is black and featureless: it must not be offered as usable.
    assert all(not (e.t0 < 0.8) for e in usable), [(e.t0, e.t1) for e in usable]


def test_shots_covers_the_whole_timeline_without_gaps(ingested):
    asset = _asset_by_name(ingested, "plain.mp4")
    shots = ShotsAnalyzer().run(asset, ingested)
    assert shots
    assert shots[0].t0 == 0.0
    for a, b in zip(shots, shots[1:], strict=False):
        assert abs(b.t0 - a.t1) < 1e-6
    assert abs(shots[-1].t1 - asset.duration_s) < 0.2


def test_shots_respects_min_shot_length(ingested):
    asset = _asset_by_name(ingested, "occlusion.mp4")
    shots = ShotsAnalyzer(min_shot_s=0.4).run(asset, ingested)
    # Only the final shot may be shorter (it is clipped by the clip end).
    assert all(s.t1 - s.t0 >= 0.4 - 1e-6 for s in shots[:-1])


def test_audio_events_detects_speech_at_the_known_offset(ingested, truth):
    if "skipped" in truth.get("speech.mp4", {}):
        pytest.skip(truth["speech.mp4"]["skipped"])
    asset = _asset_by_name(ingested, "speech.mp4")
    events = AudioEventsAnalyzer().run(asset, ingested)
    speech = [e for e in events if e.type == "speech"]
    assert speech, [f"{e.type}@{e.t0}" for e in events]
    at = truth["speech.mp4"]["speech_at_s"]
    assert any(e.t0 <= at + 2.0 and e.t1 >= at for e in speech)


def test_audio_events_labels_are_from_the_closed_set(ingested):
    asset = _asset_by_name(ingested, "sync_a.mp4")
    types = {e.type for e in AudioEventsAnalyzer().run(asset, ingested)}
    assert types <= {"speech", "music", "crowd", "silence", "other"}


def test_loudness_measures_integrated_lufs(ingested):
    asset = _asset_by_name(ingested, "sync_a.mp4")
    events = LoudnessAnalyzer().run(asset, ingested)
    assert len(events) == 1
    assert -70.0 <= events[0].data["integrated_lufs"] <= 0.0


def test_color_stats_separates_warm_from_cool(ingested):
    """Normalization depends on this: the warm/cool pair must differ in Lab b*."""
    warm = _asset_by_name(ingested, "warm.mp4")
    cool = _asset_by_name(ingested, "cool.mp4")
    an = ColorStatsAnalyzer()
    warm_b = an.run(warm, ingested)[0].data["lab_mean"][2]
    cool_b = an.run(cool, ingested)[0].data["lab_mean"][2]
    assert warm_b > cool_b + 2.0, (warm_b, cool_b)


def test_color_stats_wb_gains_point_opposite_ways(ingested):
    an = ColorStatsAnalyzer()
    warm = an.run(_asset_by_name(ingested, "warm.mp4"), ingested)[0].data["wb_gains"]
    cool = an.run(_asset_by_name(ingested, "cool.mp4"), ingested)[0].data["wb_gains"]
    # A warm frame needs its red pulled down (gain < 1) relative to a cool one.
    assert warm[0] < cool[0]


def test_analyzer_params_are_hashable_and_stable():
    a, b = OcclusionAnalyzer(), OcclusionAnalyzer()
    assert a.params() == b.params()
    assert OcclusionAnalyzer(fps=3).params() != a.params()
