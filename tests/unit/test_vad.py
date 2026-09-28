from __future__ import annotations

import pytest

from montaje.analysis.local.vad import VadAnalyzer
from montaje.index.store import Store


def _asset_by_name(ws, name: str):
    with Store(ws.db_path) as store:
        for a in store.list_assets():
            if a.path.name == name:
                return a
    raise AssertionError(f"{name} not ingested")


def test_vad_finds_speech_at_the_known_offset(ingested, truth):
    if "skipped" in truth.get("speech.mp4", {}):
        pytest.skip(truth["speech.mp4"]["skipped"])
    asset = _asset_by_name(ingested, "speech.mp4")
    segments = VadAnalyzer().run(asset, ingested)
    assert segments, "expected a speech segment"
    at = truth["speech.mp4"]["speech_at_s"]
    covering = [s for s in segments if s.t0 <= at + 1.0 and s.t1 >= at]
    assert covering, [(s.t0, s.t1) for s in segments]


def test_vad_does_not_start_before_the_speech(ingested, truth):
    if "skipped" in truth.get("speech.mp4", {}):
        pytest.skip(truth["speech.mp4"]["skipped"])
    asset = _asset_by_name(ingested, "speech.mp4")
    segments = VadAnalyzer().run(asset, ingested)
    at = truth["speech.mp4"]["speech_at_s"]
    # The first 2 s are a quiet tone bed; VAD must not claim speech there.
    assert min(s.t0 for s in segments) >= at - 0.6


def test_vad_is_silent_on_music_only_footage(ingested):
    """§9: ASR must stay silent on music-only clips, which depends on VAD being quiet."""
    asset = _asset_by_name(ingested, "sync_a.mp4")
    segments = VadAnalyzer().run(asset, ingested)
    total = sum(s.t1 - s.t0 for s in segments)
    assert total < asset.duration_s * 0.35, [(s.t0, s.t1) for s in segments]


def test_vad_respects_min_speech_length(ingested):
    asset = _asset_by_name(ingested, "speech.mp4")
    segments = VadAnalyzer(min_speech_s=1.0).run(asset, ingested)
    assert all(s.t1 - s.t0 >= 1.0 for s in segments)


def test_vad_segments_are_ordered_and_disjoint(ingested):
    asset = _asset_by_name(ingested, "speech.mp4")
    segments = VadAnalyzer().run(asset, ingested)
    for a, b in zip(segments, segments[1:], strict=False):
        assert a.t1 <= b.t0
