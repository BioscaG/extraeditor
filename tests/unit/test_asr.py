"""ASR: word timings where VAD found speech, and silence everywhere else (§9, §28)."""

from __future__ import annotations

import pytest

from montaje.analysis.local.asr import AsrAnalyzer, AsrUnavailable, _confident, transcript_text
from montaje.analysis.local.vad import VadAnalyzer
from montaje.analysis.runner import SERIAL_ANALYZERS, _asr_available
from montaje.config import AsrConfig
from montaje.index.store import Store


def _asset(ws, name: str):
    with Store(ws.db_path) as store:
        return next(a for a in store.list_assets() if a.path.name == name)


@pytest.fixture()
def with_vad(ingested):
    """VAD events persisted, since ASR reads them from the store."""
    with Store(ingested.db_path) as store:
        for asset in store.list_assets():
            events = VadAnalyzer().run(asset, ingested)
            store.replace_events(asset.asset_id, "vad@1", events)
    return ingested


requires_backend = pytest.mark.skipif(
    not _asr_available(), reason="no ASR backend installed"
)


def test_asr_is_marked_serial():
    """MLX inference from a thread pool kills the process without raising."""
    assert "asr" in SERIAL_ANALYZERS


def test_no_vad_speech_means_no_transcription(ingested):
    """Whisper invents sentences over music; gating on VAD is the guard (§28)."""
    asset = _asset(ingested, "plain.mp4")
    with Store(ingested.db_path) as store:
        store.replace_events(asset.asset_id, "vad@1", [])
    assert AsrAnalyzer().run(asset, ingested) == []


def test_missing_audio_means_no_transcription(ingested):
    asset = _asset(ingested, "plain.mp4")  # silent clip, no 16k wav
    assert AsrAnalyzer().run(asset, ingested) == []


@requires_backend
def test_words_are_transcribed_with_timings(with_vad, truth):
    if "skipped" in truth.get("speech.mp4", {}):
        pytest.skip(truth["speech.mp4"]["skipped"])
    asset = _asset(with_vad, "speech.mp4")
    events = AsrAnalyzer(language="en").run(asset, with_vad)
    words = [e for e in events if e.type == "word"]
    assert words, [e.type for e in events]
    assert all(e.t1 > e.t0 for e in words)
    assert all(e.data.get("text") for e in words)


@requires_backend
def test_words_are_ordered_and_inside_the_clip(with_vad, truth):
    if "skipped" in truth.get("speech.mp4", {}):
        pytest.skip(truth["speech.mp4"]["skipped"])
    asset = _asset(with_vad, "speech.mp4")
    words = [e for e in AsrAnalyzer(language="en").run(asset, with_vad) if e.type == "word"]
    assert [w.t0 for w in words] == sorted(w.t0 for w in words)
    assert all(0 <= w.t0 and w.t1 <= asset.duration_s + 0.5 for w in words)


@requires_backend
def test_transcript_matches_the_spoken_line(with_vad, truth):
    if "skipped" in truth.get("speech.mp4", {}):
        pytest.skip(truth["speech.mp4"]["skipped"])
    asset = _asset(with_vad, "speech.mp4")
    text = transcript_text(AsrAnalyzer(language="en").run(asset, with_vad)).lower()
    # The fixture says "This is a test of the montaje speech detector".
    assert "test" in text
    assert "speech" in text or "detector" in text


@requires_backend
def test_a_segment_event_accompanies_the_words(with_vad, truth):
    if "skipped" in truth.get("speech.mp4", {}):
        pytest.skip(truth["speech.mp4"]["skipped"])
    asset = _asset(with_vad, "speech.mp4")
    events = AsrAnalyzer(language="en").run(asset, with_vad)
    segments = [e for e in events if e.type == "segment"]
    assert segments
    assert segments[0].data["text"]
    assert segments[0].data["backend"] in ("mlx-whisper", "faster-whisper")


def test_confidence_gate_rejects_probable_non_speech():
    cfg = AsrConfig(no_speech_prob_max=0.6, min_avg_logprob=-1.0)
    assert _confident({"no_speech_prob": 0.1, "avg_logprob": -0.4}, cfg)
    assert not _confident({"no_speech_prob": 0.9, "avg_logprob": -0.4}, cfg)


def test_confidence_gate_rejects_low_logprob():
    cfg = AsrConfig(no_speech_prob_max=0.6, min_avg_logprob=-1.0)
    assert not _confident({"no_speech_prob": 0.1, "avg_logprob": -2.5}, cfg)


def test_confidence_gate_accepts_missing_fields():
    """A backend that omits the metrics must not have all its output discarded."""
    assert _confident({}, AsrConfig())


def test_params_include_the_confidence_thresholds():
    """They belong in the cache key: changing a threshold changes the output."""
    params = AsrAnalyzer(AsrConfig(no_speech_prob_max=0.3)).params()
    assert params["no_speech_prob_max"] == 0.3
    assert AsrAnalyzer(AsrConfig(no_speech_prob_max=0.9)).params() != params


def test_transcript_of_nothing_is_empty():
    assert transcript_text([]) == ""


def test_unavailable_backend_raises_a_clear_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_whisper(name, *args, **kwargs):
        if name in ("mlx_whisper", "faster_whisper"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_whisper)
    from montaje.analysis.local.asr import _load_backend

    with pytest.raises(AsrUnavailable, match="mlx-whisper"):
        _load_backend("base")
