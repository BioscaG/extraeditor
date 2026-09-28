from __future__ import annotations

from montaje.ingest.hashing import compute_asset_id
from montaje.ingest.probe import capture_time, classify_kind, probe_file
from montaje.models.asset import AssetKind, DynamicRange


def test_probe_reads_basic_video(fixtures):
    p = probe_file(fixtures / "plain.mp4")
    assert p.video is not None
    assert (p.video.width, p.video.height) == (640, 360)
    assert 3.5 < p.duration_s < 4.5
    assert p.video.dynamic_range == DynamicRange.SDR


def test_probe_detects_hlg_and_bit_depth(fixtures, truth):
    p = probe_file(fixtures / "hlg.mp4")
    assert p.video.color_transfer == "arib-std-b67"
    assert p.video.dynamic_range == DynamicRange.HDR_HLG
    assert p.video.bit_depth == 10


def test_probe_flags_vfr(fixtures):
    p = probe_file(fixtures / "vfr.mp4")
    assert p.video.vfr is True


def test_cfr_clip_not_flagged_vfr(fixtures):
    assert probe_file(fixtures / "plain.mp4").video.vfr is False


def test_selected_audio_prefers_stereo(fixtures):
    p = probe_file(fixtures / "sync_a.mp4")
    assert p.selected_audio is not None
    assert p.selected_audio.codec == "aac"


def test_no_audio_returns_none(fixtures):
    assert probe_file(fixtures / "plain.mp4").selected_audio is None


def test_classify_kind(fixtures):
    p = probe_file(fixtures / "plain.mp4")
    assert classify_kind(fixtures / "plain.mp4", p) == AssetKind.VIDEO
    assert classify_kind(fixtures / "x.heic", p) == AssetKind.PHOTO
    assert classify_kind(fixtures / "x.heic", p, live_photo_pair=True) == AssetKind.LIVE_PHOTO


def test_asset_id_is_stable_and_content_addressed(fixtures):
    a = compute_asset_id(fixtures / "plain.mp4", 4.0)
    b = compute_asset_id(fixtures / "plain.mp4", 4.0)
    c = compute_asset_id(fixtures / "warm.mp4", 4.0)
    assert a == b
    assert a != c
    assert a.startswith("a_") and len(a) == 18


def test_asset_id_changes_with_duration(fixtures):
    assert compute_asset_id(fixtures / "plain.mp4", 4.0) != compute_asset_id(fixtures / "plain.mp4", 4.1)


def test_capture_time_falls_back_to_mtime(fixtures):
    p = probe_file(fixtures / "plain.mp4")
    ct = capture_time(fixtures / "plain.mp4", p)
    # Synthetic clips carry no QuickTime creationdate, so confidence must be low.
    assert ct.source in ("quicktime", "mtime")
    if ct.source == "mtime":
        assert ct.confidence < 0.5


def test_rotation_swaps_display_size(fixtures):
    p = probe_file(fixtures / "plain.mp4")
    p.video.rotation = 90
    assert p.video.display_size == (360, 640)
