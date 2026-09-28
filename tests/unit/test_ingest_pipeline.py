from __future__ import annotations

from montaje.config import load_config
from montaje.index.store import Store
from montaje.ingest.hdr import available_methods, needs_tonemap, resolve_method, sdr_filter
from montaje.ingest.pipeline import run_ingest
from montaje.ingest.probe import probe_file
from montaje.ingest.report import build_report, detect_degraded
from montaje.sources.folder import FolderSource


def test_folder_source_lists_only_media(fixtures, tmp_path):
    (fixtures.parent / "notes.txt").write_text("ignore me")
    names = {a.filename for a in FolderSource(fixtures).list()}
    assert "plain.mp4" in names
    assert not any(n.endswith(".txt") for n in names)
    assert not any(n.endswith(".json") for n in names)


def test_folder_source_skips_dotfiles(fixtures):
    hidden = fixtures / ".hidden.mp4"
    hidden.write_bytes(b"")
    try:
        assert all(not a.filename.startswith(".") for a in FolderSource(fixtures).list())
    finally:
        hidden.unlink()


def test_folder_source_pairs_live_photos(tmp_path, fixtures):
    d = tmp_path / "lp"
    d.mkdir()
    (d / "IMG_1.JPG").write_bytes(b"x")
    (d / "IMG_1.MOV").write_bytes(b"y")
    by_name = {a.filename: a for a in FolderSource(d).list()}
    assert by_name["IMG_1.JPG"].extra["live_photo_pair_uid"].endswith("IMG_1.MOV")
    assert by_name["IMG_1.MOV"].extra["live_photo_pair_uid"].endswith("IMG_1.JPG")


def test_ingest_produces_proxy_audio_and_thumbs(ingested):
    with Store(ingested.db_path) as store:
        assets = {a.path.name: a for a in store.list_assets()}
    a = assets["speech.mp4"]
    assert ingested.proxy_path(a.asset_id).exists()
    assert ingested.audio_16k_path(a.asset_id).exists()
    assert ingested.audio_48k_path(a.asset_id).exists()
    assert list(ingested.thumbs_dir(a.asset_id).glob("*.jpg"))


def test_ingest_proxy_is_sdr_at_configured_short_edge(ingested):
    with Store(ingested.db_path) as store:
        a = next(x for x in store.list_assets() if x.path.name == "hlg.mp4")
    p = probe_file(ingested.proxy_path(a.asset_id))
    assert p.video.color_transfer in (None, "bt709", "unknown")
    assert min(p.video.width, p.video.height) <= 540
    assert p.video.bit_depth == 8


def test_ingest_leaves_no_temp_artifacts(ingested):
    assert not list(ingested.cache_dir.rglob("*.tmp.*"))


def test_ingest_never_modifies_originals(ingested, fixtures):
    before = {p.name: p.stat().st_mtime for p in fixtures.glob("*.mp4")}
    cfg = load_config(ingested.root)
    run_ingest(ingested, cfg)
    after = {p.name: p.stat().st_mtime for p in fixtures.glob("*.mp4")}
    assert before == after


def test_ingest_is_resumable_and_skips_completed_stages(ingested, monkeypatch):
    """A second run must not rebuild proxies whose params are unchanged."""
    from montaje.ingest import pipeline

    calls: list[str] = []
    original = pipeline.proxy_mod.build_proxy

    def spy(*args, **kwargs):
        calls.append("built")
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline.proxy_mod, "build_proxy", spy)
    run_ingest(ingested, load_config(ingested.root))
    assert calls == []


def test_report_lists_vfr_and_records_totals(ingested):
    report = build_report(ingested)
    assert "# Ingest report" in report
    assert "VFR assets" in report
    assert "vfr.mp4" in report


def test_report_is_written_to_disk(ingested):
    build_report(ingested)
    assert (ingested.root / "report.md").exists()


def test_detect_degraded_flags_low_resolution_video(ingested):
    with Store(ingested.db_path) as store:
        small = next(a for a in store.list_assets() if a.path.name == "vfr.mp4")  # 320x240
        big = next(a for a in store.list_assets() if a.path.name == "hlg.mp4")  # 640x360
    assert detect_degraded(small) is not None
    assert detect_degraded(big) is not None  # both below the 1280px shared-album threshold


def test_hdr_helpers(fixtures):
    hlg = probe_file(fixtures / "hlg.mp4")
    sdr = probe_file(fixtures / "plain.mp4")
    assert needs_tonemap(hlg.video) is True
    assert needs_tonemap(sdr.video) is False
    assert sdr_filter(sdr.video) is None
    assert sdr_filter(hlg.video, "zscale_hable") is not None


def test_resolve_method_falls_back_to_an_available_one():
    assert resolve_method("naive_clip") == "naive_clip"
    resolved = resolve_method("libplacebo_bt709")
    assert resolved in available_methods()


def test_available_methods_always_includes_naive_clip():
    assert "naive_clip" in available_methods()
