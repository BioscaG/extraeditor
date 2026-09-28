from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from montaje.index.store import Store
from montaje.models.asset import Asset, AssetKind, Probe, VideoStream
from montaje.models.events import Event


def _asset(asset_id: str = "a_test", duration: float = 5.0) -> Asset:
    return Asset(
        asset_id=asset_id,
        path=f"/tmp/{asset_id}.mp4",
        kind=AssetKind.VIDEO,
        probe=Probe(
            container="mov,mp4", duration_s=duration, size_bytes=1000,
            video=VideoStream(codec="h264", width=1920, height=1080, avg_fps=30.0),
        ),
    )


@pytest.fixture()
def store(tmp_path):
    with Store(tmp_path / "t.db") as s:
        yield s


def test_upsert_and_get_asset_roundtrips(store):
    a = _asset()
    store.upsert_asset(a)
    got = store.get_asset("a_test")
    assert got is not None
    assert got.asset_id == a.asset_id
    assert got.probe.video.width == 1920


def test_upsert_is_idempotent(store):
    store.upsert_asset(_asset())
    store.upsert_asset(_asset())
    assert len(store.list_assets()) == 1


def test_get_missing_asset_returns_none(store):
    assert store.get_asset("nope") is None


def test_list_assets_filters_by_kind(store):
    store.upsert_asset(_asset("a_one"))
    photo = _asset("a_two")
    photo.kind = AssetKind.PHOTO
    store.upsert_asset(photo)
    assert [a.asset_id for a in store.list_assets(kind="photo")] == ["a_two"]


def test_stage_state_defaults_to_pending(store):
    assert store.stage_state("a_test", "proxy") == "pending"


def test_stage_state_invalidates_on_params_change(store):
    store.set_stage("a_test", "proxy", "done", "hash1")
    assert store.stage_state("a_test", "proxy", "hash1") == "done"
    # A different params hash must re-run the stage, not reuse the old artifact.
    assert store.stage_state("a_test", "proxy", "hash2") == "pending"


def test_stage_records_error(store):
    store.set_stage("a_test", "proxy", "failed", "h", error="boom")
    row = store.conn.execute("SELECT error FROM status WHERE asset_id='a_test'").fetchone()
    assert row["error"] == "boom"


def test_replace_events_is_not_additive(store):
    e = [Event(asset_id="a_test", analyzer="shots@1", type="shot", t0=0, t1=1)]
    store.replace_events("a_test", "shots@1", e)
    store.replace_events("a_test", "shots@1", e)
    assert len(store.get_events("a_test", "shots")) == 1


def test_replace_events_leaves_other_analyzers_alone(store):
    store.replace_events("a_test", "shots@1",
                         [Event(asset_id="a_test", analyzer="shots@1", type="shot", t0=0, t1=1)])
    store.replace_events("a_test", "motion@1",
                         [Event(asset_id="a_test", analyzer="motion@1", type="motion", t0=0, t1=1)])
    assert len(store.get_events("a_test", "shots")) == 1
    assert len(store.get_events("a_test", "motion")) == 1


def test_get_events_filters_by_type_and_orders_by_time(store):
    events = [
        Event(asset_id="a_test", analyzer="occlusion@1", type="cover", t0=3, t1=4),
        Event(asset_id="a_test", analyzer="occlusion@1", type="reveal", t0=1, t1=2),
        Event(asset_id="a_test", analyzer="occlusion@1", type="reveal", t0=0, t1=1),
    ]
    store.replace_events("a_test", "occlusion@1", events)
    reveals = store.get_events("a_test", "occlusion", "reveal")
    assert [e.t0 for e in reveals] == [0.0, 1.0]


def test_event_data_survives_roundtrip(store):
    store.replace_events("a_test", "occlusion@1", [
        Event(asset_id="a_test", analyzer="occlusion@1", type="occluded",
              t0=0, t1=1, data={"mean_luma": 0.04, "detail": 3.1}),
    ])
    assert store.get_events("a_test")[0].data == {"mean_luma": 0.04, "detail": 3.1}


def test_event_rejects_inverted_range():
    with pytest.raises(ValueError):
        Event(asset_id="a", analyzer="x@1", type="t", t0=2.0, t1=1.0)


def test_store_is_usable_from_multiple_threads(tmp_path):
    """Ingest and analysis fan out over a thread pool; sqlite3 connections are per-thread."""
    s = Store(tmp_path / "t.db")
    s.upsert_asset(_asset())

    def work(i: int) -> str:
        s.set_stage("a_test", f"stage{i}", "done", "h")
        return s.stage_state("a_test", f"stage{i}", "h")

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(work, range(8))) == ["done"] * 8
