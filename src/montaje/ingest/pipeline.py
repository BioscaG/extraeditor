"""Ingest orchestration: probe → id → proxy/audio/thumbs, parallel, resumable (§8)."""

from __future__ import annotations

import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from montaje.config import Config
from montaje.index.store import Store
from montaje.ingest import audio as audio_mod
from montaje.ingest import probe as probe_mod
from montaje.ingest import proxy as proxy_mod
from montaje.ingest import stills
from montaje.ingest.hashing import compute_asset_id, params_hash
from montaje.models.asset import Asset, AssetKind
from montaje.sources.base import LocalAsset

log = logging.getLogger(__name__)


def register_asset(local: LocalAsset, source_name: str) -> Asset:
    probe = probe_mod.probe_file(local.path)
    asset_id = compute_asset_id(local.path, probe.duration_s)
    kind = probe_mod.classify_kind(local.path, probe, live_photo_pair=local.live_photo_pair_uid is not None)
    return Asset(
        asset_id=asset_id,
        path=local.path,
        kind=kind,
        probe=probe,
        capture=probe_mod.capture_time(local.path, probe),
        source=source_name,
        extra=dict(local.remote.extra),
    )


def _run_stage(store: Store, asset: Asset, stage: str, phash: str, fn) -> str | None:
    """Run one stage if not already done for these params. Returns error text or None."""
    if store.stage_state(asset.asset_id, stage, phash) == "done":
        return None
    store.set_stage(asset.asset_id, stage, "running", phash)
    try:
        fn()
    except subprocess.CalledProcessError as e:
        raw = e.stderr
        err = raw.decode(errors="replace")[-500:] if isinstance(raw, bytes) else str(e)
        store.set_stage(asset.asset_id, stage, "failed", phash, error=err)
        return f"{stage}: {err}"
    except Exception as e:  # keep going; one bad file must not kill the batch
        store.set_stage(asset.asset_id, stage, "failed", phash, error=str(e))
        return f"{stage}: {e}"
    store.set_stage(asset.asset_id, stage, "done", phash)
    return None


def ingest_asset(ws, store: Store, asset: Asset, cfg: Config) -> list[str]:
    """Run all ingest stages for one registered asset. Returns error strings."""
    errors: list[str] = []
    cache = ws.asset_cache(asset.asset_id)
    cache.mkdir(parents=True, exist_ok=True)

    is_still = asset.kind == AssetKind.PHOTO
    pparams = proxy_mod.proxy_params(cfg.proxy, cfg.hdr.method)
    phash = params_hash(pparams)

    if not is_still:
        err = _run_stage(
            store, asset, "proxy", phash,
            lambda: proxy_mod.build_proxy(
                asset.path, ws.proxy_path(asset.asset_id), asset.probe, cfg.proxy, cfg.hdr.method
            ),
        )
        if err:
            errors.append(err)

        if asset.probe.selected_audio is not None:
            err = _run_stage(
                store, asset, "audio", "v1",
                lambda: audio_mod.extract_audio(
                    asset.path,
                    ws.audio_16k_path(asset.asset_id),
                    ws.audio_48k_path(asset.asset_id),
                    asset.probe,
                ),
            )
            if err:
                errors.append(err)
        else:
            store.set_stage(asset.asset_id, "audio", "done", "no-audio")

        if not errors and ws.proxy_path(asset.asset_id).exists():
            err = _run_stage(
                store, asset, "thumbs", "v1",
                lambda: stills.extract_thumbs(
                    ws.proxy_path(asset.asset_id), ws.thumbs_dir(asset.asset_id), asset.duration_s
                ),
            )
            if err:
                errors.append(err)
    return errors


def run_ingest(ws, cfg: Config) -> dict:
    """Scan configured sources, register and process every asset. Idempotent."""
    from montaje.sources.folder import FolderSource

    store = Store(ws.db_path)
    locals_: list[tuple[LocalAsset, str]] = []
    for entry in ws.load_sources():
        if entry.get("type") == "folder":
            src = FolderSource(Path(entry["path"]))
            for remote in src.list():
                locals_.append((src.fetch(remote, ws.media_dir), src.name))
        elif entry.get("type") == "apple-photos":
            raise NotImplementedError("Apple Photos source lands in M2 (photos-bridge)")

    assets: list[Asset] = []
    failures: list[str] = []
    for local, source_name in locals_:
        try:
            asset = register_asset(local, source_name)
        except Exception as e:
            failures.append(f"{local.path.name}: probe failed: {e}")
            continue
        # Wire live-photo pairs by re-resolving the pair path to its asset id later (report pass).
        store.upsert_asset(asset)
        assets.append(asset)

    with ThreadPoolExecutor(max_workers=cfg.workers.ingest) as pool:
        futures = {pool.submit(ingest_asset, ws, store, a, cfg): a for a in assets}
        for fut in as_completed(futures):
            a = futures[fut]
            for err in fut.result():
                failures.append(f"{a.path.name} ({a.asset_id}): {err}")

    store.close()
    return {"assets": len(assets), "failures": failures}
