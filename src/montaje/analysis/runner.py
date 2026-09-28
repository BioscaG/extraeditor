"""Analyzer orchestration: cache key (asset_id, analyzer, version, params_hash) (§6, §9)."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from montaje.analysis.local.audio_events import AudioEventsAnalyzer
from montaje.analysis.local.color_stats import ColorStatsAnalyzer
from montaje.analysis.local.loudness import LoudnessAnalyzer
from montaje.analysis.local.motion import MotionAnalyzer
from montaje.analysis.local.occlusion import OcclusionAnalyzer
from montaje.analysis.local.quality import QualityAnalyzer
from montaje.analysis.local.shots import ShotsAnalyzer
from montaje.analysis.local.vad import VadAnalyzer
from montaje.config import Config
from montaje.index.store import Store
from montaje.ingest.hashing import params_hash
from montaje.models.asset import Asset
from montaje.models.events import AnalyzerResult, Event
from montaje.workspace import Workspace, atomic_write_text

log = logging.getLogger(__name__)

# Order matters: color_stats consumes shots; asr consumes vad.
DEFAULT_ANALYZERS = [
    "shots", "quality", "occlusion", "motion",
    "audio_events", "vad", "loudness", "color_stats",
]


def build_analyzers(cfg: Config) -> dict:
    return {
        "shots": ShotsAnalyzer(),
        "quality": QualityAnalyzer(),
        "occlusion": OcclusionAnalyzer(),
        "motion": MotionAnalyzer(),
        "audio_events": AudioEventsAnalyzer(),
        "vad": VadAnalyzer(),
        "loudness": LoudnessAnalyzer(),
        "color_stats": ColorStatsAnalyzer(),
    }


def run_analyzer(ws: Workspace, store: Store, asset: Asset, analyzer) -> tuple[str, int, str | None]:
    """Run or restore one analyzer for one asset. Returns (analyzer, n_events, error)."""
    phash = params_hash(analyzer.params())
    stage = f"analyze:{analyzer.name}"
    cache_path = ws.analysis_path(asset.asset_id, analyzer.name, analyzer.version)

    if store.stage_state(asset.asset_id, stage, phash) == "done" and cache_path.exists():
        cached = AnalyzerResult.model_validate_json(cache_path.read_text())
        if cached.params_hash == phash:
            store.replace_events(asset.asset_id, f"{analyzer.name}@{analyzer.version}", cached.events)
            return analyzer.name, len(cached.events), None

    store.set_stage(asset.asset_id, stage, "running", phash)
    try:
        events: list[Event] = analyzer.run(asset, ws)
    except Exception as e:
        store.set_stage(asset.asset_id, stage, "failed", phash, error=str(e))
        return analyzer.name, 0, str(e)

    result = AnalyzerResult(
        asset_id=asset.asset_id, analyzer=analyzer.name, version=analyzer.version,
        params_hash=phash, events=events,
    )
    atomic_write_text(cache_path, result.model_dump_json(indent=2))
    store.replace_events(asset.asset_id, f"{analyzer.name}@{analyzer.version}", events)
    store.set_stage(asset.asset_id, stage, "done", phash)
    return analyzer.name, len(events), None


def run_analysis(
    ws: Workspace, cfg: Config, only: list[str] | None = None, semantic: bool = False
) -> list[str]:
    registry = build_analyzers(cfg)
    names = [n for n in (only or DEFAULT_ANALYZERS) if n in registry]
    unknown = set(only or []) - set(registry)
    lines: list[str] = [f"[yellow]Unknown analyzer {n!r}[/yellow]" for n in sorted(unknown)]

    store = Store(ws.db_path)
    assets = store.list_assets()
    if not assets:
        store.close()
        return lines + ["[yellow]No assets; run `montaje ingest` first.[/yellow]"]

    # Parallelism across assets, sequential per asset so dependent analyzers see their inputs.
    def per_asset(asset: Asset) -> list[str]:
        out: list[str] = []
        with Store(ws.db_path) as local_store:
            for name in names:
                _, n, err = run_analyzer(ws, local_store, asset, registry[name])
                if err:
                    out.append(f"[red]FAIL[/red] {asset.asset_id} {name}: {err}")
        return out

    with ThreadPoolExecutor(max_workers=cfg.workers.analysis) as pool:
        futures = [pool.submit(per_asset, a) for a in assets]
        for fut in as_completed(futures):
            lines.extend(fut.result())

    totals = {
        name: store.conn.execute(
            "SELECT COUNT(*) c FROM events WHERE analyzer LIKE ?", (f"{name}@%",)
        ).fetchone()["c"]
        for name in names
    }
    store.close()
    lines.append("Events: " + ", ".join(f"{k}={v}" for k, v in totals.items()))

    if semantic:
        from montaje.analysis.semantic.clip_log import run_semantic

        lines.extend(run_semantic(ws, cfg))
    return lines
