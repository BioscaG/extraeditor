"""Run semantic clip analysis over a project, cached (§10).

Cache key is `(asset_id, model, prompt_version)`, stored both in the SQLite index (for
querying) and as a JSON file per asset (for reading). Semantic analysis is the only paid
per-asset step, so a cache miss has a real cost and the key has to be exactly right:
rewording a prompt must invalidate it, re-running the same command must not.
"""

from __future__ import annotations

import logging
from pathlib import Path

from montaje.analysis.local.asr import transcript_text
from montaje.analysis.semantic import prompts
from montaje.analysis.semantic.gemini_client import (
    SemanticConfig,
    analyze_clip,
    available,
)
from montaje.config import Config
from montaje.index.store import Store
from montaje.models.cliplog import ClipLog
from montaje.workspace import Workspace, atomic_write_text

log = logging.getLogger(__name__)


def cache_path(
    ws: Workspace,
    asset_id: str,
    model: str,
    prompt_version: int,
    language: str | None = None,
) -> Path:
    """Where one clip log lives. The language is in the name because it is in the answer:
    switching the brief's language must not read back logs quoting the old one."""
    safe_model = model.replace("/", "_")
    suffix = f".{language}" if language else ""
    return (
        ws.asset_cache(asset_id) / "analysis"
        / f"cliplog_{safe_model}@{prompt_version}{suffix}.json"
    )


def semantic_config(cfg: Config, language: str | None = None) -> SemanticConfig:
    return SemanticConfig(
        model=cfg.semantic.model,
        fps_short_clips=cfg.semantic.fps_short_clips,
        media_resolution=cfg.semantic.media_resolution,
        language=language,
    )


def _brief_language(ws: Workspace) -> str | None:
    try:
        return ws.load_brief().language
    except Exception:
        return None


def run_semantic(ws: Workspace, cfg: Config, force: bool = False) -> list[str]:
    """Analyze every asset that has no cached clip log. Returns report lines."""
    scfg = semantic_config(cfg, _brief_language(ws))
    if not available(scfg):
        return [
            "[yellow]Semantic analysis skipped: set GEMINI_API_KEY (and install "
            "google-genai) to log clips. Without it the planner ranks footage on "
            "sharpness and motion only.[/yellow]"
        ]

    lines: list[str] = []
    uploaded: dict[str, object] = {}
    analyzed = cached = failed = 0

    with Store(ws.db_path) as store:
        assets = store.list_assets()
        for asset in assets:
            path = cache_path(
                ws, asset.asset_id, scfg.model, prompts.PROMPT_VERSION, scfg.language
            )
            if path.exists() and not force:
                store.upsert_clip_log(ClipLog.model_validate_json(path.read_text()))
                cached += 1
                continue

            events = store.get_events(asset.asset_id)
            transcript = transcript_text(
                [e for e in events if e.analyzer.startswith("asr")]
            )
            result = analyze_clip(
                asset, ws.proxy_path(asset.asset_id), events, scfg,
                transcript=transcript, uploaded=uploaded,
            )
            if not result.ok:
                failed += 1
                lines.append(f"[red]FAIL[/red] {asset.asset_id}: {result.error}")
                continue

            atomic_write_text(path, result.log.model_dump_json(indent=2))
            store.upsert_clip_log(result.log)
            analyzed += 1
            if result.discarded:
                lines.append(
                    f"[dim]{asset.asset_id}: discarded {len(result.discarded)} "
                    f"out-of-range timestamps[/dim]"
                )

    lines.append(
        f"Clip logs: {analyzed} analyzed, {cached} cached"
        + (f", {failed} failed" if failed else "")
    )
    return lines
