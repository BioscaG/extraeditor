"""End-to-end render: plan → conform → Remotion → mix → mux (§19).

This is the only place that knows the full order of operations. Everything it calls
is independently testable; its own job is wiring and reporting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from montaje.color.grade import Grade, grade_from_style
from montaje.color.normalize import corrections_for_asset
from montaje.config import Config
from montaje.index.store import Store
from montaje.library.registry import licensed_sfx_ids, sfx_paths, sfx_peak_offsets, stable_refs
from montaje.models.asset import Asset
from montaje.models.editplan import EditPlan
from montaje.music.beats import BeatGrid
from montaje.music.structure import MusicStructure
from montaje.plan.build import fix_speech_cuts
from montaje.plan.rails import RailContext, apply_rails
from montaje.plan.rhythm import build_rhythm_report
from montaje.plan.snap import SnapCandidates
from montaje.plan.validate import ValidationContext, errors, validate, warnings
from montaje.render import remotion_bridge as remotion
from montaje.render.conform import conform_plan
from montaje.sound.mix import mix_spec_from_plan, render_mix
from montaje.sound.spotting import apply_spotting
from montaje.styles.registry import load_style
from montaje.workspace import Workspace, atomic_write_text

log = logging.getLogger(__name__)


@dataclass
class RenderResult:
    video: Path | None = None
    audio: Path | None = None
    stems_dir: Path | None = None
    plan_path: Path | None = None
    rhythm_path: Path | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rails_summary: str = ""
    mix_stats: dict = field(default_factory=dict)
    intermediates_cached: int = 0

    @property
    def ok(self) -> bool:
        return self.video is not None and not self.errors


def _load_context(ws: Workspace) -> tuple[dict[str, Asset], dict[str, list]]:
    with Store(ws.db_path) as store:
        assets = {a.asset_id: a for a in store.list_assets()}
        events = {aid: store.get_events(aid) for aid in assets}
    return assets, events


def _music_paths(ws: Workspace, assets: dict[str, Asset], plan: EditPlan) -> Path | None:
    """The 48 kHz wav for the plan's music asset, or the brief's track."""
    if plan.music.asset and plan.music.asset in assets:
        path = ws.audio_48k_path(plan.music.asset)
        if path.exists():
            return path
    brief = ws.load_brief()
    if brief.music and brief.music.path:
        candidate = Path(brief.music.path)
        if not candidate.is_absolute():
            candidate = ws.root / candidate
        if candidate.exists():
            return candidate
    return None


def render(
    ws: Workspace,
    plan: EditPlan,
    cfg: Config,
    *,
    structure: MusicStructure | None = None,
    quality: str = "draft",
    apply_rails_first: bool = True,
    skip_video: bool = False,
) -> RenderResult:
    """Render a plan to a finished file. Returns what was produced and what failed."""
    result = RenderResult()
    assets, events = _load_context(ws)
    brief = ws.load_brief()
    style = load_style(plan.style or brief.style)
    grid: BeatGrid | None = structure.grid if structure else None

    # -- rails (non-bypassable, §18) -------------------------------------------------
    if apply_rails_first:
        rail_ctx = RailContext(
            candidates={aid: SnapCandidates.from_events(evs) for aid, evs in events.items()},
            grid=grid,
            fps=plan.format.fps,
            beat_window_ms=cfg.snap.beat_window_ms,
            word_preroll_s=cfg.snap.word_preroll_s,
            word_postroll_s=cfg.snap.word_postroll_s,
            min_shot_s=cfg.rails.min_shot_s,
            asset_durations={aid: a.duration_s for aid, a in assets.items()},
        )
        plan, rail_report = apply_rails(plan, rail_ctx)
        result.rails_summary = rail_report.summary()

        # Both of these depend on the *final* timeline, so they run after the rails:
        # snapping moves cut points and may drop shots, which changes where SFX belong
        # and which speech shots end up adjacent.
        fix_speech_cuts(plan)
        plan = apply_spotting(plan, structure=structure, style=style)

    # -- validation ------------------------------------------------------------------
    vctx = ValidationContext.from_events(
        assets, events,
        stable_components=stable_refs(),
        licensed_sfx=licensed_sfx_ids(),
        min_shot_s=cfg.rails.min_shot_s,
        duration_target_s=brief.duration.target_s if brief.duration else None,
        duration_tolerance_s=brief.duration.tolerance_s if brief.duration else 15.0,
        max_transitions_per_10s=style.transitions.max_per_10s if style else None,
        max_sfx_per_10s=style.sfx.density_max_per_10s if style else None,
    )
    problems = validate(plan, vctx)
    result.errors = [str(p) for p in errors(problems)]
    result.warnings = [str(p) for p in warnings(problems)]

    # Persist the plan and its rhythm report even when it fails, so the failure is
    # inspectable (§2.9) and the agent can read the report before its next attempt.
    result.plan_path = _write_plan(ws, plan)
    report = build_rhythm_report(plan, structure=structure, style=style, events=events)
    result.rhythm_path = ws.plans_dir / f"rhythm_v{plan.version:03d}.md"
    atomic_write_text(result.rhythm_path, report.to_markdown())
    result.warnings.extend(report.problems)

    if result.errors:
        return result

    # -- conform (§19.1) -------------------------------------------------------------
    corrections = {
        aid: corrections_for_asset([e for e in evs if e.analyzer.startswith("color_stats")],
                                   strength=cfg.color.normalize_strength)
        for aid, evs in events.items()
    }
    grade = _grade_for(style)
    conformed = conform_plan(
        plan, assets, ws.intermediates_dir,
        corrections=corrections, grade=grade, hdr_method=cfg.hdr.method,
        shot_color_index=_shot_color_index(plan, events),
    )
    result.intermediates_cached = conformed.cached_count

    # -- audio (§19.2) ---------------------------------------------------------------
    audio_paths = {
        aid: ws.audio_48k_path(aid) for aid in assets if ws.audio_48k_path(aid).exists()
    }
    mix_spec = mix_spec_from_plan(
        plan,
        audio_paths=audio_paths,
        music_wav=_music_paths(ws, assets, plan),
        sfx_paths=sfx_paths(),
        sfx_peak_offsets=sfx_peak_offsets(),
        target_lufs=cfg.rails.loudness_lufs,
        true_peak_dbtp=cfg.rails.true_peak_dbtp,
    )
    result.audio = ws.intermediates_dir / f"mix_v{plan.version:03d}.wav"
    result.stems_dir = ws.exports_dir / "stems"
    result.mix_stats = render_mix(mix_spec, result.audio, stems_dir=result.stems_dir)

    if skip_video:
        return result

    # -- picture (§19.3) --------------------------------------------------------------
    resolved = remotion.resolve_shots(
        plan, conformed, grid=grid,
        luminance=_luminance_for(plan, events),
        words=_words_for(plan, events),
    )
    props = remotion.build_props(
        plan, resolved,
        bpm=grid.bpm if grid else 120.0,
        palette=_palette_for(style),
    )
    silent = ws.intermediates_dir / f"picture_v{plan.version:03d}.mp4"
    remotion.render_video(props, silent, quality=quality, public_dir=ws.intermediates_dir)

    name = "preview" if quality == "draft" else "final"
    result.video = ws.renders_dir / f"{name}_v{plan.version:03d}.mp4"
    remotion.mux(silent, result.audio, result.video)
    return result


def _write_plan(ws: Workspace, plan: EditPlan) -> Path:
    path = ws.plans_dir / f"plan_v{plan.version:03d}.json"
    atomic_write_text(path, plan.model_dump_json(indent=2))
    return path


def _grade_for(style) -> Grade | None:
    if style is None:
        return None
    from montaje.library.registry import LIBRARY_DIR

    return grade_from_style(style.color, lut_dir=LIBRARY_DIR / "luts")


def _palette_for(style) -> str:
    """Map a style to a text palette. Styles do not name one yet, so infer by name."""
    if style is None:
        return "neutral"
    name = style.name.lower()
    if "festival" in name:
        return "festival"
    if "warm" in name or "golden" in name:
        return "warm"
    return "neutral"


def _shot_color_index(plan: EditPlan, events: dict[str, list]) -> dict[str, int]:
    """Which `color_stats` shot each plan shot falls inside, for its correction."""
    out: dict[str, int] = {}
    for shot in plan.shots:
        stats = [
            e for e in events.get(shot.asset, [])
            if e.analyzer.startswith("color_stats") and e.type == "shot_color"
        ]
        mid = (shot.src_in + shot.src_out) / 2
        for e in stats:
            if e.t0 <= mid < e.t1:
                out[shot.id] = int(e.data.get("shot_index", 0))
                break
    return out


def _luminance_for(plan: EditPlan, events: dict[str, list]) -> dict[str, float]:
    """Mean luminance behind each shot, so text can decide on a scrim (§13.1)."""
    out: dict[str, float] = {}
    for shot in plan.shots:
        stats = [
            e for e in events.get(shot.asset, [])
            if e.analyzer.startswith("color_stats") and e.type == "shot_color"
            and e.t1 > shot.src_in and e.t0 < shot.src_out
        ]
        if not stats:
            continue
        # `exposure` is Lab L* normalized to 0–1, which is a perceptual lightness —
        # the right basis for a contrast decision.
        values = [float(e.data.get("exposure", 0.5)) for e in stats]
        out[shot.id] = sum(values) / len(values)
    return out


def _words_for(plan: EditPlan, events: dict[str, list]) -> dict[str, list[dict]]:
    """ASR words inside each captioned shot, with times relative to the shot."""
    out: dict[str, list[dict]] = {}
    for shot in plan.shots:
        if shot.captions is None:
            continue
        words = [
            e for e in events.get(shot.asset, [])
            if e.analyzer.startswith("asr") and e.type == "word"
            and e.t1 > shot.src_in and e.t0 < shot.src_out
        ]
        out[shot.id] = [
            {
                "text": str(w.data.get("text", "")),
                "start": round(max(0.0, w.t0 - shot.src_in), 4),
                "end": round(max(0.0, w.t1 - shot.src_in), 4),
            }
            for w in words
            if w.data.get("text")
        ]
    return out
