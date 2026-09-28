"""The director's tools (§16.2), as plain functions.

Kept free of any MCP dependency so they are directly callable and testable, and so
the built-in agent loop (§16.1) can use the same implementations the MCP server
exposes. Every tool returns JSON-serializable data.

Design rules these follow:
 - a tool answers a question the agent actually has, at the altitude it asks it — so
   `project_overview` summarizes rather than dumping the index;
 - anything that changes the plan goes through `plan.ops`, so it is validated and
   versioned rather than written directly;
 - errors are returned as data (`{"error": ...}`) rather than raised, because a tool
   that throws ends the agent's turn instead of letting it recover.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from montaje.config import load_config
from montaje.index.store import Store
from montaje.library.registry import find as find_component
from montaje.library.registry import load_components, load_sfx, search
from montaje.models.editplan import EditPlan
from montaje.plan.ops import OpError, apply_ops, available_ops, diff_summary
from montaje.plan.rhythm import build_rhythm_report
from montaje.plan.validate import ValidationContext, validate
from montaje.styles.registry import list_styles, load_style
from montaje.workspace import Workspace, atomic_write_text


class Session:
    """Holds the workspace the tools operate on, plus cached analysis."""

    def __init__(self, workspace: Workspace):
        self.ws = workspace
        self.cfg = load_config(workspace.root)
        self._structure = None
        self._structure_loaded = False

    @property
    def structure(self):
        """Music structure, analyzed once per session (it takes a few seconds)."""
        if not self._structure_loaded:
            from montaje.cli_plan import load_music_context

            self._structure, self._fit, _ = load_music_context(self.ws)
            self._structure_loaded = True
        return self._structure

    @property
    def fit(self):
        _ = self.structure
        return getattr(self, "_fit", None)

    def store(self) -> Store:
        return Store(self.ws.db_path)

    def latest_plan(self) -> EditPlan | None:
        plans = self.ws.plan_paths()
        if not plans:
            return None
        return EditPlan.model_validate_json(plans[-1].read_text())

    def write_plan(self, plan: EditPlan) -> Path:
        path = self.ws.plans_dir / f"plan_v{plan.version:03d}.json"
        atomic_write_text(path, plan.model_dump_json(indent=2))
        return path


# -- survey ------------------------------------------------------------------------


def project_overview(session: Session) -> dict[str, Any]:
    """Counts, durations, devices, degraded assets, brief and music structure (§16.2)."""
    from montaje.ingest.report import detect_degraded

    brief = session.ws.load_brief()
    with session.store() as store:
        assets = store.list_assets()
        logs = {a.asset_id: store.get_clip_log(a.asset_id) for a in assets}

    kinds = Counter(a.kind.value for a in assets)
    devices = Counter(f"{a.probe.make or '?'} {a.probe.model or '?'}".strip() for a in assets)
    degraded = [
        {"asset_id": a.asset_id, "reason": reason}
        for a in assets if (reason := detect_degraded(a))
    ]

    music: dict[str, Any] | None = None
    if session.structure is not None:
        music = {
            "bpm": session.structure.grid.bpm,
            "duration_s": round(session.structure.duration_s, 2),
            "beats_per_bar": session.structure.grid.beats_per_bar,
            "sections": [
                {"id": s.id, "role": s.role, "t0": s.t0, "t1": s.t1,
                 "bars": s.bars, "energy": s.energy}
                for s in session.structure.sections
            ],
        }
        if session.fit is not None:
            music["fit"] = {
                "total_s": session.fit.total_s,
                "phrase_bars": session.fit.phrase_bars,
                "joins": session.fit.joins,
                "notes": session.fit.notes,
            }

    return {
        "project": session.ws.slug,
        "brief": brief.model_dump(mode="json", exclude_none=True),
        "assets": {
            "count": len(assets),
            "total_duration_s": round(sum(a.duration_s for a in assets), 1),
            "by_kind": dict(kinds),
            "devices": dict(devices),
            "with_clip_logs": sum(1 for v in logs.values() if v is not None),
        },
        "degraded": degraded,
        "music": music,
        "analysis_present": sorted({
            e.analyzer.split("@")[0]
            for a in assets
            for e in _events(session, a.asset_id)[:1]
        }) if assets else [],
        "styles_available": list_styles(),
    }


def _events(session: Session, asset_id: str):
    with session.store() as store:
        return store.get_events(asset_id)


def list_footage(session: Session, limit: int = 100) -> dict[str, Any]:
    """One line per asset: duration, usable spans, speech, motion and semantics."""
    with session.store() as store:
        assets = store.list_assets()
        out = []
        for asset in assets[:limit]:
            events = store.get_events(asset.asset_id)
            log = store.get_clip_log(asset.asset_id)
            usable = [e for e in events if e.analyzer.startswith("quality")
                      and e.type == "usable"]
            speech = [e for e in events if e.analyzer.startswith("vad") and e.type == "speech"]
            motion = [e for e in events if e.analyzer.startswith("motion")]
            directions = Counter(m.data.get("direction", "static") for m in motion)
            out.append({
                "asset_id": asset.asset_id,
                "file": asset.path.name,
                "kind": asset.kind.value,
                "duration_s": round(asset.duration_s, 2),
                "usable_spans": [[round(e.t0, 2), round(e.t1, 2)] for e in usable],
                "usable_s": round(sum(e.t1 - e.t0 for e in usable), 2),
                "speech_spans": [[round(e.t0, 2), round(e.t1, 2)] for e in speech],
                "dominant_motion": directions.most_common(1)[0][0] if directions else "static",
                "summary": log.summary if log else None,
                "energy": log.energy if log else None,
                "aesthetic": log.aesthetic if log else None,
                "us_present": log.people.us_present if log else None,
            })
    return {"count": len(assets), "shown": len(out), "assets": out}


def search_footage(
    session: Session,
    query: str = "",
    min_energy: int | None = None,
    needs_speech: bool | None = None,
    min_duration_s: float = 0.0,
    limit: int = 25,
) -> dict[str, Any]:
    """Ranked usable moments matching the filters (§16.2).

    Without semantic clip logs this is a filter over the local analysis rather than a
    true semantic search; the ranking is the same one the baseline planner uses, so
    the agent and the planner agree on what "good" means.
    """
    from montaje.plan.build import BuildInputs, collect_candidates

    with session.store() as store:
        assets = {a.asset_id: a for a in store.list_assets()}
        events = {aid: store.get_events(aid) for aid in assets}
        logs = {aid: log for aid in assets if (log := store.get_clip_log(aid))}

    candidates = collect_candidates(BuildInputs(assets=assets, events=events, clip_logs=logs))
    words = [w for w in query.lower().split() if w]

    results = []
    for c in candidates:
        if c.duration_s < min_duration_s:
            continue
        if min_energy is not None and c.energy < min_energy:
            continue
        if needs_speech is not None and c.has_speech != needs_speech:
            continue
        if words and not all(w in c.label.lower() for w in words):
            continue
        results.append({
            "asset_id": c.asset_id,
            "t0": round(c.t0, 2),
            "t1": round(c.t1, 2),
            "duration_s": round(c.duration_s, 2),
            "quality": c.quality,
            "motion": round(c.motion, 2),
            "direction": c.direction,
            "has_speech": c.has_speech,
            "energy": c.energy,
            "aesthetic": c.aesthetic,
            "score": round(c.score(0.6), 4),
            "label": c.label,
        })
    results.sort(key=lambda r: -r["score"])
    return {"count": len(results), "moments": results[:limit]}


def get_events(session: Session, asset_id: str, analyzer: str | None = None,
               limit: int = 200) -> dict[str, Any]:
    """Raw analyzer events for one asset."""
    with session.store() as store:
        if store.get_asset(asset_id) is None:
            return {"error": f"unknown asset {asset_id}"}
        events = store.get_events(asset_id, analyzer)
    return {
        "asset_id": asset_id,
        "count": len(events),
        "events": [
            {"analyzer": e.analyzer, "type": e.type, "t0": round(e.t0, 3),
             "t1": round(e.t1, 3), "score": e.score, "data": e.data}
            for e in events[:limit]
        ],
    }


def semantic_status(session: Session) -> dict[str, Any]:
    """Whether semantic analysis is available and how much of the footage it covers.

    Worth checking first: without clip logs the planner ranks footage on sharpness and
    motion only, so selection is arbitrary within the technically sound material, and the
    agent should know that before trusting `search_footage`.
    """
    from montaje.analysis.semantic.clip_log import semantic_config
    from montaje.analysis.semantic.gemini_client import available

    scfg = semantic_config(session.cfg)
    with session.store() as store:
        assets = store.list_assets()
        logged = sum(1 for a in assets if store.get_clip_log(a.asset_id) is not None)
    return {
        "available": available(scfg),
        "model": scfg.model,
        "assets": len(assets),
        "with_clip_logs": logged,
        "note": (
            "Run `montaje analyze --semantic` to log clips."
            if logged < len(assets)
            else "All assets logged."
        ),
    }


def get_clip_log(session: Session, asset_id: str) -> dict[str, Any]:
    with session.store() as store:
        log = store.get_clip_log(asset_id)
    if log is None:
        return {"error": f"no clip log for {asset_id}; run `montaje analyze --semantic`"}
    return log.model_dump(mode="json")


def find_patterns(session: Session) -> dict[str, Any]:
    """Recurring structural motifs across the footage (§11).

    Measurements, not meanings: what the miner reports is how often clips begin or end
    with a particular kind of event. Interpreting one — deciding that a dark span at the
    start of a clip is someone uncovering the lens — is the agent's job, and the user's to
    confirm (§12, §2.5).
    """
    from montaje.index.patterns import mine_patterns

    with session.store() as store:
        assets = {a.asset_id: a for a in store.list_assets()}
        events = {aid: store.get_events(aid) for aid in assets}
        names = {a.asset_id: a.path.name for a in assets.values()}

    patterns = mine_patterns(assets, events)
    return {
        "clips": len(assets),
        "patterns": [
            {
                "id": p.id,
                "kind": p.kind,
                "start_event": p.event_type if p.kind != "end_motif" else None,
                "end_event": p.paired_event_type or (
                    p.event_type if p.kind == "end_motif" else None
                ),
                "count": p.count,
                "share": round(p.share, 3),
                "selectivity": p.selectivity,
                "confidence": p.confidence,
                "description": p.describe(),
                "example_files": [names.get(a, a) for a in p.assets[:6]],
            }
            for p in patterns
        ],
        "note": (
            "Call propose_conventions to turn these into conventions, then ask the user "
            "to confirm. Nothing affects the edit until confirmed."
        ),
    }


def propose_conventions_tool(session: Session) -> dict[str, Any]:
    """Turn discovered patterns into conventions awaiting confirmation (§12)."""
    from montaje.index.conventions import load_conventions, merge_proposals, save_conventions
    from montaje.index.patterns import mine_patterns, propose_conventions

    with session.store() as store:
        assets = {a.asset_id: a for a in store.list_assets()}
        events = {aid: store.get_events(aid) for aid in assets}

    merged = merge_proposals(
        load_conventions(session.ws), propose_conventions(mine_patterns(assets, events))
    )
    save_conventions(session.ws, merged)
    return {
        "conventions": [
            {
                "id": c.id,
                "status": c.status.value,
                "description": c.description,
                "start_event": c.detection.start_event,
                "end_event": c.detection.end_event,
                "role": c.treatment.role,
                "clips": c.evidence.count,
            }
            for c in merged
        ],
        "note": (
            "Ask the user to confirm the ones that are real, with `ask_user`. A convention "
            "the user already confirmed or rejected keeps that decision."
        ),
    }


def get_conventions(session: Session) -> dict[str, Any]:
    """Current conventions and their status. Only confirmed ones affect an edit."""
    from montaje.index.conventions import load_conventions

    conventions = load_conventions(session.ws)
    return {
        "count": len(conventions),
        "confirmed": [c.id for c in conventions if c.status.value == "confirmed"],
        "proposed": [c.id for c in conventions if c.status.value == "proposed"],
        "rejected": [c.id for c in conventions if c.status.value == "rejected"],
        "conventions": [c.model_dump(mode="json") for c in conventions],
    }


def confirm_convention(session: Session, convention_id: str, reject: bool = False) -> dict[str, Any]:
    """Confirm or reject a convention on the user's instruction.

    Only call this once the user has answered — the whole point of §12 is that the system
    proposes and the human decides.
    """
    from montaje.index.conventions import load_conventions, save_conventions
    from montaje.models.conventions import ConventionStatus

    conventions = load_conventions(session.ws)
    target = next((c for c in conventions if c.id == convention_id), None)
    if target is None:
        return {
            "error": f"no convention {convention_id!r}",
            "known": [c.id for c in conventions],
        }
    target.status = ConventionStatus.REJECTED if reject else ConventionStatus.CONFIRMED
    save_conventions(session.ws, conventions)
    return {"id": target.id, "status": target.status.value}


def music_structure(session: Session) -> dict[str, Any]:
    """Tempo, beat grid summary, sections and how the track was fitted (§14.2)."""
    structure = session.structure
    if structure is None:
        return {"error": "no music track in project.yaml"}
    grid = structure.grid
    return {
        "bpm": grid.bpm,
        "beats_per_bar": grid.beats_per_bar,
        "confidence": grid.confidence,
        "duration_s": round(structure.duration_s, 2),
        "beat_count": len(grid.beats),
        "first_downbeats": grid.downbeats[:8],
        "phrase_boundaries_4bar": structure.phrase_boundaries(4)[:16],
        "sections": [
            {"id": s.id, "role": s.role, "t0": s.t0, "t1": s.t1,
             "bars": s.bars, "energy": s.energy}
            for s in structure.sections
        ],
        "fit": {
            "total_s": session.fit.total_s,
            "removed_s": session.fit.removed_s,
            "repeated_s": session.fit.repeated_s,
            "phrase_bars": session.fit.phrase_bars,
            "joins": session.fit.joins,
            "notes": session.fit.notes,
        } if session.fit else None,
    }


# -- library ------------------------------------------------------------------------


def library_search(query: str = "", kind: str | None = None,
                   energy: str | None = None) -> dict[str, Any]:
    """Components matching the query, with their intent and presets (§16.2)."""
    results = search(query, kind, energy)
    return {
        "count": len(results),
        "components": [
            {
                "ref": c.ref,
                "kind": c.kind.value,
                "status": c.status.value,
                "energy": c.energy,
                "tags": c.tags,
                "frames": [c.duration.min_frames, c.duration.max_frames],
                "default_beats": c.duration.default_beats,
                "beat_anchor": c.beat_anchor,
                "motion_match": c.motion_match,
                "default_sfx": c.sfx.default if c.sfx else None,
                "presets": list(c.presets),
                "intent": c.intent,
            }
            for c in results
        ],
    }


def library_get(ref: str) -> dict[str, Any]:
    meta = find_component(ref)
    if meta is None:
        available = sorted(c.ref for c in load_components())
        return {"error": f"unknown component {ref}", "available": available}
    return meta.model_dump(mode="json")


def sfx_search(query: str = "", category: str | None = None) -> dict[str, Any]:
    """Licensed SFX, with the peak offset the mixer aligns to the anchor (§14.1)."""
    words = [w for w in query.lower().split() if w]
    out = []
    for s in load_sfx():
        if category is not None and s.category != category:
            continue
        haystack = f"{s.id} {s.category} {s.energy}".lower()
        if words and not all(w in haystack for w in words):
            continue
        out.append(s.model_dump(mode="json"))
    return {"count": len(out), "sfx": out}


def get_style(name: str | None = None) -> dict[str, Any]:
    """A style's pacing targets, palettes and taste notes (§22)."""
    style = load_style(name) if name else None
    if style is None:
        from montaje.styles.registry import load_default

        style = load_default()
    return style.model_dump(mode="json")


# -- plan ---------------------------------------------------------------------------


def plan_get(session: Session, include_shots: bool = True) -> dict[str, Any]:
    """The current plan. `include_shots=False` for a summary of a long timeline."""
    plan = session.latest_plan()
    if plan is None:
        return {"error": "no plan yet; run `montaje plan` or use plan_apply to build one"}
    if include_shots:
        return plan.model_dump(mode="json")
    fps = plan.format.fps
    return {
        "version": plan.version,
        "project": plan.project,
        "format": plan.format.model_dump(),
        "style": plan.style,
        "concept": plan.concept.model_dump(mode="json"),
        "duration_s": round(plan.timeline_end_frame() / fps, 2),
        "shot_count": len(plan.shots),
        "sfx_count": len(plan.sfx),
        "overlay_count": len(plan.overlays),
        "shots_per_section": dict(Counter(s.section or "none" for s in plan.shots)),
    }


def plan_apply(session: Session, ops: list[dict]) -> dict[str, Any]:
    """Apply ops to the current plan, writing a new version (§17.3)."""
    plan = session.latest_plan()
    if plan is None:
        return {"error": "no plan yet"}
    try:
        result = apply_ops(plan, ops)
    except OpError as e:
        return {"error": str(e), "available_ops": available_ops()}
    path = session.write_plan(result.plan)
    return {
        "version": result.plan.version,
        "path": str(path),
        "summary": result.summary,
        "diff": diff_summary(plan, result.plan),
        "duration_s": round(
            result.plan.timeline_end_frame() / result.plan.format.fps, 2
        ),
    }


def plan_validate(session: Session) -> dict[str, Any]:
    """Hard errors and warnings for the current plan (§18.2)."""
    from montaje.library.registry import licensed_sfx_ids, stable_refs

    plan = session.latest_plan()
    if plan is None:
        return {"error": "no plan yet"}
    brief = session.ws.load_brief()
    style = load_style(plan.style or brief.style)
    with session.store() as store:
        assets = {a.asset_id: a for a in store.list_assets()}
        events = {aid: store.get_events(aid) for aid in assets}
    ctx = ValidationContext.from_events(
        assets, events,
        stable_components=stable_refs(),
        licensed_sfx=licensed_sfx_ids(),
        min_shot_s=session.cfg.rails.min_shot_s,
        duration_target_s=brief.duration.target_s if brief.duration else None,
        duration_tolerance_s=brief.duration.tolerance_s if brief.duration else 15.0,
        max_transitions_per_10s=style.transitions.max_per_10s if style else None,
        max_sfx_per_10s=style.sfx.density_max_per_10s if style else None,
    )
    problems = validate(plan, ctx)
    return {
        "version": plan.version,
        "errors": [
            {"code": p.code, "message": p.message, "shot": p.shot_id}
            for p in problems if p.severity.value == "error"
        ],
        "warnings": [
            {"code": p.code, "message": p.message, "shot": p.shot_id}
            for p in problems if p.severity.value == "warning"
        ],
    }


def rhythm_report(session: Session) -> dict[str, Any]:
    """The rhythm report the agent must read before each preview (§18.3)."""
    plan = session.latest_plan()
    if plan is None:
        return {"error": "no plan yet"}
    with session.store() as store:
        events = {a.asset_id: store.get_events(a.asset_id) for a in store.list_assets()}
    style = load_style(plan.style or session.ws.load_brief().style)
    report = build_rhythm_report(plan, structure=session.structure, style=style, events=events)
    return {
        "total_s": report.total_s,
        "shots": report.shots,
        "on_grid_ratio": report.on_grid_ratio,
        "target_on_grid_ratio": report.target_on_grid_ratio,
        "offset_histogram": report.offset_histogram(),
        "transitions_per_10s": report.transitions_per_10s,
        "sfx_per_10s": report.sfx_per_10s,
        "energy_correlation": report.energy_correlation,
        "us_share": report.us_share,
        "sections": [
            {
                "id": s.section_id, "role": s.role, "shots": s.shots,
                "avg_shot_beats": s.avg_shot_beats,
                "target_shot_beats": s.target_shot_beats,
                "on_target": s.on_target,
                "visual_energy": s.visual_energy, "music_energy": s.music_energy,
            }
            for s in report.sections
        ],
        "problems": report.problems,
        "markdown": report.to_markdown(),
    }


def build_baseline_plan(session: Session, title: str | None = None,
                        target_s: float | None = None) -> dict[str, Any]:
    """Build a rule-following starting plan for the agent to refine."""
    from montaje.models.editplan import PlanFormat
    from montaje.plan.build import BuildInputs, build_plan
    from montaje.sound.spotting import apply_spotting
    from montaje.styles.registry import load_default

    brief = session.ws.load_brief()
    with session.store() as store:
        assets = {a.asset_id: a for a in store.list_assets()}
        events = {aid: store.get_events(aid) for aid in assets}
        logs = {aid: log for aid in assets if (log := store.get_clip_log(aid))}
    if not assets:
        return {"error": "no assets; run `montaje ingest` and `montaje analyze` first"}

    style = load_style(brief.style) or load_default()
    plan = build_plan(
        project=session.ws.slug,
        fmt=PlanFormat(width=brief.format.width, height=brief.format.height,
                       fps=brief.format.fps, dynamic_range=brief.format.dynamic_range),
        inputs=BuildInputs(assets=assets, events=events, clip_logs=logs,
                           structure=session.structure, music_fit=session.fit, style=style),
        title=title or brief.goal or session.ws.slug,
        target_s=target_s or (brief.duration.target_s if brief.duration else None),
    )
    plan = apply_spotting(plan, structure=session.structure, style=style)
    plan.version = session.ws.next_plan_version()
    path = session.write_plan(plan)
    return {
        "version": plan.version,
        "path": str(path),
        "shots": len(plan.shots),
        "duration_s": round(plan.timeline_end_frame() / plan.format.fps, 2),
        "sections": [
            {"id": s.id, "role": s.music_section,
             "shots": sum(1 for x in plan.shots if x.section == s.id)}
            for s in plan.concept.sections
        ],
    }


def render_preview(session: Session, quality: str = "draft") -> dict[str, Any]:
    """Render the current plan and report what the rails changed (§16.2)."""
    from montaje.render.pipeline import render

    plan = session.latest_plan()
    if plan is None:
        return {"error": "no plan yet"}
    result = render(session.ws, plan, session.cfg, structure=session.structure,
                    quality=quality)
    return {
        "ok": result.ok,
        "video": str(result.video) if result.video else None,
        "audio": str(result.audio) if result.audio else None,
        "rails": result.rails_summary,
        "errors": result.errors,
        "warnings": result.warnings,
        "mix": result.mix_stats,
        "intermediates_cached": result.intermediates_cached,
        "rhythm_report": str(result.rhythm_path) if result.rhythm_path else None,
    }


def contact_sheet(session: Session, asset_id: str, t0: float = 0.0,
                  t1: float | None = None, columns: int = 5) -> dict[str, Any]:
    """A grid of frames from a range, written to the project and returned by path."""
    from montaje import ffmpeg

    with session.store() as store:
        asset = store.get_asset(asset_id)
    if asset is None:
        return {"error": f"unknown asset {asset_id}"}
    proxy = session.ws.proxy_path(asset_id)
    if not proxy.exists():
        return {"error": f"no proxy for {asset_id}; run `montaje ingest`"}

    t1 = min(t1 if t1 is not None else asset.duration_s, asset.duration_s)
    if t1 <= t0:
        return {"error": f"empty range {t0}-{t1}"}
    out_dir = session.ws.root / "contact_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{asset_id}_{t0:.1f}-{t1:.1f}.jpg"
    # One frame per second of the range, capped so a long clip stays one readable image.
    span = t1 - t0
    count = max(1, min(25, int(span)))
    fps_for_sheet = count / span
    ffmpeg.run([
        "-y", "-v", "error", "-ss", f"{t0:.3f}", "-i", str(proxy), "-t", f"{span:.3f}",
        "-vf", f"fps={fps_for_sheet:.4f},scale=-2:240,tile={columns}x{(count + columns - 1) // columns}",
        "-frames:v", "1", str(out),
    ])
    return {"asset_id": asset_id, "range": [t0, t1], "frames": count, "path": str(out)}


def ask_user(question: str, options: list[str] | None = None) -> dict[str, Any]:
    """Human in the loop (§16.2).

    Over MCP the *client* is the human's interface, so this does not prompt: it returns
    the question for the agent to relay. In the built-in loop this is replaced by a real
    prompt.
    """
    return {
        "question": question,
        "options": options or [],
        "note": "Relay this question to the user and continue with their answer.",
    }


def write_summary(session: Session) -> dict[str, Any]:
    """A human-readable summary of the current plan (§17, plan_v###.md)."""
    plan = session.latest_plan()
    if plan is None:
        return {"error": "no plan yet"}
    fps = plan.format.fps
    lines = [
        f"# {plan.concept.title}",
        "",
        f"*{plan.concept.logline or ''}*",
        "",
        f"- **Version:** {plan.version}",
        f"- **Duration:** {plan.timeline_end_frame() / fps:.1f}s "
        f"({len(plan.shots)} shots, {plan.format.width}x{plan.format.height} @ {fps}fps)",
        f"- **Style:** {plan.style or 'none'}",
        f"- **SFX:** {len(plan.sfx)} · **Overlays:** {len(plan.overlays)}",
        "",
        "## Sections",
        "",
    ]
    for section in plan.concept.sections:
        shots = [s for s in plan.sorted_shots() if s.section == section.id]
        lines.append(
            f"### {section.id} — {section.music_section or 'section'} "
            f"({section.from_frame / fps:.1f}–{section.to_frame / fps:.1f}s, {len(shots)} shots)"
        )
        lines.append("")
        for shot in shots:
            length = shot.timeline_duration_frames(fps) / fps
            extras = []
            if shot.transition_in:
                extras.append(shot.transition_in.id.split("@")[0])
            if shot.captions:
                extras.append("captions")
            if shot.audio.mode.value != "music_only":
                extras.append(shot.audio.mode.value)
            tag = f" [{', '.join(extras)}]" if extras else ""
            lines.append(
                f"- `{shot.id}` {shot.timeline_in / fps:6.2f}s +{length:.2f}s "
                f"{shot.asset} {shot.src_in:.2f}–{shot.src_out:.2f}{tag} — {shot.intent}"
            )
        lines.append("")
    if plan.notes:
        lines += ["## Notes", "", plan.notes, ""]

    path = session.ws.plans_dir / f"plan_v{plan.version:03d}.md"
    atomic_write_text(path, "\n".join(lines))
    return {"path": str(path), "version": plan.version}


def tool_catalog() -> list[dict[str, str]]:
    """What the MCP server exposes, for documentation and tests."""
    return [
        {"name": "project_overview", "purpose": "counts, durations, devices, brief, music"},
        {"name": "list_footage", "purpose": "per-asset analysis summary"},
        {"name": "search_footage", "purpose": "ranked usable moments"},
        {"name": "get_events", "purpose": "raw analyzer events for one asset"},
        {"name": "get_clip_log", "purpose": "semantic clip log"},
        {"name": "semantic_status", "purpose": "is semantic analysis available and applied"},
        {"name": "contact_sheet", "purpose": "frame grid for a range"},
        {"name": "find_patterns", "purpose": "recurring structural motifs in the footage"},
        {"name": "propose_conventions_tool", "purpose": "turn patterns into conventions"},
        {"name": "get_conventions", "purpose": "conventions and their confirmation status"},
        {"name": "confirm_convention", "purpose": "confirm or reject, on the user's word"},
        {"name": "music_structure", "purpose": "tempo, grid, sections, duration fit"},
        {"name": "library_search", "purpose": "components by query/kind/energy"},
        {"name": "library_get", "purpose": "one component's full metadata"},
        {"name": "sfx_search", "purpose": "licensed SFX with peak offsets"},
        {"name": "get_style", "purpose": "pacing targets and taste notes"},
        {"name": "build_baseline_plan", "purpose": "rule-following starting plan"},
        {"name": "plan_get", "purpose": "current plan, full or summary"},
        {"name": "plan_apply", "purpose": "apply ops, new version + diff"},
        {"name": "plan_validate", "purpose": "hard errors and warnings"},
        {"name": "rhythm_report", "purpose": "cut-to-beat, pacing, density, energy"},
        {"name": "render_preview", "purpose": "render and report rail changes"},
        {"name": "write_summary", "purpose": "human-readable plan summary"},
        {"name": "ask_user", "purpose": "relay a question to the human"},
    ]


def dump(value: Any) -> str:
    """Serialize a tool result for transports that want text."""
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)
