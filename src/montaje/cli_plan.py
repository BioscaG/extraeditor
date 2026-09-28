"""Planning, rendering and library CLI commands (§23).

Kept out of `cli.py` so importing the CLI stays cheap: these pull in the render
pipeline, which drags in numpy and the library registry.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from montaje.config import load_config
from montaje.models.editplan import EditPlan
from montaje.workspace import Workspace

console = Console()


def resolve_workspace(project: str | None) -> Workspace:
    if project:
        return Workspace.find(project)
    cwd = Path.cwd()
    if (cwd / "project.yaml").exists():
        return Workspace(cwd)
    projects = cwd / "projects"
    if projects.is_dir():
        candidates = [d for d in projects.iterdir() if (d / "project.yaml").exists()]
        if len(candidates) == 1:
            return Workspace(candidates[0])
    raise typer.BadParameter("cannot determine project; pass --project")


def load_music_context(ws: Workspace):
    """Beat grid, structure and duration fit for the project's music track.

    Returns `(structure, fit, music_asset_id)`; all None when the project has no
    music, in which case the plan is built to the brief's target duration instead.
    """
    from montaje.music.edit import fit_to_duration
    from montaje.music.structure import analyze_structure

    brief = ws.load_brief()
    if brief.music is None or brief.music.path is None:
        return None, None, None
    path = Path(brief.music.path)
    if not path.is_absolute():
        path = ws.root / path
    if not path.exists():
        console.print(f"[yellow]Music track not found: {path}[/yellow]")
        return None, None, None

    structure = analyze_structure(path)
    target = brief.duration.target_s if brief.duration else structure.duration_s
    tolerance = brief.duration.tolerance_s if brief.duration else 5.0
    fit = fit_to_duration(structure, target, tolerance_s=tolerance, fps=brief.format.fps)
    return structure, fit, None


def latest_plan(ws: Workspace) -> EditPlan | None:
    plans = sorted(ws.plans_dir.glob("plan_v*.json"))
    if not plans:
        return None
    return EditPlan.model_validate_json(plans[-1].read_text())


def cmd_plan(project: str | None, title: str | None, target_s: float | None) -> None:
    """Build a baseline EditPlan from the analysis and the music."""
    from montaje.index.store import Store
    from montaje.models.editplan import PlanFormat
    from montaje.plan.build import BuildInputs, build_plan
    from montaje.sound.spotting import apply_spotting
    from montaje.styles.registry import load_default, load_style
    from montaje.workspace import atomic_write_text

    ws = resolve_workspace(project)
    brief = ws.load_brief()
    with Store(ws.db_path) as store:
        assets = {a.asset_id: a for a in store.list_assets()}
        events = {aid: store.get_events(aid) for aid in assets}
        clip_logs = {aid: log for aid in assets if (log := store.get_clip_log(aid))}
    if not assets:
        raise typer.BadParameter("no assets; run `montaje ingest` and `montaje analyze` first")

    structure, fit, music_asset = load_music_context(ws)
    style = load_style(brief.style) or load_default()

    plan = build_plan(
        project=ws.slug,
        fmt=PlanFormat(width=brief.format.width, height=brief.format.height,
                       fps=brief.format.fps, dynamic_range=brief.format.dynamic_range),
        inputs=BuildInputs(assets=assets, events=events, clip_logs=clip_logs,
                           structure=structure, music_fit=fit, style=style,
                           music_asset_id=music_asset),
        title=title or brief.goal or ws.slug,
        target_s=target_s or (brief.duration.target_s if brief.duration else None),
    )
    plan = apply_spotting(plan, structure=structure, style=style)

    path = ws.plans_dir / f"plan_v{plan.version:03d}.json"
    atomic_write_text(path, plan.model_dump_json(indent=2))
    console.print(
        f"[green]Built[/green] {path.name}: {len(plan.shots)} shots, "
        f"{plan.timeline_end_frame() / plan.format.fps:.1f}s, "
        f"{len(plan.sfx)} SFX, {len(plan.overlays)} overlays"
    )
    for section in plan.concept.sections:
        n = sum(1 for s in plan.shots if s.section == section.id)
        console.print(f"  {section.id:<8} {section.music_section or '':<7} {n:>3} shots")


def cmd_render(project: str | None, quality: str, plan_ref: str, skip_video: bool) -> None:
    """Render a plan: conform, mix, compose, mux."""
    from montaje.render.pipeline import render

    ws = resolve_workspace(project)
    cfg = load_config(ws.root)
    if plan_ref == "latest":
        plan = latest_plan(ws)
        if plan is None:
            raise typer.BadParameter("no plan found; run `montaje plan` first")
    else:
        plan = EditPlan.model_validate_json(Path(plan_ref).read_text())

    structure, _, _ = load_music_context(ws)
    with console.status(f"Rendering v{plan.version} ({quality})…"):
        result = render(ws, plan, cfg, structure=structure, quality=quality,
                        skip_video=skip_video)

    if result.rails_summary:
        console.print(f"[dim]Rails: {result.rails_summary}[/dim]")
    for w in result.warnings:
        console.print(f"[yellow]WARN[/yellow] {w}")
    for e in result.errors:
        console.print(f"[red]ERROR[/red] {e}")
    if result.errors:
        console.print("[red]Render blocked by hard errors (§18.2).[/red]")
        raise typer.Exit(1)

    if result.mix_stats:
        console.print(
            f"Audio: {result.mix_stats['final_lufs']} LUFS, "
            f"peak {result.mix_stats['final_true_peak_dbtp']} dBTP, "
            f"stems: {', '.join(result.mix_stats['stems'])}"
        )
    if result.video:
        console.print(f"[green]Rendered[/green] {result.video}")
    console.print(f"Rhythm report: {result.rhythm_path}")


def cmd_library_list() -> None:
    """Show every component in the craft library."""
    from montaje.library.registry import load_components, load_sfx

    table = Table(title="Craft library")
    table.add_column("ref")
    table.add_column("kind")
    table.add_column("status")
    table.add_column("frames")
    table.add_column("presets")
    table.add_column("intent", overflow="fold", max_width=54)
    for c in load_components():
        table.add_row(
            c.ref, c.kind.value, c.status.value,
            f"{c.duration.min_frames}-{c.duration.max_frames}",
            ", ".join(c.presets) or "—",
            c.intent,
        )
    console.print(table)

    sfx = load_sfx()
    sfx_table = Table(title=f"SFX ({len(sfx)} licensed)")
    for col in ("id", "category", "duration", "peak", "energy", "license"):
        sfx_table.add_column(col)
    for s in sfx:
        sfx_table.add_row(s.id, s.category, f"{s.duration_s:.2f}s",
                          f"{s.peak_offset_s:.3f}s", s.energy, s.license)
    console.print(sfx_table)


def cmd_library_gallery(out: Path, width: int, height: int, aspects: str) -> None:
    """Render every component × preset × aspect ratio, plus an HTML index (§13.4)."""
    from montaje.library.gallery import render_gallery

    entries = render_gallery(out, base_width=width, base_height=height,
                             aspects=[a.strip() for a in aspects.split(",") if a.strip()])
    console.print(f"[green]Rendered[/green] {len(entries)} gallery items to {out}")
    console.print(f"Open {out / 'index.html'} to review, then `montaje library review`.")
