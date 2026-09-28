"""montaje CLI (§23)."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from montaje.config import load_config
from montaje.models.brief import Brief
from montaje.workspace import Workspace

app = typer.Typer(no_args_is_help=True, help="montaje — agentic video editor")
source_app = typer.Typer(no_args_is_help=True, help="Manage footage sources")
debug_app = typer.Typer(no_args_is_help=True, help="Debug utilities")
library_app = typer.Typer(no_args_is_help=True, help="Craft library: list, gallery, review")
conventions_app = typer.Typer(no_args_is_help=True, help="Discovered footage conventions")
style_app = typer.Typer(no_args_is_help=True, help="Styles")
app.add_typer(source_app, name="source")
app.add_typer(debug_app, name="debug")
app.add_typer(library_app, name="library")
app.add_typer(conventions_app, name="conventions")
app.add_typer(style_app, name="style")

console = Console()

PROJECT_OPT = typer.Option(
    None, "--project", "-p", help="Project slug or path (default: sole project or cwd)"
)


def _resolve_ws(project: str | None) -> Workspace:
    if project:
        return Workspace.find(project)
    cwd = Path.cwd()
    if (cwd / "project.yaml").exists():
        return Workspace(cwd)
    projects = Path.cwd() / "projects"
    if projects.is_dir():
        candidates = [d for d in projects.iterdir() if (d / "project.yaml").exists()]
        if len(candidates) == 1:
            return Workspace(candidates[0])
    raise typer.BadParameter("cannot determine project; pass --project")


@app.command()
def init(name: str) -> None:
    """Create projects/<name>/ with a starter project.yaml."""
    root = Path.cwd() / "projects" / name
    ws = Workspace(root)
    ws.create(Brief(name=name))
    console.print(f"[green]Created[/green] {root}")
    console.print(f"Edit {ws.project_yaml} to set goal, format, duration, music and style.")


@source_app.command("add")
def source_add(
    kind: str = typer.Argument(..., help="folder | apple-photos"),
    target: str = typer.Argument(..., help="Folder path, or album name for apple-photos"),
    project: str | None = PROJECT_OPT,
) -> None:
    """Register a footage source in sources.yaml."""
    ws = _resolve_ws(project)
    if kind == "folder":
        path = Path(target).expanduser().resolve()
        if not path.is_dir():
            raise typer.BadParameter(f"{path} is not a directory")
        ws.add_source({"type": "folder", "path": str(path)})
    elif kind == "apple-photos":
        ws.add_source({"type": "apple-photos", "album": target})
        console.print("[yellow]Apple Photos export lands in M2; use folder export meanwhile (§7.4).[/yellow]")
    else:
        raise typer.BadParameter(f"unknown source kind {kind!r}")
    console.print(f"[green]Added[/green] {kind} source: {target}")


@app.command()
def ingest(project: str | None = PROJECT_OPT) -> None:
    """Probe, proxy, extract audio and thumbnails for every asset. Resumable."""
    from montaje.ingest.pipeline import run_ingest
    from montaje.ingest.report import build_report

    ws = _resolve_ws(project)
    cfg = load_config(ws.root)
    with console.status("Ingesting…"):
        result = run_ingest(ws, cfg)
    console.print(f"Ingested [bold]{result['assets']}[/bold] assets.")
    for f in result["failures"]:
        console.print(f"[red]FAIL[/red] {f}")
    build_report(ws)
    console.print(f"Report: {ws.root / 'report.md'}")


@app.command()
def report(project: str | None = PROJECT_OPT) -> None:
    """Regenerate and print the ingest report."""
    from rich.markdown import Markdown

    from montaje.ingest.report import build_report

    ws = _resolve_ws(project)
    console.print(Markdown(build_report(ws)))


@app.command()
def analyze(
    project: str | None = PROJECT_OPT,
    only: str | None = typer.Option(None, help="Comma-separated analyzer names"),
    semantic: bool = typer.Option(False, help="Also run Gemini clip logs"),
) -> None:
    """Run local analyzers over all ingested assets. Cached and resumable."""
    from montaje.analysis.runner import run_analysis

    ws = _resolve_ws(project)
    cfg = load_config(ws.root)
    names = only.split(",") if only else None
    summary = run_analysis(ws, cfg, only=names, semantic=semantic)
    for line in summary:
        console.print(line)


@app.command()
def plan(
    project: str | None = PROJECT_OPT,
    title: str | None = typer.Option(None, help="Concept title"),
    target_s: float | None = typer.Option(None, "--target", help="Override target duration"),
) -> None:
    """Build a baseline EditPlan from the analysis and the music track."""
    from montaje.cli_plan import cmd_plan

    cmd_plan(project, title, target_s)


@app.command()
def render(
    project: str | None = PROJECT_OPT,
    quality: str = typer.Option("draft", help="draft | final"),
    plan_ref: str = typer.Option("latest", "--plan", help="'latest' or a plan JSON path"),
    skip_video: bool = typer.Option(False, help="Mix audio and validate without rendering picture"),
) -> None:
    """Render a plan: rails, validation, conform, mix, compose, mux."""
    from montaje.cli_plan import cmd_render

    if quality not in ("draft", "final"):
        raise typer.BadParameter("quality must be draft or final")
    cmd_render(project, quality, plan_ref, skip_video)


@library_app.command("list")
def library_list() -> None:
    """List every component and SFX in the craft library."""
    from montaje.cli_plan import cmd_library_list

    cmd_library_list()


@library_app.command("gallery")
def library_gallery(
    out: Path = typer.Option(Path("library/gallery"), help="Output directory"),
    width: int = typer.Option(1080),
    height: int = typer.Option(1920),
    aspects: str = typer.Option("9:16", help="Comma-separated aspect ratios"),
) -> None:
    """Render every component × preset × aspect ratio plus an HTML index."""
    from montaje.cli_plan import cmd_library_gallery

    cmd_library_gallery(out, width, height, aspects)


@style_app.command("list")
def style_list() -> None:
    """List available styles."""
    from montaje.styles.registry import list_styles

    for name in list_styles():
        console.print(name)


@app.command()
def export(
    what: str = typer.Argument("all", help="all | srt | otio | fcpxml | overlays | stems"),
    project: str | None = PROJECT_OPT,
) -> None:
    """Export editable timelines, captions, graphics with alpha and audio stems."""
    from montaje.cli_plan import cmd_export

    cmd_export(project, what)


@conventions_app.command("propose")
def conventions_propose(project: str | None = PROJECT_OPT) -> None:
    """Mine the footage for recurring motifs and propose conventions for confirmation."""
    from montaje.cli_plan import cmd_conventions_propose

    cmd_conventions_propose(project)


@conventions_app.command("list")
def conventions_list(project: str | None = PROJECT_OPT) -> None:
    """Show every proposed, confirmed and rejected convention."""
    from montaje.cli_plan import cmd_conventions_list

    cmd_conventions_list(project)


@conventions_app.command("confirm")
def conventions_confirm(
    convention_id: str = typer.Argument(..., help="Convention id, or 'all'"),
    project: str | None = PROJECT_OPT,
    reject: bool = typer.Option(False, "--reject", help="Reject instead of confirming"),
) -> None:
    """Confirm (or reject) a convention. Nothing affects an edit until confirmed."""
    from montaje.cli_plan import cmd_conventions_confirm

    cmd_conventions_confirm(project, convention_id, reject)


@debug_app.command("rhythm")
def debug_rhythm(project: str | None = PROJECT_OPT) -> None:
    """Print the rhythm report for the latest plan."""
    from rich.markdown import Markdown

    from montaje.cli_plan import latest_plan, load_music_context, resolve_workspace
    from montaje.index.store import Store
    from montaje.plan.prepare import prepare_plan
    from montaje.plan.rhythm import build_rhythm_report
    from montaje.styles.registry import load_style

    ws = resolve_workspace(project)
    cfg = load_config(ws.root)
    edit_plan = latest_plan(ws)
    if edit_plan is None:
        raise typer.BadParameter("no plan found; run `montaje plan` first")
    structure, _, _ = load_music_context(ws)
    with Store(ws.db_path) as store:
        assets = {a.asset_id: a for a in store.list_assets()}
        events = {aid: store.get_events(aid) for aid in assets}
    style = load_style(edit_plan.style)
    # The rails first, because they are what the render measures. Reading the plan file
    # straight off disk reports on the pre-rails *request*: it showed 58% of cuts off the
    # musical grid for an edit whose cuts the rails were about to put on it.
    prepared = prepare_plan(
        edit_plan, ws=ws, cfg=cfg, events=events, assets=assets,
        structure=structure, style=style,
    )
    report = build_rhythm_report(
        prepared.plan, structure=structure, style=style, events=events
    )
    console.print(Markdown(report.to_markdown()))
    if prepared.summary:
        console.print(f"[dim]rails: {prepared.summary}[/dim]")


@app.command()
def mcp(
    project: str | None = PROJECT_OPT,
    transport: str = typer.Option("stdio", help="stdio | sse | streamable-http"),
) -> None:
    """Run the MCP server so an agent (e.g. Claude Code) can direct the edit."""
    from montaje.director.mcp_server import run

    run(project, transport=transport)


@debug_app.command("tools")
def debug_tools() -> None:
    """List the tools the MCP server exposes."""
    from montaje.director.tools import tool_catalog

    for entry in tool_catalog():
        console.print(f"{entry['name']:<22} {entry['purpose']}")


@debug_app.command("tonemap")
def debug_tonemap(
    clips: list[Path] = typer.Argument(..., help="HDR clips to compare"),
    out: Path = typer.Option(Path("tonemap_comparison"), help="Output dir"),
    at: float = typer.Option(1.0, help="Timestamp of the still"),
) -> None:
    """Render side-by-side tone-mapping stills to decide the M0 HDR method."""
    from montaje.ingest.hdr import render_tonemap_comparison

    for clip in clips:
        outputs = render_tonemap_comparison(clip, out, at_s=at)
        console.print(f"{clip.name}: " + ", ".join(p.name for p in outputs))
    console.print(f"Review stills in {out}/ and record the decision in DECISIONS.md")


@debug_app.command("events")
def debug_events(
    asset_id: str,
    analyzer: str | None = typer.Option(None),
    project: str | None = PROJECT_OPT,
) -> None:
    """Dump analyzer events for one asset."""
    from montaje.index.store import Store

    ws = _resolve_ws(project)
    with Store(ws.db_path) as store:
        for e in store.get_events(asset_id, analyzer):
            console.print(
                f"{e.analyzer:>16} {e.type:<10} {e.t0:7.2f}–{e.t1:7.2f}  {e.score:.2f}  {e.data}"
            )


if __name__ == "__main__":
    app()
