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
app.add_typer(source_app, name="source")
app.add_typer(debug_app, name="debug")

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
