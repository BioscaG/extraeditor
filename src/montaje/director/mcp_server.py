"""MCP server exposing the director's tools (§16.1).

`montaje mcp --project <slug>` runs this over stdio, which is how Claude Code acts as
the director in M1 — no custom agent loop needed. The same functions back the built-in
loop, so the two cannot drift.

Every tool is a thin wrapper: the logic lives in `director.tools`, and the docstrings
here are what the model actually reads when deciding which tool to call, so they say
*when* to use each one rather than restating its name.
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from montaje.director import tools
from montaje.director.tools import Session
from montaje.workspace import Workspace

INSTRUCTIONS = """\
You are the director of a montaje edit. Your job is creative judgement: concept,
selection, juxtaposition, rhythm and structure. The deterministic modules are your
eyes and hands.

How to work:

1. **Survey before deciding.** `project_overview`, then `list_footage` and
   `music_structure`. Look at ranges with `contact_sheet` before committing to them.
2. **Start from the baseline.** `build_baseline_plan` gives a rule-following plan.
   Refine it with `plan_apply`; do not hand-build a timeline shot by shot.
3. **Read the rhythm report before every preview.** `rhythm_report` is the only thing
   that can see micro-timing — a critic watching the video cannot. Fix what it flags.
4. **Validate, then render.** `plan_validate` first; hard errors block the render.

Craft principles that matter most:

- Hard cuts are the default. Designed transitions are punctuation — use them at
  section changes and for emphasis, never between every shot. Transition and SFX
  overuse is the most recognisable amateur tell.
- Map story to music: tension in the build, the best collective moment on the drop.
- Vary shot lengths, and break the beat grid on purpose somewhere. An edit that is
  100% on-grid reads as mechanical.
- Cut on action, match motion direction across cuts, hold on faces and real reactions.
- Never place near-duplicates back to back.
- Every shot needs an `intent`. If you cannot say why a shot is there, cut it.
- Each library component's `intent` says when *not* to use it. Respect that.

The rails (snapping, validation, loudness, safe areas) are non-bypassable and run
whatever you do. If a snap moved something, the render result tells you.
"""


def build_server(workspace: Workspace) -> MCPServer:
    """An MCP server bound to one project workspace."""
    session = Session(workspace)
    server = MCPServer(
        name="montaje",
        title="montaje — agentic video editor",
        instructions=INSTRUCTIONS,
        version="0.1.0",
    )

    # -- survey ---------------------------------------------------------------------

    @server.tool()
    def project_overview() -> dict[str, Any]:
        """Start here. Asset counts and durations, devices, degraded assets, the brief,
        and the music's tempo and section structure."""
        return tools.project_overview(session)

    @server.tool()
    def list_footage(limit: int = 100) -> dict[str, Any]:
        """Every asset with its usable spans, speech spans, dominant motion direction
        and semantic summary. Use this to see what you have to work with."""
        return tools.list_footage(session, limit)

    @server.tool()
    def search_footage(
        query: str = "",
        min_energy: int | None = None,
        needs_speech: bool | None = None,
        min_duration_s: float = 0.0,
        limit: int = 25,
    ) -> dict[str, Any]:
        """Ranked usable moments. `min_energy` is 1-5; `needs_speech` filters to (or
        away from) spoken ranges. Ranking matches the baseline planner's."""
        return tools.search_footage(session, query, min_energy, needs_speech,
                                   min_duration_s, limit)

    @server.tool()
    def get_events(asset_id: str, analyzer: str | None = None,
                   limit: int = 200) -> dict[str, Any]:
        """Raw analyzer events for one asset: shots, quality, occlusion, motion,
        audio_events, vad, loudness, color_stats. Use when a summary is not enough."""
        return tools.get_events(session, asset_id, analyzer, limit)

    @server.tool()
    def get_clip_log(asset_id: str) -> dict[str, Any]:
        """The semantic clip log: summary, setting, people, energy, moments, quotes.
        Only present if semantic analysis has run."""
        return tools.get_clip_log(session, asset_id)

    @server.tool()
    def semantic_status() -> dict[str, Any]:
        """Whether semantic clip logs exist. Check this early: without them, footage is
        ranked on sharpness and motion only and `search_footage` cannot tell a hero
        moment from a shot of the floor."""
        return tools.semantic_status(session)

    @server.tool()
    def contact_sheet(asset_id: str, t0: float = 0.0, t1: float | None = None,
                      columns: int = 5) -> dict[str, Any]:
        """Render a frame grid for a range and return its path, so you can look at
        footage before committing to it. Do this before choosing hero shots."""
        return tools.contact_sheet(session, asset_id, t0, t1, columns)

    @server.tool()
    def find_patterns() -> dict[str, Any]:
        """Recurring motifs across the footage: how often clips begin or end with a
        particular kind of event. These are measurements — interpreting one is your job.
        Run this during the survey, before building a plan."""
        return tools.find_patterns(session)

    @server.tool()
    def propose_conventions() -> dict[str, Any]:
        """Turn discovered patterns into conventions awaiting confirmation. Then use
        `ask_user` to ask which are real; nothing affects the edit until confirmed."""
        return tools.propose_conventions_tool(session)

    @server.tool()
    def get_conventions() -> dict[str, Any]:
        """Current conventions and their status. Only confirmed ones trim any footage."""
        return tools.get_conventions(session)

    @server.tool()
    def confirm_convention(convention_id: str, reject: bool = False) -> dict[str, Any]:
        """Confirm or reject a convention — only after the user has said which. A
        confirmed convention trims its motif out of every shot from a matching clip."""
        return tools.confirm_convention(session, convention_id, reject)

    @server.tool()
    def music_structure() -> dict[str, Any]:
        """Tempo, beat grid, phrase boundaries, sections with roles and energy, and how
        the track was fitted to the target duration. Plan sections against this."""
        return tools.music_structure(session)

    # -- library --------------------------------------------------------------------

    @server.tool()
    def library_search(query: str = "", kind: str | None = None,
                       energy: str | None = None) -> dict[str, Any]:
        """Search the craft library. `kind` is transition|shot_fx|text|overlay|
        finishing|layout. Read each result's `intent`: it says when not to use it."""
        return tools.library_search(query, kind, energy)

    @server.tool()
    def library_get(ref: str) -> dict[str, Any]:
        """One component's full metadata: duration limits, presets, beat anchor, motion
        match and default SFX. Use before setting an unfamiliar component."""
        return tools.library_get(ref)

    @server.tool()
    def sfx_search(query: str = "", category: str | None = None) -> dict[str, Any]:
        """Licensed SFX. Categories: whoosh, impact, riser, downlifter, tick, pop,
        sub_drop, crowd_bed, tape_stop. Spotting already places the obvious ones."""
        return tools.sfx_search(query, category)

    @server.tool()
    def get_style(name: str | None = None) -> dict[str, Any]:
        """A style's pacing targets per section role, transition palette, density
        limits and taste notes. These are the targets the rhythm report checks."""
        return tools.get_style(name)

    # -- plan -----------------------------------------------------------------------

    @server.tool()
    def build_baseline_plan(title: str | None = None,
                            target_s: float | None = None) -> dict[str, Any]:
        """Build a rule-following starting plan: music sections mapped onto the
        best-scoring footage, paced to the style. Refine it rather than starting
        from nothing."""
        return tools.build_baseline_plan(session, title, target_s)

    @server.tool()
    def plan_get(include_shots: bool = True) -> dict[str, Any]:
        """The current plan. Pass include_shots=False for a summary when the timeline
        is long."""
        return tools.plan_get(session, include_shots)

    @server.tool()
    def plan_apply(ops: list[dict]) -> dict[str, Any]:
        """Apply operations to the plan, producing a new version and a diff. Each op is
        `{"op": "<name>", ...args}`. Ops: insert_shot, remove_shot, move_shot,
        trim_shot, replace_source, set_transition, set_fx, set_audio, set_speed,
        set_reframe, set_color, set_captions, set_section, set_intent, add_overlay,
        update_overlay, remove_overlay, add_sfx, remove_sfx, set_music_edits, set_note.
        A batch is all-or-nothing: if one op fails the plan is untouched."""
        return tools.plan_apply(session, ops)

    @server.tool()
    def plan_validate() -> dict[str, Any]:
        """Hard errors and warnings. Errors block the render; warnings you may accept,
        but an unusable shot needs an `intent` explaining why."""
        return tools.plan_validate(session)

    @server.tool()
    def rhythm_report() -> dict[str, Any]:
        """Read this before every preview. Cut-to-beat offsets, per-section pacing
        against the style's targets, visual-versus-music energy correlation, and
        transition and SFX density. Nothing else can see micro-timing."""
        return tools.rhythm_report(session)

    @server.tool()
    def render_preview(quality: str = "draft") -> dict[str, Any]:
        """Render the plan: rails, validation, conform, mix, compose, mux. Returns the
        output path plus what the rails changed. Use quality="final" only once."""
        return tools.render_preview(session, quality)

    @server.tool()
    def write_summary() -> dict[str, Any]:
        """Write a human-readable summary of the plan next to it. Do this at wrap-up."""
        return tools.write_summary(session)

    @server.tool()
    def ask_user(question: str, options: list[str] | None = None) -> dict[str, Any]:
        """Ask the human a question — which concept to pursue, whether a convention is
        real. Returns the question for you to relay; continue with their answer."""
        return tools.ask_user(question, options)

    return server


def run(project: str | None = None, transport: str = "stdio") -> None:
    """Entry point for `montaje mcp`."""
    from montaje.cli_plan import resolve_workspace

    workspace = resolve_workspace(project)
    build_server(workspace).run(transport=transport)


if __name__ == "__main__":
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else None)
