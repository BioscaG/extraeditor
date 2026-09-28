"""Turn a written plan into the plan that actually renders (§18).

The rails are non-bypassable, and they change the edit substantially: they trim source
ranges to confirmed conventions, snap every cut onto the musical grid, repair speech cuts,
drop captions with no words under them and spot the sound effects. A plan straight off
`montaje plan` — or straight out of the agent — is a *request*; this is what gets made.

The distinction matters because it is easy to measure the wrong one. The rhythm report used
to read the plan file from disk, which is the pre-rails request, and so reported cut timing
for an edit that never existed: it showed 58% of cuts off the musical grid while the rails
were putting them on it at render time. Both the report and the render now come through
here, so there is one answer to "what does this edit do" rather than two.
"""

from __future__ import annotations

from dataclasses import dataclass

from montaje.config import Config
from montaje.index.conventions import apply_conventions, load_conventions
from montaje.models.asset import Asset
from montaje.models.editplan import EditPlan
from montaje.models.events import Event
from montaje.models.style import Style
from montaje.music.structure import MusicStructure
from montaje.plan.build import drop_empty_captions, fix_speech_cuts
from montaje.plan.rails import RailContext, apply_rails
from montaje.plan.snap import SnapCandidates
from montaje.sound.spotting import apply_spotting


@dataclass
class PreparedPlan:
    plan: EditPlan
    summary: str


def prepare_plan(
    plan: EditPlan,
    *,
    ws,
    cfg: Config,
    events: dict[str, list[Event]],
    assets: dict[str, Asset],
    structure: MusicStructure | None,
    style: Style | None,
) -> PreparedPlan:
    """Apply every rail, in the one order that works.

    Everything that changes a shot's **source range** runs before the rails, and everything
    that depends on its final **timeline position** runs after.

    Conventions trim source ranges, so they go first — applied afterwards they shortened
    shots the rails had already laid out and opened one- and two-frame gaps, which render as
    black frames and fail validation. Speech repair, caption repair and SFX spotting all
    read timeline positions, so they must come after snapping has moved them.
    """
    notes: list[str] = []

    convention_report = apply_conventions(
        plan, load_conventions(ws), events, min_shot_s=cfg.rails.min_shot_s
    )

    rail_ctx = RailContext(
        candidates={aid: SnapCandidates.from_events(evs) for aid, evs in events.items()},
        grid=structure.grid if structure else None,
        fps=plan.format.fps,
        beat_window_ms=cfg.snap.beat_window_ms,
        word_preroll_s=cfg.snap.word_preroll_s,
        word_postroll_s=cfg.snap.word_postroll_s,
        min_shot_s=cfg.rails.min_shot_s,
        asset_durations={aid: a.duration_s for aid, a in assets.items()},
    )
    plan, rail_report = apply_rails(plan, rail_ctx)
    notes.append(rail_report.summary())
    if convention_report.changed:
        notes.append(convention_report.summary())

    fix_speech_cuts(plan)
    dropped = drop_empty_captions(plan, word_spans(events))
    if dropped:
        notes.append(f"{dropped} empty captions dropped")
    plan = apply_spotting(plan, structure=structure, style=style)
    return PreparedPlan(plan=plan, summary="; ".join(n for n in notes if n))


def word_spans(events: dict[str, list[Event]]) -> dict[str, list[tuple[float, float]]]:
    """Transcribed word spans per asset, for caption repair.

    Assets with no words are left out rather than mapped to an empty list. The distinction
    is load-bearing downstream: `drop_empty_captions` treats a missing asset as "no evidence
    either way, leave the plan's intent alone" and an empty list as "checked, nothing there".
    """
    out: dict[str, list[tuple[float, float]]] = {}
    for asset_id, evs in events.items():
        spans = [
            (e.t0, e.t1)
            for e in evs
            if e.analyzer.split("@")[0] == "asr" and e.type == "word"
        ]
        if spans:
            out[asset_id] = spans
    return out
