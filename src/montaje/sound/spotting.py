"""Automatic SFX spotting (§14.1).

Spotting rules are **data**, applied automatically and overridable per shot. They
turn plan structure into SFX placements: a whip transition gets a whoosh on the cut,
a section change gets a riser ending exactly on the next downbeat, a drop gets an
impact on the downbeat.

Density limits are the point of the whole module. SFX overuse is the most common
"cheap trailer" tell (§28), so the spotter emits candidates with priorities and then
*prunes* to the style's budget, keeping the highest-priority hits.
"""

from __future__ import annotations

from dataclasses import dataclass

from montaje.library.registry import load_sfx
from montaje.models.editplan import EditPlan, Sfx
from montaje.models.style import Style
from montaje.music.structure import MusicStructure

# Category → what it is anchored to. A riser must *end* on its target; everything
# else has its peak land on it.
END_ANCHORED = {"riser", "downlifter"}


@dataclass(frozen=True)
class Candidate:
    """A proposed SFX placement, before pruning."""

    category: str
    frame: int
    reason: str
    priority: int  # higher survives pruning
    gain_db: float = -6.0

    @property
    def end_anchored(self) -> bool:
        return self.category in END_ANCHORED


def pick_sfx_id(category: str, energy: str | None = None) -> str | None:
    """Choose a licensed file from a category, preferring a matching energy."""
    candidates = [s for s in load_sfx() if s.category == category]
    if not candidates:
        return None
    if energy:
        matching = [s for s in candidates if s.energy == energy]
        if matching:
            candidates = matching
    # Deterministic: the same plan must always spot the same files.
    return sorted(candidates, key=lambda s: s.id)[0].id


def candidates_for(plan: EditPlan, structure: MusicStructure | None = None) -> list[Candidate]:
    """Every SFX the rules suggest, highest priority first."""
    from montaje.library.registry import find

    out: list[Candidate] = []
    section_starts = {s.from_frame for s in plan.concept.sections}
    drop_starts = {
        s.from_frame for s in plan.concept.sections
        if (s.music_section or "").lower() == "drop" or (s.purpose or "").lower().startswith("peak")
    }

    for shot in plan.sorted_shots():
        # A transition's own metadata names its default SFX, so the rule follows the
        # component rather than hardcoding a mapping here.
        if shot.transition_in is not None:
            meta = find(shot.transition_in.id)
            if meta is not None and meta.sfx is not None:
                category = meta.sfx.default.split(".")[0]
                out.append(Candidate(
                    category=category,
                    frame=shot.timeline_in,
                    reason=f"{shot.transition_in.id} on {shot.id}",
                    priority=60,
                    gain_db=-7.0,
                ))

        if shot.captions is not None:
            out.append(Candidate(category="tick", frame=shot.timeline_in,
                                 reason=f"captions start on {shot.id}",
                                 priority=20, gain_db=-16.0))

    for frame in sorted(section_starts):
        if frame <= 0:
            continue
        priority = 90 if frame in drop_starts else 70
        # The riser *ends* on the downbeat that opens the section.
        out.append(Candidate(category="riser", frame=frame,
                             reason=f"section change at frame {frame}",
                             priority=priority, gain_db=-9.0))
        if frame in drop_starts:
            out.append(Candidate(category="impact", frame=frame,
                                 reason=f"drop at frame {frame}",
                                 priority=100, gain_db=-4.0))

    # A crowd bed under speech glues the ambience together when the clip audio is
    # otherwise dry (§14.1).
    for shot in plan.sorted_shots():
        if shot.audio.cleanup == "dialogue":
            out.append(Candidate(category="crowd_bed", frame=shot.timeline_in,
                                 reason=f"speech ambience under {shot.id}",
                                 priority=30, gain_db=-22.0))

    out.sort(key=lambda c: (-c.priority, c.frame))
    return out


def prune_to_budget(
    candidates: list[Candidate],
    total_frames: int,
    fps: float,
    max_per_10s: float,
    *,
    min_gap_s: float = 0.35,
) -> list[Candidate]:
    """Keep the highest-priority candidates within the style's density budget.

    Also enforces a minimum gap: two hits closer than ~350 ms read as one messy
    smear rather than two accents, however much budget is left.
    """
    total_s = total_frames / fps
    if total_s <= 0:
        return []
    budget = max(1, int(round(max_per_10s * total_s / 10.0)))
    kept: list[Candidate] = []
    for candidate in candidates:
        if len(kept) >= budget:
            break
        gap_ok = all(
            abs(candidate.frame - k.frame) / fps >= min_gap_s
            or candidate.category != k.category
            for k in kept
        )
        if gap_ok:
            kept.append(candidate)
    return sorted(kept, key=lambda c: c.frame)


def spot(
    plan: EditPlan,
    *,
    structure: MusicStructure | None = None,
    style: Style | None = None,
) -> list[Sfx]:
    """Automatic spotting for a plan, returning `Sfx` entries ready to apply.

    Only `source="auto_spotting"` entries are produced; anything the agent or the
    user placed is left untouched by the caller.
    """
    fps = plan.format.fps
    total_frames = plan.timeline_end_frame()
    max_per_10s = style.sfx.density_max_per_10s if style else 3.0

    kept = prune_to_budget(candidates_for(plan, structure), total_frames, fps, max_per_10s)

    out: list[Sfx] = []
    for i, candidate in enumerate(kept):
        sfx_id = pick_sfx_id(candidate.category)
        if sfx_id is None:
            continue  # nothing licensed in that category; skip rather than guess
        entry = Sfx(
            id=f"auto{i:03d}",
            sfx=sfx_id,
            anchor_frame=None if candidate.end_anchored else candidate.frame,
            end_on_frame=candidate.frame if candidate.end_anchored else None,
            gain_db=candidate.gain_db,
            source="auto_spotting",
        )
        out.append(entry)
    return out


def apply_spotting(plan: EditPlan, **kwargs) -> EditPlan:
    """Replace the plan's auto-spotted SFX, preserving agent and user placements."""
    out = plan.model_copy(deep=True)
    manual = [f for f in out.sfx if f.source != "auto_spotting"]
    out.sfx = manual + spot(out, **kwargs)
    return out
