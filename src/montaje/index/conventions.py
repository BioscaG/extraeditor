"""Storing, confirming and applying conventions (§12).

A convention is **data**, and it does nothing until the user confirms it. That is the
whole point of §2.5: the system discovers a regularity, describes it factually, and asks;
it never decides that a dark span at the start of a clip means someone had their hand over
the lens.

Once confirmed, a convention affects the edit in one specific way: shots taken from a clip
that matches it are trimmed inside the motif, so the covered frames never reach the
timeline. That is applied here rather than in the builder, so it applies equally to plans
the agent wrote by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from montaje.models.conventions import Convention, ConventionStatus
from montaje.models.editplan import EditPlan
from montaje.models.events import Event
from montaje.workspace import Workspace, atomic_write_text


def load_conventions(ws: Workspace) -> list[Convention]:
    if not ws.conventions_yaml.exists():
        return []
    data = yaml.safe_load(ws.conventions_yaml.read_text()) or []
    if isinstance(data, dict):
        data = data.get("conventions", [])
    return [Convention.model_validate(entry) for entry in data]


def save_conventions(ws: Workspace, conventions: list[Convention]) -> None:
    atomic_write_text(
        ws.conventions_yaml,
        yaml.safe_dump(
            [c.model_dump(mode="json", exclude_none=True) for c in conventions],
            sort_keys=False, allow_unicode=True,
        ),
    )


def merge_proposals(existing: list[Convention], proposed: list[Convention]) -> list[Convention]:
    """Add newly proposed conventions without disturbing decisions already made.

    A convention the user confirmed or rejected keeps that status even if the miner
    proposes it again — re-running analysis must not silently un-confirm a decision or
    re-ask a question that has been answered.
    """
    by_id = {c.id: c for c in existing}
    for candidate in proposed:
        if candidate.id in by_id:
            # Refresh the evidence, keep the verdict.
            current = by_id[candidate.id]
            current.evidence = candidate.evidence
            current.description = candidate.description
            continue
        by_id[candidate.id] = candidate
    return list(by_id.values())


def confirmed(conventions: list[Convention]) -> list[Convention]:
    return [c for c in conventions if c.status == ConventionStatus.CONFIRMED]


@dataclass(frozen=True)
class MotifSpan:
    """Where a convention's motif sits in one asset."""

    asset_id: str
    convention_id: str
    start_edge_s: float | None  # content is usable *after* this
    end_edge_s: float | None    # content is usable *before* this


def _edge_times(events: list[Event], event_type: str | None, at_start: bool,
                window_s: float) -> float | None:
    """The instant a named event marks near one edge of the clip, if any."""
    if not event_type:
        return None
    analyzer, _, kind = event_type.partition(".")
    candidates = [
        e for e in events
        if e.analyzer.split("@")[0] == analyzer and e.type == kind
    ]
    if not candidates:
        return None
    if at_start:
        # A reveal ends where usable content begins.
        near = [e for e in candidates if e.t1 <= window_s + (e.t1 - e.t0)]
        return max(e.t1 for e in near) if near else None
    duration = max((e.t1 for e in events), default=0.0)
    near = [e for e in candidates if duration - e.t0 <= window_s + (e.t1 - e.t0)]
    return min(e.t0 for e in near) if near else None


def find_motifs(
    convention: Convention,
    events_by_asset: dict[str, list[Event]],
) -> list[MotifSpan]:
    """Where a convention's motif appears, per asset."""
    window = convention.detection.max_offset_s
    out: list[MotifSpan] = []
    for asset_id, events in events_by_asset.items():
        start = _edge_times(events, convention.detection.start_event, True, window)
        end = _edge_times(events, convention.detection.end_event, False, window)
        if start is None and end is None:
            continue
        out.append(MotifSpan(asset_id, convention.id, start, end))
    return out


@dataclass
class ApplyReport:
    trimmed: list[str]
    dropped: list[str]

    @property
    def changed(self) -> bool:
        return bool(self.trimmed or self.dropped)

    def summary(self) -> str:
        parts = []
        if self.trimmed:
            parts.append(f"{len(self.trimmed)} shots trimmed inside a convention's motif")
        if self.dropped:
            parts.append(f"{len(self.dropped)} shots dropped (entirely inside the motif)")
        return "; ".join(parts) or "no shots affected by conventions"


def apply_conventions(
    plan: EditPlan,
    conventions: list[Convention],
    events_by_asset: dict[str, list[Event]],
    *,
    min_shot_s: float = 0.3,
) -> ApplyReport:
    """Trim shots out of confirmed conventions' motifs, in place.

    Only `trim_outside` conventions do anything here. A shot left shorter than
    `min_shot_s` is dropped rather than kept as a flash: if all of it was inside the motif
    then none of it was footage the shooter intended to be seen.
    """
    active = [
        c for c in confirmed(conventions)
        if c.treatment.trim_outside
    ]
    if not active:
        return ApplyReport([], [])

    spans: dict[str, MotifSpan] = {}
    for convention in active:
        for motif in find_motifs(convention, events_by_asset):
            spans[motif.asset_id] = motif

    trimmed: list[str] = []
    dropped: list[str] = []
    keep = []
    for shot in plan.shots:
        motif = spans.get(shot.asset)
        if motif is None:
            keep.append(shot)
            continue
        src_in, src_out = shot.src_in, shot.src_out
        if motif.start_edge_s is not None:
            src_in = max(src_in, motif.start_edge_s)
        if motif.end_edge_s is not None:
            src_out = min(src_out, motif.end_edge_s)
        if src_out - src_in < min_shot_s:
            dropped.append(shot.id)
            continue
        if (src_in, src_out) != (shot.src_in, shot.src_out):
            shot.src_in = round(src_in, 3)
            shot.src_out = round(src_out, 3)
            trimmed.append(shot.id)
        keep.append(shot)
    plan.shots = keep
    return ApplyReport(trimmed, dropped)


def usable_after_conventions(
    asset_id: str,
    duration_s: float,
    conventions: list[Convention],
    events: list[Event],
) -> tuple[float, float]:
    """The range of an asset a confirmed convention leaves usable.

    Used by the planner so it never *proposes* a shot inside a motif, rather than relying
    on the trim to fix it afterwards — a shot chosen inside the motif wastes the budget it
    was allocated.
    """
    lo, hi = 0.0, duration_s
    for convention in confirmed(conventions):
        if not convention.treatment.trim_outside:
            continue
        window = convention.detection.max_offset_s
        start = _edge_times(events, convention.detection.start_event, True, window)
        end = _edge_times(events, convention.detection.end_event, False, window)
        if start is not None:
            lo = max(lo, start)
        if end is not None:
            hi = min(hi, end)
    return (lo, hi) if hi > lo else (0.0, duration_s)
