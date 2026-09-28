"""The rails: apply snapping and relayout to a whole plan (§18).

`apply_rails` is the single non-bypassable transform between what the agent asks
for and what gets rendered. It snaps source times to deterministic boundaries,
snaps timeline positions to the musical grid, resolves durations expressed in beats
into frames, and then relays out the timeline so shots remain contiguous.

Relayout matters: snapping changes shot lengths, so without it every snap would
open a gap or an overlap further down the timeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from montaje.models.editplan import ComponentRef, Duration, EditPlan, Shot, SnapKind
from montaje.music.beats import BeatGrid
from montaje.plan.snap import SnapCandidates, snap_source_time, snap_timeline_frame


@dataclass
class RailContext:
    """Boundaries the rails snap to, per asset and on the timeline."""

    candidates: dict[str, SnapCandidates] = field(default_factory=dict)
    grid: BeatGrid | None = None
    fps: float = 30.0
    beat_window_ms: float = 120.0
    word_preroll_s: float = 0.08
    word_postroll_s: float = 0.15
    min_shot_s: float = 0.3
    # Asset durations, so a snap (especially word post-roll) cannot push a shot's
    # out-point past the end of its source file.
    asset_durations: dict[str, float] = field(default_factory=dict)

    @property
    def beats(self) -> list[float]:
        return self.grid.beats if self.grid else []

    @property
    def downbeats(self) -> list[float]:
        return self.grid.downbeats if self.grid else []


@dataclass
class RailReport:
    """What the rails changed, so the agent can see its plan was adjusted."""

    source_snaps: list[str] = field(default_factory=list)
    timeline_snaps: list[str] = field(default_factory=list)
    beat_durations_resolved: int = 0
    shots_relaid_out: int = 0
    dropped_shots: list[str] = field(default_factory=list)
    dropped_sections: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.source_snaps or self.timeline_snaps or self.dropped_shots
                    or self.dropped_sections or self.beat_durations_resolved
                    or self.shots_relaid_out)

    def summary(self) -> str:
        parts = [
            f"{len(self.source_snaps)} source snaps",
            f"{len(self.timeline_snaps)} timeline snaps",
            f"{self.beat_durations_resolved} beat durations resolved",
            f"{self.shots_relaid_out} shots relaid out",
        ]
        if self.dropped_shots:
            parts.append(f"{len(self.dropped_shots)} shots dropped")
        if self.dropped_sections:
            parts.append(f"{len(self.dropped_sections)} sections emptied")
        return "; ".join(parts)


def resolve_duration(duration: Duration | None, grid: BeatGrid | None, fps: float) -> int | None:
    """Musical durations become frames here, never earlier (§17.1)."""
    if duration is None:
        return None
    if duration.frames is not None:
        return duration.frames
    period = grid.beat_period_s if grid else 0.5
    return max(1, round((duration.beats or 0.0) * period * fps))


def _resolve_component(ref: ComponentRef | None, grid: BeatGrid | None, fps: float) -> bool:
    """Rewrite a component's beat-based duration into frames. Returns True if changed."""
    if ref is None or ref.duration is None or ref.duration.frames is not None:
        return False
    ref.duration = Duration(frames=resolve_duration(ref.duration, grid, fps))
    return True


def apply_rails(plan: EditPlan, ctx: RailContext) -> tuple[EditPlan, RailReport]:
    """Return a snapped, relaid-out copy of the plan plus a report of the changes."""
    out = plan.model_copy(deep=True)
    report = RailReport()
    fps = out.format.fps

    for shot in out.shots:
        _snap_shot_source(shot, ctx, report)
        if _resolve_component(shot.transition_in, ctx.grid, fps):
            report.beat_durations_resolved += 1
        for fx in shot.fx:
            if _resolve_component(fx, ctx.grid, fps):
                report.beat_durations_resolved += 1

    # Drop shots that snapping collapsed below the minimum length, rather than
    # rendering a frame-long flash.
    keep: list[Shot] = []
    for shot in out.shots:
        if shot.src_duration_s < ctx.min_shot_s - 1e-9:
            report.dropped_shots.append(shot.id)
        else:
            keep.append(shot)
    out.shots = keep

    _snap_and_relayout_timeline(out, ctx, report)
    _relayout_sections(out, report)
    return out, report


def _relayout_sections(plan: EditPlan, report: RailReport) -> None:
    """Move the concept's section boundaries onto the relaid-out timeline.

    Sections are timeline positions, so the rails own them just as much as the shots.
    Leaving them at their pre-rails frames makes everything that reads them wrong:
    SFX spotted at a section boundary land past the end of a shortened timeline, and
    the rhythm report attributes shots to the wrong section.
    """
    fps = plan.format.fps
    if not plan.concept.sections:
        return
    by_section: dict[str, list[tuple[int, int]]] = {}
    for shot in plan.sorted_shots():
        if shot.section is None:
            continue
        start = shot.timeline_in
        by_section.setdefault(shot.section, []).append(
            (start, start + shot.timeline_duration_frames(fps))
        )

    end_of_timeline = plan.timeline_end_frame()
    kept = []
    for section in plan.concept.sections:
        spans = by_section.get(section.id)
        if not spans:
            # A section whose every shot was dropped is no longer a section.
            report.dropped_sections.append(section.id)
            continue
        section.from_frame = min(s for s, _ in spans)
        section.to_frame = min(end_of_timeline, max(e for _, e in spans))
        kept.append(section)
    plan.concept.sections = kept


def _snap_shot_source(shot: Shot, ctx: RailContext, report: RailReport) -> None:
    cands = ctx.candidates.get(shot.asset, SnapCandidates())
    new_in = snap_source_time(
        shot.src_in, shot.snap.in_, cands,
        word_preroll_s=ctx.word_preroll_s, word_postroll_s=ctx.word_postroll_s,
    )
    new_out = snap_source_time(
        shot.src_out, shot.snap.out, cands,
        word_preroll_s=ctx.word_preroll_s, word_postroll_s=ctx.word_postroll_s,
    )
    # Clamp to the file: word post-roll in particular pushes the out-point past the
    # last word, which on a clip that ends mid-sentence is past the end of the file.
    duration = ctx.asset_durations.get(shot.asset)
    if duration is not None:
        new_out = min(new_out, duration)
        new_in = max(0.0, min(new_in, duration - ctx.min_shot_s))
    # A snap that would invert or empty the range is not applied.
    if new_out - new_in < ctx.min_shot_s:
        return
    if abs(new_in - shot.src_in) > 1e-6:
        report.source_snaps.append(
            f"{shot.id} in {shot.src_in:.3f}→{new_in:.3f} ({shot.snap.in_.value})"
        )
        shot.src_in = round(new_in, 3)
    if abs(new_out - shot.src_out) > 1e-6:
        report.source_snaps.append(
            f"{shot.id} out {shot.src_out:.3f}→{new_out:.3f} ({shot.snap.out.value})"
        )
        shot.src_out = round(new_out, 3)


def _snap_and_relayout_timeline(plan: EditPlan, ctx: RailContext, report: RailReport) -> None:
    """Snap each shot's start to the grid where asked, then close gaps and overlaps.

    Shots are laid out head to tail in their existing order. Two subtleties:

    - a grid snap only ever moves a shot *forward*, because pulling it back would
      overlap its predecessor;
    - moving it forward opens a gap, which would render as black frames. The gap is
      closed by **extending the previous shot** into its handle material rather than
      by leaving it or abandoning the snap. Where no material is available, the snap
      is abandoned instead.
    """
    fps = plan.format.fps
    shots = plan.sorted_shots()
    cursor = 0
    for index, shot in enumerate(shots):
        target = cursor
        if shot.snap.in_ in (SnapKind.BEAT, SnapKind.DOWNBEAT, SnapKind.BAR):
            snapped = snap_timeline_frame(
                cursor, shot.snap.in_, fps, ctx.beats, ctx.downbeats,
                window_ms=ctx.beat_window_ms,
            )
            candidate = max(cursor, snapped)
            gap = candidate - cursor
            if gap > 0 and index > 0:
                if _extend(shots[index - 1], gap, fps, ctx):
                    target = candidate
                # else: keep `target = cursor` and let the shot sit off the grid,
                # which the rhythm report will show as an off-grid cut.
            else:
                target = candidate
            if target != cursor:
                report.timeline_snaps.append(
                    f"{shot.id} frame {cursor}→{target} ({shot.snap.in_.value})"
                )
        if shot.timeline_in != target:
            report.shots_relaid_out += 1
            shot.timeline_in = target
        cursor = target + shot.timeline_duration_frames(fps)
    plan.shots = shots


def _extend(shot: Shot, frames: int, fps: float, ctx: RailContext) -> bool:
    """Lengthen a shot by exactly `frames` timeline frames. True if applied.

    Works in frames, not seconds: adding `frames / fps` to `src_out` and letting
    `timeline_duration_frames` round the result can land one frame either side of the
    target, which shows up as a one-frame overlap with the following shot. The new
    out-point is therefore searched for and verified against the frame count.
    """
    if frames <= 0:
        return True
    target_frames = shot.timeline_duration_frames(fps) + frames
    duration = ctx.asset_durations.get(shot.asset)
    original = shot.src_out
    # Start from the arithmetic guess and nudge by single frame-steps until the
    # rounded timeline length matches exactly.
    guess = shot.src_in + target_frames / fps
    step = 1.0 / (fps * 4)
    for offset in (0, 1, -1, 2, -2):
        candidate = guess + offset * step
        if candidate <= shot.src_in or (duration is not None and candidate > duration):
            continue
        shot.src_out = round(candidate, 4)
        if shot.timeline_duration_frames(fps) == target_frames:
            return True
    shot.src_out = original
    return False
