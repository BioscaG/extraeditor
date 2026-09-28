"""Fit a track to a target duration by musical edits (§14.2).

The rule that makes this sound professional rather than butchered: only ever
remove or repeat **whole phrases** (4/8/16 bars), and only ever join **on a
downbeat**. Cutting mid-phrase is instantly audible even to untrained listeners,
because the listener's internal bar count breaks.

Shortening removes phrases; lengthening repeats them. Joins get a short
equal-power crossfade. The resulting `MusicEdit` list is what the render stage
consumes and what appears in the EditPlan.
"""

from __future__ import annotations

from dataclasses import dataclass

from montaje.models.editplan import MusicEdit, MusicEnding, MusicJoin
from montaje.music.structure import MusicStructure

# Phrase lengths to try, longest first: removing one 16-bar phrase is less audible
# than removing four 4-bar ones.
PHRASE_BAR_OPTIONS = (16, 8, 4)


def fits_within(duration_s: float, budget_s: float, fps: float = 30.0) -> bool:
    """Whether `duration_s` fits `budget_s`, compared at frame resolution.

    Phrase lengths come from a measured beat period, so they are never exact round
    numbers: two 8-second phrases sum to 16.0016 s and fail a literal `<= 16.0`. A
    sub-frame difference is below the timeline's own resolution and must not change a
    decision — this exact comparison silently dropped the intro and outro from short
    edits twice before being tied to the frame grid.
    """
    return round(duration_s * fps) <= round(budget_s * fps)


@dataclass(frozen=True)
class MusicFit:
    edits: list[MusicEdit]
    ending: MusicEnding
    total_s: float
    removed_s: float
    repeated_s: float
    phrase_bars: int
    notes: str = ""

    @property
    def joins(self) -> int:
        return sum(1 for e in self.edits if e.join is not None)


def phrase_spans(structure: MusicStructure, phrase_bars: int) -> list[tuple[float, float]]:
    """Contiguous `(t0, t1)` phrase spans covering the whole track.

    Fragments shorter than half a phrase are merged into their neighbour rather
    than emitted. The track's first downbeat is almost never at exactly 0.0, so
    naively prepending 0.0 produces a millisecond-long "phrase" — which then
    rounds to zero timeline frames and silently corrupts the edit's timing.
    Any audio before the first downbeat is a pickup and belongs to the first phrase.
    """
    duration = structure.duration_s
    downbeats = structure.grid.downbeats
    if not downbeats or duration <= 0:
        return [(0.0, duration)] if duration > 0 else []

    starts = list(downbeats[::phrase_bars])
    if not starts:
        return [(0.0, duration)]
    # Absorb the pickup before the first downbeat into the first phrase.
    starts[0] = 0.0
    edges = [*starts, duration]

    spans: list[tuple[float, float]] = []
    min_len = 0.5 * phrase_bars * structure.grid.beat_period_s * structure.grid.beats_per_bar
    for a, b in zip(edges, edges[1:], strict=False):
        if b - a <= 1e-6:
            continue
        if b - a < min_len and spans:
            # Extend the previous phrase instead of emitting a fragment.
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    return spans


# How strongly each section role resists being cut. The drop is the payoff the whole
# edit is built around, so it outranks everything — including the combined protection
# an intro phrase gets from being both an intro and the first phrase. Getting this
# ordering wrong silently produced short edits with no drop at all.
ROLE_PROTECTION = {"drop": 8.0, "intro": 2.0, "outro": 2.0, "build": 0.5, "break": 0.0}
EDGE_PROTECTION = 2.0


def _rank_removable(structure: MusicStructure, phrases: list[tuple[float, float]]) -> list[int]:
    """Phrase indices ordered by how safe they are to drop, safest first.

    Prefer low-energy, mid-track phrases: the intro establishes the track and the outro
    ends it, but the drop is what the viewer came for.
    """
    n = len(phrases)
    scored: list[tuple[float, int]] = []
    for i, (t0, t1) in enumerate(phrases):
        mid = (t0 + t1) / 2
        section = structure.section_at(mid)
        energy = section.energy if section else 0.5
        role = section.role if section else "build"
        protection = ROLE_PROTECTION.get(role, 0.0)
        # Edge phrases are structurally load-bearing even inside a long section.
        if i == 0 or i == n - 1:
            protection += EDGE_PROTECTION
        scored.append((energy + protection, i))
    return [i for _, i in sorted(scored)]


def fit_to_duration(
    structure: MusicStructure,
    target_s: float,
    tolerance_s: float = 2.0,
    crossfade_frames: int = 3,
    fps: float = 30.0,
    ending: str = "hit_with_tail",
) -> MusicFit:
    """Build the `MusicEdit` list that brings the track to `target_s`."""
    duration = structure.duration_s
    if duration <= 0:
        return MusicFit(edits=[], ending=MusicEnding(type=ending), total_s=0.0,
                        removed_s=0.0, repeated_s=0.0, phrase_bars=0,
                        notes="empty track")

    candidates: list[MusicFit] = []
    for phrase_bars in PHRASE_BAR_OPTIONS:
        phrases = phrase_spans(structure, phrase_bars)
        if len(phrases) < 2:
            continue
        candidates.append(_fit_with_phrases(
            structure, phrases, target_s, tolerance_s, crossfade_frames, fps,
            ending, phrase_bars,
        ))

    # Among fits that hit the target, prefer the one that preserves the most section
    # roles, then the longest phrases. Returning the first acceptable grid instead
    # meant a short target got the coarser grid and lost the intro and outro — the
    # coarse grid only wins on join inaudibility, and keeping the track's shape
    # matters more than that.
    acceptable = [
        f for f in candidates if fits_within(abs(f.total_s - target_s), tolerance_s, fps)
    ]
    if acceptable:
        return max(acceptable, key=lambda f: _structure_score(structure, f, target_s, fps))

    best = min(candidates, key=lambda f: abs(f.total_s - target_s)) if candidates else None
    if best is None:
        # Shorter than one phrase at every grid: nothing can be removed or repeated
        # without cutting mid-phrase, so pass the track through unedited.
        return MusicFit(
            edits=[MusicEdit(src_in=0.0, src_out=round(duration, 4), timeline_in=0)],
            ending=MusicEnding(type=ending), total_s=round(duration, 4),
            removed_s=0.0, repeated_s=0.0, phrase_bars=0,
            notes="track shorter than one phrase; passed through unedited",
        )
    return best


def _structure_score(
    structure: MusicStructure, fit: MusicFit, target_s: float, fps: float
) -> tuple:
    """Rank a candidate fit. Higher is better; compared lexicographically.

    The drop comes first and outranks the role count, because a short edit that keeps
    an intro and an outro but loses the drop has kept the packaging and thrown away the
    contents. Phrase length is the last structural tie-break: a coarser grid makes the
    joins less audible, which matters only once the shape is right.
    """
    roles = {s.role for s in timeline_sections(structure, fit, fps)}
    has_drop = "drop" in roles or not any(s.role == "drop" for s in structure.sections)
    return (has_drop, len(roles), fit.phrase_bars, -abs(fit.total_s - target_s))


def _fit_with_phrases(
    structure: MusicStructure,
    phrases: list[tuple[float, float]],
    target_s: float,
    tolerance_s: float,
    crossfade_frames: int,
    fps: float,
    ending: str,
    phrase_bars: int,
) -> MusicFit:
    duration = sum(b - a for a, b in phrases)
    keep = [True] * len(phrases)
    removed_s = 0.0
    repeated: list[int] = []

    if target_s < duration:
        # The first and last phrases give the edit an arrival and an ending. Keeping
        # them is what stops a short target from collapsing into a bare drop with no
        # beginning or end — so they are excluded from removal entirely rather than
        # merely penalised, as long as there is room for both plus something between.
        protected: set[int] = set()
        if len(phrases) >= 3:
            edge_s = (phrases[0][1] - phrases[0][0]) + (phrases[-1][1] - phrases[-1][0])
            shortest_middle = min(b - a for a, b in phrases[1:-1])
            # Protect the edges when the result can still hold them plus at least one
            # middle phrase. The budget is the target *plus its tolerance*, because
            # spending the tolerance to keep a beginning and an ending is exactly what
            # it is for — an arbitrary fraction of the target instead missed this case
            # by 1.6 milliseconds.
            if fits_within(edge_s + shortest_middle, target_s + tolerance_s, fps):
                protected = {0, len(phrases) - 1}

        for i in _rank_removable(structure, phrases):
            if duration - removed_s <= target_s:
                break
            if i in protected:
                continue
            length = phrases[i][1] - phrases[i][0]
            if duration - removed_s - length < target_s - length / 2:
                continue  # dropping this one would overshoot below the target
            keep[i] = False
            removed_s += length
    elif target_s > duration:
        # Repeat the highest-energy non-edge phrases: a longer build or drop reads
        # as intentional, a repeated intro reads as a mistake.
        order = list(reversed(_rank_removable(structure, phrases)))
        total = duration
        for i in order:
            if fits_within(target_s, total, fps):
                break
            length = phrases[i][1] - phrases[i][0]
            # Stop if one more phrase would land further from the target than stopping
            # here does: overshooting by a whole phrase to cover a sub-frame shortfall
            # is the wrong trade, and comparing raw floats caused exactly that.
            if abs(total + length - target_s) > abs(total - target_s):
                break
            repeated.append(i)
            total += length

    # Emit edits in timeline order, merging adjacent kept phrases into one range so
    # the render does not crossfade where the audio is already continuous.
    edits: list[MusicEdit] = []
    timeline_s = 0.0
    pending: list[tuple[float, float]] = []

    def flush(needs_join: bool) -> None:
        nonlocal timeline_s
        if not pending:
            return
        src_in, src_out = pending[0][0], pending[-1][1]
        join = (
            MusicJoin(type="crossfade", frames=crossfade_frames, on="downbeat")
            if needs_join and edits
            else None
        )
        edits.append(MusicEdit(src_in=round(src_in, 4), src_out=round(src_out, 4),
                               timeline_in=round(timeline_s * fps), join=join))
        timeline_s += src_out - src_in
        pending.clear()

    discontinuity = False
    for i, (a, b) in enumerate(phrases):
        if not keep[i]:
            discontinuity = True
            continue
        if pending and discontinuity:
            flush(True)
            discontinuity = False
        pending.append((a, b))
        for _ in range(repeated.count(i)):
            flush(discontinuity)
            discontinuity = False
            pending.append((a, b))
            flush(True)
    flush(discontinuity)

    total = sum(e.src_out - e.src_in for e in edits)
    repeated_s = sum(phrases[i][1] - phrases[i][0] for i in repeated)
    return MusicFit(
        edits=edits,
        ending=MusicEnding(type=ending),
        total_s=round(total, 4),
        removed_s=round(removed_s, 4),
        repeated_s=round(repeated_s, 4),
        phrase_bars=phrase_bars,
        notes=f"{len(phrases)} phrases of {phrase_bars} bars; "
              f"{keep.count(False)} removed, {len(repeated)} repeated",
    )


@dataclass(frozen=True)
class TimelineSection:
    """A music section as it appears on the *edited* timeline."""

    id: str
    role: str
    from_frame: int
    to_frame: int
    energy: float

    @property
    def duration_frames(self) -> int:
        return self.to_frame - self.from_frame


def timeline_sections(
    structure: MusicStructure, fit: MusicFit, fps: float = 30.0, min_frames: int = 15
) -> list[TimelineSection]:
    """Project the track's sections through the music edit onto the timeline.

    Fitting to duration removes and repeats phrases, so a section that began at 48 s
    in the source may begin anywhere — or several times, or not at all — on the
    timeline. Building a plan against the *source* section times instead produces an
    edit whose structure does not match what the viewer hears, which is the whole
    point of cutting to the music.

    Adjacent fragments of the same section are merged, and fragments shorter than
    `min_frames` are absorbed into their neighbour rather than becoming a section.
    """
    if not structure.sections or not fit.edits:
        return []

    fragments: list[TimelineSection] = []
    for edit in fit.edits:
        cursor_frame = edit.timeline_in
        for section in structure.sections:
            overlap_in = max(edit.src_in, section.t0)
            overlap_out = min(edit.src_out, section.t1)
            if overlap_out <= overlap_in:
                continue
            offset = overlap_in - edit.src_in
            from_frame = cursor_frame + round(offset * fps)
            to_frame = from_frame + round((overlap_out - overlap_in) * fps)
            fragments.append(TimelineSection(
                id=section.id, role=section.role,
                from_frame=from_frame, to_frame=to_frame, energy=section.energy,
            ))
    fragments.sort(key=lambda f: f.from_frame)

    merged: list[TimelineSection] = []
    for fragment in fragments:
        if merged and merged[-1].role == fragment.role and fragment.from_frame <= merged[-1].to_frame + 1:
            previous = merged[-1]
            merged[-1] = TimelineSection(
                id=previous.id, role=previous.role,
                from_frame=previous.from_frame, to_frame=max(previous.to_frame, fragment.to_frame),
                energy=max(previous.energy, fragment.energy),
            )
        elif merged and fragment.duration_frames < min_frames:
            previous = merged[-1]
            merged[-1] = TimelineSection(
                id=previous.id, role=previous.role,
                from_frame=previous.from_frame, to_frame=fragment.to_frame,
                energy=previous.energy,
            )
        else:
            merged.append(fragment)

    # Re-number so ids are unique and ordered on the timeline: a repeated phrase can
    # otherwise produce two sections with the same id.
    return [
        TimelineSection(id=f"sec{i:02d}", role=s.role, from_frame=s.from_frame,
                        to_frame=s.to_frame, energy=s.energy)
        for i, s in enumerate(merged)
    ]


def validate_joins(fit: MusicFit, structure: MusicStructure, tolerance_s: float = 0.05) -> list[str]:
    """Hard rail: every join must land on a downbeat (§18.2)."""
    problems: list[str] = []
    downbeats = structure.grid.downbeats
    if not downbeats:
        return problems
    for e in fit.edits:
        if e.join is None:
            continue
        nearest = min(downbeats, key=lambda d: abs(d - e.src_in))
        if abs(nearest - e.src_in) > tolerance_s:
            problems.append(
                f"join at src {e.src_in:.3f}s is {abs(nearest - e.src_in) * 1000:.0f}ms "
                f"off the nearest downbeat ({nearest:.3f}s)"
            )
    return problems
