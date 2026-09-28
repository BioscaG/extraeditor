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


def _rank_removable(structure: MusicStructure, phrases: list[tuple[float, float]]) -> list[int]:
    """Phrase indices ordered by how safe they are to drop.

    Prefer low-energy, mid-track phrases: the intro establishes the track, the
    outro ends it, and the drop is the payoff the edit is built around.
    """
    n = len(phrases)
    scored: list[tuple[float, int]] = []
    for i, (t0, t1) in enumerate(phrases):
        mid = (t0 + t1) / 2
        section = structure.section_at(mid)
        energy = section.energy if section else 0.5
        role = section.role if section else "build"
        penalty = {"drop": 3.0, "intro": 2.0, "outro": 2.0}.get(role, 0.0)
        # Edge phrases are structurally load-bearing even inside a long section.
        if i == 0 or i == n - 1:
            penalty += 2.0
        scored.append((energy + penalty, i))
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

    best: MusicFit | None = None
    for phrase_bars in PHRASE_BAR_OPTIONS:
        phrases = phrase_spans(structure, phrase_bars)
        if len(phrases) < 2:
            continue
        fit = _fit_with_phrases(structure, phrases, target_s, crossfade_frames, fps, ending, phrase_bars)
        if abs(fit.total_s - target_s) <= tolerance_s:
            return fit
        if best is None or abs(fit.total_s - target_s) < abs(best.total_s - target_s):
            best = fit
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


def _fit_with_phrases(
    structure: MusicStructure,
    phrases: list[tuple[float, float]],
    target_s: float,
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
        for i in _rank_removable(structure, phrases):
            if duration - removed_s <= target_s:
                break
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
            if total >= target_s:
                break
            length = phrases[i][1] - phrases[i][0]
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
