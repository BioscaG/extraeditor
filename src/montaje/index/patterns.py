"""Pattern mining over the footage index (§11).

This module is the test of §2.5 — *nothing project-specific in code*. The festival
footage's convention (open by uncovering the lens, close by covering it) must be
**discovered** here from meaning-free `occlusion.reveal` / `occlusion.cover` events, not
recognised by a rule that knows about hands or lenses.

So the miner asks only structural questions:

- how often does a clip *begin* with a particular kind of event?
- how often does it *end* with one?
- is that more often than chance, across enough clips to mean something?

The answer is a `Pattern` with counts, examples and a confidence — support for the agent,
which interprets it and proposes a `Convention` for the user to confirm (§12). The miner
never names what it found.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from montaje.models.asset import Asset
from montaje.models.conventions import (
    Convention,
    ConventionDetection,
    ConventionEvidence,
    ConventionTreatment,
)
from montaje.models.events import Event

# How near a clip's start or end an event must be to count as "at" it. Generous, because
# a reveal takes time and the clip may start slightly before it finishes.
EDGE_WINDOW_S = 2.0
# A motif needs to appear in at least this many clips and this share of them before it is
# worth proposing. Two clips is a coincidence; a third of a shoot is a habit.
MIN_CLIP_COUNT = 3
MIN_SHARE = 0.25
# A motif must also be *selective*. An event type present at the edge of essentially every
# clip is describing the analyzer, not the footage: per-second analyzers (motion, quality
# metrics, colour stats, subjects) cover every clip end to end by construction.
MAX_SHARE = 0.9
# An event type whose events cover more than this fraction of a clip is a description of
# the whole clip rather than a motif at its edge, so it cannot mark one. Measured rather
# than listed, so analyzers that do not exist yet are handled too.
MAX_COVERAGE = 0.5
# Event types that mark an *instant* rather than a span. A convention's treatment is to
# trim to its edges, so an edge event is what a snap can use — `occlusion.occluded`
# identifies the same clips as `occlusion.reveal` but there is nothing to trim *to* in the
# middle of a span.
EDGE_EVENT_TYPES = {"reveal", "cover", "word", "segment", "speech"}


@dataclass
class Pattern:
    """A structural regularity found across clips. Deliberately unnamed."""

    id: str
    kind: str  # "start_motif" | "end_motif" | "bookend"
    event_type: str  # the analyzer event that marks it, e.g. "occlusion.reveal"
    count: int
    total_clips: int
    assets: list[str] = field(default_factory=list)
    mean_offset_s: float = 0.0
    paired_event_type: str | None = None  # for bookends, the event at the other end

    @property
    def share(self) -> float:
        return self.count / self.total_clips if self.total_clips else 0.0

    @property
    def confidence(self) -> float:
        """0–1. Rises with both how many clips show it and what share of them do.

        Both matter independently: 3 clips out of 4 is suggestive but thin, and 10 out of
        100 is well-sampled but not a habit.
        """
        if self.total_clips == 0:
            return 0.0
        breadth = min(1.0, self.count / 10.0)
        return round(min(1.0, 0.45 * breadth + 0.55 * min(1.0, self.share / 0.7)), 4)

    @property
    def is_significant(self) -> bool:
        return (
            self.count >= MIN_CLIP_COUNT
            and MIN_SHARE <= self.share <= MAX_SHARE
        )

    @property
    def selectivity(self) -> float:
        """How much this motif distinguishes some clips from others.

        Peaks when about half the footage shows it. A motif in every clip separates
        nothing, and one in a handful is not yet a habit — this is what ranks the real
        convention above the dozens of coincidences that share its clip count.
        """
        return round(1.0 - abs(self.share - 0.5) * 2, 4)

    def _edge_score(self) -> int:
        """How many of this pattern's event types mark an instant rather than a span."""
        types = [self.event_type]
        if self.paired_event_type:
            types.append(self.paired_event_type)
        return sum(1 for t in types if t.split(".")[-1] in EDGE_EVENT_TYPES)

    def describe(self) -> str:
        """A factual description with no interpretation of what the motif *means*."""
        where = {"start_motif": "begin", "end_motif": "end", "bookend": "both begin and end"}
        subject = self.event_type.split(".")[-1]
        if self.kind == "bookend" and self.paired_event_type:
            other = self.paired_event_type.split(".")[-1]
            return (
                f"{self.count} of {self.total_clips} clips begin with a "
                f"'{subject}' event and end with a '{other}' event "
                f"(within {EDGE_WINDOW_S:.0f}s of each edge)."
            )
        return (
            f"{self.count} of {self.total_clips} clips {where[self.kind]} with a "
            f"'{subject}' event (within {EDGE_WINDOW_S:.0f}s of the edge)."
        )


def _coverage(events: list[Event], duration_s: float) -> dict[str, float]:
    """Fraction of the clip each event type's events span, by type."""
    if duration_s <= 0:
        return {}
    totals: dict[str, float] = defaultdict(float)
    for event in events:
        key = f"{event.analyzer.split('@')[0]}.{event.type}"
        totals[key] += max(0.0, event.t1 - event.t0)
    return {key: total / duration_s for key, total in totals.items()}


def _edge_events(
    events: list[Event], duration_s: float, window_s: float
) -> tuple[dict[str, float], dict[str, float]]:
    """Event types occurring near the start and near the end, with their offsets.

    Event types that blanket the clip are skipped. A per-second analyzer emits something
    at every clip's first and last second by construction, so treating that as a motif
    buries the real ones: on a 12-clip shoot it produced 70 spurious patterns, several at
    confidence 1.00, above the one genuine convention.
    """
    coverage = _coverage(events, duration_s)
    at_start: dict[str, float] = {}
    at_end: dict[str, float] = {}
    for event in events:
        name = event.analyzer.split("@")[0]
        # Shot boundaries are at every clip's edges by definition.
        if name == "shots":
            continue
        key = f"{name}.{event.type}"
        if coverage.get(key, 0.0) > MAX_COVERAGE:
            continue
        mid = (event.t0 + event.t1) / 2
        if mid <= window_s:
            at_start.setdefault(key, mid)
        if duration_s - mid <= window_s:
            at_end.setdefault(key, duration_s - mid)
    return at_start, at_end


def mine_patterns(
    assets: dict[str, Asset],
    events: dict[str, list[Event]],
    *,
    window_s: float = EDGE_WINDOW_S,
) -> list[Pattern]:
    """Structural motifs across the footage, most confident first."""
    total = len([a for a in assets.values() if a.duration_s > 0])
    if total == 0:
        return []

    starts: dict[str, list[tuple[str, float]]] = defaultdict(list)
    ends: dict[str, list[tuple[str, float]]] = defaultdict(list)
    bookends: dict[tuple[str, str], list[str]] = defaultdict(list)

    for asset_id, asset in assets.items():
        if asset.duration_s <= 0:
            continue
        at_start, at_end = _edge_events(events.get(asset_id, []), asset.duration_s, window_s)
        for key, offset in at_start.items():
            starts[key].append((asset_id, offset))
        for key, offset in at_end.items():
            ends[key].append((asset_id, offset))
        # A bookend is the stronger claim: the clip is framed at both ends. Recording it
        # separately lets the agent see that the two motifs are the same habit rather
        # than two coincidences.
        for start_key in at_start:
            for end_key in at_end:
                bookends[(start_key, end_key)].append(asset_id)

    patterns: list[Pattern] = []
    for key, hits in starts.items():
        patterns.append(Pattern(
            id=f"start.{key}", kind="start_motif", event_type=key,
            count=len(hits), total_clips=total,
            assets=sorted(a for a, _ in hits),
            mean_offset_s=round(sum(o for _, o in hits) / len(hits), 3),
        ))
    for key, hits in ends.items():
        patterns.append(Pattern(
            id=f"end.{key}", kind="end_motif", event_type=key,
            count=len(hits), total_clips=total,
            assets=sorted(a for a, _ in hits),
            mean_offset_s=round(sum(o for _, o in hits) / len(hits), 3),
        ))
    for (start_key, end_key), hits in bookends.items():
        patterns.append(Pattern(
            id=f"bookend.{start_key}~{end_key}", kind="bookend",
            event_type=start_key, paired_event_type=end_key,
            count=len(hits), total_clips=total, assets=sorted(hits),
        ))

    significant = [p for p in patterns if p.is_significant]
    # Rank by selectivity first, then confidence. A motif that separates half the footage
    # from the other half is the interesting one; sheer prevalence mostly measures how the
    # analyzers sample. Bookends outrank their own single-edge components at equal score,
    # so the agent is not offered three conventions for one habit.
    significant.sort(
        key=lambda p: (
            -p.selectivity,
            p.kind != "bookend",
            -p._edge_score(),
            -p.confidence,
            -p.count,
        )
    )
    return _collapse_duplicates(significant)


def _collapse_duplicates(patterns: list[Pattern]) -> list[Pattern]:
    """Keep one pattern per distinct set of clips.

    Several patterns describing exactly the same clips are one finding seen from different
    angles — a span event and its own edges, for instance. Offering all of them makes the
    agent propose four conventions for one habit. The list is already ordered so the most
    actionable comes first, so the first for each clip set is the one to keep.
    """
    out: list[Pattern] = []
    seen: set[frozenset[str]] = set()
    for pattern in patterns:
        key = frozenset(pattern.assets)
        if key in seen:
            continue
        seen.add(key)
        out.append(pattern)
    return out


def propose_conventions(patterns: list[Pattern]) -> list[Convention]:
    """Turn patterns into `proposed` conventions for the user to confirm (§12).

    The description states what was measured; the treatment is a *suggestion*. Nothing
    here decides that a motif means anything — that is the user's call, and until they
    confirm it the convention has no effect on any edit.
    """
    out: list[Convention] = []
    for pattern in patterns:
        if pattern.kind == "bookend":
            detection = ConventionDetection(
                start_event=pattern.event_type,
                end_event=pattern.paired_event_type,
                max_offset_s=EDGE_WINDOW_S,
            )
            treatment = ConventionTreatment(
                role="segment_boundary", trim_outside=True,
                use_as_transition=None,
            )
        elif pattern.kind == "start_motif":
            detection = ConventionDetection(start_event=pattern.event_type,
                                            max_offset_s=EDGE_WINDOW_S)
            treatment = ConventionTreatment(role="clip_opening", trim_outside=True)
        else:
            detection = ConventionDetection(end_event=pattern.event_type,
                                            max_offset_s=EDGE_WINDOW_S)
            treatment = ConventionTreatment(role="clip_closing", trim_outside=True)

        out.append(Convention(
            id=pattern.id.replace(".", "_").replace("~", "_to_"),
            description=(
                pattern.describe()
                + " What this means is for you to say; nothing is applied until confirmed."
            ),
            evidence=ConventionEvidence(count=pattern.count, assets=pattern.assets[:12]),
            detection=detection,
            treatment=treatment,
        ))
    return out


def summarize(patterns: list[Pattern]) -> str:
    """A short report for the agent and for `montaje debug patterns`."""
    if not patterns:
        return "No recurring structural motifs found in this footage."
    lines = ["# Footage patterns", ""]
    for pattern in patterns:
        lines.append(
            f"- **{pattern.id}** (confidence {pattern.confidence:.2f}): {pattern.describe()}"
        )
        lines.append(f"  examples: {', '.join(pattern.assets[:5])}")
    lines += [
        "",
        "These are measurements, not meanings. Interpreting them — and deciding whether "
        "to treat one as a segment boundary or a transition — is the agent's job, "
        "confirmed by the user (§12).",
    ]
    return "\n".join(lines)
