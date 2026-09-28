"""Snapping: the LLM picks moments, code picks frames (§2.2, §18.1).

Vision-language models understand *what* happens but sample video at a few fps, so
their timestamps drift. Every decision the agent makes is therefore pulled onto a
deterministic boundary before it reaches the render: shot cuts, word boundaries,
silences and occlusion edges in source time; beats, downbeats and bars on the
timeline.

Snapping is a **rail**: non-bypassable, applied whatever the agent does (§2.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from montaje.models.editplan import SnapKind
from montaje.models.events import Event


@dataclass(frozen=True)
class SnapCandidates:
    """Deterministic boundaries available for one asset, in source seconds."""

    shots: list[float] = field(default_factory=list)
    word_starts: list[float] = field(default_factory=list)
    word_ends: list[float] = field(default_factory=list)
    silences: list[float] = field(default_factory=list)
    reveals: list[float] = field(default_factory=list)
    covers: list[float] = field(default_factory=list)

    def for_kind(self, kind: SnapKind) -> list[float]:
        return {
            SnapKind.SHOT: self.shots,
            SnapKind.WORD_START: self.word_starts,
            SnapKind.WORD_END: self.word_ends,
            SnapKind.SILENCE: self.silences,
            SnapKind.OCCLUSION_REVEAL: self.reveals,
            SnapKind.OCCLUSION_COVER: self.covers,
        }.get(kind, [])

    @classmethod
    def from_events(cls, events: list[Event]) -> SnapCandidates:
        shots: list[float] = []
        reveals: list[float] = []
        covers: list[float] = []
        silences: list[float] = []
        word_starts: list[float] = []
        word_ends: list[float] = []
        for e in events:
            name = e.analyzer.split("@")[0]
            if name == "shots" and e.type == "shot":
                shots.extend((e.t0, e.t1))
            elif name == "occlusion" and e.type == "reveal":
                # The usable frame is where the occlusion ends, not the event centre.
                reveals.append(e.t1)
            elif name == "occlusion" and e.type == "cover":
                covers.append(e.t0)
            elif name == "audio_events" and e.type == "silence":
                silences.extend((e.t0, e.t1))
            elif name == "asr" and e.type == "word":
                word_starts.append(e.t0)
                word_ends.append(e.t1)
            elif name == "vad" and e.type == "speech" and not word_starts:
                # Without ASR, VAD segment edges are the best speech boundaries available.
                word_starts.append(e.t0)
                word_ends.append(e.t1)
        return cls(
            shots=sorted(set(shots)),
            word_starts=sorted(set(word_starts)),
            word_ends=sorted(set(word_ends)),
            silences=sorted(set(silences)),
            reveals=sorted(set(reveals)),
            covers=sorted(set(covers)),
        )


def nearest(values: list[float], t: float, max_delta: float | None = None) -> float | None:
    """Closest value to `t`, or None if nothing is within `max_delta`."""
    if not values:
        return None
    best = min(values, key=lambda v: abs(v - t))
    if max_delta is not None and abs(best - t) > max_delta:
        return None
    return best


def snap_source_time(
    t: float,
    kind: SnapKind,
    candidates: SnapCandidates,
    *,
    word_preroll_s: float = 0.08,
    word_postroll_s: float = 0.15,
    max_delta_s: float = 0.5,
) -> float:
    """Snap a source time to the boundary its `SnapKind` names.

    Speech gets asymmetric padding: a little before the first word so the attack is
    not clipped, and more after the last so the sentence is allowed to land. Never
    inside a word (§18.1) — that is the most audible cut error there is.
    """
    if kind in (SnapKind.NONE, SnapKind.BEAT, SnapKind.DOWNBEAT, SnapKind.BAR):
        return t  # timeline-domain kinds, or no snapping requested
    target = nearest(candidates.for_kind(kind), t, max_delta_s)
    if target is None:
        return t
    if kind == SnapKind.WORD_START:
        return max(0.0, target - word_preroll_s)
    if kind == SnapKind.WORD_END:
        return target + word_postroll_s
    return target


def snap_timeline_frame(
    frame: int,
    kind: SnapKind,
    fps: float,
    beats: list[float],
    downbeats: list[float],
    bars: list[float] | None = None,
    *,
    window_ms: float = 120.0,
) -> int:
    """Snap a timeline frame to the musical grid, within `window_ms`.

    Outside the window the frame is left alone: the agent may deliberately place a
    cut off the grid (§16.4), and a rail that overrides that would flatten the edit.
    """
    grid = {
        SnapKind.BEAT: beats,
        SnapKind.DOWNBEAT: downbeats,
        SnapKind.BAR: bars if bars is not None else downbeats,
    }.get(kind)
    if not grid:
        return frame
    t = frame / fps
    target = nearest(grid, t, window_ms / 1000.0)
    return frame if target is None else round(target * fps)
