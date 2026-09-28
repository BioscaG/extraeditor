"""Rhythm report (§18.3).

The critic watches a sparsely sampled preview and cannot perceive micro-timing, so
rhythm is guaranteed *here*, from the plan and the analysis, not by a model looking
at the result. The agent must read this before every preview.

Reported: cut-to-beat offsets (on-grid vs deliberately off-grid), shot-length
distribution per section against style targets, visual-versus-music energy
correlation, face/`us` screen time, and transition and SFX density.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from montaje.models.editplan import EditPlan
from montaje.models.events import Event
from montaje.models.style import Style
from montaje.music.structure import MusicStructure
from montaje.plan.snap import MAX_MEASURE_FRACTION, grid_window_ms


@dataclass
class SectionRhythm:
    section_id: str
    role: str
    shots: int
    avg_shot_s: float
    avg_shot_beats: float
    target_shot_beats: float | None
    shortest_s: float
    longest_s: float
    transitions_per_10s: float
    sfx_per_10s: float
    visual_energy: float
    music_energy: float

    @property
    def on_target(self) -> bool | None:
        """Whether average shot length is within 35% of the style target."""
        if self.target_shot_beats is None:
            return None
        return abs(self.avg_shot_beats - self.target_shot_beats) <= 0.35 * self.target_shot_beats


@dataclass
class RhythmReport:
    total_s: float
    shots: int
    on_grid_ratio: float
    target_on_grid_ratio: float | None
    beat_offsets_ms: list[float] = field(default_factory=list)
    sections: list[SectionRhythm] = field(default_factory=list)
    # Spread of shot durations, as std/mean. What makes an edit feel made by a person is
    # that its shots are not all the same length — not that its cuts miss the grid.
    length_variety: float | None = None
    energy_correlation: float | None = None
    us_share: float | None = None
    face_screen_time: dict[str, float] = field(default_factory=dict)
    transitions_per_10s: float = 0.0
    sfx_per_10s: float = 0.0
    problems: list[str] = field(default_factory=list)

    def offset_histogram(self, bins: tuple[float, ...] = (20, 40, 80, 120, 250)) -> dict[str, int]:
        """Cut-to-beat offsets bucketed by magnitude, in milliseconds."""
        out: dict[str, int] = {}
        prev = 0.0
        for edge in bins:
            out[f"≤{edge:.0f}ms"] = sum(1 for o in self.beat_offsets_ms if prev <= abs(o) < edge)
            prev = edge
        out[f">{bins[-1]:.0f}ms"] = sum(1 for o in self.beat_offsets_ms if abs(o) >= bins[-1])
        return out

    def to_markdown(self) -> str:
        lines = [
            "# Rhythm report",
            "",
            f"- **Duration:** {self.total_s:.1f}s across {self.shots} shots "
            f"(avg {self.total_s / max(1, self.shots):.2f}s)",
            f"- **On-grid cuts:** {self.on_grid_ratio * 100:.0f}%"
            + (f" (style target {self.target_on_grid_ratio * 100:.0f}%)"
               if self.target_on_grid_ratio is not None else ""),
            f"- **Transitions:** {self.transitions_per_10s:.1f} per 10s · "
            f"**SFX:** {self.sfx_per_10s:.1f} per 10s",
        ]
        if self.length_variety is not None:
            lines.append(f"- **Shot-length spread:** {self.length_variety:.2f}")
        if self.energy_correlation is not None:
            lines.append(f"- **Visual/music energy correlation:** {self.energy_correlation:+.2f}")
        if self.us_share is not None:
            lines.append(f"- **Shots with us in frame:** {self.us_share * 100:.0f}%")
        lines.append("")
        lines.append("## Cut-to-beat offsets")
        lines.append("")
        for bucket, n in self.offset_histogram().items():
            lines.append(f"- {bucket}: {n}")
        if self.sections:
            lines += ["", "## Per section", "",
                      "| section | role | shots | avg shot | target | on target | vis/mus energy |",
                      "|---|---|---|---|---|---|---|"]
            for s in self.sections:
                target = f"{s.target_shot_beats:.2f} beats" if s.target_shot_beats else "—"
                mark = {True: "yes", False: "**no**", None: "—"}[s.on_target]
                lines.append(
                    f"| {s.section_id} | {s.role} | {s.shots} | {s.avg_shot_beats:.2f} beats "
                    f"({s.avg_shot_s:.2f}s) | {target} | {mark} | "
                    f"{s.visual_energy:.2f} / {s.music_energy:.2f} |"
                )
        if self.face_screen_time:
            lines += ["", "## Face screen time", ""]
            for person, seconds in sorted(self.face_screen_time.items(), key=lambda kv: -kv[1]):
                lines.append(f"- {person}: {seconds:.1f}s ({seconds / max(self.total_s, 1e-9) * 100:.0f}%)")
        if self.problems:
            lines += ["", "## Problems", ""] + [f"- {p}" for p in self.problems]
        return "\n".join(lines)


def _visual_energy(events: list[Event], t0: float, t1: float) -> float:
    """Mean motion + subject activity over a source range, from `motion` events."""
    values = [
        e.data.get("camera_magnitude", 0.0) + e.data.get("subject_energy", 0.0) / 10.0
        for e in events
        if e.analyzer.startswith("motion") and e.t1 > t0 and e.t0 < t1
    ]
    return float(np.mean(values)) if values else 0.0


def build_rhythm_report(
    plan: EditPlan,
    *,
    structure: MusicStructure | None = None,
    style: Style | None = None,
    events: dict[str, list[Event]] | None = None,
    clip_people: dict[str, bool] | None = None,
) -> RhythmReport:
    fps = plan.format.fps
    shots = plan.sorted_shots()
    total_s = plan.timeline_end_frame() / fps
    events = events or {}
    grid = structure.grid if structure else None
    beat_period = grid.beat_period_s if grid else 0.5

    # Cut-to-beat offsets: measured at every cut point, including the last shot's out.
    #
    # Measured against the *half-beat* grid, because that is the grid the edit is built on.
    # A cut on an off-beat is as musical as one on a beat — every fast edit cuts on off-beats
    # — so scoring it as a miss described a defect that was not there, while a genuinely
    # sloppy cut 200ms from anything scored the same.
    offsets_ms: list[float] = []
    positions = grid.half_beats if grid else []
    if positions:
        cut_times = [s.timeline_in / fps for s in shots]
        if shots:
            last = shots[-1]
            cut_times.append((last.timeline_in + last.timeline_duration_frames(fps)) / fps)
        for t in cut_times:
            nearest = min(positions, key=lambda b: abs(b - t))
            offsets_ms.append((t - nearest) * 1000.0)

    # Scaled to the grid's spacing, for the same reason the rail's window is: a fixed 120ms
    # against 250ms half-beat positions calls every frame on-grid and measures nothing. The
    # measuring fraction is tighter than the rail's — the rail may move a cut half a grid
    # step to place it, but only a cut that lands close counts as placed.
    window_ms = grid_window_ms(positions, 120.0, MAX_MEASURE_FRACTION)
    on_grid = sum(1 for o in offsets_ms if abs(o) <= window_ms)
    on_grid_ratio = on_grid / len(offsets_ms) if offsets_ms else 0.0

    per_10s = max(total_s / 10.0, 1e-9)
    n_transitions = sum(1 for s in shots if s.transition_in is not None)
    report = RhythmReport(
        total_s=round(total_s, 3),
        shots=len(shots),
        on_grid_ratio=round(on_grid_ratio, 4),
        target_on_grid_ratio=style.pacing.on_grid_ratio if style else None,
        beat_offsets_ms=[round(o, 1) for o in offsets_ms],
        transitions_per_10s=round(n_transitions / per_10s, 3),
        sfx_per_10s=round(len(plan.sfx) / per_10s, 3),
        length_variety=_length_variety(shots, fps),
    )

    # Per-section pacing against the style's targets.
    by_section: dict[str | None, list] = defaultdict(list)
    for s in shots:
        by_section[s.section].append(s)
    section_roles = {sec.id: (sec.music_section or sec.purpose or "") for sec in plan.concept.sections}

    visual_series: list[float] = []
    music_series: list[float] = []
    for section in plan.concept.sections:
        members = by_section.get(section.id, [])
        if not members:
            continue
        lengths = [s.timeline_duration_frames(fps) / fps for s in members]
        role = section.music_section or section_roles.get(section.id, "") or "build"
        target = style.pacing.avg_shot_beats.get(role) if style else None
        span_s = max(sum(lengths), 1e-9)

        vis = float(np.mean([
            _visual_energy(events.get(s.asset, []), s.src_in, s.src_out) for s in members
        ])) if members else 0.0
        mus = 0.0
        if structure and structure.energy_curve:
            lo = int(section.from_frame / fps)
            hi = max(lo + 1, int(section.to_frame / fps))
            segment = structure.energy_curve[lo:hi]
            mus = float(np.mean(segment)) if segment else 0.0
        visual_series.append(vis)
        music_series.append(mus)

        report.sections.append(SectionRhythm(
            section_id=section.id,
            role=role,
            shots=len(members),
            avg_shot_s=round(float(np.mean(lengths)), 3),
            avg_shot_beats=round(float(np.mean(lengths)) / beat_period, 3),
            target_shot_beats=target,
            shortest_s=round(min(lengths), 3),
            longest_s=round(max(lengths), 3),
            transitions_per_10s=round(
                sum(1 for s in members if s.transition_in is not None) / (span_s / 10.0), 3),
            sfx_per_10s=round(
                sum(1 for f in plan.sfx
                    if section.from_frame <= (f.anchor_frame or f.end_on_frame or -1) < section.to_frame)
                / (span_s / 10.0), 3),
            visual_energy=round(vis, 3),
            music_energy=round(mus, 3),
        ))

    # Does the picture get busier where the music does? (§18.3)
    if len(visual_series) >= 3 and np.std(visual_series) > 0 and np.std(music_series) > 0:
        report.energy_correlation = round(float(np.corrcoef(visual_series, music_series)[0, 1]), 3)

    if clip_people:
        with_us = sum(1 for s in shots if clip_people.get(s.asset, False))
        report.us_share = round(with_us / len(shots), 3) if shots else None

    report.problems = _rhythm_problems(report, style)
    return report


# Below this, shot lengths are close enough to identical that the cut rate reads as a
# metronome. Measured as std/mean: the style's own variation pattern produces about 0.45,
# and a section of constant-length shots produces 0.0.
MIN_LENGTH_VARIETY = 0.2

# Density caps are taste guidance, not arithmetic. Spotting that correctly aims at the cap
# lands a hair over it — 27 cues across 89.9s is 3.003 per 10s against a limit of 3.0 — and
# reporting that as a problem produced the message "3.0 SFX per 10s exceeds the style limit
# of 3.0", which is both true and useless.
DENSITY_TOLERANCE = 1.05


def _length_variety(shots: list, fps: float) -> float | None:
    """Spread of shot durations, as std/mean. `None` when there is nothing to compare."""
    if len(shots) < 3:
        return None
    lengths = [s.timeline_duration_frames(fps) / fps for s in shots]
    mean = sum(lengths) / len(lengths)
    if mean <= 0:
        return None
    variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    return round(variance**0.5 / mean, 3)


def _rhythm_problems(report: RhythmReport, style: Style | None) -> list[str]:
    problems: list[str] = []
    if report.target_on_grid_ratio is not None and (
        report.on_grid_ratio < report.target_on_grid_ratio - 0.15
    ):
        problems.append(
            f"only {report.on_grid_ratio * 100:.0f}% of cuts are on the beat grid "
            f"(target {report.target_on_grid_ratio * 100:.0f}%) — the edit will feel loose"
        )
    # What used to live here was a complaint about *every* cut being on the grid. It was
    # the wrong measure: cutting on the grid is what makes an edit feel cut to the music,
    # and §16.4's real requirement is that the shots not all be the same length.
    if report.length_variety is not None and report.length_variety < MIN_LENGTH_VARIETY:
        problems.append(
            f"shot lengths barely vary (spread {report.length_variety:.2f}) — a constant "
            f"cut rate is what makes an edit feel generated rather than edited (§16.4)"
        )
    for s in report.sections:
        if s.on_target is False:
            direction = "faster" if s.avg_shot_beats > (s.target_shot_beats or 0) else "slower"
            problems.append(
                f"section {s.section_id} ({s.role}) averages {s.avg_shot_beats:.2f} beats per shot "
                f"against a target of {s.target_shot_beats:.2f} — cut {direction}"
            )
    if report.energy_correlation is not None and report.energy_correlation < 0.2:
        problems.append(
            f"visual energy barely tracks the music (r={report.energy_correlation:+.2f}) — "
            f"the busiest footage should land in the loudest sections"
        )
    if style is not None:
        if report.transitions_per_10s > style.transitions.max_per_10s * DENSITY_TOLERANCE:
            problems.append(
                f"{report.transitions_per_10s:.1f} transitions per 10s exceeds the style limit of "
                f"{style.transitions.max_per_10s} — transition overuse is the most common amateur tell"
            )
        if report.sfx_per_10s > style.sfx.density_max_per_10s * DENSITY_TOLERANCE:
            problems.append(
                f"{report.sfx_per_10s:.1f} SFX per 10s exceeds the style limit of "
                f"{style.sfx.density_max_per_10s}"
            )
    if report.us_share is not None and report.us_share < 0.4:
        problems.append(
            f"only {report.us_share * 100:.0f}% of shots have us in frame — stage shots should be "
            f"punctuation, not the substance (§16.4)"
        )
    return problems
