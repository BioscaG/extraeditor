"""EditPlan validation (§18.2).

Hard errors block a render; warnings are advisory and the agent may accept them
with a justified `intent`. Validation is a rail: it runs whatever the agent does,
and the agent cannot modify it (§2.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from montaje.models.asset import Asset
from montaje.models.editplan import AudioMode, EditPlan, Shot
from montaje.models.events import Event


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Problem:
    severity: Severity
    code: str
    message: str
    shot_id: str | None = None

    def __str__(self) -> str:
        where = f" [{self.shot_id}]" if self.shot_id else ""
        return f"{self.severity.value.upper()}{where} {self.code}: {self.message}"


@dataclass
class ValidationContext:
    """Everything validation needs to check a plan against reality."""

    assets: dict[str, Asset] = field(default_factory=dict)
    usable_spans: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    speech_spans: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] = field(default_factory=list)
    stable_components: set[str] = field(default_factory=set)
    licensed_sfx: set[str] = field(default_factory=set)
    min_shot_s: float = 0.3
    beat_window_ms: float = 120.0
    duration_target_s: float | None = None
    duration_tolerance_s: float = 15.0
    max_transitions_per_10s: float | None = None
    max_sfx_per_10s: float | None = None

    @classmethod
    def from_events(
        cls, assets: dict[str, Asset], events: dict[str, list[Event]], **kwargs
    ) -> ValidationContext:
        usable: dict[str, list[tuple[float, float]]] = {}
        speech: dict[str, list[tuple[float, float]]] = {}
        for asset_id, evs in events.items():
            for e in evs:
                name = e.analyzer.split("@")[0]
                if name == "quality" and e.type == "usable":
                    usable.setdefault(asset_id, []).append((e.t0, e.t1))
                elif name == "vad" and e.type == "speech":
                    speech.setdefault(asset_id, []).append((e.t0, e.t1))
        return cls(assets=assets, usable_spans=usable, speech_spans=speech, **kwargs)


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Length of the overlap between two spans."""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _covered_fraction(span: tuple[float, float], spans: list[tuple[float, float]]) -> float:
    length = span[1] - span[0]
    if length <= 0:
        return 1.0
    return min(1.0, sum(_overlaps(span, s) for s in spans) / length)


def validate(plan: EditPlan, ctx: ValidationContext) -> list[Problem]:
    """All problems in the plan, hard errors first."""
    problems: list[Problem] = []
    fps = plan.format.fps
    shots = plan.sorted_shots()

    problems += _check_shots(plan, shots, ctx, fps)
    problems += _check_timeline(plan, shots, fps)
    problems += _check_duration(plan, fps, ctx)
    problems += _check_components(plan, ctx)
    problems += _check_sfx(plan, ctx, fps)
    problems += _check_audio(plan, shots, ctx)
    problems += _check_density(plan, fps, ctx)

    problems.sort(key=lambda p: 0 if p.severity == Severity.ERROR else 1)
    return problems


def _check_shots(plan: EditPlan, shots: list[Shot], ctx: ValidationContext, fps: float) -> list[Problem]:
    out: list[Problem] = []
    seen_ids: set[str] = set()
    used_ranges: dict[str, list[tuple[float, float]]] = {}

    for shot in shots:
        if shot.id in seen_ids:
            out.append(Problem(Severity.ERROR, "duplicate_shot_id",
                               f"shot id {shot.id!r} appears more than once", shot.id))
        seen_ids.add(shot.id)

        asset = ctx.assets.get(shot.asset)
        if asset is None:
            out.append(Problem(Severity.ERROR, "unknown_asset",
                               f"asset {shot.asset!r} is not in the index", shot.id))
        elif shot.src_out > asset.duration_s + 1e-6:
            out.append(Problem(Severity.ERROR, "source_out_of_range",
                               f"src_out {shot.src_out:.3f}s exceeds asset duration "
                               f"{asset.duration_s:.3f}s", shot.id))

        if shot.src_in < 0:
            out.append(Problem(Severity.ERROR, "negative_source",
                               f"src_in {shot.src_in:.3f}s is before the asset start", shot.id))

        length_frames = shot.timeline_duration_frames(fps)
        if length_frames / fps < ctx.min_shot_s - 1e-9:
            out.append(Problem(Severity.ERROR, "shot_too_short",
                               f"{length_frames / fps:.3f}s is below the {ctx.min_shot_s}s minimum",
                               shot.id))

        # Unusable material is allowed only when the agent justified it.
        usable = ctx.usable_spans.get(shot.asset)
        if usable is not None:
            covered = _covered_fraction((shot.src_in, shot.src_out), usable)
            if covered < 0.5 and not shot.intent:
                out.append(Problem(Severity.ERROR, "outside_usable",
                                   f"only {covered * 100:.0f}% of the range is marked usable and no "
                                   f"intent justifies it", shot.id))
            elif covered < 0.5:
                out.append(Problem(Severity.WARNING, "outside_usable",
                                   f"only {covered * 100:.0f}% of the range is marked usable "
                                   f"(intent: {shot.intent})", shot.id))

        if not shot.intent:
            out.append(Problem(Severity.WARNING, "missing_intent",
                               "every shot should carry an intent (§16.4)", shot.id))

        if ctx.assets.get(shot.asset) is not None and ctx.assets[shot.asset].degraded:
            out.append(Problem(Severity.WARNING, "degraded_asset",
                               f"asset {shot.asset} is flagged degraded: "
                               f"{ctx.assets[shot.asset].degraded_reason}", shot.id))

        prior = used_ranges.setdefault(shot.asset, [])
        if any(_overlaps((shot.src_in, shot.src_out), p) > 0.1 for p in prior):
            out.append(Problem(Severity.WARNING, "reused_source",
                               "this source range is already used by another shot", shot.id))
        prior.append((shot.src_in, shot.src_out))

        if shot.section is not None and shot.section not in {s.id for s in plan.concept.sections}:
            out.append(Problem(Severity.WARNING, "unknown_section",
                               f"section {shot.section!r} is not declared in the concept", shot.id))
    return out


def _check_timeline(plan: EditPlan, shots: list[Shot], fps: float) -> list[Problem]:
    """No gaps and no unintended overlaps: a gap renders as black frames (§18.2)."""
    out: list[Problem] = []
    if not shots:
        out.append(Problem(Severity.ERROR, "empty_timeline", "the plan has no shots"))
        return out
    if shots[0].timeline_in != 0:
        out.append(Problem(Severity.ERROR, "timeline_gap",
                           f"timeline starts at frame {shots[0].timeline_in}, not 0", shots[0].id))
    for a, b in zip(shots, shots[1:], strict=False):
        end = a.timeline_in + a.timeline_duration_frames(fps)
        # A transition on the incoming shot legitimately overlaps the outgoing one.
        overlap_allowance = 0
        if b.transition_in is not None and b.transition_in.duration is not None:
            overlap_allowance = b.transition_in.duration.frames or 0
        if b.timeline_in > end:
            out.append(Problem(Severity.ERROR, "timeline_gap",
                               f"{b.timeline_in - end} frame gap before this shot", b.id))
        elif b.timeline_in < end - overlap_allowance:
            out.append(Problem(Severity.ERROR, "timeline_overlap",
                               f"overlaps the previous shot by {end - b.timeline_in} frames", b.id))
    return out


def _check_duration(plan: EditPlan, fps: float, ctx: ValidationContext) -> list[Problem]:
    if ctx.duration_target_s is None:
        return []
    total_s = plan.timeline_end_frame() / fps
    delta = abs(total_s - ctx.duration_target_s)
    if delta > ctx.duration_tolerance_s:
        return [Problem(Severity.ERROR, "duration_out_of_tolerance",
                        f"{total_s:.1f}s is {delta:.1f}s from the {ctx.duration_target_s:.0f}s target "
                        f"(tolerance {ctx.duration_tolerance_s:.0f}s)")]
    return []


def _check_components(plan: EditPlan, ctx: ValidationContext) -> list[Problem]:
    """Final renders may only use approved library versions (§18.2)."""
    out: list[Problem] = []
    if not ctx.stable_components:
        return out  # no library loaded; nothing to check against
    accepted_drafts = {d.id for d in plan.drafts}
    refs: list[tuple[str, str | None]] = []
    for shot in plan.shots:
        if shot.transition_in:
            refs.append((shot.transition_in.id, shot.id))
        for fx in shot.fx:
            refs.append((fx.id, shot.id))
        if shot.captions:
            refs.append((shot.captions.id, shot.id))
    refs += [(o.component, None) for o in plan.overlays]

    for ref, shot_id in refs:
        base = ref.split("@")[0]
        if ref in ctx.stable_components or base in accepted_drafts:
            continue
        out.append(Problem(Severity.ERROR, "unapproved_component",
                           f"{ref} is not a stable library version and not an accepted draft",
                           shot_id))
    return out


def _check_sfx(plan: EditPlan, ctx: ValidationContext, fps: float) -> list[Problem]:
    out: list[Problem] = []
    end = plan.timeline_end_frame()
    for fx in plan.sfx:
        if ctx.licensed_sfx and fx.sfx not in ctx.licensed_sfx:
            out.append(Problem(Severity.ERROR, "sfx_without_license",
                               f"{fx.sfx} has no license metadata (§14.1)"))
        anchor = fx.anchor_frame if fx.anchor_frame is not None else fx.end_on_frame
        if anchor is not None and not (0 <= anchor <= end):
            out.append(Problem(Severity.ERROR, "sfx_off_timeline",
                               f"{fx.id} is anchored at frame {anchor}, outside 0..{end}"))
    return out


def _check_audio(plan: EditPlan, shots: list[Shot], ctx: ValidationContext) -> list[Problem]:
    """Speech must duck the music and must not overlap other speech (§18.2)."""
    out: list[Problem] = []
    speech_windows: list[tuple[int, int, str]] = []
    fps = plan.format.fps

    for shot in shots:
        has_speech = bool(
            [s for s in ctx.speech_spans.get(shot.asset, [])
             if _overlaps((shot.src_in, shot.src_out), s) > 0.2]
        )
        if shot.audio.mode in (AudioMode.ORIGINAL, AudioMode.MIXED) and has_speech:
            if shot.audio.duck_music_db is None and plan.music.asset is not None:
                out.append(Problem(Severity.ERROR, "speech_without_ducking",
                                   "speech is audible over the music with no duck_music_db",
                                   shot.id))
            start = shot.timeline_in - shot.audio.j_cut_frames
            end = shot.timeline_in + shot.timeline_duration_frames(fps) + shot.audio.l_cut_frames
            speech_windows.append((start, end, shot.id))

    speech_windows.sort()
    for (_, a_end, a_id), (b_start, _, b_id) in zip(
        speech_windows, speech_windows[1:], strict=False
    ):
        if b_start < a_end:
            out.append(Problem(Severity.ERROR, "overlapping_speech",
                               f"speech overlaps shot {a_id} by {a_end - b_start} frames", b_id))
    return out


def _check_density(plan: EditPlan, fps: float, ctx: ValidationContext) -> list[Problem]:
    """Transition and SFX overuse is the most common amateur tell (§28)."""
    out: list[Problem] = []
    total_s = plan.timeline_end_frame() / fps
    if total_s <= 0:
        return out
    per_10s = total_s / 10.0

    if ctx.max_transitions_per_10s is not None:
        n = sum(1 for s in plan.shots if s.transition_in is not None)
        rate = n / per_10s
        if rate > ctx.max_transitions_per_10s:
            out.append(Problem(Severity.WARNING, "transition_density",
                               f"{rate:.1f} transitions per 10s exceeds the style limit of "
                               f"{ctx.max_transitions_per_10s}"))
    if ctx.max_sfx_per_10s is not None:
        rate = len(plan.sfx) / per_10s
        if rate > ctx.max_sfx_per_10s:
            out.append(Problem(Severity.WARNING, "sfx_density",
                               f"{rate:.1f} SFX per 10s exceeds the style limit of "
                               f"{ctx.max_sfx_per_10s}"))
    return out


def errors(problems: list[Problem]) -> list[Problem]:
    return [p for p in problems if p.severity == Severity.ERROR]


def warnings(problems: list[Problem]) -> list[Problem]:
    return [p for p in problems if p.severity == Severity.WARNING]
