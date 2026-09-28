"""Deterministic baseline plan builder.

The director agent is where creativity lives (§2.3), but a deterministic builder is
needed for three reasons: the end-to-end test needs a plan without an LLM in the
loop, the agent needs a starting plan to refine rather than a blank timeline, and a
machine with no API key must still produce a watchable edit.

It is deliberately *rule-following*, not creative: map the music's sections onto the
footage, pick the best-scoring usable ranges, pace each section to the style's
target, and let the rails snap everything. It makes no attempt at story.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

from montaje.models.asset import Asset, AssetKind
from montaje.models.cliplog import ClipLog
from montaje.models.conventions import Convention
from montaje.models.editplan import (
    AudioMode,
    Captions,
    ComponentRef,
    Concept,
    Duration,
    EditPlan,
    Music,
    MusicSource,
    PlanFormat,
    Reframe,
    Section,
    Shot,
    ShotAudio,
    SnapKind,
    SnapSpec,
)
from montaje.models.events import Event
from montaje.models.style import Style
from montaje.music.edit import MusicFit
from montaje.music.structure import MusicStructure

# How much a semantically logged moment outranks a merely usable range.
MOMENT_WEIGHT = 1.2
# Extra weight per moment kind. A hero shot is what an edit is built around; a
# transition candidate is only useful at a boundary, so it gets no general bonus.
KIND_BONUS = {
    "hero": 0.35,
    "highlight": 0.2,
    "reaction": 0.18,
    "quote": 0.12,
    "scenic": 0.05,
    "transition_candidate": 0.0,
}

# Camera-motion magnitude treated as full energy. Above roughly this the shot is already
# as busy as footage gets; the analyzer reports magnitude in grid cells per frame.
MOTION_ENERGY_SCALE = 6.0

# The shortest candidate range worth considering at all.
MIN_SHOT_S = 0.4
# Floor for a generated shot. Must be >= the rails' `min_shot_s`, or the builder
# spends its budget on shots the rails will delete — which silently shortened the
# whole edit by 9 seconds before this was tied together.
ABSOLUTE_MIN_SHOT_S = 0.35


@dataclass
class Candidate:
    """A usable range of one asset, with everything needed to rank it."""

    asset_id: str
    t0: float
    t1: float
    quality: float = 0.5
    motion: float = 0.0
    direction: str = "static"
    has_speech: bool = False
    has_words: bool = False
    energy: int = 3
    # Whether `energy` came from a clip log. Without one it is a default, not a measurement.
    energy_from_log: bool = False
    aesthetic: int = 3
    us_present: bool = False
    label: str = ""
    # From a semantic clip log, when one exists. `moment_score` is the model's own
    # recommendation strength; `moment_kind` says what sort of moment it identified.
    moment_score: float = 0.0
    moment_kind: str = ""

    @property
    def duration_s(self) -> float:
        return self.t1 - self.t0

    @property
    def is_moment(self) -> bool:
        return self.moment_score > 0.0

    @property
    def energy_value(self) -> float:
        """Energy as 0–1, from the clip log if there is one, otherwise from motion.

        Falling back to measured motion matters: without semantic analysis every clip log
        is absent and `energy` sits at its default 3, so energy matching does nothing and
        the picture's business bears no relation to the music's. The rhythm report measures
        *motion* energy against music energy, so the planner should optimize the same
        quantity the report scores it on.
        """
        if self.energy_from_log:
            return self.energy / 5.0
        return min(1.0, self.motion / MOTION_ENERGY_SCALE)

    def score(self, want_energy: float) -> float:
        """Rank for a target energy, 0–1. Higher is a better fit."""
        # Aesthetic and quality are absolute goods; energy is a *match*, not a maximum,
        # because a calm intro needs calm footage.
        energy_fit = 1.0 - abs(self.energy_value - want_energy)
        people = 0.15 if self.us_present else 0.0
        base = (
            0.30 * self.quality
            + 0.25 * (self.aesthetic / 5.0)
            + 0.30 * energy_fit
            + people
        )
        # A logged moment is the only evidence available that a range is *interesting*
        # rather than merely technically sound, so it dominates the ranking. Without
        # semantic analysis every usable range scores the same on this axis and
        # selection is effectively arbitrary within the sharp, steady material.
        return base + MOMENT_WEIGHT * self.moment_score + KIND_BONUS.get(self.moment_kind, 0.0)


@dataclass
class BuildInputs:
    assets: dict[str, Asset]
    events: dict[str, list[Event]] = field(default_factory=dict)
    clip_logs: dict[str, ClipLog] = field(default_factory=dict)
    structure: MusicStructure | None = None
    music_fit: MusicFit | None = None
    style: Style | None = None
    music_asset_id: str | None = None
    conventions: list[Convention] = field(default_factory=list)

    def usable_range(self, asset_id: str, duration_s: float) -> tuple[float, float]:
        """The part of an asset a confirmed convention leaves usable (§12).

        Applied when *choosing* ranges, not only when trimming afterwards: a shot picked
        inside a motif would be trimmed away later, wasting the budget it was given.
        """
        from montaje.index.conventions import usable_after_conventions

        if not self.conventions:
            return 0.0, duration_s
        return usable_after_conventions(
            asset_id, duration_s, self.conventions, self.events.get(asset_id, [])
        )

    def word_spans(self, asset_id: str) -> list[tuple[float, float]]:
        return [
            (e.t0, e.t1) for e in self.events.get(asset_id, [])
            if e.analyzer.startswith("asr") and e.type == "word"
        ]

    def has_words_in(self, asset_id: str, t0: float, t1: float) -> bool:
        """Whether any transcribed word falls in a range.

        Checked against the *chosen* shot range, not the candidate span it came from:
        a clip with one spoken line yields usable spans covering the whole clip, so
        gating on the span captioned every sub-range taken from it — 43 of 51 shots
        claimed captions when only 18 had anything to show.
        """
        return any(b > t0 and a < t1 for a, b in self.word_spans(asset_id))


def collect_candidates(inputs: BuildInputs) -> list[Candidate]:
    """Every usable range across all assets, annotated from the analysis."""
    out: list[Candidate] = []
    for asset_id, asset in inputs.assets.items():
        if asset.kind == AssetKind.PHOTO:
            continue
        events = inputs.events.get(asset_id, [])
        log = inputs.clip_logs.get(asset_id)
        usable = [e for e in events if e.analyzer.startswith("quality") and e.type == "usable"]
        metrics = [e for e in events if e.analyzer.startswith("quality") and e.type == "metrics"]
        if not usable:
            if metrics:
                # The analyzer ran and found nothing worth using — too dark, too blurred,
                # too shaky. Offering the clip anyway contradicts the measurement: of 130
                # real clips, 58 had no usable span at all, and one of them put half a
                # second of black into a finished render, where the auto-checks caught it.
                continue
            # Quality analysis never ran: offer the whole clip rather than nothing, so an
            # un-analyzed project still produces an edit.
            usable = [Event(asset_id=asset_id, analyzer="quality@0", type="usable",
                            t0=0.0, t1=asset.duration_s)]
        motion = [e for e in events if e.analyzer.startswith("motion")]
        speech = [e for e in events if e.analyzer.startswith("vad") and e.type == "speech"]
        words = [e for e in events if e.analyzer.startswith("asr") and e.type == "word"]

        # Logged moments become candidates in their own right, alongside the usable
        # spans, so the planner can pick "the thing that happens" rather than "a stretch
        # of sharp footage that happens to contain it".
        spans: list[tuple[float, float, float, str]] = [
            (e.t0, e.t1, 0.0, "") for e in usable
        ]
        if log is not None:
            for moment in log.moments:
                spans.append((moment.t0, moment.t1, moment.score, moment.kind.value))

        allowed_lo, allowed_hi = inputs.usable_range(asset_id, asset.duration_s)
        for raw_t0, raw_t1, moment_score, moment_kind in spans:
            # Clip every candidate to what the confirmed conventions leave usable.
            span_t0 = max(raw_t0, allowed_lo)
            span_t1 = min(raw_t1, allowed_hi)
            span = Event(asset_id=asset_id, analyzer="span@0", type="span",
                         t0=span_t0, t1=max(span_t0, span_t1))
            if span.t1 - span.t0 < MIN_SHOT_S:
                continue
            inner = [m for m in metrics if m.t0 >= span.t0 - 1 and m.t1 <= span.t1 + 1]
            detail = [m.data.get("detail", 0.0) for m in inner] or [8.0]
            shake = [m.data.get("shake", 0.0) for m in inner] or [0.0]
            quality = _quality_score(sum(detail) / len(detail), sum(shake) / len(shake))

            inner_motion = [m for m in motion if m.t1 > span.t0 and m.t0 < span.t1]
            magnitudes = [m.data.get("camera_magnitude", 0.0) for m in inner_motion] or [0.0]
            directions = [m.data.get("direction", "static") for m in inner_motion]

            out.append(Candidate(
                asset_id=asset_id, t0=span.t0, t1=span.t1,
                quality=quality,
                motion=sum(magnitudes) / len(magnitudes),
                direction=_dominant(directions),
                has_speech=any(s.t1 > span.t0 and s.t0 < span.t1 for s in speech),
                has_words=any(w.t1 > span.t0 and w.t0 < span.t1 for w in words),
                energy=log.energy if log else 3,
                energy_from_log=log is not None,
                aesthetic=log.aesthetic if log else 3,
                us_present=log.people.us_present if log else False,
                label=log.summary[:80] if log else "",
                moment_score=moment_score,
                moment_kind=moment_kind,
            ))
    return out


def _quality_score(detail: float, shake: float) -> float:
    """Map raw detail and shake metrics into 0–1."""
    sharp = min(1.0, detail / 14.0)
    steady = max(0.0, 1.0 - shake / 25.0)
    return round(0.65 * sharp + 0.35 * steady, 4)


def _dominant(values: list[str]) -> str:
    if not values:
        return "static"
    moving = [v for v in values if v != "static"]
    pool = moving or values
    return max(set(pool), key=pool.count)


def _sections_from_structure(
    structure: MusicStructure | None,
    fit: MusicFit | None,
    total_frames: int,
    fps: float,
) -> list[tuple[Section, float]]:
    """Plan sections with a target energy each, mapped onto the *edited* timeline.

    The music edit removes and repeats phrases, so a section that starts at 48 s in
    the source starts somewhere else on the timeline (§14.2). Sections are therefore
    projected through the edit rather than taken from the source times — otherwise the
    picture's structure would not match what the viewer hears.

    Without a music structure the timeline is one section: there is nothing to map
    story onto and inventing boundaries would be arbitrary.
    """
    if structure is None or not structure.sections:
        return [(Section(id="all", from_frame=0, to_frame=total_frames,
                         music_section="build", purpose="single section"), 0.6)]

    if fit is not None and fit.edits:
        from montaje.music.edit import timeline_sections

        projected = timeline_sections(structure, fit, fps)
        out: list[tuple[Section, float]] = []
        for s in projected:
            to_frame = min(total_frames, s.to_frame)
            if to_frame <= s.from_frame:
                continue
            bars = round((to_frame - s.from_frame) / fps / (structure.grid.beat_period_s
                                                            * structure.grid.beats_per_bar))
            out.append((
                Section(id=s.id, from_frame=s.from_frame, to_frame=to_frame,
                        music_section=s.role, purpose=f"{s.role} ({bars} bars)"),
                max(0.15, min(1.0, s.energy)),
            ))
        if out:
            return out

    out = []
    for s in structure.sections:
        from_frame = round(s.t0 * fps)
        to_frame = min(total_frames, round(s.t1 * fps))
        if to_frame <= from_frame:
            continue
        out.append((
            Section(id=s.id, from_frame=from_frame, to_frame=to_frame,
                    music_section=s.role, purpose=f"{s.role} ({s.bars} bars)"),
            max(0.15, min(1.0, s.energy)),
        ))
    return out


# Multipliers cycled through so shot lengths vary around the section's target
# instead of being constant. §16.4 requires varying shot lengths and breaking the
# beat grid on purpose: a constant cut rate is the difference between an edit that
# feels edited and one that feels generated. The mean is 1.0, so the section's
# average still lands on the style's target.
LENGTH_VARIATION = (1.0, 1.75, 0.75, 1.5, 0.5, 1.25, 1.0, 0.75, 2.0, 0.5)

# The finest unit a shot length is allowed to be. Applying the multipliers to a duration in
# seconds produced lengths like 1.5 or 0.75 beats, and since the snap grid's finest position
# is the half beat, the cut after such a shot lands between grid positions *by construction*
# — the first real run measured 51% of cuts on the grid against a style target of 75%, and
# no amount of snapping could have fixed it, because the lengths themselves were unmusical.
HALF_BEATS_PER_BEAT = 2

# What a shot is worth after its asset has already been used once. Multiplicative, so the
# third shot from a clip is worth 0.29 of the first and the sixth 0.05: a genuinely
# outstanding clip can still carry several shots, while an ordinary one gives way to a clip
# not yet seen. Not a hard cap, because sometimes one clip really is the best thing there is.
REPEAT_DECAY = 0.54


@functools.lru_cache(maxsize=64)
def _variation_half_beats(target_half_beats: int, min_half_beats: int = 1) -> tuple[int, ...]:
    """`LENGTH_VARIATION` expressed in whole half-beats, with the mean restored exactly.

    Rounding to a musical unit biases the mean upward — every value is pulled to a whole
    half-beat and the shortest ones hit the floor, which can only push them up. Left
    uncorrected, a drop targeting 1.0 beats per shot measures 1.3, and the rhythm report
    rightly complains about a pacing target the builder never actually aimed at. The surplus
    is taken back off the longest shots, so the mean is exact and the variation — which is
    the point of the whole sequence — survives.

    `min_half_beats` carries the minimum shot length, itself rounded up to a whole half-beat:
    clamping afterwards in seconds produced 0.35s shots on a 0.25s grid, which is exactly the
    unmusical length this quantization exists to avoid.
    """
    lengths = [
        max(min_half_beats, math.floor(target_half_beats * multiplier + 0.5))
        for multiplier in LENGTH_VARIATION
    ]
    wanted = target_half_beats * len(lengths)
    # Longest first, and stably, so the correction is deterministic across runs.
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    # Where the floor is longer than the target the mean simply cannot be met — every shot
    # is already as short as it may be. Stop rather than spin; the rhythm report will say so.
    for position in range(len(lengths) * 64):
        total = sum(lengths)
        if total == wanted:
            break
        index = order[position % len(order)]
        if total > wanted:
            if lengths[index] > min_half_beats:
                lengths[index] -= 1
        else:
            lengths[index] += 1
    return tuple(lengths)


def _shot_length_s(role: str, style: Style | None, bpm: float, index: int = 0) -> float:
    """Shot length for the `index`-th shot of a section, varied around the target.

    Quantized to half beats so the cut that follows can land on the grid at all.
    """
    beats = 1.0
    if style and style.pacing.avg_shot_beats:
        beats = style.pacing.avg_shot_beats.get(role, 1.0)
    half_beat_s = 30.0 / bpm
    target_half_beats = max(1, math.floor(beats * HALF_BEATS_PER_BEAT + 0.5))
    min_half_beats = max(1, math.ceil(ABSOLUTE_MIN_SHOT_S / half_beat_s - 1e-9))
    pattern = _variation_half_beats(target_half_beats, min_half_beats)
    return pattern[index % len(pattern)] * half_beat_s


def build_plan(
    project: str,
    fmt: PlanFormat,
    inputs: BuildInputs,
    *,
    title: str = "Untitled",
    target_s: float | None = None,
) -> EditPlan:
    """Build a complete, renderable plan. Rails and validation run separately."""
    fps = fmt.fps
    bpm = inputs.structure.grid.bpm if inputs.structure else 120.0
    style = inputs.style

    duration_s = target_s
    if duration_s is None:
        duration_s = inputs.music_fit.total_s if inputs.music_fit else 60.0
    total_frames = max(1, round(duration_s * fps))

    sections = _sections_from_structure(inputs.structure, inputs.music_fit, total_frames, fps)
    candidates = collect_candidates(inputs)

    plan = EditPlan(
        version=1,
        project=project,
        format=fmt,
        style=style.name if style else None,
        concept=Concept(
            title=title,
            logline="Baseline edit: music structure mapped onto the best usable footage.",
            sections=[s for s, _ in sections],
        ),
        music=Music(
            source=MusicSource(mode="provided"),
            asset=inputs.music_asset_id,
            edits=list(inputs.music_fit.edits) if inputs.music_fit else [],
            ending=inputs.music_fit.ending if inputs.music_fit else Music().ending,
        ),
    )

    used: dict[str, list[tuple[float, float]]] = {}
    shot_index = 0
    cursor = 0
    # The same position as `cursor`, kept in exact seconds rather than whole frames.
    #
    # Shot lengths are whole half-beats, but a half-beat is not a whole number of frames —
    # at 120 BPM and 30fps it is 7.5 — so a cursor that advances by rounded frame counts
    # drifts away from the grid one shot at a time. Measured over a 90s edit, 46% of cuts
    # ended up more than 62ms off the grid, and the rails could only pull back the ones
    # whose previous shot still had handle material to give. Accumulating in seconds and
    # rounding only at the moment a frame number is needed keeps every cut within half a
    # frame of its grid position, which is as close as a frame-based timeline can get.
    cursor_s = 0.0
    # Shot ids that open a section. Recorded here rather than recovered afterwards by
    # comparing timeline positions: the cursor advances by whole shots and rarely lands
    # exactly on a section boundary, so a frame comparison silently matched nothing and
    # the edit ended up with no designed transitions at all.
    section_openers: list[str] = []

    for section, want_energy in sections:
        role = section.music_section or "build"
        section_end = section.to_frame
        last_asset: str | None = None
        within_section = 0

        while cursor < section_end:
            remaining_s = (section_end - cursor) / fps
            if remaining_s < ABSOLUTE_MIN_SHOT_S:
                break
            length_s = min(_shot_length_s(role, style, bpm, within_section), remaining_s)
            within_section += 1
            pick = _pick(candidates, used, want_energy, length_s, avoid_asset=last_asset)
            if pick is None:
                break
            candidate, src_in, src_out = pick

            # Stretch the source range to land the *next* cut on its grid frame. The
            # picked range is already about `length_s` long; this trims or extends it by
            # at most a frame, within the material the candidate span offers.
            src_in = round(src_in, 3)
            grid_end_frame = round((cursor_s + length_s) * fps)
            grid_frames = max(1, grid_end_frame - cursor)
            aligned_out = round(src_in + grid_frames / fps, 3)
            # Where the material runs a frame short the picked range stands, and the cursors
            # are resynchronised below: losing the grid on one shot is honest, a gap is not.
            if aligned_out <= candidate.t1:
                src_out = aligned_out

            used.setdefault(candidate.asset_id, []).append((src_in, src_out))
            last_asset = candidate.asset_id

            shot_index += 1
            shot = Shot(
                id=f"s{shot_index:03d}",
                asset=candidate.asset_id,
                src_in=round(src_in, 3),
                src_out=round(src_out, 3),
                timeline_in=cursor,
                section=section.id,
                # Half beats, not beats: shot lengths are whole half-beats, so the beat is
                # too coarse a rail to hold them — a shot of three half-beats would be
                # dragged back onto the beat and lose the off-beat cut it was built for.
                snap=SnapSpec(timeline=SnapKind.HALF_BEAT),
                audio=_audio_for(candidate, has_music=bool(plan.music.edits)),
                # The style's reframe policy, not the model default: leaving it at
                # `center` means 16:9 footage in a 9:16 frame is cropped down the middle
                # and people end up out of shot, which is the most visible failure a
                # vertical edit can have.
                reframe=_reframe_for(style),
                intent=_intent_for(candidate, section),
            )
            # Captions only where there are actually transcribed words in the chosen
            # range: a caption component with nothing to show renders empty, so claiming
            # it in the plan is a lie the validator then has to catch.
            if inputs.has_words_in(candidate.asset_id, src_in, src_out) \
                    and style and style.text.captions:
                shot.captions = Captions(id=_versioned(style.text.captions), words="auto")
                # Speech needs room to breathe: hold the *source* range to the spoken words,
                # while the shot still lands on the grid. Replacing the timeline snap here
                # exempted every captioned shot from the music — a third of the edit.
                shot.snap = SnapSpec(
                    **{"in": SnapKind.WORD_START, "out": SnapKind.WORD_END},
                    timeline=SnapKind.HALF_BEAT,
                )

            if within_section == 1 and section is not sections[0][0]:
                section_openers.append(shot.id)
            plan.shots.append(shot)
            frames = shot.timeline_duration_frames(fps)
            cursor += frames
            if frames == grid_frames:
                # The invariant that removes the drift: `cursor_s` accumulates the exact
                # half-beat lengths while `cursor` accumulates their frame differences, so
                # `cursor == round(cursor_s * fps)` holds however long the edit runs.
                cursor_s += length_s
            else:
                # Clamped, or the source range came back a frame different from what was
                # asked for. Resynchronise rather than let the two cursors disagree.
                cursor_s = cursor / fps

    _fix_speech_cuts(plan)
    _add_transitions(plan, section_openers, style)
    _add_title(plan, title, style, fps)
    return plan


def fix_speech_cuts(plan: EditPlan) -> None:
    """Remove J/L cuts that would put two voices on top of each other.

    A J-cut lets the incoming audio lead its picture, which works over music or
    ambience but not over another person talking — §18.2 rejects overlapping speech,
    and this is the most common way of causing it.

    Consecutive *speaking* shots are compared, not consecutive shots: a short
    music-only shot between two lines does not stop a long J-cut reaching back into
    the earlier line. Must run after the rails, since snapping changes which shots
    end up adjacent and how long they are.
    """
    fps = plan.format.fps
    speaking = [
        s for s in plan.sorted_shots()
        if s.audio.mode in (AudioMode.ORIGINAL, AudioMode.MIXED)
    ]
    for previous, current in zip(speaking, speaking[1:], strict=False):
        previous_end = (
            previous.timeline_in
            + previous.timeline_duration_frames(fps)
            + previous.audio.l_cut_frames
        )
        if current.timeline_in - current.audio.j_cut_frames < previous_end:
            # Trim both sides of the offending boundary rather than only the J-cut:
            # an L-cut on the earlier shot causes the same overlap.
            available = current.timeline_in - (
                previous.timeline_in + previous.timeline_duration_frames(fps)
            )
            previous.audio.l_cut_frames = 0
            current.audio.j_cut_frames = max(0, min(current.audio.j_cut_frames, available))


def drop_empty_captions(plan: EditPlan, word_spans: dict[str, list[tuple[float, float]]]) -> int:
    """Remove captions from shots whose range no longer contains any word.

    Must run after the rails: snapping and gap-closing move a shot's in and out points,
    which can shift a short shot off the words it was captioned for. A caption component
    with nothing to show renders empty, so the plan should stop claiming it. Returns how
    many were removed.
    """
    removed = 0
    for shot in plan.shots:
        if shot.captions is None:
            continue
        spans = word_spans.get(shot.asset)
        if spans is None:
            continue  # no ASR for this asset; leave the plan's intent alone
        if not any(b > shot.src_in and a < shot.src_out for a, b in spans):
            shot.captions = None
            removed += 1
    return removed


# Kept as a private alias so the builder's own call site stays readable.
_fix_speech_cuts = fix_speech_cuts


def _pick(
    candidates: list[Candidate],
    used: dict[str, list[tuple[float, float]]],
    want_energy: float,
    length_s: float,
    avoid_asset: str | None = None,
) -> tuple[Candidate, float, float] | None:
    """Best unused range of at least `length_s`, preferring assets not already drawn on.

    Avoiding the previous asset implements "never place near-duplicates back to
    back" (§16.4) at the cheapest possible level: consecutive shots from the same
    clip are the most likely near-duplicates there are.

    The repeat decay is the other half of that, and the more important half. Scoring each
    range on its own merit means the few clips that score highest win every comparison, and
    a real shoot is not uniform: over 130 clips this edit drew all 108 of its shots from 20
    of them — one clip fifteen times — while 52 clips with perfectly usable footage never
    appeared at all. A viewer reads that as the same scene coming round again, and it is the
    difference between an aftermovie of a festival and an aftermovie of four clips.
    """
    best: tuple[float, Candidate, float, float] | None = None
    for candidate in candidates:
        if candidate.duration_s < max(ABSOLUTE_MIN_SHOT_S, length_s * 0.6):
            continue
        ranges = used.get(candidate.asset_id, [])
        start = _free_start(candidate, ranges, length_s)
        if start is None:
            continue
        end = min(candidate.t1, start + length_s)
        if end - start < max(ABSOLUTE_MIN_SHOT_S, length_s * 0.6):
            continue
        # Decay by how often this *asset* has been drawn on, not this range: two different
        # seconds of the same 40-second clip of the same stage are still the same shot.
        score = candidate.score(want_energy) * REPEAT_DECAY ** len(ranges)
        if avoid_asset is not None and candidate.asset_id == avoid_asset:
            score -= 0.25
        if best is None or score > best[0]:
            best = (score, candidate, start, end)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _free_start(
    candidate: Candidate, used: list[tuple[float, float]], length_s: float
) -> float | None:
    """First position in the candidate's span not already used, with room for `length_s`."""
    cursor = candidate.t0
    for a, b in sorted(used):
        if b <= cursor or a >= candidate.t1:
            continue
        if a - cursor >= length_s:
            return cursor
        cursor = max(cursor, b)
    if candidate.t1 - cursor >= max(ABSOLUTE_MIN_SHOT_S, length_s * 0.6):
        return cursor
    return None


def _reframe_for(style: Style | None) -> Reframe:
    """The style's reframe policy, defaulting to a centre crop with no style."""
    if style is None:
        return Reframe()
    return Reframe(mode=style.reframe.policy, ease=style.reframe.ease)


def _audio_for(candidate: Candidate, has_music: bool) -> ShotAudio:
    """Speech is heard and ducks the music; everything else rides the bed."""
    if candidate.has_speech:
        return ShotAudio(
            mode=AudioMode.ORIGINAL,
            duck_music_db=-14.0 if has_music else None,
            cleanup="dialogue",
            j_cut_frames=4,
        )
    return ShotAudio(mode=AudioMode.MUSIC_ONLY)


def _intent_for(candidate: Candidate, section: Section) -> str:
    """Every shot carries an intent (§16.4) — including the builder's."""
    role = section.music_section or "section"
    if candidate.has_speech:
        return f"Speech in the {role}: {candidate.label or 'spoken line'}"
    if candidate.motion > 3:
        return f"Moving shot ({candidate.direction}) for the {role}"
    return f"{candidate.label or 'Held shot'} in the {role}"


def _versioned(component_id: str) -> str:
    """Resolve a bare component id from a style into a versioned plan reference."""
    from montaje.library.registry import find

    meta = find(component_id)
    return meta.ref if meta else component_id


def _add_transitions(plan: EditPlan, section_openers: list[str], style: Style | None) -> None:
    """Designed transitions on the shots that open a section, and nowhere else.

    §16.4: hard cuts are the default and designed transitions are punctuation. Placing
    them only at section changes keeps the density inside the style's limit by
    construction rather than by pruning afterwards.
    """
    if style is None or not style.transitions.palette or not section_openers:
        return
    from montaje.library.registry import find

    palette = style.transitions.palette
    for index, shot_id in enumerate(section_openers):
        try:
            shot = plan.shot(shot_id)
        except KeyError:
            continue
        meta = find(palette[index % len(palette)])
        if meta is None:
            continue
        shot.transition_in = ComponentRef(
            id=meta.ref,
            preset="subtle",
            duration=Duration(beats=meta.duration.default_beats or 0.5),
        )
        # A section change should land on a bar line, not just any beat. Only the timeline
        # position changes: whatever the shot's source edges were pulled to — a word, a shot
        # boundary — still holds, and overwriting `in_` here silently discarded it.
        shot.snap = shot.snap.model_copy(update={"timeline": SnapKind.DOWNBEAT})


def _add_title(plan: EditPlan, title: str, style: Style | None, fps: float) -> None:
    """One opening title, held over the first shots."""
    if style is None or not style.text.titles or not plan.shots:
        return
    from montaje.library.registry import find
    from montaje.models.editplan import Overlay

    meta = find(style.text.titles)
    if meta is None:
        return
    duration = min(meta.duration.max_frames, round(fps * 2.5))
    plan.overlays.append(Overlay(
        id="o001",
        component=meta.ref,
        from_frame=0,
        to_frame=duration,
        props={"lines": _title_lines(title), "accentLine": 0},
    ))


def _title_lines(title: str) -> list[str]:
    """Break a title into at most three display lines."""
    words = title.split()
    if len(words) <= 2:
        return [title.upper()]
    mid = len(words) // 2
    return [" ".join(words[:mid]).upper(), " ".join(words[mid:]).upper()]
