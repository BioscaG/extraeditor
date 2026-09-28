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

from dataclasses import dataclass, field

from montaje.models.asset import Asset, AssetKind
from montaje.models.cliplog import ClipLog
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
    aesthetic: int = 3
    us_present: bool = False
    label: str = ""

    @property
    def duration_s(self) -> float:
        return self.t1 - self.t0

    def score(self, want_energy: float) -> float:
        """Rank for a target energy, 0–1. Higher is a better fit."""
        # Aesthetic and quality are absolute goods; energy is a *match*, not a maximum,
        # because a calm intro needs calm footage.
        energy_fit = 1.0 - abs(self.energy / 5.0 - want_energy)
        people = 0.15 if self.us_present else 0.0
        return (
            0.30 * self.quality
            + 0.25 * (self.aesthetic / 5.0)
            + 0.30 * energy_fit
            + people
        )


@dataclass
class BuildInputs:
    assets: dict[str, Asset]
    events: dict[str, list[Event]] = field(default_factory=dict)
    clip_logs: dict[str, ClipLog] = field(default_factory=dict)
    structure: MusicStructure | None = None
    music_fit: MusicFit | None = None
    style: Style | None = None
    music_asset_id: str | None = None

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
        if not usable:
            # No quality analysis: offer the whole clip rather than nothing, so an
            # un-analyzed project still produces an edit.
            usable = [Event(asset_id=asset_id, analyzer="quality@0", type="usable",
                            t0=0.0, t1=asset.duration_s)]
        metrics = [e for e in events if e.analyzer.startswith("quality") and e.type == "metrics"]
        motion = [e for e in events if e.analyzer.startswith("motion")]
        speech = [e for e in events if e.analyzer.startswith("vad") and e.type == "speech"]
        words = [e for e in events if e.analyzer.startswith("asr") and e.type == "word"]

        for span in usable:
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
                aesthetic=log.aesthetic if log else 3,
                us_present=log.people.us_present if log else False,
                label=log.summary[:80] if log else "",
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


def _shot_length_s(role: str, style: Style | None, bpm: float, index: int = 0) -> float:
    """Shot length for the `index`-th shot of a section, varied around the target."""
    beats = 1.0
    if style and style.pacing.avg_shot_beats:
        beats = style.pacing.avg_shot_beats.get(role, 1.0)
    target = beats * 60.0 / bpm
    varied = target * LENGTH_VARIATION[index % len(LENGTH_VARIATION)]
    return max(ABSOLUTE_MIN_SHOT_S, varied)


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
                snap=SnapSpec(**{"in": SnapKind.BEAT, "out": SnapKind.NONE}),
                audio=_audio_for(candidate, has_music=bool(plan.music.edits)),
                intent=_intent_for(candidate, section),
            )
            # Captions only where there are actually transcribed words in the chosen
            # range: a caption component with nothing to show renders empty, so claiming
            # it in the plan is a lie the validator then has to catch.
            if inputs.has_words_in(candidate.asset_id, src_in, src_out) \
                    and style and style.text.captions:
                shot.captions = Captions(id=_versioned(style.text.captions), words="auto")
                # Speech needs room to breathe: hold the shot to the spoken range.
                shot.snap = SnapSpec(**{"in": SnapKind.WORD_START, "out": SnapKind.WORD_END})

            if within_section == 1 and section is not sections[0][0]:
                section_openers.append(shot.id)
            plan.shots.append(shot)
            cursor += shot.timeline_duration_frames(fps)

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
    """Best unused range of at least `length_s`, preferring a different asset.

    Avoiding the previous asset implements "never place near-duplicates back to
    back" (§16.4) at the cheapest possible level: consecutive shots from the same
    clip are the most likely near-duplicates there are.
    """
    best: tuple[float, Candidate, float, float] | None = None
    for candidate in candidates:
        if candidate.duration_s < max(ABSOLUTE_MIN_SHOT_S, length_s * 0.6):
            continue
        start = _free_start(candidate, used.get(candidate.asset_id, []), length_s)
        if start is None:
            continue
        end = min(candidate.t1, start + length_s)
        if end - start < max(ABSOLUTE_MIN_SHOT_S, length_s * 0.6):
            continue
        score = candidate.score(want_energy)
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
        # A section change should land on a bar line, not just any beat.
        shot.snap = SnapSpec(**{"in": SnapKind.DOWNBEAT, "out": shot.snap.out})


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
