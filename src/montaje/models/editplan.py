"""EditPlan — the central contract (§17).

Source times are seconds (ms precision) relative to asset start.
Timeline times are integer frames. Musical times may be expressed in beats
and are resolved to frames by the rails.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class AudioMode(StrEnum):
    ORIGINAL = "original"
    MUSIC_ONLY = "music_only"
    MIXED = "mixed"
    MUTED = "muted"


class SnapKind(StrEnum):
    NONE = "none"
    SHOT = "shot"
    BEAT = "beat"
    # The eighth-note grid: every beat plus the midpoint between beats. A festival edit
    # genuinely cuts on off-beats, and without this the finest available position is the
    # beat — which makes any shot shorter than one beat put its following cut off the grid
    # by construction, however well the edit is built.
    HALF_BEAT = "half_beat"
    DOWNBEAT = "downbeat"
    BAR = "bar"
    WORD_START = "word_start"
    WORD_END = "word_end"
    SILENCE = "silence"
    OCCLUSION_REVEAL = "occlusion.reveal"
    OCCLUSION_COVER = "occlusion.cover"


class Duration(BaseModel):
    """A duration in beats or frames; rails resolve beats → frames."""

    beats: float | None = None
    frames: int | None = None

    @model_validator(mode="after")
    def _one_of(self) -> Duration:
        if (self.beats is None) == (self.frames is None):
            raise ValueError("duration needs exactly one of beats|frames")
        return self


class PlanFormat(BaseModel):
    width: int
    height: int
    fps: float
    dynamic_range: str = "sdr"


class Section(BaseModel):
    id: str
    from_frame: int
    to_frame: int
    music_section: str | None = None
    purpose: str | None = None


class Concept(BaseModel):
    title: str
    logline: str | None = None
    signature: str | None = None
    sections: list[Section] = Field(default_factory=list)


class MusicJoin(BaseModel):
    type: str = "crossfade"  # crossfade | butt
    frames: int = 3
    on: str = "downbeat"


class MusicEdit(BaseModel):
    src_in: float
    src_out: float
    timeline_in: int
    join: MusicJoin | None = None  # join with the previous edit


class MusicEnding(BaseModel):
    type: str = "hit_with_tail"  # hit_with_tail | natural_outro | fade


class AudioMoment(BaseModel):
    asset: str
    src_in: float
    src_out: float
    timeline_in: int
    gain_db: float = 0.0


class MusicSource(BaseModel):
    mode: str = "provided"  # provided | library | generate
    provider: str | None = None
    song_id: str | None = None
    variant: int | None = None
    composition_plan: str | None = None


class Music(BaseModel):
    source: MusicSource = Field(default_factory=MusicSource)
    asset: str | None = None
    edits: list[MusicEdit] = Field(default_factory=list)
    ending: MusicEnding = Field(default_factory=MusicEnding)
    audio_moments: list[AudioMoment] = Field(default_factory=list)


class SpeedRamp(BaseModel):
    """Piecewise speed. `from`/`to` are source-relative positions in [0,1] of the shot range."""

    from_: float = Field(alias="from", ge=0.0, le=1.0)
    to: float = Field(alias="to", ge=0.0, le=1.0)
    rate: float = Field(gt=0.0)

    model_config = {"populate_by_name": True}


class Reframe(BaseModel):
    mode: str = "center"  # center | auto_subject | manual
    ease: str = "glide"
    rect: list[float] | None = None  # manual: [x, y, w, h] normalized


class ShotColor(BaseModel):
    normalize: str = "auto"  # auto | off | manual
    match_to: str | None = None  # shot id used as anchor
    grade: str = "style"  # style | off
    params: dict = Field(default_factory=dict)


class ComponentRef(BaseModel):
    id: str  # "transition.zoom_punch@1.0" — versioned library reference
    preset: str | None = None
    duration: Duration | None = None
    props: dict = Field(default_factory=dict)
    on: str | None = None  # e.g. "downbeats" for repeating fx


class ShotAudio(BaseModel):
    mode: AudioMode = AudioMode.MUSIC_ONLY
    duck_music_db: float | None = None
    j_cut_frames: int = 0
    l_cut_frames: int = 0
    cleanup: str | None = None  # "dialogue"
    gain_db: float = 0.0


class SnapSpec(BaseModel):
    """Where a shot's edges are pulled to.

    `in_`/`out` are **source-domain**: which frame of the original the shot starts and ends
    on — a shot boundary, a word edge, the end of an occlusion. `timeline` is separately
    **timeline-domain**: where in the finished edit the shot appears.

    They have to be separate because a shot commonly needs both, and the two never conflict:
    moving when a shot appears does not move which frames of it are used. Collapsing them
    into one field meant a captioned shot, which must start on a word, could not also land
    on the beat — and with a third of the shots in a real edit captioned, a third of the cuts
    were simply exempt from the musical grid, which measured as 45% of cuts off it.
    """

    in_: SnapKind = Field(default=SnapKind.NONE, alias="in")
    out: SnapKind = SnapKind.NONE
    timeline: SnapKind = SnapKind.NONE

    model_config = {"populate_by_name": True}


class Captions(BaseModel):
    id: str  # component ref
    words: str = "auto"  # auto | explicit srt path


class Shot(BaseModel):
    id: str
    asset: str
    src_in: float
    src_out: float
    timeline_in: int
    speed: list[SpeedRamp] = Field(default_factory=list)
    reframe: Reframe = Field(default_factory=Reframe)
    color: ShotColor = Field(default_factory=ShotColor)
    transition_in: ComponentRef | None = None
    fx: list[ComponentRef] = Field(default_factory=list)
    audio: ShotAudio = Field(default_factory=ShotAudio)
    snap: SnapSpec = Field(default_factory=SnapSpec)
    captions: Captions | None = None
    section: str | None = None
    intent: str = ""

    @model_validator(mode="after")
    def _range_ok(self) -> Shot:
        if self.src_out <= self.src_in:
            raise ValueError(f"shot {self.id}: src_out <= src_in")
        return self

    @property
    def src_duration_s(self) -> float:
        return self.src_out - self.src_in

    def timeline_duration_frames(self, fps: float) -> int:
        """Timeline length in frames after speed ramps."""
        dur = self.src_duration_s
        if not self.speed:
            return round(dur * fps)
        total = 0.0
        for ramp in self.speed:
            total += dur * (ramp.to - ramp.from_) / ramp.rate
        return round(total * fps)


class Overlay(BaseModel):
    id: str
    component: str  # versioned library reference
    from_frame: int
    to_frame: int
    props: dict = Field(default_factory=dict)


class Sfx(BaseModel):
    id: str
    sfx: str  # library sfx id
    anchor_frame: int | None = None  # peak lands here
    end_on_frame: int | None = None  # e.g. riser ends on the downbeat
    gain_db: float = 0.0
    source: str = "agent"  # auto_spotting | agent | user

    @model_validator(mode="after")
    def _anchored(self) -> Sfx:
        if self.anchor_frame is None and self.end_on_frame is None:
            raise ValueError(f"sfx {self.id}: needs anchor_frame or end_on_frame")
        return self


class Draft(BaseModel):
    id: str
    path: str


class EditPlan(BaseModel):
    version: int = 1
    project: str
    format: PlanFormat
    style: str | None = None
    concept: Concept = Field(default_factory=lambda: Concept(title="untitled"))
    music: Music = Field(default_factory=Music)
    shots: list[Shot] = Field(default_factory=list)
    overlays: list[Overlay] = Field(default_factory=list)
    sfx: list[Sfx] = Field(default_factory=list)
    drafts: list[Draft] = Field(default_factory=list)
    notes: str = ""

    def shot(self, shot_id: str) -> Shot:
        for s in self.shots:
            if s.id == shot_id:
                return s
        raise KeyError(shot_id)

    def sorted_shots(self) -> list[Shot]:
        return sorted(self.shots, key=lambda s: s.timeline_in)

    def timeline_end_frame(self) -> int:
        end = 0
        for s in self.shots:
            end = max(end, s.timeline_in + s.timeline_duration_frames(self.format.fps))
        for o in self.overlays:
            end = max(end, o.to_frame)
        return end
