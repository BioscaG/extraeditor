"""ClipLog — structured output of Gemini semantic analysis (§10)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class MomentKind(StrEnum):
    HIGHLIGHT = "highlight"
    REACTION = "reaction"
    QUOTE = "quote"
    SCENIC = "scenic"
    TRANSITION_CANDIDATE = "transition_candidate"
    HERO = "hero"


class Moment(BaseModel):
    t0: float
    t1: float
    label: str
    why: str = ""
    score: float = 0.5
    kind: MomentKind = MomentKind.HIGHLIGHT


class Quote(BaseModel):
    t0: float
    t1: float
    text: str
    speaker_hint: str | None = None
    usable: bool = True


class People(BaseModel):
    count_estimate: int = 0
    us_present: bool = False
    crowd: bool = False


class StageMusic(BaseModel):
    present: bool = False
    description: str = ""


class ClipLog(BaseModel):
    asset_id: str
    model: str = ""
    prompt_version: int = 1
    summary: str = ""
    setting: str = ""
    time_of_day: str = ""
    shot_type: str = ""  # selfie | front | back | pov
    people: People = Field(default_factory=People)
    energy: int = Field(default=3, ge=1, le=5)
    aesthetic: int = Field(default=3, ge=1, le=5)
    start_description: str = ""
    end_description: str = ""
    notable_gestures: list[str] = Field(default_factory=list)
    moments: list[Moment] = Field(default_factory=list)
    quotes: list[Quote] = Field(default_factory=list)
    stage_music: StageMusic = Field(default_factory=StageMusic)
    issues: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
