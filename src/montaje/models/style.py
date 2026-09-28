"""Style models (§22)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Pacing(BaseModel):
    avg_shot_beats: dict[str, float] = Field(default_factory=dict)  # per section role
    on_grid_ratio: float = 0.75


class TransitionPolicy(BaseModel):
    palette: list[str] = Field(default_factory=list)
    max_per_10s: float = 2.0


class SfxPolicy(BaseModel):
    density_max_per_10s: float = 3.0
    rules: str = "default"


class TextPolicy(BaseModel):
    captions: str | None = None
    titles: str | None = None
    stamps: bool = False


class ColorPolicy(BaseModel):
    lut: str | None = None
    normalize_strength: float = 0.7
    grain: float = 0.0


class ReframePolicy(BaseModel):
    policy: str = "auto_subject"
    ease: str = "glide"


class Style(BaseModel):
    name: str
    version: int = 1
    pacing: Pacing = Field(default_factory=Pacing)
    transitions: TransitionPolicy = Field(default_factory=TransitionPolicy)
    sfx: SfxPolicy = Field(default_factory=SfxPolicy)
    text: TextPolicy = Field(default_factory=TextPolicy)
    color: ColorPolicy = Field(default_factory=ColorPolicy)
    reframe: ReframePolicy = Field(default_factory=ReframePolicy)
    taste_notes: str = ""
