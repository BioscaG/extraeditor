"""Craft-library metadata models (§13)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ComponentKind(StrEnum):
    TRANSITION = "transition"
    SHOT_FX = "shot_fx"
    TEXT = "text"
    OVERLAY = "overlay"
    FINISHING = "finishing"
    LAYOUT = "layout"


class ComponentStatus(StrEnum):
    DRAFT = "draft"
    CANDIDATE = "candidate"
    STABLE = "stable"
    DEPRECATED = "deprecated"


class DurationSpec(BaseModel):
    min_frames: int = 1
    max_frames: int = 300
    default_beats: float | None = None
    default_frames: int | None = None


class SfxDefault(BaseModel):
    default: str
    anchor: str = "cut_point"


class ComponentMeta(BaseModel):
    """Python mirror of the TSX `meta` export — kept in sync by the registry."""

    id: str
    version: str
    kind: ComponentKind
    status: ComponentStatus = ComponentStatus.DRAFT
    duration: DurationSpec = Field(default_factory=DurationSpec)
    energy: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    beat_anchor: str | None = None
    motion_match: str | None = None
    sfx: SfxDefault | None = None
    aspect_ratios: list[str] = Field(default_factory=lambda: ["9:16", "16:9", "1:1"])
    author: str = "human"  # human | agent
    presets: dict[str, dict] = Field(default_factory=dict)
    # One line the director reads when choosing. Says when *not* to use it too, which
    # is what keeps transition and SFX overuse down (§28).
    intent: str = ""

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"


class ComponentScore(BaseModel):
    id: str
    uses: int = 0
    survivals: int = 0  # kept after user feedback
    removals: int = 0
    ratings: list[int] = Field(default_factory=list)

    @property
    def score(self) -> float:
        """0..1; Laplace-smoothed survival rate blended with explicit ratings."""
        survival = (self.survivals + 1) / (self.survivals + self.removals + 2)
        if not self.ratings:
            return survival
        rating = (sum(self.ratings) / len(self.ratings)) / 5.0
        return 0.5 * survival + 0.5 * rating


class SfxMeta(BaseModel):
    id: str
    file: str
    category: str  # whoosh | impact | riser | downlifter | tick | pop | tape_stop | crowd_bed | sub_drop
    duration_s: float
    peak_offset_s: float  # the frame of the hit
    loudness_lufs: float | None = None
    energy: str = "medium"
    license: str  # never add a file without a known license
    source: str = ""
