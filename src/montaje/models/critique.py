"""Critic output (§20)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class CritiqueCategory(StrEnum):
    STORY = "story"
    CONTENT = "content"
    CUT = "cut"
    AUDIO = "audio"
    CAPTION = "caption"
    VISUAL = "visual"
    COLOR = "color"


class CritiqueSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CritiqueNote(BaseModel):
    t: float
    severity: CritiqueSeverity
    category: CritiqueCategory
    note: str
    suggestion: str = ""


class Critique(BaseModel):
    plan_version: int
    model: str = ""
    notes: list[CritiqueNote] = Field(default_factory=list)
    autocheck_failures: list[str] = Field(default_factory=list)

    @property
    def high_severity(self) -> list[CritiqueNote]:
        return [n for n in self.notes if n.severity == CritiqueSeverity.HIGH]
