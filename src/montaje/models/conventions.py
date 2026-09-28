"""Discovered footage conventions (§12) — data, never code."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ConventionStatus(StrEnum):
    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class ConventionEvidence(BaseModel):
    count: int = 0
    assets: list[str] = Field(default_factory=list)


class ConventionDetection(BaseModel):
    start_event: str | None = None  # e.g. "occlusion.reveal"
    end_event: str | None = None
    max_offset_s: float = 1.5


class ConventionTreatment(BaseModel):
    role: str = "segment_boundary"
    use_as_transition: str | None = None
    trim_outside: bool = True


class Convention(BaseModel):
    id: str
    status: ConventionStatus = ConventionStatus.PROPOSED
    description: str
    evidence: ConventionEvidence = Field(default_factory=ConventionEvidence)
    detection: ConventionDetection = Field(default_factory=ConventionDetection)
    treatment: ConventionTreatment = Field(default_factory=ConventionTreatment)
