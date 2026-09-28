"""Common event format emitted by every local analyzer (§9)."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Event(BaseModel):
    asset_id: str
    analyzer: str  # "occlusion@1"
    type: str  # analyzer-specific, meaning-free ("reveal", "cover", "usable", …)
    t0: float
    t1: float
    score: float = 1.0
    data: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_range(self) -> Event:
        if self.t1 < self.t0:
            raise ValueError(f"event t1 < t0 ({self.t0}..{self.t1})")
        return self


class AnalyzerResult(BaseModel):
    """What an analyzer run persists to cache: events plus run metadata."""

    asset_id: str
    analyzer: str
    version: int
    params_hash: str
    events: list[Event] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)
