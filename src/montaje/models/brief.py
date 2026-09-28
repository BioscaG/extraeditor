"""Project brief — project.yaml (§6)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class Format(BaseModel):
    aspect: str = "9:16"
    width: int = 1080
    height: int = 1920
    fps: float = 30.0
    dynamic_range: str = "sdr"

    @model_validator(mode="before")
    @classmethod
    def _accept_resolution_string(cls, data: dict) -> dict:
        # project.yaml uses `resolution: 1080x1920`
        if isinstance(data, dict) and "resolution" in data:
            res = str(data.pop("resolution"))
            w, h = res.lower().split("x")
            data.setdefault("width", int(w))
            data.setdefault("height", int(h))
        return data

    @field_validator("aspect")
    @classmethod
    def _known_aspect(cls, v: str) -> str:
        if v not in {"9:16", "16:9", "1:1", "4:5"}:
            raise ValueError(f"unsupported aspect {v}")
        return v


class DurationTarget(BaseModel):
    target_s: float = 75.0
    tolerance_s: float = 15.0


class MusicBrief(BaseModel):
    mode: str = "provided"  # provided | library | generate
    path: Path | None = None
    lyrics: Path | None = None


class Brief(BaseModel):
    name: str
    goal: str | None = None
    format: Format = Field(default_factory=Format)
    duration: DurationTarget = Field(default_factory=DurationTarget)
    music: MusicBrief | None = None
    style: str | None = None
    references: list[Path] = Field(default_factory=list)
    language: str | None = None
    notes: str | None = None
