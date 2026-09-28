"""Layered configuration: bundled defaults ← repo config/defaults.yaml ← project overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_PATH = REPO_ROOT / "config" / "defaults.yaml"


class WorkersConfig(BaseModel):
    ingest: int = 3
    analysis: int = 4


class ProxyConfig(BaseModel):
    short_edge: int = 540
    fps: float = 30.0
    gop_s: float = 1.0


class HdrConfig(BaseModel):
    output: str = "sdr"
    method: str = "zscale_hable"


class SnapConfig(BaseModel):
    beat_window_ms: float = 120.0
    word_preroll_s: float = 0.08
    word_postroll_s: float = 0.15


class RailsConfig(BaseModel):
    min_shot_s: float = 0.3
    loudness_lufs: float = -14.0
    true_peak_dbtp: float = -1.0


class ColorConfig(BaseModel):
    normalize_strength: float = 0.7
    match_within: list[str] = Field(default_factory=lambda: ["sync_group", "section"])


class SemanticConfig(BaseModel):
    model: str = "gemini-3.8-flash"
    provider_tier: str = "paid"
    fps_short_clips: float = 3.0
    media_resolution: str = "low"


class DirectorConfig(BaseModel):
    model_iterate: str = "claude-sonnet-5"
    model_final: str = "claude-opus-5-5"
    max_steps: int = 80
    budget_usd: float = 15.0


class CriticConfig(BaseModel):
    model: str = "gemini-3.8-flash"
    max_loops: int = 3


class MusicConfig(BaseModel):
    default_mode: str = "generate"
    provider: str = "elevenlabs"
    variants: int = 2
    instrumental: bool = True
    strict_section_durations: bool = True


class WorkshopConfig(BaseModel):
    sandbox: str = "subprocess"
    cpu_s: int = 600
    mem_gb: int = 8
    network: bool = False


class AsrConfig(BaseModel):
    model: str = "large-v3-turbo"
    no_speech_prob_max: float = 0.6
    min_avg_logprob: float = -1.0


class Config(BaseModel):
    workers: WorkersConfig = Field(default_factory=WorkersConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    hdr: HdrConfig = Field(default_factory=HdrConfig)
    asr: AsrConfig = Field(default_factory=AsrConfig)
    snap: SnapConfig = Field(default_factory=SnapConfig)
    rails: RailsConfig = Field(default_factory=RailsConfig)
    color: ColorConfig = Field(default_factory=ColorConfig)
    semantic: SemanticConfig = Field(default_factory=SemanticConfig)
    director: DirectorConfig = Field(default_factory=DirectorConfig)
    critic: CriticConfig = Field(default_factory=CriticConfig)
    music: MusicConfig = Field(default_factory=MusicConfig)
    workshop: WorkshopConfig = Field(default_factory=WorkshopConfig)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(project_dir: Path | None = None) -> Config:
    data: dict[str, Any] = {}
    if DEFAULTS_PATH.exists():
        data = yaml.safe_load(DEFAULTS_PATH.read_text()) or {}
    if project_dir is not None:
        override_path = project_dir / "config.yaml"
        if override_path.exists():
            data = _deep_merge(data, yaml.safe_load(override_path.read_text()) or {})
    return Config.model_validate(data)
