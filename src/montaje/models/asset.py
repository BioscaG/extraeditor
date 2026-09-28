"""Asset and probe models — the single source of truth for ingested media."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class AssetKind(StrEnum):
    VIDEO = "video"
    PHOTO = "photo"
    LIVE_PHOTO = "live_photo"
    SLOMO = "slomo"
    TIMELAPSE = "timelapse"
    SCREEN_RECORDING = "screen_recording"


class DynamicRange(StrEnum):
    SDR = "sdr"
    HDR_HLG = "hdr_hlg"
    HDR_PQ = "hdr_pq"


class VideoStream(BaseModel):
    codec: str
    width: int
    height: int
    rotation: int = 0
    pix_fmt: str | None = None
    bit_depth: int = 8
    avg_fps: float
    real_fps: float | None = None  # r_frame_rate; differs from avg on VFR
    vfr: bool = False
    color_primaries: str | None = None
    color_transfer: str | None = None  # arib-std-b67 = HLG, smpte2084 = PQ
    color_space: str | None = None
    dolby_vision: bool = False
    duration_s: float | None = None

    @property
    def dynamic_range(self) -> DynamicRange:
        if self.color_transfer == "arib-std-b67":
            return DynamicRange.HDR_HLG
        if self.color_transfer == "smpte2084":
            return DynamicRange.HDR_PQ
        return DynamicRange.SDR

    @property
    def display_size(self) -> tuple[int, int]:
        """(width, height) after applying rotation."""
        if self.rotation % 180 != 0:
            return (self.height, self.width)
        return (self.width, self.height)


class AudioStream(BaseModel):
    index: int
    codec: str
    channels: int
    sample_rate: int
    duration_s: float | None = None
    default: bool = False


class Probe(BaseModel):
    container: str
    duration_s: float
    size_bytes: int
    video: VideoStream | None = None
    audio: list[AudioStream] = Field(default_factory=list)
    creation_time: datetime | None = None  # QuickTime creationdate, TZ-aware when available
    make: str | None = None
    model: str | None = None
    raw_tags: dict[str, str] = Field(default_factory=dict)

    @property
    def selected_audio(self) -> AudioStream | None:
        """Pick a decodable audio track: prefer default, then stereo, then first."""
        if not self.audio:
            return None
        for s in self.audio:
            if s.default and s.channels >= 2:
                return s
        for s in self.audio:
            if s.channels == 2:
                return s
        return self.audio[0]


class CaptureTime(BaseModel):
    utc: datetime
    tz_offset_minutes: int | None = None
    source: str  # "quicktime" | "photokit" | "exif" | "mtime"
    confidence: float  # 1.0 quicktime/photokit, 0.7 exif, 0.2 mtime


class Asset(BaseModel):
    asset_id: str  # sha256(size ‖ first 8MB ‖ last 8MB ‖ duration), truncated
    path: Path  # original media, never modified
    kind: AssetKind
    probe: Probe
    capture: CaptureTime | None = None
    source: str = "folder"  # source plugin that produced it
    live_photo_pair: str | None = None  # asset_id of the paired still/video
    degraded: bool = False  # e.g. shared-album 720p derivative
    degraded_reason: str | None = None
    extra: dict = Field(default_factory=dict)  # source-specific metadata (PhotoKit id, location…)

    @property
    def duration_s(self) -> float:
        return self.probe.duration_s
