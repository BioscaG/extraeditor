"""ffprobe-based probing and asset classification (§8 steps 1–4)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

from montaje import ffmpeg
from montaje.models.asset import AssetKind, AudioStream, CaptureTime, Probe, VideoStream

PHOTO_EXTS = {".heic", ".jpg", ".jpeg", ".png", ".dng"}


def _fps(rate: str | None) -> float | None:
    if not rate or rate in ("0/0", "N/A"):
        return None
    try:
        return float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        return None


def run_ffprobe(path: Path) -> dict:
    out = ffmpeg.probe(
        ["-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    )
    return json.loads(out.stdout)


def _parse_rotation(stream: dict) -> int:
    for sd in stream.get("side_data_list", []):
        if "rotation" in sd:
            return int(sd["rotation"]) % 360
    rot = stream.get("tags", {}).get("rotate")
    return int(rot) % 360 if rot else 0


def _has_dolby_vision(stream: dict) -> bool:
    return any("DOVI" in sd.get("side_data_type", "") for sd in stream.get("side_data_list", []))


def _parse_creation_time(tags: dict) -> datetime | None:
    # QuickTime `com.apple.quicktime.creationdate` keeps the local offset; `creation_time` is UTC.
    for key in ("com.apple.quicktime.creationdate", "creation_time"):
        raw = tags.get(key)
        if not raw:
            continue
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            continue
    return None


def probe_file(path: Path) -> Probe:
    data = run_ffprobe(path)
    fmt = data.get("format", {})
    fmt_tags = fmt.get("tags", {})

    video: VideoStream | None = None
    audio: list[AudioStream] = []
    for s in data.get("streams", []):
        is_cover_art = s.get("disposition", {}).get("attached_pic", 0) == 1
        if s["codec_type"] == "video" and video is None and not is_cover_art:
            avg = _fps(s.get("avg_frame_rate"))
            real = _fps(s.get("r_frame_rate"))
            video = VideoStream(
                codec=s.get("codec_name", "unknown"),
                width=int(s["width"]),
                height=int(s["height"]),
                rotation=_parse_rotation(s),
                pix_fmt=s.get("pix_fmt"),
                bit_depth=int(
                    s.get("bits_per_raw_sample") or (10 if "10" in (s.get("pix_fmt") or "") else 8)
                ),
                avg_fps=avg or real or 0.0,
                real_fps=real,
                vfr=bool(avg and real and abs(avg - real) > 0.5),
                color_primaries=s.get("color_primaries"),
                color_transfer=s.get("color_transfer"),
                color_space=s.get("color_space"),
                dolby_vision=_has_dolby_vision(s),
                duration_s=float(s["duration"]) if s.get("duration") else None,
            )
        elif s["codec_type"] == "audio":
            audio.append(
                AudioStream(
                    index=int(s["index"]),
                    codec=s.get("codec_name", "unknown"),
                    channels=int(s.get("channels", 0)),
                    sample_rate=int(s.get("sample_rate", 0)),
                    duration_s=float(s["duration"]) if s.get("duration") else None,
                    default=bool(s.get("disposition", {}).get("default", 0)),
                )
            )

    return Probe(
        container=fmt.get("format_name", "unknown"),
        duration_s=float(fmt.get("duration", 0.0)),
        size_bytes=int(fmt.get("size", path.stat().st_size)),
        video=video,
        audio=audio,
        creation_time=_parse_creation_time(fmt_tags),
        make=fmt_tags.get("com.apple.quicktime.make") or fmt_tags.get("make"),
        model=fmt_tags.get("com.apple.quicktime.model") or fmt_tags.get("model"),
        raw_tags={k: str(v) for k, v in fmt_tags.items()},
    )


def classify_kind(path: Path, probe: Probe, live_photo_pair: bool = False) -> AssetKind:
    if path.suffix.lower() in PHOTO_EXTS:
        return AssetKind.LIVE_PHOTO if live_photo_pair else AssetKind.PHOTO
    if live_photo_pair and probe.duration_s <= 4.5:
        return AssetKind.LIVE_PHOTO
    v = probe.video
    if v is not None:
        if v.avg_fps >= 100:
            return AssetKind.SLOMO
        if probe.raw_tags.get("com.apple.quicktime.software", "").lower().startswith("screen"):
            return AssetKind.SCREEN_RECORDING
        # timelapse heuristic: long wall-clock capture squeezed into a short, silent clip
        if not probe.audio and probe.duration_s < 60 and "timelapse" in str(path).lower():
            return AssetKind.TIMELAPSE
    return AssetKind.VIDEO


def capture_time(path: Path, probe: Probe) -> CaptureTime:
    """QuickTime creationdate > mtime, with source and confidence (§8 step 2)."""
    ct = probe.creation_time
    if ct is not None:
        offset = ct.utcoffset()
        return CaptureTime(
            utc=ct.astimezone(UTC),
            tz_offset_minutes=int(offset.total_seconds() // 60) if offset is not None else None,
            source="quicktime",
            confidence=1.0,
        )
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    return CaptureTime(utc=mtime, tz_offset_minutes=None, source="mtime", confidence=0.2)
