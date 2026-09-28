"""Builders for plan-level tests, shared by the rails/validate/ops/rhythm suites."""

from __future__ import annotations

from montaje.models.asset import Asset, AssetKind, Probe, VideoStream
from montaje.models.editplan import (
    AudioMode,
    ComponentRef,
    Concept,
    Duration,
    EditPlan,
    Music,
    MusicSource,
    PlanFormat,
    Section,
    Shot,
    ShotAudio,
    SnapKind,
    SnapSpec,
)
from montaje.models.events import Event
from montaje.music.beats import BeatGrid


def make_asset(asset_id: str = "a_1", duration: float = 30.0, degraded: bool = False) -> Asset:
    return Asset(
        asset_id=asset_id,
        path=f"/tmp/{asset_id}.mp4",
        kind=AssetKind.VIDEO,
        probe=Probe(
            container="mov,mp4", duration_s=duration, size_bytes=1,
            video=VideoStream(codec="hevc", width=1080, height=1920, avg_fps=30.0),
        ),
        degraded=degraded,
        degraded_reason="test" if degraded else None,
    )


def make_grid(bpm: float = 120.0, duration: float = 60.0, beats_per_bar: int = 4) -> BeatGrid:
    period = 60.0 / bpm
    beats = [round(i * period, 6) for i in range(int(duration / period) + 1)]
    return BeatGrid(bpm=bpm, beats=beats, downbeats=beats[::beats_per_bar],
                    beats_per_bar=beats_per_bar, duration_s=duration, confidence=1.0)


def make_shot(
    shot_id: str = "s001",
    asset: str = "a_1",
    src_in: float = 0.0,
    src_out: float = 2.0,
    timeline_in: int = 0,
    snap_in: SnapKind = SnapKind.NONE,
    snap_out: SnapKind = SnapKind.NONE,
    intent: str = "test shot",
    section: str | None = None,
    audio_mode: AudioMode = AudioMode.MUSIC_ONLY,
    duck_db: float | None = None,
    transition: ComponentRef | None = None,
) -> Shot:
    return Shot(
        id=shot_id, asset=asset, src_in=src_in, src_out=src_out, timeline_in=timeline_in,
        snap=SnapSpec(**{"in": snap_in, "out": snap_out}),
        intent=intent, section=section,
        audio=ShotAudio(mode=audio_mode, duck_music_db=duck_db),
        transition_in=transition,
    )


def make_plan(
    shots: list[Shot] | None = None,
    fps: float = 30.0,
    sections: list[Section] | None = None,
    music_asset: str | None = "a_music",
) -> EditPlan:
    """A contiguous plan: shots laid head to tail unless they set timeline_in."""
    if shots is None:
        shots = [
            make_shot("s001", src_in=1.0, src_out=3.0, timeline_in=0),
            make_shot("s002", src_in=5.0, src_out=7.0, timeline_in=60),
        ]
    return EditPlan(
        version=1,
        project="test",
        format=PlanFormat(width=1080, height=1920, fps=fps),
        concept=Concept(title="test", sections=sections or []),
        music=Music(source=MusicSource(mode="provided"), asset=music_asset),
        shots=shots,
    )


def usable_events(asset_id: str = "a_1", t0: float = 0.0, t1: float = 30.0) -> list[Event]:
    return [Event(asset_id=asset_id, analyzer="quality@1", type="usable", t0=t0, t1=t1)]


def transition_ref(component: str = "transition.whip_pan@1.0", frames: int | None = None,
                   beats: float | None = None) -> ComponentRef:
    duration = None
    if frames is not None:
        duration = Duration(frames=frames)
    elif beats is not None:
        duration = Duration(beats=beats)
    return ComponentRef(id=component, duration=duration)
