"""Invoke the Remotion renderer with an EditPlan (§19.3).

Remotion receives the plan plus the *resolved* intermediates: colour is already
baked in by the conform step, so the composition only deals with timing, motion and
type. Video renders muted and the ffmpeg mix is muxed in afterwards, because ducking
envelopes and two-pass loudness belong in the audio graph (§19.2).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from montaje import ffmpeg
from montaje.models.editplan import EditPlan
from montaje.music.beats import BeatGrid
from montaje.render.conform import ConformPlan
from montaje.workspace import atomic_write_text

REPO_ROOT = Path(__file__).resolve().parents[3]
REMOTION_DIR = REPO_ROOT / "render" / "remotion"

QUALITY_PRESETS = {
    # Draft: short edge 540 and a fast encode, for the critique loop.
    "draft": {"scale": 0.5, "crf": 28, "jpeg_quality": 70},
    "final": {"scale": 1.0, "crf": 18, "jpeg_quality": 95},
}


class RemotionUnavailable(RuntimeError):
    pass


@dataclass
class ResolvedShot:
    """What the renderer needs per shot, beyond the plan itself."""

    shot_id: str
    src: str
    head_handle_s: float
    duration_in_frames: int
    background_luminance: float = 0.5
    beats: list[float] = field(default_factory=list)
    words: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "shot_id": self.shot_id,
            "src": self.src,
            "head_handle_s": round(self.head_handle_s, 4),
            "duration_in_frames": self.duration_in_frames,
            "background_luminance": round(self.background_luminance, 4),
            "beats": [round(b, 4) for b in self.beats],
            "words": self.words,
        }


def check_available() -> None:
    if shutil.which("npx") is None:
        raise RemotionUnavailable("npx not found; install Node to render with Remotion")
    if not (REPO_ROOT / "node_modules" / "remotion").is_dir():
        raise RemotionUnavailable(f"run `npm install` in {REPO_ROOT} first")


def resolve_shots(
    plan: EditPlan,
    conformed: ConformPlan,
    *,
    grid: BeatGrid | None = None,
    luminance: dict[str, float] | None = None,
    words: dict[str, list[dict]] | None = None,
) -> list[ResolvedShot]:
    """Pair each shot with its intermediate and its shot-relative timings.

    Beat and word times are made **relative to the shot** because a component is
    given `useCurrentFrame()` from its own start (§13.2) — handing it absolute
    timeline times would put every animation at the wrong moment.
    """
    fps = plan.format.fps
    luminance = luminance or {}
    words = words or {}
    out: list[ResolvedShot] = []
    for shot in plan.sorted_shots():
        result = conformed.results.get(shot.id)
        if result is None:
            continue
        start_s = shot.timeline_in / fps
        length_frames = shot.timeline_duration_frames(fps)
        end_s = start_s + length_frames / fps
        shot_beats = (
            [round(b - start_s, 4) for b in grid.beats if start_s <= b < end_s]
            if grid else []
        )
        out.append(ResolvedShot(
            shot_id=shot.id,
            # Filename only: the renderer resolves it through staticFile() against
            # the public dir, which is the intermediates directory.
            src=result.path.name,
            head_handle_s=result.head_handle_s,
            duration_in_frames=length_frames,
            background_luminance=luminance.get(shot.id, 0.5),
            beats=shot_beats,
            words=words.get(shot.id, []),
        ))
    return out


def build_props(
    plan: EditPlan,
    resolved: list[ResolvedShot],
    *,
    bpm: float = 120.0,
    palette: str = "neutral",
    finishing: dict | None = None,
) -> dict:
    """The Remotion input props: the plan's render-relevant fields plus intermediates."""
    return {
        "version": plan.version,
        "project": plan.project,
        "format": {
            "width": plan.format.width,
            "height": plan.format.height,
            "fps": plan.format.fps,
            "dynamic_range": plan.format.dynamic_range,
        },
        "style": plan.style,
        "concept": {
            "title": plan.concept.title,
            "logline": plan.concept.logline,
            "signature": plan.concept.signature,
            "sections": [s.model_dump() for s in plan.concept.sections],
        },
        "shots": [
            {
                "id": s.id,
                "asset": s.asset,
                "src_in": s.src_in,
                "src_out": s.src_out,
                "timeline_in": s.timeline_in,
                "speed": [{"from": r.from_, "to": r.to, "rate": r.rate} for r in s.speed],
                "reframe": s.reframe.model_dump(),
                "transition_in": _ref_json(s.transition_in),
                "fx": [_ref_json(f) for f in s.fx],
                "captions": s.captions.model_dump() if s.captions else None,
                "section": s.section,
                "intent": s.intent,
            }
            for s in plan.sorted_shots()
        ],
        "overlays": [o.model_dump() for o in plan.overlays],
        "bpm": bpm,
        "resolved": [r.to_json() for r in resolved],
        "palette": palette,
        "finishing": finishing,
    }


def _ref_json(ref) -> dict | None:
    if ref is None:
        return None
    duration = None
    if ref.duration is not None:
        duration = {"beats": ref.duration.beats, "frames": ref.duration.frames}
    return {"id": ref.id, "preset": ref.preset, "duration": duration,
            "props": ref.props, "on": ref.on}


def render_video(
    props: dict,
    dest: Path,
    *,
    quality: str = "draft",
    props_path: Path | None = None,
    concurrency: int | None = None,
    public_dir: Path | None = None,
) -> Path:
    """Render the Edit composition to a muted video file.

    `public_dir` must be the directory holding the conformed intermediates: the
    composition resolves each shot's filename through `staticFile()` against it.
    """
    check_available()
    preset = QUALITY_PRESETS[quality]
    # Absolute: the renderer runs with its cwd inside render/remotion, so a relative
    # destination would be written there instead of where the caller asked.
    dest = dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Props go via a file: a 100-shot plan's JSON exceeds the shell's argument limit.
    props_path = props_path or dest.with_suffix(".props.json")
    atomic_write_text(props_path, json.dumps(props, indent=2))

    tmp = dest.with_suffix(".tmp.mp4")
    cmd = [
        "npx", "remotion", "render", "Edit", str(tmp),
        f"--props={props_path}",
        f"--scale={preset['scale']}",
        f"--crf={preset['crf']}",
        f"--jpeg-quality={preset['jpeg_quality']}",
        "--log=error",
    ]
    if concurrency is not None:
        cmd.append(f"--concurrency={concurrency}")
    if public_dir is not None:
        cmd.append(f"--public-dir={public_dir.resolve()}")
    subprocess.run(cmd, check=True, cwd=REMOTION_DIR)
    tmp.replace(dest)
    return dest


def mux(video: Path, audio: Path, dest: Path, *, crf: int | None = None) -> Path:
    """Combine the muted render with the mixed audio (§19.3).

    The video stream is copied rather than re-encoded: it was already encoded at the
    chosen quality, and a second pass would only lose detail.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.mp4")
    args = ["-y", "-v", "error", "-i", str(video), "-i", str(audio),
            "-map", "0:v:0", "-map", "1:a:0"]
    if crf is None:
        args += ["-c:v", "copy"]
    else:
        args += ["-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p"]
    args += ["-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", str(tmp)]
    ffmpeg.run(args)
    tmp.replace(dest)
    return dest


def render_gallery_item(
    component_ref: str,
    dest: Path,
    *,
    preset: str | None = None,
    params: dict | None = None,
    width: int = 1080,
    height: int = 1920,
    duration_in_frames: int = 30,
    bpm: float = 120.0,
    still_frame: int | None = None,
) -> Path:
    """Render one component × preset × aspect ratio for the gallery (§13.4)."""
    check_available()
    dest = dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    props = {
        "componentId": component_ref,
        "preset": preset,
        "params": params or {},
        "bpm": bpm,
        "durationInFrames": duration_in_frames,
        "label": f"{component_ref}{f'/{preset}' if preset else ''}",
    }
    props_path = dest.with_suffix(".props.json")
    atomic_write_text(props_path, json.dumps(props))
    verb = "still" if still_frame is not None else "render"
    cmd = ["npx", "remotion", verb, "Gallery", str(dest),
           f"--props={props_path}", f"--width={width}", f"--height={height}", "--log=error"]
    if still_frame is not None:
        cmd.append(f"--frame={still_frame}")
    subprocess.run(cmd, check=True, cwd=REMOTION_DIR)
    props_path.unlink(missing_ok=True)
    return dest
