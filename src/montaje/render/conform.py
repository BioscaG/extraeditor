"""Conform: cut the selected ranges from the originals and make them uniform (§19.1).

Analysis runs on proxies; pixels come from originals. For each shot this extracts
the source range **plus handles** from the original file, applies the color pipeline
(HDR→SDR, normalize, match, grade), conforms to the project's constant frame rate
and output resolution, and writes an intermediate.

Heavy work therefore scales with *output* duration, not with how much footage was
shot (§2.8) — only the selected ranges are ever touched at full resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from montaje import ffmpeg
from montaje.color.grade import Grade
from montaje.color.normalize import ShotCorrection
from montaje.ingest.hashing import params_hash
from montaje.ingest.hdr import needs_tonemap, requires_hw_frames, resolve_method, sdr_filter
from montaje.models.asset import Asset, AssetKind
from montaje.models.editplan import EditPlan, Shot

# Extra material either side of the cut, so transitions and speed ramps have
# something to work with and a late snap does not run off the end of the clip.
HANDLE_S = 0.5

CONFORM_VERSION = 1


@dataclass
class ConformSpec:
    """Everything that determines one intermediate's content — and its cache key."""

    asset: Asset
    src_in: float
    src_out: float
    width: int
    height: int
    fps: float
    hdr_method: str = "zscale_hable"
    correction: ShotCorrection | None = None
    grade: Grade | None = None
    handle_s: float = HANDLE_S
    slomo_native_fps: bool = True

    @property
    def extract_in(self) -> float:
        return max(0.0, self.src_in - self.handle_s)

    @property
    def extract_out(self) -> float:
        return min(self.asset.duration_s, self.src_out + self.handle_s)

    @property
    def head_handle_s(self) -> float:
        """Actual handle at the head, which is shorter when the shot starts near 0."""
        return self.src_in - self.extract_in

    def cache_key(self) -> str:
        return params_hash({
            "version": CONFORM_VERSION,
            "asset": self.asset.asset_id,
            "in": round(self.extract_in, 3),
            "out": round(self.extract_out, 3),
            "size": [self.width, self.height],
            "fps": self.fps,
            "hdr": resolve_method(self.hdr_method),
            "correction": self.correction.to_filter() if self.correction else None,
            "grade": self.grade.to_filter() if self.grade else None,
            "slomo_native_fps": self.slomo_native_fps,
        })

    def output_name(self) -> str:
        return f"{self.asset.asset_id}_{self.cache_key()}.mov"


@dataclass
class ConformResult:
    path: Path
    spec: ConformSpec
    head_handle_s: float
    cached: bool = False


def build_filters(spec: ConformSpec) -> tuple[list[str], list[str]]:
    """Return `(decode_args, filter_chain)` for one conform.

    Order is the color pipeline's order (§15): input transform, normalization,
    matching, creative grade, finishing. Scaling happens before the grade so the
    grade's grain and sharpening are applied at output resolution, where they were
    designed to be seen.
    """
    decode: list[str] = []
    filters: list[str] = []
    method = resolve_method(spec.hdr_method)
    hw_tonemap = needs_tonemap(spec.asset.probe.video) and requires_hw_frames(method)

    if hw_tonemap:
        decode = ["-hwaccel", "videotoolbox", "-hwaccel_output_format", "videotoolbox_vld"]
        filters.append(f"scale_vt=w={spec.width}:h={spec.height}")
        filters.append(sdr_filter(spec.asset.probe.video, method) or "")
    else:
        decode = ["-hwaccel", "videotoolbox"]
        tm = sdr_filter(spec.asset.probe.video, method)
        if tm:
            filters.append(tm)
        # Fit inside the frame preserving aspect, then pad: cropping is the
        # reframe module's job, not the conform's.
        filters.append(
            f"scale={spec.width}:{spec.height}:force_original_aspect_ratio=decrease"
        )
        filters.append(
            f"pad={spec.width}:{spec.height}:(ow-iw)/2:(oh-ih)/2:color=black"
        )

    if spec.correction is not None and (corr := spec.correction.to_filter()):
        filters.append(corr)
    if spec.grade is not None and (grade := spec.grade.to_filter()):
        filters.append(grade)

    # Slow motion keeps its native frame rate so it can be retimed later as true
    # slow motion rather than interpolated (§8.2); everything else is made constant.
    is_slomo = spec.asset.kind == AssetKind.SLOMO and spec.slomo_native_fps
    if not is_slomo:
        filters.append(f"fps={spec.fps}")
    filters.append("setsar=1")
    return decode, [f for f in filters if f]


def conform_shot(spec: ConformSpec, dest_dir: Path, force: bool = False) -> ConformResult:
    """Produce (or reuse) the intermediate for one shot."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / spec.output_name()
    if dest.exists() and not force:
        return ConformResult(path=dest, spec=spec, head_handle_s=spec.head_handle_s, cached=True)

    decode, filters = build_filters(spec)
    duration = spec.extract_out - spec.extract_in
    tmp = dest.with_suffix(".tmp.mov")
    args = [
        "-y", "-v", "error",
        *decode,
        # -ss before -i seeks fast; re-encoding makes frame accuracy exact anyway.
        "-ss", f"{spec.extract_in:.4f}",
        "-i", str(spec.asset.path),
        "-t", f"{duration:.4f}",
        "-vf", ",".join(filters),
        "-an",  # audio is mixed separately from the 48k extraction (§19.2)
        # ProRes 422 decodes cheaply and frame-accurately in Remotion's
        # OffthreadVideo; H.264 intermediates cause seek drift on long timelines.
        "-c:v", "prores_videotoolbox", "-profile:v", "2", "-pix_fmt", "yuv422p10le",
        str(tmp),
    ]
    ffmpeg.run(args)
    tmp.replace(dest)
    return ConformResult(path=dest, spec=spec, head_handle_s=spec.head_handle_s)


@dataclass
class ConformPlan:
    """All intermediates a plan needs, plus what was reused."""

    results: dict[str, ConformResult] = field(default_factory=dict)  # shot id → result

    @property
    def cached_count(self) -> int:
        return sum(1 for r in self.results.values() if r.cached)

    def path_for(self, shot_id: str) -> Path:
        return self.results[shot_id].path


def conform_plan(
    plan: EditPlan,
    assets: dict[str, Asset],
    dest_dir: Path,
    *,
    corrections: dict[str, dict[int, ShotCorrection]] | None = None,
    grade: Grade | None = None,
    hdr_method: str = "zscale_hable",
    shot_color_index: dict[str, int] | None = None,
    force: bool = False,
) -> ConformPlan:
    """Conform every shot in the plan. Shots sharing a spec share one intermediate."""
    out = ConformPlan()
    corrections = corrections or {}
    shot_color_index = shot_color_index or {}
    seen: dict[str, ConformResult] = {}

    for shot in plan.sorted_shots():
        asset = assets.get(shot.asset)
        if asset is None:
            raise KeyError(f"shot {shot.id}: asset {shot.asset} is not in the index")
        correction = None
        if shot.color.normalize != "off":
            index = shot_color_index.get(shot.id, 0)
            correction = corrections.get(shot.asset, {}).get(index)
        spec = ConformSpec(
            asset=asset, src_in=shot.src_in, src_out=shot.src_out,
            width=plan.format.width, height=plan.format.height, fps=plan.format.fps,
            hdr_method=hdr_method, correction=correction,
            grade=grade if shot.color.grade != "off" else None,
        )
        key = spec.output_name()
        if key not in seen:
            seen[key] = conform_shot(spec, dest_dir, force=force)
        out.results[shot.id] = seen[key]
    return out


def _shot_speed_expression(shot: Shot) -> str | None:
    """`setpts` expression for a shot's speed ramps, or None at unit speed.

    A single constant rate is the common case and maps to a plain `setpts`
    multiplier. Piecewise ramps are handled per-segment by the renderer.
    """
    if not shot.speed:
        return None
    if len(shot.speed) == 1 and abs(shot.speed[0].rate - 1.0) < 1e-6:
        return None
    if len(shot.speed) == 1:
        return f"setpts=PTS/{shot.speed[0].rate:.6f}"
    return None
