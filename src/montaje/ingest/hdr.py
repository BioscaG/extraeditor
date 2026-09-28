"""HDR → SDR policy (§8.1).

iPhone footage is HLG (often Dolby Vision) by default; output is SDR Rec.709.
Doing this correctly means linearizing the HLG/PQ signal, tone-mapping in linear
light, then encoding to bt709 — only `zscale` (libzimg) and `libplacebo` can do
the linearize step in ffmpeg, so the available method depends on the build
(see `montaje.ffmpeg`).

Chains always downscale BEFORE tone-mapping: linearizing 4K frames is the
expensive part and proxies are 540p anyway (§25).

`montaje debug tonemap` renders every available method side by side so the
decision is recorded from evidence rather than assumed.
"""

from __future__ import annotations

from pathlib import Path

from montaje import ffmpeg
from montaje.models.asset import DynamicRange, VideoStream

# Preference order. Each entry: (name, required filter, filter chain applied after scaling).
# `libplacebo` does linearization, tone-mapping and gamut mapping in one pass with
# the best perceptual result; `zscale`+`tonemap` is the classic path; `scale_vt`
# hands the conversion to VideoToolbox (fast, but its curve is not tunable);
# `naive_clip` just reinterprets the signal and blows out highlights.
_CHAINS: dict[str, tuple[str | None, str]] = {
    "libplacebo_bt709": (
        "libplacebo",
        "libplacebo=colorspace=bt709:color_primaries=bt709:color_trc=bt709:tonemapping=bt2390",
    ),
    "zscale_hable": (
        "zscale",
        "zscale=transfer=linear:npl=100,"
        "tonemap=tonemap=hable:desat=0,"
        "zscale=primaries=bt709:transfer=bt709:matrix=bt709:range=tv",
    ),
    "zscale_mobius": (
        "zscale",
        "zscale=transfer=linear:npl=100,"
        "tonemap=tonemap=mobius:desat=0,"
        "zscale=primaries=bt709:transfer=bt709:matrix=bt709:range=tv",
    ),
    # VideoToolbox path: requires hardware frames, so the caller must decode with
    # -hwaccel videotoolbox -hwaccel_output_format videotoolbox_vld.
    "videotoolbox": (
        "scale_vt",
        "scale_vt=color_transfer=bt709:color_primaries=bt709:color_matrix=bt709,"
        "hwdownload,format=nv12",
    ),
    "naive_clip": (None, "format=yuv420p"),
}

PREFERENCE = ("libplacebo_bt709", "zscale_hable", "zscale_mobius", "videotoolbox", "naive_clip")


def method_available(method: str) -> bool:
    required = _CHAINS[method][0]
    return required is None or ffmpeg.has_filter(required)


def available_methods() -> list[str]:
    return [m for m in PREFERENCE if method_available(m)]


def resolve_method(requested: str) -> str:
    """Return `requested` if this ffmpeg supports it, else the best available fallback."""
    if requested not in _CHAINS:
        raise ValueError(f"unknown tone-mapping method {requested!r}; known: {sorted(_CHAINS)}")
    if method_available(requested):
        return requested
    fallback = available_methods()[0]
    return fallback


def needs_tonemap(video: VideoStream | None) -> bool:
    return video is not None and video.dynamic_range != DynamicRange.SDR


def requires_hw_frames(method: str) -> bool:
    return method == "videotoolbox"


def sdr_filter(video: VideoStream | None, method: str = "zscale_hable") -> str | None:
    """SDR conversion chain for this source, or None if it is already SDR."""
    if not needs_tonemap(video):
        return None
    return _CHAINS[resolve_method(method)][1]


def render_tonemap_comparison(src: Path, out_dir: Path, at_s: float = 1.0) -> list[Path]:
    """Extract one still per available method, for side-by-side review (§8.1 M0 decision)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for name in available_methods():
        chain = _CHAINS[name][1]
        out = out_dir / f"{src.stem}_{name}.png"
        pre: list[str] = []
        if requires_hw_frames(name):
            pre = ["-hwaccel", "videotoolbox", "-hwaccel_output_format", "videotoolbox_vld"]
            vf = f"scale_vt=w=-2:h=540,{chain}"
        else:
            vf = f"scale=-2:540,{chain}"
        try:
            ffmpeg.run(["-y", "-v", "error", *pre, "-ss", str(at_s), "-i", str(src),
                        "-vf", vf, "-frames:v", "1", str(out)])
        except Exception:
            continue  # a method that fails on this file is simply not offered
        outputs.append(out)
    return outputs
