"""Analysis proxy: H.264 8-bit SDR, short edge 540, constant fps, 1 s GOP (§8 step 5)."""

from __future__ import annotations

from pathlib import Path

from montaje import ffmpeg
from montaje.config import ProxyConfig
from montaje.ingest.hdr import needs_tonemap, requires_hw_frames, resolve_method, sdr_filter
from montaje.models.asset import Probe

PROXY_VERSION = 1


def proxy_params(cfg: ProxyConfig, hdr_method: str) -> dict:
    # The *resolved* method goes in the cache key: if the machine gains zscale,
    # proxies built with the fallback are correctly invalidated.
    return {
        "version": PROXY_VERSION,
        "short_edge": cfg.short_edge,
        "fps": cfg.fps,
        "gop_s": cfg.gop_s,
        "hdr_method": resolve_method(hdr_method),
    }


def build_proxy(
    src: Path,
    dest: Path,
    probe: Probe,
    cfg: ProxyConfig,
    hdr_method: str = "zscale_hable",
    hwaccel: bool = True,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    method = resolve_method(hdr_method)
    tonemap = needs_tonemap(probe.video)
    tm_chain = sdr_filter(probe.video, method)
    hw_tonemap = tonemap and requires_hw_frames(method)

    decode: list[str] = []
    filters: list[str] = []
    if hw_tonemap:
        # The VideoToolbox chain needs hardware frames all the way to scale_vt,
        # so scaling happens inside that filter and the chain ends in hwdownload.
        decode = ["-hwaccel", "videotoolbox", "-hwaccel_output_format", "videotoolbox_vld"]
        filters.append(f"scale_vt=w=-2:h={cfg.short_edge}")
        filters.append(tm_chain)
    else:
        if hwaccel:
            decode = ["-hwaccel", "videotoolbox"]
        # Downscale before tone-mapping: linearizing at 540p is far cheaper than at 4K.
        filters.append(f"scale=-2:'min({cfg.short_edge},ih)'")
        if tm_chain:
            filters.append(tm_chain)
    filters.append(f"fps={cfg.fps}")

    tmp = Path(str(dest) + ".tmp.mp4")
    args = ["-y", "-v", "error", *decode, "-i", str(src), "-vf", ",".join(filters)]
    args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"]
    args += ["-g", str(round(cfg.fps * cfg.gop_s)), "-movflags", "+faststart"]
    if probe.selected_audio is not None:
        args += ["-map", "0:v:0", "-map", f"0:{probe.selected_audio.index}",
                 "-c:a", "aac", "-ar", "48000"]
    else:
        args += ["-an"]
    args += [str(tmp)]
    ffmpeg.run(args)
    tmp.replace(dest)
