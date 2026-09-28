"""ffmpeg/ffprobe discovery and capability probing.

Homebrew's default `ffmpeg` bottle ships without libzimg (`zscale`) and
libplacebo, which are the two filters that can correctly linearize HLG/PQ before
tone-mapping. Rather than hardcoding a binary, we discover the most capable
ffmpeg on the machine once and let the HDR module pick a method from what it
actually supports (§0: use the fallback and record the choice).

Override with MONTAJE_FFMPEG / MONTAJE_FFPROBE.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
from pathlib import Path

# Preference order: keg-only Homebrew builds first (they carry zimg + libplacebo),
# then whatever is on PATH.
_FFMPEG_CANDIDATES = (
    "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg",
    "/usr/local/opt/ffmpeg-full/bin/ffmpeg",
    "ffmpeg",
)


def _resolve(candidates: tuple[str, ...], env_var: str) -> str:
    override = os.environ.get(env_var)
    if override:
        return override
    for c in candidates:
        if Path(c).is_file() or shutil.which(c):
            return c
    raise FileNotFoundError(f"no ffmpeg binary found (tried {candidates}); set {env_var}")


@functools.lru_cache(maxsize=1)
def ffmpeg_bin() -> str:
    return _resolve(_FFMPEG_CANDIDATES, "MONTAJE_FFMPEG")


@functools.lru_cache(maxsize=1)
def ffprobe_bin() -> str:
    ff = ffmpeg_bin()
    sibling = Path(ff).with_name("ffprobe")
    if sibling.is_file():
        return str(sibling)
    return _resolve(("ffprobe",), "MONTAJE_FFPROBE")


@functools.lru_cache(maxsize=1)
def available_filters() -> frozenset[str]:
    out = subprocess.run(
        [ffmpeg_bin(), "-hide_banner", "-filters"], capture_output=True, text=True, check=True
    ).stdout
    names = set()
    for line in out.splitlines():
        parts = line.split()
        # Format: "TSC name  IN->OUT  description"
        if len(parts) >= 3 and not line.startswith("Filters:") and len(parts[0]) <= 3:
            names.add(parts[1])
    return frozenset(names)


def has_filter(name: str) -> bool:
    return name in available_filters()


def run(args: list[str], *, stdin: bytes | None = None, check: bool = True,
        capture: bool = True) -> subprocess.CompletedProcess:
    """Run ffmpeg with the resolved binary. `args` excludes the binary itself."""
    return subprocess.run(
        [ffmpeg_bin(), "-hide_banner", *args],
        input=stdin,
        capture_output=capture,
        check=check,
    )


def probe(args: list[str], text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([ffprobe_bin(), *args], capture_output=True, text=text, check=True)


def capability_report() -> dict[str, object]:
    """What the discovered ffmpeg can do — surfaced in the ingest report and DECISIONS."""
    version = subprocess.run(
        [ffmpeg_bin(), "-hide_banner", "-version"], capture_output=True, text=True, check=True
    ).stdout.splitlines()[0]
    return {
        "binary": ffmpeg_bin(),
        "version": version,
        "zscale": has_filter("zscale"),
        "libplacebo": has_filter("libplacebo"),
        "scale_vt": has_filter("scale_vt"),
        "tonemap": has_filter("tonemap"),
    }
