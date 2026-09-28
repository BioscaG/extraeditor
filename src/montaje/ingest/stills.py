"""Thumbnail strips: 1 fps overall + 4 fps for the first/last 2 s (§8 step 7)."""

from __future__ import annotations

from pathlib import Path

from montaje import ffmpeg


def extract_thumbs(proxy: Path, dest_dir: Path, duration_s: float) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg.run(["-y", "-v", "error", "-i", str(proxy),
                "-vf", "fps=1,scale=-2:180", "-q:v", "5", str(dest_dir / "t_%04d.jpg")])
    # Dense edges: clip starts/ends carry conventions (hand covers, reveals).
    for tag, args in (
        ("head", ["-t", "2"]),
        ("tail", ["-sseof", "-2"]),
    ):
        if duration_s <= 2 and tag == "tail":
            continue
        pre = args if tag == "tail" else []
        post = args if tag == "head" else []
        ffmpeg.run(["-y", "-v", "error", *pre, "-i", str(proxy), *post,
                    "-vf", "fps=4,scale=-2:180", "-q:v", "5",
                    str(dest_dir / f"{tag}_%03d.jpg")])
