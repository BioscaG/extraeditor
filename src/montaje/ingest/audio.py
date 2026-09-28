"""Audio extraction: 16 kHz mono for analysis, 48 kHz stereo for mixing (§8 step 6)."""

from __future__ import annotations

from pathlib import Path

from montaje import ffmpeg
from montaje.models.asset import Probe


def extract_audio(src: Path, dest_16k: Path, dest_48k: Path, probe: Probe) -> bool:
    """Returns False when the asset has no decodable audio."""
    stream = probe.selected_audio
    if stream is None:
        return False
    dest_16k.parent.mkdir(parents=True, exist_ok=True)
    amap = f"0:{stream.index}"
    for dest, args in (
        (dest_16k, ["-ac", "1", "-ar", "16000"]),
        (dest_48k, ["-ac", "2", "-ar", "48000"]),
    ):
        tmp = Path(str(dest) + ".tmp.wav")
        ffmpeg.run(["-y", "-v", "error", "-i", str(src), "-map", amap, "-vn", *args,
                    "-c:a", "pcm_s16le", str(tmp)])
        tmp.replace(dest)
    return True
