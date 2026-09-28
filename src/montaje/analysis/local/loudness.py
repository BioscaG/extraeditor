"""Loudness analyzer: integrated LUFS, true peak, crowd peaks (§9)."""

from __future__ import annotations

import re

from montaje import ffmpeg
from montaje.models.asset import Asset
from montaje.models.events import Event


def measure_loudness(path) -> dict:
    """Run ffmpeg loudnorm in analysis mode and parse the summary."""
    proc = ffmpeg.run(
        ["-v", "info", "-i", str(path), "-af",
         "loudnorm=I=-14:TP=-1:print_format=summary", "-f", "null", "-"],
        check=False,
    )
    text = proc.stderr.decode(errors="replace")
    out: dict[str, float] = {}
    for key, field in (
        ("Input Integrated", "integrated_lufs"),
        ("Input True Peak", "true_peak_dbtp"),
        ("Input LRA", "lra"),
        ("Input Threshold", "threshold"),
    ):
        m = re.search(rf"{key}:\s*(-?[\d.]+|-inf)", text)
        if m:
            out[field] = float("-inf") if m.group(1) == "-inf" else float(m.group(1))
    return out


class LoudnessAnalyzer:
    name = "loudness"
    version = 1

    def params(self) -> dict:
        return {}

    def run(self, asset: Asset, ws) -> list[Event]:
        wav = ws.audio_48k_path(asset.asset_id)
        if not wav.exists():
            return []
        stats = measure_loudness(wav)
        if not stats:
            return []
        # JSON cannot hold -inf; a silent track reports a floor value instead.
        clean = {k: (-70.0 if v == float("-inf") else v) for k, v in stats.items()}
        return [Event(asset_id=asset.asset_id, analyzer=f"{self.name}@{self.version}",
                      type="loudness", t0=0.0, t1=asset.duration_s, score=1.0, data=clean)]
