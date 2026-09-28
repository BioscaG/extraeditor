"""Automatic checks on the *rendered* file (§20).

Everything else in the system validates the plan. These checks look at the output, which
is the only place certain defects can appear at all:

- **black frames** — a plan can be gap-free and still render black, if an intermediate
  failed to decode or a transition exposed the frame edge;
- **frozen frames** — a plan cannot express "this clip is a still", but footage can be;
- **loudness and true peak** — measured on the muxed file, after the encoder, not on the
  mix the mixer thought it wrote;
- **text inside the safe area** — measured by sampling frames and looking at where the
  bright pixels actually are, not by trusting the component's own layout.
- **duration** — the finished file against the plan, which catches a mux that dropped
  frames.

These are cheap, deterministic, and they check claims the rest of the system makes about
itself. The model-based half of the critic (§20) is separate: it judges whether the edit is
any *good*, which none of this can.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from montaje import ffmpeg
from montaje.analysis.local.loudness import measure_loudness

# A single black frame at a cut is normal (a dip passes through it); a run is a defect.
MAX_BLACK_RUN_S = 0.25
# A freeze longer than this is either a still or a decode failure. Long enough not to fire
# on a genuinely static shot of a poster, short enough to catch a stuck decoder.
MAX_FREEZE_S = 1.5
# How far the measured loudness may sit from the target before it is worth reporting.
LOUDNESS_TOLERANCE_LU = 1.5


@dataclass
class Finding:
    check: str
    severity: str  # "error" | "warning"
    message: str
    at_s: float | None = None

    def __str__(self) -> str:
        where = f" at {self.at_s:.2f}s" if self.at_s is not None else ""
        return f"{self.severity.upper()} {self.check}{where}: {self.message}"


@dataclass
class AutoCheckReport:
    findings: list[Finding] = field(default_factory=list)
    measured: dict = field(default_factory=dict)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_markdown(self) -> str:
        lines = ["# Auto-checks", ""]
        lines.append(f"- **Result:** {'passed' if self.passed else 'FAILED'}")
        for key, value in sorted(self.measured.items()):
            lines.append(f"- {key}: {value}")
        if self.findings:
            lines += ["", "## Findings", ""]
            lines += [f"- {f}" for f in self.findings]
        return "\n".join(lines)


def _parse_spans(stderr: str, pattern: str) -> list[tuple[float, float]]:
    """Pull `start:… end:…` pairs out of an ffmpeg detector's log."""
    out: list[tuple[float, float]] = []
    for match in re.finditer(pattern, stderr):
        try:
            out.append((float(match.group(1)), float(match.group(2))))
        except (ValueError, IndexError):
            continue
    return out


def check_black(video: Path, max_run_s: float = MAX_BLACK_RUN_S) -> list[Finding]:
    """Runs of black frames. A gap-free plan can still render black."""
    proc = ffmpeg.run(
        ["-v", "info", "-i", str(video), "-vf", "blackdetect=d=0.05:pic_th=0.98",
         "-an", "-f", "null", "-"],
        check=False,
    )
    stderr = proc.stderr.decode(errors="replace")
    spans = _parse_spans(
        stderr, r"black_start:([\d.]+) black_end:([\d.]+)"
    )
    findings = []
    for start, end in spans:
        duration = end - start
        if duration <= max_run_s:
            continue  # a frame or two at a cut is a dip doing its job
        findings.append(Finding(
            check="black_frames",
            severity="error" if duration > 0.6 else "warning",
            message=f"{duration:.2f}s of black — a failed intermediate or an exposed frame edge",
            at_s=start,
        ))
    return findings


def check_freeze(video: Path, max_freeze_s: float = MAX_FREEZE_S) -> list[Finding]:
    """Frozen video. The plan cannot express a still, but a stuck decoder produces one."""
    proc = ffmpeg.run(
        ["-v", "info", "-i", str(video),
         "-vf", f"freezedetect=n=-60dB:d={max_freeze_s}", "-an", "-f", "null", "-"],
        check=False,
    )
    stderr = proc.stderr.decode(errors="replace")
    starts = [
        float(m.group(1))
        for m in re.finditer(r"freeze_start: ([\d.]+)", stderr)
    ]
    return [
        Finding(check="frozen_frames", severity="warning",
                message=f"video frozen for at least {max_freeze_s:.1f}s", at_s=start)
        for start in starts
    ]


def check_loudness(video: Path, target_lufs: float, true_peak_dbtp: float) -> list[Finding]:
    """Loudness of the *finished file*, not of the mix the mixer thought it wrote."""
    stats = measure_loudness(video)
    findings = []
    integrated = stats.get("integrated_lufs")
    if integrated is None:
        return [Finding(check="loudness", severity="warning",
                        message="could not measure loudness")]
    if integrated == float("-inf"):
        return [Finding(check="loudness", severity="error", message="the output is silent")]
    if abs(integrated - target_lufs) > LOUDNESS_TOLERANCE_LU:
        findings.append(Finding(
            check="loudness", severity="warning",
            message=f"{integrated:.1f} LUFS against a {target_lufs:.1f} target",
        ))
    peak = stats.get("true_peak_dbtp")
    if peak is not None and peak > true_peak_dbtp + 0.3:
        findings.append(Finding(
            check="true_peak", severity="error",
            message=f"{peak:.1f} dBTP exceeds the {true_peak_dbtp:.1f} ceiling — it will clip",
        ))
    return findings


def check_duration(video: Path, expected_s: float, tolerance_s: float = 0.2) -> list[Finding]:
    """The finished file against the plan. Catches a mux that dropped frames."""
    out = ffmpeg.probe(["-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(video)])
    try:
        actual = float(out.stdout.strip())
    except ValueError:
        return [Finding(check="duration", severity="error",
                        message="the output has no readable duration")]
    if abs(actual - expected_s) > tolerance_s:
        return [Finding(
            check="duration", severity="error",
            message=f"{actual:.2f}s rendered against {expected_s:.2f}s planned",
        )]
    return []


def check_safe_areas(
    video: Path,
    aspect: str = "9:16",
    platform: str = "social",
    samples: int = 12,
) -> list[Finding]:
    """Whether bright detail sits outside the safe area (§28: captions under the UI).

    Measured from the pixels rather than trusted from the component's layout: a component
    can compute a correct position and still be placed wrongly by the plan, and this is the
    only check that would notice.

    Deliberately a warning. Footage legitimately fills the frame, so a bright *region* in
    the unsafe zone is normal; what this catches is bright, high-contrast detail there — the
    signature of text — and even then it can only flag it for a human.
    """
    import numpy as np

    from montaje.analysis.base import decode_gray_frames, proxy_height

    insets = _insets(aspect, platform)
    width = 96
    try:
        height = proxy_height(video, width)
        frames, _ = decode_gray_frames(video, fps=max(0.2, samples / _duration(video)),
                                       width=width)
    except Exception as e:
        return [Finding(check="safe_areas", severity="warning",
                        message=f"could not sample frames: {e}")]
    if len(frames) == 0:
        return []

    top = int(height * insets["top"])
    bottom = int(height * (1 - insets["bottom"]))
    findings: list[Finding] = []

    for index, frame in enumerate(frames):
        f = frame.astype(np.float32)
        # Text is bright *and* bimodal: light glyphs directly against a dark scrim or a
        # dark surround, with few mid-tones between. Brightness alone catches the sky, and
        # a gradient measure does not survive the downscale — 20px glyphs become six grid
        # cells and average out below any threshold that busy footage also clears.
        for name, band in (("top", f[:top]), ("bottom", f[bottom:])):
            if band.size == 0:
                continue
            bright = (band > 225).mean()
            dark = (band < 40).mean()
            bimodal = bright + dark
            if bright > 0.05 and bimodal > 0.5:
                findings.append(Finding(
                    check="safe_areas", severity="warning",
                    message=(
                        f"bright detail against a dark surround in the {name} unsafe zone "
                        f"({bright * 100:.0f}% bright, {bimodal * 100:.0f}% bimodal) — "
                        f"check no text is hidden under the platform UI"
                    ),
                    at_s=round(index / max(1, len(frames)) * _duration(video), 2),
                ))
                break
    # One report is enough; a caption that breaks the rule breaks it on many frames.
    return findings[:1]


def _insets(aspect: str, platform: str) -> dict:
    """Safe-area insets, mirroring `library/tokens/safeAreas.ts`."""
    table = {
        "9:16": {"none": (0.05, 0.05), "social": (0.1, 0.2),
                 "reels": (0.09, 0.22), "tiktok": (0.1, 0.24)},
        "16:9": {"none": (0.04, 0.04), "social": (0.06, 0.12)},
        "1:1": {"none": (0.05, 0.05), "social": (0.07, 0.14)},
        "4:5": {"none": (0.05, 0.05), "social": (0.08, 0.16)},
    }
    top, bottom = table.get(aspect, table["9:16"]).get(platform, (0.1, 0.2))
    return {"top": top, "bottom": bottom}


def _duration(video: Path) -> float:
    out = ffmpeg.probe(["-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(video)])
    try:
        return max(0.001, float(out.stdout.strip()))
    except ValueError:
        return 0.001


def run_autochecks(
    video: Path,
    *,
    expected_duration_s: float | None = None,
    target_lufs: float = -14.0,
    true_peak_dbtp: float = -1.0,
    aspect: str = "9:16",
    platform: str = "social",
) -> AutoCheckReport:
    """Every automatic check on a rendered file (§20)."""
    report = AutoCheckReport()
    if not video.exists():
        report.findings.append(Finding(check="output", severity="error",
                                       message=f"no file at {video}"))
        return report

    report.measured["duration_s"] = round(_duration(video), 3)
    report.findings += check_black(video)
    report.findings += check_freeze(video)
    report.findings += check_loudness(video, target_lufs, true_peak_dbtp)
    report.findings += check_safe_areas(video, aspect, platform)
    if expected_duration_s is not None:
        report.findings += check_duration(video, expected_duration_s)

    stats = measure_loudness(video)
    report.measured["integrated_lufs"] = stats.get("integrated_lufs")
    report.measured["true_peak_dbtp"] = stats.get("true_peak_dbtp")
    # Errors first, so the first line of the report is the worst news.
    report.findings.sort(key=lambda f: (f.severity != "error", f.at_s or 0.0))
    return report
