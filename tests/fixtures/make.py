"""Synthetic fixture generation (§27).

Every fixture has a *known* ground truth so analyzer assertions are exact:
occlusion spans at known times, click tracks at a known BPM, speech at a known
offset, HLG and VFR clips, two clips sharing audio with a known offset, and
color-shifted copies of one clip.

    python tests/fixtures/make.py [--out tests/fixtures/generated]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

FFMPEG = "ffmpeg"


def _run(args: list[str], stdin: bytes | None = None) -> None:
    subprocess.run([FFMPEG, "-y", "-v", "error", *args], check=True, input=stdin)


def make_occlusion_clip(out: Path, duration: float = 6.0, fps: int = 30) -> dict:
    """Bright moving content with black, featureless spans at 0–1 s and 4.5–6 s.

    Mirrors the vlog convention (hand covers the lens) without encoding it as meaning.
    """
    dark_head, dark_tail_start = 1.0, 4.5
    # testsrc2 gives high-detail content; `geq` blanks the covered spans to near-black.
    vf = (
        f"geq=lum='if(between(T,0,{dark_head})+between(T,{dark_tail_start},{duration}),4,lum(X,Y))'"
        f":cb='if(between(T,0,{dark_head})+between(T,{dark_tail_start},{duration}),128,cb(X,Y))'"
        f":cr='if(between(T,0,{dark_head})+between(T,{dark_tail_start},{duration}),128,cr(X,Y))'"
    )
    _run([
        "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate={fps}:duration={duration}",
        "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out),
    ])
    return {"occluded_spans": [[0.0, dark_head], [dark_tail_start, duration]],
            "reveal_near_s": dark_head, "cover_near_s": dark_tail_start, "duration_s": duration}


def make_click_track(out: Path, bpm: float = 120.0, duration: float = 10.0, sr: int = 48000,
                     accent_every: int = 4) -> dict:
    """Clicks exactly on the beat at a known BPM, accented on the downbeat.

    Synthesized with numpy rather than an ffmpeg expression so beat positions are
    sample-exact ground truth for the beat/downbeat assertions.
    """
    import numpy as np

    period = 60.0 / bpm
    x = np.zeros(int(duration * sr), dtype=np.float32)
    beats: list[float] = []
    for i in range(int(duration / period)):
        t = i * period
        beats.append(round(t, 6))
        start = int(t * sr)
        n = int(0.04 * sr)
        env = np.exp(-np.linspace(0, 12, n)).astype(np.float32)
        freq = 1600.0 if i % accent_every == 0 else 1000.0
        amp = 0.9 if i % accent_every == 0 else 0.5
        click = amp * env * np.sin(2 * np.pi * freq * np.arange(n) / sr).astype(np.float32)
        x[start : start + len(click)] += click[: len(x) - start]

    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    _run(["-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0", "-c:a", "pcm_s16le", str(out)],
         stdin=pcm.tobytes())
    return {"bpm": bpm, "beats": beats, "downbeats": beats[::accent_every], "duration_s": duration}


def make_speech_over_music(out: Path, speech_at_s: float = 2.0, duration: float = 8.0,
                           text: str = "This is a test of the montaje speech detector") -> dict:
    """A video whose audio is macOS `say` speech at a known offset over a quiet tone bed.

    Packaged as video because the folder source only ingests footage; the speech
    detector is asserted through the real ingest → analyze path.
    """
    if shutil.which("say") is None:
        return {"skipped": "no `say` binary (macOS only)"}
    aiff = out.with_suffix(".speech.aiff")
    subprocess.run(["say", "-o", str(aiff), text], check=True)
    ms = int(speech_at_s * 1000)
    _run([
        "-f", "lavfi", "-i", f"testsrc2=size=320x240:rate=30:duration={duration}",
        "-f", "lavfi", "-i", f"sine=frequency=110:sample_rate=48000:duration={duration}",
        "-i", str(aiff),
        "-filter_complex",
        f"[1:a]volume=0.05[bed];[2:a]adelay={ms}|{ms},volume=1.6[sp];"
        "[bed][sp]amix=inputs=2:duration=first:dropout_transition=0[a]",
        "-map", "0:v", "-map", "[a]", "-shortest",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", str(out),
    ])
    aiff.unlink(missing_ok=True)
    return {"speech_at_s": speech_at_s, "text": text, "duration_s": duration}


def make_hlg_clip(out: Path, duration: float = 4.0, fps: int = 30) -> dict:
    """10-bit HLG clip to exercise the HDR→SDR path."""
    _run([
        "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate={fps}:duration={duration}",
        "-vf", "format=yuv420p10le",
        "-c:v", "libx265", "-crf", "24", "-tag:v", "hvc1",
        "-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc",
        # The container flags alone are not enough: the HEVC VUI must carry them too,
        # or ffprobe reports `unknown` and the HDR path never triggers.
        "-x265-params", "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc",
        str(out),
    ])
    return {"color_transfer": "arib-std-b67", "expected_dynamic_range": "hdr_hlg",
            "expected_bit_depth": 10}


def make_vfr_clip(out: Path, duration: float = 4.0) -> dict:
    """Variable frame rate: frames selected unevenly from a 60 fps source.

    Keeping the original timestamps (`-fps_mode passthrough`) leaves uneven PTS
    gaps, so ffprobe reports avg_frame_rate well below r_frame_rate — which is
    exactly the condition the ingest probe must flag (§28).
    """
    _run([
        "-f", "lavfi", "-i", f"testsrc2=size=320x240:rate=60:duration={duration}",
        # Keep only frames that are neither every-3rd nor every-7th: irregular gaps.
        "-vf", "select='mod(n,3)*mod(n,7)'",
        "-fps_mode", "passthrough",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out),
    ])
    return {"expected_vfr": True, "source_rate": 60}


def make_sync_pair(out_a: Path, out_b: Path, offset_s: float = 1.3, duration: float = 6.0,
                   sr: int = 48000) -> dict:
    """Two clips recording the same audio, offset by a known amount.

    The shared audio is band-limited noise (broadband, so cross-correlation has a
    single sharp peak) with independent noise added per clip, like two phones.
    """
    import numpy as np

    rng = np.random.default_rng(7)
    shared = rng.standard_normal(int((duration + offset_s) * sr)).astype(np.float32) * 0.3
    for path, delay in ((out_a, 0.0), (out_b, offset_s)):
        start = int(delay * sr)
        track = shared[start : start + int(duration * sr)].copy()
        track += rng.standard_normal(len(track)).astype(np.float32) * 0.02
        pcm = (np.clip(track, -1, 1) * 32767).astype("<i2")
        _run([
            "-f", "lavfi", "-i", f"testsrc2=size=320x240:rate=30:duration={duration}",
            "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
            "-map", "0:v", "-map", "1:a", "-shortest",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(path),
        ], stdin=pcm.tobytes())
    # out_b starts `offset_s` later in the shared timeline, so b lags a by -offset_s.
    return {"offset_s": offset_s, "note": "b's audio starts offset_s into a's audio"}


def make_color_shifted(src: Path, out_warm: Path, out_cool: Path,
                       warm_k: int = 2800, cool_k: int = 9000) -> dict:
    """Warm and cool copies of one clip: normalization must pull them back together.

    `colortemperature` is used rather than `colorbalance` because it applies a
    physically meaningful white-point shift across the whole tonal range —
    colorbalance's shadow/midtone gains barely move Lab b* on synthetic content,
    which made the pair indistinguishable to `color_stats`.
    """
    _run(["-i", str(src), "-vf", f"colortemperature=temperature={warm_k}",
          "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_warm)])
    _run(["-i", str(src), "-vf", f"colortemperature=temperature={cool_k}",
          "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_cool)])
    return {"pair": [out_warm.name, out_cool.name], "warm_k": warm_k, "cool_k": cool_k}


def make_all(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    truth: dict = {}
    truth["occlusion.mp4"] = make_occlusion_clip(out_dir / "occlusion.mp4")
    truth["click_120.wav"] = make_click_track(out_dir / "click_120.wav", bpm=120)
    truth["click_90.wav"] = make_click_track(out_dir / "click_90.wav", bpm=90)
    truth["speech.mp4"] = make_speech_over_music(out_dir / "speech.mp4")
    truth["hlg.mp4"] = make_hlg_clip(out_dir / "hlg.mp4")
    truth["vfr.mp4"] = make_vfr_clip(out_dir / "vfr.mp4")
    truth["sync"] = make_sync_pair(out_dir / "sync_a.mp4", out_dir / "sync_b.mp4")
    plain = out_dir / "plain.mp4"
    _run(["-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=4",
          "-c:v", "libx264", "-pix_fmt", "yuv420p", str(plain)])
    truth["color"] = make_color_shifted(plain, out_dir / "warm.mp4", out_dir / "cool.mp4")
    (out_dir / "truth.json").write_text(__import__("json").dumps(truth, indent=2))
    return truth


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "generated")
    args = ap.parse_args()
    truth = make_all(args.out)
    print(f"Wrote {len(list(args.out.iterdir()))} files to {args.out}")
