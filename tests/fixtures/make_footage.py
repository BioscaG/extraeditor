"""Generate a small synthetic "shoot" for end-to-end tests (§27 E2E).

Not a substitute for real footage — it is a *pipeline* test: enough clips, with enough
variety in motion, brightness, colour temperature and audio content, that the planner
has real choices to make, the color pipeline has real differences to reconcile, and
the rails have real boundaries to snap to.

    python tests/fixtures/make_footage.py --out tests/fixtures/shoot
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

FFMPEG = shutil.which("/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg") or "ffmpeg"


def _run(args: list[str], stdin: bytes | None = None) -> None:
    subprocess.run([FFMPEG, "-y", "-v", "error", *args], check=True, input=stdin)


def _clip(
    out: Path,
    *,
    duration: float,
    hue: int,
    motion: str,
    brightness: float,
    temperature: int,
    audio: str,
    fps: int = 30,
    size: str = "720x1280",
) -> dict:
    """One clip: a moving pattern with a known dominant motion direction and colour."""
    w, h = (int(v) for v in size.split("x"))
    # A scrolling gradient plus a moving marker gives the motion analyzer a real,
    # known direction to measure rather than noise.
    pan = {
        "left": f"crop={w}:{h}:'max(0,(iw-{w})*(1-t/{duration}))':0",
        "right": f"crop={w}:{h}:'min(iw-{w},(iw-{w})*t/{duration})':0",
        "static": f"crop={w}:{h}:(iw-{w})/2:0",
    }[motion]

    filters = [
        f"scale={int(w * 1.6)}:{h}",
        pan,
        f"hue=h={hue}",
        f"eq=brightness={brightness:.3f}",
        f"colortemperature=temperature={temperature}",
        f"fps={fps}",
        "format=yuv420p",
    ]

    audio_args: list[str]
    if audio == "speech" and shutil.which("say"):
        aiff = out.with_suffix(".say.aiff")
        subprocess.run(
            ["say", "-o", str(aiff), "This is a line of dialogue for the montaje test edit"],
            check=True,
        )
        audio_args = ["-i", str(aiff)]
        amix = "[1:a]volume=1.4,apad[a]"
    elif audio == "crowd":
        audio_args = ["-f", "lavfi", "-i",
                      f"anoisesrc=color=pink:amplitude=0.25:duration={duration}:sample_rate=48000"]
        amix = "[1:a]anull[a]"
    else:
        audio_args = ["-f", "lavfi", "-i",
                      f"sine=frequency=180:sample_rate=48000:duration={duration}"]
        amix = "[1:a]volume=0.08[a]"

    _run([
        "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={fps}:duration={duration}",
        *audio_args,
        "-filter_complex", f"[0:v]{','.join(filters)}[v];{amix}",
        "-map", "[v]", "-map", "[a]", "-t", str(duration),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-ar", "48000", "-shortest",
        str(out),
    ])
    if audio == "speech":
        out.with_suffix(".say.aiff").unlink(missing_ok=True)
    return {
        "file": out.name, "duration_s": duration, "motion": motion,
        "brightness": brightness, "temperature": temperature, "audio": audio,
    }


# A shoot with deliberate variety: two speaking clips, crowd-ish clips at different
# colour temperatures, panning clips in both directions, and one dark clip the quality
# analyzer should rank low.
def _bookended_clip(out: Path, duration: float = 7.0, hue: int = 20, fps: int = 30,
                    size: str = "720x1280", cover_s: float = 0.9) -> dict:
    """A clip that opens *and* closes with the lens obscured.

    This is the festival footage's convention, reproduced structurally: the first and last
    `cover_s` seconds are near-black and featureless. Nothing in the fixture or the miner
    names a hand or a lens — the point is that the motif is discoverable from
    meaning-free occlusion events alone (§2.5, §11).
    """
    w, h = (int(v) for v in size.split("x"))
    blank = (f"if(lt(T,{cover_s})+gt(T,{duration - cover_s}),4,lum(X,Y))",
             f"if(lt(T,{cover_s})+gt(T,{duration - cover_s}),128,cb(X,Y))",
             f"if(lt(T,{cover_s})+gt(T,{duration - cover_s}),128,cr(X,Y))")
    vf = (f"scale={w}:{h},hue=h={hue},"
          f"geq=lum='{blank[0]}':cb='{blank[1]}':cr='{blank[2]}',fps={fps},format=yuv420p")
    _run([
        "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={fps}:duration={duration}",
        "-f", "lavfi", "-i",
        f"anoisesrc=color=pink:amplitude=0.2:duration={duration}:sample_rate=48000",
        "-vf", vf, "-map", "0:v", "-map", "1:a", "-t", str(duration),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-ar", "48000", "-shortest", str(out),
    ])
    return {"file": out.name, "duration_s": duration, "bookended": True,
            "cover_s": cover_s}


# Clips that open and close with the lens obscured, plus clips that do not. The miner must
# find the motif in the former without being told it exists.
BOOKENDED = [
    dict(name="vlog_a.mp4", duration=7.0, hue=20),
    dict(name="vlog_b.mp4", duration=6.0, hue=35),
    dict(name="vlog_c.mp4", duration=8.0, hue=10),
    dict(name="vlog_d.mp4", duration=6.5, hue=28),
]


SHOOT = [
    dict(name="clip_01_vlog.mp4", duration=8.0, hue=10, motion="static",
         brightness=0.04, temperature=5200, audio="speech"),
    dict(name="clip_02_pan_left.mp4", duration=6.0, hue=180, motion="left",
         brightness=0.0, temperature=7600, audio="crowd"),
    dict(name="clip_03_pan_right.mp4", duration=6.0, hue=200, motion="right",
         brightness=-0.02, temperature=8200, audio="crowd"),
    dict(name="clip_04_stage.mp4", duration=7.0, hue=300, motion="static",
         brightness=0.02, temperature=3200, audio="crowd"),
    dict(name="clip_05_vlog2.mp4", duration=7.0, hue=25, motion="static",
         brightness=0.03, temperature=4800, audio="speech"),
    dict(name="clip_06_dance.mp4", duration=6.0, hue=120, motion="right",
         brightness=0.01, temperature=6500, audio="crowd"),
    dict(name="clip_07_dark.mp4", duration=5.0, hue=240, motion="static",
         brightness=-0.35, temperature=6500, audio="tone"),
    dict(name="clip_08_warm.mp4", duration=6.0, hue=40, motion="left",
         brightness=0.05, temperature=2900, audio="crowd"),
]


def make_shoot(out_dir: Path, bookended: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for spec in SHOOT:
        name = spec.pop("name")
        clips.append(_clip(out_dir / name, **spec))
        spec["name"] = name  # keep SHOOT reusable across calls
    if bookended:
        for spec in BOOKENDED:
            name = spec.pop("name")
            clips.append(_bookended_clip(out_dir / name, **spec))
            spec["name"] = name
    truth = {
        "clips": clips,
        "total_s": sum(c["duration_s"] for c in clips),
        "bookended_files": [c["file"] for c in clips if c.get("bookended")],
    }
    (out_dir / "shoot.json").write_text(json.dumps(truth, indent=2))
    return truth


def make_music(out: Path) -> dict:
    """A 96-second four-section track at 120 BPM, reusing the fixture generator."""
    from make import make_structured_track  # same directory

    return make_structured_track(out)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "shoot")
    args = ap.parse_args()
    truth = make_shoot(args.out)
    music_dir = args.out / "music"
    music_dir.mkdir(parents=True, exist_ok=True)
    music = make_music(music_dir / "track.wav")
    (args.out / "music.json").write_text(json.dumps(music, indent=2))
    print(f"Wrote {len(truth['clips'])} clips ({truth['total_s']:.0f}s) + music to {args.out}")
