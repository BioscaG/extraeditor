"""Render the component gallery and its HTML index (§13.4).

A component becomes `stable` only after human approval in the gallery, so the gallery
has to show every combination that could ship: component × preset × aspect ratio. The
index is a plain HTML file so review needs nothing but a browser.

Backgrounds are synthetic plates, not real footage, so a re-render only differs when
the *component* changed — which is what makes the perceptual regression test in §27
meaningful.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from montaje.library.registry import load_components
from montaje.models.library import ComponentKind, ComponentMeta
from montaje.render.remotion_bridge import render_gallery_item
from montaje.workspace import atomic_write_text

# Sample props per component id, so text components have something to show.
SAMPLE_PROPS: dict[str, dict] = {
    "text.kinetic_title": {"lines": ["MONTAJE", "GALLERY"], "accentLine": 1},
    "text.captions_word_pop": {
        "words": [
            {"text": "this", "start": 0.0, "end": 0.35},
            {"text": "is", "start": 0.35, "end": 0.6},
            {"text": "a", "start": 0.6, "end": 0.75},
            {"text": "caption", "start": 0.75, "end": 1.3},
            {"text": "sample", "start": 1.3, "end": 1.9},
        ]
    },
    "shot_fx.beat_pulse": {"beats": [0.0, 0.5, 1.0, 1.5, 2.0]},
}

ASPECT_SIZES = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
}


@dataclass
class GalleryEntry:
    ref: str
    kind: str
    status: str
    preset: str | None
    aspect: str
    video: str
    still: str
    intent: str


def _duration_frames(meta: ComponentMeta, fps: float = 30.0, bpm: float = 120.0) -> int:
    """How long to render this component in the gallery.

    Transitions get their declared length; everything else gets a couple of seconds,
    long enough to see the whole animation settle.
    """
    if meta.kind == ComponentKind.TRANSITION:
        beats = meta.duration.default_beats or 0.5
        frames = round(beats * (60 / bpm) * fps)
        return max(meta.duration.min_frames, min(meta.duration.max_frames, frames)) or 1
    return min(meta.duration.max_frames, round(fps * 2.0))


def render_gallery(
    out_dir: Path,
    *,
    base_width: int = 1080,
    base_height: int = 1920,
    aspects: list[str] | None = None,
    scale: float = 0.5,
    fps: float = 30.0,
    bpm: float = 120.0,
) -> list[GalleryEntry]:
    """Render every component × preset × aspect ratio into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[GalleryEntry] = []

    for meta in load_components():
        # The hard cut renders nothing by design; a blank clip in the gallery would
        # just look like a broken component.
        if meta.id == "transition.hard_cut":
            continue
        presets: list[str | None] = [None, *sorted(meta.presets)]
        wanted = aspects or meta.aspect_ratios
        for aspect in wanted:
            if aspect not in meta.aspect_ratios or aspect not in ASPECT_SIZES:
                continue
            width, height = ASPECT_SIZES[aspect]
            width, height = round(width * scale / 2) * 2, round(height * scale / 2) * 2
            duration = _duration_frames(meta, fps, bpm)
            for preset in presets:
                slug = _slug(meta.ref, preset, aspect)
                video = out_dir / f"{slug}.mp4"
                still = out_dir / f"{slug}.png"
                render_gallery_item(
                    meta.ref, video, preset=preset,
                    params=SAMPLE_PROPS.get(meta.id, {}),
                    width=width, height=height,
                    duration_in_frames=duration, bpm=bpm,
                )
                render_gallery_item(
                    meta.ref, still, preset=preset,
                    params=SAMPLE_PROPS.get(meta.id, {}),
                    width=width, height=height,
                    duration_in_frames=duration, bpm=bpm,
                    # Mid-animation: the frame that actually shows what it does.
                    still_frame=max(0, duration // 2),
                )
                entries.append(GalleryEntry(
                    ref=meta.ref, kind=meta.kind.value, status=meta.status.value,
                    preset=preset, aspect=aspect,
                    video=video.name, still=still.name, intent=meta.intent,
                ))

    atomic_write_text(out_dir / "index.json",
                      json.dumps([asdict(e) for e in entries], indent=2))
    atomic_write_text(out_dir / "index.html", _html(entries))
    return entries


def _slug(ref: str, preset: str | None, aspect: str) -> str:
    parts = [ref.replace("@", "-").replace(".", "_")]
    parts.append(preset or "default")
    parts.append(aspect.replace(":", "x"))
    return "__".join(parts)


def _html(entries: list[GalleryEntry]) -> str:
    """A self-contained review page: hover to play, grouped by component."""
    by_ref: dict[str, list[GalleryEntry]] = {}
    for e in entries:
        by_ref.setdefault(e.ref, []).append(e)

    cards = []
    for ref, group in by_ref.items():
        intent = group[0].intent
        items = "".join(
            f"""
        <figure>
          <video src="{e.video}" muted loop playsinline
                 onmouseover="this.play()" onmouseout="this.pause();this.currentTime=0"
                 poster="{e.still}"></video>
          <figcaption>{e.preset or "default"} · {e.aspect}</figcaption>
        </figure>"""
            for e in group
        )
        cards.append(f"""
    <section>
      <h2>{ref} <small>{group[0].kind} · {group[0].status}</small></h2>
      <p class="intent">{intent}</p>
      <div class="row">{items}</div>
    </section>""")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>montaje craft library</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ font: 15px/1.5 -apple-system, system-ui, sans-serif; background:#0d0d10; color:#e8e8ea;
         margin:0; padding:32px 40px; }}
  h1 {{ font-size:24px; margin:0 0 4px; }}
  .lede {{ color:#9a9aa2; margin:0 0 32px; max-width:60ch; }}
  section {{ border-top:1px solid #23232a; padding:24px 0; }}
  h2 {{ font-size:17px; margin:0 0 4px; font-family:ui-monospace,monospace; }}
  h2 small {{ color:#8a8a92; font-family:inherit; font-weight:400; margin-left:8px; }}
  .intent {{ color:#a8a8b0; margin:0 0 16px; max-width:70ch; }}
  .row {{ display:flex; flex-wrap:wrap; gap:16px; }}
  figure {{ margin:0; }}
  video {{ height:280px; background:#000; border-radius:8px; display:block;
           border:1px solid #23232a; }}
  figcaption {{ color:#8a8a92; font-size:13px; margin-top:6px; font-family:ui-monospace,monospace; }}
</style>
</head>
<body>
<h1>montaje craft library</h1>
<p class="lede">Hover a clip to play it. A component becomes <code>stable</code> only
after approval here (§13.4). Backgrounds are synthetic plates so a re-render only
differs when the component itself changed.</p>
{"".join(cards)}
</body>
</html>
"""
