"""Editable exports: SRT, OTIO, FCPXML and an alpha overlay track (§19.4).

The point of these is that the edit stays *editable*. A rendered MP4 is a dead end;
an OTIO or FCPXML that references the **originals** with the right source ranges lets
someone finish the job in Resolve. Two things make that actually work:

- timelines reference the original files, never the conformed intermediates, so the
  colour pipeline's decisions can be redone rather than baked in;
- graphics go out as a separate ProRes 4444 track *with alpha*, because titles and
  captions baked into the picture cannot be moved, restyled or removed.

`intent` travels as a marker on every clip, so the reasoning survives the handoff.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from xml.dom import minidom

from montaje.models.asset import Asset
from montaje.models.editplan import EditPlan
from montaje.models.events import Event
from montaje.workspace import atomic_write_text

# -- SRT ---------------------------------------------------------------------------


def _srt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


@dataclass
class Cue:
    start_s: float
    end_s: float
    text: str


def cues_from_plan(
    plan: EditPlan,
    events: dict[str, list[Event]],
    *,
    max_chars: int = 42,
    min_duration_s: float = 0.8,
) -> list[Cue]:
    """Subtitle cues in *timeline* time, grouped into readable lines.

    Words are grouped rather than emitted individually: one cue per word is unreadable
    as a subtitle track, even though it is exactly right for an animated caption. Cues
    are also held to a minimum duration, because a subtitle that flashes for 3 frames
    cannot be read however correct its timing is.
    """
    fps = plan.format.fps
    cues: list[Cue] = []

    for shot in plan.sorted_shots():
        words = sorted(
            (
                e for e in events.get(shot.asset, [])
                if e.analyzer.startswith("asr") and e.type == "word"
                and e.t1 > shot.src_in and e.t0 < shot.src_out
            ),
            key=lambda e: e.t0,
        )
        if not words:
            continue
        offset = shot.timeline_in / fps - shot.src_in

        group: list[Event] = []
        for word in words:
            token = str(word.data.get("text", "")).strip()
            if not token:
                continue
            candidate = " ".join([*(str(g.data["text"]).strip() for g in group), token])
            if group and len(candidate) > max_chars:
                cues.append(_cue_from(group, offset, min_duration_s))
                group = []
            group.append(word)
        if group:
            cues.append(_cue_from(group, offset, min_duration_s))

    cues.sort(key=lambda c: c.start_s)

    # The same spoken line often appears in several shots, because a reused source
    # range is reused audio. As an animated caption that is fine; as a subtitle track
    # it reads as the line being repeated three times, so overlapping duplicates are
    # collapsed into the first occurrence.
    deduped: list[Cue] = []
    for cue in cues:
        previous = deduped[-1] if deduped else None
        if previous is not None and cue.start_s < previous.end_s and _similar(previous.text, cue.text):
            previous.end_s = max(previous.end_s, cue.end_s)
            continue
        deduped.append(cue)

    # Never let one cue start before the previous one ends.
    for a, b in zip(deduped, deduped[1:], strict=False):
        if a.end_s > b.start_s:
            a.end_s = b.start_s
    return [c for c in deduped if c.end_s > c.start_s]


def _similar(a: str, b: str) -> bool:
    """Whether two cue texts are the same line, ignoring truncation at either end."""
    x, y = a.strip().lower(), b.strip().lower()
    if not x or not y:
        return False
    if x in y or y in x:
        return True
    words_x, words_y = set(x.split()), set(y.split())
    overlap = len(words_x & words_y) / max(1, min(len(words_x), len(words_y)))
    return overlap >= 0.7


def _cue_from(group: list[Event], offset: float, min_duration_s: float) -> Cue:
    text = " ".join(str(g.data["text"]).strip() for g in group)
    start = group[0].t0 + offset
    end = max(group[-1].t1 + offset, start + min_duration_s)
    return Cue(start_s=start, end_s=end, text=text)


def write_srt(cues: list[Cue], dest: Path) -> Path:
    blocks = [
        f"{i}\n{_srt_time(c.start_s)} --> {_srt_time(c.end_s)}\n{c.text}\n"
        for i, c in enumerate(cues, start=1)
    ]
    atomic_write_text(dest, "\n".join(blocks))
    return dest


def export_srt(plan: EditPlan, events: dict[str, list[Event]], dest: Path) -> Path:
    return write_srt(cues_from_plan(plan, events), dest)


# -- OpenTimelineIO ----------------------------------------------------------------


class OtioUnavailable(RuntimeError):
    pass


def build_otio(plan: EditPlan, assets: dict[str, Asset]):
    """An `otio.schema.Timeline` referencing the **originals** (§19.4).

    Built through the library rather than by writing its JSON by hand: the schema has
    required fields that are easy to miss (a hand-written `Clip.2` needs
    `media_references`, not `media_reference`, and fails to load with a bare KeyError),
    and OTIO's own writer cannot produce a file its reader rejects.
    """
    try:
        import opentimelineio as otio
    except ImportError as e:  # pragma: no cover - dependency is declared
        raise OtioUnavailable("install opentimelineio to export timelines") from e

    fps = plan.format.fps
    timeline = otio.schema.Timeline(name=plan.concept.title or plan.project)
    timeline.global_start_time = otio.opentime.RationalTime(0, fps)
    timeline.metadata["montaje"] = {
        "plan_version": plan.version,
        "project": plan.project,
        "style": plan.style,
        "logline": plan.concept.logline,
    }

    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    timeline.tracks.append(track)

    cursor = 0
    for shot in plan.sorted_shots():
        # A gap is preserved rather than closed, so the timeline stays honest about
        # what is there instead of silently sliding clips earlier.
        if shot.timeline_in > cursor:
            track.append(otio.schema.Gap(
                duration=otio.opentime.RationalTime(shot.timeline_in - cursor, fps)
            ))
        length = max(1, shot.timeline_duration_frames(fps))

        # A plan that genuinely overlaps two shots cannot go on one OTIO track, so the
        # outgoing clip is trimmed to the cut point. Appending the clips back to back
        # instead stretched the exported timeline past the edit's real length.
        overlap = cursor - shot.timeline_in
        if overlap > 0:
            _trim_last_clip(track, overlap, fps, otio)
            cursor = shot.timeline_in

        # A designed transition is an *annotation* on a butt cut in the plan — the
        # renderer derives the overlap from it. In OTIO a Transition sits between two
        # clips and borrows from each without changing their durations, which is exactly
        # the same model, so it maps across directly.
        if shot.transition_in is not None and track and cursor > 0:
            frames = shot.transition_in.duration.frames if shot.transition_in.duration else None
            handle = max(1, (frames or round(fps * 0.25)) // 2)
            track.append(otio.schema.Transition(
                name=shot.transition_in.id,
                transition_type=otio.schema.TransitionTypes.SMPTE_Dissolve,
                in_offset=otio.opentime.RationalTime(handle, fps),
                out_offset=otio.opentime.RationalTime(handle, fps),
                metadata={"montaje": {
                    "component": shot.transition_in.id,
                    "preset": shot.transition_in.preset,
                }},
            ))

        asset = assets.get(shot.asset)
        clip = otio.schema.Clip(
            name=f"{shot.id} {shot.asset}",
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(round(shot.src_in * fps), fps),
                duration=otio.opentime.RationalTime(length, fps),
            ),
        )
        if asset is not None:
            clip.media_reference = otio.schema.ExternalReference(
                target_url=asset.path.resolve().as_uri(),
                available_range=otio.opentime.TimeRange(
                    start_time=otio.opentime.RationalTime(0, fps),
                    duration=otio.opentime.RationalTime(round(asset.duration_s * fps), fps),
                ),
            )
        clip.metadata["montaje"] = {
            "shot_id": shot.id,
            "section": shot.section,
            "intent": shot.intent,
            "transition_in": shot.transition_in.id if shot.transition_in else None,
            "audio_mode": shot.audio.mode.value,
        }
        if shot.intent:
            # `intent` rides along as a marker so the reasoning survives the handoff.
            clip.markers.append(otio.schema.Marker(
                name=shot.intent,
                color=otio.schema.MarkerColor.YELLOW,
                marked_range=otio.opentime.TimeRange(
                    start_time=otio.opentime.RationalTime(round(shot.src_in * fps), fps),
                    duration=otio.opentime.RationalTime(1, fps),
                ),
            ))
        track.append(clip)
        cursor = shot.timeline_in + length

    for section in plan.concept.sections:
        track.markers.append(otio.schema.Marker(
            name=f"{section.id}: {section.music_section or section.purpose or ''}",
            color=otio.schema.MarkerColor.CYAN,
            marked_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(section.from_frame, fps),
                duration=otio.opentime.RationalTime(
                    max(1, section.to_frame - section.from_frame), fps
                ),
            ),
        ))
    return timeline


def _trim_last_clip(track, frames: int, fps: float, otio) -> None:
    """Shorten the most recent clip by `frames`, so the next one can butt against it."""
    for item in reversed(list(track)):
        if not isinstance(item, otio.schema.Clip):
            continue
        current = item.source_range
        remaining = max(1, int(current.duration.value) - frames)
        item.source_range = otio.opentime.TimeRange(
            start_time=current.start_time,
            duration=otio.opentime.RationalTime(remaining, fps),
        )
        return


def export_otio(plan: EditPlan, assets: dict[str, Asset], dest: Path) -> Path:
    import opentimelineio as otio

    dest.parent.mkdir(parents=True, exist_ok=True)
    otio.adapters.write_to_file(build_otio(plan, assets), str(dest))
    return dest


# -- FCPXML -----------------------------------------------------------------------


def fcpxml_tree(plan: EditPlan, assets: dict[str, Asset]) -> ET.Element:
    """FCPXML v1.10, which Resolve and Final Cut both import.

    Times are expressed as exact rationals (`<n>/<d>s`) rather than decimals: FCPXML
    requires it, and a decimal frame time is how an import ends up one frame off.
    """
    fps = plan.format.fps
    # A rational timebase: 30 -> 1/30s, 29.97 -> 1001/30000s.
    if abs(fps - round(fps)) < 1e-6:
        tb_num, tb_den = 1, int(round(fps))
    else:
        tb_num, tb_den = 1001, int(round(fps * 1001))
    frame_duration = f"{tb_num}/{tb_den}s"

    def t(frames: int) -> str:
        return f"{frames * tb_num}/{tb_den}s"

    root = ET.Element("fcpxml", version="1.10")
    resources = ET.SubElement(root, "resources")
    ET.SubElement(
        resources, "format", id="r0", name=f"FFVideoFormat{plan.format.height}p",
        frameDuration=frame_duration,
        width=str(plan.format.width), height=str(plan.format.height),
        colorSpace="1-1-1 (Rec. 709)",
    )

    asset_ids: dict[str, str] = {}
    for i, (asset_id, asset) in enumerate(sorted(assets.items()), start=1):
        ref = f"a{i}"
        asset_ids[asset_id] = ref
        element = ET.SubElement(
            resources, "asset", id=ref, name=asset.path.name,
            start="0s", duration=t(round(asset.duration_s * fps)),
            hasVideo="1", hasAudio="1" if asset.probe.audio else "0",
            format="r0",
        )
        ET.SubElement(element, "media-rep", kind="original-media",
                      src=asset.path.resolve().as_uri())

    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", name=plan.project)
    project = ET.SubElement(event, "project", name=plan.concept.title or plan.project)
    total = plan.timeline_end_frame()
    sequence = ET.SubElement(
        project, "sequence", format="r0", duration=t(total),
        tcStart="0s", tcFormat="NDF", audioLayout="stereo", audioRate="48k",
    )
    spine = ET.SubElement(sequence, "spine")

    cursor = 0
    for shot in plan.sorted_shots():
        if shot.timeline_in > cursor:
            ET.SubElement(spine, "gap", name="gap", offset=t(cursor),
                          duration=t(shot.timeline_in - cursor), start="0s")
        length = shot.timeline_duration_frames(fps)
        ref = asset_ids.get(shot.asset, "a1")
        clip = ET.SubElement(
            spine, "asset-clip", name=f"{shot.id}", ref=ref,
            offset=t(shot.timeline_in), duration=t(length),
            start=t(round(shot.src_in * fps)),
            audioRole="dialogue" if shot.audio.mode.value != "music_only" else "music",
        )
        if shot.intent:
            ET.SubElement(clip, "marker", start=t(round(shot.src_in * fps)),
                          duration=frame_duration, value=shot.intent)
        cursor = shot.timeline_in + length

    return root


def export_fcpxml(plan: EditPlan, assets: dict[str, Asset], dest: Path) -> Path:
    xml = ET.tostring(fcpxml_tree(plan, assets), encoding="unicode")
    pretty = minidom.parseString(xml).toprettyxml(indent="  ")
    atomic_write_text(dest, pretty)
    return dest


# -- overlay track ------------------------------------------------------------------


def export_overlays(
    props: dict,
    dest: Path,
    *,
    remotion_dir: Path | None = None,
) -> Path:
    """Render only the graphics, as ProRes 4444 with alpha (§19.4).

    Titles and captions baked into the picture cannot be moved, restyled or removed in
    Resolve. Rendering them separately over transparency is what keeps them editable.
    The shot list is emptied so only overlays and captions draw.
    """
    import subprocess

    from montaje.render.remotion_bridge import REMOTION_DIR, check_available

    check_available()
    dest.parent.mkdir(parents=True, exist_ok=True)
    graphics_props = dict(props)
    # Keep the shots (captions hang off them) but drop the video sources, so the
    # picture is transparent and only text renders.
    graphics_props["resolved"] = [
        {**r, "src": ""} for r in props.get("resolved", [])
    ]
    props_path = dest.with_suffix(".props.json")
    atomic_write_text(props_path, json.dumps(graphics_props, indent=2))

    tmp = dest.with_suffix(".tmp.mov")
    subprocess.run(
        ["npx", "remotion", "render", "Edit", str(tmp),
         f"--props={props_path}", "--codec=prores", "--prores-profile=4444",
         "--image-format=png", "--log=error"],
        check=True, cwd=remotion_dir or REMOTION_DIR,
    )
    tmp.replace(dest)
    props_path.unlink(missing_ok=True)
    return dest


def export_all(
    plan: EditPlan,
    assets: dict[str, Asset],
    events: dict[str, list[Event]],
    exports_dir: Path,
) -> dict[str, Path]:
    """Every timeline-level export for a plan version."""
    exports_dir.mkdir(parents=True, exist_ok=True)
    version = f"v{plan.version:03d}"
    return {
        "srt": export_srt(plan, events, exports_dir / f"captions_{version}.srt"),
        "otio": export_otio(plan, assets, exports_dir / f"timeline_{version}.otio"),
        "fcpxml": export_fcpxml(plan, assets, exports_dir / f"timeline_{version}.fcpxml"),
    }
