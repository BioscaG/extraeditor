"""Audio mix: music edit, clip audio, ducking, SFX, loudness (§14.3, §19.2).

Built as an ffmpeg filter_complex graph so the whole mix is one deterministic pass.
Stems (music, dialogue, SFX) are rendered from the same graph, which guarantees the
stems actually sum to the master rather than being a separate approximation.

Two-pass loudness: measure, then apply. One-pass `loudnorm` is a dynamic
normalizer, which pumps on a music bed; measuring first and applying a fixed gain
preserves the mix's own dynamics (§14.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from montaje import ffmpeg
from montaje.analysis.local.loudness import measure_loudness
from montaje.models.editplan import AudioMode, EditPlan

# Envelope shape for ducking under speech. Fast enough not to clip the first word,
# slow enough not to sound like a gate.
DUCK_ATTACK_S = 0.15
DUCK_RELEASE_S = 0.45


@dataclass
class ClipAudio:
    """One shot's original audio, placed on the timeline."""

    shot_id: str
    wav: Path
    src_in: float
    src_out: float
    timeline_s: float
    gain_db: float = 0.0
    duck_music_db: float | None = None
    cleanup: bool = False
    j_cut_s: float = 0.0
    l_cut_s: float = 0.0

    @property
    def start_s(self) -> float:
        """Where this audio begins, including the J-cut lead-in."""
        return max(0.0, self.timeline_s - self.j_cut_s)

    @property
    def end_s(self) -> float:
        return self.timeline_s + (self.src_out - self.src_in) + self.l_cut_s


@dataclass
class SfxPlacement:
    sfx_id: str
    wav: Path
    timeline_s: float  # where the file starts, after peak alignment
    gain_db: float = 0.0


@dataclass
class MixSpec:
    duration_s: float
    music_wav: Path | None = None
    music_segments: list[tuple[float, float, float]] = field(default_factory=list)
    music_gain_db: float = -1.0
    clips: list[ClipAudio] = field(default_factory=list)
    sfx: list[SfxPlacement] = field(default_factory=list)
    target_lufs: float = -14.0
    true_peak_dbtp: float = -1.0
    sample_rate: int = 48000


def _db(value: float) -> str:
    return f"{value:.3f}dB"


def duck_envelope_expression(clips: list[ClipAudio], duration_s: float) -> str | None:
    """A `volume` expression that dips the music under each speaking clip.

    One expression for the whole timeline rather than a sidechain compressor: the
    dip is driven by the *plan* (where speech was placed, including J/L cuts), not by
    the signal, so it is exactly reproducible and lands ahead of the first syllable.
    """
    speaking = [c for c in clips if c.duck_music_db is not None]
    if not speaking:
        return None
    terms: list[str] = []
    for c in speaking:
        gain = 10 ** (c.duck_music_db / 20.0)
        depth = 1.0 - gain
        a0 = max(0.0, c.start_s - DUCK_ATTACK_S)
        a1 = c.start_s
        r0 = c.end_s
        r1 = min(duration_s, c.end_s + DUCK_RELEASE_S)
        # Ramp in, hold, ramp out. Each clip contributes its own reduction; the
        # deepest one wins via the running minimum below.
        terms.append(
            f"if(between(t,{a0:.4f},{a1:.4f}),{depth:.5f}*(t-{a0:.4f})/{max(1e-6, a1 - a0):.4f},"
            f"if(between(t,{a1:.4f},{r0:.4f}),{depth:.5f},"
            f"if(between(t,{r0:.4f},{r1:.4f}),{depth:.5f}*(1-(t-{r0:.4f})/{max(1e-6, r1 - r0):.4f}),0)))"
        )
    reduction = terms[0]
    for term in terms[1:]:
        reduction = f"max({reduction},{term})"
    return f"volume=volume='1-({reduction})':eval=frame"


def _dialogue_chain() -> str:
    """Speech cleanup: high-pass, denoise, presence, gentle compression (§14.3)."""
    return (
        "highpass=f=80,"
        "afftdn=nr=12:nf=-25,"
        "equalizer=f=3000:t=q:w=1.2:g=2.5,"
        "acompressor=threshold=-18dB:ratio=3:attack=5:release=120:makeup=2"
    )


def build_graph(spec: MixSpec, export_stems: bool = False) -> tuple[list[str], list[str], dict[str, str]]:
    """Build `(inputs, filter_steps, stem_labels)` for the whole mix.

    With `export_stems`, each stem is split so it can be written to its own file as
    well as summed into the master. Without it no split is emitted, because an
    unconsumed filter output pad is a graph error.
    """
    inputs: list[str] = ["-f", "lavfi", "-t", f"{spec.duration_s:.4f}",
                         "-i", f"anullsrc=r={spec.sample_rate}:cl=stereo"]
    steps: list[str] = []
    stems: dict[str, str] = {}
    index = 1

    music_index = None
    if spec.music_wav is not None and spec.music_segments:
        inputs += ["-i", str(spec.music_wav)]
        music_index = index
        index += 1

    clip_indices: list[tuple[ClipAudio, int]] = []
    for clip in spec.clips:
        inputs += ["-i", str(clip.wav)]
        clip_indices.append((clip, index))
        index += 1

    sfx_indices: list[tuple[SfxPlacement, int]] = []
    for placement in spec.sfx:
        inputs += ["-i", str(placement.wav)]
        sfx_indices.append((placement, index))
        index += 1

    # -- music bed
    if music_index is not None:
        parts: list[str] = []
        for i, (src_in, src_out, timeline_in) in enumerate(spec.music_segments):
            length = src_out - src_in
            if length <= 0:
                continue
            fade = 0.05 if i > 0 else 0.0
            chain = (f"[{music_index}:a]atrim=start={src_in:.4f}:end={src_out:.4f},"
                     f"asetpts=PTS-STARTPTS,aresample={spec.sample_rate}")
            if fade > 0:
                chain += (f",afade=t=in:st=0:d={fade:.3f}"
                          f",afade=t=out:st={max(0.0, length - fade):.4f}:d={fade:.3f}")
            ms = int(round(timeline_in * 1000))
            chain += f",adelay={ms}|{ms}[m{i}]"
            steps.append(chain)
            parts.append(f"[m{i}]")
        if parts:
            joined = "".join(parts)
            steps.append(f"{joined}amix=inputs={len(parts)}:normalize=0[mraw]")
            duck = duck_envelope_expression(spec.clips, spec.duration_s)
            chain = f"[mraw]volume={_db(spec.music_gain_db)}"
            if duck:
                chain += f",{duck}"
            steps.append(f"{chain}[music]")
            stems["music"] = "music"

    # -- dialogue / clip audio
    dialogue_parts: list[str] = []
    for clip, idx in clip_indices:
        ms = int(round(clip.start_s * 1000))
        chain = (f"[{idx}:a]atrim=start={clip.src_in:.4f}:end={clip.src_out:.4f},"
                 f"asetpts=PTS-STARTPTS,aresample={spec.sample_rate}")
        if clip.cleanup:
            chain += f",{_dialogue_chain()}"
        if abs(clip.gain_db) > 1e-3:
            chain += f",volume={_db(clip.gain_db)}"
        # Short fades at both ends: a hard in-point on a waveform clicks.
        chain += ",afade=t=in:st=0:d=0.01"
        chain += f",adelay={ms}|{ms}[d{len(dialogue_parts)}]"
        steps.append(chain)
        dialogue_parts.append(f"[d{len(dialogue_parts)}]")
    if dialogue_parts:
        steps.append("".join(dialogue_parts)
                     + f"amix=inputs={len(dialogue_parts)}:normalize=0[dialogue]")
        stems["dialogue"] = "dialogue"

    # -- sfx
    sfx_parts: list[str] = []
    for placement, idx in sfx_indices:
        ms = int(round(max(0.0, placement.timeline_s) * 1000))
        chain = (f"[{idx}:a]aresample={spec.sample_rate},volume={_db(placement.gain_db)},"
                 f"adelay={ms}|{ms}[f{len(sfx_parts)}]")
        steps.append(chain)
        sfx_parts.append(f"[f{len(sfx_parts)}]")
    if sfx_parts:
        steps.append("".join(sfx_parts) + f"amix=inputs={len(sfx_parts)}:normalize=0[sfx]")
        stems["sfx"] = "sfx"

    # Each stem feeds both the master bus and its own output file. A filter output
    # label may only be consumed once, so split it explicitly — which is also what
    # guarantees the exported stems are literally the ones that were summed.
    bus_labels: list[str] = []
    for name, label in list(stems.items()):
        if export_stems:
            steps.append(f"[{label}]asplit=2[{label}_bus][{label}_out]")
            bus_labels.append(f"[{label}_bus]")
            stems[name] = f"{label}_out"
        else:
            bus_labels.append(f"[{label}]")

    # Sum over the silent bed so the master is always exactly `duration_s` long.
    bus = ["[0:a]", *bus_labels]
    steps.append(
        "".join(bus) + f"amix=inputs={len(bus)}:normalize=0,"
        f"alimiter=limit={spec.true_peak_dbtp:.2f}dB:level=disabled,"
        f"atrim=end={spec.duration_s:.4f}[master]"
    )
    return inputs, steps, stems


def render_mix(spec: MixSpec, dest: Path, stems_dir: Path | None = None) -> dict:
    """Render the master mix (and stems), then normalize loudness in a second pass."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    inputs, steps, stems = build_graph(spec, export_stems=stems_dir is not None)

    raw = dest.with_suffix(".raw.wav")
    args = ["-y", "-v", "error", *inputs, "-filter_complex", ";".join(steps),
            "-map", "[master]", "-c:a", "pcm_s24le", "-ar", str(spec.sample_rate), str(raw)]
    if stems_dir is not None:
        stems_dir.mkdir(parents=True, exist_ok=True)
        for name, label in stems.items():
            args += ["-map", f"[{label}]", "-c:a", "pcm_s24le",
                     "-ar", str(spec.sample_rate), str(stems_dir / f"{name}.wav")]
    ffmpeg.run(args)

    # Pass 2: measure, then apply one fixed gain. A single-pass loudnorm is dynamic
    # and pumps under a music bed.
    measured = measure_loudness(raw)
    integrated = measured.get("integrated_lufs", spec.target_lufs)
    if integrated == float("-inf"):
        gain_db = 0.0
    else:
        gain_db = spec.target_lufs - integrated
    tmp = dest.with_suffix(".tmp.wav")
    ffmpeg.run([
        "-y", "-v", "error", "-i", str(raw),
        "-af", f"volume={_db(gain_db)},alimiter=limit={spec.true_peak_dbtp:.2f}dB:level=disabled",
        "-c:a", "pcm_s24le", "-ar", str(spec.sample_rate), str(tmp),
    ])
    tmp.replace(dest)
    raw.unlink(missing_ok=True)

    final = measure_loudness(dest)
    return {
        "measured_lufs": integrated,
        "applied_gain_db": round(gain_db, 3),
        "final_lufs": final.get("integrated_lufs"),
        "final_true_peak_dbtp": final.get("true_peak_dbtp"),
        "stems": sorted(stems),
    }


def mix_spec_from_plan(
    plan: EditPlan,
    *,
    audio_paths: dict[str, Path],
    music_wav: Path | None,
    sfx_paths: dict[str, Path] | None = None,
    sfx_peak_offsets: dict[str, float] | None = None,
    target_lufs: float = -14.0,
    true_peak_dbtp: float = -1.0,
) -> MixSpec:
    """Translate an EditPlan's audio intentions into a MixSpec."""
    fps = plan.format.fps
    duration_s = plan.timeline_end_frame() / fps
    sfx_paths = sfx_paths or {}
    sfx_peak_offsets = sfx_peak_offsets or {}

    clips: list[ClipAudio] = []
    for shot in plan.sorted_shots():
        if shot.audio.mode in (AudioMode.MUTED, AudioMode.MUSIC_ONLY):
            continue
        wav = audio_paths.get(shot.asset)
        if wav is None:
            continue
        clips.append(ClipAudio(
            shot_id=shot.id, wav=wav,
            src_in=shot.src_in, src_out=shot.src_out,
            timeline_s=shot.timeline_in / fps,
            gain_db=shot.audio.gain_db,
            duck_music_db=shot.audio.duck_music_db,
            cleanup=shot.audio.cleanup == "dialogue",
            j_cut_s=shot.audio.j_cut_frames / fps,
            l_cut_s=shot.audio.l_cut_frames / fps,
        ))

    placements: list[SfxPlacement] = []
    for fx in plan.sfx:
        wav = sfx_paths.get(fx.sfx)
        if wav is None:
            continue
        if fx.anchor_frame is not None:
            # The file's peak must land on the anchor frame, so it starts earlier by
            # its own peak offset (§14.1).
            start = fx.anchor_frame / fps - sfx_peak_offsets.get(fx.sfx, 0.0)
        else:
            start = (fx.end_on_frame or 0) / fps - _duration_of(wav)
        placements.append(SfxPlacement(sfx_id=fx.id, wav=wav, timeline_s=max(0.0, start),
                                       gain_db=fx.gain_db))

    segments = [
        (e.src_in, e.src_out, e.timeline_in / fps)
        for e in plan.music.edits
        if e.src_out > e.src_in
    ]
    return MixSpec(
        duration_s=duration_s,
        music_wav=music_wav if segments else None,
        music_segments=segments,
        clips=clips,
        sfx=placements,
        target_lufs=target_lufs,
        true_peak_dbtp=true_peak_dbtp,
    )


def _duration_of(path: Path) -> float:
    out = ffmpeg.probe(["-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)])
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0
