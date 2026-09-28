"""High-level plan operations (§17.3).

The agent never writes an EditPlan directly; it applies ops. Each op is validated
on apply and produces a new plan version plus a diff summary, so every change is
attributable and reversible.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from montaje.models.editplan import (
    Captions,
    ComponentRef,
    EditPlan,
    Overlay,
    Reframe,
    Sfx,
    Shot,
    ShotAudio,
    ShotColor,
    SpeedRamp,
)


class OpError(ValueError):
    """An op that cannot be applied to this plan."""


@dataclass(frozen=True)
class OpResult:
    plan: EditPlan
    summary: str


OpFn = Callable[[EditPlan, dict], str]
_REGISTRY: dict[str, OpFn] = {}


def op(name: str) -> Callable[[OpFn], OpFn]:
    def register(fn: OpFn) -> OpFn:
        _REGISTRY[name] = fn
        return fn
    return register


def available_ops() -> list[str]:
    return sorted(_REGISTRY)


def apply_ops(plan: EditPlan, ops: list[dict]) -> OpResult:
    """Apply a list of ops to a copy of the plan, bumping the version once.

    All ops apply to a copy: if any of them fails the original plan is untouched,
    so a partially-applied batch can never be written to disk.
    """
    out = plan.model_copy(deep=True)
    summaries: list[str] = []
    for i, spec in enumerate(ops):
        name = spec.get("op")
        if name not in _REGISTRY:
            raise OpError(f"op {i}: unknown op {name!r}; known: {', '.join(available_ops())}")
        args = {k: v for k, v in spec.items() if k != "op"}
        try:
            summaries.append(_REGISTRY[name](out, args))
        except OpError:
            raise
        except Exception as e:
            raise OpError(f"op {i} ({name}): {e}") from e
    out.version = plan.version + 1
    return OpResult(plan=out, summary="; ".join(summaries))


def _require(args: dict, *keys: str) -> tuple:
    missing = [k for k in keys if k not in args]
    if missing:
        raise OpError(f"missing argument(s): {', '.join(missing)}")
    return tuple(args[k] for k in keys)


def _shot(plan: EditPlan, shot_id: str) -> Shot:
    try:
        return plan.shot(shot_id)
    except KeyError:
        raise OpError(f"no shot {shot_id!r} in the plan") from None


def _next_shot_id(plan: EditPlan) -> str:
    used = {s.id for s in plan.shots}
    i = 1
    while f"s{i:03d}" in used:
        i += 1
    return f"s{i:03d}"


# -- shots -----------------------------------------------------------------------


@op("insert_shot")
def _insert_shot(plan: EditPlan, args: dict) -> str:
    asset, src_in, src_out = _require(args, "asset", "src_in", "src_out")
    shot_id = args.get("id") or _next_shot_id(plan)
    if any(s.id == shot_id for s in plan.shots):
        raise OpError(f"shot id {shot_id!r} already exists")
    after = args.get("after")
    shot = Shot(
        id=shot_id, asset=asset, src_in=float(src_in), src_out=float(src_out),
        timeline_in=int(args.get("timeline_in", 0)),
        section=args.get("section"), intent=args.get("intent", ""),
    )
    if after is not None:
        index = next((i for i, s in enumerate(plan.sorted_shots()) if s.id == after), None)
        if index is None:
            raise OpError(f"no shot {after!r} to insert after")
        ordered = plan.sorted_shots()
        ordered.insert(index + 1, shot)
        plan.shots = ordered
    else:
        plan.shots.append(shot)
    return f"inserted {shot_id} ({asset} {src_in:.2f}-{src_out:.2f}s)"


@op("remove_shot")
def _remove_shot(plan: EditPlan, args: dict) -> str:
    (shot_id,) = _require(args, "id")
    _shot(plan, shot_id)
    plan.shots = [s for s in plan.shots if s.id != shot_id]
    return f"removed {shot_id}"


@op("move_shot")
def _move_shot(plan: EditPlan, args: dict) -> str:
    shot_id, before = _require(args, "id", "before")
    shot = _shot(plan, shot_id)
    ordered = [s for s in plan.sorted_shots() if s.id != shot_id]
    index = next((i for i, s in enumerate(ordered) if s.id == before), None)
    if index is None:
        raise OpError(f"no shot {before!r} to move before")
    ordered.insert(index, shot)
    plan.shots = ordered
    return f"moved {shot_id} before {before}"


@op("trim_shot")
def _trim_shot(plan: EditPlan, args: dict) -> str:
    (shot_id,) = _require(args, "id")
    shot = _shot(plan, shot_id)
    src_in = float(args.get("src_in", shot.src_in))
    src_out = float(args.get("src_out", shot.src_out))
    if src_out <= src_in:
        raise OpError(f"src_out {src_out} must exceed src_in {src_in}")
    before = f"{shot.src_in:.2f}-{shot.src_out:.2f}"
    shot.src_in, shot.src_out = src_in, src_out
    return f"trimmed {shot_id} {before} → {src_in:.2f}-{src_out:.2f}"


@op("replace_source")
def _replace_source(plan: EditPlan, args: dict) -> str:
    shot_id, asset = _require(args, "id", "asset")
    shot = _shot(plan, shot_id)
    old = shot.asset
    shot.asset = asset
    if "src_in" in args:
        shot.src_in = float(args["src_in"])
    if "src_out" in args:
        shot.src_out = float(args["src_out"])
    if shot.src_out <= shot.src_in:
        raise OpError("replacement source range is empty")
    return f"{shot_id} source {old} → {asset}"


# -- shot attributes ---------------------------------------------------------------


@op("set_transition")
def _set_transition(plan: EditPlan, args: dict) -> str:
    (shot_id,) = _require(args, "id")
    shot = _shot(plan, shot_id)
    component = args.get("component")
    if component is None:
        shot.transition_in = None
        return f"{shot_id} transition cleared (hard cut)"
    shot.transition_in = ComponentRef.model_validate(component)
    return f"{shot_id} transition → {shot.transition_in.id}"


@op("set_fx")
def _set_fx(plan: EditPlan, args: dict) -> str:
    shot_id, fx = _require(args, "id", "fx")
    shot = _shot(plan, shot_id)
    shot.fx = [ComponentRef.model_validate(f) for f in fx]
    return f"{shot_id} fx → {len(shot.fx)} component(s)"


@op("set_audio")
def _set_audio(plan: EditPlan, args: dict) -> str:
    shot_id, audio = _require(args, "id", "audio")
    shot = _shot(plan, shot_id)
    shot.audio = ShotAudio.model_validate(audio)
    return f"{shot_id} audio → {shot.audio.mode.value}"


@op("set_speed")
def _set_speed(plan: EditPlan, args: dict) -> str:
    shot_id, speed = _require(args, "id", "speed")
    shot = _shot(plan, shot_id)
    shot.speed = [SpeedRamp.model_validate(r) for r in speed]
    return f"{shot_id} speed → {len(shot.speed)} ramp(s)"


@op("set_reframe")
def _set_reframe(plan: EditPlan, args: dict) -> str:
    shot_id, reframe = _require(args, "id", "reframe")
    shot = _shot(plan, shot_id)
    shot.reframe = Reframe.model_validate(reframe)
    return f"{shot_id} reframe → {shot.reframe.mode}"


@op("set_color")
def _set_color(plan: EditPlan, args: dict) -> str:
    shot_id, color = _require(args, "id", "color")
    shot = _shot(plan, shot_id)
    shot.color = ShotColor.model_validate(color)
    return f"{shot_id} color → normalize={shot.color.normalize} grade={shot.color.grade}"


@op("set_captions")
def _set_captions(plan: EditPlan, args: dict) -> str:
    (shot_id,) = _require(args, "id")
    shot = _shot(plan, shot_id)
    captions = args.get("captions")
    shot.captions = Captions.model_validate(captions) if captions else None
    return f"{shot_id} captions → {shot.captions.id if shot.captions else 'none'}"


@op("set_section")
def _set_section(plan: EditPlan, args: dict) -> str:
    shot_id, section = _require(args, "id", "section")
    shot = _shot(plan, shot_id)
    shot.section = section
    return f"{shot_id} section → {section}"


@op("set_intent")
def _set_intent(plan: EditPlan, args: dict) -> str:
    shot_id, intent = _require(args, "id", "intent")
    _shot(plan, shot_id).intent = intent
    return f"{shot_id} intent set"


# -- overlays, sfx, music, notes -----------------------------------------------------


@op("add_overlay")
def _add_overlay(plan: EditPlan, args: dict) -> str:
    overlay = Overlay.model_validate(args if "component" in args else args.get("overlay", {}))
    if any(o.id == overlay.id for o in plan.overlays):
        raise OpError(f"overlay id {overlay.id!r} already exists")
    plan.overlays.append(overlay)
    return f"added overlay {overlay.id} ({overlay.component})"


@op("update_overlay")
def _update_overlay(plan: EditPlan, args: dict) -> str:
    (overlay_id,) = _require(args, "id")
    overlay = next((o for o in plan.overlays if o.id == overlay_id), None)
    if overlay is None:
        raise OpError(f"no overlay {overlay_id!r}")
    for key in ("from_frame", "to_frame", "component", "props"):
        if key in args:
            setattr(overlay, key, args[key])
    return f"updated overlay {overlay_id}"


@op("remove_overlay")
def _remove_overlay(plan: EditPlan, args: dict) -> str:
    (overlay_id,) = _require(args, "id")
    if not any(o.id == overlay_id for o in plan.overlays):
        raise OpError(f"no overlay {overlay_id!r}")
    plan.overlays = [o for o in plan.overlays if o.id != overlay_id]
    return f"removed overlay {overlay_id}"


@op("add_sfx")
def _add_sfx(plan: EditPlan, args: dict) -> str:
    sfx = Sfx.model_validate(args if "sfx" in args else args.get("entry", {}))
    if any(f.id == sfx.id for f in plan.sfx):
        raise OpError(f"sfx id {sfx.id!r} already exists")
    plan.sfx.append(sfx)
    return f"added sfx {sfx.id} ({sfx.sfx})"


@op("remove_sfx")
def _remove_sfx(plan: EditPlan, args: dict) -> str:
    (sfx_id,) = _require(args, "id")
    if not any(f.id == sfx_id for f in plan.sfx):
        raise OpError(f"no sfx {sfx_id!r}")
    plan.sfx = [f for f in plan.sfx if f.id != sfx_id]
    return f"removed sfx {sfx_id}"


@op("set_music_edits")
def _set_music_edits(plan: EditPlan, args: dict) -> str:
    from montaje.models.editplan import MusicEdit

    (edits,) = _require(args, "edits")
    plan.music.edits = [MusicEdit.model_validate(e) for e in edits]
    return f"music → {len(plan.music.edits)} edit(s)"


@op("set_note")
def _set_note(plan: EditPlan, args: dict) -> str:
    (note,) = _require(args, "note")
    plan.notes = note
    return "notes updated"


def diff_summary(before: EditPlan, after: EditPlan) -> str:
    """Human-readable difference between two plan versions, for the plan_v###.md file."""
    fps = after.format.fps
    lines = [f"v{before.version} → v{after.version}"]
    before_ids = {s.id for s in before.shots}
    after_ids = {s.id for s in after.shots}
    if added := sorted(after_ids - before_ids):
        lines.append(f"- added shots: {', '.join(added)}")
    if removed := sorted(before_ids - after_ids):
        lines.append(f"- removed shots: {', '.join(removed)}")
    for shot_id in sorted(before_ids & after_ids):
        a, b = before.shot(shot_id), after.shot(shot_id)
        changes: list[str] = []
        if (a.src_in, a.src_out) != (b.src_in, b.src_out):
            changes.append(f"range {a.src_in:.2f}-{a.src_out:.2f} → {b.src_in:.2f}-{b.src_out:.2f}")
        if a.timeline_in != b.timeline_in:
            changes.append(f"timeline {a.timeline_in} → {b.timeline_in}")
        if a.asset != b.asset:
            changes.append(f"asset {a.asset} → {b.asset}")
        if _ref_id(a.transition_in) != _ref_id(b.transition_in):
            changes.append(f"transition {_ref_id(a.transition_in)} → {_ref_id(b.transition_in)}")
        if a.audio.mode != b.audio.mode:
            changes.append(f"audio {a.audio.mode.value} → {b.audio.mode.value}")
        if changes:
            lines.append(f"- {shot_id}: " + "; ".join(changes))
    a_dur, b_dur = before.timeline_end_frame() / fps, after.timeline_end_frame() / fps
    if abs(a_dur - b_dur) > 1e-6:
        lines.append(f"- duration {a_dur:.2f}s → {b_dur:.2f}s")
    if len(before.sfx) != len(after.sfx):
        lines.append(f"- sfx {len(before.sfx)} → {len(after.sfx)}")
    if len(before.overlays) != len(after.overlays):
        lines.append(f"- overlays {len(before.overlays)} → {len(after.overlays)}")
    return "\n".join(lines) if len(lines) > 1 else lines[0] + " (no structural change)"


def _ref_id(ref: Any) -> str:
    return ref.id if ref is not None else "none"
