"""Gemini client for semantic clip analysis (§10).

Responsibilities, in order of how much they matter:

1. **Validate and snap every timestamp.** A VLM samples video at a few frames per
   second, so its timings drift; §2.2 makes this the rule the whole system is built on —
   the model picks moments, code picks frames. Out-of-range timestamps are discarded and
   logged rather than clamped, because a moment the model placed outside the clip is
   evidence it was guessing.
2. **Cache by `(asset_id, model, prompt_version)`.** Semantic analysis is the only paid
   per-asset step; re-running it because a prompt was reworded is a real cost.
3. **Upload each proxy once.** The Files API keeps it for the session, so a re-ask about
   the same clip does not re-upload it.
4. **Fail per asset, not per run.** One clip the model refuses must not lose the other
   ninety-nine.

The client never invents structure: it asks for `ClipLog` as a schema and validates the
response against the pydantic model.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from montaje.analysis.semantic import prompts
from montaje.models.asset import Asset
from montaje.models.cliplog import ClipLog, Moment, Quote
from montaje.models.events import Event

log = logging.getLogger(__name__)

# The paid tier is the default: free-tier content may be used to improve the provider's
# products, which is not an acceptable default for someone's personal footage (§10).
API_KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

# A moment shorter than this is not a shot, it is a frame the model happened to like.
MIN_MOMENT_S = 0.25
# How far a model timestamp may be from a deterministic boundary and still snap to it.
SNAP_WINDOW_S = 0.6


class GeminiUnavailable(RuntimeError):
    """No SDK or no API key. Semantic analysis is optional; everything else still runs."""


@dataclass
class SemanticConfig:
    model: str = "gemini-3.8-flash"
    fps_short_clips: float = 3.0
    media_resolution: str = "low"
    max_clip_s: float = 60.0
    api_key: str | None = None


@dataclass
class ClipLogResult:
    asset_id: str
    log: ClipLog | None = None
    error: str | None = None
    discarded: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.log is not None


def api_key(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    for name in API_KEY_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return None


def available(cfg: SemanticConfig | None = None) -> bool:
    """Whether semantic analysis can run at all, without raising."""
    try:
        import google.genai  # noqa: F401
    except ImportError:
        return False
    return api_key(cfg.api_key if cfg else None) is not None


def _client(cfg: SemanticConfig):
    try:
        from google import genai
    except ImportError as e:
        raise GeminiUnavailable("install google-genai to run semantic analysis") from e
    key = api_key(cfg.api_key)
    if not key:
        raise GeminiUnavailable(
            f"set one of {', '.join(API_KEY_VARS)} to run semantic analysis"
        )
    return genai.Client(api_key=key)


# -- context from local analysis -----------------------------------------------------


def build_context(asset: Asset, events: list[Event], transcript: str = "") -> str:
    """The CONTEXT block: what the deterministic analysis already knows (§10)."""
    shots = [(e.t0, e.t1) for e in events if e.analyzer.startswith("shots") and e.type == "shot"]
    usable = [(e.t0, e.t1) for e in events if e.analyzer.startswith("quality") and e.type == "usable"]
    occlusion = [
        (e.type, e.t0, e.t1) for e in events
        if e.analyzer.startswith("occlusion") and e.type in ("reveal", "cover")
    ]
    speech = [(e.t0, e.t1) for e in events if e.analyzer.startswith("vad") and e.type == "speech"]
    audio = [
        (e.type, e.t0, e.t1) for e in events
        if e.analyzer.startswith("audio_events") and e.type != "other"
    ]
    return prompts.format_context(
        duration_s=asset.duration_s,
        transcript=transcript,
        shots=shots,
        usable=usable,
        occlusion=occlusion,
        audio=audio,
        speech=speech,
    )


def snap_targets(events: list[Event]) -> list[float]:
    """Deterministic instants a model timestamp may snap to."""
    targets: set[float] = set()
    for event in events:
        name = event.analyzer.split("@")[0]
        if name == "shots" and event.type == "shot":
            targets.update((event.t0, event.t1))
        elif name == "occlusion" and event.type == "reveal":
            targets.add(event.t1)
        elif name == "occlusion" and event.type == "cover":
            targets.add(event.t0)
        elif name == "vad" and event.type == "speech":
            targets.update((event.t0, event.t1))
        elif name == "asr" and event.type == "word":
            targets.update((event.t0, event.t1))
    return sorted(targets)


def _snap(value: float, targets: list[float], window: float = SNAP_WINDOW_S) -> float:
    if not targets:
        return value
    nearest = min(targets, key=lambda t: abs(t - value))
    return nearest if abs(nearest - value) <= window else value


# -- validation -----------------------------------------------------------------------


def validate_log(
    log: ClipLog, duration_s: float, targets: list[float] | None = None
) -> tuple[ClipLog, list[str]]:
    """Drop impossible timestamps and snap plausible ones. Returns `(log, discarded)`.

    Out-of-range entries are **discarded, not clamped**: a moment the model placed past
    the end of the clip tells you it was guessing, and clamping it to the end would turn
    that guess into a shot the editor is asked to trust.
    """
    targets = targets or []
    discarded: list[str] = []
    out = log.model_copy(deep=True)

    moments: list[Moment] = []
    for moment in out.moments:
        problem = _range_problem(moment.t0, moment.t1, duration_s)
        if problem:
            discarded.append(f"moment {moment.label!r} {moment.t0:.2f}–{moment.t1:.2f}: {problem}")
            continue
        moment.t0 = round(_snap(moment.t0, targets), 3)
        moment.t1 = round(_snap(moment.t1, targets), 3)
        if moment.t1 - moment.t0 < MIN_MOMENT_S:
            # Snapping can collapse a short moment; extend it rather than drop it, since
            # the model did identify something here.
            moment.t1 = round(min(duration_s, moment.t0 + MIN_MOMENT_S), 3)
        if moment.t1 <= moment.t0:
            discarded.append(f"moment {moment.label!r}: collapsed after snapping")
            continue
        moment.score = float(min(1.0, max(0.0, moment.score)))
        moments.append(moment)
    out.moments = sorted(moments, key=lambda m: m.t0)

    quotes: list[Quote] = []
    for quote in out.quotes:
        problem = _range_problem(quote.t0, quote.t1, duration_s)
        if problem:
            discarded.append(f"quote {quote.text[:40]!r}: {problem}")
            continue
        quote.t0 = round(_snap(quote.t0, targets), 3)
        quote.t1 = round(_snap(quote.t1, targets), 3)
        if quote.t1 <= quote.t0:
            discarded.append(f"quote {quote.text[:40]!r}: collapsed after snapping")
            continue
        quotes.append(quote)
    out.quotes = sorted(quotes, key=lambda q: q.t0)

    out.energy = int(min(5, max(1, out.energy)))
    out.aesthetic = int(min(5, max(1, out.aesthetic)))
    return out, discarded


def _range_problem(t0: float, t1: float, duration_s: float) -> str | None:
    if t0 < -0.05:
        return "starts before the clip"
    if t0 > duration_s + 0.05:
        return f"starts past the end ({duration_s:.2f}s)"
    if t1 > duration_s + 0.5:
        return f"ends past the end ({duration_s:.2f}s)"
    if t1 <= t0:
        return "ends before it starts"
    return None


# -- the call --------------------------------------------------------------------------


def analyze_clip(
    asset: Asset,
    proxy: Path,
    events: list[Event],
    cfg: SemanticConfig,
    *,
    transcript: str = "",
    uploaded: dict[str, object] | None = None,
) -> ClipLogResult:
    """Ask Gemini for one clip's `ClipLog`, validated and snapped."""
    if not proxy.exists():
        return ClipLogResult(asset_id=asset.asset_id, error=f"no proxy at {proxy}")

    try:
        client = _client(cfg)
    except GeminiUnavailable as e:
        return ClipLogResult(asset_id=asset.asset_id, error=str(e))

    from google.genai import types

    try:
        cache = uploaded if uploaded is not None else {}
        handle = cache.get(asset.asset_id)
        if handle is None:
            handle = client.files.upload(file=str(proxy))
            cache[asset.asset_id] = handle

        context = build_context(asset, events, transcript)
        response = client.models.generate_content(
            model=cfg.model,
            contents=[handle, prompts.clip_log_prompt(context)],
            config=types.GenerateContentConfig(
                system_instruction=prompts.SYSTEM,
                response_mime_type="application/json",
                response_schema=ClipLog,
                # Low media resolution and a few fps are enough to describe a clip, and
                # this is the only per-asset paid step in the pipeline (§26).
                media_resolution=_media_resolution(cfg.media_resolution, types),
            ),
        )
    except Exception as e:  # one refused clip must not lose the other ninety-nine
        return ClipLogResult(asset_id=asset.asset_id, error=f"{type(e).__name__}: {e}")

    log_obj = _parse(response, asset.asset_id, cfg)
    if isinstance(log_obj, str):
        return ClipLogResult(asset_id=asset.asset_id, error=log_obj)

    validated, discarded = validate_log(log_obj, asset.duration_s, snap_targets(events))
    for problem in discarded:
        log.info("discarded from %s: %s", asset.asset_id, problem)
    return ClipLogResult(asset_id=asset.asset_id, log=validated, discarded=discarded)


def _media_resolution(name: str, types):
    """Map the config's resolution name onto the SDK enum, tolerating SDK changes."""
    mapping = {
        "low": "MEDIA_RESOLUTION_LOW",
        "medium": "MEDIA_RESOLUTION_MEDIUM",
        "high": "MEDIA_RESOLUTION_HIGH",
    }
    enum = getattr(types, "MediaResolution", None)
    if enum is None:
        return None
    return getattr(enum, mapping.get(name, "MEDIA_RESOLUTION_LOW"), None)


def _parse(response, asset_id: str, cfg: SemanticConfig) -> ClipLog | str:
    """Pull a `ClipLog` out of a response, preferring the SDK's parsed object."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, ClipLog):
        parsed.asset_id = asset_id
        parsed.model = cfg.model
        parsed.prompt_version = prompts.PROMPT_VERSION
        return parsed
    text = getattr(response, "text", None)
    if not text:
        return "empty response"
    try:
        log_obj = ClipLog.model_validate_json(text)
    except Exception as e:
        return f"unparseable response: {e}"
    log_obj.asset_id = asset_id
    log_obj.model = cfg.model
    log_obj.prompt_version = prompts.PROMPT_VERSION
    return log_obj
