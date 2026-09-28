"""Prompts for semantic clip analysis (§10).

Two things make these prompts work, and both are about *grounding*:

1. **The local analysis goes in the prompt.** The model is told where the transcript,
   the occlusion spans, the quality spans and the audio events are. Without that it
   guesses at timestamps from a 2–4 fps sample and the numbers drift by seconds (§2.2).
   With it, the model is picking among instants the deterministic analysis already
   found, which is a much easier task.
2. **Nothing project-specific.** No prompt mentions hands over lenses, festivals or
   any other convention: conventions are *discovered* from patterns across the footage
   and confirmed by the user (§2.5, §12). A prompt that names one would find it in
   footage that does not contain it.

`PROMPT_VERSION` is part of the cache key, so editing any prompt here invalidates every
cached clip log rather than mixing outputs from two different questions.
"""

from __future__ import annotations

PROMPT_VERSION = 1

SYSTEM = """\
You are a film editor's assistant logging raw footage. You watch one clip and describe \
what is in it, factually and usefully, for an editor who has not seen it.

Rules:
- Report only what is visible or audible. Do not speculate about what happened off camera.
- Timestamps are in seconds from the clip's start. Prefer the instants listed in the \
CONTEXT block: those come from frame-accurate analysis, while your own view of the clip \
is sampled at a few frames per second and your timing will drift.
- A "moment" is something an editor would cut to. Be selective: three excellent moments \
are more useful than ten mediocre ones.
- `score` is how strongly you would recommend using the moment, 0 to 1.
- Note problems honestly in `issues`. An editor needs to know a shot is unusable.
- If the clip contains nothing worth cutting to, return an empty `moments` list. Saying \
so is more useful than inventing something.
"""

CLIP_LOG = """\
Log this clip.

CONTEXT (from frame-accurate local analysis — trust these timings over your own):
{context}

Describe:
- `summary`: one or two sentences on what happens.
- `setting`, `time_of_day`: where and when, as far as it is visible.
- `shot_type`: selfie | front | back | pov.
- `people`: how many, whether a recurring group appears to be present (`us_present`), \
whether there is a crowd.
- `energy` and `aesthetic`: 1–5. Energy is how much is happening; aesthetic is how good \
it looks.
- `start_description` and `end_description`: what the first and last moments look like. \
Be concrete about framing, motion and any object entering or leaving frame — these are \
used to find recurring patterns across clips.
- `notable_gestures`: specific physical actions, described plainly.
- `moments`: the instants worth cutting to, each with `t0`, `t1`, `label`, `why`, \
`score`, and `kind` (highlight | reaction | quote | scenic | transition_candidate | hero).
- `quotes`: spoken lines worth keeping, with `usable` false if unclear or clipped.
- `stage_music`: whether performed or amplified music is audible, and what it is.
- `issues`: anything that limits how the clip can be used.
- `tags`: short keywords for search.
"""


def format_context(
    duration_s: float,
    transcript: str = "",
    shots: list[tuple[float, float]] | None = None,
    usable: list[tuple[float, float]] | None = None,
    occlusion: list[tuple[str, float, float]] | None = None,
    audio: list[tuple[str, float, float]] | None = None,
    speech: list[tuple[float, float]] | None = None,
) -> str:
    """Render the local analysis into the CONTEXT block.

    Spans are collapsed to at most a handful of entries each: a 200-line context costs
    tokens and buries the useful timings among the routine ones.
    """
    lines = [f"- duration: {duration_s:.2f}s"]
    if shots:
        lines.append("- shot boundaries: " + _spans(shots, limit=12))
    if usable:
        lines.append("- technically usable spans: " + _spans(usable, limit=8))
    if occlusion:
        lines.append(
            "- dark/featureless spans and their edges (no meaning assigned): "
            + ", ".join(f"{kind} {t0:.2f}–{t1:.2f}" for kind, t0, t1 in occlusion[:8])
        )
    if speech:
        lines.append("- speech detected: " + _spans(speech, limit=8))
    if audio:
        lines.append(
            "- audio content per window: "
            + ", ".join(f"{kind} {t0:.1f}–{t1:.1f}" for kind, t0, t1 in _merge(audio)[:10])
        )
    if transcript:
        lines.append(f'- transcript: "{transcript.strip()[:800]}"')
    else:
        lines.append("- transcript: (no speech transcribed)")
    return "\n".join(lines)


def _spans(spans: list[tuple[float, float]], limit: int) -> str:
    shown = spans[:limit]
    text = ", ".join(f"{t0:.2f}–{t1:.2f}" for t0, t1 in shown)
    if len(spans) > limit:
        text += f", … ({len(spans) - limit} more)"
    return text


def _merge(labelled: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    """Collapse consecutive windows with the same label into one span."""
    out: list[tuple[str, float, float]] = []
    for kind, t0, t1 in labelled:
        if out and out[-1][0] == kind and abs(out[-1][2] - t0) < 0.01:
            out[-1] = (kind, out[-1][1], t1)
        else:
            out.append((kind, t0, t1))
    return out


def clip_log_prompt(context: str) -> str:
    return CLIP_LOG.format(context=context)
