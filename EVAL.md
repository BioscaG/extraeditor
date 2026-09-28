# Evaluation

Human evaluation of finished edits (§27). One table row per plan version, so progress
is visible across milestones rather than asserted.

Score each category 1–5. The last column is the one that matters: **would I post this?**

| # | meaning |
|---|---|
| 1 | Broken — I would not show this to anyone |
| 2 | Recognisably amateur |
| 3 | Watchable; a friend would say "nice" |
| 4 | Clearly good; comparable to a skilled creator's template edit |
| 5 | Indistinguishable from a professional editor's work |

## Categories

- **Hook** — do the first 1–2 seconds make you keep watching?
- **Story** — is there a shape, or is it a pile of clips in a row?
- **Pacing** — does the cut rate serve the music and vary on purpose?
- **Cut quality** — cuts on action, matched motion, no mid-word or mid-gesture cuts.
- **Transitions** — designed transitions used as punctuation, not as grammar.
- **Sound design** — SFX placed, sized and spaced so they read as intentional.
- **Color** — do shots from different phones look like one film?
- **Text** — legible, safely placed, typographically deliberate.
- **Overall** — would I post this?

## Method

1. Watch the whole thing once, on a phone, at the intended aspect ratio, with sound.
2. Score without looking at the rhythm report — score the *result*, not the metrics.
3. Then read `plans/rhythm_v###.md` and note where the numbers disagree with your eyes.
   A disagreement is a finding about the *report*, and worth recording.
4. Record the three most damaging problems, most damaging first. Those become the next
   milestone's work.

Scoring the machine-checkable categories by eye is deliberate: the rails already
guarantee that cuts land on beats and loudness is correct. What the rails cannot tell
you is whether the result is worth watching.

## Results

| date | plan | project | hook | story | pacing | cut | trans | sound | color | text | overall | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-28 | v003 | e2e (synthetic) | — | — | — | — | — | — | — | — | — | Pipeline verification only, not an aesthetic baseline: the footage is ffmpeg test patterns, so every visual category is meaningless. What it does establish is that the machinery runs end to end — 51 shots, 43.2s, −15.5 LUFS, three stems, captions from real ASR word timings, two section transitions, OTIO/FCPXML matching the plan's duration exactly. |

**The M1 baseline row is still to be filled** with the real festival footage. Until an
edit of real material is scored here, M1's acceptance criterion ("an aftermovie you would
actually share") is unmet no matter what the tests say.

## Known limitations at the time of writing

These are things the evaluator should not be surprised by, because they are known rather
than discovered:

- **No semantic analysis yet.** Without Gemini clip logs (§10) the planner ranks footage
  on sharpness, steadiness and motion only. It cannot tell a hero moment from a shot of
  the floor, so *selection* — the thing that most separates a good edit from a bad one —
  is effectively random within the usable material. Expect story and hook to score low.
- **No reframing.** Shots are letterboxed to fit rather than reframed onto the subject,
  because subject tracking (§9 `subjects`) is not implemented. On 9:16 output from 16:9
  footage this is very visible.
- **Shot selection ignores duplicates.** `dedupe` is not implemented, so the only
  protection against near-identical adjacent shots is avoiding the same source clip twice
  in a row.
- **One transition per section change, chosen round-robin.** Motion matching is recorded
  in the component metadata but the baseline planner does not use it yet.
- **The critic loop (§20) is not implemented.** Nothing watches the preview.
