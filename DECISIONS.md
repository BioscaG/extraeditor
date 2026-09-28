# Decisions

Append-only log. Each entry: date, decision, alternatives considered, evidence.

---

## 2026-09-28 — Repository is the `montaje` package itself

**Decision:** the repo root *is* the project (`pyproject.toml` at root, code in `src/montaje/`),
rather than nesting a `montaje/` directory as drawn in README §5.

**Alternatives:** literal nested `montaje/` folder inside the repo.

**Evidence:** the repo is already named `extraeditor` and exists only to hold this tool; a nested
folder would add a redundant path segment to every import and command. Every other path in §5 is
kept verbatim (`src/montaje/...`, `library/`, `render/remotion/`, `projects/`, `tests/`).

---

## 2026-09-28 — Python 3.12 via `uv`, managed venv

**Decision:** pin `requires-python = ">=3.12"` and use a `uv`-managed `.venv`.

**Evidence:** system Python on the target machine is 3.9.6, which lacks `StrEnum` and PEP 604/695
syntax used throughout the models. `uv python install 3.12` provisions the interpreter without
touching the system one.

---

## 2026-09-28 — M1 analyzers: deterministic ffmpeg+numpy first, models behind the same interface

**Decision:** ship `shots`, `quality`, `occlusion`, `motion`, `audio_events`, `vad`, `loudness`,
`color_stats` as pure ffmpeg + numpy implementations for the vertical slice. TransNetV2 (shots),
CLAP (audio tagging), Silero (VAD) and mlx-whisper (ASR) plug in behind the identical
`Analyzer` protocol later.

**Alternatives:** install the full model stack before the first watchable edit.

**Evidence:** README §0 mandates a vertical slice before hardening, and §2.7 requires analyzers to be
pure, versioned and cached. The analyzer interface (`name`, `version`, `params()`, `run()`) plus the
`(asset_id, analyzer, version, params_hash)` cache key means swapping the implementation only bumps
`version`, invalidating that analyzer's cache and nothing else. Avoids blocking M1 on
`allin1`/`madmom`-class Apple Silicon install problems (§28).

---

## 2026-09-28 — ffmpeg is discovered at runtime, not assumed

**Decision:** `montaje.ffmpeg` resolves the ffmpeg/ffprobe binaries and probes their filter list
once per process, preferring a keg-only `ffmpeg-full` install over whatever is on `PATH`. Every
subprocess call in the codebase goes through it. `MONTAJE_FFMPEG` / `MONTAJE_FFPROBE` override.

**Alternatives:** hardcode `"ffmpeg"`; require a specific build in the README and fail otherwise.

**Evidence:** Homebrew's default `ffmpeg` bottle on this machine (9.0.2) ships **without libzimg
and libplacebo**, so `zscale` — which §4 lists as the tone-mapping filter — does not exist and
every HDR proxy failed with "No such filter: 'zscale'". `brew install ffmpeg-full` provides zimg,
libplacebo, libvmaf and whisper.cpp and is **keg-only**, so it installs at
`/opt/homebrew/opt/ffmpeg-full/bin/` without replacing the existing ffmpeg. Discovery rather than a
hard requirement keeps the tool working on machines with only the basic build, at reduced quality.

---

## 2026-09-28 — Tone-mapping: capability-ranked methods, `libplacebo` preferred

**Decision:** tone-mapping is a ranked list of methods filtered by what the discovered ffmpeg
supports: `libplacebo_bt709` → `zscale_hable` → `zscale_mobius` → `videotoolbox` → `naive_clip`.
`hdr.method` in config names a *preference*; `resolve_method()` degrades to the best available and
the resolved name goes into the proxy cache key. All chains downscale before tone-mapping.

**Alternatives:** a single hardcoded chain (fails on builds without zimg); AVFoundation via a Swift
helper (README's listed fallback — deferred, it would block M0 on writing Swift).

**Evidence:** only `zscale` and `libplacebo` can linearize HLG/PQ in ffmpeg; the `colorspace` filter
has no `arib-std-b67` or `smpte2084` value, so it cannot substitute. `libplacebo` is ranked first
because it does linearization, BT.2390 tone-mapping and gamut mapping in one pass. `scale_vt`
(VideoToolbox) was verified working on this machine and is kept as a fast hardware path, but its
curve is not tunable. Putting the *resolved* method in the cache key means proxies built with a
fallback are correctly invalidated if the machine later gains a better filter.
**Status:** ranking is a priori. `montaje debug tonemap` renders every available method side by
side; revisit this entry with stills from real festival HDR clips.

---

## 2026-09-28 — Thread-local SQLite connections

**Decision:** `Store` lazily creates one connection per thread, with WAL journaling and a 30 s busy
timeout.

**Evidence:** ingest and analysis fan out over a `ThreadPoolExecutor` (§8, §25) and sqlite3 refuses
to use a connection from a thread other than its creator — this failed 15 tests before the fix. WAL
lets the reader threads proceed while one writes.

---

## 2026-09-28 — Speech is detected by syllabic modulation, not spectral tonality

**Decision:** `audio_events` and `vad` classify speech on *envelope modulation* (std/mean of the
10 ms envelope) combined with the 300–3400 Hz energy share. Spectral flatness is used only to find
music, never to reject speech.

**Evidence:** measured on the `say`-over-tone fixture, real speech windows sit at spectral flatness
0.33–0.58 — a "speech is tonal" rule (`flatness < 0.25`) rejected all of them and labelled them
`crowd`. Modulation cleanly separates them: speech 0.89–1.27, steady crowd/music noise well below
0.7. `tests/unit/test_analyzers.py` and `test_vad.py` pin this behaviour, including the §9
requirement that VAD stays quiet on music-only clips.

---

## 2026-09-28 — Color fixtures use `colortemperature`, not `colorbalance`

**Decision:** the warm/cool normalization fixture pair is generated with
`colortemperature=temperature=2800|9000`.

**Evidence:** `colorbalance=rs=0.25:bs=-0.2` moved Lab b* by only 0.5 between the warm and cool
copies, which `color_stats` cannot be expected to separate — the fixture, not the analyzer, was
wrong. `colortemperature` applies a white-point shift across the whole tonal range and produces a
realistic, clearly measurable difference.

---

## 2026-09-28 — Beat grid is least-squares fitted, not just phase-aligned

**Decision:** `track_beats` estimates a coarse tempo and phase from the onset envelope, then
`refine_grid` fits **both** phase and period by linear regression against measured attack times,
with outlier rejection and a guard against collapsing onto another metrical level. The grid stays
rigid (one phase, one period) — beats are never individually snapped.

**Alternatives:** phase-only alignment with a global offset correction; per-beat snapping to the
nearest onset.

**Evidence:** measured on the fixtures, in three stages:

| stage | click 120 | click 90 | 96 s arrangement |
|---|---|---|---|
| phase search only | 13.8 ms | 15.2 ms | — |
| + global offset correction | 0.30 ms | 2.4 ms | 29.7 ms |
| + period fitted (current) | 1.0 ms | 1.4 ms | **3.1 ms** over 192 beats |

A global offset cannot fix a tempo error: 120.094 BPM against a true 120 drifts 75 ms over 96
seconds, and the median correction just splits that error either side of the middle. Per-beat
snapping would fix the error but introduce jitter, and a jittery grid makes cuts feel loose rather
than tight — the opposite of the goal.

---

## 2026-09-28 — Three different envelopes, one per measurement, because each has a different bias

**Decision:** the music module computes three envelopes and uses each only where its bias does not
matter: **spectral flux** (5 ms) for tempo and coarse phase, **amplitude** (1 ms) for beat strength
and energy, **amplitude rise** (1 ms) for locating attacks.

**Evidence:** each has a measurable, opposite timing bias. Spectral flux peaks ~14 ms *before* the
transient (its analysis window's group delay: the flux rises as a transient enters the window).
Raw amplitude peaks *after* it, by about a quarter cycle of the carrier — ~4 ms for a 60 Hz kick.
Only the rise of the amplitude envelope is unbiased, which is what makes millisecond grid fitting
possible. This is not academic: using the flux for downbeat detection put the downbeats one beat
off, because sampling it at a now-accurate beat time lands past its peak and reports every beat as
equally weak.

---

## 2026-09-28 — Section boundaries: median + MAD novelty threshold

**Decision:** `analyze_structure` accepts a phrase boundary as a section change only when its
checkerboard novelty exceeds `median + 4 × 1.4826 × MAD` of all candidates.

**Alternatives:** take the top-N candidates greedily; threshold at mean + k·sigma.

**Evidence:** on the 4-section fixture, taking candidates greedily in rank order produced 6
sections, splitting homogeneous material at arbitrary phrase boundaries. Thresholding at mean+sigma
failed the other way and produced 2: novelty spans orders of magnitude (the true boundaries scored
0.2134, 0.0279 and 0.0042 against ≤0.0009 for every non-boundary), so the single strongest
boundary inflates the standard deviation enough to hide the weaker real ones. MAD ignores exactly
those outliers and recovers all three boundaries to within 0.1 s, with the correct bar counts.

---

## 2026-09-28 — Fit-to-duration operates on whole phrases, longest grid first

**Decision:** `fit_to_duration` tries 16-, then 8-, then 4-bar phrase grids and returns the first
that lands within tolerance. Shortening removes whole phrases ranked by how safe they are to drop
(low energy, mid-track, never the drop or the edge phrases); lengthening repeats the
highest-energy non-edge phrases. Joins are downbeat crossfades, validated by `validate_joins`.

**Evidence:** §14.2 requires whole-phrase edits on downbeats. Longest-grid-first because removing
one 16-bar phrase is less audible than removing four 4-bar ones. `phrase_spans` merges fragments
shorter than half a phrase into their neighbour: the first downbeat is almost never at exactly 0.0,
and naively prepending it produced a 7 ms "phrase" that rounded to zero timeline frames and
silently corrupted the edit's frame positions.

---

## 2026-09-28 — Component metadata is read from TypeScript, not duplicated in Python

**Decision:** `library.registry` bundles a tiny TS script with esbuild and runs it under
Node to get the real `meta` objects out of the TSX components. A source-parsing fallback
covers machines without Node.

**Alternatives:** maintain a parallel Python declaration per component; generate Python
from TS at build time.

**Evidence:** `meta` lives next to the code it describes, so it cannot drift from the
component — a duplicate in Python would. The fallback demonstrates why the Node path
matters: parsing the source reads `duration.minFrames` as the literal default (1–300)
instead of resolving `timing.sweep.minFrames` (6–20), so the validator would accept
transition durations the component cannot render.

---

## 2026-09-28 — The baseline SFX library is synthesized, not downloaded

**Decision:** `library/sfx/make_sfx.py` generates 13 SFX procedurally (whoosh, impact,
riser, downlifter, tick, pop, sub drop, crowd bed, tape stop) and writes `sfx.yaml` with
`license: CC0-1.0` and a measured `peak_offset_s` per file.

**Alternatives:** ship curated CC0 packs; fetch at install time.

**Evidence:** §14.1 forbids adding a file without a known license, and §18.2 makes an
unlicensed SFX a hard render error. Synthesizing means the code *is* the provenance, so
the license claim is auditable and nothing can vanish. Curated packs can be added
alongside with their own per-file terms. The peak offset matters because the mixer
aligns the *hit* to the anchor frame, not the file start.

---

## 2026-09-28 — Order of operations: rails, then spotting and speech repair

**Decision:** the render pipeline runs rails → speech-cut repair → SFX spotting →
validate → conform → mix → compose → mux. `montaje plan` also spots, so a plan on disk
is complete, but the render re-spots after the rails.

**Evidence:** every step after the rails depends on the final timeline, and getting this
order wrong produced four distinct hard-error classes on the first real run:

| symptom | cause |
|---|---|
| speech overlapping speech | a J-cut on every speaking shot reached into the previous line |
| `src_out` past end of file | word post-roll had nothing clamping it to the asset duration |
| 3-frame gaps (black frames) | a forward beat snap opened a gap nothing closed |
| SFX anchored past the timeline end | the rails moved shots but not section boundaries |

The rails now clamp to asset durations, close snap gaps by extending the previous shot
into its handles, and relayout `concept.sections` along with the shots. Sections are
timeline positions, so the rails own them too.

---

## 2026-09-28 — The builder's minimum shot length must not undercut the rails'

**Decision:** `plan.build.ABSOLUTE_MIN_SHOT_S` is 0.35 s, above `rails.min_shot_s` (0.3).

**Evidence:** it was 0.25 s. The drop section's 0.5-beat target at 120 BPM, times the
0.5 variation multiplier, produced 0.125 s shots which the builder clamped to 0.25 s and
the rails then deleted for being under 0.3 s. The edit silently lost 9 of its 40 seconds
and the duration check failed with no indication why.

---

## 2026-09-28 — Duration comparisons are made at frame resolution

**Decision:** `music.edit.fits_within` compares durations as rounded frame counts.

**Evidence:** phrase lengths derive from a *measured* beat period, so they are never
round: two nominally 8-second phrases sum to 16.0016 s and fail a literal `<= 16.0`.
That 1.6 ms decided three separate behaviours wrongly — edge protection did not engage,
so short edits lost their intro and outro; and the lengthening loop added one phrase too
many, overshooting a 120 s target to 128 s. A sub-frame difference is below the
timeline's own resolution and must never change a decision.

---

## 2026-09-28 — Fit-to-duration is chosen by structure preserved, not by first fit

**Decision:** `fit_to_duration` evaluates all three phrase grids and picks by
`(drop present, roles preserved, phrase length, closeness to target)`.

**Alternatives:** return the first grid within tolerance (the original), which meant
longest-phrases-first always won.

**Evidence:** returning the first acceptable grid gave a 30-second target a 16-bar grid
holding only the drop. Coarser phrases only buy less audible joins; keeping the track's
shape matters more. The drop is ranked above the role *count* because a short edit that
keeps an intro and an outro but loses the drop has kept the packaging and thrown away
the contents. Measured on the fixture: every target from 30 s up now keeps
intro/drop/outro, and 40/80/120 s land exactly on target.

---

## 2026-09-28 — MCP server on the 2.x SDK; tools are plain functions

**Decision:** tool logic lives in `director/tools.py` with no MCP import; `mcp_server.py`
is thin wrappers around it. Errors are returned as `{"error": ...}` data, never raised.

**Evidence:** the same functions back both the MCP server and the future built-in loop
(§16.1), so the two cannot drift, and the tools are directly unit-testable. The MCP 2.x
SDK renamed `FastMCP` to `MCPServer`. Errors are data because a tool that raises ends the
agent's turn instead of letting it recover — which is the difference between the agent
fixing a typo'd asset id and the session dying.

---

## 2026-09-28 — Remotion resolves intermediates via `staticFile` and `--public-dir`

**Decision:** resolved shots carry an intermediate *filename*; the composition wraps it
in `staticFile()` and the render is invoked with `--public-dir` pointing at the project's
intermediates directory.

**Evidence:** Remotion serves assets over HTTP from its bundle, so an absolute
filesystem path is looked up *inside* the bundle and 404s — every frame failed. Copying
intermediates into `public/` would duplicate ProRes files; `--public-dir` points at them
where they already are. Also pinned `zod` to the exact version Remotion requires, since a
mismatch produces "unclear errors" by its own warning's admission.

---

## 2026-09-28 — ASR runs only on VAD speech, and runs serially

**Decision:** `AsrAnalyzer` transcribes only the ranges `vad` marked as speech, discards
segments the model itself is unsure about, and is listed in `SERIAL_ANALYZERS` so it never
runs from the thread pool.

**Evidence:** two separate failure modes, both severe.

Gating on VAD is what §28 warns about: Whisper invents fluent sentences over music and
crowd noise, and those inventions become on-screen captions. Transcribing only inside VAD
segments and rejecting low-confidence output means a wrong caption needs two independent
failures.

The serial requirement was found the hard way: run from a `ThreadPoolExecutor`, MLX
inference **terminates the interpreter without raising**. The command exited 0, printed
nothing, wrote no events, and the only trace was a leaked-semaphore warning. Running the
same analyzer directly worked perfectly, which is what made it diagnosable.

---

## 2026-09-28 — Captions are gated on words in the *chosen* range, then repaired

**Decision:** the builder only attaches captions where transcribed words fall inside the
shot's own `[src_in, src_out)`; `drop_empty_captions` removes any that the rails then
shifted off their words.

**Evidence:** gating on the *candidate span* instead captioned 43 of 51 shots when only
18 had anything to show — a clip with one spoken line produces usable spans covering the
whole clip, so every sub-range inherited the speech flag. A caption component with no
words renders nothing, so the plan was claiming something it could not deliver. The
post-rails repair is needed because snapping and gap-closing move a short shot off the
words it was chosen for.

---

## 2026-09-28 — A rebuilt plan takes the next version number

**Decision:** `montaje plan` and `build_baseline_plan` write `next_plan_version()`, not v1.

**Evidence:** `montaje plan` always wrote `plan_v001.json`, but `latest_plan()` picks the
highest version on disk. A `plan_v002` left by an earlier session therefore shadowed every
subsequent rebuild — the render used a plan nobody had asked for, and the missing
transitions it produced looked like a bug in the transition code rather than in version
selection.

---

## 2026-09-28 — OTIO is built through the library, and transitions map to `Transition`

**Decision:** `build_otio` constructs `otio.schema` objects rather than writing OTIO's
JSON by hand. A plan's `transition_in` becomes an `otio.schema.Transition`; a genuine
timeline overlap trims the outgoing clip instead.

**Alternatives:** hand-written JSON (no dependency), which is what was tried first.

**Evidence:** the hand-written form failed OTIO's own reader with a bare
`KeyError: media_references` — `Clip.2` needs the plural field, and OTIO's writer cannot
produce a file its reader rejects. On transitions: montaje's plan represents a designed
transition as an *annotation on a butt cut*, with the renderer deriving the overlap, and
OTIO's Transition has exactly that model (it borrows from its neighbours without changing
their durations), so the two map across directly. Appending transition-overlapped clips
back to back instead inflated a 42.1 s edit to 45.5 s in the export.

---

## 2026-09-28 — Subtitles collapse overlapping duplicates, but not genuine repeats

**Decision:** `cues_from_plan` merges consecutive cues with near-identical text **only**
when they overlap in time.

**Evidence:** a reused source range reuses its audio, so the same line can legitimately
appear twice in the edit — and the viewer hears it twice, so the subtitle must show it
twice. What is *not* legitimate is the same line appearing as two overlapping cues, which
happens because cue padding extends one cue into the next shot's copy of it. Collapsing
all repeats was the first attempt and would have dropped real dialogue.

---

## 2026-09-28 — Reframing: crop onto the subject, held still per shot

**Decision:** the conform crops each shot to the output aspect ratio, positioned on the
subject the `subjects` analyzer found, falling back to a centre crop when nothing is
confident. The crop is computed once per shot and does not move within it.

**Alternatives:** letterbox (what the conform did before); centre crop; a crop that
tracks the subject frame by frame.

**Evidence:** cropping 16:9 into 9:16 discards two thirds of the picture, so *where* the
window sits is the most consequential single decision in a vertical edit. Letterboxing
wastes half the screen; a centre crop puts the subject out of frame whenever they are not
dead centre, which on handheld footage is most of the time. Measured on 16:9 clips with a
subject at x=0.20 and x=0.78, the crop moves to x=118 and x=1194 against a centre of 656 —
in both cases a centre crop would have missed the subject entirely.

Holding the crop still within a shot is deliberate: a crop that follows the subject reads
as a security camera, while a locked frame reads as composition. Moving reframes are a
component-level effect, not a conform-level one.

The clamp on how far the crop may travel **limits** the extreme cases rather than scaling
every offset. Scaling was the first implementation and it weakened every reframe in order
to restrain the few that needed it.

---

## 2026-09-28 — Subject detection: detail + skin + relative motion, not spectral saliency

**Decision:** interest is `1.0 × detail + 2.5 × skin + 1.5 × motion`, thresholded at 45%
of the map's peak before taking a centroid. Confidence is the weighted **spatial spread**
of what survives.

**Alternatives:** spectral-residual saliency (Hou & Zhang), which was implemented first.

**Evidence:** three separate findings, each from a measurement.

*Spectral residual was rejected.* It assumes natural image statistics; on the
flat-background footage available here it locked onto the FFT ringing of a rectangle
rather than the rectangle. On a plate with a subject at x=0.85 it reported x=0.50 — no
signal at all. A method whose domain assumption cannot be verified is not one to build the
most visible decision in the edit on. The three replacement signals are individually
checkable: skin tone beats a brighter white distraction, and relative motion picks a
moving patch over a static one of equal brightness.

*The map must be thresholded.* Used raw as a mass distribution, the background contributes
at its baseline level across thousands of cells and outweighs the subject's few — a
2300-cell background at 0.1 against a 24-cell subject at 1.0. Every centroid landed within
0.02 of the frame centre, which looks exactly like reframing working while doing nothing.

*Confidence must measure spread, not area.* Counting surviving cells rated pure random
noise at 0.97 — higher than a clean shot of an actual subject — because grain leaves few
cells above the cut but scatters them across the whole frame. Weighted spread rates the
same noise at 0.00, a subject filling half the frame at 0.02, and a real subject at 0.86.

---

## 2026-09-28 — The builder applies the style's reframe policy

**Decision:** generated shots take `reframe` from the style, not from the model default.

**Evidence:** `modern-festival` specifies `policy: auto_subject`, but every generated shot
carried the model's `center` default, so subject detection ran, produced correct centres,
and was then ignored. The render looked plausible — the frame was filled edge to edge —
which is what made it hard to notice: the failure mode of a mis-wired reframe is not a
crash but a subtly worse composition.

---

## 2026-09-28 — Semantic timestamps are validated and snapped, never clamped

**Decision:** `validate_log` snaps a model timestamp to the nearest deterministic boundary
within 0.6 s (shot edges, occlusion edges, speech edges, word edges), and **discards**
anything outside the clip rather than clamping it into range.

**Evidence:** §2.2 is the rule the whole system rests on — the model picks moments, code
picks frames — because a VLM samples video at a few frames per second and its timings
drift. Discarding rather than clamping is the important half: a moment the model placed
past the end of the clip is evidence it was guessing, and clamping that guess to the clip
end converts it into a shot the editor is asked to trust. Sub-frame overshoot is tolerated,
since that is rounding rather than invention. A moment that collapses because both ends
snapped to the same boundary is extended to a minimum length instead of dropped — the model
did observe something there.

---

## 2026-09-28 — The prompt carries the local analysis, and names no convention

**Decision:** every clip-log prompt includes a CONTEXT block with the shot boundaries,
usable spans, occlusion edges, speech spans, audio labels and transcript, and instructs the
model to prefer those timings over its own. `PROMPT_VERSION` is part of the cache key.

**Evidence:** grounding turns "guess when this happened" into "choose among these instants",
which is a far easier task and the reason snapping usually finds a target within its window.
A test asserts no prompt contains the words *festival*, *hand*, *lens*, *vlog* or
*aftermovie*: §2.5 requires conventions to be discovered from patterns across the footage
and confirmed by the user, and a prompt that names one would find it in footage that does
not contain it. Versioning the prompt in the cache key means rewording a question
invalidates the answers rather than mixing outputs from two different questions — this is
the only paid per-asset step, so the key has to be exactly right in both directions.

---

## 2026-09-28 — A logged moment outranks a technically better non-moment

**Decision:** `Candidate.score` adds `1.2 × moment_score` plus a per-kind bonus (hero 0.35,
highlight 0.20, reaction 0.18, quote 0.12, scenic 0.05, transition candidate 0.00). Logged
moments become candidates alongside the usable spans rather than replacing them.

**Evidence:** a semantic moment is the only evidence available that a range is *interesting*
rather than merely sharp and steady. Weighted this way, a logged hero moment with mediocre
technical quality scores 2.06 against 0.77 for a technically perfect non-moment — which is
the intended ordering, because an edit made of flawless shots of nothing is worse than one
made of slightly soft shots of something. Transition candidates earn no general bonus
because they are only useful at a boundary. Keeping the usable spans as candidates means a
clip the model logged no moments in is still available rather than excluded.

---

## 2026-09-28 — Pattern mining asks only structural questions

**Decision:** `index/patterns.py` measures how often clips begin or end with each kind of
analyzer event, reports counts, examples and a confidence, and names nothing. A test
asserts the output contains none of the words *hand*, *lens*, *vlog* or *festival*.

**Evidence:** this module is the test of §2.5. On a 12-clip shoot where 4 clips were
obscured at both ends, the miner's top finding is
`bookend.occlusion.reveal~occlusion.cover` matching **exactly** those 4 clips — M3's
headline acceptance criterion, met with no code that knows what a hand or a lens is. The
fixture's asset ids are deliberately neutral (`a_m0`, not `a_vlog0`), because the first
version of that test passed only because the *fixture* was doing the naming.

---

## 2026-09-28 — Motifs are filtered by measured coverage and selectivity

**Decision:** an event type whose events cover more than 50% of a clip cannot mark a motif
at its edge, and a pattern present in more than 90% of clips is rejected. Patterns are
ranked by *selectivity* (peaking at half the footage) before confidence, and patterns
identifying an identical set of clips are collapsed to one.

**Evidence:** measured, in three steps, on the same 12-clip shoot:

| filter | patterns surfaced | real motif's rank |
|---|---|---|
| none | 70+ | buried, below ~30 at confidence 1.00 |
| + coverage and selectivity | 10 | joint 1st, with three restatements |
| + collapse duplicates | 2 | 1st |

Per-second analyzers (motion, quality metrics, colour stats, subjects) emit something in
every clip's first and last second *by construction*, so without the coverage filter they
produce dozens of patterns at perfect confidence that describe the sampling rather than the
footage. Coverage is **measured** rather than a hardcoded list of analyzer names, so
analyzers that do not exist yet are handled. Selectivity is the ranking signal because a
motif in every clip separates nothing; sheer prevalence mostly measures how the analyzers
sample. Edge events (`reveal`, `cover`, `word`) are preferred over spans when both identify
the same clips, because a convention's treatment is to *trim to* an instant and there is
nothing to trim to mid-span.

---

## 2026-09-28 — A convention does nothing until confirmed, and then trims at two points

**Decision:** conventions persist in `conventions.yaml` with `proposed | confirmed |
rejected`. Only confirmed ones act, and they act twice: the planner refuses to *choose*
ranges inside the motif, and `apply_conventions` trims any shot that still overlaps it.
Re-running the miner refreshes a convention's evidence but never its status.

**Evidence:** §12 requires the user to confirm, so discovery must not imply application —
tests assert that a proposed or rejected convention changes nothing. Applying at both
points is not redundant: the planner check stops a shot being chosen inside the motif and
wasting the budget it was allocated, while the trim catches shots the *agent* wrote by hand
rather than the builder choosing. A shot left entirely inside the motif is dropped, because
none of it was footage the shooter intended to be seen. Preserving status across re-runs
matters because re-running analysis must not silently un-confirm a decision or re-ask a
question that has been answered.

---

## 2026-09-28 — Pipeline order: source-range edits before the rails, timeline repairs after

**Decision:** `render.pipeline` applies confirmed conventions (a source-range edit) *before*
`apply_rails`, and speech repair, caption repair and SFX spotting (all timeline-position
dependent) after it.

**Evidence:** applying the convention trim after the rails shortened shots the rails had
already laid out head to tail, opening one- and two-frame gaps that render as black frames
and fail validation — a regression the end-to-end run caught immediately after conventions
were wired in. The rule generalizes: anything that changes `src_in`/`src_out` must run
before relayout so relayout can absorb it, and anything that reads `timeline_in` must run
after snapping has finished moving it. A test spies on the call order to pin it.

---

## 2026-09-28 — Without clip logs, energy comes from measured motion

**Decision:** `Candidate.energy_value` uses the clip log's 1–5 energy when there is one, and
otherwise scales the measured camera-motion magnitude.

**Evidence:** the rhythm report derives *visual* energy from motion events, while the
planner was matching on *semantic* energy — which without clip logs sits at its default 3
for every candidate, so energy matching did nothing at all. The report duly reported the
consequence: `visual energy barely tracks the music (r=-0.09)`. Deriving energy from motion
when no log exists makes the planner optimize the same quantity the report scores it on, and
takes the correlation from **-0.07 to +0.49** on the fixture shoot — the busiest footage now
lands in the loudest section. A logged energy still wins where one exists, because that is a
judgement about the content and motion is only a proxy for it.

---

## 2026-09-28 — Finishing is a Remotion component, not an ffmpeg filter

**Decision:** grain, vignette and halation are applied once by `finishing.finish` over the
whole composition — text included — and `grade_from_style` no longer emits grain or
vignette.

**Alternatives:** ffmpeg's `noise` and `vignette` filters in the conform, which is where
they started (§30 lists this as an open decision).

**Evidence:** **halation** decides it. Bloom around highlights is what makes digital
footage read as filmed, and doing it properly needs a blurred, brightness-thresholded copy
of the frame screened back over itself — a compositing operation, not a filter. Once
halation has to be in the composition, putting grain and vignette there too keeps the whole
finishing pass as one layer with one set of parameters rather than split across two stages
that must be kept consistent. And it has to be *over* the text: grain under the captions
but not over them reads as two images composited together. A test asserts the ffmpeg grade
no longer carries them, because applying both would double the grain.

---

## 2026-09-28 — Four more components, and three bugs the gallery caught

**Decision:** added `transition.dip` (white/black dip and flash), `transition.luma_wipe`,
`shot_fx.impact_shake` and `text.stamp`, taking the catalogue to 11.

**Evidence for the design choices, and what rendering them found:**

`transition.dip`'s `flash` preset sets `hold: 0`, which produced an interpolation range with
`0.5` twice — Remotion requires strictly increasing inputs and threw. A legal parameter
value crashed the component, and only a gallery render surfaced it. The stops are now built
to be strictly increasing whatever the hold is.

`text.stamp` overflowed the frame: "18:40 — MAIN STAGE" at the caption size with stamp
tracking ran off the right edge. Fitting the text alone was still not enough, because the
pill's padding, the two gaps and the divider all scale with the font size — the size that
leaves room for them is derived exactly rather than guessed with a factor. And the divider
then vanished, because a fixed-width flex item inside a `max-width` container collapses
under the default `flex-shrink: 1`.

`shot_fx.impact_shake` decays exponentially rather than holding amplitude: real impact
energy dissipates, so the first two frames carry almost all the displacement. Constant
amplitude over fifteen frames is how this effect is usually done badly. Displacement is
seeded from the frame number via Remotion's `random`, so it is noisy but deterministic —
which §13.2 requires and which the gallery's perceptual diff depends on.

`transition.luma_wipe` is a soft gradient mask with a light glow on the edge, never a hard
edge, and its intent points at a *discovered* occlusion motif: where the footage already
goes dark, wiping through that darkness turns an artefact of how it was shot into
punctuation.

Gallery renders also revealed that `render_gallery_item` passed a relative destination while
running with its cwd inside `render/remotion`, so output went to the wrong directory.

---

## 2026-09-28 — Auto-checks run on the rendered file, and found a real defect immediately

**Decision:** `critic/autochecks.py` runs after every render on the muxed MP4: black runs,
frozen video, integrated loudness, true peak, text in the unsafe zone, and duration against
the plan. Findings are reported but never block — the file exists and is worth looking at
even when a check fails.

**Evidence:** everything else in the system validates the *plan*, and the very first
auto-check run found a defect no plan validation could: the finished MP4 peaked at **-0.5
dBTP against a -1.0 ceiling**. The mix was correctly limited to -1.0, but the AAC encode in
the mux overshot the PCM it was given, which is what lossy codecs do. That is the whole
argument for checking the output rather than the intention.

---

## 2026-09-28 — Loudness is iterated, and the mix leaves headroom for the encoder

**Decision:** `render_mix` limits to `true_peak_dbtp - 1.0` dB when the output will be
lossily encoded, and applies its loudness gain over up to three measure-and-correct passes.

**Evidence:** two measured problems with the single-pass version.

The encode headroom addresses the -0.5 dBTP finding above: 1 dB is enough for AAC's
overshoot, and the finished file now measures -1.4 dBTP against its -1.0 ceiling.

The iteration addresses undershoot. One measure-then-apply pass lands consistently quiet,
because the output limiter pulls peaks down and takes loudness with them: the mix measured
**-15.6 LUFS against a -14.0 target**. Measuring again after limiting and applying the
residual converges — the same mix now lands at -14.5, inside the checks' tolerance. Not a
dynamic `loudnorm`, which pumps under a music bed.

---

## 2026-09-28 — Text in the unsafe zone is detected by bimodality, not by gradient

**Decision:** `check_safe_areas` flags a band when it is both bright (>5% of cells above
225) and bimodal (>50% of cells either above 225 or below 40).

**Alternatives:** mean horizontal gradient, which was the first implementation.

**Evidence:** the gradient measure does not survive the downscale the check samples at. At
96 cells wide a 20-pixel glyph becomes six cells and its mean gradient averages out to 8.8,
below any threshold that busy footage also clears. Bimodality describes what text actually
is — light glyphs directly against a dark scrim, with few mid-tones — and separates it
cleanly from footage, which is continuous. The check stays a warning in every case: footage
legitimately fills the frame, so this can only ever flag something for a human to look at.

---

## 2026-09-28 — First run on real footage: what 130 iPhone clips changed

Everything below was found by running the pipeline over a real shoot — 88 videos and 42
photos from an iCloud Shared Album, 37.4 minutes — rather than over synthetic fixtures. The
fixtures were not wrong; they were uniform, and every defect here needed variety or scale to
appear at all.

**Measured properties of the material.** All 88 videos sit between 2.9 and 3.2 Mbps
regardless of content, which is a fixed-ceiling transcode rather than a camera: the Shared
Album re-encodes to 720p. 63 of 88 clips are landscape and carry 29.8 of the 37.4 minutes,
so the output format is 16:9 — cropping that to 9:16 would have thrown away three quarters
of the shoot or upscaled it 2.7×. The 42 photos arrived at 2730px on the long edge, 2.5× the
output height, making them the sharpest material in the set. The ingest report's own §7.2
degradation check flagged all 88 videos without being asked.

---

## 2026-09-28 — Shot lengths are whole half-beats, and the grid includes off-beats

**Decision:** `SnapKind.HALF_BEAT` and `BeatGrid.half_beats` add the eighth-note grid.
`avg_shot_beats` values are whole beats and doubled from their previous values. Shot lengths
are quantized to whole half-beats by `_variation_half_beats`, whose mean is corrected back to
the target exactly. `on_grid_ratio` becomes 1.0.

**Alternatives:** keeping the beat as the finest grid position; deliberately displacing a
quarter of cuts off the grid, as the style previously asked for.

**Evidence:** the style asked for two things that cannot both hold. `avg_shot_beats.drop` was
0.5, so shots lasted half a beat, while `on_grid_ratio` was 0.75 — but the finest grid
position was the beat, so a half-beat shot puts the cut after it between positions *by
construction*. The first real plan measured **51% of cuts on the grid against a target of
75%**, and no amount of snapping could have fixed it.

Nothing in the builder ever consumed `on_grid_ratio`, so the 25% of cuts nominally off the
grid on purpose were all accidents. And there is no room for the gesture: at 120 BPM the
half-beat positions are 250ms apart and the snap window is 120ms, so a cut displaced far
enough to be deliberate is indistinguishable from a careless one. What makes an edit feel
made by a person is that its shots are not all the same length, so the rhythm report now
measures **shot-length spread** (std/mean) and warns below 0.2, replacing a warning that
fired when *every* cut was on the grid.

The doubling is separate and about legibility: 0.5 beats at 120 BPM is 0.25s, seven frames,
less time than it takes to recognise what is in the frame. The first real edit ran 166 shots
in 90 seconds — 0.54s average — which is a strobe, not a cut rate.

---

## 2026-09-28 — A snap window must scale with the grid it snaps to, and measuring is stricter than snapping

**Decision:** `grid_window_ms(positions, window_ms, fraction)` caps a window at a fraction of
the grid's own median spacing. Rails use 0.5; the rhythm report measures with 0.25.

**Evidence:** a fixed window silently stops discriminating as the grid gets finer. At 120 BPM
the half-beat positions are 250ms apart, and at 30fps **every frame is within 117ms of one**
— so with the 120ms window tuned for the beat grid, every cut counted as on-grid whatever the
edit did. The measure read 100%, which a test caught as vacuous.

Separating the two fractions matters as much. Sharing one threshold at 0.25 left the rail
unable to move a cut that had drifted 80–120ms, so it did not try: the honest measurement was
55%. Half the spacing is the natural limit for a *rail* — up to there the nearest position is
unambiguous — while a *measure* should stay tight. With the two separated, 87% of cuts land
within the measuring window and 95 of 108 within 20ms.

---

## 2026-09-28 — Source snapping and timeline snapping are separate fields

**Decision:** `SnapSpec` gains `timeline`, distinct from `in_`/`out`. Rails read
`snap.timeline`, falling back to a timeline kind left in `snap.in_` so older plans still work.

**Evidence:** a captioned shot must start on a word, and the single `in_` field could hold
either a source kind or a timeline kind but not both — so every captioned shot was exempt
from the musical grid. In a real edit that was 33 of 108 shots and 45% of the off-grid cuts.
The two never conflict: moving *when* a shot appears does not change *which* frames of it are
used. The same bug applied to section openers, whose downbeat snap silently discarded
whatever source snapping they had.

---

## 2026-09-28 — The rails can move a cut backwards

**Decision:** `_resize` replaces `_extend` at the call site and accepts negative frame
counts, shortening the preceding shot.

**Evidence:** the rails only ever moved a cut *later*, because moving it earlier requires the
preceding shot to give up frames and nothing could do that. Of 108 shots in a real edit only
**10 cuts were snapped at all**. Extending needs handle material and often has none;
shortening only needs the shot to stay above the minimum, so it nearly always succeeds. With
both directions available, 45 cuts snap.

---

## 2026-09-28 — The rhythm report measures the plan that renders, not the plan on disk

**Decision:** the rails sequence moves to `plan/prepare.py`, and both `montaje render` and
`montaje debug rhythm` call it.

**Evidence:** the report read the plan file straight off disk, which is the *pre-rails*
request. It reported 58% of cuts off the musical grid for an edit whose cuts the rails were
about to place on it — a description of something that never rendered. Extracting the
sequence also removes the risk that the two drift, which is the same bug in slower motion.

---

## 2026-09-28 — Quotes come from the transcript, never from the model

**Decision:** `reconcile_quotes` replaces each quote's text with the ASR words transcribed
under it and tightens the span onto them. A quote with no transcribed speech under it is
marked unusable, unless the clip has no transcript at all.

**Alternatives:** instructing the model not to translate, which was tried first and ignored.

**Evidence:** asked in English about Spanish footage, the model returned fluent English —
"Here we are, I don't see anything" — and an explicit verbatim instruction did not stop it.
Worse, it filled gaps: for a clip whose transcript reads "¿Puedes escuchar? oh my god" it
produced "Oh my god, this is coming hard guys", inventing a line nobody said. Captions play
over the speaker's own voice, so an invented or translated one is always wrong.

This is §2.2 applied to words: the model picks which moment is worth quoting, the transcript
says what was said. It also settles the mixed-language case, which is the real one here — the
speakers switch language mid-clip, so no single language could be asked for. The brief's
`language` is now a *hint about what will be heard*, omitted entirely when unset, and part of
the clip-log cache key because it changes the answer.

---

## 2026-09-28 — Transient API failures retry; facts about the request do not

**Decision:** `_call_with_retry` retries 408/429/500/502/503/504 and transport faults up to 7
times with exponential backoff and jitter. Everything else raises immediately.

**Evidence:** the very first real clip-log call returned `503 UNAVAILABLE — "This model is
currently experiencing high demand"`. Without retries, a run over 130 clips comes back with a
random handful missing and nothing in the output says which clips were never understood; the
planner silently ranks those on sharpness and motion alone. The distinction is worth drawing
precisely: a `402 RESOURCE_EXHAUSTED` for depleted credit failed instantly, as it should,
because retrying it would have hidden the real cause behind a minute of backoff.

A model fallback was considered and rejected on measurement: when 3.8-flash was returning
503, so were 3.7, 3.6 and 3.5 simultaneously. There was no faster model to fall back to, only
a later moment to ask again.

---

## 2026-09-28 — Ducking is a sum over merged spans, not nested maxima

**Decision:** `duck_windows` merges overlapping and adjacent speech into disjoint spans at
the deepest depth involved, and `duck_envelope_expression` sums one flat term per segment.

**Evidence:** the previous expression nested one `max(if(...))` per speaking clip, so its
depth grew with the edit. At **99 speaking clips ffmpeg failed with "Error initializing
filters"** and the render died in the mix. Summed over disjoint spans the depth is always
one: the same edit produces 6 merged windows and an 864-character expression.

Merging is also better mixing — the music stays down through a run of lines instead of
pumping up between them.

A second defect in the same expression: the guard against a zero-length ramp was `1e-6`,
below the four decimals the duration is *printed* with, so a release tail clipped by the end
of the timeline rendered as `/0.0000`. Guards have to be expressed at the precision of the
output, not of the arithmetic.

---

## 2026-09-28 — A clip's score decays with each use of its asset

**Decision:** `_pick` multiplies a candidate's score by `REPEAT_DECAY ** uses_of_that_asset`,
with `REPEAT_DECAY = 0.54`.

**Evidence:** scoring each range on its own merit means the highest-scoring clips win every
comparison. Over 130 clips the first real edit drew all 108 of its shots from **20 assets,
one of them fifteen times**, while 52 clips with usable footage never appeared. A viewer
reads that as the same scene coming round again. With the decay the same edit uses 72 assets
and reuses none more than twice.

The decay is multiplicative rather than a cap because sometimes one clip really is the best
thing in the shoot — but what buys a repeat is a *logged moment*, weighted 1.2, not technical
merit. A fresh ordinary clip beating a second look at a sharp clip of nothing is the correct
ranking.

---

## 2026-09-28 — "No usable spans" is not the same as "no analysis"

**Decision:** `collect_candidates` skips an asset that has quality *metrics* but no *usable*
spans, and only falls back to offering the whole clip when there are no metrics at all.

**Evidence:** the fallback conflated two opposite situations. Of 130 real clips, **58 had no
usable span** — night footage measuring detail 1.4 against a threshold of 4.0, with 70% of
pixels near black — and all of them were being offered whole, in direct contradiction of the
measurement that had just rejected them.

This surfaced only after the repeat decay above broadened coverage: while the planner was
living on 20 good clips the bug was latent, and the moment it spread out it reached the dark
ones. One put **half a second of black into a finished render at 54.73s**, which nothing in
plan validation could see and the output auto-checks caught on the muxed file.

---

## 2026-09-28 — A free-tier quota error stops the run, because it is a privacy condition

**Decision:** `is_free_tier_quota` separates the free tier's daily cap from an ordinary rate
limit. It is not retried, and with `semantic.provider_tier=paid` (the default) it stops the
whole semantic pass with an explanation rather than failing clip by clip.

**Evidence:** with credit added to the account, requests were still being counted against
`generativelanguage.googleapis.com/generate_content_free_tier_requests`, **limit 20 per day
per model**. Two consequences, and the second matters more. A per-day cap cannot be waited
out inside a run, so blind backoff turns a clear error into a slow one — 130 clips × 7
attempts × up to 90s. And the request went out on free-tier terms, under which the provider
may use the content to improve its products. §10 sets `provider_tier: paid` precisely so that
personal footage is not sent that way, so continuing would have sent 129 more clips under
terms the project declines. `provider_tier: free` remains available as explicit consent.
