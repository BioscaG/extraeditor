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
