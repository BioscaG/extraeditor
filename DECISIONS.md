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
