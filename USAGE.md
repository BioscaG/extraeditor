# Using montaje

The build specification is `README.md`; this is how to actually run what exists.

## Install

```bash
uv python install 3.12
uv venv --python 3.12
uv pip install -e .

npm install          # Remotion + the craft library's peer deps, hoisted to the repo root
```

**ffmpeg.** montaje discovers ffmpeg at runtime and probes its filters. Homebrew's
default bottle ships without `libzimg` and `libplacebo`, which are the only filters that
can correctly tone-map HLG/PQ to SDR — without them HDR footage falls back to a
hardware or naive path. For the full colour pipeline:

```bash
brew install ffmpeg-full   # keg-only; does not replace your existing ffmpeg
```

Check what was found and what it can do:

```bash
python -c "from montaje import ffmpeg; print(ffmpeg.capability_report())"
```

**Optional.** `uv pip install mlx-whisper` for word-level transcription on Apple
Silicon (or `faster-whisper` elsewhere). Without it, captions have no words and are
silently omitted rather than rendered empty.

**Semantic analysis.** Set `GEMINI_API_KEY` and run `montaje analyze --semantic` to log
what is *in* each clip. This is the single biggest quality lever in the system: without
it the planner ranks footage on sharpness, steadiness and motion only, so selection is
arbitrary within the technically sound material. It is also the only paid per-asset step,
so results are cached by `(asset, model, prompt version)`.

## A full pass

```bash
montaje init festival-2026
# edit projects/festival-2026/project.yaml: goal, format, duration, music, style

montaje source add folder ~/Movies/festival
montaje ingest                    # probe, proxies, HDR→SDR, audio, thumbnails
montaje report                    # VFR, HDR mix, degraded assets, failures
montaje analyze                   # shots, quality, occlusion, motion, audio, VAD, ASR, colour
montaje conventions propose       # discover recurring motifs; confirm the real ones
montaje plan                      # baseline EditPlan from the music structure
montaje render --quality draft    # rails, validate, conform, mix, compose, mux
montaje export all                # SRT + OTIO + FCPXML referencing the originals
```

Every command is idempotent and resumable: re-running `ingest` or `analyze` skips work
whose inputs and parameters have not changed.

## Directing it with an agent

```bash
montaje mcp --project festival-2026
```

This exposes 19 tools over stdio (`montaje debug tools` lists them). Point Claude Code at
it and the model becomes the director: it surveys the footage, builds a plan, reads the
rhythm report, and renders. The rails still run underneath whatever it decides.

## Inspecting things

```bash
montaje debug rhythm              # cut-to-beat offsets, pacing vs the style, densities
montaje debug events <asset_id>   # raw analyzer output
montaje debug tonemap <clip.mov>  # side-by-side HDR→SDR stills, to decide the method
montaje library list              # components with their intent, and licensed SFX
montaje library gallery           # render every component × preset, plus an HTML index
```

Everything intermediate is a file you can open — that is deliberate (§2.9). The most
useful ones:

| path | what it is |
|---|---|
| `projects/<slug>/report.md` | what the footage actually is |
| `projects/<slug>/plans/plan_v###.json` | the edit, as data |
| `projects/<slug>/plans/rhythm_v###.md` | why it feels the way it does |
| `projects/<slug>/cache/<asset>/analysis/` | every analyzer's output, versioned |
| `projects/<slug>/renders/` | the video |
| `projects/<slug>/exports/` | timelines, captions, stems |

## What is built

| README section | state |
|---|---|
| §6–8 workspace, folder source, ingest, HDR | done |
| §9 local analysis | shots, quality, occlusion, motion, audio events, VAD, ASR, loudness, colour stats, subjects. Missing: embeddings, sync, dedupe, separation |
| §10 semantic clip logs | implemented; needs `GEMINI_API_KEY`. Untested against the live API |
| §13–15 craft library, tokens, colour pipeline | tokens and 11 components (5 transitions, 3 text, 2 shot fx, finishing); §13.3 lists more |
| §14 music editing, mix, SFX | beats, structure, fit-to-duration, spotting, mix, stems. Missing: lyrics, generation |
| §17–18 EditPlan, ops, rails, rhythm | done |
| §19 render and exports | done, except the overlay alpha track is untested against Resolve |
| §16 director | tools and MCP server; no built-in loop yet |
| §11–12 pattern mining and conventions | done: motifs discovered, confirmed by you, then applied |
| §20 critic, §21 workshop, §22 style learning | not started |
| §7.2 Apple Photos bridge | not started |

`DECISIONS.md` records every non-obvious choice and the evidence for it.

## Conventions

Footage has habits. A vlogger might open every clip by uncovering the lens and close it by
covering it again — and an edit that keeps those frames looks careless.

montaje finds such habits without being told what they are:

```bash
montaje conventions propose        # mine the footage for recurring motifs
montaje conventions list           # see what it found and what it means
montaje conventions confirm <id>   # only now does it affect any edit
```

The miner asks purely structural questions — how often do clips *begin* with a particular
kind of event, how often do they *end* with one, and does that separate some clips from
others? It reports counts and examples. Deciding that a dark span at the start of a clip is
a hand over the lens is your call, and until you confirm it nothing is trimmed.

A confirmed convention trims its motif out of every shot from a matching clip, and the
planner stops choosing ranges inside it in the first place.
