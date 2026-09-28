"""Read the craft library's component metadata from Python (§13.2, §13.4).

The TSX `meta` export is the source of truth — it lives next to the code it
describes, so it cannot drift from the component. Rather than duplicating it in
Python, this module extracts it by asking Node, which means the director and the
validator see exactly what the renderer sees.

Falls back to parsing the TSX when Node is unavailable, so `montaje library list`
and plan validation still work on a machine with no Node install.
"""

from __future__ import annotations

import functools
import json
import re
import shutil
import subprocess
from pathlib import Path

from montaje.models.library import ComponentKind, ComponentMeta, ComponentStatus, SfxMeta

REPO_ROOT = Path(__file__).resolve().parents[3]
LIBRARY_DIR = REPO_ROOT / "library"
COMPONENTS_DIR = LIBRARY_DIR / "components"
REMOTION_DIR = REPO_ROOT / "render" / "remotion"

# Script that imports the registry and prints every component's meta as JSON. It is
# written into render/remotion/src/ so esbuild resolves react/remotion/zod through the
# workspace's hoisted node_modules; the library is three levels up from there.
_DUMP_SCRIPT = """
import { allMeta } from "../../../library/components/index";
const out = allMeta().map((m) => ({
  id: m.id,
  version: m.version,
  kind: m.kind,
  status: m.status,
  duration: {
    min_frames: m.duration.minFrames,
    max_frames: m.duration.maxFrames,
    default_beats: m.duration.default.beats ?? null,
    default_frames: m.duration.default.minFrames ?? null,
  },
  energy: m.energy,
  tags: m.tags,
  beat_anchor: m.beatAnchor ?? null,
  motion_match: m.motionMatch ?? null,
  sfx: m.sfx ? { default: m.sfx.default, anchor: m.sfx.anchor } : null,
  aspect_ratios: m.aspectRatios,
  author: m.author,
  presets: m.presets ?? {},
  intent: m.intent,
}));
console.log(JSON.stringify(out));
"""


class LibraryUnavailable(RuntimeError):
    """The library could not be read at all."""


def _dump_via_node() -> list[dict] | None:
    """Ask esbuild+node for the real meta objects. None if Node is unavailable."""
    if shutil.which("node") is None:
        return None
    esbuild = REPO_ROOT / "node_modules" / ".bin" / "esbuild"
    if not esbuild.is_file():
        return None
    script = REMOTION_DIR / "src" / ".meta_dump.ts"
    bundle = REMOTION_DIR / "src" / ".meta_dump.js"
    try:
        script.write_text(_DUMP_SCRIPT)
        subprocess.run(
            [str(esbuild), str(script), "--bundle", "--platform=node", "--format=cjs",
             f"--outfile={bundle}", "--log-level=error"],
            check=True, capture_output=True, cwd=REPO_ROOT,
        )
        out = subprocess.run(["node", str(bundle)], check=True, capture_output=True,
                             text=True, cwd=REPO_ROOT)
        return json.loads(out.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None
    finally:
        script.unlink(missing_ok=True)
        bundle.unlink(missing_ok=True)


_META_FIELD = re.compile(r'^\s*(\w+):\s*"([^"]*)"', re.MULTILINE)


def _parse_tsx(path: Path) -> dict | None:
    """Minimal fallback: pull the scalar meta fields out of a TSX source file."""
    text = path.read_text()
    start = text.find("export const meta")
    if start < 0:
        return None
    block = text[start : text.find("\n};", start)]
    fields = dict(_META_FIELD.findall(block))
    if "id" not in fields or "version" not in fields:
        return None
    tags = re.search(r"tags:\s*\[([^\]]*)\]", block)
    energy = re.search(r"energy:\s*\[([^\]]*)\]", block)
    aspects = re.search(r"aspectRatios:\s*\[([^\]]*)\]", block)
    duration = re.search(r"minFrames:\s*([\d.]+),\s*maxFrames:\s*([\d.]+)", block)
    presets = re.findall(r"^\s{4}(\w+):\s*\{", block, re.MULTILINE)
    return {
        "id": fields["id"],
        "version": fields["version"],
        "kind": fields.get("kind", "overlay"),
        "status": fields.get("status", "draft"),
        "duration": {
            "min_frames": int(float(duration.group(1))) if duration else 1,
            "max_frames": int(float(duration.group(2))) if duration else 300,
        },
        "energy": _string_list(energy),
        "tags": _string_list(tags),
        "beat_anchor": fields.get("beatAnchor"),
        "motion_match": fields.get("motionMatch"),
        "aspect_ratios": _string_list(aspects) or ["9:16", "16:9", "1:1"],
        "author": fields.get("author", "human"),
        "presets": dict.fromkeys(presets, {}),
        "intent": "",
    }


def _string_list(match: re.Match | None) -> list[str]:
    if match is None:
        return []
    return [s.strip().strip('"\'') for s in match.group(1).split(",") if s.strip()]


def _dump_via_parse() -> list[dict]:
    out: list[dict] = []
    for path in sorted(COMPONENTS_DIR.rglob("*.tsx")):
        parsed = _parse_tsx(path)
        if parsed is not None:
            out.append(parsed)
    return out


@functools.lru_cache(maxsize=1)
def load_components() -> list[ComponentMeta]:
    """Every component's metadata, preferring the real TS values."""
    raw = _dump_via_node()
    source = "node"
    if raw is None:
        raw = _dump_via_parse()
        source = "parse"
    if not raw:
        raise LibraryUnavailable(
            f"no components found in {COMPONENTS_DIR} (tried node and source parsing)"
        )
    metas: list[ComponentMeta] = []
    for entry in raw:
        entry.setdefault("presets", {})
        metas.append(ComponentMeta.model_validate(entry))
    load_components.source = source  # type: ignore[attr-defined]
    return metas


def refresh() -> None:
    """Drop the cache; used by tests and after editing the library."""
    load_components.cache_clear()


def find(ref: str) -> ComponentMeta | None:
    """Resolve `id`, `id@major.minor` or `id@version` to a component."""
    components = load_components()
    if "@" not in ref:
        matches = [c for c in components if c.id == ref]
        return _newest(matches)
    cid, version = ref.split("@", 1)
    matches = [
        c for c in components
        if c.id == cid and (c.version == version or c.version.startswith(f"{version}."))
    ]
    return _newest(matches)


def _newest(matches: list[ComponentMeta]) -> ComponentMeta | None:
    if not matches:
        return None
    return max(matches, key=lambda c: tuple(int(p) for p in c.version.split(".")))


def stable_refs() -> set[str]:
    """Versioned refs of every stable component — what validation checks against."""
    return {c.ref for c in load_components() if c.status == ComponentStatus.STABLE}


def search(
    query: str = "",
    kind: ComponentKind | str | None = None,
    energy: str | None = None,
) -> list[ComponentMeta]:
    """Free-text plus facet search over the library, for the director's tool."""
    words = [w for w in query.lower().split() if w]
    out: list[ComponentMeta] = []
    for c in load_components():
        if kind is not None and c.kind != kind:
            continue
        if energy is not None and energy not in c.energy:
            continue
        haystack = " ".join([c.id, *c.tags, *c.energy, c.kind.value]).lower()
        if words and not all(w in haystack for w in words):
            continue
        out.append(c)
    return out


def load_sfx() -> list[SfxMeta]:
    """SFX metadata from `library/sfx/sfx.yaml`.

    A file without a license entry is omitted rather than returned: §14.1 forbids
    adding one without a known license, and silently shipping it in a render would
    make the project's output unusable.
    """
    import yaml

    path = LIBRARY_DIR / "sfx" / "sfx.yaml"
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    out: list[SfxMeta] = []
    for entry in data.get("sfx", []):
        if not entry.get("license"):
            continue
        out.append(SfxMeta.model_validate(entry))
    return out


def licensed_sfx_ids() -> set[str]:
    return {s.id for s in load_sfx()}


def sfx_paths() -> dict[str, Path]:
    return {s.id: LIBRARY_DIR / "sfx" / s.file for s in load_sfx()}


def sfx_peak_offsets() -> dict[str, float]:
    return {s.id: s.peak_offset_s for s in load_sfx()}
