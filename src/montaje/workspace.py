"""Project workspace layout and atomic file helpers (§6)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import yaml

from montaje.models.brief import Brief


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str))


class Workspace:
    """A project under projects/<slug>/ — every path in one place."""

    def __init__(self, root: Path):
        self.root = root

    @property
    def slug(self) -> str:
        return self.root.name

    project_yaml = property(lambda self: self.root / "project.yaml")
    sources_yaml = property(lambda self: self.root / "sources.yaml")
    db_path = property(lambda self: self.root / "montaje.db")
    media_dir = property(lambda self: self.root / "media")
    cache_dir = property(lambda self: self.root / "cache")
    conventions_yaml = property(lambda self: self.root / "conventions.yaml")
    plans_dir = property(lambda self: self.root / "plans")
    renders_dir = property(lambda self: self.root / "renders")
    exports_dir = property(lambda self: self.root / "exports")
    drafts_dir = property(lambda self: self.root / "drafts")
    logs_dir = property(lambda self: self.root / "logs")
    intermediates_dir = property(lambda self: self.root / "intermediates")
    music_dir = property(lambda self: self.root / "music")

    def asset_cache(self, asset_id: str) -> Path:
        return self.cache_dir / asset_id

    def proxy_path(self, asset_id: str) -> Path:
        return self.asset_cache(asset_id) / "proxy.mp4"

    def audio_16k_path(self, asset_id: str) -> Path:
        return self.asset_cache(asset_id) / "audio_16k.wav"

    def audio_48k_path(self, asset_id: str) -> Path:
        return self.asset_cache(asset_id) / "audio_48k.wav"

    def thumbs_dir(self, asset_id: str) -> Path:
        return self.asset_cache(asset_id) / "thumbs"

    def analysis_path(self, asset_id: str, analyzer: str, version: int) -> Path:
        return self.asset_cache(asset_id) / "analysis" / f"{analyzer}@{version}.json"

    def create(self, brief: Brief) -> None:
        for d in (
            self.media_dir,
            self.cache_dir,
            self.plans_dir,
            self.renders_dir,
            self.exports_dir,
            self.drafts_dir,
            self.logs_dir,
            self.intermediates_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        if not self.project_yaml.exists():
            atomic_write_text(
                self.project_yaml,
                yaml.safe_dump(
                    brief.model_dump(mode="json", exclude_none=True), sort_keys=False, allow_unicode=True
                ),
            )
        if not self.sources_yaml.exists():
            atomic_write_text(self.sources_yaml, yaml.safe_dump({"sources": []}))

    def load_brief(self) -> Brief:
        data = yaml.safe_load(self.project_yaml.read_text()) or {}
        data.setdefault("name", self.slug)
        return Brief.model_validate(data)

    def load_sources(self) -> list[dict]:
        if not self.sources_yaml.exists():
            return []
        return (yaml.safe_load(self.sources_yaml.read_text()) or {}).get("sources", [])

    def add_source(self, entry: dict) -> None:
        sources = self.load_sources()
        if entry not in sources:
            sources.append(entry)
        atomic_write_text(self.sources_yaml, yaml.safe_dump({"sources": sources}, sort_keys=False))

    def plan_paths(self) -> list[Path]:
        return sorted(self.plans_dir.glob("plan_v*.json"))

    def next_plan_version(self) -> int:
        """One past the highest plan on disk.

        A rebuilt plan must not reuse a version number: a stale higher version left by
        an earlier session would still be the "latest" and would silently shadow the
        rebuild, which is how an edit ends up rendering from a plan nobody asked for.
        """
        highest = 0
        for path in self.plan_paths():
            digits = path.stem.removeprefix("plan_v")
            if digits.isdigit():
                highest = max(highest, int(digits))
        return highest + 1

    @classmethod
    def find(cls, name_or_path: str, projects_root: Path | None = None) -> Workspace:
        p = Path(name_or_path)
        if p.is_dir() and (p / "project.yaml").exists():
            return cls(p.resolve())
        root = projects_root or Path.cwd() / "projects"
        candidate = root / name_or_path
        if (candidate / "project.yaml").exists():
            return cls(candidate.resolve())
        raise FileNotFoundError(f"project '{name_or_path}' not found (looked in {root})")
