"""Folder source: recursive scan with Live Photo pairing (§7.1)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from montaje.sources.base import LocalAsset, RemoteAsset

VIDEO_EXTS = {".mov", ".mp4", ".m4v"}
PHOTO_EXTS = {".heic", ".jpg", ".jpeg", ".png", ".dng"}
MEDIA_EXTS = VIDEO_EXTS | PHOTO_EXTS


class FolderSource:
    """Files are used in place — no copy. `fetch` just validates existence."""

    name = "folder"

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)

    def list(self) -> list[RemoteAsset]:
        assets: list[RemoteAsset] = []
        pairs = self._live_photo_pairs()
        for p in sorted(self.root.rglob("*")):
            if p.suffix.lower() not in MEDIA_EXTS or p.name.startswith("."):
                continue
            extra = {}
            pair = pairs.get(p)
            if pair is not None:
                extra["live_photo_pair_uid"] = str(pair)
            assets.append(
                RemoteAsset(
                    uid=str(p),
                    filename=p.name,
                    size_bytes=p.stat().st_size,
                    kind_hint="photo" if p.suffix.lower() in PHOTO_EXTS else "video",
                    extra=extra,
                )
            )
        return assets

    def fetch(self, asset: RemoteAsset, dest: Path) -> LocalAsset:
        p = Path(asset.uid)
        if not p.is_file():
            raise FileNotFoundError(p)
        return LocalAsset(path=p, remote=asset, live_photo_pair_uid=asset.extra.get("live_photo_pair_uid"))

    def _live_photo_pairs(self) -> dict[Path, Path]:
        """Pair a still and a video sharing the basename (IMG_0001.HEIC + IMG_0001.MOV)."""
        by_stem: dict[tuple[Path, str], dict[str, Path]] = defaultdict(dict)
        for p in self.root.rglob("*"):
            ext = p.suffix.lower()
            if ext in PHOTO_EXTS:
                by_stem[(p.parent, p.stem.lower())]["photo"] = p
            elif ext in VIDEO_EXTS:
                by_stem[(p.parent, p.stem.lower())]["video"] = p
        pairs: dict[Path, Path] = {}
        for group in by_stem.values():
            if "photo" in group and "video" in group:
                pairs[group["photo"]] = group["video"]
                pairs[group["video"]] = group["photo"]
        return pairs
