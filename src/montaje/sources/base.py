"""Source protocol (§7)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field


class RemoteAsset(BaseModel):
    """An asset as listed by a source, before it is local."""

    uid: str  # source-scoped stable id (path for folders, localIdentifier for Photos)
    filename: str
    size_bytes: int | None = None
    kind_hint: str | None = None
    extra: dict = Field(default_factory=dict)


class LocalAsset(BaseModel):
    """A fetched file, ready for ingest."""

    path: Path
    remote: RemoteAsset
    live_photo_pair_uid: str | None = None


class Source(Protocol):
    name: str

    def list(self) -> list[RemoteAsset]: ...

    def fetch(self, asset: RemoteAsset, dest: Path) -> LocalAsset: ...
