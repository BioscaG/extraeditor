"""SQLite metadata store: assets, events, clip logs, per-asset stage status (§11, §25).

Stage status makes every command idempotent and resumable: each (asset, stage)
records done/failed with the params hash that produced it.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path

from montaje.models.asset import Asset
from montaje.models.cliplog import ClipLog
from montaje.models.events import Event

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
  asset_id TEXT PRIMARY KEY,
  path TEXT NOT NULL,
  kind TEXT NOT NULL,
  duration_s REAL NOT NULL,
  json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS status (
  asset_id TEXT NOT NULL,
  stage TEXT NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('pending','running','done','failed')),
  params_hash TEXT NOT NULL DEFAULT '',
  error TEXT,
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (asset_id, stage)
);
CREATE TABLE IF NOT EXISTS events (
  asset_id TEXT NOT NULL,
  analyzer TEXT NOT NULL,
  type TEXT NOT NULL,
  t0 REAL NOT NULL,
  t1 REAL NOT NULL,
  score REAL NOT NULL,
  data TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_asset ON events (asset_id, analyzer);
CREATE TABLE IF NOT EXISTS clip_logs (
  asset_id TEXT NOT NULL,
  model TEXT NOT NULL,
  prompt_version INTEGER NOT NULL,
  json TEXT NOT NULL,
  PRIMARY KEY (asset_id, model, prompt_version)
);
"""


class Store:
    """Thread-safe SQLite access.

    Ingest and analysis fan out over a thread pool, and a sqlite3 connection may
    only be used from the thread that created it — so each thread lazily gets its
    own connection to the same file. WAL mode lets those connections read while
    another writes.
    """

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._local = threading.local()
        with self._connect() as conn:  # create schema once, up front
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._local.conn = self._connect()
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- assets ------------------------------------------------------------

    def upsert_asset(self, asset: Asset) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO assets (asset_id, path, kind, duration_s, json) VALUES (?,?,?,?,?)",
            (
                asset.asset_id,
                str(asset.path),
                asset.kind.value,
                asset.duration_s,
                asset.model_dump_json(),
            ),
        )
        self.conn.commit()

    def get_asset(self, asset_id: str) -> Asset | None:
        row = self.conn.execute("SELECT json FROM assets WHERE asset_id = ?", (asset_id,)).fetchone()
        return Asset.model_validate_json(row["json"]) if row else None

    def list_assets(self, kind: str | None = None) -> list[Asset]:
        if kind:
            rows = self.conn.execute("SELECT json FROM assets WHERE kind = ? ORDER BY asset_id", (kind,))
        else:
            rows = self.conn.execute("SELECT json FROM assets ORDER BY asset_id")
        return [Asset.model_validate_json(r["json"]) for r in rows]

    # -- stage status --------------------------------------------------------

    def stage_state(self, asset_id: str, stage: str, params_hash: str = "") -> str:
        row = self.conn.execute(
            "SELECT state, params_hash FROM status WHERE asset_id = ? AND stage = ?",
            (asset_id, stage),
        ).fetchone()
        if row is None or row["params_hash"] != params_hash:
            return "pending"
        return row["state"]

    def set_stage(self, asset_id: str, stage: str, state: str, params_hash: str = "",
                  error: str | None = None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO status (asset_id, stage, state, params_hash, error, updated_at)"
            " VALUES (?,?,?,?,?, datetime('now'))",
            (asset_id, stage, state, params_hash, error),
        )
        self.conn.commit()

    # -- events --------------------------------------------------------------

    def replace_events(self, asset_id: str, analyzer: str, events: Iterable[Event]) -> None:
        self.conn.execute("DELETE FROM events WHERE asset_id = ? AND analyzer = ?", (asset_id, analyzer))
        self.conn.executemany(
            "INSERT INTO events (asset_id, analyzer, type, t0, t1, score, data) VALUES (?,?,?,?,?,?,?)",
            [(e.asset_id, e.analyzer, e.type, e.t0, e.t1, e.score, json.dumps(e.data)) for e in events],
        )
        self.conn.commit()

    def get_events(self, asset_id: str, analyzer: str | None = None, type_: str | None = None) -> list[Event]:
        q = "SELECT * FROM events WHERE asset_id = ?"
        args: list = [asset_id]
        if analyzer:
            q += " AND analyzer LIKE ?"
            args.append(f"{analyzer}%")
        if type_:
            q += " AND type = ?"
            args.append(type_)
        q += " ORDER BY t0"
        return [
            Event(
                asset_id=r["asset_id"],
                analyzer=r["analyzer"],
                type=r["type"],
                t0=r["t0"],
                t1=r["t1"],
                score=r["score"],
                data=json.loads(r["data"]),
            )
            for r in self.conn.execute(q, args)
        ]

    # -- clip logs -------------------------------------------------------------

    def upsert_clip_log(self, log: ClipLog) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO clip_logs (asset_id, model, prompt_version, json) VALUES (?,?,?,?)",
            (log.asset_id, log.model, log.prompt_version, log.model_dump_json()),
        )
        self.conn.commit()

    def get_clip_log(self, asset_id: str) -> ClipLog | None:
        row = self.conn.execute(
            "SELECT json FROM clip_logs WHERE asset_id = ? ORDER BY prompt_version DESC LIMIT 1",
            (asset_id,),
        ).fetchone()
        return ClipLog.model_validate_json(row["json"]) if row else None
