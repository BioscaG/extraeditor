"""Content-addressed asset ids (§6).

asset_id = sha256(file size ‖ first 8 MB ‖ last 8 MB ‖ duration), truncated to 16 hex
chars with an `a_` prefix. Reading only the head and tail keeps hashing fast on
multi-GB originals while still catching truncation and re-encodes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CHUNK = 8 * 1024 * 1024


def compute_asset_id(path: Path, duration_s: float) -> str:
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(CHUNK))
        if size > 2 * CHUNK:
            f.seek(size - CHUNK)
            h.update(f.read(CHUNK))
    h.update(f"{duration_s:.3f}".encode())
    return "a_" + h.hexdigest()[:16]


def params_hash(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:12]
