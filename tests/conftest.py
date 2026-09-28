from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "generated"


@pytest.fixture(scope="session")
def fixtures() -> Path:
    """Generate the synthetic fixture set once per session if missing."""
    if not (FIXTURE_DIR / "truth.json").exists():
        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not available")
        subprocess.run(
            ["python", str(Path(__file__).parent / "fixtures" / "make.py"), "--out", str(FIXTURE_DIR)],
            check=True,
        )
    return FIXTURE_DIR


@pytest.fixture(scope="session")
def truth(fixtures: Path) -> dict:
    return json.loads((fixtures / "truth.json").read_text())


@pytest.fixture()
def project(tmp_path: Path):
    """An empty workspace with an initialized store."""
    from montaje.models.brief import Brief
    from montaje.workspace import Workspace

    ws = Workspace(tmp_path / "proj")
    ws.create(Brief(name="proj"))
    return ws


@pytest.fixture()
def ingested(project, fixtures):
    """Workspace with the fixture directory ingested."""
    from montaje.config import load_config
    from montaje.ingest.pipeline import run_ingest

    project.add_source({"type": "folder", "path": str(fixtures)})
    cfg = load_config(project.root)
    result = run_ingest(project, cfg)
    assert not result["failures"], result["failures"]
    return project
