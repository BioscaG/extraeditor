from __future__ import annotations

import json

import pytest
import yaml

from montaje.config import load_config
from montaje.models.brief import Brief, Format
from montaje.workspace import Workspace, atomic_write_json, atomic_write_text


def test_create_makes_all_directories(project):
    for d in (project.media_dir, project.cache_dir, project.plans_dir,
              project.renders_dir, project.exports_dir, project.logs_dir):
        assert d.is_dir()


def test_create_is_idempotent_and_preserves_edits(project):
    project.project_yaml.write_text(yaml.safe_dump({"name": "proj", "goal": "keep me"}))
    project.create(Brief(name="proj"))
    assert project.load_brief().goal == "keep me"


def test_load_brief_roundtrips(project):
    brief = Brief(name="proj", goal="aftermovie", language="es")
    project.project_yaml.write_text(yaml.safe_dump(brief.model_dump(mode="json", exclude_none=True)))
    got = project.load_brief()
    assert got.goal == "aftermovie"
    assert got.language == "es"


def test_add_source_deduplicates(project):
    project.add_source({"type": "folder", "path": "/tmp/a"})
    project.add_source({"type": "folder", "path": "/tmp/a"})
    assert len(project.load_sources()) == 1


def test_find_by_path(project):
    assert Workspace.find(str(project.root)).slug == project.slug


def test_find_raises_for_unknown_project(tmp_path):
    with pytest.raises(FileNotFoundError):
        Workspace.find("does-not-exist", projects_root=tmp_path)


def test_cache_paths_are_namespaced_by_asset(project):
    assert project.proxy_path("a_1") != project.proxy_path("a_2")
    assert project.proxy_path("a_1").parent == project.asset_cache("a_1")


def test_analysis_path_includes_analyzer_version(project):
    p = project.analysis_path("a_1", "occlusion", 2)
    assert p.name == "occlusion@2.json"


def test_atomic_write_leaves_no_temp_files(tmp_path):
    target = tmp_path / "out" / "f.json"
    atomic_write_json(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}
    assert [p.name for p in target.parent.iterdir()] == ["f.json"]


def test_atomic_write_overwrites_existing(tmp_path):
    target = tmp_path / "f.txt"
    atomic_write_text(target, "old")
    atomic_write_text(target, "new")
    assert target.read_text() == "new"


def test_format_accepts_resolution_string():
    f = Format.model_validate({"aspect": "9:16", "resolution": "1080x1920", "fps": 30})
    assert (f.width, f.height) == (1080, 1920)


def test_format_rejects_unknown_aspect():
    with pytest.raises(ValueError):
        Format(aspect="21:9")


def test_config_defaults_load_from_repo_yaml():
    cfg = load_config()
    assert cfg.proxy.short_edge == 540
    assert cfg.rails.loudness_lufs == -14.0
    assert cfg.director.model_final == "claude-opus-5-5"


def test_project_config_overrides_defaults_deeply(project):
    (project.root / "config.yaml").write_text(yaml.safe_dump({"proxy": {"short_edge": 720}}))
    cfg = load_config(project.root)
    assert cfg.proxy.short_edge == 720
    # Untouched keys in the same section must survive the merge.
    assert cfg.proxy.fps == 30.0
