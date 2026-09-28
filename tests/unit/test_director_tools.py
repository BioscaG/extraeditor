"""The director's tools (§16.2): what the agent sees, and that it cannot corrupt a plan."""

from __future__ import annotations

import pytest

from montaje.director import tools
from montaje.director.tools import Session
from montaje.models.brief import Brief, DurationTarget, Format, MusicBrief
from montaje.workspace import Workspace


@pytest.fixture(scope="module")
def shoot(fixtures):
    """The synthetic shoot, generated once if missing."""
    import subprocess
    from pathlib import Path

    out = Path(__file__).resolve().parents[1] / "fixtures" / "shoot"
    if not (out / "shoot.json").exists():
        subprocess.run(
            ["python", str(out.parent / "make_footage.py"), "--out", str(out)],
            check=True,
        )
    return out


@pytest.fixture(scope="module")
def directed(tmp_path_factory, shoot):
    """An ingested, analyzed, planned project — built once for the whole module."""
    from montaje.analysis.runner import run_analysis
    from montaje.config import load_config
    from montaje.ingest.pipeline import run_ingest

    root = tmp_path_factory.mktemp("directed") / "proj"
    ws = Workspace(root)
    ws.create(Brief(
        name="proj", goal="test edit",
        format=Format(width=540, height=960, fps=30.0),
        duration=DurationTarget(target_s=32.0, tolerance_s=6.0),
        music=MusicBrief(path=shoot / "music" / "track.wav"),
        style="modern-festival",
    ))
    ws.add_source({"type": "folder", "path": str(shoot)})
    cfg = load_config(ws.root)
    run_ingest(ws, cfg)
    run_analysis(ws, cfg)
    session = Session(ws)
    tools.build_baseline_plan(session, title="Test Edit")
    return session


# -- survey ---------------------------------------------------------------------------


def test_overview_reports_assets_and_music(directed):
    ov = tools.project_overview(directed)
    assert ov["assets"]["count"] >= 8
    assert ov["assets"]["total_duration_s"] > 0
    assert ov["music"]["bpm"] == pytest.approx(120.0, rel=0.01)
    assert [s["role"] for s in ov["music"]["sections"]][0] == "intro"


def test_overview_includes_the_brief_and_available_styles(directed):
    ov = tools.project_overview(directed)
    assert ov["brief"]["goal"] == "test edit"
    assert "modern-festival" in ov["styles_available"]


def test_list_footage_summarizes_each_asset(directed):
    listing = tools.list_footage(directed)
    assert listing["count"] >= 8
    first = listing["assets"][0]
    for key in ("asset_id", "duration_s", "usable_spans", "speech_spans", "dominant_motion"):
        assert key in first


def test_list_footage_finds_the_speaking_clips(directed):
    listing = tools.list_footage(directed)
    speaking = [a for a in listing["assets"] if a["speech_spans"]]
    assert speaking, "the shoot contains two clips with dialogue"


def test_search_footage_ranks_by_score(directed):
    result = tools.search_footage(directed, limit=10)
    scores = [m["score"] for m in result["moments"]]
    assert scores == sorted(scores, reverse=True)


def test_search_footage_filters_on_speech(directed):
    with_speech = tools.search_footage(directed, needs_speech=True)["moments"]
    without = tools.search_footage(directed, needs_speech=False)["moments"]
    assert all(m["has_speech"] for m in with_speech)
    assert all(not m["has_speech"] for m in without)


def test_search_footage_filters_on_duration(directed):
    result = tools.search_footage(directed, min_duration_s=5.0)["moments"]
    assert all(m["duration_s"] >= 5.0 for m in result)


def test_get_events_returns_analyzer_data(directed):
    asset_id = tools.list_footage(directed, 1)["assets"][0]["asset_id"]
    events = tools.get_events(directed, asset_id, "motion")
    assert events["count"] > 0
    assert all(e["analyzer"].startswith("motion") for e in events["events"])


def test_get_events_on_an_unknown_asset_returns_an_error_not_a_raise(directed):
    """A tool that throws ends the agent's turn; errors must come back as data."""
    assert "error" in tools.get_events(directed, "a_nope")


def test_get_clip_log_reports_its_absence_usefully(directed):
    asset_id = tools.list_footage(directed, 1)["assets"][0]["asset_id"]
    result = tools.get_clip_log(directed, asset_id)
    assert "analyze --semantic" in result["error"]


def test_music_structure_exposes_the_grid_and_the_fit(directed):
    music = tools.music_structure(directed)
    assert music["beats_per_bar"] == 4
    assert len(music["first_downbeats"]) > 0
    assert music["fit"]["total_s"] > 0


def test_contact_sheet_writes_an_image(directed):
    from pathlib import Path

    asset_id = tools.list_footage(directed, 1)["assets"][0]["asset_id"]
    result = tools.contact_sheet(directed, asset_id, 0.0, 4.0)
    assert Path(result["path"]).exists()
    assert result["frames"] > 1


def test_contact_sheet_rejects_an_empty_range(directed):
    asset_id = tools.list_footage(directed, 1)["assets"][0]["asset_id"]
    assert "error" in tools.contact_sheet(directed, asset_id, 3.0, 3.0)


# -- library ---------------------------------------------------------------------------


def test_library_search_returns_intent_and_presets():
    result = tools.library_search(query="whip")
    assert result["count"] >= 1
    entry = result["components"][0]
    assert entry["ref"].startswith("transition.whip_pan@")
    assert entry["intent"]
    assert "aggressive" in entry["presets"]


def test_library_search_filters_by_kind():
    refs = [c["ref"] for c in tools.library_search(kind="text")["components"]]
    assert refs and all(r.startswith("text.") for r in refs)


def test_library_get_lists_alternatives_when_unknown():
    result = tools.library_get("transition.does_not_exist")
    assert "error" in result
    assert any(r.startswith("transition.") for r in result["available"])


def test_sfx_search_only_returns_licensed_files():
    result = tools.sfx_search()
    assert result["count"] > 0
    assert all(s["license"] for s in result["sfx"])


def test_sfx_search_filters_by_category():
    result = tools.sfx_search(category="riser")
    assert result["count"] >= 1
    assert all(s["category"] == "riser" for s in result["sfx"])


def test_get_style_exposes_pacing_targets_and_taste_notes():
    style = tools.get_style("modern-festival")
    assert style["pacing"]["avg_shot_beats"]["drop"] < style["pacing"]["avg_shot_beats"]["intro"]
    assert style["taste_notes"]


# -- plan ------------------------------------------------------------------------------


def test_baseline_plan_covers_every_music_section(directed):
    summary = tools.plan_get(directed, include_shots=False)
    assert summary["shot_count"] > 5
    assert len(summary["shots_per_section"]) >= 2


def test_plan_get_summary_omits_shots(directed):
    summary = tools.plan_get(directed, include_shots=False)
    assert "shots" not in summary
    assert summary["duration_s"] > 0


def test_plan_apply_bumps_the_version_and_returns_a_diff(directed):
    before = tools.plan_get(directed, include_shots=False)["version"]
    result = tools.plan_apply(directed, [{"op": "set_note", "note": "directed"}])
    assert result["version"] == before + 1
    assert "notes updated" in result["summary"]
    assert f"v{before} → v{before + 1}" in result["diff"]


def test_plan_apply_rejects_an_unknown_op_and_lists_the_valid_ones(directed):
    result = tools.plan_apply(directed, [{"op": "frobnicate"}])
    assert "unknown op" in result["error"]
    assert "insert_shot" in result["available_ops"]


def test_a_failed_batch_does_not_change_the_plan(directed):
    before = tools.plan_get(directed, include_shots=False)["version"]
    tools.plan_apply(directed, [
        {"op": "set_note", "note": "should not persist"},
        {"op": "remove_shot", "id": "does_not_exist"},
    ])
    assert tools.plan_get(directed, include_shots=False)["version"] == before


def test_plan_validate_separates_errors_from_warnings(directed):
    result = tools.plan_validate(directed)
    assert isinstance(result["errors"], list)
    assert isinstance(result["warnings"], list)
    assert all("code" in w for w in result["warnings"])


def test_rhythm_report_exposes_what_the_critic_cannot_see(directed):
    report = tools.rhythm_report(directed)
    assert 0.0 <= report["on_grid_ratio"] <= 1.0
    assert report["offset_histogram"]
    assert "# Rhythm report" in report["markdown"]


def test_rhythm_report_measures_pacing_against_the_style(directed):
    report = tools.rhythm_report(directed)
    assert report["sections"]
    assert report["sections"][0]["target_shot_beats"] is not None


def test_write_summary_produces_a_readable_file(directed):
    from pathlib import Path

    result = tools.write_summary(directed)
    text = Path(result["path"]).read_text()
    assert "## Sections" in text
    # Every shot line carries its intent, which is what makes the summary readable.
    assert "—" in text


def test_ask_user_returns_the_question_for_relay():
    result = tools.ask_user("Which concept?", ["A", "B"])
    assert result["question"] == "Which concept?"
    assert result["options"] == ["A", "B"]


def test_tool_catalog_matches_the_module(directed):
    """The catalog is what documents the surface; it must not drift from the code."""
    names = {entry["name"] for entry in tools.tool_catalog()}
    for name in names:
        assert hasattr(tools, name), f"{name} is catalogued but not implemented"


def test_every_public_tool_is_catalogued():
    catalogued = {entry["name"] for entry in tools.tool_catalog()}
    expected = {
        "project_overview", "list_footage", "search_footage", "get_events",
        "get_clip_log", "contact_sheet", "music_structure", "library_search",
        "library_get", "sfx_search", "get_style", "build_baseline_plan", "plan_get",
        "plan_apply", "plan_validate", "rhythm_report", "render_preview",
        "write_summary", "ask_user",
    }
    assert expected == catalogued
