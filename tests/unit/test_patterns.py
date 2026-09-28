"""Pattern mining and conventions (§11–12).

The headline claim under test is §2.5: a recurring shooting habit must be **discovered**
from meaning-free events, with nothing in the code that knows what the habit is.
"""

from __future__ import annotations

import pytest

from montaje.index.conventions import (
    apply_conventions,
    confirmed,
    find_motifs,
    load_conventions,
    merge_proposals,
    save_conventions,
    usable_after_conventions,
)
from montaje.index.patterns import (
    MAX_SHARE,
    MIN_CLIP_COUNT,
    Pattern,
    mine_patterns,
    propose_conventions,
    summarize,
)
from montaje.models.conventions import Convention, ConventionDetection, ConventionStatus
from montaje.models.events import Event
from tests.unit.conftest_plan import make_asset, make_plan, make_shot


def bookended_events(asset_id: str, duration: float = 7.0, cover: float = 0.9) -> list[Event]:
    """Occlusion events for a clip obscured at both ends. Meaning-free by construction."""
    return [
        Event(asset_id=asset_id, analyzer="occlusion@1", type="occluded",
              t0=0.0, t1=cover),
        Event(asset_id=asset_id, analyzer="occlusion@1", type="reveal",
              t0=cover - 0.2, t1=cover + 0.2),
        Event(asset_id=asset_id, analyzer="occlusion@1", type="occluded",
              t0=duration - cover, t1=duration),
        Event(asset_id=asset_id, analyzer="occlusion@1", type="cover",
              t0=duration - cover - 0.2, t1=duration - cover + 0.2),
    ]


def dense_events(asset_id: str, duration: float = 7.0) -> list[Event]:
    """Per-second analyzer output: present at both edges of every clip by construction."""
    return [
        Event(asset_id=asset_id, analyzer="motion@1", type="motion",
              t0=float(i), t1=float(i + 1), data={"direction": "static"})
        for i in range(int(duration))
    ]


def shoot(bookended: int = 4, plain: int = 8, duration: float = 7.0):
    """A shoot where some clips carry the motif. Asset ids are deliberately neutral so a
    test asserting the miner names nothing cannot be fooled by the fixture's own naming."""
    assets, events = {}, {}
    for i in range(bookended):
        aid = f"a_m{i}"
        assets[aid] = make_asset(aid, duration)
        events[aid] = bookended_events(aid, duration) + dense_events(aid, duration)
    for i in range(plain):
        aid = f"a_plain{i}"
        assets[aid] = make_asset(aid, duration)
        events[aid] = dense_events(aid, duration)
    return assets, events


# -- discovery ---------------------------------------------------------------------------


def test_the_recurring_motif_is_discovered():
    """M3's headline criterion: found with no project-specific code."""
    assets, events = shoot()
    patterns = mine_patterns(assets, events)
    assert patterns
    top = patterns[0]
    assert top.kind == "bookend"
    assert top.event_type == "occlusion.reveal"
    assert top.paired_event_type == "occlusion.cover"


def test_the_motif_identifies_exactly_the_clips_that_have_it():
    assets, events = shoot(bookended=4, plain=8)
    top = mine_patterns(assets, events)[0]
    assert set(top.assets) == {f"a_m{i}" for i in range(4)}


def test_nothing_in_the_output_names_the_habit():
    """§2.5: the miner reports what it measured, not what it means."""
    assets, events = shoot()
    text = summarize(mine_patterns(assets, events)).lower()
    for word in ("hand", "lens", "vlog", "festival"):
        assert word not in text, word


def test_per_second_analyzers_do_not_become_motifs():
    """They emit something at every clip's edges, which says nothing about the footage."""
    assets, events = shoot(bookended=0, plain=8)
    assert mine_patterns(assets, events) == []


def test_dense_events_are_excluded_by_measured_coverage():
    """Measured, not listed, so analyzers that do not exist yet are handled too."""
    assets = {"a_1": make_asset("a_1", 10.0)}
    events = {"a_1": [
        Event(asset_id="a_1", analyzer="future@1", type="blanket", t0=0.0, t1=10.0),
    ]}
    patterns = mine_patterns(assets, events)
    assert not any("future" in p.event_type for p in patterns)


def test_a_motif_in_every_clip_is_not_selective_enough():
    """Something present everywhere describes the analyzer, not a shooting habit."""
    assets, events = shoot(bookended=8, plain=0)
    assert mine_patterns(assets, events) == []


def test_a_motif_in_too_few_clips_is_ignored():
    assets, events = shoot(bookended=2, plain=10)
    assert not any(p.event_type == "occlusion.reveal" for p in mine_patterns(assets, events))


def test_duplicate_findings_are_collapsed():
    """A span and its own edges identify the same clips: that is one finding, not four."""
    assets, events = shoot()
    patterns = mine_patterns(assets, events)
    clip_sets = [frozenset(p.assets) for p in patterns]
    assert len(clip_sets) == len(set(clip_sets))


def test_edge_events_are_preferred_over_spans():
    """A convention trims to an edge; there is nothing to trim to mid-span."""
    assets, events = shoot()
    top = mine_patterns(assets, events)[0]
    assert top.event_type.endswith("reveal")
    assert (top.paired_event_type or "").endswith("cover")


def test_empty_footage_yields_no_patterns():
    assert mine_patterns({}, {}) == []


def test_selectivity_peaks_at_half_the_footage():
    half = Pattern(id="x", kind="start_motif", event_type="a.b", count=6, total_clips=12)
    most = Pattern(id="y", kind="start_motif", event_type="a.b", count=11, total_clips=12)
    assert half.selectivity > most.selectivity


def test_significance_requires_both_a_count_and_a_share():
    thin = Pattern(id="x", kind="start_motif", event_type="a.b",
                   count=MIN_CLIP_COUNT - 1, total_clips=4)
    universal = Pattern(id="y", kind="start_motif", event_type="a.b",
                        count=100, total_clips=100)
    assert not thin.is_significant
    assert not universal.is_significant
    assert universal.share > MAX_SHARE


# -- proposals -----------------------------------------------------------------------------


def test_proposals_start_unconfirmed():
    """Nothing affects an edit until the user says so (§12)."""
    assets, events = shoot()
    for convention in propose_conventions(mine_patterns(assets, events)):
        assert convention.status == ConventionStatus.PROPOSED


def test_a_bookend_proposal_names_both_edges():
    assets, events = shoot()
    convention = propose_conventions(mine_patterns(assets, events))[0]
    assert convention.detection.start_event == "occlusion.reveal"
    assert convention.detection.end_event == "occlusion.cover"
    assert convention.treatment.trim_outside is True


def test_a_proposal_states_that_the_meaning_is_the_users_call():
    assets, events = shoot()
    convention = propose_conventions(mine_patterns(assets, events))[0]
    assert "for you to say" in convention.description
    assert str(convention.evidence.count) in convention.description


# -- persistence ----------------------------------------------------------------------------


def test_conventions_round_trip(project):
    original = [Convention(id="c1", description="d",
                           detection=ConventionDetection(start_event="occlusion.reveal"))]
    save_conventions(project, original)
    assert load_conventions(project)[0].id == "c1"


def test_no_conventions_file_means_no_conventions(project):
    assert load_conventions(project) == []


def test_merging_preserves_a_decision_already_made():
    """Re-running analysis must not silently un-confirm a decision."""
    existing = [Convention(id="c1", description="old", status=ConventionStatus.CONFIRMED)]
    proposed = [Convention(id="c1", description="new", status=ConventionStatus.PROPOSED)]
    merged = merge_proposals(existing, proposed)
    assert merged[0].status == ConventionStatus.CONFIRMED
    # The evidence is refreshed even though the verdict stands.
    assert merged[0].description == "new"


def test_merging_preserves_a_rejection():
    """A question that has been answered must not be re-asked."""
    existing = [Convention(id="c1", description="d", status=ConventionStatus.REJECTED)]
    merged = merge_proposals(existing, [Convention(id="c1", description="d")])
    assert merged[0].status == ConventionStatus.REJECTED


def test_merging_adds_new_proposals():
    merged = merge_proposals(
        [Convention(id="c1", description="a")],
        [Convention(id="c2", description="b")],
    )
    assert {c.id for c in merged} == {"c1", "c2"}


# -- application ------------------------------------------------------------------------------


def convention(status=ConventionStatus.CONFIRMED) -> Convention:
    return Convention(
        id="bookend", description="d", status=status,
        detection=ConventionDetection(start_event="occlusion.reveal",
                                      end_event="occlusion.cover", max_offset_s=2.0),
    )


def test_motifs_are_found_at_the_right_instants():
    events = {"a_1": bookended_events("a_1", 7.0, 0.9)}
    motif = find_motifs(convention(), events)[0]
    # Usable content begins where the reveal ends and stops where the cover begins.
    assert motif.start_edge_s == pytest.approx(1.1, abs=0.01)
    assert motif.end_edge_s == pytest.approx(5.9, abs=0.01)


def test_a_clip_without_the_motif_is_not_matched():
    assert find_motifs(convention(), {"a_1": dense_events("a_1")}) == []


def test_a_confirmed_convention_trims_shots_into_the_motif():
    plan = make_plan([make_shot("s001", src_in=0.0, src_out=7.0, timeline_in=0)])
    report = apply_conventions(plan, [convention()], {"a_1": bookended_events("a_1", 7.0)})
    assert report.trimmed == ["s001"]
    assert plan.shots[0].src_in > 1.0
    assert plan.shots[0].src_out < 6.0


def test_an_unconfirmed_convention_changes_nothing():
    """This is the whole point: discovery does not imply application."""
    plan = make_plan([make_shot("s001", src_in=0.0, src_out=7.0, timeline_in=0)])
    report = apply_conventions(
        plan, [convention(ConventionStatus.PROPOSED)], {"a_1": bookended_events("a_1", 7.0)}
    )
    assert not report.changed
    assert plan.shots[0].src_in == 0.0


def test_a_rejected_convention_changes_nothing():
    plan = make_plan([make_shot("s001", src_in=0.0, src_out=7.0, timeline_in=0)])
    report = apply_conventions(
        plan, [convention(ConventionStatus.REJECTED)], {"a_1": bookended_events("a_1", 7.0)}
    )
    assert not report.changed


def test_a_shot_entirely_inside_the_motif_is_dropped():
    """None of it was footage the shooter meant to be seen."""
    plan = make_plan([
        make_shot("s001", src_in=0.0, src_out=0.8, timeline_in=0),
        make_shot("s002", src_in=2.0, src_out=4.0, timeline_in=24),
    ])
    report = apply_conventions(plan, [convention()], {"a_1": bookended_events("a_1", 7.0)})
    assert report.dropped == ["s001"]
    assert [s.id for s in plan.shots] == ["s002"]


def test_a_shot_already_inside_the_usable_range_is_untouched():
    plan = make_plan([make_shot("s001", src_in=2.0, src_out=4.0, timeline_in=0)])
    report = apply_conventions(plan, [convention()], {"a_1": bookended_events("a_1", 7.0)})
    assert not report.changed
    assert (plan.shots[0].src_in, plan.shots[0].src_out) == (2.0, 4.0)


def test_shots_from_other_assets_are_untouched():
    plan = make_plan([make_shot("s001", asset="a_other", src_in=0.0, src_out=7.0,
                                timeline_in=0)])
    report = apply_conventions(plan, [convention()], {"a_1": bookended_events("a_1", 7.0)})
    assert not report.changed


def test_usable_range_excludes_the_motif():
    lo, hi = usable_after_conventions(
        "a_1", 7.0, [convention()], bookended_events("a_1", 7.0)
    )
    assert lo == pytest.approx(1.1, abs=0.01)
    assert hi == pytest.approx(5.9, abs=0.01)


def test_usable_range_is_the_whole_clip_without_a_convention():
    assert usable_after_conventions("a_1", 7.0, [], []) == (0.0, 7.0)


def test_usable_range_falls_back_when_the_motif_would_empty_the_clip():
    """A misdetected motif must not leave nothing usable."""
    events = [
        Event(asset_id="a_1", analyzer="occlusion@1", type="reveal", t0=5.8, t1=6.0),
        Event(asset_id="a_1", analyzer="occlusion@1", type="cover", t0=0.1, t1=0.3),
    ]
    assert usable_after_conventions("a_1", 7.0, [convention()], events) == (0.0, 7.0)


def test_confirmed_filters_by_status():
    conventions = [
        convention(ConventionStatus.CONFIRMED),
        convention(ConventionStatus.PROPOSED),
        convention(ConventionStatus.REJECTED),
    ]
    assert len(confirmed(conventions)) == 1


# -- the planner honours them -------------------------------------------------------------------


def test_the_planner_never_proposes_a_shot_inside_a_motif():
    """Trimming afterwards would waste the budget the shot was given."""
    from montaje.plan.build import BuildInputs, collect_candidates

    asset_id = "a_1"
    events = bookended_events(asset_id, 7.0) + [
        Event(asset_id=asset_id, analyzer="quality@1", type="usable", t0=0.0, t1=7.0),
    ]
    inputs = BuildInputs(
        assets={asset_id: make_asset(asset_id, 7.0)},
        events={asset_id: events},
        conventions=[convention()],
    )
    for candidate in collect_candidates(inputs):
        assert candidate.t0 >= 1.05
        assert candidate.t1 <= 5.95


def test_without_conventions_the_whole_usable_span_is_offered():
    from montaje.plan.build import BuildInputs, collect_candidates

    events = bookended_events("a_1", 7.0) + [
        Event(asset_id="a_1", analyzer="quality@1", type="usable", t0=0.0, t1=7.0),
    ]
    inputs = BuildInputs(assets={"a_1": make_asset("a_1", 7.0)}, events={"a_1": events})
    assert any(c.t0 < 0.5 for c in collect_candidates(inputs))


# -- ordering in the render pipeline --------------------------------------------------------


def test_convention_trimming_runs_before_the_rails(project, monkeypatch):
    """Trimming after relayout shortens laid-out shots and opens black-frame gaps.

    The rule the pipeline follows: source-range edits before the rails, timeline-position
    repairs after. This pins the convention trim on the correct side of it.
    """
    from montaje.render import pipeline

    calls: list[str] = []
    real_conventions = pipeline.apply_conventions
    real_rails = pipeline.apply_rails

    def spy_conventions(*args, **kwargs):
        calls.append("conventions")
        return real_conventions(*args, **kwargs)

    def spy_rails(*args, **kwargs):
        calls.append("rails")
        return real_rails(*args, **kwargs)

    monkeypatch.setattr(pipeline, "apply_conventions", spy_conventions)
    monkeypatch.setattr(pipeline, "apply_rails", spy_rails)
    monkeypatch.setattr(pipeline, "apply_spotting", lambda plan, **kw: plan)

    from montaje.config import load_config
    from montaje.index.store import Store

    with Store(project.db_path) as store:
        store.upsert_asset(make_asset("a_1", 30.0))
    pipeline.render(project, make_plan(), load_config(), skip_video=True)

    assert calls.index("conventions") < calls.index("rails")
