"""The component contract (§13.2), checked against the registry's real metadata."""

from __future__ import annotations

import pytest

from montaje.library.registry import find, load_components, refresh, stable_refs
from montaje.models.library import ComponentKind, ComponentStatus


@pytest.fixture(scope="module", autouse=True)
def fresh_registry():
    refresh()
    yield
    refresh()


def test_every_component_has_an_intent():
    """The director chooses from these; an unexplained component cannot be chosen well."""
    for component in load_components():
        assert component.intent, component.ref


def test_every_intent_carries_a_caveat_as_well_as_a_use():
    """§13.3: a component that only advertises itself gets overused, and overuse is the
    listed amateur tell (§28). So an intent must say more than what it is for.

    Tested structurally — at least two sentences — rather than by keyword: the first
    attempt matched on a word list and failed zoom_punch, whose caveat ("reserve it for
    the beats that actually matter") simply used different words.
    """
    for component in load_components():
        if component.id == "transition.hard_cut":
            continue  # the default needs no caveat
        sentences = [s for s in component.intent.replace(";", ".").split(".") if s.strip()]
        assert len(sentences) >= 2, (component.ref, component.intent)


def test_versions_are_semver():
    for component in load_components():
        parts = component.version.split(".")
        assert len(parts) == 3 and all(p.isdigit() for p in parts), component.ref


def test_ids_are_namespaced_by_kind():
    for component in load_components():
        assert component.id.startswith(f"{component.kind.value}."), component.ref


def test_durations_are_orderly():
    for component in load_components():
        assert component.duration.min_frames <= component.duration.max_frames, component.ref


def test_transitions_declare_a_beat_anchor():
    """A transition has to know which of its instants lands on the beat (§13.2)."""
    for component in load_components():
        if component.kind == ComponentKind.TRANSITION:
            assert component.beat_anchor, component.ref


def test_every_component_supports_the_vertical_aspect():
    """The first real use case is a 9:16 aftermovie."""
    for component in load_components():
        assert "9:16" in component.aspect_ratios, component.ref


def test_presets_are_named_not_numbered():
    for component in load_components():
        for name in component.presets:
            assert not name.isdigit(), (component.ref, name)


def test_stable_components_are_what_validation_accepts():
    refs = stable_refs()
    assert refs
    for component in load_components():
        if component.status == ComponentStatus.STABLE:
            assert component.ref in refs


def test_the_catalogue_covers_every_kind_an_aftermovie_needs():
    kinds = {c.kind for c in load_components()}
    assert ComponentKind.TRANSITION in kinds
    assert ComponentKind.TEXT in kinds
    assert ComponentKind.SHOT_FX in kinds
    assert ComponentKind.FINISHING in kinds


def test_the_hard_cut_is_zero_length():
    """It exists so "cut" is a countable decision, not an absence."""
    cut = find("transition.hard_cut")
    assert cut is not None
    assert cut.duration.max_frames == 0


def test_the_style_only_references_components_that_exist():
    """A style is data and can name anything; a missing reference must be caught."""
    from montaje.styles.registry import load_style

    style = load_style("modern-festival")
    assert style is not None
    for ref in style.transitions.palette:
        assert find(ref) is not None, ref
    for ref in (style.text.captions, style.text.titles):
        if ref:
            assert find(ref) is not None, ref


def test_components_that_suggest_sfx_name_a_licensed_category():
    from montaje.library.registry import load_sfx

    categories = {s.category for s in load_sfx()}
    for component in load_components():
        if component.sfx is None:
            continue
        category = component.sfx.default.split(".")[0]
        assert category in categories, (component.ref, category)


def test_finishing_is_not_also_applied_by_ffmpeg():
    """Grain per shot plus grain over the composition would double it, and grain under
    the captions but not over them reads as two images composited together."""
    from montaje.color.grade import grade_from_style
    from montaje.models.style import ColorPolicy

    grade = grade_from_style(ColorPolicy(grain=0.3, vignette=0.4))
    assert grade.grain == 0.0
    assert grade.vignette == 0.0
    assert grade.to_filter() is None


def test_the_style_carries_finishing_parameters():
    from montaje.styles.registry import load_style

    policy = load_style("modern-festival").color
    assert policy.halation > 0
    assert 0 < policy.grain < 0.3, "visible grain is the clearest colourist tell"


def test_finishing_props_are_omitted_when_the_style_asks_for_none():
    from montaje.models.style import ColorPolicy, Style
    from montaje.render.pipeline import _finishing_for

    assert _finishing_for(None) is None
    assert _finishing_for(Style(name="n", color=ColorPolicy())) is None
    assert _finishing_for(Style(name="n", color=ColorPolicy(halation=0.3))) is not None
