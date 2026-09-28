"""Exports (§19.4): the edit must stay editable, not just renderable."""

from __future__ import annotations

import pytest

from montaje.models.editplan import ComponentRef, Duration, Section, SnapKind
from montaje.models.events import Event
from montaje.render.exports import (
    Cue,
    _similar,
    _srt_time,
    build_otio,
    cues_from_plan,
    export_fcpxml,
    export_otio,
    export_srt,
    fcpxml_tree,
    write_srt,
)
from tests.unit.conftest_plan import make_asset, make_plan, make_shot


def words(asset: str, *spans: tuple[str, float, float]) -> list[Event]:
    return [
        Event(asset_id=asset, analyzer="asr@1", type="word", t0=t0, t1=t1,
              data={"text": text})
        for text, t0, t1 in spans
    ]


@pytest.fixture()
def assets():
    return {"a_1": make_asset("a_1", 30.0)}


# -- SRT ------------------------------------------------------------------------------


def test_srt_time_formatting():
    assert _srt_time(0.0) == "00:00:00,000"
    assert _srt_time(1.267) == "00:00:01,267"
    assert _srt_time(3661.5) == "01:01:01,500"


def test_srt_time_clamps_negatives():
    assert _srt_time(-1.0) == "00:00:00,000"


def test_cues_are_in_timeline_time_not_source_time():
    """A shot at frame 60 means its words appear at 2s, not at their source time."""
    plan = make_plan([make_shot("s001", src_in=5.0, src_out=7.0, timeline_in=60)])
    events = {"a_1": words("a_1", ("hello", 5.2, 5.6), ("there", 5.7, 6.1))}
    cues = cues_from_plan(plan, events)
    assert cues
    assert cues[0].start_s == pytest.approx(2.2, abs=0.01)


def test_cues_group_words_into_readable_lines():
    """One cue per word is unreadable as a subtitle, even though it is right on screen."""
    plan = make_plan([make_shot("s001", src_in=0.0, src_out=6.0, timeline_in=0)])
    spans = [(f"word{i}", i * 0.4, i * 0.4 + 0.35) for i in range(14)]
    cues = cues_from_plan(plan, {"a_1": words("a_1", *spans)}, max_chars=20)
    assert 1 < len(cues) < 14
    assert all(len(c.text) <= 26 for c in cues)


def test_cues_have_a_minimum_readable_duration():
    plan = make_plan([make_shot("s001", src_in=0.0, src_out=6.0, timeline_in=0)])
    cues = cues_from_plan(plan, {"a_1": words("a_1", ("hi", 0.0, 0.1))},
                          min_duration_s=0.8)
    assert cues[0].end_s - cues[0].start_s >= 0.79


def test_cues_never_overlap():
    plan = make_plan([
        make_shot("s001", src_in=0.0, src_out=2.0, timeline_in=0),
        make_shot("s002", src_in=5.0, src_out=7.0, timeline_in=60),
    ])
    events = {"a_1": words("a_1", ("one", 0.1, 0.4), ("two", 5.1, 5.4))}
    cues = cues_from_plan(plan, events, min_duration_s=3.0)
    for a, b in zip(cues, cues[1:], strict=False):
        assert a.end_s <= b.start_s


def test_overlapping_duplicate_cues_are_collapsed():
    """Cue padding can make one line's cue overlap the same line's cue on the next shot.

    That overlap is an artifact, not something the viewer hears twice, so it collapses.
    """
    plan = make_plan([
        make_shot("s001", src_in=0.0, src_out=1.5, timeline_in=0),
        make_shot("s002", src_in=0.0, src_out=1.5, timeline_in=20),
    ])
    events = {"a_1": words("a_1", ("same", 0.1, 0.5), ("line", 0.6, 1.0))}
    cues = cues_from_plan(plan, events, min_duration_s=2.0)
    assert len(cues) == 1


def test_a_genuinely_repeated_line_is_subtitled_twice():
    """If the same audio plays twice in the edit, the viewer hears it twice."""
    plan = make_plan([
        make_shot("s001", src_in=0.0, src_out=1.5, timeline_in=0),
        make_shot("s002", src_in=0.0, src_out=1.5, timeline_in=150),
    ])
    events = {"a_1": words("a_1", ("same", 0.1, 0.5), ("line", 0.6, 1.0))}
    assert len(cues_from_plan(plan, events)) == 2


def test_similarity_detects_truncated_repeats():
    assert _similar("this is a line of dialogue", "this is a line")
    assert _similar("dialogue for the montage test", "for the montage test")
    assert not _similar("completely different words here", "nothing alike at all")


def test_no_words_means_no_cues():
    plan = make_plan([make_shot("s001")])
    assert cues_from_plan(plan, {}) == []


def test_write_srt_numbers_cues_from_one(tmp_path):
    dest = write_srt([Cue(0.0, 1.0, "first"), Cue(1.0, 2.0, "second")], tmp_path / "c.srt")
    text = dest.read_text()
    assert text.startswith("1\n00:00:00,000 --> 00:00:01,000\nfirst")
    assert "\n2\n" in text


def test_export_srt_writes_a_file(tmp_path):
    plan = make_plan([make_shot("s001", src_in=0.0, src_out=2.0, timeline_in=0)])
    events = {"a_1": words("a_1", ("hello", 0.2, 0.6))}
    assert export_srt(plan, events, tmp_path / "c.srt").exists()


# -- OTIO -------------------------------------------------------------------------------


def test_otio_references_the_originals(assets):
    """§19.4: a timeline referencing intermediates has the colour baked in."""
    timeline = build_otio(make_plan(), assets)
    import opentimelineio as otio

    clips = [c for c in timeline.tracks[0] if isinstance(c, otio.schema.Clip)]
    assert clips
    for clip in clips:
        assert clip.media_reference.target_url.endswith("a_1.mp4")


def test_otio_duration_matches_the_plan(assets):
    plan = make_plan()
    timeline = build_otio(plan, assets)
    expected = plan.timeline_end_frame() / plan.format.fps
    assert timeline.duration().to_seconds() == pytest.approx(expected, abs=0.05)


def test_otio_preserves_a_transition_annotation(assets):
    """The plan annotates a butt cut; OTIO's Transition is the same model."""
    import opentimelineio as otio

    plan = make_plan([
        make_shot("s001", src_in=0.0, src_out=2.0, timeline_in=0),
        make_shot("s002", src_in=5.0, src_out=7.0, timeline_in=60,
                  transition=ComponentRef(id="transition.whip_pan@1.0.0",
                                          duration=Duration(frames=8))),
    ])
    timeline = build_otio(plan, assets)
    transitions = [i for i in timeline.tracks[0] if isinstance(i, otio.schema.Transition)]
    assert [t.name for t in transitions] == ["transition.whip_pan@1.0.0"]
    assert transitions[0].in_offset.value == 4


def test_a_transition_does_not_change_the_exported_duration(assets):
    """In OTIO a Transition borrows from its neighbours; it must not add length."""
    plain = make_plan([
        make_shot("s001", src_in=0.0, src_out=2.0, timeline_in=0),
        make_shot("s002", src_in=5.0, src_out=7.0, timeline_in=60),
    ])
    with_transition = make_plan([
        make_shot("s001", src_in=0.0, src_out=2.0, timeline_in=0),
        make_shot("s002", src_in=5.0, src_out=7.0, timeline_in=60,
                  transition=ComponentRef(id="transition.whip_pan@1.0.0",
                                          duration=Duration(frames=8))),
    ])
    assert build_otio(plain, assets).duration() == build_otio(with_transition, assets).duration()


def test_otio_carries_intent_as_a_marker(assets):
    import opentimelineio as otio

    plan = make_plan([make_shot("s001", intent="Open on the crowd surge")])
    # The timeline must stay bound: OTIO's bindings free a track's children along with
    # the timeline that owns them, so `build_otio(...).tracks[0]` yields nothing.
    timeline = build_otio(plan, assets)
    clips = [c for c in timeline.tracks[0] if isinstance(c, otio.schema.Clip)]
    assert clips[0].markers[0].name == "Open on the crowd surge"


def test_otio_carries_shot_metadata(assets):
    import opentimelineio as otio

    plan = make_plan([make_shot("s001", section="intro")])
    plan.concept.sections = [Section(id="intro", from_frame=0, to_frame=60)]
    timeline = build_otio(plan, assets)
    clips = [c for c in timeline.tracks[0] if isinstance(c, otio.schema.Clip)]
    meta = clips[0].metadata["montaje"]
    assert meta["shot_id"] == "s001"
    assert meta["section"] == "intro"


def test_otio_marks_the_sections(assets):
    plan = make_plan(sections=[
        Section(id="intro", from_frame=0, to_frame=60, music_section="intro"),
        Section(id="drop", from_frame=60, to_frame=120, music_section="drop"),
    ])
    timeline = build_otio(plan, assets)
    names = [m.name for m in timeline.tracks[0].markers]
    assert names == ["intro: intro", "drop: drop"]


def test_otio_preserves_a_gap(assets):
    """A gap is real; sliding clips earlier to hide it would misrepresent the edit."""
    import opentimelineio as otio

    plan = make_plan([
        make_shot("s001", src_in=0.0, src_out=2.0, timeline_in=0),
        make_shot("s002", src_in=5.0, src_out=7.0, timeline_in=90),
    ])
    timeline = build_otio(plan, assets)
    items = list(timeline.tracks[0])
    assert any(isinstance(i, otio.schema.Gap) for i in items)


def test_otio_round_trips_through_the_library(tmp_path, assets):
    """Writing the schema by hand produced files OTIO's own reader rejected."""
    import opentimelineio as otio

    dest = export_otio(make_plan(), assets, tmp_path / "t.otio")
    reread = otio.adapters.read_from_file(str(dest))
    original = build_otio(make_plan(), assets)
    assert len(list(reread.tracks[0])) == len(list(original.tracks[0]))


# -- FCPXML ------------------------------------------------------------------------------


def test_fcpxml_uses_rational_times(assets):
    """A decimal frame time is how an import lands one frame off."""
    tree = fcpxml_tree(make_plan(), assets)
    fmt = tree.find("resources/format")
    assert fmt.get("frameDuration") == "1/30s"
    clip = tree.find("library/event/project/sequence/spine/asset-clip")
    assert clip.get("duration").endswith("/30s")


def test_fcpxml_handles_a_fractional_frame_rate(assets):
    plan = make_plan(fps=29.97)
    tree = fcpxml_tree(plan, assets)
    assert tree.find("resources/format").get("frameDuration") == "1001/30000s"


def test_fcpxml_references_originals_as_file_urls(assets):
    tree = fcpxml_tree(make_plan(), assets)
    src = tree.find("resources/asset/media-rep").get("src")
    assert src.startswith("file://")
    assert src.endswith("a_1.mp4")


def test_fcpxml_marks_intent(assets):
    plan = make_plan([make_shot("s001", intent="hero shot")])
    tree = fcpxml_tree(plan, assets)
    marker = tree.find("library/event/project/sequence/spine/asset-clip/marker")
    assert marker.get("value") == "hero shot"


def test_fcpxml_assigns_audio_roles(assets):
    from montaje.models.editplan import AudioMode

    plan = make_plan([
        make_shot("s001", audio_mode=AudioMode.ORIGINAL, duck_db=-14.0),
        make_shot("s002", src_in=8.0, src_out=10.0, timeline_in=60),
    ])
    tree = fcpxml_tree(plan, assets)
    roles = [c.get("audioRole")
             for c in tree.findall("library/event/project/sequence/spine/asset-clip")]
    assert roles == ["dialogue", "music"]


def test_fcpxml_is_well_formed_xml(tmp_path, assets):
    import xml.etree.ElementTree as ET

    dest = export_fcpxml(make_plan(), assets, tmp_path / "t.fcpxml")
    assert ET.parse(dest).getroot().get("version") == "1.10"


def test_exports_survive_snapped_plans(tmp_path, assets):
    """Exports run on post-rails plans, whose snap kinds are already resolved."""
    plan = make_plan([
        make_shot("s001", src_in=0.0, src_out=2.0, timeline_in=0, snap_in=SnapKind.BEAT),
    ])
    assert export_otio(plan, assets, tmp_path / "t.otio").exists()
    assert export_fcpxml(plan, assets, tmp_path / "t.fcpxml").exists()
