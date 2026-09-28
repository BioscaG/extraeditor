"""Plan operations: validated on apply, versioned, with a diff summary (§17.3)."""

from __future__ import annotations

import pytest

from montaje.models.editplan import AudioMode, Overlay, Sfx
from montaje.plan.ops import OpError, apply_ops, available_ops, diff_summary
from tests.unit.conftest_plan import make_plan, make_shot


def test_apply_bumps_the_version_once_per_batch():
    plan = make_plan()
    result = apply_ops(plan, [
        {"op": "set_intent", "id": "s001", "intent": "a"},
        {"op": "set_intent", "id": "s002", "intent": "b"},
    ])
    assert result.plan.version == plan.version + 1


def test_apply_does_not_mutate_the_input():
    plan = make_plan()
    apply_ops(plan, [{"op": "remove_shot", "id": "s001"}])
    assert [s.id for s in plan.shots] == ["s001", "s002"]


def test_a_failing_op_leaves_the_plan_untouched():
    """A partially-applied batch must never be written to disk."""
    plan = make_plan()
    with pytest.raises(OpError):
        apply_ops(plan, [
            {"op": "remove_shot", "id": "s001"},
            {"op": "remove_shot", "id": "does_not_exist"},
        ])
    assert [s.id for s in plan.shots] == ["s001", "s002"]


def test_unknown_op_is_rejected_with_the_known_list():
    with pytest.raises(OpError, match="unknown op"):
        apply_ops(make_plan(), [{"op": "frobnicate"}])


def test_missing_argument_is_reported():
    with pytest.raises(OpError, match="missing argument"):
        apply_ops(make_plan(), [{"op": "trim_shot"}])


def test_every_op_in_the_spec_is_registered():
    """§17.3 lists the ops the agent may call."""
    expected = {
        "insert_shot", "remove_shot", "move_shot", "trim_shot", "replace_source",
        "set_transition", "set_fx", "set_audio", "set_speed", "set_reframe", "set_color",
        "add_overlay", "update_overlay", "remove_overlay", "add_sfx", "remove_sfx",
        "set_music_edits", "set_section", "set_note",
    }
    assert expected <= set(available_ops())


# -- shots -----------------------------------------------------------------------------


def test_insert_shot_allocates_an_id():
    result = apply_ops(make_plan(), [
        {"op": "insert_shot", "asset": "a_1", "src_in": 10.0, "src_out": 12.0},
    ])
    assert len(result.plan.shots) == 3
    assert result.plan.shots[-1].id not in ("s001", "s002")


def test_insert_shot_honours_an_explicit_id():
    result = apply_ops(make_plan(), [
        {"op": "insert_shot", "id": "hero", "asset": "a_1", "src_in": 1.0, "src_out": 2.0},
    ])
    assert result.plan.shot("hero").src_out == 2.0


def test_insert_shot_rejects_a_duplicate_id():
    with pytest.raises(OpError, match="already exists"):
        apply_ops(make_plan(), [
            {"op": "insert_shot", "id": "s001", "asset": "a_1", "src_in": 1.0, "src_out": 2.0},
        ])


def test_insert_after_places_the_shot_in_order():
    result = apply_ops(make_plan(), [
        {"op": "insert_shot", "id": "mid", "asset": "a_1", "src_in": 9.0, "src_out": 10.0,
         "after": "s001"},
    ])
    assert [s.id for s in result.plan.shots] == ["s001", "mid", "s002"]


def test_insert_after_an_unknown_shot_fails():
    with pytest.raises(OpError, match="insert after"):
        apply_ops(make_plan(), [
            {"op": "insert_shot", "asset": "a_1", "src_in": 1.0, "src_out": 2.0, "after": "nope"},
        ])


def test_remove_shot():
    result = apply_ops(make_plan(), [{"op": "remove_shot", "id": "s001"}])
    assert [s.id for s in result.plan.shots] == ["s002"]


def test_move_shot_reorders():
    result = apply_ops(make_plan(), [{"op": "move_shot", "id": "s002", "before": "s001"}])
    assert [s.id for s in result.plan.shots] == ["s002", "s001"]


def test_trim_shot():
    result = apply_ops(make_plan(), [{"op": "trim_shot", "id": "s001", "src_out": 2.5}])
    assert result.plan.shot("s001").src_out == 2.5


def test_trim_shot_rejects_an_inverted_range():
    with pytest.raises(OpError, match="must exceed"):
        apply_ops(make_plan(), [{"op": "trim_shot", "id": "s001", "src_out": 0.5}])


def test_replace_source():
    result = apply_ops(make_plan(), [
        {"op": "replace_source", "id": "s001", "asset": "a_2", "src_in": 0.0, "src_out": 1.5},
    ])
    shot = result.plan.shot("s001")
    assert (shot.asset, shot.src_out) == ("a_2", 1.5)


def test_replace_source_rejects_an_empty_range():
    with pytest.raises(OpError, match="empty"):
        apply_ops(make_plan(), [
            {"op": "replace_source", "id": "s001", "asset": "a_2", "src_in": 5.0, "src_out": 5.0},
        ])


# -- attributes -------------------------------------------------------------------------


def test_set_transition():
    result = apply_ops(make_plan(), [
        {"op": "set_transition", "id": "s002",
         "component": {"id": "transition.whip_pan@1.2", "preset": "subtle",
                       "duration": {"beats": 0.5}}},
    ])
    ref = result.plan.shot("s002").transition_in
    assert ref.id == "transition.whip_pan@1.2"
    assert ref.duration.beats == 0.5


def test_set_transition_to_none_means_a_hard_cut():
    plan = apply_ops(make_plan(), [
        {"op": "set_transition", "id": "s002", "component": {"id": "transition.whip_pan@1.2"}},
    ]).plan
    result = apply_ops(plan, [{"op": "set_transition", "id": "s002", "component": None}])
    assert result.plan.shot("s002").transition_in is None
    assert "hard cut" in result.summary


def test_set_audio():
    result = apply_ops(make_plan(), [
        {"op": "set_audio", "id": "s001",
         "audio": {"mode": "original", "duck_music_db": -14, "j_cut_frames": 8,
                   "cleanup": "dialogue"}},
    ])
    audio = result.plan.shot("s001").audio
    assert audio.mode == AudioMode.ORIGINAL
    assert audio.duck_music_db == -14
    assert audio.j_cut_frames == 8


def test_set_speed():
    result = apply_ops(make_plan(), [
        {"op": "set_speed", "id": "s001",
         "speed": [{"from": 0.0, "to": 0.5, "rate": 1.0}, {"from": 0.5, "to": 1.0, "rate": 0.5}]},
    ])
    assert len(result.plan.shot("s001").speed) == 2


def test_speed_ramps_change_timeline_length():
    """Half speed over half the shot makes the shot longer on the timeline."""
    plan = make_plan([make_shot("s001", src_in=0.0, src_out=2.0)])
    assert plan.shots[0].timeline_duration_frames(30.0) == 60
    result = apply_ops(plan, [
        {"op": "set_speed", "id": "s001",
         "speed": [{"from": 0.0, "to": 0.5, "rate": 1.0}, {"from": 0.5, "to": 1.0, "rate": 0.5}]},
    ])
    assert result.plan.shot("s001").timeline_duration_frames(30.0) == 90


def test_set_reframe():
    result = apply_ops(make_plan(), [
        {"op": "set_reframe", "id": "s001", "reframe": {"mode": "auto_subject", "ease": "glide"}},
    ])
    assert result.plan.shot("s001").reframe.mode == "auto_subject"


def test_set_color():
    result = apply_ops(make_plan(), [
        {"op": "set_color", "id": "s001",
         "color": {"normalize": "auto", "match_to": "s002", "grade": "style"}},
    ])
    assert result.plan.shot("s001").color.match_to == "s002"


def test_set_captions_and_clear():
    plan = apply_ops(make_plan(), [
        {"op": "set_captions", "id": "s001",
         "captions": {"id": "text.captions_word_pop@2.0", "words": "auto"}},
    ]).plan
    assert plan.shot("s001").captions.id == "text.captions_word_pop@2.0"
    cleared = apply_ops(plan, [{"op": "set_captions", "id": "s001"}]).plan
    assert cleared.shot("s001").captions is None


def test_set_section_and_note():
    result = apply_ops(make_plan(), [
        {"op": "set_section", "id": "s001", "section": "intro"},
        {"op": "set_note", "note": "opens on the crowd"},
    ])
    assert result.plan.shot("s001").section == "intro"
    assert result.plan.notes == "opens on the crowd"


def test_ops_on_a_missing_shot_fail():
    with pytest.raises(OpError, match="no shot"):
        apply_ops(make_plan(), [{"op": "set_intent", "id": "nope", "intent": "x"}])


# -- overlays, sfx, music ------------------------------------------------------------------


def test_add_and_remove_overlay():
    plan = apply_ops(make_plan(), [
        {"op": "add_overlay", "id": "o001", "component": "text.timestamp@1.0",
         "from_frame": 0, "to_frame": 45, "props": {"text": "10 HORAS"}},
    ]).plan
    assert plan.overlays[0].props["text"] == "10 HORAS"
    assert apply_ops(plan, [{"op": "remove_overlay", "id": "o001"}]).plan.overlays == []


def test_add_duplicate_overlay_fails():
    plan = make_plan()
    plan.overlays = [Overlay(id="o001", component="text.x@1", from_frame=0, to_frame=10)]
    with pytest.raises(OpError, match="already exists"):
        apply_ops(plan, [{"op": "add_overlay", "id": "o001", "component": "text.y@1",
                          "from_frame": 0, "to_frame": 10}])


def test_update_overlay():
    plan = make_plan()
    plan.overlays = [Overlay(id="o001", component="text.x@1", from_frame=0, to_frame=10)]
    result = apply_ops(plan, [{"op": "update_overlay", "id": "o001", "to_frame": 30}])
    assert result.plan.overlays[0].to_frame == 30


def test_update_missing_overlay_fails():
    with pytest.raises(OpError, match="no overlay"):
        apply_ops(make_plan(), [{"op": "update_overlay", "id": "o001", "to_frame": 30}])


def test_add_and_remove_sfx():
    plan = apply_ops(make_plan(), [
        {"op": "add_sfx", "id": "fx001", "sfx": "impact.deep_03", "anchor_frame": 30,
         "gain_db": -4, "source": "agent"},
    ]).plan
    assert plan.sfx[0].sfx == "impact.deep_03"
    assert apply_ops(plan, [{"op": "remove_sfx", "id": "fx001"}]).plan.sfx == []


def test_remove_missing_sfx_fails():
    with pytest.raises(OpError, match="no sfx"):
        apply_ops(make_plan(), [{"op": "remove_sfx", "id": "fx001"}])


def test_add_duplicate_sfx_fails():
    plan = make_plan()
    plan.sfx = [Sfx(id="fx001", sfx="impact.deep_03", anchor_frame=10)]
    with pytest.raises(OpError, match="already exists"):
        apply_ops(plan, [{"op": "add_sfx", "id": "fx001", "sfx": "riser.short_02",
                          "anchor_frame": 20}])


def test_set_music_edits():
    result = apply_ops(make_plan(), [
        {"op": "set_music_edits", "edits": [
            {"src_in": 0.0, "src_out": 31.5, "timeline_in": 0},
            {"src_in": 63.0, "src_out": 102.0, "timeline_in": 945,
             "join": {"type": "crossfade", "frames": 3, "on": "downbeat"}},
        ]},
    ])
    assert len(result.plan.music.edits) == 2
    assert result.plan.music.edits[1].join.on == "downbeat"


# -- diff -----------------------------------------------------------------------------------


def test_diff_reports_added_and_removed_shots():
    before = make_plan()
    after = apply_ops(before, [
        {"op": "remove_shot", "id": "s001"},
        {"op": "insert_shot", "id": "s009", "asset": "a_1", "src_in": 9.0, "src_out": 11.0},
    ]).plan
    text = diff_summary(before, after)
    assert "removed shots: s001" in text
    assert "added shots: s009" in text


def test_diff_reports_a_trim_and_the_duration_change():
    before = make_plan()
    after = apply_ops(before, [{"op": "trim_shot", "id": "s002", "src_out": 6.0}]).plan
    text = diff_summary(before, after)
    assert "s002" in text and "range" in text


def test_diff_reports_a_transition_change():
    before = make_plan()
    after = apply_ops(before, [
        {"op": "set_transition", "id": "s002", "component": {"id": "transition.flash@1.0"}},
    ]).plan
    assert "transition none → transition.flash@1.0" in diff_summary(before, after)


def test_diff_of_identical_plans_says_so():
    plan = make_plan()
    assert "no structural change" in diff_summary(plan, plan)


def test_diff_includes_the_version_bump():
    before = make_plan()
    after = apply_ops(before, [{"op": "set_note", "note": "x"}]).plan
    assert "v1 → v2" in diff_summary(before, after)
