"""Conform and audio mix, exercised against real media (§19.1, §19.2)."""

from __future__ import annotations

import pytest

from montaje.color.grade import Grade
from montaje.color.normalize import ShotCorrection
from montaje.index.store import Store
from montaje.ingest.probe import probe_file
from montaje.models.editplan import AudioMode, MusicEdit, Sfx
from montaje.render.conform import HANDLE_S, ConformSpec, build_filters, conform_plan, conform_shot
from montaje.sound.mix import (
    ClipAudio,
    MixSpec,
    SfxPlacement,
    build_graph,
    duck_envelope_expression,
    mix_spec_from_plan,
    render_mix,
)
from tests.unit.conftest_plan import make_plan, make_shot


def _asset(ws, name: str):
    with Store(ws.db_path) as store:
        return next(a for a in store.list_assets() if a.path.name == name)


@pytest.fixture()
def spec(ingested):
    return ConformSpec(
        asset=_asset(ingested, "plain.mp4"),
        src_in=1.0, src_out=2.5, width=540, height=960, fps=30.0,
    )


# -- conform ---------------------------------------------------------------------------


def test_conform_writes_an_intermediate_at_the_output_size(spec, tmp_path):
    result = conform_shot(spec, tmp_path)
    assert result.path.exists()
    p = probe_file(result.path)
    assert (p.video.width, p.video.height) == (540, 960)


def test_conform_includes_handles_on_both_sides(spec, tmp_path):
    result = conform_shot(spec, tmp_path)
    p = probe_file(result.path)
    expected = (spec.src_out - spec.src_in) + 2 * HANDLE_S
    assert p.duration_s == pytest.approx(expected, abs=0.15)


def test_head_handle_is_clipped_at_the_asset_start(ingested):
    spec = ConformSpec(asset=_asset(ingested, "plain.mp4"), src_in=0.1, src_out=1.0,
                       width=540, height=960, fps=30.0)
    assert spec.extract_in == 0.0
    assert spec.head_handle_s == pytest.approx(0.1)


def test_tail_handle_is_clipped_at_the_asset_end(ingested):
    asset = _asset(ingested, "plain.mp4")
    spec = ConformSpec(asset=asset, src_in=1.0, src_out=asset.duration_s - 0.1,
                       width=540, height=960, fps=30.0)
    assert spec.extract_out == pytest.approx(asset.duration_s)


def test_conform_is_cached_by_spec(spec, tmp_path):
    first = conform_shot(spec, tmp_path)
    second = conform_shot(spec, tmp_path)
    assert second.cached is True
    assert second.path == first.path


def test_a_different_range_gets_a_different_intermediate(ingested, tmp_path):
    asset = _asset(ingested, "plain.mp4")
    a = ConformSpec(asset=asset, src_in=0.5, src_out=1.5, width=540, height=960, fps=30.0)
    b = ConformSpec(asset=asset, src_in=2.0, src_out=3.0, width=540, height=960, fps=30.0)
    assert a.output_name() != b.output_name()


def test_a_different_correction_invalidates_the_cache(ingested):
    asset = _asset(ingested, "plain.mp4")
    base = ConformSpec(asset=asset, src_in=0.5, src_out=1.5, width=540, height=960, fps=30.0)
    graded = ConformSpec(asset=asset, src_in=0.5, src_out=1.5, width=540, height=960, fps=30.0,
                         correction=ShotCorrection(0, 0.9, 1.0, 1.1, 1.0))
    assert base.output_name() != graded.output_name()


def test_conform_output_is_sdr_even_from_hlg(ingested, tmp_path):
    spec = ConformSpec(asset=_asset(ingested, "hlg.mp4"), src_in=0.5, src_out=2.0,
                       width=540, height=960, fps=30.0)
    result = conform_shot(spec, tmp_path)
    p = probe_file(result.path)
    assert p.video.color_transfer in (None, "unknown", "bt709")


def test_filters_apply_color_in_pipeline_order(ingested):
    """§15: input transform, normalize, grade, finishing."""
    spec = ConformSpec(
        asset=_asset(ingested, "hlg.mp4"), src_in=0.5, src_out=1.5,
        width=540, height=960, fps=30.0,
        correction=ShotCorrection(0, 0.9, 1.0, 1.1, 1.0),
        grade=Grade(contrast=1.1, grain=0.2),
    )
    _, filters = build_filters(spec)
    chain = ",".join(filters)
    assert chain.index("colorchannelmixer") < chain.index("eq=contrast")
    assert chain.index("eq=contrast") < chain.index("noise")


def test_filters_pad_rather_than_crop(spec):
    """Cropping is the reframe module's job, not the conform's."""
    _, filters = build_filters(spec)
    chain = ",".join(filters)
    assert "force_original_aspect_ratio=decrease" in chain
    assert "pad=540:960" in chain


def test_conform_normalizes_the_frame_rate(spec):
    _, filters = build_filters(spec)
    assert any(f == "fps=30.0" for f in filters)


def test_conform_plan_reuses_one_intermediate_for_identical_shots(ingested, tmp_path):
    asset = _asset(ingested, "plain.mp4")
    plan = make_plan([
        make_shot("s001", asset=asset.asset_id, src_in=0.5, src_out=1.5, timeline_in=0),
        make_shot("s002", asset=asset.asset_id, src_in=0.5, src_out=1.5, timeline_in=30),
    ], fps=30.0)
    plan.format.width, plan.format.height = 540, 960
    result = conform_plan(plan, {asset.asset_id: asset}, tmp_path)
    assert result.path_for("s001") == result.path_for("s002")


def test_conform_plan_fails_loudly_on_an_unknown_asset(tmp_path):
    with pytest.raises(KeyError, match="not in the index"):
        conform_plan(make_plan(), {}, tmp_path)


# -- mix ----------------------------------------------------------------------------------


def test_mix_is_exactly_the_requested_duration(ingested, tmp_path, fixtures):
    spec = MixSpec(duration_s=6.0, music_wav=fixtures / "track.wav",
                   music_segments=[(0.0, 6.0, 0.0)])
    out = tmp_path / "master.wav"
    render_mix(spec, out)
    assert probe_file(out).duration_s == pytest.approx(6.0, abs=0.05)


def test_mix_hits_the_loudness_target(tmp_path, fixtures):
    spec = MixSpec(duration_s=6.0, music_wav=fixtures / "track.wav",
                   music_segments=[(48.0, 54.0, 0.0)], target_lufs=-14.0)
    out = tmp_path / "master.wav"
    stats = render_mix(spec, out)
    # The output limiter shaves peaks, so the landing point is a little under target.
    assert stats["final_lufs"] == pytest.approx(-14.0, abs=1.5)


def test_mix_respects_the_true_peak_ceiling(tmp_path, fixtures):
    spec = MixSpec(duration_s=6.0, music_wav=fixtures / "track.wav",
                   music_segments=[(48.0, 54.0, 0.0)], true_peak_dbtp=-1.0)
    out = tmp_path / "master.wav"
    stats = render_mix(spec, out)
    assert stats["final_true_peak_dbtp"] <= -0.5


def test_stems_are_exported(tmp_path, fixtures, ingested):
    asset = _asset(ingested, "speech.mp4")
    spec = MixSpec(
        duration_s=8.0, music_wav=fixtures / "track.wav", music_segments=[(0.0, 8.0, 0.0)],
        clips=[ClipAudio(shot_id="s1", wav=ingested.audio_48k_path(asset.asset_id),
                         src_in=1.0, src_out=5.0, timeline_s=1.0, duck_music_db=-12.0)],
    )
    stems_dir = tmp_path / "stems"
    stats = render_mix(spec, tmp_path / "master.wav", stems_dir=stems_dir)
    assert set(stats["stems"]) == {"music", "dialogue"}
    assert (stems_dir / "music.wav").exists()
    assert (stems_dir / "dialogue.wav").exists()


def test_mix_without_stems_dir_emits_no_split(tmp_path, fixtures):
    """An unconsumed filter output pad is a graph error, so the split is conditional."""
    spec = MixSpec(duration_s=4.0, music_wav=fixtures / "track.wav",
                   music_segments=[(0.0, 4.0, 0.0)])
    _, steps, _ = build_graph(spec, export_stems=False)
    assert not any("asplit" in s for s in steps)
    render_mix(spec, tmp_path / "master.wav")


def test_mix_with_no_audio_at_all_still_produces_silence(tmp_path):
    out = tmp_path / "master.wav"
    render_mix(MixSpec(duration_s=3.0), out)
    assert probe_file(out).duration_s == pytest.approx(3.0, abs=0.05)


# -- ducking ---------------------------------------------------------------------------------


def test_duck_expression_is_none_without_speaking_clips():
    clips = [ClipAudio(shot_id="s1", wav="x.wav", src_in=0, src_out=1, timeline_s=0)]
    assert duck_envelope_expression(clips, 10.0) is None


def test_duck_expression_covers_the_speaking_window():
    clips = [ClipAudio(shot_id="s1", wav="x.wav", src_in=0, src_out=2, timeline_s=3.0,
                       duck_music_db=-14.0)]
    expr = duck_envelope_expression(clips, 10.0)
    assert "3.0000" in expr and "5.0000" in expr


def test_duck_lead_in_accounts_for_the_j_cut():
    """A J-cut brings speech in early; the dip must arrive with it."""
    clips = [ClipAudio(shot_id="s1", wav="x.wav", src_in=0, src_out=2, timeline_s=3.0,
                       duck_music_db=-14.0, j_cut_s=0.5)]
    expr = duck_envelope_expression(clips, 10.0)
    assert "2.5000" in expr


def test_deeper_duck_wins_where_two_clips_overlap():
    clips = [
        ClipAudio(shot_id="s1", wav="x.wav", src_in=0, src_out=2, timeline_s=0.0,
                  duck_music_db=-6.0),
        ClipAudio(shot_id="s2", wav="x.wav", src_in=0, src_out=2, timeline_s=1.0,
                  duck_music_db=-20.0),
    ]
    assert "max(" in duck_envelope_expression(clips, 10.0)


# -- plan translation --------------------------------------------------------------------------


def test_mix_spec_skips_music_only_shots(ingested, fixtures):
    asset = _asset(ingested, "speech.mp4")
    plan = make_plan([
        make_shot("s001", asset=asset.asset_id, src_in=0.0, src_out=2.0, timeline_in=0,
                  audio_mode=AudioMode.MUSIC_ONLY),
        make_shot("s002", asset=asset.asset_id, src_in=3.0, src_out=5.0, timeline_in=60,
                  audio_mode=AudioMode.ORIGINAL, duck_db=-14.0),
    ])
    spec = mix_spec_from_plan(
        plan, audio_paths={asset.asset_id: ingested.audio_48k_path(asset.asset_id)},
        music_wav=fixtures / "track.wav",
    )
    assert [c.shot_id for c in spec.clips] == ["s002"]


def test_mix_spec_aligns_sfx_peaks_to_their_anchor(fixtures, tmp_path):
    """§14.1: the peak offset is what lands on the frame, not the file start."""
    plan = make_plan()
    plan.sfx = [Sfx(id="fx1", sfx="impact.a", anchor_frame=60, gain_db=-4)]
    spec = mix_spec_from_plan(
        plan, audio_paths={}, music_wav=None,
        sfx_paths={"impact.a": fixtures / "click_120.wav"},
        sfx_peak_offsets={"impact.a": 0.25},
    )
    # anchor 60 frames = 2.0s, minus the 0.25s peak offset.
    assert spec.sfx[0].timeline_s == pytest.approx(1.75)


def test_mix_spec_places_a_riser_to_end_on_its_frame(fixtures):
    plan = make_plan()
    plan.sfx = [Sfx(id="fx1", sfx="riser.a", end_on_frame=120, gain_db=-8)]
    spec = mix_spec_from_plan(
        plan, audio_paths={}, music_wav=None,
        sfx_paths={"riser.a": fixtures / "click_120.wav"},
    )
    # click_120.wav is 10s long and must end at 4.0s, so it would start before zero.
    assert spec.sfx[0].timeline_s == 0.0


def test_mix_spec_uses_the_plans_music_edits(fixtures):
    plan = make_plan()
    plan.music.edits = [
        MusicEdit(src_in=0.0, src_out=4.0, timeline_in=0),
        MusicEdit(src_in=16.0, src_out=20.0, timeline_in=120),
    ]
    spec = mix_spec_from_plan(plan, audio_paths={}, music_wav=fixtures / "track.wav")
    assert spec.music_segments == [(0.0, 4.0, 0.0), (16.0, 20.0, 4.0)]


def test_mix_spec_has_no_music_when_the_plan_has_no_edits(fixtures):
    spec = mix_spec_from_plan(make_plan(), audio_paths={}, music_wav=fixtures / "track.wav")
    assert spec.music_wav is None


def test_sfx_placement_is_never_negative(fixtures):
    plan = make_plan()
    plan.sfx = [Sfx(id="fx1", sfx="impact.a", anchor_frame=2)]
    spec = mix_spec_from_plan(
        plan, audio_paths={}, music_wav=None,
        sfx_paths={"impact.a": fixtures / "click_120.wav"},
        sfx_peak_offsets={"impact.a": 1.0},
    )
    assert spec.sfx[0].timeline_s == 0.0


def test_graph_places_sfx_and_exports_it_as_a_stem(tmp_path, fixtures):
    spec = MixSpec(
        duration_s=4.0,
        sfx=[SfxPlacement(sfx_id="fx1", wav=fixtures / "click_120.wav",
                          timeline_s=1.0, gain_db=-6)],
    )
    stats = render_mix(spec, tmp_path / "m.wav", stems_dir=tmp_path / "stems")
    assert "sfx" in stats["stems"]
