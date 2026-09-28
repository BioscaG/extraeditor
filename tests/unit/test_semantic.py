"""Semantic clip analysis (§10): the LLM picks moments, code fixes the timings."""

from __future__ import annotations

import pytest

from montaje.analysis.semantic import prompts
from montaje.analysis.semantic.clip_log import cache_path, run_semantic, semantic_config
from montaje.analysis.semantic.gemini_client import (
    MIN_MOMENT_S,
    SemanticConfig,
    _call_with_retry,
    _is_retryable,
    _parse,
    analyze_clip,
    api_key,
    available,
    build_context,
    reconcile_quotes,
    snap_targets,
    validate_log,
)
from montaje.config import load_config
from montaje.models.cliplog import ClipLog, Moment, MomentKind, People, Quote
from montaje.models.events import Event
from tests.unit.conftest_plan import make_asset


def events_for(asset_id: str = "a_1") -> list[Event]:
    return [
        Event(asset_id=asset_id, analyzer="shots@1", type="shot", t0=0.0, t1=4.0),
        Event(asset_id=asset_id, analyzer="shots@1", type="shot", t0=4.0, t1=9.0),
        Event(asset_id=asset_id, analyzer="quality@1", type="usable", t0=0.5, t1=8.5),
        Event(asset_id=asset_id, analyzer="occlusion@1", type="reveal", t0=0.6, t1=1.0),
        Event(asset_id=asset_id, analyzer="occlusion@1", type="cover", t0=8.0, t1=8.4),
        Event(asset_id=asset_id, analyzer="vad@1", type="speech", t0=2.0, t1=3.5),
        Event(asset_id=asset_id, analyzer="asr@1", type="word", t0=2.1, t1=2.4,
              data={"text": "hello"}),
        Event(asset_id=asset_id, analyzer="audio_events@1", type="crowd", t0=0.0, t1=1.0),
        Event(asset_id=asset_id, analyzer="audio_events@1", type="crowd", t0=1.0, t1=2.0),
    ]


def log_with(*moments: Moment, quotes: list[Quote] | None = None) -> ClipLog:
    return ClipLog(
        asset_id="a_1", summary="a clip", energy=3, aesthetic=4,
        people=People(count_estimate=2, us_present=True),
        moments=list(moments), quotes=quotes or [],
    )


# -- availability ------------------------------------------------------------------------


def test_api_key_prefers_the_explicit_value(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert api_key("explicit") == "explicit"


def test_api_key_reads_either_environment_variable(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    assert api_key() == "google"


def test_not_available_without_a_key(monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert available(SemanticConfig()) is False


def test_available_with_a_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert available(SemanticConfig()) is True


def test_analyze_clip_without_a_key_returns_an_error_not_a_raise(monkeypatch, tmp_path):
    """One missing key must not crash a 100-clip run."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    proxy = tmp_path / "p.mp4"
    proxy.write_bytes(b"x")
    result = analyze_clip(make_asset(), proxy, [], SemanticConfig())
    assert not result.ok
    assert "GEMINI_API_KEY" in result.error


def test_analyze_clip_reports_a_missing_proxy(tmp_path):
    result = analyze_clip(make_asset(), tmp_path / "missing.mp4", [], SemanticConfig())
    assert "no proxy" in result.error


# -- context grounding --------------------------------------------------------------------


def test_context_includes_the_deterministic_timings():
    """§10: local context grounds the model's timestamps. Without it they drift."""
    context = build_context(make_asset("a_1", 9.0), events_for(), transcript="hello there")
    assert "duration: 9.00s" in context
    assert "shot boundaries" in context
    assert "usable spans" in context
    assert "speech detected" in context
    assert "hello there" in context


def test_context_reports_the_absence_of_a_transcript():
    context = build_context(make_asset("a_1", 9.0), events_for())
    assert "no speech transcribed" in context


def test_context_labels_occlusion_without_assigning_meaning():
    """§2.5: nothing project-specific. The prompt must not name a convention."""
    context = build_context(make_asset("a_1", 9.0), events_for())
    assert "no meaning assigned" in context
    assert "hand" not in context.lower()
    assert "vlog" not in context.lower()


def test_prompts_name_no_convention():
    blob = (prompts.SYSTEM + prompts.CLIP_LOG).lower()
    for word in ("festival", "hand", "lens", "vlog", "aftermovie"):
        assert word not in blob, word


def test_context_merges_consecutive_audio_windows():
    context = build_context(make_asset("a_1", 9.0), events_for())
    # Two adjacent 1s crowd windows should read as one 0–2s span.
    assert "crowd 0.0–2.0" in context


def test_context_truncates_long_span_lists():
    events = [
        Event(asset_id="a_1", analyzer="shots@1", type="shot", t0=float(i), t1=float(i + 1))
        for i in range(40)
    ]
    context = build_context(make_asset("a_1", 40.0), events)
    assert "more)" in context


# -- snapping ------------------------------------------------------------------------------


def test_snap_targets_come_from_deterministic_analysis():
    targets = snap_targets(events_for())
    assert 0.0 in targets and 4.0 in targets and 9.0 in targets  # shot edges
    assert 1.0 in targets      # reveal end
    assert 8.0 in targets      # cover start
    assert 2.1 in targets      # word start


def test_a_drifted_moment_snaps_to_a_shot_boundary():
    """A VLM samples at a few fps; its timings land near boundaries, not on them."""
    log = log_with(Moment(t0=4.3, t1=8.2, label="crowd", kind=MomentKind.HIGHLIGHT))
    result, discarded = validate_log(log, 9.0, snap_targets(events_for()))
    assert discarded == []
    assert result.moments[0].t0 == 4.0
    assert result.moments[0].t1 == 8.0


def test_a_moment_far_from_any_boundary_is_left_alone():
    """The model may genuinely mean a point mid-shot."""
    log = log_with(Moment(t0=5.5, t1=6.5, label="mid", kind=MomentKind.SCENIC))
    result, _ = validate_log(log, 9.0, snap_targets(events_for()))
    assert result.moments[0].t0 == 5.5


def test_quotes_snap_to_word_boundaries():
    # 2.18 is nearest the word start at 2.1 (the VAD segment starts at 2.0, further away).
    log = log_with(quotes=[Quote(t0=2.18, t1=2.45, text="hello")])
    result, _ = validate_log(log, 9.0, snap_targets(events_for()))
    assert result.quotes[0].t0 == 2.1
    assert result.quotes[0].t1 == 2.4


def test_snapping_picks_the_nearest_target_of_any_kind():
    """Shot edges, speech edges and word edges all compete; the closest wins."""
    log = log_with(quotes=[Quote(t0=2.02, t1=3.4, text="hello")])
    result, _ = validate_log(log, 9.0, snap_targets(events_for()))
    assert result.quotes[0].t0 == 2.0   # VAD speech start
    assert result.quotes[0].t1 == 3.5   # VAD speech end


# -- validation -----------------------------------------------------------------------------


def test_a_moment_past_the_end_is_discarded_not_clamped():
    """Clamping a guess to the clip end turns it into a shot the editor is asked to trust."""
    log = log_with(Moment(t0=12.0, t1=14.0, label="imagined", kind=MomentKind.HERO))
    result, discarded = validate_log(log, 9.0)
    assert result.moments == []
    assert "past the end" in discarded[0]


def test_a_moment_ending_past_the_end_is_discarded():
    log = log_with(Moment(t0=8.0, t1=20.0, label="long", kind=MomentKind.HIGHLIGHT))
    result, discarded = validate_log(log, 9.0)
    assert result.moments == []
    assert discarded


def test_a_negative_moment_is_discarded():
    log = log_with(Moment(t0=-2.0, t1=1.0, label="before", kind=MomentKind.HIGHLIGHT))
    result, discarded = validate_log(log, 9.0)
    assert result.moments == []
    assert "before the clip" in discarded[0]


def test_a_moment_slightly_over_the_end_is_tolerated():
    """Sub-frame overshoot is rounding, not hallucination."""
    log = log_with(Moment(t0=8.0, t1=9.2, label="ending", kind=MomentKind.HIGHLIGHT))
    result, discarded = validate_log(log, 9.0)
    assert len(result.moments) == 1
    assert discarded == []


def test_an_inverted_moment_is_discarded():
    log = ClipLog(asset_id="a_1")
    log.moments = [Moment.model_construct(t0=5.0, t1=3.0, label="backwards",
                                          why="", score=0.5, kind=MomentKind.HIGHLIGHT)]
    result, discarded = validate_log(log, 9.0)
    assert result.moments == []
    assert "ends before it starts" in discarded[0]


def test_a_collapsed_moment_is_extended_rather_than_dropped():
    """Snapping both ends to the same boundary should not delete a real observation."""
    log = log_with(Moment(t0=3.95, t1=4.05, label="on the cut", kind=MomentKind.HIGHLIGHT))
    result, _ = validate_log(log, 9.0, snap_targets(events_for()))
    assert len(result.moments) == 1
    assert result.moments[0].t1 - result.moments[0].t0 >= MIN_MOMENT_S - 1e-6


def test_moments_are_returned_in_time_order():
    log = log_with(
        Moment(t0=7.0, t1=8.0, label="late", kind=MomentKind.HIGHLIGHT),
        Moment(t0=1.0, t1=2.0, label="early", kind=MomentKind.HIGHLIGHT),
    )
    result, _ = validate_log(log, 9.0)
    assert [m.label for m in result.moments] == ["early", "late"]


def test_scores_are_clamped_to_the_unit_range():
    log = log_with(Moment(t0=1.0, t1=2.0, label="x", score=3.5, kind=MomentKind.HERO))
    result, _ = validate_log(log, 9.0)
    assert result.moments[0].score == 1.0


def test_energy_and_aesthetic_are_clamped():
    """A model that returns 0 or 9 must not produce an out-of-range score downstream."""
    log = log_with()
    # Bypass validation to simulate a response the SDK accepted but the schema would not.
    object.__setattr__(log, "__dict__", {**log.__dict__, "energy": 9, "aesthetic": 0})
    result, _ = validate_log(log, 9.0)
    assert result.energy == 5
    assert result.aesthetic == 1


def test_validation_does_not_mutate_the_input():
    log = log_with(Moment(t0=4.3, t1=8.2, label="x", kind=MomentKind.HIGHLIGHT))
    validate_log(log, 9.0, snap_targets(events_for()))
    assert log.moments[0].t0 == 4.3


# -- response parsing ------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, parsed=None, text=None):
        self.parsed = parsed
        self.text = text


def test_parse_prefers_the_sdks_parsed_object():
    source = log_with(Moment(t0=1.0, t1=2.0, label="x", kind=MomentKind.HERO))
    result = _parse(FakeResponse(parsed=source), "a_9", SemanticConfig(model="m"))
    assert isinstance(result, ClipLog)
    assert result.asset_id == "a_9"
    assert result.model == "m"
    assert result.prompt_version == prompts.PROMPT_VERSION


def test_parse_falls_back_to_json_text():
    payload = log_with().model_dump_json()
    result = _parse(FakeResponse(text=payload), "a_9", SemanticConfig())
    assert isinstance(result, ClipLog)
    assert result.asset_id == "a_9"


def test_parse_reports_an_empty_response():
    assert _parse(FakeResponse(), "a_9", SemanticConfig()) == "empty response"


def test_parse_reports_unparseable_json():
    result = _parse(FakeResponse(text="not json at all"), "a_9", SemanticConfig())
    assert isinstance(result, str) and "unparseable" in result


# -- caching -----------------------------------------------------------------------------------


def test_cache_key_includes_model_and_prompt_version(project):
    a = cache_path(project, "a_1", "gemini-3.8-flash", 1)
    b = cache_path(project, "a_1", "gemini-3.8-flash", 2)
    c = cache_path(project, "a_1", "other-model", 1)
    assert a != b != c
    assert a.name.endswith("@1.json")


def test_cache_path_is_filesystem_safe(project):
    """Model ids can contain slashes."""
    assert "/" not in cache_path(project, "a_1", "models/gemini-3.8", 1).name


def test_run_semantic_explains_itself_when_unavailable(project, monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    lines = run_semantic(project, load_config())
    assert any("GEMINI_API_KEY" in line for line in lines)


def test_run_semantic_uses_the_cache(project, monkeypatch):
    """A cached log must not be re-requested: this is the only paid per-asset step."""
    from montaje.index.store import Store

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    asset = make_asset("a_1", 9.0)
    with Store(project.db_path) as store:
        store.upsert_asset(asset)

    cfg = load_config()
    scfg = semantic_config(cfg)
    path = cache_path(project, "a_1", scfg.model, prompts.PROMPT_VERSION)
    path.parent.mkdir(parents=True, exist_ok=True)
    cached = log_with(Moment(t0=1.0, t1=2.0, label="cached", kind=MomentKind.HERO))
    path.write_text(cached.model_dump_json())

    def fail(*args, **kwargs):
        raise AssertionError("analyze_clip must not be called for a cached asset")

    monkeypatch.setattr("montaje.analysis.semantic.clip_log.analyze_clip", fail)
    lines = run_semantic(project, cfg)
    assert any("1 cached" in line for line in lines)

    with Store(project.db_path) as store:
        assert store.get_clip_log("a_1").moments[0].label == "cached"


def test_semantic_config_comes_from_the_project_config():
    cfg = load_config()
    scfg = semantic_config(cfg)
    assert scfg.model == cfg.semantic.model
    assert scfg.media_resolution == cfg.semantic.media_resolution


# -- selection ---------------------------------------------------------------------------------


def test_a_logged_moment_outranks_a_technically_better_non_moment():
    """Semantic analysis is what makes selection editorial rather than arbitrary."""
    from montaje.plan.build import Candidate

    plain = Candidate("a", 0, 2, quality=0.95, aesthetic=5, energy=3)
    hero = Candidate("a", 0, 2, quality=0.6, aesthetic=3, energy=3,
                     moment_score=0.9, moment_kind="hero")
    assert hero.score(0.6) > plain.score(0.6)


def test_moment_kinds_are_ranked_against_each_other():
    from montaje.plan.build import Candidate

    def candidate(kind: str) -> float:
        return Candidate("a", 0, 2, moment_score=0.8, moment_kind=kind).score(0.6)

    assert candidate("hero") > candidate("highlight") > candidate("scenic")
    # A transition candidate is only useful at a boundary, so it earns no general bonus.
    assert candidate("transition_candidate") < candidate("scenic")


def test_moments_become_candidates_alongside_usable_spans():
    from montaje.plan.build import BuildInputs, collect_candidates

    asset = make_asset("a_1", 9.0)
    inputs = BuildInputs(
        assets={"a_1": asset},
        events={"a_1": events_for()},
        clip_logs={"a_1": log_with(
            Moment(t0=4.0, t1=6.0, label="the drop", score=0.9, kind=MomentKind.HERO)
        )},
    )
    candidates = collect_candidates(inputs)
    assert any(c.is_moment and c.moment_kind == "hero" for c in candidates)
    # The usable span is still offered too, so a moment-free clip is not excluded.
    assert any(not c.is_moment for c in candidates)


def test_the_highest_scoring_candidate_is_the_logged_moment():
    from montaje.plan.build import BuildInputs, collect_candidates

    inputs = BuildInputs(
        assets={"a_1": make_asset("a_1", 9.0)},
        events={"a_1": events_for()},
        clip_logs={"a_1": log_with(
            Moment(t0=4.0, t1=6.0, label="the drop", score=1.0, kind=MomentKind.HERO)
        )},
    )
    best = max(collect_candidates(inputs), key=lambda c: c.score(0.6))
    assert best.moment_kind == "hero"


# -- energy without semantic analysis ------------------------------------------------------


def test_energy_falls_back_to_measured_motion_without_a_clip_log():
    """Otherwise every candidate sits at the default 3 and energy matching does nothing."""
    from montaje.plan.build import MOTION_ENERGY_SCALE, Candidate

    still = Candidate("a", 0, 2, motion=0.0)
    busy = Candidate("a", 0, 2, motion=MOTION_ENERGY_SCALE)
    assert still.energy_value == 0.0
    assert busy.energy_value == 1.0
    # A calm section prefers the still shot, a loud one the busy shot.
    assert still.score(0.1) > busy.score(0.1)
    assert busy.score(0.9) > still.score(0.9)


def test_a_clip_log_overrides_the_motion_fallback():
    """A logged energy is a judgement about the content; motion is only a proxy for it."""
    from montaje.plan.build import Candidate

    logged = Candidate("a", 0, 2, motion=0.0, energy=5, energy_from_log=True)
    assert logged.energy_value == 1.0


def test_the_planner_optimizes_what_the_rhythm_report_measures():
    """Report and planner must agree on 'energy', or the report always complains."""
    from montaje.plan.build import Candidate

    # The report derives visual energy from motion events, so with no clip log the
    # planner's energy must come from motion too.
    assert Candidate("a", 0, 2, motion=3.0).energy_value > 0.0


# -- retrying transient failures -------------------------------------------------------


class FakeApiError(Exception):
    """Stands in for the SDK's APIError, which carries the HTTP status on `.code`."""

    def __init__(self, code: int):
        super().__init__(f"{code} error")
        self.code = code


def test_server_pressure_is_retryable():
    """The first real call this project ever made came back 503 'high demand'."""
    for code in (408, 429, 500, 502, 503, 504):
        assert _is_retryable(FakeApiError(code))


def test_a_fact_about_the_request_is_not_retryable():
    """402 means no credit and 403 a bad key; retrying either burns a minute of backoff
    and then reports the same error, with the real cause hidden behind the delay."""
    for code in (400, 401, 402, 403, 404, 422):
        assert not _is_retryable(FakeApiError(code))


def test_a_dropped_connection_is_retryable():
    """No status at all means the request never reached the service."""
    assert _is_retryable(ConnectionError("reset"))
    assert _is_retryable(TimeoutError())


def test_an_unknown_error_is_not_retried():
    assert not _is_retryable(ValueError("nonsense"))


def test_retry_returns_the_first_success(monkeypatch):
    monkeypatch.setattr("montaje.analysis.semantic.gemini_client.time.sleep", lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise FakeApiError(503)
        return "done"

    assert _call_with_retry("x", flaky) == "done"
    assert len(calls) == 3


def test_retry_gives_up_and_raises_the_last_error(monkeypatch):
    monkeypatch.setattr("montaje.analysis.semantic.gemini_client.time.sleep", lambda s: None)
    calls = []

    def always_busy():
        calls.append(1)
        raise FakeApiError(503)

    with pytest.raises(FakeApiError):
        _call_with_retry("x", always_busy, attempts=4)
    assert len(calls) == 4


def test_a_permanent_error_is_raised_on_the_first_attempt(monkeypatch):
    monkeypatch.setattr("montaje.analysis.semantic.gemini_client.time.sleep", lambda s: None)
    calls = []

    def no_credit():
        calls.append(1)
        raise FakeApiError(402)

    with pytest.raises(FakeApiError):
        _call_with_retry("x", no_credit)
    assert len(calls) == 1, "a 402 must not be retried"


# -- quotes come from the transcript ----------------------------------------------------


def words(*items) -> list[Event]:
    return [
        Event(asset_id="a", analyzer="asr@1", type="word", t0=t0, t1=t1, data={"text": text})
        for t0, t1, text in items
    ]


def log_with_quote(t0: float, t1: float, text: str) -> ClipLog:
    return ClipLog(asset_id="a", summary="s", quotes=[Quote(t0=t0, t1=t1, text=text)])


def test_a_translated_quote_is_replaced_by_the_transcript():
    """Asked in English about Spanish footage, the model returned 'Here we are, I don't see
    anything' for '¿Puedes escuchar?' — fluent, and not what anybody said."""
    events = words((17.15, 17.85, "¿Puedes"), (17.85, 18.17, "escuchar?"))
    out, notes = reconcile_quotes(log_with_quote(17.1, 18.2, "Can you hear it?"), events)
    assert out.quotes[0].text == "¿Puedes escuchar?"
    assert notes


def test_the_quote_span_is_tightened_onto_the_words():
    events = words((17.15, 17.85, "¿Puedes"), (17.85, 18.17, "escuchar?"))
    out, _ = reconcile_quotes(log_with_quote(16.9, 18.6, "whatever"), events)
    assert (out.quotes[0].t0, out.quotes[0].t1) == (17.15, 18.17)


def test_a_quote_with_no_transcribed_speech_under_it_is_unusable():
    """Where the rest of the clip did transcribe, silence here is evidence."""
    events = words((1.0, 1.4, "hola"))
    out, notes = reconcile_quotes(log_with_quote(30.0, 33.0, "invented line"), events)
    assert out.quotes[0].usable is False
    assert any("no transcribed speech" in n for n in notes)


def test_a_clip_with_no_transcript_leaves_quotes_alone():
    """ASR is VAD-gated, so no words can mean it found no speech — not that the model lied."""
    out, notes = reconcile_quotes(log_with_quote(1.0, 2.0, "oh my god"), [])
    assert out.quotes[0].text == "oh my god"
    assert out.quotes[0].usable is True
    assert notes == []


def test_a_matching_quote_is_not_reported_as_rewritten():
    events = words((1.0, 1.3, "vamos"))
    out, notes = reconcile_quotes(log_with_quote(1.0, 1.3, "vamos"), events)
    assert out.quotes[0].text == "vamos"
    assert notes == []


def test_mixed_language_speech_is_kept_as_spoken():
    """The speakers switch language mid-clip; the transcript is the only thing that knows."""
    events = words((0.0, 0.4, "¿Puedes"), (0.4, 0.8, "escuchar?"), (1.0, 1.4, "oh"),
                   (1.4, 1.6, "my"), (1.6, 1.9, "god"))
    out, _ = reconcile_quotes(log_with_quote(0.0, 2.0, "Can you hear it? amazing"), events)
    assert out.quotes[0].text == "¿Puedes escuchar? oh my god"


# -- the language hint ------------------------------------------------------------------


def test_the_language_hint_is_omitted_when_none_is_declared():
    """Mixed-language footage declares no language, and a wrong hint is worse than none."""
    assert "mostly in" not in prompts.clip_log_prompt("ctx", None)


def test_the_language_hint_names_the_language():
    assert "Spanish" in prompts.clip_log_prompt("ctx", "es")


def test_an_unknown_language_code_is_passed_through():
    assert "sw" in prompts.clip_log_prompt("ctx", "sw")


def test_the_verbatim_rule_always_applies():
    """It is in the system prompt, not the per-language hint: it holds with no language too."""
    assert "never translated" in prompts.SYSTEM


def test_the_language_is_part_of_the_cache_key(tmp_path):
    from montaje.workspace import Workspace

    ws = Workspace(tmp_path / "p")
    a = cache_path(ws, "a_1", "m", 2, "es")
    b = cache_path(ws, "a_1", "m", 2, "en")
    assert a != b, "switching language must not read back logs quoting the old one"
    assert cache_path(ws, "a_1", "m", 2, None) != a
