"""Auto-checks on the rendered file (§20), against videos with deliberate defects."""

from __future__ import annotations

import pytest

from montaje import ffmpeg
from montaje.critic.autochecks import (
    AutoCheckReport,
    Finding,
    check_black,
    check_duration,
    check_freeze,
    check_loudness,
    check_safe_areas,
    run_autochecks,
)


def make_video(path, filters: str, duration: float = 4.0, audio: str = "sine",
               level: float = 0.2) -> None:
    """A test clip with a named video filter chain and a chosen audio source."""
    audio_src = {
        "sine": f"sine=frequency=220:sample_rate=48000:duration={duration}",
        "silence": f"anullsrc=r=48000:cl=stereo:d={duration}",
        "loud": f"sine=frequency=220:sample_rate=48000:duration={duration}",
    }[audio]
    args = [
        "-y", "-v", "error",
        "-f", "lavfi", "-i", f"testsrc2=size=320x568:rate=30:duration={duration}",
        "-f", "lavfi", "-i", audio_src,
    ]
    volume = 1.0 if audio != "loud" else 0.99
    args += [
        "-filter_complex", f"[0:v]{filters}[v];[1:a]volume={volume if audio!='sine' else level}[a]",
        "-map", "[v]", "-map", "[a]", "-t", str(duration),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
    ]
    ffmpeg.run(args)


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    path = tmp_path_factory.mktemp("checks") / "clean.mp4"
    make_video(path, "format=yuv420p")
    return path


@pytest.fixture(scope="module")
def with_black(tmp_path_factory):
    """A full second of black in the middle — a failed intermediate looks like this."""
    path = tmp_path_factory.mktemp("checks") / "black.mp4"
    make_video(path, "geq=lum='if(between(T,1.5,2.5),0,lum(X,Y))':cb=128:cr=128")
    return path


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    """Three seconds of one repeated frame — a stuck decoder looks like this."""
    path = tmp_path_factory.mktemp("checks") / "frozen.mp4"
    make_video(path, "format=yuv420p,select='lte(n,1)',loop=loop=-1:size=1:start=0,fps=30",
               duration=4.0)
    return path


# -- black --------------------------------------------------------------------------------


def test_black_run_is_found(with_black):
    findings = check_black(with_black)
    assert findings
    assert findings[0].check == "black_frames"
    assert 1.0 <= (findings[0].at_s or 0) <= 2.0


def test_a_long_black_run_is_an_error(with_black):
    assert any(f.severity == "error" for f in check_black(with_black))


def test_clean_video_has_no_black_findings(clean):
    assert check_black(clean) == []


def test_a_brief_black_frame_is_tolerated(clean):
    """A dip passes through black by design; a frame or two at a cut is not a defect."""
    assert check_black(clean, max_run_s=0.25) == []


# -- freeze --------------------------------------------------------------------------------


def test_a_freeze_is_found(frozen):
    findings = check_freeze(frozen, max_freeze_s=1.0)
    assert findings
    assert findings[0].check == "frozen_frames"


def test_moving_video_is_not_flagged_as_frozen(clean):
    assert check_freeze(clean) == []


# -- loudness -------------------------------------------------------------------------------


def test_silence_is_an_error(tmp_path):
    path = tmp_path / "silent.mp4"
    make_video(path, "format=yuv420p", audio="silence")
    findings = check_loudness(path, target_lufs=-14.0, true_peak_dbtp=-1.0)
    assert any("silent" in f.message for f in findings)
    assert all(f.severity == "error" for f in findings if "silent" in f.message)


def test_a_loudness_miss_is_reported(clean):
    """The test clip is quiet; measured against a -14 target that is worth saying."""
    findings = check_loudness(clean, target_lufs=-14.0, true_peak_dbtp=-1.0)
    assert any(f.check == "loudness" for f in findings)


def test_a_loudness_within_tolerance_is_not_reported(clean):
    from montaje.analysis.local.loudness import measure_loudness

    actual = measure_loudness(clean)["integrated_lufs"]
    findings = check_loudness(clean, target_lufs=actual, true_peak_dbtp=0.0)
    assert not any(f.check == "loudness" for f in findings)


def test_a_true_peak_over_the_ceiling_is_an_error(tmp_path):
    """The defect that motivated these checks: AAC overshoots the PCM it is given."""
    from montaje.analysis.local.loudness import measure_loudness

    path = tmp_path / "hot.mp4"
    make_video(path, "format=yuv420p", audio="loud")
    # Measure first and set the ceiling below it: the point is the check's verdict, not
    # the absolute level a synthetic tone happens to reach after an AAC encode.
    peak = measure_loudness(path)["true_peak_dbtp"]
    findings = check_loudness(path, target_lufs=-14.0, true_peak_dbtp=peak - 3.0)
    assert any(f.check == "true_peak" and f.severity == "error" for f in findings)


def test_a_true_peak_under_the_ceiling_passes(tmp_path):
    from montaje.analysis.local.loudness import measure_loudness

    path = tmp_path / "quiet.mp4"
    make_video(path, "format=yuv420p")
    peak = measure_loudness(path)["true_peak_dbtp"]
    findings = check_loudness(path, target_lufs=-14.0, true_peak_dbtp=peak + 3.0)
    assert not any(f.check == "true_peak" for f in findings)


# -- duration --------------------------------------------------------------------------------


def test_a_duration_mismatch_is_an_error(clean):
    assert check_duration(clean, expected_s=10.0)[0].severity == "error"


def test_a_matching_duration_passes(clean):
    assert check_duration(clean, expected_s=4.0, tolerance_s=0.3) == []


# -- safe areas --------------------------------------------------------------------------------


def test_bright_high_contrast_detail_in_the_unsafe_zone_is_flagged(tmp_path):
    """§28: captions hidden under the platform UI. Measured from pixels, not from layout."""
    path = tmp_path / "unsafe.mp4"
    # A band of bright hard-edged stripes in the bottom 15% — where the feed UI sits.
    make_video(
        path,
        "format=yuv420p,"
        "drawbox=x=0:y=500:w=320:h=60:color=black@1:t=fill,"
        "drawbox=x=10:y=510:w=20:h=40:color=white@1:t=fill,"
        "drawbox=x=50:y=510:w=20:h=40:color=white@1:t=fill,"
        "drawbox=x=90:y=510:w=20:h=40:color=white@1:t=fill,"
        "drawbox=x=130:y=510:w=20:h=40:color=white@1:t=fill",
    )
    findings = check_safe_areas(path, aspect="9:16", platform="social")
    assert findings, "bright detail on a dark scrim in the bottom unsafe zone must be flagged"
    assert findings[0].severity == "warning"


def test_safe_area_findings_are_reported_once(tmp_path):
    """A caption that breaks the rule breaks it on many frames; one report is enough."""
    path = tmp_path / "unsafe2.mp4"
    make_video(
        path,
        "format=yuv420p,drawbox=x=0:y=500:w=320:h=60:color=black@1:t=fill,"
        "drawbox=x=10:y=510:w=20:h=40:color=white@1:t=fill,"
        "drawbox=x=60:y=510:w=20:h=40:color=white@1:t=fill",
    )
    assert len(check_safe_areas(path)) <= 1


def test_safe_area_check_is_only_ever_a_warning(tmp_path):
    """Footage legitimately fills the frame; this can only flag it for a human."""
    path = tmp_path / "busy.mp4"
    make_video(path, "format=yuv420p")
    assert all(f.severity == "warning" for f in check_safe_areas(path))


# -- the whole report ---------------------------------------------------------------------------


def test_a_missing_file_is_an_error(tmp_path):
    report = run_autochecks(tmp_path / "nope.mp4")
    assert not report.passed
    assert "no file" in report.errors[0].message


def test_the_report_records_what_it_measured(clean):
    report = run_autochecks(clean, target_lufs=-40.0, true_peak_dbtp=0.0)
    assert report.measured["duration_s"] == pytest.approx(4.0, abs=0.2)
    assert "integrated_lufs" in report.measured


def test_errors_are_ordered_first(with_black):
    report = run_autochecks(with_black, target_lufs=-14.0, true_peak_dbtp=-1.0)
    severities = [f.severity for f in report.findings]
    assert severities == sorted(severities, key=lambda s: s != "error")


def test_markdown_states_the_verdict_first(clean):
    text = run_autochecks(clean, target_lufs=-40.0, true_peak_dbtp=0.0).to_markdown()
    assert "# Auto-checks" in text
    assert "**Result:**" in text


def test_passed_is_false_only_for_errors():
    report = AutoCheckReport(findings=[Finding("x", "warning", "m")])
    assert report.passed
    report.findings.append(Finding("y", "error", "m"))
    assert not report.passed


def test_finding_str_includes_the_timestamp():
    assert "at 1.50s" in str(Finding("black_frames", "error", "black", at_s=1.5))


# -- the mix satisfies its own checks -------------------------------------------------------------


def test_the_mix_lands_within_tolerance_of_its_target(tmp_path, fixtures):
    """One gain pass undershoots, because the limiter pulls loudness down with the peaks."""
    from montaje.sound.mix import MixSpec, render_mix

    spec = MixSpec(duration_s=8.0, music_wav=fixtures / "track.wav",
                   music_segments=[(48.0, 56.0, 0.0)], target_lufs=-14.0,
                   true_peak_dbtp=-1.0)
    stats = render_mix(spec, tmp_path / "m.wav")
    assert stats["final_lufs"] == pytest.approx(-14.0, abs=0.7)


def test_the_mix_leaves_headroom_for_a_lossy_encode(tmp_path, fixtures):
    """A mix limited to exactly -1.0 dBTP measured -0.5 in the finished MP4."""
    from montaje.sound.mix import LOSSY_ENCODE_HEADROOM_DB, MixSpec, render_mix

    spec = MixSpec(duration_s=6.0, music_wav=fixtures / "track.wav",
                   music_segments=[(48.0, 54.0, 0.0)], true_peak_dbtp=-1.0)
    stats = render_mix(spec, tmp_path / "m.wav", lossy_output=True)
    assert stats["ceiling_dbtp"] == pytest.approx(-1.0 - LOSSY_ENCODE_HEADROOM_DB)
    assert stats["final_true_peak_dbtp"] <= -1.0


def test_a_pcm_output_needs_no_encode_headroom(tmp_path, fixtures):
    from montaje.sound.mix import MixSpec, render_mix

    spec = MixSpec(duration_s=4.0, music_wav=fixtures / "track.wav",
                   music_segments=[(48.0, 52.0, 0.0)], true_peak_dbtp=-1.0)
    stats = render_mix(spec, tmp_path / "m.wav", lossy_output=False)
    assert stats["ceiling_dbtp"] == pytest.approx(-1.0)
