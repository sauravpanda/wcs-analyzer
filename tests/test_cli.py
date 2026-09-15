"""End-to-end CLI tests with the provider layer mocked out."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from wcs_analyzer.cli import _cache_key_model, main
from wcs_analyzer.scoring import SegmentAnalysis


def _segment(**overrides) -> SegmentAnalysis:
    base = dict(
        start_time=0.0, end_time=30.0,
        timing_score=7.0, technique_score=6.5, teamwork_score=7.5, presentation_score=7.0,
        is_summary=True,
        raw_data={"timing": {"score": 7}, "technique": {"score": 6.5}},
    )
    base.update(overrides)
    return SegmentAnalysis(**base)


@pytest.fixture
def video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake video file in an isolated cwd (reports are written next to it)."""
    monkeypatch.chdir(tmp_path)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not really a video" * 64)
    return clip


@pytest.fixture
def provider_mocks():
    """Mock everything that would touch the network, ffprobe, or ~/.wcs-analyzer."""
    with patch("wcs_analyzer.cli._analyze_with_claude_code", return_value=[_segment()]) as cc, \
         patch("wcs_analyzer.cache.get_cached_result", return_value=None) as get_cached, \
         patch("wcs_analyzer.cache.save_to_cache") as save_cache, \
         patch("wcs_analyzer.video.get_video_recorded_at", return_value="2026-01-01T00:00:00+00:00"):
        yield {"claude_code": cc, "get_cached": get_cached, "save_cache": save_cache}


class TestCacheKeyModel:
    def test_default_resolution_keeps_legacy_key(self):
        # Caches written before --hd existed must keep hitting
        assert _cache_key_model("claude-code", "m", None, 768) == "claude-code:m:"

    def test_hd_changes_key_for_claude_providers(self):
        assert _cache_key_model("claude-code", "m", None, 1080).endswith(":1080px")
        assert _cache_key_model("claude", "m", "lead in blue", 1080) == "claude:m:lead in blue:1080px"

    def test_gemini_ignores_resolution(self):
        assert _cache_key_model("gemini", "m", None, 1080) == "gemini:m:"


class TestAnalyzeHd:
    def test_hd_flag_sets_1080_and_separate_cache_key(self, video: Path, provider_mocks: dict):
        result = CliRunner().invoke(main, ["analyze", str(video), "--provider", "claude-code", "--hd"])
        assert result.exit_code == 0, result.output

        cc: MagicMock = provider_mocks["claude_code"]
        assert cc.call_args[0][4] == 1080  # max_dimension positional
        assert "Frame resolution: 1080px" in result.output

        get_cached: MagicMock = provider_mocks["get_cached"]
        assert get_cached.call_args[0][3].endswith(":1080px")

    def test_max_dimension_overrides_hd(self, video: Path, provider_mocks: dict):
        result = CliRunner().invoke(
            main, ["analyze", str(video), "--provider", "claude-code", "--hd", "--max-dimension", "900"],
        )
        assert result.exit_code == 0, result.output
        assert provider_mocks["claude_code"].call_args[0][4] == 900

    def test_default_resolution_is_768(self, video: Path, provider_mocks: dict):
        result = CliRunner().invoke(main, ["analyze", str(video), "--provider", "claude-code"])
        assert result.exit_code == 0, result.output
        assert provider_mocks["claude_code"].call_args[0][4] == 768
        assert "Frame resolution" not in result.output
        assert "px" not in provider_mocks["get_cached"].call_args[0][3]

    def test_non_positive_max_dimension_rejected(self, video: Path, provider_mocks: dict):
        result = CliRunner().invoke(
            main, ["analyze", str(video), "--provider", "claude-code", "--max-dimension", "0"],
        )
        assert result.exit_code == 1
        assert "positive" in result.output


class TestAnalyzeModelPassThrough:
    def test_default_model_reaches_claude_code_provider(self, video: Path, provider_mocks: dict):
        result = CliRunner().invoke(main, ["analyze", str(video), "--provider", "claude-code"])
        assert result.exit_code == 0, result.output
        assert provider_mocks["claude_code"].call_args[0][5] == "claude-opus-5"

    def test_explicit_model_reaches_claude_code_provider(self, video: Path, provider_mocks: dict):
        result = CliRunner().invoke(
            main, ["analyze", str(video), "--provider", "claude-code", "--model", "claude-fable-5-1"],
        )
        assert result.exit_code == 0, result.output
        assert provider_mocks["claude_code"].call_args[0][5] == "claude-fable-5-1"


class TestAnalyzeSaveReport:
    def test_save_report_writes_plain_text(self, video: Path, provider_mocks: dict, tmp_path: Path):
        out = tmp_path / "report.txt"
        result = CliRunner().invoke(
            main, ["analyze", str(video), "--provider", "claude-code", "-r", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "WCS Dance Analysis Report" in text
        assert "Overall Score" in text
        assert "Text report saved to" in result.output
        # The JSON copy is still written as before
        assert (tmp_path / "clip_report.json").exists()


class TestAnalyzeParseFailure:
    def test_placeholder_result_is_flagged(self, video: Path, provider_mocks: dict):
        failed = _segment(
            timing_score=5.0, technique_score=5.0, teamwork_score=5.0, presentation_score=5.0,
            raw_data={"error": "Failed to parse response", "raw": "garbage"},
        )
        provider_mocks["claude_code"].return_value = [failed]

        result = CliRunner().invoke(main, ["analyze", str(video), "--provider", "claude-code"])
        assert result.exit_code == 0, result.output
        assert "could not be parsed" in result.output
        assert "placeholder" in result.output


class TestTimingCommand:
    def test_accepts_claude_code_provider(self, video: Path, provider_mocks: dict):
        result = CliRunner().invoke(main, ["timing", str(video), "--provider", "claude-code"])
        assert result.exit_code == 0, result.output
        assert "Timing Score" in result.output
        provider_mocks["claude_code"].assert_called_once()


class TestCoachCommand:
    def test_coach_runs_with_opus_default_and_prints_report(self, video: Path, tmp_path: Path):
        from wcs_analyzer.coach import CoachReport

        fake = CoachReport(
            video_name="clip.mp4", duration=30.0, model="claude-opus-5",
            focus={"description": "lead wearing bib 1", "confidence": 0.9},
            themes=[{"title": "Look up", "detail": "", "examples": []}],
        )
        with patch("wcs_analyzer.coach.run_coach", return_value=fake) as rc, \
             patch("wcs_analyzer.coach_report.write_reports",
                   return_value=(tmp_path / "coach_report.md", tmp_path / "coach_report.html")):
            result = CliRunner().invoke(
                main, ["coach", str(video), "--dancers", "lead wearing bib 1", "--division", "intermediate"],
            )
        assert result.exit_code == 0, result.output
        assert rc.call_args.kwargs["model"] == "claude-opus-5"
        assert rc.call_args.kwargs["zoom"] is True
        assert rc.call_args.kwargs["division"] == "intermediate"
        assert "Look up" in result.output
        assert "coach_report.html" in result.output

    def test_coach_no_zoom_flag(self, video: Path, tmp_path: Path):
        from wcs_analyzer.coach import CoachReport

        fake = CoachReport(video_name="clip.mp4", duration=30.0, model="claude-opus-5")
        with patch("wcs_analyzer.coach.run_coach", return_value=fake) as rc, \
             patch("wcs_analyzer.coach_report.write_reports", return_value=(tmp_path / "a.md", tmp_path / "a.html")):
            result = CliRunner().invoke(main, ["coach", str(video), "--no-zoom", "--zoom-moments", "3"])
        assert result.exit_code == 0, result.output
        assert rc.call_args.kwargs["zoom"] is False
        assert rc.call_args.kwargs["zoom_moments"] == 3

    def test_coach_rerender_uses_saved_json(self, video: Path, tmp_path: Path):
        import json
        from dataclasses import asdict

        from wcs_analyzer.coach import CoachReport

        out = tmp_path / "out"
        out.mkdir()
        saved = CoachReport(video_name="clip.mp4", duration=30.0, model="claude-opus-5",
                            themes=[{"title": "Look up", "detail": "", "examples": []}])
        (out / "coach.json").write_text(json.dumps(asdict(saved), default=str))
        with patch("wcs_analyzer.coach.run_coach") as rc:
            result = CliRunner().invoke(main, ["coach", str(video), "-o", str(out), "--rerender"])
        assert result.exit_code == 0, result.output
        rc.assert_not_called()
        assert (out / "coach_report.html").exists()
        assert "Re-rendered" in result.output

    def test_coach_rerender_without_json_fails_cleanly(self, video: Path, tmp_path: Path):
        result = CliRunner().invoke(main, ["coach", str(video), "-o", str(tmp_path / "nope"), "--rerender"])
        assert result.exit_code == 1
        assert "No coach.json" in result.output
