"""Tests for report formatting."""

import json
from pathlib import Path

from wcs_analyzer.report import _score_bar, _score_color, _format_time, _trend, save_report_json, save_report_csv
from wcs_analyzer.scoring import FinalScores, SegmentAnalysis


class TestHelpers:
    def test_score_bar_full(self):
        bar = _score_bar(10.0)
        assert len(bar) == 10
        assert "\u2591" not in bar  # no empty blocks

    def test_score_bar_empty(self):
        bar = _score_bar(0.0)
        assert "\u2588" not in bar  # no filled blocks

    def test_score_bar_half(self):
        bar = _score_bar(5.0)
        assert bar.count("\u2588") == 5
        assert bar.count("\u2591") == 5

    def test_score_color_green(self):
        assert _score_color(8.0) == "green"
        assert _score_color(10.0) == "green"

    def test_score_color_yellow(self):
        assert _score_color(6.0) == "yellow"
        assert _score_color(7.9) == "yellow"

    def test_score_color_orange(self):
        assert _score_color(4.0) == "dark_orange"

    def test_score_color_red(self):
        assert _score_color(3.9) == "red"
        assert _score_color(0.0) == "red"

    def test_format_time(self):
        assert _format_time(0) == "0:00"
        assert _format_time(65) == "1:05"
        assert _format_time(3661) == "61:01"


class TestSaveReportJson:
    def test_saves_valid_json(self, tmp_path: Path):
        scores = FinalScores(
            timing=7.5,
            technique=6.0,
            teamwork=8.0,
            presentation=7.0,
            overall=7.1,
            grade="B-",
            posture=7.0,
            extension=5.5,
            footwork=6.5,
            slot=6.0,
            total_off_beat=1,
            off_beat_moments=[{"time": "0:05", "description": "rushed"}],
            all_patterns=["Sugar Push"],
            top_strengths=["Good connection"],
            top_improvements=["Work on extension"],
            overall_impression="Solid performance",
            segments=[
                SegmentAnalysis(start_time=0.0, end_time=4.0, timing_score=7.5),
            ],
        )

        out = tmp_path / "report.json"
        save_report_json(scores, out)

        data = json.loads(out.read_text())
        assert data["scores"]["overall"] == 7.1
        assert data["scores"]["grade"] == "B-"
        assert len(data["segments"]) == 1
        assert data["patterns"] == ["Sugar Push"]
        assert "partner_breakdown" in data
        assert "lead" in data["partner_breakdown"]
        assert "follow" in data["partner_breakdown"]


class TestTrend:
    def test_improvement(self):
        result = _trend(8.0, 6.0)
        assert "\u2191" in result  # up arrow

    def test_regression(self):
        result = _trend(5.0, 7.0)
        assert "\u2193" in result  # down arrow

    def test_stable(self):
        result = _trend(7.0, 7.2)
        assert "\u2192" in result  # right arrow


class TestSaveReportCsv:
    def test_csv_has_expected_rows(self, tmp_path: Path):
        scores = FinalScores(
            timing=7.5, technique=6.0, teamwork=8.0, presentation=7.0,
            overall=7.1, grade="B-",
            posture=7.0, extension=5.5, footwork=6.5, slot=6.0,
            segments=[
                SegmentAnalysis(start_time=0.0, end_time=4.0, timing_score=7.5,
                                technique_score=6.0, teamwork_score=8.0, presentation_score=7.0),
            ],
        )
        out = tmp_path / "report.csv"
        save_report_csv(scores, out)

        content = out.read_text()
        assert "Overall" in content
        assert "Timing" in content
        assert "7.5" in content
