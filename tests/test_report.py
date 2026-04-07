"""Tests for report formatting."""

import json
from pathlib import Path

from wcs_analyzer.report import _score_bar, _score_color, _format_time, save_report_json
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
