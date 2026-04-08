"""Tests for Claude Code CLI analyzer."""

import json
from unittest.mock import MagicMock, patch

import pytest

from wcs_analyzer.claude_code_analyzer import (
    _check_claude_cli,
    _parse_response,
    _ANALYSIS_SCHEMA,
)
from wcs_analyzer.exceptions import AnalysisError


VALID_ANALYSIS = {
    "timing": {"score": 7.5, "on_beat": True, "off_beat_moments": [], "notes": "Good timing"},
    "technique": {
        "score": 6.5,
        "posture": {"score": 7.0, "notes": "Upright"},
        "extension": {"score": 6.0, "notes": "Could extend more"},
        "footwork": {"score": 6.5, "notes": "Clean"},
        "slot": {"score": 6.5, "notes": "Slight drift"},
        "notes": "Solid basics",
    },
    "teamwork": {"score": 8.0, "notes": "Good connection"},
    "presentation": {"score": 7.0, "notes": "Engaging"},
    "patterns_identified": ["Sugar Push", "Whip"],
    "highlights": ["Nice anchor"],
    "improvements": ["Extend arms more"],
    "lead": {"technique_score": 6.5, "presentation_score": 7.0, "notes": "Good leads"},
    "follow": {"technique_score": 7.0, "presentation_score": 7.5, "notes": "Responsive"},
    "overall_impression": "Solid intermediate performance",
}


class TestCheckClaudeCli:
    @patch("shutil.which", return_value="/usr/local/bin/claude")
    def test_found(self, mock_which: MagicMock):
        assert _check_claude_cli() == "/usr/local/bin/claude"

    @patch("shutil.which", return_value=None)
    def test_not_found_raises(self, mock_which: MagicMock):
        with pytest.raises(AnalysisError, match="Claude Code CLI not found"):
            _check_claude_cli()


class TestParseResponse:
    def test_valid_response(self):
        result = _parse_response(VALID_ANALYSIS, duration=120.0)
        assert result.timing_score == 7.5
        assert result.technique_score == 6.5
        assert result.teamwork_score == 8.0
        assert result.posture_score == 7.0
        assert result.lead_technique == 6.5
        assert result.follow_presentation == 7.5
        assert result.end_time == 120.0
        assert "Sugar Push" in result.patterns

    def test_missing_fields_use_defaults(self):
        minimal = {"timing": {"score": 8}, "technique": {"score": 7}, "teamwork": {"score": 6}, "presentation": {"score": 9}}
        result = _parse_response(minimal, duration=60.0)
        assert result.timing_score == 8.0
        assert result.posture_score == 5.0  # default
        assert result.lead_technique == 0.0  # default


class TestAnalysisSchema:
    def test_schema_is_valid_json(self):
        parsed = json.loads(_ANALYSIS_SCHEMA)
        assert parsed["type"] == "object"
        assert "timing" in parsed["properties"]
        assert "technique" in parsed["properties"]
        assert "lead" in parsed["properties"]
        assert "follow" in parsed["properties"]
