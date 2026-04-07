"""Tests for analyzer module — Claude API response parsing and orchestration."""

import json
from unittest.mock import MagicMock, patch

import pytest

from wcs_analyzer.analyzer import (
    _parse_segment_json, _call_claude, _effective_max_frames, analyze_dance,
    MAX_FRAMES_PER_CALL, _IMAGE_TOKEN_BUDGET, _TOKENS_PER_IMAGE_ESTIMATE,
)
from wcs_analyzer.audio import AudioFeatures
from wcs_analyzer.exceptions import AnalysisError
from wcs_analyzer.video import FrameData


# -- _parse_segment_json tests --

VALID_RESPONSE = json.dumps({
    "timing": {"score": 7.5, "on_beat": True, "off_beat_moments": [], "notes": "Good"},
    "technique": {
        "score": 6.0,
        "posture": {"score": 7.0, "notes": "Upright"},
        "extension": {"score": 5.5, "notes": "Could stretch more"},
        "footwork": {"score": 6.5, "notes": "Clean triples"},
        "slot": {"score": 6.0, "notes": "Slight drift"},
        "notes": "Solid basics",
    },
    "teamwork": {"score": 8.0, "connection": "Good lead-follow", "notes": "Nice connection"},
    "presentation": {"score": 7.0, "musicality": "Hits breaks", "styling": "Some body rolls", "notes": "Engaging"},
    "patterns_identified": ["Sugar Push", "Left Side Pass"],
    "highlights": ["Nice anchor at 3.2s"],
    "improvements": ["Extend through the slot more"],
})


class TestParseSegmentJson:
    def test_valid_json(self):
        result = _parse_segment_json(VALID_RESPONSE, 0.0, 4.0)
        assert result.timing_score == 7.5
        assert result.technique_score == 6.0
        assert result.teamwork_score == 8.0
        assert result.presentation_score == 7.0
        assert result.posture_score == 7.0
        assert result.extension_score == 5.5
        assert result.footwork_score == 6.5
        assert result.slot_score == 6.0
        assert result.patterns == ["Sugar Push", "Left Side Pass"]
        assert result.highlights == ["Nice anchor at 3.2s"]
        assert result.improvements == ["Extend through the slot more"]

    def test_json_with_markdown_fences(self):
        wrapped = f"```json\n{VALID_RESPONSE}\n```"
        result = _parse_segment_json(wrapped, 0.0, 4.0)
        assert result.timing_score == 7.5

    def test_json_with_bare_fences(self):
        wrapped = f"```\n{VALID_RESPONSE}\n```"
        result = _parse_segment_json(wrapped, 1.0, 5.0)
        assert result.technique_score == 6.0

    def test_invalid_json_returns_defaults(self):
        result = _parse_segment_json("not json at all", 0.0, 4.0)
        assert result.timing_score == 5.0
        assert result.technique_score == 5.0
        assert result.teamwork_score == 5.0
        assert result.presentation_score == 5.0
        assert "error" in result.raw_data

    def test_partial_json_uses_defaults_for_missing(self):
        partial = json.dumps({"timing": {"score": 9.0}, "technique": {}})
        result = _parse_segment_json(partial, 0.0, 4.0)
        assert result.timing_score == 9.0
        assert result.technique_score == 5  # default when missing
        assert result.teamwork_score == 5  # missing entirely

    def test_empty_string_returns_defaults(self):
        result = _parse_segment_json("", 0.0, 4.0)
        assert result.timing_score == 5.0

    def test_timestamps_preserved(self):
        result = _parse_segment_json(VALID_RESPONSE, 2.5, 6.5)
        assert result.start_time == 2.5
        assert result.end_time == 6.5


class TestCallClaude:
    def test_successful_call(self):
        mock_client = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "response text"
        mock_client.messages.create.return_value = MagicMock(content=[mock_block])

        result = _call_claude(mock_client, "claude-sonnet-4-6", [{"type": "text", "text": "hi"}])
        assert result == "response text"

    def test_unexpected_block_type_raises(self):
        mock_client = MagicMock()
        mock_block = MagicMock(spec=[])  # no .text attribute
        mock_client.messages.create.return_value = MagicMock(content=[mock_block])

        with pytest.raises(AnalysisError, match="Unexpected response block type"):
            _call_claude(mock_client, "claude-sonnet-4-6", [{"type": "text", "text": "hi"}])

    @patch("wcs_analyzer.analyzer.time.sleep")
    def test_rate_limit_retries(self, mock_sleep: MagicMock):
        import anthropic as anth

        mock_client = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "ok"

        mock_client.messages.create.side_effect = [
            anth.RateLimitError.__new__(anth.RateLimitError),
            MagicMock(content=[mock_block]),
        ]

        result = _call_claude(mock_client, "claude-sonnet-4-6", [{"type": "text", "text": "hi"}])
        assert result == "ok"
        mock_sleep.assert_called_once_with(2)  # 2^(0+1) = 2

    @patch("wcs_analyzer.analyzer.time.sleep")
    def test_rate_limit_exhausted_raises(self, mock_sleep: MagicMock):
        import anthropic as anth

        mock_client = MagicMock()
        err = anth.RateLimitError.__new__(anth.RateLimitError)
        mock_client.messages.create.side_effect = [err, err, err]

        with pytest.raises(anth.RateLimitError):
            _call_claude(mock_client, "claude-sonnet-4-6", [{"type": "text", "text": "hi"}], max_retries=3)


class TestTokenAwareness:
    def test_effective_max_frames_respects_budget(self):
        max_f = _effective_max_frames()
        assert max_f <= MAX_FRAMES_PER_CALL
        assert max_f <= _IMAGE_TOKEN_BUDGET // _TOKENS_PER_IMAGE_ESTIMATE

    def test_effective_max_frames_positive(self):
        assert _effective_max_frames() > 0


class TestAnalyzeDance:
    @patch("wcs_analyzer.analyzer._call_claude")
    @patch("wcs_analyzer.analyzer.anthropic.Anthropic")
    def test_single_phrase_no_summary(self, mock_anthropic_cls: MagicMock, mock_call: MagicMock):
        mock_call.return_value = VALID_RESPONSE

        frames = FrameData(
            images=["img1", "img2"],
            timestamps=[0.0, 0.5],
            fps_original=30.0,
            fps_sampled=2.0,
            duration=1.0,
            width=640,
            height=480,
        )
        audio = AudioFeatures(bpm=120.0, beat_times=[0.5], beat_strengths=[0.8], duration=1.0)

        results = analyze_dance(frames, audio)
        # Single phrase → no summary call
        assert len(results) == 1
        assert results[0].timing_score == 7.5

    @patch("wcs_analyzer.analyzer._call_claude")
    @patch("wcs_analyzer.analyzer.anthropic.Anthropic")
    def test_multiple_phrases_include_summary(self, mock_anthropic_cls: MagicMock, mock_call: MagicMock):
        mock_call.return_value = VALID_RESPONSE

        frames = FrameData(
            images=[f"img_{i}" for i in range(20)],
            timestamps=[i * 0.5 for i in range(20)],
            fps_original=30.0,
            fps_sampled=2.0,
            duration=10.0,
            width=640,
            height=480,
        )
        audio = AudioFeatures(
            bpm=120.0,
            beat_times=[i * 0.5 for i in range(20)],
            beat_strengths=[0.8] * 20,
            duration=10.0,
        )

        results = analyze_dance(frames, audio)
        # Multiple phrases → last result is summary
        assert len(results) >= 3  # at least 2 phrases + 1 summary
