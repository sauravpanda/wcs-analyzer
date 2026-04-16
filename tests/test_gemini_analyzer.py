"""Tests for Gemini video analyzer."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from wcs_analyzer.gemini_analyzer import (
    _parse_response,
    _DETAIL_FPS,
    _INLINE_LIMIT,
    analyze_dance_gemini,
)
from wcs_analyzer.prompts import DANCER_CONTEXT_TEMPLATE, GEMINI_VIDEO_PROMPT

VALID_GEMINI_RESPONSE = json.dumps({
    "timing": {"score": 8.0, "on_beat": True, "off_beat_moments": [], "rhythm_consistency": "Strong", "notes": "Solid timing"},
    "technique": {
        "score": 7.0,
        "posture": {"score": 7.5, "notes": "Good frame"},
        "extension": {"score": 6.5, "notes": "Could extend more"},
        "footwork": {"score": 7.0, "notes": "Clean triples"},
        "slot": {"score": 7.0, "notes": "Stays in slot"},
        "notes": "Solid technique",
    },
    "teamwork": {"score": 8.5, "connection": "Great connection", "notes": "Well matched"},
    "presentation": {"score": 7.5, "musicality": "Hits the breaks", "styling": "Nice body rolls", "notes": "Engaging"},
    "patterns_identified": ["Sugar Push", "Whip", "Left Side Pass"],
    "highlights": ["Great whip at 0:15"],
    "improvements": ["Work on arm styling"],
    "lead": {"technique_score": 7.0, "presentation_score": 7.0, "notes": "Solid leads"},
    "follow": {"technique_score": 7.5, "presentation_score": 8.0, "notes": "Great responsiveness"},
    "overall_impression": "A strong intermediate performance",
    "estimated_bpm": 108,
    "song_style": "contemporary",
})


class TestParseResponse:
    def test_valid_json(self):
        result = _parse_response(VALID_GEMINI_RESPONSE)
        assert result.timing_score == 8.0
        assert result.technique_score == 7.0
        assert result.teamwork_score == 8.5
        assert result.posture_score == 7.5
        assert result.lead_technique == 7.0
        assert result.follow_presentation == 8.0
        assert "Sugar Push" in result.patterns
        assert result.raw_data["estimated_bpm"] == 108

    def test_json_with_fences(self):
        wrapped = f"```json\n{VALID_GEMINI_RESPONSE}\n```"
        result = _parse_response(wrapped)
        assert result.timing_score == 8.0

    def test_invalid_json_returns_defaults(self):
        result = _parse_response("This is not JSON")
        assert result.timing_score == 5.0
        assert "error" in result.raw_data

    def test_empty_string_returns_defaults(self):
        result = _parse_response("")
        assert result.timing_score == 5.0


class TestDetailFPS:
    def test_fps_levels(self):
        # Medium bumped from 2 → 3 so fast WCS transitions don't slip
        # between samples when running pattern pre-pass + analysis.
        assert _DETAIL_FPS["low"] == 1
        assert _DETAIL_FPS["medium"] == 3
        assert _DETAIL_FPS["high"] == 5


class TestInlineLimit:
    def test_inline_limit_is_20mb(self):
        assert _INLINE_LIMIT == 20 * 1024 * 1024


class TestAnalyzeDanceGemini:
    @patch("wcs_analyzer.gemini_analyzer._call_gemini")
    @patch("wcs_analyzer.gemini_analyzer._inline_video")
    @patch("wcs_analyzer.gemini_analyzer.genai.Client")
    def test_small_file_uses_inline(self, mock_client_cls: MagicMock, mock_inline: MagicMock, mock_call: MagicMock, tmp_path: Path):
        # Create a small file (under 20MB)
        video = tmp_path / "small.mp4"
        video.write_bytes(b"x" * 1000)

        from wcs_analyzer.pricing import UsageTotals
        mock_inline.return_value = MagicMock()
        usage = UsageTotals.from_counts("gemini-2.5-flash", 1000, 500)
        mock_call.return_value = (VALID_GEMINI_RESPONSE, usage)

        # Disable the pattern pre-pass so we only get one _call_gemini
        # invocation and the token counts are deterministic.
        results = analyze_dance_gemini(
            video, model="gemini-2.5-flash", detail="medium", pattern_pre_pass=False,
        )
        assert len(results) == 1
        assert results[0].timing_score == 8.0
        assert results[0].usage.input_tokens == 1000
        mock_inline.assert_called_once()

    @patch("wcs_analyzer.gemini_analyzer._call_gemini")
    @patch("wcs_analyzer.gemini_analyzer._upload_video")
    @patch("wcs_analyzer.gemini_analyzer.genai.Client")
    def test_large_file_uses_upload(self, mock_client_cls: MagicMock, mock_upload: MagicMock, mock_call: MagicMock, tmp_path: Path):
        # Create a file over 20MB
        video = tmp_path / "large.mp4"
        video.write_bytes(b"x" * (_INLINE_LIMIT + 1))

        from wcs_analyzer.pricing import UsageTotals
        mock_upload.return_value = MagicMock()
        mock_call.return_value = (VALID_GEMINI_RESPONSE, UsageTotals.from_counts("gemini-2.5-flash", 1, 1))

        results = analyze_dance_gemini(
            video, model="gemini-2.5-flash", detail="medium", pattern_pre_pass=False,
        )
        assert len(results) == 1
        mock_upload.assert_called_once()


class TestDancerContext:
    def test_dancer_template_formats(self):
        ctx = DANCER_CONTEXT_TEMPLATE.format(dancer_description="lead in blue, follow in red")
        assert "lead in blue, follow in red" in ctx
        assert "DANCER IDENTIFICATION" in ctx

    def test_gemini_prompt_accepts_dancer_context(self):
        ctx = DANCER_CONTEXT_TEMPLATE.format(dancer_description="couple on the left")
        prompt = GEMINI_VIDEO_PROMPT.replace("{dancer_context}", ctx + "\n")
        assert "couple on the left" in prompt
        assert "Watch and listen" in prompt

    def test_gemini_prompt_empty_dancer_context(self):
        prompt = GEMINI_VIDEO_PROMPT.replace("{dancer_context}", "")
        assert "Watch and listen" in prompt
        assert "{dancer_context}" not in prompt


class TestGeminiPatternPrePass:
    @patch("wcs_analyzer.gemini_analyzer._call_gemini")
    def test_pre_pass_feeds_timeline_into_main_call(self, mock_call: MagicMock):
        """When pattern_pre_pass is on, the main call should receive a
        prompt that names the detected patterns."""
        import json
        from wcs_analyzer.pricing import UsageTotals

        # Two sequential _call_gemini invocations: first is pattern
        # detection, second is main analysis.
        pattern_response = json.dumps({
            "patterns": [
                {"start_time": 0.0, "end_time": 4.0, "name": "sugar push", "confidence": 0.9},
                {"start_time": 4.0, "end_time": 8.0, "name": "whip", "confidence": 0.8},
            ]
        })
        mock_call.side_effect = [
            (pattern_response, UsageTotals.from_counts("gemini-3.1-pro-preview", 500, 200)),
            (VALID_GEMINI_RESPONSE, UsageTotals.from_counts("gemini-3.1-pro-preview", 1500, 800)),
        ]

        with patch("wcs_analyzer.gemini_analyzer._inline_video") as mock_inline, \
             patch("wcs_analyzer.gemini_analyzer.genai.Client"):
            mock_inline.return_value = MagicMock()
            video = Path("/tmp/fake.mp4")
            video.write_bytes(b"x" * 1000)
            try:
                results = analyze_dance_gemini(
                    video, model="gemini-3.1-pro-preview", detail="medium",
                    pattern_pre_pass=True,
                )
            finally:
                video.unlink(missing_ok=True)

        assert len(results) == 1
        assert mock_call.call_count == 2
        # The second call's prompt should reference the detected patterns
        second_call_contents = mock_call.call_args_list[1].kwargs["contents"]
        main_prompt = second_call_contents[-1]  # Last element is the text prompt
        assert "sugar push" in main_prompt.lower()
        assert "whip" in main_prompt.lower()
        assert "DETECTED PATTERN TIMELINE" in main_prompt
        # Combined usage from both calls
        assert results[0].usage.input_tokens == 2000
        assert results[0].usage.output_tokens == 1000

    @patch("wcs_analyzer.gemini_analyzer._call_gemini")
    def test_pre_pass_gracefully_skips_on_unparseable_response(self, mock_call: MagicMock):
        """If the pre-pass returns garbage, the main analysis still runs."""
        from wcs_analyzer.pricing import UsageTotals
        mock_call.side_effect = [
            ("garbage not json", UsageTotals.from_counts("gemini-3.1-pro-preview", 100, 50)),
            (VALID_GEMINI_RESPONSE, UsageTotals.from_counts("gemini-3.1-pro-preview", 1500, 800)),
        ]
        with patch("wcs_analyzer.gemini_analyzer._inline_video") as mock_inline, \
             patch("wcs_analyzer.gemini_analyzer.genai.Client"):
            mock_inline.return_value = MagicMock()
            video = Path("/tmp/fake2.mp4")
            video.write_bytes(b"x" * 1000)
            try:
                results = analyze_dance_gemini(
                    video, model="gemini-3.1-pro-preview", detail="medium",
                    pattern_pre_pass=True,
                )
            finally:
                video.unlink(missing_ok=True)
        assert len(results) == 1
        # Both calls still counted toward usage
        assert mock_call.call_count == 2


class TestFormatPatternTimelineForPrompt:
    def test_format_includes_all_patterns(self):
        from wcs_analyzer.gemini_analyzer import _format_pattern_timeline_for_prompt
        from wcs_analyzer.analyzer import PatternSegment
        timeline = [
            PatternSegment(start_time=0.0, end_time=3.5, name="sugar push", confidence=0.9),
            PatternSegment(start_time=3.5, end_time=8.0, name="whip", confidence=0.7),
        ]
        text = _format_pattern_timeline_for_prompt(timeline)
        assert "sugar push" in text
        assert "whip" in text
        assert "DETECTED PATTERN TIMELINE" in text
        assert "trust" in text.lower()

    def test_format_empty_timeline(self):
        from wcs_analyzer.gemini_analyzer import _format_pattern_timeline_for_prompt
        text = _format_pattern_timeline_for_prompt([])
        assert "DETECTED PATTERN TIMELINE" in text
