"""Tests for analyzer module — Claude API response parsing and orchestration."""

import json
from unittest.mock import MagicMock, patch

import pytest

from wcs_analyzer.analyzer import (
    _parse_segment_json, _call_claude, _effective_max_frames, analyze_dance,
    _parse_pattern_timeline, clamp_score, clamp_partner, detect_pattern_timeline,
    parse_segment_data, safe_parse_json,
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


class TestClampScore:
    def test_in_range_unchanged(self):
        assert clamp_score(7.5) == 7.5
        assert clamp_score(1.0) == 1.0
        assert clamp_score(10.0) == 10.0

    def test_above_ten_clamped(self):
        assert clamp_score(15) == 10.0
        assert clamp_score(10.5) == 10.0

    def test_below_one_clamped_or_defaulted(self):
        # 0 or negative is treated as missing -> default
        assert clamp_score(0) == 5.0
        assert clamp_score(-3) == 5.0
        # Positive-but-small clamps up to the floor
        assert clamp_score(0.5) == 1.0

    def test_invalid_types_return_default(self):
        assert clamp_score("not a number") == 5.0
        assert clamp_score(None) == 5.0
        assert clamp_score({}) == 5.0

    def test_nan_returns_default(self):
        assert clamp_score(float("nan")) == 5.0

    def test_custom_default(self):
        assert clamp_score(None, default=3.0) == 3.0


class TestClampPartner:
    def test_zero_preserved(self):
        assert clamp_partner(0) == 0.0
        assert clamp_partner(None) == 0.0

    def test_in_range(self):
        assert clamp_partner(7.5) == 7.5

    def test_out_of_range_clamped(self):
        assert clamp_partner(15) == 10.0
        assert clamp_partner(0.3) == 1.0


class TestParseSegmentData:
    def test_out_of_range_scores_clamped(self):
        data = {
            "timing": {"score": 15},
            "technique": {"score": -2},
            "teamwork": {"score": 10.5},
            "presentation": {"score": "bad"},
        }
        seg = parse_segment_data(data, 0.0, 4.0)
        assert seg.timing_score == 10.0
        assert seg.technique_score == 5.0  # negative -> default
        assert seg.teamwork_score == 10.0
        assert seg.presentation_score == 5.0

    def test_confidence_intervals_parsed(self):
        data = {
            "timing": {"score": 7.5, "score_low": 7.0, "score_high": 8.0},
            "technique": {"score": 6.0, "score_low": 4.5, "score_high": 7.5},
            "teamwork": {"score": 8.0},
            "presentation": {"score": 7.0},
        }
        seg = parse_segment_data(data, 0.0, 4.0)
        assert seg.timing_low == 7.0
        assert seg.timing_high == 8.0
        assert seg.technique_low == 4.5
        assert seg.technique_high == 7.5
        # Missing CI falls back to point estimate
        assert seg.teamwork_low == 8.0
        assert seg.teamwork_high == 8.0

    def test_ci_out_of_order_swapped(self):
        data = {"timing": {"score": 7.0, "score_low": 8.0, "score_high": 6.0}}
        seg = parse_segment_data(data, 0.0, 4.0)
        assert seg.timing_low <= seg.timing_high
        assert seg.timing_low <= 7.0 <= seg.timing_high

    def test_ci_clamped_to_range(self):
        data = {"timing": {"score": 7.0, "score_low": -5, "score_high": 99}}
        seg = parse_segment_data(data, 0.0, 4.0)
        assert seg.timing_low >= 1.0
        assert seg.timing_high <= 10.0

    def test_reasoning_parsed_per_category(self):
        data = {
            "timing": {"score": 7, "reasoning": "Anchors land clean on 3&4"},
            "technique": {"score": 7, "reasoning": "Posture solid, slight slot drift"},
            "teamwork": {"score": 7},  # missing reasoning — fine
            "presentation": {"score": 7, "reasoning": ""},  # empty — skip
        }
        seg = parse_segment_data(data, 0.0, 4.0)
        assert seg.reasoning == {
            "timing": "Anchors land clean on 3&4",
            "technique": "Posture solid, slight slot drift",
        }
        assert "teamwork" not in seg.reasoning
        assert "presentation" not in seg.reasoning

    def test_summary_flat_sub_scores(self):
        # SUMMARY_PROMPT emits flat posture_score, not nested posture.score
        data = {
            "timing": {"score": 7},
            "technique": {
                "score": 7,
                "posture_score": 8,
                "extension_score": 6,
                "footwork_score": 7.5,
                "slot_score": 6.5,
            },
            "teamwork": {"score": 7},
            "presentation": {"score": 7},
        }
        seg = parse_segment_data(data, 0.0, 4.0)
        assert seg.posture_score == 8.0
        assert seg.extension_score == 6.0
        assert seg.footwork_score == 7.5
        assert seg.slot_score == 6.5


class TestSafeParseJson:
    def test_plain_json(self):
        assert safe_parse_json('{"a": 1}') == {"a": 1}

    def test_markdown_fence(self):
        assert safe_parse_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_bad_returns_none(self):
        assert safe_parse_json("not json") is None

    def test_non_dict_returns_none(self):
        assert safe_parse_json("[1, 2, 3]") is None


class TestCallAndParseRetry:
    @patch("wcs_analyzer.analyzer._call_claude")
    def test_retry_on_parse_failure(self, mock_call: MagicMock):
        from wcs_analyzer.analyzer import _call_and_parse
        mock_call.side_effect = ["not json", VALID_RESPONSE]
        mock_client = MagicMock()
        parsed = _call_and_parse(mock_client, "claude-sonnet-4-6", [{"type": "text", "text": "x"}], 0.0, 4.0)
        assert parsed.timing_score == 7.5
        assert mock_call.call_count == 2
        # Second call should include the corrective hint
        second_content = mock_call.call_args_list[1][0][2]
        assert any("could not be parsed" in str(c) for c in second_content)

    @patch("wcs_analyzer.analyzer._call_claude")
    def test_retry_failure_uses_default(self, mock_call: MagicMock):
        from wcs_analyzer.analyzer import _call_and_parse
        mock_call.side_effect = ["not json", "still not json"]
        mock_client = MagicMock()
        parsed = _call_and_parse(mock_client, "claude-sonnet-4-6", [{"type": "text", "text": "x"}], 0.0, 4.0)
        assert parsed.timing_score == 5.0
        assert "error" in parsed.raw_data
        assert mock_call.call_count == 2


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


# ---- Pattern segmentation tests -------------------------------------------


class TestParsePatternTimeline:
    def test_valid_timeline_parsed_and_sorted(self):
        items = [
            {"start_time": 3.5, "end_time": 7.0, "name": "whip", "confidence": 0.8},
            {"start_time": 0.0, "end_time": 3.5, "name": "sugar push", "confidence": 0.9},
        ]
        segs = _parse_pattern_timeline(items, duration=10.0)
        assert len(segs) == 2
        assert segs[0].name == "sugar push"
        assert segs[0].start_time == 0.0
        assert segs[1].name == "whip"

    def test_inverted_range_dropped(self):
        items = [
            {"start_time": 3.0, "end_time": 2.0, "name": "bad"},
            {"start_time": 0.0, "end_time": 2.0, "name": "good"},
        ]
        segs = _parse_pattern_timeline(items, duration=10.0)
        assert len(segs) == 1
        assert segs[0].name == "good"

    def test_clamps_to_duration(self):
        items = [{"start_time": 0.0, "end_time": 15.0, "name": "sugar push"}]
        segs = _parse_pattern_timeline(items, duration=10.0)
        assert segs[0].end_time == 10.0

    def test_invalid_confidence_defaults_to_zero(self):
        items = [{"start_time": 0.0, "end_time": 2.0, "name": "p", "confidence": "bad"}]
        segs = _parse_pattern_timeline(items, duration=10.0)
        assert segs[0].confidence == 0.0

    def test_confidence_clamped_to_unit_interval(self):
        items = [
            {"start_time": 0.0, "end_time": 1.0, "name": "a", "confidence": 1.5},
            {"start_time": 1.0, "end_time": 2.0, "name": "b", "confidence": -0.3},
        ]
        segs = _parse_pattern_timeline(items, duration=10.0)
        assert segs[0].confidence == 1.0
        assert segs[1].confidence == 0.0

    def test_missing_name_defaults_to_unknown(self):
        items = [{"start_time": 0.0, "end_time": 1.0}]
        segs = _parse_pattern_timeline(items, duration=10.0)
        assert segs[0].name == "unknown"

    def test_non_dict_items_skipped(self):
        items = ["not a dict", None, {"start_time": 0.0, "end_time": 1.0, "name": "ok"}]
        segs = _parse_pattern_timeline(items, duration=10.0)
        assert len(segs) == 1


class TestDetectPatternTimeline:
    @patch("wcs_analyzer.analyzer._call_claude")
    def test_successful_detection(self, mock_call: MagicMock):
        mock_call.return_value = json.dumps({
            "patterns": [
                {"start_time": 0.0, "end_time": 3.0, "name": "sugar push", "confidence": 0.85},
                {"start_time": 3.0, "end_time": 7.5, "name": "whip", "confidence": 0.7},
            ]
        })
        frames = FrameData(
            images=["img"] * 20, timestamps=[i * 0.5 for i in range(20)],
            fps_original=30.0, fps_sampled=2.0, duration=10.0, width=640, height=480,
        )
        client = MagicMock()
        segs = detect_pattern_timeline(client, "claude-sonnet-4-6", frames)
        assert len(segs) == 2
        assert segs[0].name == "sugar push"

    @patch("wcs_analyzer.analyzer._call_claude")
    def test_unparseable_returns_empty(self, mock_call: MagicMock):
        mock_call.return_value = "garbage not json"
        frames = FrameData(
            images=["img"] * 5, timestamps=[0.0, 1.0, 2.0, 3.0, 4.0],
            fps_original=30.0, fps_sampled=1.0, duration=5.0, width=640, height=480,
        )
        assert detect_pattern_timeline(MagicMock(), "claude-sonnet-4-6", frames) == []

    def test_empty_frames_returns_empty(self):
        frames = FrameData()
        assert detect_pattern_timeline(MagicMock(), "claude-sonnet-4-6", frames) == []

    @patch("wcs_analyzer.analyzer._call_claude")
    def test_subsamples_long_videos(self, mock_call: MagicMock):
        mock_call.return_value = json.dumps({"patterns": []})
        frames = FrameData(
            images=[f"i{i}" for i in range(200)], timestamps=[i * 0.1 for i in range(200)],
            fps_original=30.0, fps_sampled=10.0, duration=20.0, width=640, height=480,
        )
        detect_pattern_timeline(MagicMock(), "claude-sonnet-4-6", frames, max_frames=10)
        # Verify the content passed had <= 10 image blocks (plus 1 text block)
        content = mock_call.call_args[0][2]
        image_blocks = [c for c in content if c.get("type") == "image"]
        assert len(image_blocks) == 10
