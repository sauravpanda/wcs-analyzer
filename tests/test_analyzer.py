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


def _mock_usage(input_tokens: int = 100, output_tokens: int = 200) -> MagicMock:
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    return usage


class TestCallClaude:
    def test_successful_call(self):
        mock_client = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "response text"
        mock_client.messages.create.return_value = MagicMock(
            content=[mock_block], usage=_mock_usage(150, 300),
        )

        text, usage = _call_claude(mock_client, "claude-sonnet-4-6", [{"type": "text", "text": "hi"}])
        assert text == "response text"
        assert usage.input_tokens == 150
        assert usage.output_tokens == 300
        # Sonnet pricing is $3/M input + $15/M output
        assert usage.estimated_cost > 0

    def test_unexpected_block_type_raises(self):
        mock_client = MagicMock()
        mock_block = MagicMock(spec=[])  # no .text attribute
        mock_client.messages.create.return_value = MagicMock(content=[mock_block], usage=_mock_usage())

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
            MagicMock(content=[mock_block], usage=_mock_usage()),
        ]

        text, _ = _call_claude(mock_client, "claude-sonnet-4-6", [{"type": "text", "text": "hi"}])
        assert text == "ok"
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


def _usage(input_tokens: int = 100, output_tokens: int = 200, model: str = "claude-sonnet-4-6"):
    from wcs_analyzer.pricing import UsageTotals
    return UsageTotals.from_counts(model, input_tokens, output_tokens)


class TestCallAndParseRetry:
    @patch("wcs_analyzer.analyzer._call_claude")
    def test_retry_on_parse_failure(self, mock_call: MagicMock):
        from wcs_analyzer.analyzer import _call_and_parse
        mock_call.side_effect = [("not json", _usage()), (VALID_RESPONSE, _usage(500, 1000))]
        mock_client = MagicMock()
        parsed = _call_and_parse(mock_client, "claude-sonnet-4-6", [{"type": "text", "text": "x"}], 0.0, 4.0)
        assert parsed.timing_score == 7.5
        assert mock_call.call_count == 2
        # Second call should include the corrective hint
        second_content = mock_call.call_args_list[1][0][2]
        assert any("could not be parsed" in str(c) for c in second_content)
        # Usage is summed across both calls
        assert parsed.usage.input_tokens == 600
        assert parsed.usage.output_tokens == 1200

    @patch("wcs_analyzer.analyzer._call_claude")
    def test_retry_failure_uses_default(self, mock_call: MagicMock):
        from wcs_analyzer.analyzer import _call_and_parse
        mock_call.side_effect = [("not json", _usage()), ("still not json", _usage())]
        mock_client = MagicMock()
        parsed = _call_and_parse(mock_client, "claude-sonnet-4-6", [{"type": "text", "text": "x"}], 0.0, 4.0)
        assert parsed.timing_score == 5.0
        assert "error" in parsed.raw_data
        assert mock_call.call_count == 2
        # Even failed runs accumulate cost
        assert parsed.usage.input_tokens == 200


class TestAnalyzeDance:
    @patch("wcs_analyzer.analyzer._call_claude")
    @patch("wcs_analyzer.analyzer.anthropic.Anthropic")
    def test_single_phrase_no_summary(self, mock_anthropic_cls: MagicMock, mock_call: MagicMock):
        mock_call.return_value = (VALID_RESPONSE, _usage())

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
        mock_call.return_value = (VALID_RESPONSE, _usage())

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
        mock_call.return_value = (json.dumps({
            "patterns": [
                {"start_time": 0.0, "end_time": 3.0, "name": "sugar push", "confidence": 0.85},
                {"start_time": 3.0, "end_time": 7.5, "name": "whip", "confidence": 0.7},
            ]
        }), _usage())
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
        mock_call.return_value = ("garbage not json", _usage())
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
        mock_call.return_value = (json.dumps({"patterns": []}), _usage())
        frames = FrameData(
            images=[f"i{i}" for i in range(200)], timestamps=[i * 0.1 for i in range(200)],
            fps_original=30.0, fps_sampled=10.0, duration=20.0, width=640, height=480,
        )
        detect_pattern_timeline(MagicMock(), "claude-sonnet-4-6", frames, max_frames=10)
        # Verify the content passed had <= 10 image blocks (plus 1 text block)
        content = mock_call.call_args[0][2]
        image_blocks = [c for c in content if c.get("type") == "image"]
        assert len(image_blocks) == 10


class TestExtractPatternDetailsStringFallback:
    def test_string_patterns_lack_quality_timing(self):
        """String-form patterns shouldn't get fabricated 'solid'/'on_beat' defaults."""
        from wcs_analyzer.analyzer import _extract_pattern_details
        details = _extract_pattern_details(["Sugar Push", "Whip"])
        assert len(details) == 2
        assert details[0]["name"] == "Sugar Push"
        assert "quality" not in details[0]
        assert "timing" not in details[0]

    def test_dict_patterns_preserve_quality_timing(self):
        from wcs_analyzer.analyzer import _extract_pattern_details
        details = _extract_pattern_details([
            {"name": "Sugar Push", "quality": "strong", "timing": "on_beat"},
            "Whip",  # Mixed list
        ])
        assert details[0]["quality"] == "strong"
        assert "quality" not in details[1]


class TestPatternTimelineOutputFormat:
    """New structured pattern_timeline output from Gemini forces the
    model to commit to time windows and enumerate every pattern.
    """

    def test_timeline_format_populates_patterns_and_details(self):
        data = {
            "timing": {"score": 6.5},
            "technique": {"score": 6.0},
            "teamwork": {"score": 7.0},
            "presentation": {"score": 6.0},
            "pattern_timeline": [
                {"start_time": 0.0, "end_time": 3.5, "patterns": ["starter step"], "quality": "solid", "timing": "on_beat", "notes": "clean lead-in"},
                {"start_time": 3.5, "end_time": 9.0, "patterns": ["sugar push", "free spin"], "quality": "strong", "timing": "on_beat", "notes": "nice free spin on 3&4"},
                {"start_time": 9.0, "end_time": 15.0, "patterns": ["whip"], "quality": "needs_work", "timing": "slightly_off", "notes": "lead rushed the anchor"},
            ],
        }
        seg = parse_segment_data(data, 0.0, 15.0)
        # patterns contains every name from the timeline
        assert "starter step" in seg.patterns
        assert "sugar push" in seg.patterns
        assert "free spin" in seg.patterns
        assert "whip" in seg.patterns
        # pattern_details carries quality/timing/notes per pattern
        names_with_quality = {p["name"]: p for p in seg.pattern_details}
        assert names_with_quality["sugar push"]["quality"] == "strong"
        assert names_with_quality["whip"]["timing"] == "slightly_off"
        # raw timeline preserved
        assert len(seg.pattern_timeline) == 3
        assert seg.pattern_timeline[0]["start_time"] == 0.0
        assert seg.pattern_timeline[1]["patterns"] == ["sugar push", "free spin"]

    def test_old_flat_format_still_works(self):
        """Cached segments from before this PR should still parse."""
        data = {
            "timing": {"score": 6},
            "technique": {"score": 6},
            "teamwork": {"score": 6},
            "presentation": {"score": 6},
            "patterns_identified": [
                {"name": "sugar push", "quality": "solid", "timing": "on_beat"},
                {"name": "whip", "quality": "strong", "timing": "on_beat"},
            ],
        }
        seg = parse_segment_data(data, 0.0, 4.0)
        assert seg.patterns == ["sugar push", "whip"]
        # Timeline wasn't in the response, so stays empty
        assert seg.pattern_timeline == []

    def test_malformed_timeline_entries_skipped(self):
        data = {
            "timing": {"score": 6},
            "technique": {"score": 6},
            "teamwork": {"score": 6},
            "presentation": {"score": 6},
            "pattern_timeline": [
                "not a dict",
                {"start_time": "bad", "end_time": 5, "patterns": ["whip"]},  # bad start
                {"start_time": 0.0, "end_time": 3.0, "patterns": ["sugar push"]},  # valid
                {"start_time": 3.0, "end_time": 6.0, "patterns": []},  # empty patterns → skip
                {"start_time": 6.0, "end_time": 9.0, "patterns": "not a list"},  # wrong type
            ],
        }
        seg = parse_segment_data(data, 0.0, 9.0)
        # Only the valid entry survives
        assert len(seg.pattern_timeline) == 1
        assert seg.pattern_timeline[0]["patterns"] == ["sugar push"]
        assert seg.patterns == ["sugar push"]

    def test_empty_timeline_falls_back_to_flat(self):
        data = {
            "timing": {"score": 6},
            "technique": {"score": 6},
            "teamwork": {"score": 6},
            "presentation": {"score": 6},
            "pattern_timeline": [],  # empty triggers fallback
            "patterns_identified": [{"name": "whip"}],
        }
        seg = parse_segment_data(data, 0.0, 4.0)
        assert seg.patterns == ["whip"]
