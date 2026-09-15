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


class TestExtractJsonFromProse:
    def test_fenced_json_after_prose(self):
        from wcs_analyzer.claude_code_analyzer import _extract_json_from_prose
        text = 'I reviewed the frames. Here is my analysis:\n\n```json\n{"timing": {"score": 7}, "technique": {"score": 8}}\n```\n\nLet me know if you need more.'
        result = _extract_json_from_prose(text)
        assert result == {"timing": {"score": 7}, "technique": {"score": 8}}

    def test_bare_json_in_prose(self):
        from wcs_analyzer.claude_code_analyzer import _extract_json_from_prose
        text = 'Analysis complete: {"timing": {"score": 7.5}, "notes": "good"} — see details.'
        result = _extract_json_from_prose(text)
        assert result is not None
        assert result["timing"]["score"] == 7.5

    def test_handles_strings_with_braces(self):
        """Braces inside string values shouldn't confuse the balance counter."""
        from wcs_analyzer.claude_code_analyzer import _extract_json_from_prose
        text = '```json\n{"notes": "uses { and } chars", "score": 5}\n```'
        result = _extract_json_from_prose(text)
        assert result == {"notes": "uses { and } chars", "score": 5}

    def test_handles_escaped_quotes(self):
        from wcs_analyzer.claude_code_analyzer import _extract_json_from_prose
        text = '{"notes": "she said \\"nice whip\\"", "score": 7}'
        result = _extract_json_from_prose(text)
        assert result is not None
        assert result["score"] == 7

    def test_returns_none_on_no_json(self):
        from wcs_analyzer.claude_code_analyzer import _extract_json_from_prose
        assert _extract_json_from_prose("no json here at all") is None

    def test_picks_first_valid_of_multiple(self):
        from wcs_analyzer.claude_code_analyzer import _extract_json_from_prose
        text = 'First: {"a": 1}. Second: {"b": 2}.'
        result = _extract_json_from_prose(text)
        assert result == {"a": 1}


class TestPromptConstruction:
    """The prompt must describe the frames the model actually gets."""

    def _run(self, detail: str = "medium", **kwargs):
        from pathlib import Path

        from wcs_analyzer.claude_code_analyzer import analyze_dance_claude_code
        from wcs_analyzer.pricing import UsageTotals
        from wcs_analyzer.video import FrameData

        frames = FrameData(
            images=["aGk=", "aGk=", "aGk="],  # base64("hi")
            timestamps=[0.0, 0.5, 1.0],
            fps_original=30.0, fps_sampled=2.0, duration=1.5, width=640, height=480,
        )
        with patch("wcs_analyzer.claude_code_analyzer._check_claude_cli", return_value="/bin/claude"), \
             patch("wcs_analyzer.claude_code_analyzer.extract_frames", return_value=frames) as mock_extract, \
             patch("wcs_analyzer.claude_code_analyzer._call_claude_cli",
                   return_value=(VALID_ANALYSIS, UsageTotals())) as mock_cli:
            segments = analyze_dance_claude_code(Path("clip.mp4"), detail=detail, **kwargs)
        prompt = mock_cli.call_args[0][1]
        return segments, prompt, mock_extract

    def test_prompt_states_actual_sampling_rate(self):
        # The CLI-level --fps (3.0) is not what claude-code samples at; the
        # detail level decides (medium -> 2 fps). The prompt must say 2 fps.
        _, prompt, _ = self._run(detail="medium", fps=3.0)
        assert "sampled at 2 fps" in prompt
        assert "0.50s apart" in prompt
        assert "sampled at 3.0 fps" not in prompt

    def test_prompt_lists_frame_timestamps(self):
        _, prompt, _ = self._run()
        assert "frame_000.jpg  (t=0.00s)" in prompt
        assert "frame_001.jpg  (t=0.50s)" in prompt
        assert "frame_002.jpg  (t=1.00s)" in prompt

    def test_max_dimension_forwarded_to_frame_extraction(self):
        _, _, mock_extract = self._run(max_dimension=1080)
        assert mock_extract.call_args.kwargs["max_dimension"] == 1080
        assert mock_extract.call_args.kwargs["fps"] == 2.0

    def test_default_max_dimension(self):
        _, _, mock_extract = self._run()
        assert mock_extract.call_args.kwargs["max_dimension"] == 768

    def test_result_is_whole_video_summary(self):
        segments, _, _ = self._run()
        assert len(segments) == 1
        assert segments[0].is_summary is True
        assert segments[0].end_time == 1.5

    def test_model_forwarded_to_cli(self):
        from pathlib import Path

        from wcs_analyzer.claude_code_analyzer import analyze_dance_claude_code
        from wcs_analyzer.pricing import UsageTotals
        from wcs_analyzer.video import FrameData

        frames = FrameData(images=["aGk="], timestamps=[0.0], duration=0.5)
        with patch("wcs_analyzer.claude_code_analyzer._check_claude_cli", return_value="/bin/claude"), \
             patch("wcs_analyzer.claude_code_analyzer.extract_frames", return_value=frames), \
             patch("wcs_analyzer.claude_code_analyzer._call_claude_cli",
                   return_value=(VALID_ANALYSIS, UsageTotals())) as mock_cli:
            analyze_dance_claude_code(Path("clip.mp4"), model="claude-sonnet-4-6")
        assert mock_cli.call_args.kwargs["model"] == "claude-sonnet-4-6"

    def test_prompt_requests_timeline_and_plain_pattern_names(self):
        _, prompt, _ = self._run()
        assert '"pattern_timeline"' in prompt
        assert "plain pattern names with no timestamps" in prompt

    def test_prompt_pins_pattern_vocabulary(self):
        from wcs_analyzer.prompts import WCS_PATTERN_VOCABULARY
        _, prompt, _ = self._run()
        assert "ONLY these canonical names" in prompt
        for name in WCS_PATTERN_VOCABULARY:
            assert name in prompt


class TestCallClaudeCliCommand:
    def _cmd(self, **kwargs) -> list[str]:
        from wcs_analyzer.claude_code_analyzer import _call_claude_cli
        stdout = json.dumps({"type": "result", "is_error": False, "result": json.dumps(VALID_ANALYSIS), "usage": {}})
        proc = MagicMock(stdout=stdout, stderr="", returncode=0)
        with patch("wcs_analyzer.claude_code_analyzer.subprocess.run", return_value=proc) as run:
            _call_claude_cli("/bin/claude", "prompt", **kwargs)
        return run.call_args[0][0]

    def test_model_flag_added_when_given(self):
        cmd = self._cmd(model="claude-sonnet-4-6")
        assert cmd[cmd.index("--model") + 1] == "claude-sonnet-4-6"

    def test_no_model_flag_by_default(self):
        assert "--model" not in self._cmd()


class TestCliBudget:
    def test_budget_scales_with_frames(self):
        from wcs_analyzer.claude_code_analyzer import _cli_budget
        t_small, turns_small = _cli_budget(20)
        t_mid, turns_mid = _cli_budget(117)     # 54 s clip at 2 fps
        t_long, turns_long = _cli_budget(272)   # 136 s clip at 2 fps
        assert turns_small == 25                # never below the old cap
        assert t_mid > 300 and turns_mid > 25   # the old fixed budget was too small here
        assert t_long > t_mid and turns_long > turns_mid
        assert t_long >= 272 * 3                # at least a few seconds per frame

    def test_max_turns_forwarded_to_cli(self):
        from wcs_analyzer.claude_code_analyzer import _call_claude_cli
        stdout = json.dumps({"type": "result", "is_error": False, "result": json.dumps(VALID_ANALYSIS), "usage": {}})
        proc = MagicMock(stdout=stdout, stderr="", returncode=0)
        with patch("wcs_analyzer.claude_code_analyzer.subprocess.run", return_value=proc) as run:
            _call_claude_cli("/bin/claude", "prompt", timeout=999, max_turns=40)
        cmd = run.call_args[0][0]
        assert cmd[cmd.index("--max-turns") + 1] == "40"
        assert run.call_args.kwargs["timeout"] == 999

    def test_analyze_uses_frame_scaled_budget(self):
        from pathlib import Path

        from wcs_analyzer.claude_code_analyzer import analyze_dance_claude_code
        from wcs_analyzer.pricing import UsageTotals
        from wcs_analyzer.video import FrameData

        frames = FrameData(images=["aGk="] * 272, timestamps=[i * 0.5 for i in range(272)], duration=136.0)
        with patch("wcs_analyzer.claude_code_analyzer._check_claude_cli", return_value="/bin/claude"), \
             patch("wcs_analyzer.claude_code_analyzer.extract_frames", return_value=frames), \
             patch("wcs_analyzer.claude_code_analyzer._call_claude_cli",
                   return_value=(VALID_ANALYSIS, UsageTotals())) as mock_cli:
            analyze_dance_claude_code(Path("clip.mp4"))
        assert mock_cli.call_args.kwargs["timeout"] > 300
        assert mock_cli.call_args.kwargs["max_turns"] > 25


class TestCallClaudeCliEnvelopes:
    """`claude --output-format json` changed shape across CLI versions."""

    def _run(self, stdout: str, returncode: int = 0):
        from wcs_analyzer.claude_code_analyzer import _call_claude_cli
        proc = MagicMock(stdout=stdout, stderr="", returncode=returncode)
        with patch("wcs_analyzer.claude_code_analyzer.subprocess.run", return_value=proc):
            return _call_claude_cli("/bin/claude", "prompt")

    def test_legacy_single_envelope_dict(self):
        stdout = json.dumps({
            "type": "result", "is_error": False, "result": json.dumps(VALID_ANALYSIS),
            "usage": {"input_tokens": 100, "output_tokens": 50}, "total_cost_usd": 0.01,
            "model": "claude-sonnet-4-6",
        })
        data, usage = self._run(stdout)
        assert data["timing"]["score"] == 7.5
        assert usage.estimated_cost == 0.01
        assert usage.model == "claude-sonnet-4-6"

    def test_claude_code_2x_event_list(self):
        # Shape observed from Claude Code 2.1.x: init, assistant, rate limit, result.
        stdout = json.dumps([
            {"type": "system", "subtype": "init", "model": "claude-sonnet-4-6", "cwd": "/tmp"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "reading frames"}]}},
            {"type": "rate_limit_event", "rate_limit_info": {}},
            {"type": "result", "subtype": "success", "is_error": False, "num_turns": 9,
             "result": "Here is the analysis:\n```json\n" + json.dumps(VALID_ANALYSIS) + "\n```",
             "usage": {"input_tokens": 2000, "output_tokens": 400},
             "total_cost_usd": 0.247, "modelUsage": {"claude-sonnet-4-6": {"inputTokens": 2000}}},
        ])
        data, usage = self._run(stdout)
        assert data["technique"]["score"] == 6.5
        assert usage.input_tokens == 2000
        assert usage.output_tokens == 400
        assert usage.estimated_cost == 0.247
        assert usage.pricing_known is True
        assert usage.model == "claude-sonnet-4-6"  # recovered from modelUsage

    def test_event_list_error_result_raises(self):
        stdout = json.dumps([
            {"type": "system", "subtype": "init"},
            {"type": "result", "subtype": "error_max_turns", "is_error": True,
             "result": "Reached max turns", "usage": {}},
        ])
        with pytest.raises(AnalysisError, match="Reached max turns"):
            self._run(stdout)

    def test_error_without_result_text_reports_subtype_and_errors(self):
        stdout = json.dumps([
            {"type": "system", "subtype": "init"},
            {"type": "result", "subtype": "error_during_execution", "is_error": True,
             "result": "", "errors": ["prompt too long"], "usage": {}},
        ])
        with pytest.raises(AnalysisError, match="subtype=error_during_execution.*prompt too long"):
            self._run(stdout)

    def test_error_with_nothing_but_exit_code_says_so(self):
        with pytest.raises(AnalysisError, match="exit code 1, no output"):
            self._run("", returncode=1)

    def test_event_list_without_result_item_raises_cleanly(self):
        stdout = json.dumps([{"type": "system", "subtype": "init"}])
        with pytest.raises(AnalysisError, match="Could not parse"):
            self._run(stdout)

    def test_unparseable_stdout_with_nonzero_exit_raises(self):
        with pytest.raises(AnalysisError, match="Claude Code CLI failed"):
            self._run("not json", returncode=1)
