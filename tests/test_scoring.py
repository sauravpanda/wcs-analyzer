"""Tests for the scoring engine."""

from wcs_analyzer.scoring import (
    FinalScores,
    SegmentAnalysis,
    aggregate_ensemble,
    compute_final_scores,
    WEIGHTS,
)


def test_weights_sum_to_one():
    assert sum(WEIGHTS.values()) == 1.0


def test_empty_segments_returns_default():
    scores = compute_final_scores([])
    assert scores.overall == 0.0
    assert scores.grade == ""


def test_single_segment_scoring():
    seg = SegmentAnalysis(
        start_time=0.0,
        end_time=4.0,
        timing_score=8.0,
        technique_score=7.0,
        teamwork_score=6.0,
        presentation_score=9.0,
    )
    scores = compute_final_scores([seg])

    expected = (
        8.0 * 0.30
        + 7.0 * 0.30
        + 6.0 * 0.20
        + 9.0 * 0.20
    )
    assert scores.overall == round(expected, 1)
    assert scores.timing == 8.0
    assert scores.technique == 7.0


def test_grade_assignment():
    seg = SegmentAnalysis(timing_score=9.5, technique_score=9.5, teamwork_score=9.5, presentation_score=9.5)
    scores = compute_final_scores([seg])
    assert scores.grade == "A+"

    seg2 = SegmentAnalysis(timing_score=2.0, technique_score=2.0, teamwork_score=2.0, presentation_score=2.0)
    scores2 = compute_final_scores([seg2])
    assert scores2.grade == "F"


def test_off_beat_moments_collected():
    seg1 = SegmentAnalysis(
        start_time=0.0,
        end_time=4.0,
        off_beat_moments=[{"time": "0:05", "description": "rushed"}],
    )
    seg2 = SegmentAnalysis(
        start_time=4.0,
        end_time=8.0,
        off_beat_moments=[{"time": "0:12", "description": "late"}],
    )
    scores = compute_final_scores([seg1, seg2])
    assert scores.total_off_beat == 2


def test_patterns_deduplicated():
    seg1 = SegmentAnalysis(start_time=0.0, end_time=4.0, patterns=["Sugar Push", "Whip"])
    seg2 = SegmentAnalysis(start_time=4.0, end_time=8.0, patterns=["sugar push", "Left Side Pass"])
    scores = compute_final_scores([seg1, seg2])
    assert len(scores.all_patterns) == 3


def test_summary_segment_used_for_scores():
    """When last segment is a summary (start_time=0), its scores are used."""
    seg1 = SegmentAnalysis(start_time=0.0, end_time=4.0, timing_score=6.0, technique_score=6.0, teamwork_score=6.0, presentation_score=6.0)
    summary = SegmentAnalysis(
        start_time=0.0,
        end_time=8.0,
        timing_score=8.0,
        technique_score=7.0,
        teamwork_score=9.0,
        presentation_score=7.0,
        raw_data={"overall_impression": "Good dance"},
    )
    scores = compute_final_scores([seg1, summary])
    # Should use summary scores, not seg1
    assert scores.timing == 8.0
    assert scores.teamwork == 9.0


def test_partner_scores_from_summary():
    """Partner scores should be taken from summary when available."""
    seg1 = SegmentAnalysis(
        start_time=0.0, end_time=4.0,
        lead_technique=6.0, lead_presentation=5.0,
        follow_technique=7.0, follow_presentation=8.0,
    )
    summary = SegmentAnalysis(
        start_time=0.0, end_time=8.0,
        lead_technique=8.0, lead_presentation=7.0, lead_notes="Strong leads",
        follow_technique=9.0, follow_presentation=8.5, follow_notes="Great follow",
        raw_data={"overall_impression": "Good"},
    )
    scores = compute_final_scores([seg1, summary])
    assert scores.lead_technique == 8.0
    assert scores.follow_technique == 9.0
    assert scores.lead_notes == "Strong leads"


def test_partner_scores_averaged_without_summary():
    """When no summary, partner scores should average across segments."""
    seg1 = SegmentAnalysis(
        start_time=0.0, end_time=4.0,
        lead_technique=6.0, lead_presentation=5.0,
        follow_technique=8.0, follow_presentation=7.0,
    )
    seg2 = SegmentAnalysis(
        start_time=4.0, end_time=8.0,
        lead_technique=8.0, lead_presentation=7.0,
        follow_technique=6.0, follow_presentation=9.0,
    )
    scores = compute_final_scores([seg1, seg2])
    assert scores.lead_technique == 7.0  # avg of 6 and 8
    assert scores.follow_presentation == 8.0  # avg of 7 and 9


def test_no_partner_data_zeros():
    """When no partner data, scores should be 0."""
    seg = SegmentAnalysis(start_time=0.0, end_time=4.0)
    scores = compute_final_scores([seg])
    assert scores.lead_technique == 0.0
    assert scores.follow_technique == 0.0


def test_confidence_intervals_aggregate_from_segments():
    """CI bounds should average across segments when no summary is present."""
    seg1 = SegmentAnalysis(
        start_time=0.0, end_time=4.0,
        timing_score=7.0, timing_low=6.0, timing_high=8.0,
        technique_score=6.0, technique_low=5.0, technique_high=7.0,
        teamwork_score=7.0, teamwork_low=6.5, teamwork_high=7.5,
        presentation_score=7.0, presentation_low=6.0, presentation_high=8.0,
    )
    seg2 = SegmentAnalysis(
        start_time=4.0, end_time=8.0,
        timing_score=9.0, timing_low=8.0, timing_high=10.0,
        technique_score=8.0, technique_low=7.0, technique_high=9.0,
        teamwork_score=7.0, teamwork_low=6.5, teamwork_high=7.5,
        presentation_score=7.0, presentation_low=6.0, presentation_high=8.0,
    )
    scores = compute_final_scores([seg1, seg2])
    assert scores.timing_low == 7.0
    assert scores.timing_high == 9.0
    assert scores.technique_low == 6.0
    assert scores.technique_high == 8.0
    # Overall CI should be weighted like overall score
    assert scores.overall_low < scores.overall < scores.overall_high


def test_low_confidence_flag_raised_on_wide_interval():
    seg = SegmentAnalysis(
        timing_score=7.0, timing_low=4.0, timing_high=9.0,  # width 5 > 2
        technique_score=7.0, technique_low=6.8, technique_high=7.2,
        teamwork_score=7.0, teamwork_low=6.8, teamwork_high=7.2,
        presentation_score=7.0, presentation_low=6.8, presentation_high=7.2,
    )
    scores = compute_final_scores([seg])
    assert scores.low_confidence is True


def test_low_confidence_flag_off_on_tight_intervals():
    seg = SegmentAnalysis(
        timing_score=7.0, timing_low=6.8, timing_high=7.2,
        technique_score=7.0, technique_low=6.8, technique_high=7.2,
        teamwork_score=7.0, teamwork_low=6.8, teamwork_high=7.2,
        presentation_score=7.0, presentation_low=6.8, presentation_high=7.2,
    )
    scores = compute_final_scores([seg])
    assert scores.low_confidence is False


def test_reasoning_from_summary_preferred():
    seg = SegmentAnalysis(
        start_time=0.0, end_time=4.0,
        reasoning={"timing": "segment-level reasoning"},
    )
    summary = SegmentAnalysis(
        start_time=0.0, end_time=8.0,
        timing_score=7.0, technique_score=7.0, teamwork_score=7.0, presentation_score=7.0,
        reasoning={"timing": "summary reasoning", "technique": "tech reasoning"},
        raw_data={"overall_impression": "ok"},
    )
    scores = compute_final_scores([seg, summary])
    assert scores.reasoning["timing"] == "summary reasoning"
    assert scores.reasoning["technique"] == "tech reasoning"


def test_reasoning_falls_back_to_segments():
    seg1 = SegmentAnalysis(
        start_time=0.0, end_time=4.0,
        reasoning={"timing": "first segment timing"},
    )
    seg2 = SegmentAnalysis(
        start_time=4.0, end_time=8.0,
        reasoning={"timing": "second segment timing", "technique": "tech note"},
    )
    scores = compute_final_scores([seg1, seg2])
    # First non-empty reasoning per category wins
    assert scores.reasoning["timing"] == "first segment timing"
    assert scores.reasoning["technique"] == "tech note"


def test_summary_ci_used_when_present():
    seg1 = SegmentAnalysis(
        start_time=0.0, end_time=4.0,
        timing_score=5.0, timing_low=4.0, timing_high=6.0,
    )
    summary = SegmentAnalysis(
        start_time=0.0, end_time=8.0,
        timing_score=8.0, timing_low=7.5, timing_high=8.5,
        technique_score=8.0, technique_low=7.5, technique_high=8.5,
        teamwork_score=8.0, teamwork_low=7.5, teamwork_high=8.5,
        presentation_score=8.0, presentation_low=7.5, presentation_high=8.5,
        raw_data={"overall_impression": "Good"},
    )
    scores = compute_final_scores([seg1, summary])
    assert scores.timing_low == 7.5
    assert scores.timing_high == 8.5


# ---- Ensemble aggregation tests -------------------------------------------


def _fs(timing=7.0, technique=7.0, teamwork=7.0, presentation=7.0,
        posture=7.0, extension=7.0, footwork=7.0, slot=7.0) -> FinalScores:
    return FinalScores(
        timing=timing, technique=technique, teamwork=teamwork, presentation=presentation,
        posture=posture, extension=extension, footwork=footwork, slot=slot,
    )


def test_ensemble_consensus_is_median():
    runs = {
        "gemini": _fs(timing=8.0, technique=7.0, teamwork=7.0, presentation=7.0),
        "claude": _fs(timing=6.0, technique=7.0, teamwork=7.0, presentation=7.0),
        "claude-code": _fs(timing=7.0, technique=7.0, teamwork=7.0, presentation=7.0),
    }
    ensemble = aggregate_ensemble(runs)
    assert ensemble.timing == 7.0  # median of 6, 7, 8
    assert ensemble.technique == 7.0
    assert len(ensemble.providers) == 3


def test_ensemble_flags_contested_categories():
    runs = {
        "a": _fs(timing=9.0),
        "b": _fs(timing=5.0),  # 4-point spread → stddev ~2.0 → contested
    }
    ensemble = aggregate_ensemble(runs)
    assert "timing" in ensemble.contested
    assert ensemble.stddev["timing"] > 1.0


def test_ensemble_tight_agreement_not_contested():
    runs = {
        "a": _fs(timing=7.0, technique=6.5),
        "b": _fs(timing=7.2, technique=6.7),
        "c": _fs(timing=6.9, technique=6.4),
    }
    ensemble = aggregate_ensemble(runs)
    assert ensemble.contested == []
    assert ensemble.stddev["timing"] < 0.5


def test_ensemble_overall_uses_weighted_medians():
    runs = {
        "a": _fs(timing=8.0, technique=6.0, teamwork=7.0, presentation=8.0),
        "b": _fs(timing=8.0, technique=6.0, teamwork=7.0, presentation=8.0),
    }
    ensemble = aggregate_ensemble(runs)
    expected = 8.0 * 0.30 + 6.0 * 0.30 + 7.0 * 0.20 + 8.0 * 0.20
    assert ensemble.overall == round(expected, 1)
    assert ensemble.grade in ("A", "A-", "B+", "B", "B-")


def test_ensemble_empty_returns_default():
    ensemble = aggregate_ensemble({})
    assert ensemble.overall == 0.0
    assert ensemble.providers == []


def test_ensemble_single_provider_has_zero_stddev():
    runs = {"gemini": _fs(timing=7.5)}
    ensemble = aggregate_ensemble(runs)
    assert ensemble.timing == 7.5
    assert ensemble.stddev["timing"] == 0.0
    assert ensemble.contested == []


# ---- Usage aggregation tests ---------------------------------------------


def test_final_scores_sums_usage_across_segments():
    from wcs_analyzer.pricing import UsageTotals
    seg1 = SegmentAnalysis(
        start_time=0.0, end_time=4.0,
        timing_score=7.0, technique_score=7.0, teamwork_score=7.0, presentation_score=7.0,
        usage=UsageTotals.from_counts("claude-sonnet-4-6", 1000, 500),
    )
    seg2 = SegmentAnalysis(
        start_time=4.0, end_time=8.0,
        timing_score=7.0, technique_score=7.0, teamwork_score=7.0, presentation_score=7.0,
        usage=UsageTotals.from_counts("claude-sonnet-4-6", 2000, 1000),
    )
    scores = compute_final_scores([seg1, seg2])
    assert scores.usage.input_tokens == 3000
    assert scores.usage.output_tokens == 1500
    assert scores.usage.estimated_cost > 0


def test_final_scores_includes_summary_usage():
    """The summary call's tokens should be counted even though its scores
    take precedence over the per-segment averages."""
    from wcs_analyzer.pricing import UsageTotals
    seg1 = SegmentAnalysis(
        start_time=0.0, end_time=4.0,
        usage=UsageTotals.from_counts("claude-sonnet-4-6", 1000, 500),
    )
    summary = SegmentAnalysis(
        start_time=0.0, end_time=8.0,
        timing_score=8.0, technique_score=8.0, teamwork_score=8.0, presentation_score=8.0,
        usage=UsageTotals.from_counts("claude-sonnet-4-6", 500, 250),
        raw_data={"overall_impression": "ok"},
    )
    scores = compute_final_scores([seg1, summary])
    assert scores.usage.input_tokens == 1500
    assert scores.usage.output_tokens == 750


# ---- Ported features from Updates branch ---------------------------------


def test_pattern_counts_tallied_across_segments():
    seg1 = SegmentAnalysis(
        start_time=0.0, end_time=4.0,
        patterns=["Sugar Push", "Whip", "Sugar Push"],
    )
    seg2 = SegmentAnalysis(
        start_time=4.0, end_time=8.0,
        patterns=["sugar push", "Left Side Pass"],
    )
    scores = compute_final_scores([seg1, seg2])
    # Case-insensitive count, but display name is the first one we saw
    assert scores.pattern_counts["Sugar Push"] == 3
    assert scores.pattern_counts["Whip"] == 1
    assert scores.pattern_counts["Left Side Pass"] == 1


def test_pattern_timeline_records_time_windows():
    seg1 = SegmentAnalysis(
        start_time=0.0, end_time=4.0,
        patterns=["Sugar Push"],
    )
    seg2 = SegmentAnalysis(
        start_time=4.0, end_time=8.0,
        patterns=["Whip", "Left Side Pass"],
    )
    seg3 = SegmentAnalysis(start_time=8.0, end_time=12.0)  # No patterns
    scores = compute_final_scores([seg1, seg2, seg3])
    assert len(scores.pattern_timeline) == 2
    assert scores.pattern_timeline[0]["start_time"] == 0.0
    assert scores.pattern_timeline[0]["patterns"] == ["Sugar Push"]
    assert scores.pattern_timeline[1]["patterns"] == ["Whip", "Left Side Pass"]


def test_is_summary_flag_preferred_over_start_time_heuristic():
    """A first segment starting at 0.0 must not be treated as a summary."""
    seg1 = SegmentAnalysis(
        start_time=0.0, end_time=4.0,
        timing_score=6.0, technique_score=6.0, teamwork_score=6.0, presentation_score=6.0,
    )
    seg2 = SegmentAnalysis(
        start_time=4.0, end_time=8.0,
        timing_score=8.0, technique_score=8.0, teamwork_score=8.0, presentation_score=8.0,
    )
    summary = SegmentAnalysis(
        start_time=99.0, end_time=108.0,  # Doesn't start at 0
        is_summary=True,
        timing_score=9.0, technique_score=9.0, teamwork_score=9.0, presentation_score=9.0,
        raw_data={"overall_impression": "ok"},
    )
    scores = compute_final_scores([seg1, seg2, summary])
    # Scores should come from the is_summary segment even though start_time != 0
    assert scores.timing == 9.0


def test_final_scores_prefers_summary_pattern_timeline():
    """When the summary segment has an explicit pattern_timeline (e.g.
    from Gemini's structured output), that wins over the derived-from-
    segments fallback."""
    seg1 = SegmentAnalysis(
        start_time=0.0, end_time=4.0,
        patterns=["sugar push"],  # would feed derived timeline
    )
    summary = SegmentAnalysis(
        start_time=0.0, end_time=8.0,
        timing_score=7.0, technique_score=7.0, teamwork_score=7.0, presentation_score=7.0,
        pattern_timeline=[
            {"start_time": 0.0, "end_time": 3.0, "patterns": ["starter step"], "pattern_details": []},
            {"start_time": 3.0, "end_time": 8.0, "patterns": ["whip", "free spin"], "pattern_details": []},
        ],
        raw_data={"overall_impression": "ok"},
    )
    scores = compute_final_scores([seg1, summary])
    assert len(scores.pattern_timeline) == 2
    # Should be the summary's explicit timeline, not the derived one
    assert scores.pattern_timeline[0]["patterns"] == ["starter step"]
    assert scores.pattern_timeline[1]["patterns"] == ["whip", "free spin"]
