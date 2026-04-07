"""Tests for the scoring engine."""

from wcs_analyzer.scoring import (
    FinalScores,
    SegmentAnalysis,
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
