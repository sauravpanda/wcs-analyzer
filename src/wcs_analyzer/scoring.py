"""Scoring engine: aggregate per-segment scores into WSDC-weighted final score."""

from __future__ import annotations

from dataclasses import dataclass, field

WEIGHTS = {
    "timing": 0.30,
    "technique": 0.30,
    "teamwork": 0.20,
    "presentation": 0.20,
}

GRADE_THRESHOLDS = [
    (9.5, "A+"),
    (9.0, "A"),
    (8.5, "A-"),
    (8.0, "B+"),
    (7.5, "B"),
    (7.0, "B-"),
    (6.5, "C+"),
    (6.0, "C"),
    (5.5, "C-"),
    (5.0, "D+"),
    (4.0, "D"),
    (0.0, "F"),
]


@dataclass
class SegmentScore:
    segment_index: int
    start_time: float
    end_time: float
    timing: float
    technique: float
    teamwork: float
    presentation: float
    timing_notes: str = ""
    technique_notes: str = ""
    teamwork_notes: str = ""
    presentation_notes: str = ""
    off_beat_moments: list[dict] = field(default_factory=list)
    highlight_moments: list[dict] = field(default_factory=list)

    @property
    def weighted_score(self) -> float:
        return (
            self.timing * WEIGHTS["timing"]
            + self.technique * WEIGHTS["technique"]
            + self.teamwork * WEIGHTS["teamwork"]
            + self.presentation * WEIGHTS["presentation"]
        )


@dataclass
class FinalScore:
    timing: float
    technique: float
    teamwork: float
    presentation: float
    overall: float
    grade: str
    top_strengths: list[str]
    areas_to_improve: list[str]
    judge_commentary: str
    off_beat_moments: list[dict] = field(default_factory=list)
    highlight_moments: list[dict] = field(default_factory=list)
    segment_scores: list[SegmentScore] = field(default_factory=list)


def letter_grade(score: float) -> str:
    for threshold, grade in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "F"


def aggregate_scores(
    segment_scores: list[SegmentScore],
    summary: dict,
) -> FinalScore:
    """Combine per-segment data with the LLM's holistic summary into a FinalScore."""
    # Collect all off-beat / highlight moments across segments
    all_off_beat: list[dict] = []
    all_highlights: list[dict] = []
    for seg in segment_scores:
        all_off_beat.extend(seg.off_beat_moments)
        all_highlights.extend(seg.highlight_moments)

    # Sort by timestamp
    all_off_beat.sort(key=lambda x: x.get("timestamp", 0))
    all_highlights.sort(key=lambda x: x.get("timestamp", 0))

    timing = float(summary.get("overall_timing", _avg(segment_scores, "timing")))
    technique = float(summary.get("overall_technique", _avg(segment_scores, "technique")))
    teamwork = float(summary.get("overall_teamwork", _avg(segment_scores, "teamwork")))
    presentation = float(summary.get("overall_presentation", _avg(segment_scores, "presentation")))

    overall = (
        timing * WEIGHTS["timing"]
        + technique * WEIGHTS["technique"]
        + teamwork * WEIGHTS["teamwork"]
        + presentation * WEIGHTS["presentation"]
    )

    return FinalScore(
        timing=timing,
        technique=technique,
        teamwork=teamwork,
        presentation=presentation,
        overall=overall,
        grade=letter_grade(overall),
        top_strengths=summary.get("top_strengths", []),
        areas_to_improve=summary.get("areas_to_improve", []),
        judge_commentary=summary.get("judge_commentary", ""),
        off_beat_moments=all_off_beat,
        highlight_moments=all_highlights,
        segment_scores=segment_scores,
    )


def _avg(segments: list[SegmentScore], attr: str) -> float:
    if not segments:
        return 5.0
    return sum(getattr(s, attr) for s in segments) / len(segments)
