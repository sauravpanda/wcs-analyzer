"""Scoring engine — WSDC-style scoring aggregation."""

from dataclasses import dataclass, field


@dataclass
class SegmentAnalysis:
    """Analysis result for a single segment or summary."""

    start_time: float = 0.0
    end_time: float = 0.0

    # Core category scores (1-10)
    timing_score: float = 5.0
    technique_score: float = 5.0
    teamwork_score: float = 5.0
    presentation_score: float = 5.0

    # Technique sub-scores
    posture_score: float = 5.0
    extension_score: float = 5.0
    footwork_score: float = 5.0
    slot_score: float = 5.0

    # Details
    off_beat_moments: list[dict] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    highlights: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)

    # Partner-specific scores (optional — only populated when detected)
    lead_technique: float = 0.0
    lead_presentation: float = 0.0
    lead_notes: str = ""
    follow_technique: float = 0.0
    follow_presentation: float = 0.0
    follow_notes: str = ""

    # Raw LLM response data
    raw_data: dict = field(default_factory=dict)


# WSDC category weights
WEIGHTS = {
    "timing": 0.30,
    "technique": 0.30,
    "teamwork": 0.20,
    "presentation": 0.20,
}

GRADE_THRESHOLDS = [
    (9.5, "A+"), (9.0, "A"), (8.5, "A-"),
    (8.0, "B+"), (7.5, "B"), (7.0, "B-"),
    (6.5, "C+"), (6.0, "C"), (5.5, "C-"),
    (5.0, "D+"), (4.5, "D"), (4.0, "D-"),
    (0.0, "F"),
]


@dataclass
class FinalScores:
    """Aggregated final scores for the full dance."""

    timing: float = 0.0
    technique: float = 0.0
    teamwork: float = 0.0
    presentation: float = 0.0
    overall: float = 0.0
    grade: str = ""

    # Technique sub-scores
    posture: float = 0.0
    extension: float = 0.0
    footwork: float = 0.0
    slot: float = 0.0

    # Partner-specific aggregated scores
    lead_technique: float = 0.0
    lead_presentation: float = 0.0
    lead_notes: str = ""
    follow_technique: float = 0.0
    follow_presentation: float = 0.0
    follow_notes: str = ""

    # Aggregated details
    total_off_beat: int = 0
    off_beat_moments: list[dict] = field(default_factory=list)
    all_patterns: list[str] = field(default_factory=list)
    top_strengths: list[str] = field(default_factory=list)
    top_improvements: list[str] = field(default_factory=list)
    overall_impression: str = ""

    # Per-segment data for timeline
    segments: list[SegmentAnalysis] = field(default_factory=list)


def compute_final_scores(segments: list[SegmentAnalysis]) -> FinalScores:
    """Aggregate segment-level scores into final WSDC-style scores.

    If the last segment is a summary (start_time=0, covering full duration),
    use its scores directly. Otherwise, average across segments.

    Args:
        segments: List of per-segment analysis results.

    Returns:
        FinalScores with weighted overall score and details.
    """
    if not segments:
        return FinalScores()

    # Check if last segment is a summary (covers full duration)
    has_summary = len(segments) > 1 and segments[-1].start_time == 0.0
    scoring_segments = segments[:-1] if has_summary else segments
    summary = segments[-1] if has_summary else None

    # Use summary scores if available, otherwise average
    if summary and summary.raw_data and "error" not in summary.raw_data:
        timing = summary.timing_score
        technique = summary.technique_score
        teamwork = summary.teamwork_score
        presentation = summary.presentation_score
        posture = summary.posture_score
        extension = summary.extension_score
        footwork = summary.footwork_score
        slot = summary.slot_score
    else:
        n = len(scoring_segments)
        timing = sum(s.timing_score for s in scoring_segments) / n
        technique = sum(s.technique_score for s in scoring_segments) / n
        teamwork = sum(s.teamwork_score for s in scoring_segments) / n
        presentation = sum(s.presentation_score for s in scoring_segments) / n
        posture = sum(s.posture_score for s in scoring_segments) / n
        extension = sum(s.extension_score for s in scoring_segments) / n
        footwork = sum(s.footwork_score for s in scoring_segments) / n
        slot = sum(s.slot_score for s in scoring_segments) / n

    # Weighted overall
    overall = (
        timing * WEIGHTS["timing"]
        + technique * WEIGHTS["technique"]
        + teamwork * WEIGHTS["teamwork"]
        + presentation * WEIGHTS["presentation"]
    )

    # Letter grade
    grade = "F"
    for threshold, g in GRADE_THRESHOLDS:
        if overall >= threshold:
            grade = g
            break

    # Collect all off-beat moments
    all_off_beat = []
    for seg in scoring_segments:
        all_off_beat.extend(seg.off_beat_moments)

    # Collect patterns (deduplicated)
    seen_patterns = set()
    all_patterns = []
    for seg in scoring_segments:
        for p in seg.patterns:
            if p.lower() not in seen_patterns:
                seen_patterns.add(p.lower())
                all_patterns.append(p)

    # Use summary for strengths/improvements, or aggregate
    if summary:
        strengths = summary.highlights[:3] or _collect_top(scoring_segments, "highlights", 3)
        improvements = summary.improvements[:3] or _collect_top(scoring_segments, "improvements", 3)
        impression = summary.raw_data.get("overall_impression", "")
    else:
        strengths = _collect_top(scoring_segments, "highlights", 3)
        improvements = _collect_top(scoring_segments, "improvements", 3)
        impression = ""

    # Partner-specific scores — use summary if available, else average
    if summary and summary.lead_technique > 0:
        lead_tech = summary.lead_technique
        lead_pres = summary.lead_presentation
        lead_notes_str = summary.lead_notes
        follow_tech = summary.follow_technique
        follow_pres = summary.follow_presentation
        follow_notes_str = summary.follow_notes
    else:
        segs_with_partner = [s for s in scoring_segments if s.lead_technique > 0]
        if segs_with_partner:
            n_p = len(segs_with_partner)
            lead_tech = sum(s.lead_technique for s in segs_with_partner) / n_p
            lead_pres = sum(s.lead_presentation for s in segs_with_partner) / n_p
            follow_tech = sum(s.follow_technique for s in segs_with_partner) / n_p
            follow_pres = sum(s.follow_presentation for s in segs_with_partner) / n_p
            lead_notes_str = ""
            follow_notes_str = ""
        else:
            lead_tech = lead_pres = follow_tech = follow_pres = 0.0
            lead_notes_str = follow_notes_str = ""

    return FinalScores(
        timing=round(timing, 1),
        technique=round(technique, 1),
        teamwork=round(teamwork, 1),
        presentation=round(presentation, 1),
        overall=round(overall, 1),
        grade=grade,
        posture=round(posture, 1),
        extension=round(extension, 1),
        footwork=round(footwork, 1),
        slot=round(slot, 1),
        lead_technique=round(lead_tech, 1),
        lead_presentation=round(lead_pres, 1),
        lead_notes=lead_notes_str,
        follow_technique=round(follow_tech, 1),
        follow_presentation=round(follow_pres, 1),
        follow_notes=follow_notes_str,
        total_off_beat=len(all_off_beat),
        off_beat_moments=all_off_beat,
        all_patterns=all_patterns,
        top_strengths=strengths,
        top_improvements=improvements,
        overall_impression=impression,
        segments=scoring_segments,
    )


def _collect_top(segments: list[SegmentAnalysis], attr: str, n: int) -> list[str]:
    """Collect top N unique items from a segment attribute."""
    seen = set()
    items = []
    for seg in segments:
        for item in getattr(seg, attr, []):
            if item not in seen:
                seen.add(item)
                items.append(item)
            if len(items) >= n:
                return items
    return items
