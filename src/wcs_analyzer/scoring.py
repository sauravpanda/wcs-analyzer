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

    # Per-category confidence intervals (low/high). Default to point estimate.
    timing_low: float = 5.0
    timing_high: float = 5.0
    technique_low: float = 5.0
    technique_high: float = 5.0
    teamwork_low: float = 5.0
    teamwork_high: float = 5.0
    presentation_low: float = 5.0
    presentation_high: float = 5.0

    # Technique sub-scores
    posture_score: float = 5.0
    extension_score: float = 5.0
    footwork_score: float = 5.0
    slot_score: float = 5.0

    # Per-category chain-of-thought reasoning (keyed by category name)
    reasoning: dict[str, str] = field(default_factory=dict)

    # Details
    off_beat_moments: list[dict] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)  # plain names for backward compat
    pattern_details: list[dict] = field(default_factory=list)  # rich pattern info
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

    # Aggregated per-category confidence intervals
    timing_low: float = 0.0
    timing_high: float = 0.0
    technique_low: float = 0.0
    technique_high: float = 0.0
    teamwork_low: float = 0.0
    teamwork_high: float = 0.0
    presentation_low: float = 0.0
    presentation_high: float = 0.0
    overall_low: float = 0.0
    overall_high: float = 0.0
    low_confidence: bool = False

    # Per-category reasoning (from summary or first segment with reasoning)
    reasoning: dict[str, str] = field(default_factory=dict)

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
    pattern_details: list[dict] = field(default_factory=list)
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
        timing_lo, timing_hi = summary.timing_low, summary.timing_high
        tech_lo, tech_hi = summary.technique_low, summary.technique_high
        tw_lo, tw_hi = summary.teamwork_low, summary.teamwork_high
        pres_lo, pres_hi = summary.presentation_low, summary.presentation_high
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
        timing_lo = sum(s.timing_low for s in scoring_segments) / n
        timing_hi = sum(s.timing_high for s in scoring_segments) / n
        tech_lo = sum(s.technique_low for s in scoring_segments) / n
        tech_hi = sum(s.technique_high for s in scoring_segments) / n
        tw_lo = sum(s.teamwork_low for s in scoring_segments) / n
        tw_hi = sum(s.teamwork_high for s in scoring_segments) / n
        pres_lo = sum(s.presentation_low for s in scoring_segments) / n
        pres_hi = sum(s.presentation_high for s in scoring_segments) / n

    # Weighted overall (and weighted CI bounds)
    def _weighted(t: float, tc: float, tw: float, p: float) -> float:
        return (
            t * WEIGHTS["timing"]
            + tc * WEIGHTS["technique"]
            + tw * WEIGHTS["teamwork"]
            + p * WEIGHTS["presentation"]
        )

    overall = _weighted(timing, technique, teamwork, presentation)
    overall_lo = _weighted(timing_lo, tech_lo, tw_lo, pres_lo)
    overall_hi = _weighted(timing_hi, tech_hi, tw_hi, pres_hi)
    low_conf = any(
        hi - lo > 2.0
        for lo, hi in (
            (timing_lo, timing_hi),
            (tech_lo, tech_hi),
            (tw_lo, tw_hi),
            (pres_lo, pres_hi),
        )
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

    # Collect pattern details (deduplicated by name)
    seen_detail_names: set[str] = set()
    all_pattern_details = []
    for seg in scoring_segments:
        for pd in seg.pattern_details:
            name = pd.get("name", "").lower()
            if name and name not in seen_detail_names:
                seen_detail_names.add(name)
                all_pattern_details.append(pd)

    # Use summary for strengths/improvements, or aggregate
    if summary:
        strengths = summary.highlights[:3] or _collect_top(scoring_segments, "highlights", 3)
        improvements = summary.improvements[:3] or _collect_top(scoring_segments, "improvements", 3)
        impression = summary.raw_data.get("overall_impression", "")
        reasoning_final = dict(summary.reasoning) if summary.reasoning else {}
    else:
        strengths = _collect_top(scoring_segments, "highlights", 3)
        improvements = _collect_top(scoring_segments, "improvements", 3)
        impression = ""
        reasoning_final = {}

    if not reasoning_final:
        for seg in scoring_segments:
            for cat, text in seg.reasoning.items():
                if text and cat not in reasoning_final:
                    reasoning_final[cat] = text

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
        timing_low=round(timing_lo, 1),
        timing_high=round(timing_hi, 1),
        technique_low=round(tech_lo, 1),
        technique_high=round(tech_hi, 1),
        teamwork_low=round(tw_lo, 1),
        teamwork_high=round(tw_hi, 1),
        presentation_low=round(pres_lo, 1),
        presentation_high=round(pres_hi, 1),
        overall_low=round(overall_lo, 1),
        overall_high=round(overall_hi, 1),
        low_confidence=low_conf,
        reasoning=reasoning_final,
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
        pattern_details=all_pattern_details,
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
