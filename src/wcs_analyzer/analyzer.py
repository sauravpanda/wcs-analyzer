"""LLM orchestration — send frames to Claude for WCS analysis."""

import json
import logging
import time

import anthropic

from dataclasses import dataclass

from .audio import AudioFeatures, format_beat_context
from .exceptions import AnalysisError
from .pricing import UsageTotals
from .prompts import (
    DANCER_CONTEXT_TEMPLATE,
    PATTERN_SEGMENTATION_PROMPT,
    SEGMENT_ANALYSIS_PROMPT,
    SUMMARY_PROMPT,
    SYSTEM_PROMPT,
)
from .scoring import SegmentAnalysis
from .video import FrameData, group_frames_by_phrase

logger = logging.getLogger(__name__)


# Max frames per API call to stay within token limits
MAX_FRAMES_PER_CALL = 16

# Approximate token budget for images in a single API call.
# Claude charges ~1600 tokens per 768px image. We leave room for the
# system prompt (~800 tokens), segment prompt (~400 tokens), and
# response (4096 tokens).  With a 200k input context, a safe image
# budget is ~180k tokens.
_TOKENS_PER_IMAGE_ESTIMATE = 1600
_IMAGE_TOKEN_BUDGET = 180_000

# Maximum characters for the summary prompt's segment_results block.
# Keeps the summary well within context limits for long videos.
_MAX_SUMMARY_CHARS = 100_000


def _effective_max_frames() -> int:
    """Compute effective max frames per call based on token budget."""
    max_by_budget = _IMAGE_TOKEN_BUDGET // _TOKENS_PER_IMAGE_ESTIMATE
    return min(MAX_FRAMES_PER_CALL, max_by_budget)


SCORE_MIN = 1.0
SCORE_MAX = 10.0


def clamp_score(value: object, default: float = 5.0) -> float:
    """Coerce a raw JSON value to a float score clamped to [1, 10].

    Invalid or missing values fall back to `default`. A value of 0 is
    treated as missing so partner-specific fields (lead/follow) can
    signal absence with 0 and still clamp correctly via clamp_partner.
    """
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if v != v:  # NaN
        return default
    if v < SCORE_MIN:
        if v <= 0:
            return default
        return SCORE_MIN
    if v > SCORE_MAX:
        return SCORE_MAX
    return v


def clamp_partner(value: object) -> float:
    """Clamp a partner-specific score, preserving 0 as 'not detected'."""
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if v != v or v <= 0:
        return 0.0
    return min(SCORE_MAX, max(SCORE_MIN, v))


def _parse_ci(cat: dict, score: float) -> tuple[float, float]:
    """Extract score_low / score_high, clamped and ordered around score."""
    lo = clamp_score(cat.get("score_low"), default=score)
    hi = clamp_score(cat.get("score_high"), default=score)
    if lo > hi:
        lo, hi = hi, lo
    return min(lo, score), max(hi, score)


def _sub_score(technique: dict, key: str) -> float:
    """Extract a technique sub-score from either nested or flat summary format."""
    sub = technique.get(key)
    if isinstance(sub, dict):
        return clamp_score(sub.get("score", 5))
    return clamp_score(technique.get(f"{key}_score", 5))


def parse_segment_data(
    data: dict,
    start_time: float,
    end_time: float,
    usage: UsageTotals | None = None,
) -> SegmentAnalysis:
    """Build a SegmentAnalysis from a parsed JSON dict.

    Shared by the Claude, Gemini, and Claude Code providers so that
    score clamping and confidence-interval parsing behave identically.
    """
    timing = data.get("timing") or {}
    technique = data.get("technique") or {}
    teamwork = data.get("teamwork") or {}
    presentation = data.get("presentation") or {}
    lead = data.get("lead") or {}
    follow = data.get("follow") or {}

    timing_score = clamp_score(timing.get("score", 5))
    technique_score = clamp_score(technique.get("score", 5))
    teamwork_score = clamp_score(teamwork.get("score", 5))
    presentation_score = clamp_score(presentation.get("score", 5))

    t_lo, t_hi = _parse_ci(timing, timing_score)
    tc_lo, tc_hi = _parse_ci(technique, technique_score)
    tw_lo, tw_hi = _parse_ci(teamwork, teamwork_score)
    p_lo, p_hi = _parse_ci(presentation, presentation_score)

    reasoning = {
        cat: str(block.get("reasoning") or "").strip()
        for cat, block in (
            ("timing", timing),
            ("technique", technique),
            ("teamwork", teamwork),
            ("presentation", presentation),
        )
        if block.get("reasoning")
    }

    # Structured timeline format takes precedence over the flat
    # patterns_identified list — forces the model to commit to every
    # time window and enumerate patterns chronologically, which
    # catches fine-grained patterns that a single flat enumeration
    # would drop. Falls back to the legacy flat format for backward
    # compat with cached segments and older prompts.
    timeline_raw = data.get("pattern_timeline")
    pattern_timeline: list[dict] = []
    if isinstance(timeline_raw, list) and timeline_raw:
        patterns_from_timeline: list[dict] = []
        for entry in timeline_raw:
            if not isinstance(entry, dict):
                continue
            try:
                start = float(entry.get("start_time", 0.0))
                end = float(entry.get("end_time", start))
            except (TypeError, ValueError):
                continue
            names = entry.get("patterns") or []
            if not isinstance(names, list):
                continue
            clean_names = [str(n).strip() for n in names if n]
            if not clean_names:
                continue
            pattern_timeline.append({
                "start_time": start,
                "end_time": end,
                "patterns": clean_names,
                "pattern_details": [
                    {
                        "name": name,
                        "quality": entry.get("quality"),
                        "timing": entry.get("timing"),
                        "notes": entry.get("notes", ""),
                    }
                    for name in clean_names
                ],
            })
            for name in clean_names:
                patterns_from_timeline.append({
                    "name": name,
                    "quality": entry.get("quality"),
                    "timing": entry.get("timing"),
                    "notes": entry.get("notes", ""),
                })
        patterns_raw: list = patterns_from_timeline
    else:
        patterns_raw = data.get("patterns_identified", data.get("patterns_seen", []))

    return SegmentAnalysis(
        start_time=start_time,
        end_time=end_time,
        timing_score=timing_score,
        technique_score=technique_score,
        teamwork_score=teamwork_score,
        presentation_score=presentation_score,
        timing_low=t_lo,
        timing_high=t_hi,
        technique_low=tc_lo,
        technique_high=tc_hi,
        teamwork_low=tw_lo,
        teamwork_high=tw_hi,
        presentation_low=p_lo,
        presentation_high=p_hi,
        posture_score=_sub_score(technique, "posture"),
        extension_score=_sub_score(technique, "extension"),
        footwork_score=_sub_score(technique, "footwork"),
        slot_score=_sub_score(technique, "slot"),
        reasoning=reasoning,
        pattern_timeline=pattern_timeline,
        off_beat_moments=timing.get("off_beat_moments", []) or [],
        patterns=_extract_pattern_names(patterns_raw),
        pattern_details=_extract_pattern_details(patterns_raw),
        highlights=data.get("highlights", data.get("top_strengths", [])) or [],
        improvements=data.get("improvements", data.get("top_improvements", [])) or [],
        lead_technique=clamp_partner(lead.get("technique_score", 0)),
        lead_presentation=clamp_partner(lead.get("presentation_score", 0)),
        lead_notes=lead.get("notes", "") or "",
        follow_technique=clamp_partner(follow.get("technique_score", 0)),
        follow_presentation=clamp_partner(follow.get("presentation_score", 0)),
        follow_notes=follow.get("notes", "") or "",
        usage=usage or UsageTotals(),
        raw_data=data,
    )


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def safe_parse_json(raw: str) -> dict | None:
    """Parse LLM response text as JSON, stripping markdown fences.

    Returns None on failure rather than raising.
    """
    try:
        result = json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None


_RETRY_HINT = (
    "Your previous response could not be parsed as JSON. "
    "Return ONLY a single valid JSON object matching the schema above. "
    "No prose, no markdown fences, no commentary."
)


@dataclass
class PatternSegment:
    """A single pattern detected on the video timeline."""

    start_time: float
    end_time: float
    name: str
    confidence: float = 0.0


def detect_pattern_timeline(
    client: anthropic.Anthropic,
    model: str,
    frames: FrameData,
    max_frames: int = 24,
) -> list[PatternSegment]:
    """Run a single LLM call to segment the video into named patterns.

    Uses up to `max_frames` evenly-spaced frames from the full video so
    the call stays cheap (one request instead of per-phrase). Returns an
    empty list if the response can't be parsed.
    """
    if not frames.images:
        return []

    # Subsample evenly across the clip
    n = len(frames.images)
    if n <= max_frames:
        selected_idx = list(range(n))
    else:
        step = n / max_frames
        selected_idx = [int(i * step) for i in range(max_frames)]
    selected_images = [frames.images[i] for i in selected_idx]

    # Rough expectation: ~one pattern per 3 seconds of video
    expected_count = max(1, int(frames.duration / 3.0))
    prompt = PATTERN_SEGMENTATION_PROMPT.format(
        duration=frames.duration,
        num_frames=len(selected_images),
        expected_count=expected_count,
    )

    content: list[dict] = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": img},
        }
        for img in selected_images
    ]
    content.append({"type": "text", "text": prompt})

    raw, _usage = _call_claude(client, model, content)
    data = safe_parse_json(raw)
    if data is None or "patterns" not in data:
        logger.warning("Pattern segmentation returned unparseable response: %s", raw[:200])
        return []
    return _parse_pattern_timeline(data.get("patterns", []), frames.duration)


def _parse_pattern_timeline(items: list, duration: float) -> list[PatternSegment]:
    """Validate, clamp, and sort a list of pattern timeline entries."""
    segments: list[PatternSegment] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            start = max(0.0, float(item.get("start_time", 0.0)))
            end = float(item.get("end_time", start))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        start = min(start, duration)
        end = min(end, duration)
        if end <= start:
            continue
        name = str(item.get("name", "")).strip() or "unknown"
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        segments.append(PatternSegment(
            start_time=start, end_time=end, name=name, confidence=confidence,
        ))
    segments.sort(key=lambda s: s.start_time)
    return segments


def _default_segment(
    start_time: float, end_time: float, raw: str,
    usage: UsageTotals | None = None,
) -> SegmentAnalysis:
    return SegmentAnalysis(
        start_time=start_time,
        end_time=end_time,
        timing_score=5.0,
        technique_score=5.0,
        teamwork_score=5.0,
        presentation_score=5.0,
        usage=usage or UsageTotals(),
        raw_data={"error": "Failed to parse response", "raw": raw[:500]},
    )


def analyze_dance(
    frames: FrameData,
    audio: AudioFeatures,
    model: str = "claude-sonnet-4-6",
    detail: str = "medium",
    dancers: str | None = None,
) -> list[SegmentAnalysis]:
    """Analyze a full dance by breaking it into segments and sending to Claude.

    Args:
        frames: Extracted video frames.
        audio: Audio features with beat data.
        model: Claude model ID to use.
        detail: Analysis detail level (low/medium/high).

    Returns:
        List of SegmentAnalysis results, one per segment plus a final summary.
    """
    client = anthropic.Anthropic()
    max_frames = _effective_max_frames()

    # Group frames into 8-count phrases
    phrases = group_frames_by_phrase(frames, beats_per_phrase=8, bpm=audio.bpm)
    logger.info(
        "Grouped %d frames into %d phrases for analysis (max %d frames/call)",
        len(frames.images), len(phrases), max_frames,
    )

    # Adjust granularity based on detail level
    if detail == "low":
        # Merge every 2 phrases
        merged = []
        for i in range(0, len(phrases), 2):
            group = phrases[i:i + 2]
            merged.append({
                "images": [img for p in group for img in p["images"]][:max_frames],
                "timestamps": [t for p in group for t in p["timestamps"]],
                "phrase_index": group[0]["phrase_index"],
                "start_time": group[0]["start_time"],
                "end_time": group[-1]["end_time"],
            })
        phrases = merged
    elif detail == "high":
        # Keep all phrases, but cap frames per phrase
        for p in phrases:
            if len(p["images"]) > max_frames:
                step = len(p["images"]) // max_frames
                p["images"] = p["images"][::step][:max_frames]
                p["timestamps"] = p["timestamps"][::step][:max_frames]

    # Analyze each segment
    segment_results: list[SegmentAnalysis] = []

    for i, phrase in enumerate(phrases):
        logger.debug("Analyzing phrase %d/%d (%.1f-%.1fs)", i + 1, len(phrases), phrase["start_time"], phrase["end_time"])
        # Cap frames
        images = phrase["images"][:max_frames]

        dancer_context = ""
        if dancers:
            dancer_context = DANCER_CONTEXT_TEMPLATE.format(dancer_description=dancers) + "\n"
        beat_context = format_beat_context(audio, phrase["start_time"], phrase["end_time"])
        prompt = SEGMENT_ANALYSIS_PROMPT.format(dancer_context=dancer_context, beat_context=beat_context)

        # Build content with interleaved images
        content: list[dict] = []
        for img_b64 in images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img_b64,
                },
            })
        content.append({"type": "text", "text": prompt})

        parsed = _call_and_parse(
            client, model, content,
            start_time=phrase["start_time"],
            end_time=phrase["end_time"],
        )
        segment_results.append(parsed)

    # Get final summary if we have multiple segments
    if len(segment_results) > 1:
        summary = _get_summary(client, model, segment_results, audio)
        segment_results.append(summary)

    return segment_results


def _call_claude(
    client: anthropic.Anthropic,
    model: str,
    content: list,  # type: ignore[type-arg]
    max_retries: int = 3,
    temperature: float = 0.0,
) -> tuple[str, UsageTotals]:
    """Call Claude API with retries for rate limiting.

    Temperature defaults to 0 for reproducibility — rubric grading
    wants stable, replayable scores, not creative variation.

    Returns the response text plus a UsageTotals capturing the token
    counts and estimated cost for this call.
    """
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                temperature=temperature,
                messages=[{"role": "user", "content": content}],
            )
            block = response.content[0]
            if not hasattr(block, "text"):
                raise AnalysisError(f"Unexpected response block type: {type(block)}")
            usage_obj = getattr(response, "usage", None)
            input_tokens = int(getattr(usage_obj, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage_obj, "output_tokens", 0) or 0)
            usage = UsageTotals.from_counts(model, input_tokens, output_tokens)
            return block.text, usage  # type: ignore[union-attr]
        except anthropic.RateLimitError:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning("Rate limited by API, retrying in %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)
            else:
                raise
    return "", UsageTotals(model=model)


def _parse_segment_json(raw: str, start_time: float, end_time: float) -> SegmentAnalysis:
    """Parse Claude's JSON response into a SegmentAnalysis (no retry)."""
    data = safe_parse_json(raw)
    if data is None:
        logger.warning(
            "Failed to parse Claude response as JSON for segment %.1f-%.1fs. "
            "Using default scores. Raw response: %s",
            start_time, end_time, raw[:200],
        )
        return _default_segment(start_time, end_time, raw)
    return parse_segment_data(data, start_time, end_time)


def _call_and_parse(
    client: anthropic.Anthropic,
    model: str,
    content: list,  # type: ignore[type-arg]
    start_time: float,
    end_time: float,
) -> SegmentAnalysis:
    """Call Claude, parse the response, and retry once on JSON parse failure.

    Accumulates usage across both the initial call and (if needed) the
    corrective retry, so the segment's `usage` reflects actual spend.
    """
    raw, usage = _call_claude(client, model, content)
    data = safe_parse_json(raw)
    if data is None:
        logger.warning(
            "Segment %.1f-%.1fs: JSON parse failed, retrying with corrective prompt",
            start_time, end_time,
        )
        retry_content = list(content) + [{"type": "text", "text": _RETRY_HINT}]
        raw, retry_usage = _call_claude(client, model, retry_content)
        usage = usage.add(retry_usage)
        data = safe_parse_json(raw)
    if data is None:
        logger.warning(
            "Segment %.1f-%.1fs: retry also failed to parse. Using default scores. Raw: %s",
            start_time, end_time, raw[:200],
        )
        return _default_segment(start_time, end_time, raw, usage=usage)
    return parse_segment_data(data, start_time, end_time, usage=usage)


def _get_summary(
    client: anthropic.Anthropic,
    model: str,
    segments: list[SegmentAnalysis],
    audio: AudioFeatures,
) -> SegmentAnalysis:
    """Get a final summary analysis across all segments."""
    segment_texts = []
    for i, seg in enumerate(segments):
        segment_texts.append(
            f"Segment {i + 1} ({seg.start_time:.1f}s - {seg.end_time:.1f}s): "
            f"Timing={seg.timing_score}, Technique={seg.technique_score}, "
            f"Teamwork={seg.teamwork_score}, Presentation={seg.presentation_score}"
            + (f"\nOff-beat moments: {json.dumps(seg.off_beat_moments)}" if seg.off_beat_moments else "")
            + (f"\nPatterns: {', '.join(seg.patterns)}" if seg.patterns else "")
            + (f"\nHighlights: {'; '.join(seg.highlights)}" if seg.highlights else "")
            + (f"\nImprovements: {'; '.join(seg.improvements)}" if seg.improvements else "")
        )

    combined = "\n\n".join(segment_texts)
    if len(combined) > _MAX_SUMMARY_CHARS:
        logger.warning(
            "Segment results too long for summary (%d chars), truncating to %d",
            len(combined), _MAX_SUMMARY_CHARS,
        )
        combined = combined[:_MAX_SUMMARY_CHARS] + "\n\n[... truncated for length]"

    prompt = SUMMARY_PROMPT.format(
        num_segments=len(segments),
        duration=audio.duration,
        bpm=audio.bpm,
        segment_results=combined,
    )

    seg = _call_and_parse(
        client, model,
        [{"type": "text", "text": prompt}],
        start_time=0.0,
        end_time=audio.duration,
    )
    seg.is_summary = True
    return seg


def _extract_pattern_names(patterns: list) -> list[str]:
    """Extract plain pattern names from mixed list of strings/dicts."""
    names = []
    for p in patterns:
        if isinstance(p, str):
            names.append(p)
        elif isinstance(p, dict):
            names.append(p.get("name", "unknown"))
    return names


def _extract_pattern_details(patterns: list) -> list[dict]:
    """Extract rich pattern details, normalizing strings to dicts.

    String patterns carry no quality/timing data, so those fields are
    left absent rather than set to misleading "solid"/"on_beat"
    defaults — the report then renders "?" for them, which is honest.
    """
    details = []
    for p in patterns:
        if isinstance(p, dict) and "name" in p:
            details.append(p)
        elif isinstance(p, str):
            details.append({"name": p, "notes": ""})
    return details
