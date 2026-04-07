"""LLM orchestration — send frames to Claude for WCS analysis."""

import json
import logging
import time

import anthropic

from .audio import AudioFeatures, format_beat_context
from .exceptions import AnalysisError
from .prompts import DANCER_CONTEXT_TEMPLATE, SEGMENT_ANALYSIS_PROMPT, SUMMARY_PROMPT, SYSTEM_PROMPT
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

        result = _call_claude(client, model, content)
        parsed = _parse_segment_json(result, phrase["start_time"], phrase["end_time"])
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
) -> str:
    """Call Claude API with retries for rate limiting."""
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            block = response.content[0]
            if not hasattr(block, "text"):
                raise AnalysisError(f"Unexpected response block type: {type(block)}")
            return block.text  # type: ignore[union-attr]
        except anthropic.RateLimitError:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning("Rate limited by API, retrying in %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)
            else:
                raise
    return ""


def _parse_segment_json(raw: str, start_time: float, end_time: float) -> SegmentAnalysis:
    """Parse Claude's JSON response into a SegmentAnalysis."""
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(
            "Failed to parse Claude response as JSON for segment %.1f-%.1fs. "
            "Using default scores. Raw response: %s",
            start_time, end_time, raw[:200],
        )
        return SegmentAnalysis(
            start_time=start_time,
            end_time=end_time,
            timing_score=5.0,
            technique_score=5.0,
            teamwork_score=5.0,
            presentation_score=5.0,
            raw_data={"error": "Failed to parse response", "raw": raw[:500]},
        )

    return SegmentAnalysis(
        start_time=start_time,
        end_time=end_time,
        timing_score=float(data.get("timing", {}).get("score", 5)),
        technique_score=float(data.get("technique", {}).get("score", 5)),
        teamwork_score=float(data.get("teamwork", {}).get("score", 5)),
        presentation_score=float(data.get("presentation", {}).get("score", 5)),
        off_beat_moments=data.get("timing", {}).get("off_beat_moments", []),
        posture_score=float(data.get("technique", {}).get("posture", {}).get("score", 5)),
        extension_score=float(data.get("technique", {}).get("extension", {}).get("score", 5)),
        footwork_score=float(data.get("technique", {}).get("footwork", {}).get("score", 5)),
        slot_score=float(data.get("technique", {}).get("slot", {}).get("score", 5)),
        patterns=data.get("patterns_identified", []),
        highlights=data.get("highlights", []),
        improvements=data.get("improvements", []),
        lead_technique=float(data.get("lead", {}).get("technique_score", 0)),
        lead_presentation=float(data.get("lead", {}).get("presentation_score", 0)),
        lead_notes=data.get("lead", {}).get("notes", ""),
        follow_technique=float(data.get("follow", {}).get("technique_score", 0)),
        follow_presentation=float(data.get("follow", {}).get("presentation_score", 0)),
        follow_notes=data.get("follow", {}).get("notes", ""),
        raw_data=data,
    )


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

    result = _call_claude(client, model, [{"type": "text", "text": prompt}])
    return _parse_segment_json(
        result,
        start_time=0.0,
        end_time=audio.duration,
    )
