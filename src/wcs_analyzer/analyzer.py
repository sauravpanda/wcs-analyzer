"""LLM orchestration — send frames to Claude for WCS analysis."""

import json
import time

import anthropic

from .audio import AudioFeatures, format_beat_context
from .prompts import SEGMENT_ANALYSIS_PROMPT, SUMMARY_PROMPT, SYSTEM_PROMPT
from .scoring import SegmentAnalysis
from .video import FrameData, group_frames_by_phrase


# Max frames per API call to stay within token limits
MAX_FRAMES_PER_CALL = 16


def analyze_dance(
    frames: FrameData,
    audio: AudioFeatures,
    model: str = "claude-sonnet-4-6",
    detail: str = "medium",
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

    # Group frames into 8-count phrases
    phrases = group_frames_by_phrase(frames, beats_per_phrase=8, bpm=audio.bpm)

    # Adjust granularity based on detail level
    if detail == "low":
        # Merge every 2 phrases
        merged = []
        for i in range(0, len(phrases), 2):
            group = phrases[i:i + 2]
            merged.append({
                "images": [img for p in group for img in p["images"]][:MAX_FRAMES_PER_CALL],
                "timestamps": [t for p in group for t in p["timestamps"]],
                "phrase_index": group[0]["phrase_index"],
                "start_time": group[0]["start_time"],
                "end_time": group[-1]["end_time"],
            })
        phrases = merged
    elif detail == "high":
        # Keep all phrases, but cap frames per phrase
        for p in phrases:
            if len(p["images"]) > MAX_FRAMES_PER_CALL:
                step = len(p["images"]) // MAX_FRAMES_PER_CALL
                p["images"] = p["images"][::step][:MAX_FRAMES_PER_CALL]
                p["timestamps"] = p["timestamps"][::step][:MAX_FRAMES_PER_CALL]

    # Analyze each segment
    segment_results: list[SegmentAnalysis] = []

    for phrase in phrases:
        # Cap frames
        images = phrase["images"][:MAX_FRAMES_PER_CALL]

        beat_context = format_beat_context(audio, phrase["start_time"], phrase["end_time"])
        prompt = SEGMENT_ANALYSIS_PROMPT.format(beat_context=beat_context)

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
            assert hasattr(block, "text"), f"Unexpected block type: {type(block)}"
            return block.text  # type: ignore[union-attr]
        except anthropic.RateLimitError:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
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
        # Fallback: return a minimal result
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

    prompt = SUMMARY_PROMPT.format(
        num_segments=len(segments),
        duration=audio.duration,
        bpm=audio.bpm,
        segment_results="\n\n".join(segment_texts),
    )

    result = _call_claude(client, model, [{"type": "text", "text": prompt}])
    return _parse_segment_json(
        result,
        start_time=0.0,
        end_time=audio.duration,
    )
