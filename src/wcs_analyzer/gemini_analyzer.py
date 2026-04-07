"""Gemini-based video analysis — upload full video for native analysis."""

import json
import logging
import time
from pathlib import Path

from google import genai
from google.genai import types

from .exceptions import AnalysisError
from .prompts import GEMINI_VIDEO_PROMPT, SYSTEM_PROMPT
from .scoring import SegmentAnalysis

logger = logging.getLogger(__name__)

# Gemini FPS settings by detail level
_DETAIL_FPS = {"low": 1, "medium": 2, "high": 5}

# Inline upload limit (bytes). Larger files use the File API.
_INLINE_LIMIT = 20 * 1024 * 1024  # 20 MB


def analyze_dance_gemini(
    video_path: Path,
    model: str = "gemini-2.5-flash",
    detail: str = "medium",
) -> list[SegmentAnalysis]:
    """Analyze a dance video using Gemini's native video understanding.

    Uploads the full video (with audio) so Gemini can see motion and hear
    the music — no frame extraction or beat detection needed.

    Args:
        video_path: Path to the video file.
        model: Gemini model ID.
        detail: Analysis detail level (low/medium/high).

    Returns:
        List containing a single SegmentAnalysis covering the full video.
    """
    client = genai.Client()
    fps = _DETAIL_FPS.get(detail, 2)
    file_size = video_path.stat().st_size

    # Determine media resolution
    if detail == "high":
        resolution = types.MediaResolution.MEDIA_RESOLUTION_HIGH
    elif detail == "low":
        resolution = types.MediaResolution.MEDIA_RESOLUTION_LOW
    else:
        resolution = types.MediaResolution.MEDIA_RESOLUTION_MEDIUM

    # Upload or inline
    if file_size > _INLINE_LIMIT:
        video_part = _upload_video(client, video_path, fps)
    else:
        video_part = _inline_video(video_path, fps)

    prompt = GEMINI_VIDEO_PROMPT
    logger.info(
        "Sending video to Gemini (%s, %d fps, %s resolution)",
        model, fps, detail,
    )

    result = _call_gemini(
        client, model,
        contents=[video_part, prompt],
        resolution=resolution,
    )
    parsed = _parse_response(result)
    return [parsed]


def _upload_video(client: genai.Client, video_path: Path, fps: int) -> types.Part:
    """Upload a video via the File API and wait for processing."""
    logger.info("Uploading video via File API (%s)...", video_path.name)
    video_file = client.files.upload(file=str(video_path))
    logger.debug("Upload complete: %s, state: %s", video_file.name, video_file.state)

    # Poll until ready
    while video_file.state.name == "PROCESSING":  # type: ignore[union-attr]
        logger.debug("Video still processing, waiting 5s...")
        time.sleep(5)
        video_file = client.files.get(name=video_file.name)  # type: ignore[arg-type]

    if video_file.state.name == "FAILED":  # type: ignore[union-attr]
        raise AnalysisError(f"Gemini video processing failed for {video_path.name}")

    logger.info("Video ready for analysis")
    return types.Part(
        file_data=types.FileData(file_uri=video_file.uri),  # type: ignore[arg-type]
        video_metadata=types.VideoMetadata(fps=fps),
    )


def _inline_video(video_path: Path, fps: int) -> types.Part:
    """Create an inline video part for small files."""
    logger.info("Sending video inline (%s, %.1f MB)", video_path.name, video_path.stat().st_size / 1024 / 1024)
    suffix = video_path.suffix.lower()
    mime_map = {
        ".mp4": "video/mp4",
        ".mov": "video/mov",
        ".avi": "video/avi",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
        ".wmv": "video/wmv",
        ".flv": "video/x-flv",
        ".mpeg": "video/mpeg",
        ".mpg": "video/mpeg",
        ".3gp": "video/3gpp",
    }
    mime = mime_map.get(suffix, "video/mp4")
    video_bytes = video_path.read_bytes()
    return types.Part(
        inline_data=types.Blob(data=video_bytes, mime_type=mime),
        video_metadata=types.VideoMetadata(fps=fps),
    )


def _call_gemini(
    client: genai.Client,
    model: str,
    contents: list,
    resolution: types.MediaResolution,
    max_retries: int = 3,
) -> str:
    """Call Gemini API with retries."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=8192,
                    media_resolution=resolution,
                ),
            )
            if response.text:
                return response.text
            raise AnalysisError("Gemini returned empty response")
        except Exception as e:
            # Retry on transient errors
            err_str = str(e).lower()
            if any(k in err_str for k in ("rate", "quota", "429", "503", "overloaded")) and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning("Gemini rate limited, retrying in %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)
            else:
                raise AnalysisError(f"Gemini API error: {e}") from e
    return ""


def _parse_response(raw: str) -> SegmentAnalysis:
    """Parse Gemini's JSON response into a SegmentAnalysis."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse Gemini response as JSON. Raw: %s", raw[:300])
        return SegmentAnalysis(
            start_time=0.0, end_time=0.0,
            timing_score=5.0, technique_score=5.0,
            teamwork_score=5.0, presentation_score=5.0,
            raw_data={"error": "Failed to parse response", "raw": raw[:500]},
        )

    return SegmentAnalysis(
        start_time=0.0,
        end_time=0.0,
        timing_score=float(data.get("timing", {}).get("score", 5)),
        technique_score=float(data.get("technique", {}).get("score", 5)),
        teamwork_score=float(data.get("teamwork", {}).get("score", 5)),
        presentation_score=float(data.get("presentation", {}).get("score", 5)),
        off_beat_moments=data.get("timing", {}).get("off_beat_moments", []),
        posture_score=float(data.get("technique", {}).get("posture", {}).get("score", 5)),
        extension_score=float(data.get("technique", {}).get("extension", {}).get("score", 5)),
        footwork_score=float(data.get("technique", {}).get("footwork", {}).get("score", 5)),
        slot_score=float(data.get("technique", {}).get("slot", {}).get("score", 5)),
        patterns=data.get("patterns_identified", data.get("patterns_seen", [])),
        highlights=data.get("highlights", data.get("top_strengths", [])),
        improvements=data.get("improvements", data.get("top_improvements", [])),
        lead_technique=float(data.get("lead", {}).get("technique_score", 0)),
        lead_presentation=float(data.get("lead", {}).get("presentation_score", 0)),
        lead_notes=data.get("lead", {}).get("notes", ""),
        follow_technique=float(data.get("follow", {}).get("technique_score", 0)),
        follow_presentation=float(data.get("follow", {}).get("presentation_score", 0)),
        follow_notes=data.get("follow", {}).get("notes", ""),
        raw_data=data,
    )
