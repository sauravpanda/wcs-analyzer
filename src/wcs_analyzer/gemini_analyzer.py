"""Gemini-based video analysis — upload full video for native analysis."""

import logging
import time
from pathlib import Path

from google import genai
from google.genai import types

from .analyzer import _default_segment, parse_segment_data, safe_parse_json
from .exceptions import AnalysisError
from .pricing import UsageTotals
from .prompts import DANCER_CONTEXT_TEMPLATE, GEMINI_VIDEO_PROMPT, SYSTEM_PROMPT
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
    dancers: str | None = None,
    pose_context: str | None = None,
) -> list[SegmentAnalysis]:
    """Analyze a dance video using Gemini's native video understanding.

    Uploads the full video (with audio) so Gemini can see motion and hear
    the music — no frame extraction or beat detection needed.

    Args:
        video_path: Path to the video file.
        model: Gemini model ID.
        detail: Analysis detail level (low/medium/high).
        dancers: Optional description of which dancers to focus on.

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

    dancer_context = ""
    if dancers:
        dancer_context = DANCER_CONTEXT_TEMPLATE.format(dancer_description=dancers) + "\n"
    prompt = GEMINI_VIDEO_PROMPT.replace("{dancer_context}", dancer_context)
    if pose_context:
        prompt = f"{pose_context}\n\n{prompt}"
    logger.info(
        "Sending video to Gemini (%s, %d fps, %s resolution)",
        model, fps, detail,
    )

    result, usage = _call_gemini(
        client, model,
        contents=[video_part, prompt],
        resolution=resolution,
    )
    parsed = _parse_response(result, usage)
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
) -> tuple[str, UsageTotals]:
    """Call Gemini API with retries, returning response text + usage."""
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
                meta = getattr(response, "usage_metadata", None)
                input_tokens = int(getattr(meta, "prompt_token_count", 0) or 0)
                output_tokens = int(getattr(meta, "candidates_token_count", 0) or 0)
                usage = UsageTotals.from_counts(model, input_tokens, output_tokens)
                return response.text, usage
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
    return "", UsageTotals(model=model)


def _parse_response(raw: str, usage: UsageTotals | None = None) -> SegmentAnalysis:
    """Parse Gemini's JSON response into a SegmentAnalysis."""
    data = safe_parse_json(raw)
    if data is None:
        logger.warning("Failed to parse Gemini response as JSON. Raw: %s", raw[:300])
        return _default_segment(0.0, 0.0, raw, usage=usage)
    return parse_segment_data(data, start_time=0.0, end_time=0.0, usage=usage)
