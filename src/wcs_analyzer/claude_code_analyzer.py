"""Claude Code CLI-based analysis — uses the locally installed claude CLI."""

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from .exceptions import AnalysisError
from .prompts import DANCER_CONTEXT_TEMPLATE, SYSTEM_PROMPT
from .scoring import SegmentAnalysis
from .video import extract_frames

logger = logging.getLogger(__name__)

# JSON schema for structured output from Claude Code
_ANALYSIS_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "timing": {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "on_beat": {"type": "boolean"},
                "off_beat_moments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "timestamp_approx": {"type": "string"},
                            "description": {"type": "string"},
                            "beat_count": {"type": "string"},
                        },
                    },
                },
                "notes": {"type": "string"},
            },
            "required": ["score"],
        },
        "technique": {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "posture": {"type": "object", "properties": {"score": {"type": "number"}, "notes": {"type": "string"}}},
                "extension": {"type": "object", "properties": {"score": {"type": "number"}, "notes": {"type": "string"}}},
                "footwork": {"type": "object", "properties": {"score": {"type": "number"}, "notes": {"type": "string"}}},
                "slot": {"type": "object", "properties": {"score": {"type": "number"}, "notes": {"type": "string"}}},
                "notes": {"type": "string"},
            },
            "required": ["score"],
        },
        "teamwork": {
            "type": "object",
            "properties": {"score": {"type": "number"}, "notes": {"type": "string"}},
            "required": ["score"],
        },
        "presentation": {
            "type": "object",
            "properties": {"score": {"type": "number"}, "notes": {"type": "string"}},
            "required": ["score"],
        },
        "patterns_identified": {"type": "array", "items": {"type": "string"}},
        "highlights": {"type": "array", "items": {"type": "string"}},
        "improvements": {"type": "array", "items": {"type": "string"}},
        "lead": {
            "type": "object",
            "properties": {
                "technique_score": {"type": "number"},
                "presentation_score": {"type": "number"},
                "notes": {"type": "string"},
            },
        },
        "follow": {
            "type": "object",
            "properties": {
                "technique_score": {"type": "number"},
                "presentation_score": {"type": "number"},
                "notes": {"type": "string"},
            },
        },
        "overall_impression": {"type": "string"},
    },
    "required": ["timing", "technique", "teamwork", "presentation"],
})


def _check_claude_cli() -> str:
    """Check that the claude CLI is installed and return its path."""
    path = shutil.which("claude")
    if not path:
        raise AnalysisError(
            "Claude Code CLI not found. Install it from https://claude.ai/code "
            "or use --provider gemini instead."
        )
    return path


def analyze_dance_claude_code(
    video_path: Path,
    detail: str = "medium",
    dancers: str | None = None,
    fps: float = 3.0,
) -> list[SegmentAnalysis]:
    """Analyze a dance video using the Claude Code CLI.

    Extracts frames, saves them as images, then invokes the claude CLI
    to read and analyze them. No separate API key needed — uses the
    user's existing Claude Code authentication.

    Args:
        video_path: Path to the video file.
        detail: Analysis detail level (low/medium/high).
        dancers: Optional description of which dancers to focus on.
        fps: Frames per second to extract.

    Returns:
        List containing a single SegmentAnalysis.
    """
    claude_path = _check_claude_cli()

    # Extract frames to temp directory
    logger.info("Extracting frames for Claude Code analysis...")
    frames = extract_frames(video_path, fps=fps)

    if not frames.images:
        raise AnalysisError("No frames extracted from video")

    # Select frames based on detail level
    if detail == "low":
        step = max(1, len(frames.images) // 6)
    elif detail == "high":
        step = max(1, len(frames.images) // 20)
    else:
        step = max(1, len(frames.images) // 12)

    selected = frames.images[::step][:20]  # cap at 20 frames
    logger.info("Selected %d frames for analysis", len(selected))

    # Save frames to temp directory
    with tempfile.TemporaryDirectory(prefix="wcs_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        frame_paths = []
        for i, img_b64 in enumerate(selected):
            import base64
            frame_file = tmp_path / f"frame_{i:03d}.jpg"
            frame_file.write_bytes(base64.b64decode(img_b64))
            frame_paths.append(str(frame_file))

        # Build the prompt
        dancer_context = ""
        if dancers:
            dancer_context = DANCER_CONTEXT_TEMPLATE.format(dancer_description=dancers) + "\n\n"

        frame_list = "\n".join(f"- {p}" for p in frame_paths)
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"{dancer_context}"
            f"Analyze these {len(frame_paths)} sequential frames from a West Coast Swing dance video "
            f"({frames.duration:.0f}s total, sampled at {fps} fps).\n\n"
            f"Read each of these image files and analyze the dance:\n{frame_list}\n\n"
            f"After viewing all frames, provide your WSDC-style scoring analysis. "
            f"Include scores for timing, technique, teamwork, and presentation (1-10 each), "
            f"plus lead/follow individual scores, patterns identified, highlights, "
            f"and specific improvement suggestions."
        )

        # Call claude CLI
        logger.info("Invoking Claude Code CLI...")
        result = _call_claude_cli(claude_path, prompt)

    return [_parse_response(result, frames.duration)]


def _call_claude_cli(claude_path: str, prompt: str, timeout: int = 300) -> dict:
    """Call the claude CLI and return parsed JSON response."""
    cmd = [
        claude_path,
        "-p", prompt,
        "--output-format", "json",
        "--json-schema", _ANALYSIS_SCHEMA,
        "--allowedTools", "Read",
        "--max-turns", "5",
        "--bare",
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise AnalysisError(
            f"Claude Code CLI timed out after {timeout}s. "
            "Try --detail low for fewer frames."
        )
    except FileNotFoundError:
        raise AnalysisError("Claude Code CLI not found")

    if proc.returncode != 0:
        stderr = proc.stderr.strip()[:500] if proc.stderr else "unknown error"
        raise AnalysisError(f"Claude Code CLI failed: {stderr}")

    # Parse the outer JSON envelope from claude --output-format json
    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise AnalysisError(f"Failed to parse Claude Code output: {proc.stdout[:300]}")

    # The structured output is in the "result" field
    result_text = envelope.get("result", "")

    # Try to parse the result as JSON directly (from --json-schema)
    if isinstance(result_text, dict):
        return result_text

    # Sometimes it comes back as a JSON string
    try:
        return json.loads(result_text)  # type: ignore[no-any-return]
    except (json.JSONDecodeError, TypeError):
        # Try to extract JSON from markdown fences
        text = str(result_text).strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        try:
            return json.loads(text.strip())  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            raise AnalysisError(f"Could not parse analysis result: {str(result_text)[:300]}")


def _parse_response(data: dict, duration: float) -> SegmentAnalysis:
    """Convert parsed JSON dict into a SegmentAnalysis."""
    return SegmentAnalysis(
        start_time=0.0,
        end_time=duration,
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
