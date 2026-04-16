"""Claude Code CLI-based analysis — uses the locally installed claude CLI."""

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from .analyzer import parse_segment_data
from .exceptions import AnalysisError
from .pricing import UsageTotals
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

    # FPS per detail level — controls how many frames get analyzed.
    # Higher rates cost more but catch fast WCS footwork; HD pulls 4fps.
    detail_fps = {"low": 1.0, "medium": 2.0, "high": 4.0}
    analysis_fps = detail_fps.get(detail, 2.0)

    # Extract frames at the analysis FPS directly
    logger.info("Extracting frames at %.1f fps for Claude Code analysis...", analysis_fps)
    frames = extract_frames(video_path, fps=analysis_fps)

    if not frames.images:
        raise AnalysisError("No frames extracted from video")

    selected = frames.images
    logger.info("Extracted %d frames (%.0fs video at %.1f fps)", len(selected), frames.duration, analysis_fps)

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
            f"After viewing all frames, provide your WSDC-style scoring analysis as JSON.\n\n"
            f"You MUST respond with ONLY valid JSON in this exact format:\n"
            f'{{"timing": {{"score": <1-10>, "off_beat_moments": [], "notes": "..."}}, '
            f'"technique": {{"score": <1-10>, "posture": {{"score": <1-10>, "notes": "..."}}, '
            f'"extension": {{"score": <1-10>, "notes": "..."}}, "footwork": {{"score": <1-10>, "notes": "..."}}, '
            f'"slot": {{"score": <1-10>, "notes": "..."}}, "notes": "..."}}, '
            f'"teamwork": {{"score": <1-10>, "notes": "..."}}, '
            f'"presentation": {{"score": <1-10>, "notes": "..."}}, '
            f'"patterns_identified": ["..."], "highlights": ["..."], "improvements": ["..."], '
            f'"lead": {{"technique_score": <1-10>, "presentation_score": <1-10>, "notes": "..."}}, '
            f'"follow": {{"technique_score": <1-10>, "presentation_score": <1-10>, "notes": "..."}}, '
            f'"overall_impression": "..."}}\n\n'
            f"Only output valid JSON, no other text."
        )

        # Call claude CLI
        logger.info("Invoking Claude Code CLI...")
        result, usage = _call_claude_cli(claude_path, prompt)

    return [_parse_response(result, frames.duration, usage)]


def _call_claude_cli(claude_path: str, prompt: str, timeout: int = 300) -> tuple[dict, UsageTotals]:
    """Call the claude CLI and return (parsed JSON, usage)."""
    cmd = [
        claude_path,
        "-p", prompt,
        "--output-format", "json",
        "--allowedTools", "Read",
        "--max-turns", "25",
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

    # Parse the outer JSON envelope from claude --output-format json
    try:
        envelope = json.loads(proc.stdout) if proc.stdout else {}
    except json.JSONDecodeError:
        envelope = {}

    # Check for errors in the envelope or return code
    if envelope.get("is_error") or proc.returncode != 0:
        error_msg = envelope.get("result", "") or proc.stderr or "unknown error"
        raise AnalysisError(f"Claude Code CLI failed: {error_msg}")

    usage = _usage_from_envelope(envelope)

    # The structured output is in the "result" field
    result_text = envelope.get("result", "")

    # Try to parse the result as JSON directly (from --json-schema)
    if isinstance(result_text, dict):
        return result_text, usage

    # Sometimes it comes back as a JSON string
    try:
        return json.loads(result_text), usage  # type: ignore[no-any-return]
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to extract JSON from markdown fences
    text = str(result_text).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    try:
        return json.loads(text.strip()), usage  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        pass

    # Last resort: Claude sometimes emits prose, then a fenced JSON block,
    # then more prose. Scan for the first valid JSON object by locating a
    # ```json fence, and if that fails, the first { and balanced }.
    parsed = _extract_json_from_prose(str(result_text))
    if parsed is not None:
        return parsed, usage
    raise AnalysisError(f"Could not parse analysis result: {str(result_text)[:300]}")


def _extract_json_from_prose(text: str) -> dict | None:
    """Pull a JSON object out of a prose+JSON response.

    Claude Code CLI sometimes prefixes its JSON with explanatory prose,
    wraps it in a ```json fence, or appends more prose after. Try three
    strategies in order, returning the first one that parses.
    """
    # 1. Look for a ```json (or bare ```) fence anywhere in the text
    for marker in ("```json", "```"):
        start = text.find(marker)
        if start == -1:
            continue
        content_start = start + len(marker)
        end = text.find("```", content_start)
        if end == -1:
            continue
        candidate = text[content_start:end].strip()
        try:
            result = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            return result

    # 2. Scan for the first top-level { and match its closing }.
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        result = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(result, dict):
                        return result
                    break
        start = text.find("{", start + 1)
    return None


def _usage_from_envelope(envelope: dict) -> UsageTotals:
    """Extract token counts and total cost from the claude CLI envelope.

    The CLI reports `total_cost_usd` directly (authoritative, matches
    actual billing), so we prefer that over re-estimating from our own
    price table. Falls back to token-based estimation if the envelope
    doesn't include a cost.
    """
    usage_block = envelope.get("usage") or {}
    input_tokens = int(usage_block.get("input_tokens", 0) or 0)
    output_tokens = int(usage_block.get("output_tokens", 0) or 0)
    model = envelope.get("model", "") or ""

    cost = envelope.get("total_cost_usd")
    if isinstance(cost, (int, float)) and cost >= 0:
        return UsageTotals(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=round(float(cost), 4),
            model=model or "claude-code",
            pricing_known=True,
        )
    return UsageTotals.from_counts(model or "claude-sonnet-4-6", input_tokens, output_tokens)


def _parse_response(data: dict, duration: float, usage: UsageTotals | None = None) -> SegmentAnalysis:
    """Convert parsed JSON dict into a SegmentAnalysis (whole-video summary)."""
    seg = parse_segment_data(data, start_time=0.0, end_time=duration, usage=usage)
    seg.is_summary = True
    return seg
