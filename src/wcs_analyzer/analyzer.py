"""LLM orchestration: send frame batches to Claude, parse responses."""

from __future__ import annotations

import json
import time
from pathlib import Path

import anthropic

from .audio import AudioAnalysis
from .prompts import SYSTEM_PROMPT, build_segment_prompt, build_summary_prompt
from .scoring import SegmentScore
from .video import Segment


def _parse_json(text: str) -> dict:
    """Extract JSON from model response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        inner = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            if line.startswith("```") and in_block:
                break
            if in_block:
                inner.append(line)
        text = "\n".join(inner)
    return json.loads(text)


def analyze_segment(
    client: anthropic.Anthropic,
    segment: Segment,
    total_segments: int,
    audio: AudioAnalysis,
    model: str,
    max_frames_per_segment: int = 12,
) -> SegmentScore:
    """Send one segment's frames to Claude and return a SegmentScore."""
    beat_times = audio.beats_in_range(segment.start_time, segment.end_time)

    prompt = build_segment_prompt(
        segment_index=segment.segment_index,
        total_segments=total_segments,
        start_time=segment.start_time,
        end_time=segment.end_time,
        bpm=audio.bpm,
        beat_times=beat_times,
    )

    # Sample frames evenly if too many
    frames = segment.frames
    if len(frames) > max_frames_per_segment:
        step = len(frames) / max_frames_per_segment
        frames = [frames[int(i * step)] for i in range(max_frames_per_segment)]

    content: list[dict] = []
    for frame in frames:
        content.append({
            "type": "text",
            "text": f"Frame at {frame.timestamp:.2f}s (beat #{frame.beat_number}):",
        })
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": frame.as_base64(),
            },
        })
    content.append({"type": "text", "text": prompt})

    for attempt in range(3):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            data = _parse_json(response.content[0].text)
            return SegmentScore(
                segment_index=segment.segment_index,
                start_time=segment.start_time,
                end_time=segment.end_time,
                timing=float(data.get("timing_score", 5.0)),
                technique=float(data.get("technique_score", 5.0)),
                teamwork=float(data.get("teamwork_score", 5.0)),
                presentation=float(data.get("presentation_score", 5.0)),
                timing_notes=data.get("timing_notes", ""),
                technique_notes=data.get("technique_notes", ""),
                teamwork_notes=data.get("teamwork_notes", ""),
                presentation_notes=data.get("presentation_notes", ""),
                off_beat_moments=data.get("off_beat_moments", []),
                highlight_moments=data.get("highlight_moments", []),
            )
        except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Failed after 3 attempts")


def get_final_summary(
    client: anthropic.Anthropic,
    segment_scores: list[SegmentScore],
    model: str,
) -> dict:
    """Ask Claude for a holistic final summary given all segment results."""
    seg_lines = []
    for s in segment_scores:
        seg_lines.append(
            f"Segment {s.segment_index} ({s.start_time:.1f}s-{s.end_time:.1f}s): "
            f"timing={s.timing:.1f} technique={s.technique:.1f} "
            f"teamwork={s.teamwork:.1f} presentation={s.presentation:.1f}\n"
            f"  timing: {s.timing_notes}\n"
            f"  technique: {s.technique_notes}\n"
            f"  teamwork: {s.teamwork_notes}\n"
            f"  presentation: {s.presentation_notes}"
        )

    prompt = build_summary_prompt(
        segment_summaries="\n\n".join(seg_lines),
        total_segments=len(segment_scores),
    )

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_json(response.content[0].text)


def run_analysis(
    video_path: Path,
    audio: AudioAnalysis,
    segments: list[Segment],
    model: str,
    api_key: str | None,
    progress_callback=None,
) -> tuple[list[SegmentScore], dict]:
    """Full pipeline: analyze all segments, then get summary. Returns (scores, summary)."""
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    segment_scores: list[SegmentScore] = []
    total = len(segments)

    for i, segment in enumerate(segments):
        if progress_callback:
            progress_callback(i, total, f"Analyzing segment {i + 1}/{total}…")
        score = analyze_segment(client, segment, total, audio, model)
        segment_scores.append(score)

    if progress_callback:
        progress_callback(total, total, "Generating final summary…")

    summary = get_final_summary(client, segment_scores, model)
    return segment_scores, summary
