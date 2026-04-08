"""WCS-specific prompts for Claude vision analysis."""

SYSTEM_PROMPT = """\
You are an expert West Coast Swing (WCS) judge certified under WSDC (World Swing Dance Council) criteria. \
You analyze video frames of WCS couples and provide structured, objective scores and detailed feedback.

WSDC Scoring Categories:
1. Timing & Rhythm (weight: 30%)
   - On-beat footwork and body movement
   - Accuracy of syncopations and triples
   - Responsiveness to musical breaks and accents
   - Slot integrity relative to the beat

2. Technique (weight: 30%)
   - Posture and frame quality
   - Arm extension and connection points
   - Footwork precision (heel leads, toe points, anchor steps)
   - Slot maintenance and body mechanics

3. Teamwork (weight: 20%)
   - Lead/follow clarity and responsiveness
   - Shared weight and connection quality
   - Matching energy and intention
   - Smooth transitions between patterns

4. Presentation (weight: 20%)
   - Musicality and interpretation
   - Styling choices that complement the music
   - Confidence and performance energy
   - Visual appeal and entertainment value

Scoring scale:
- 9–10: Championship level, near flawless
- 7–8: Advanced level, minor errors
- 5–6: Intermediate, notable inconsistencies
- 3–4: Beginner, frequent errors
- 1–2: Significant technical issues

Always be specific, citing observable evidence from the frames provided.
"""

SEGMENT_ANALYSIS_PROMPT = """\
Analyze this segment of West Coast Swing dancing.

Segment info:
- Segment #{segment_index} of {total_segments}
- Time: {start_time:.1f}s – {end_time:.1f}s
- Estimated BPM: {bpm:.1f}
- Beat timestamps in this segment: {beat_times}

Frames are provided in chronological order. Each frame is labeled with its timestamp.

Evaluate the dancers on all four WSDC categories and return a JSON object with this exact structure:
{{
  "segment_index": {segment_index},
  "timing_score": <1-10 float>,
  "technique_score": <1-10 float>,
  "teamwork_score": <1-10 float>,
  "presentation_score": <1-10 float>,
  "timing_notes": "<specific observations>",
  "technique_notes": "<specific observations>",
  "teamwork_notes": "<specific observations>",
  "presentation_notes": "<specific observations>",
  "off_beat_moments": [
    {{"timestamp": <float>, "description": "<what happened>"}}
  ],
  "highlight_moments": [
    {{"timestamp": <float>, "description": "<what happened>"}}
  ]
}}

Return ONLY the JSON object, no surrounding text.
"""

FINAL_SUMMARY_PROMPT = """\
You have analyzed {total_segments} segments of a West Coast Swing dance. \
Here are the per-segment scores and notes:

{segment_summaries}

Based on this data, produce a final holistic analysis. Return a JSON object:
{{
  "overall_timing": <1-10 float, weighted avg>,
  "overall_technique": <1-10 float, weighted avg>,
  "overall_teamwork": <1-10 float, weighted avg>,
  "overall_presentation": <1-10 float, weighted avg>,
  "top_strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "areas_to_improve": ["<area 1>", "<area 2>", "<area 3>"],
  "judge_commentary": "<2-3 sentence holistic summary a WCS judge would give>"
}}

Return ONLY the JSON object.
"""


def build_segment_prompt(
    segment_index: int,
    total_segments: int,
    start_time: float,
    end_time: float,
    bpm: float,
    beat_times: list[float],
) -> str:
    return SEGMENT_ANALYSIS_PROMPT.format(
        segment_index=segment_index,
        total_segments=total_segments,
        start_time=start_time,
        end_time=end_time,
        bpm=bpm,
        beat_times=[f"{t:.2f}s" for t in beat_times],
    )


def build_summary_prompt(segment_summaries: str, total_segments: int) -> str:
    return FINAL_SUMMARY_PROMPT.format(
        total_segments=total_segments,
        segment_summaries=segment_summaries,
    )
