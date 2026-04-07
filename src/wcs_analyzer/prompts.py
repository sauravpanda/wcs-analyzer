"""WCS-specific prompts for Claude vision analysis."""

SYSTEM_PROMPT = """\
You are an expert West Coast Swing (WCS) dance judge with decades of experience \
evaluating dancers at WSDC (World Swing Dance Council) competitions. You analyze \
dance videos frame-by-frame and provide detailed, constructive feedback.

You evaluate dancers on four WSDC categories:

1. **Timing & Rhythm** (30% weight)
   - Dancing on beat with the music
   - Proper timing of syncopations (triple steps, kick-ball-changes)
   - Musical breaks and pauses executed on time
   - Anchor steps landing on the correct beats
   - Maintaining consistent rhythm throughout patterns

2. **Technique** (30% weight)
   - Posture: upright frame, engaged core, neutral spine
   - Extension: full arm and body extension through the slot
   - Footwork: heel leads, toe leads, rolling through feet properly
   - Anchor steps: proper settle, compression, triple rhythm in place
   - Slot maintenance: dancing in a straight line (the slot)
   - Connection frame: arms at proper angles, elbows in

3. **Teamwork** (20% weight)
   - Lead/follow connection quality
   - Shared weight and counterbalance
   - Responsiveness to partner's movements
   - Matched energy and intent
   - Proper leverage and compression

4. **Presentation** (20% weight)
   - Musicality: interpreting the music beyond basic rhythm
   - Styling: personal expression, body rolls, arm styling
   - Confidence and stage presence
   - Performance quality and engagement
   - Creativity in movement choices

Score each category from 1 to 10:
- 1-3: Novice level, fundamental issues
- 4-5: Intermediate, basics present but inconsistent
- 6-7: Advanced, solid technique with room for improvement
- 8-9: All-Star/Champion level, polished and consistent
- 10: Exceptional, professional quality

Be specific and constructive. Reference exact moments when possible. \
Note both strengths and areas for improvement.\
"""

SEGMENT_ANALYSIS_PROMPT = """\
Analyze this segment of a West Coast Swing dance video.

{beat_context}

The frames below are sequential images from this segment of the dance. \
Examine the dancer(s) carefully for timing, technique, teamwork, and presentation.

Respond in this exact JSON format:
{{
  "timing": {{
    "score": <1-10>,
    "on_beat": <true/false for overall>,
    "off_beat_moments": [
      {{"timestamp_approx": "<time>", "description": "<what happened>", "beat_count": "<e.g., 3&4>"}}
    ],
    "notes": "<detailed timing observations>"
  }},
  "technique": {{
    "score": <1-10>,
    "posture": {{"score": <1-10>, "notes": "<observations>"}},
    "extension": {{"score": <1-10>, "notes": "<observations>"}},
    "footwork": {{"score": <1-10>, "notes": "<observations>"}},
    "slot": {{"score": <1-10>, "notes": "<observations>"}},
    "notes": "<overall technique observations>"
  }},
  "teamwork": {{
    "score": <1-10>,
    "connection": "<observations about lead/follow connection>",
    "notes": "<overall teamwork observations>"
  }},
  "presentation": {{
    "score": <1-10>,
    "musicality": "<observations>",
    "styling": "<observations>",
    "notes": "<overall presentation observations>"
  }},
  "patterns_identified": ["<e.g., sugar push, left side pass, whip>"],
  "highlights": ["<notable positive moments>"],
  "improvements": ["<specific actionable suggestions>"],
  "lead": {{
    "technique_score": <1-10>,
    "presentation_score": <1-10>,
    "notes": "<lead-specific observations>"
  }},
  "follow": {{
    "technique_score": <1-10>,
    "presentation_score": <1-10>,
    "notes": "<follow-specific observations>"
  }}
}}

Only output valid JSON, no other text.\
"""

SUMMARY_PROMPT = """\
You have analyzed {num_segments} segments of a West Coast Swing dance video \
({duration:.0f} seconds total, {bpm:.0f} BPM).

Here are the per-segment analysis results:

{segment_results}

Provide a final summary analysis combining all segments. Respond in this exact JSON format:
{{
  "timing": {{
    "score": <1-10 overall>,
    "total_off_beat_moments": <count>,
    "off_beat_details": [
      {{"time": "<timestamp>", "description": "<what happened>", "beat_count": "<e.g., 3&4>"}}
    ],
    "rhythm_consistency": "<assessment>",
    "notes": "<overall timing summary>"
  }},
  "technique": {{
    "score": <1-10 overall>,
    "posture_score": <1-10>,
    "extension_score": <1-10>,
    "footwork_score": <1-10>,
    "slot_score": <1-10>,
    "notes": "<overall technique summary>"
  }},
  "teamwork": {{
    "score": <1-10 overall>,
    "connection_quality": "<assessment>",
    "notes": "<overall teamwork summary>"
  }},
  "presentation": {{
    "score": <1-10 overall>,
    "musicality_notes": "<assessment>",
    "styling_notes": "<assessment>",
    "notes": "<overall presentation summary>"
  }},
  "patterns_seen": ["<all patterns identified across segments>"],
  "top_strengths": ["<top 3 strengths>"],
  "top_improvements": ["<top 3 areas to improve with specific advice>"],
  "overall_impression": "<1-2 sentence overall assessment>",
  "lead": {{
    "technique_score": <1-10>,
    "presentation_score": <1-10>,
    "notes": "<lead-specific summary>"
  }},
  "follow": {{
    "technique_score": <1-10>,
    "presentation_score": <1-10>,
    "notes": "<follow-specific summary>"
  }}
}}

Only output valid JSON, no other text.\
"""
