"""WCS-specific prompts for dance video analysis."""

PATTERN_SEGMENTATION_PROMPT = """\
You are segmenting a West Coast Swing dance video into its constituent \
patterns. Look at the provided frames (in temporal order) and identify the \
sequence of patterns the couple performs.

Common WCS patterns:
- sugar push (6-count, in-place)
- left side pass, right side pass (6-count)
- tuck turn (6-count)
- whip (8-count, rotational)
- basket whip, reverse whip
- starter step, anchor-only / in-place variations

Return a JSON timeline. Each entry covers a contiguous time range; the \
ranges must not overlap and should cover the entire video from start to end.

Respond in this exact JSON format. No prose, no markdown:
{{
  "patterns": [
    {{"start_time": 0.0, "end_time": 3.2, "name": "sugar push", "confidence": 0.8}},
    {{"start_time": 3.2, "end_time": 7.0, "name": "left side pass", "confidence": 0.7}}
  ]
}}

Confidence is 0-1; use 1.0 when you're certain, ~0.5 when you can only \
narrow it down to a family, and < 0.3 when the pattern is unclear. The \
dance is {duration:.1f} seconds long. Use {num_frames} sampled frames \
to identify approximately {expected_count} patterns.
"""


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

Use these calibration examples to anchor your scale:

**Novice example (~3):** Social-dance couple. Lead drops the follow's arm \
mid-pattern and loses the slot line; triple steps are flat-footed without \
rolling through the foot. Follow's posture collapses at the anchor. An \
off-beat moment happens roughly every 8-count. Typical scoring: timing 3.5, \
technique 3.0, teamwork 4.0, presentation 3.5.

**Intermediate example (~6):** Novice-division competitor. Consistent basics, \
clean sugar pushes and side passes. Anchor steps mostly settle on beat but \
occasionally rush into the next pattern. Some forward lean on tuck turns. \
Teamwork is clean but reactive rather than conversational; styling is minimal. \
Typical scoring: timing 6.5, technique 6.0, teamwork 6.5, presentation 5.5.

**Champion example (~9):** Champion-tier routine. Musicality drives every \
movement — dancers hit breaks precisely, stretch the anchor into the blues \
pocket, layer body rolls into triples. Frame is immaculate, extension is full, \
the slot is razor-straight. Lead shapes the music through the follow's path. \
Typical scoring: timing 9.0, technique 9.0, teamwork 9.5, presentation 9.5.

Before committing to a score for a category, you MUST write a one-sentence \
`reasoning` field walking through the specific evidence you observed. The \
score should follow directly from that reasoning, not the other way around.

Be specific and constructive. Reference exact moments when possible. \
Note both strengths and areas for improvement.

For every category score you give, also return a `score_low` and `score_high` \
expressing your uncertainty — the range you'd still defend if pressed. A confident \
score has a tight interval (e.g., 7.3-7.7); a shaky or obstructed view has a wide \
one (e.g., 5.5-8.0). Keep `score_low <= score <= score_high`, all within 1-10.

IMPORTANT: If the video contains multiple couples or bystanders, focus \
ONLY on the specified dancers. Ignore all other people in the frame.\
"""

DANCER_CONTEXT_TEMPLATE = """\
DANCER IDENTIFICATION: {dancer_description}
Focus your analysis ONLY on these dancers. There may be other people \
visible in the video (other couples, spectators, judges) — ignore them entirely.
"""

SEGMENT_ANALYSIS_PROMPT = """\
Analyze this segment of a West Coast Swing dance video.

{dancer_context}{beat_context}

The frames below are sequential images from this segment of the dance. \
Examine the dancer(s) carefully for timing, technique, teamwork, and presentation.

Respond in this exact JSON format. Fill `reasoning` BEFORE `score` in each category:
{{
  "timing": {{
    "reasoning": "<one sentence walking through the beat-by-beat evidence before scoring>",
    "score": <1-10>,
    "score_low": <1-10>,
    "score_high": <1-10>,
    "on_beat": <true/false for overall>,
    "off_beat_moments": [
      {{"timestamp_approx": "<time>", "description": "<what happened>", "beat_count": "<e.g., 3&4>"}}
    ],
    "notes": "<detailed timing observations>"
  }},
  "technique": {{
    "reasoning": "<one sentence weighing posture, extension, footwork, slot before scoring>",
    "score": <1-10>,
    "score_low": <1-10>,
    "score_high": <1-10>,
    "posture": {{"score": <1-10>, "notes": "<detail: frame alignment, core engagement, forward lean, head position, shoulder tension>"}},
    "extension": {{"score": <1-10>, "notes": "<detail: arm reach, body stretch through slot, line quality>"}},
    "footwork": {{"score": <1-10>, "notes": "<detail: heel leads, toe leads, rolling through feet, triple step clarity>"}},
    "slot": {{"score": <1-10>, "notes": "<detail: staying in the slot line, drifting, lane discipline>"}},
    "notes": "<overall technique observations>"
  }},
  "teamwork": {{
    "reasoning": "<one sentence on connection, responsiveness, shared weight before scoring>",
    "score": <1-10>,
    "score_low": <1-10>,
    "score_high": <1-10>,
    "connection": "<observations about lead/follow connection>",
    "notes": "<overall teamwork observations>"
  }},
  "presentation": {{
    "reasoning": "<one sentence on musicality, styling, stage presence before scoring>",
    "score": <1-10>,
    "score_low": <1-10>,
    "score_high": <1-10>,
    "musicality": "<observations>",
    "styling": "<observations>",
    "notes": "<overall presentation observations>"
  }},
  "patterns_identified": [
    {{"name": "<e.g., sugar push>", "quality": "<strong|solid|needs_work|weak>", "timing": "<on_beat|slightly_off|off_beat>", "notes": "<what was good or needs work>"}}
  ],
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

GEMINI_VIDEO_PROMPT = """\
Watch and listen to this entire West Coast Swing dance video carefully. \
Pay attention to both the visual movement AND the music/audio to judge timing accuracy.

{dancer_context}\
Analyze the full performance and provide a comprehensive evaluation. \
Since you can hear the music, evaluate whether the dancers are truly on beat — \
listen for anchors landing on the downbeat, triples matching the rhythm, \
and whether styling choices align with musical accents and breaks.

Respond in this exact JSON format. Fill `reasoning` BEFORE `score` in each category:
{
  "timing": {
    "reasoning": "<one sentence walking through what you heard and saw before scoring>",
    "score": <1-10>,
    "score_low": <1-10>,
    "score_high": <1-10>,
    "on_beat": <true/false for overall>,
    "off_beat_moments": [
      {"timestamp_approx": "<time>", "description": "<what happened>", "beat_count": "<e.g., 3&4>"}
    ],
    "rhythm_consistency": "<assessment of timing throughout>",
    "notes": "<detailed timing observations referencing what you heard in the music>"
  },
  "technique": {
    "reasoning": "<one sentence weighing posture, extension, footwork, slot before scoring>",
    "score": <1-10>,
    "score_low": <1-10>,
    "score_high": <1-10>,
    "posture": {"score": <1-10>, "notes": "<detail: frame alignment, core engagement, forward lean, head position, shoulder tension>"},
    "extension": {"score": <1-10>, "notes": "<detail: arm reach, body stretch through slot, line quality>"},
    "footwork": {"score": <1-10>, "notes": "<detail: heel leads, toe leads, rolling through feet, triple step clarity>"},
    "slot": {"score": <1-10>, "notes": "<detail: staying in the slot line, drifting, lane discipline>"},
    "notes": "<overall technique observations>"
  },
  "teamwork": {
    "reasoning": "<one sentence on connection, responsiveness, shared weight before scoring>",
    "score": <1-10>,
    "score_low": <1-10>,
    "score_high": <1-10>,
    "connection": "<observations about lead/follow connection>",
    "notes": "<overall teamwork observations>"
  },
  "presentation": {
    "reasoning": "<one sentence on musicality, styling, stage presence before scoring>",
    "score": <1-10>,
    "score_low": <1-10>,
    "score_high": <1-10>,
    "musicality": "<observations — reference specific musical moments>",
    "styling": "<observations>",
    "notes": "<overall presentation observations>"
  },
  "patterns_identified": [
    {
      "name": "<e.g., sugar push, left side pass, whip>",
      "quality": "<strong|solid|needs_work|weak>",
      "timing": "<on_beat|slightly_off|off_beat>",
      "notes": "<what was good or needs improvement in this pattern>"
    }
  ],
  "highlights": ["<notable positive moments with approximate timestamps>"],
  "improvements": ["<specific actionable suggestions>"],
  "lead": {
    "technique_score": <1-10>,
    "presentation_score": <1-10>,
    "notes": "<lead-specific observations>"
  },
  "follow": {
    "technique_score": <1-10>,
    "presentation_score": <1-10>,
    "notes": "<follow-specific observations>"
  },
  "overall_impression": "<1-2 sentence overall assessment>",
  "estimated_bpm": <estimated BPM from the music>,
  "song_style": "<e.g., blues, contemporary, lyrical>"
}

Only output valid JSON, no other text.\
"""

SUMMARY_PROMPT = """\
You have analyzed {num_segments} segments of a West Coast Swing dance video \
({duration:.0f} seconds total, {bpm:.0f} BPM).

Here are the per-segment analysis results:

{segment_results}

Provide a final summary analysis combining all segments. Fill `reasoning` BEFORE \
`score` in each category — it should synthesize the per-segment results. Respond \
in this exact JSON format:
{{
  "timing": {{
    "reasoning": "<one sentence synthesizing timing consistency across segments>",
    "score": <1-10 overall>,
    "score_low": <1-10>,
    "score_high": <1-10>,
    "total_off_beat_moments": <count>,
    "off_beat_details": [
      {{"time": "<timestamp>", "description": "<what happened>", "beat_count": "<e.g., 3&4>"}}
    ],
    "rhythm_consistency": "<assessment>",
    "notes": "<overall timing summary>"
  }},
  "technique": {{
    "reasoning": "<one sentence synthesizing technique across segments>",
    "score": <1-10 overall>,
    "score_low": <1-10>,
    "score_high": <1-10>,
    "posture_score": <1-10>,
    "extension_score": <1-10>,
    "footwork_score": <1-10>,
    "slot_score": <1-10>,
    "notes": "<overall technique summary>"
  }},
  "teamwork": {{
    "reasoning": "<one sentence synthesizing teamwork across segments>",
    "score": <1-10 overall>,
    "score_low": <1-10>,
    "score_high": <1-10>,
    "connection_quality": "<assessment>",
    "notes": "<overall teamwork summary>"
  }},
  "presentation": {{
    "reasoning": "<one sentence synthesizing presentation across segments>",
    "score": <1-10 overall>,
    "score_low": <1-10>,
    "score_high": <1-10>,
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
