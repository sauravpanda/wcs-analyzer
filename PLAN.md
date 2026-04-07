# WCS Dance Analyzer — Implementation Plan

## Context

Build a CLI tool that takes a West Coast Swing dance video, analyzes it using a vision LLM (Claude), and produces WSDC-style competition scoring with detailed feedback on timing, technique, teamwork, and presentation.

## Architecture Overview

```
Video File → Frame Extraction + Audio Extraction
                ↓                    ↓
         Key Frame Sampling    Beat Detection
                ↓                    ↓
         Vision LLM Analysis ← Beat/Timing Data
                ↓
         Scoring Engine
                ↓
         CLI Report Output
```

## WSDC Scoring Categories (v1)

| Category | Weight | What's Evaluated |
|----------|--------|------------------|
| **Timing & Rhythm** | 30% | On-beat dancing, syncopation accuracy, musical breaks |
| **Technique** | 30% | Posture, extension, footwork, anchor steps, slot maintenance |
| **Teamwork** | 20% | Connection, lead/follow clarity, shared weight, responsiveness |
| **Presentation** | 20% | Musicality, styling, confidence, performance quality |

Each category scored 1-10, weighted to produce an overall score.

## Project Structure

```
wcs-analyzer/
├── pyproject.toml              # Project config, dependencies, CLI entry point
├── README.md
├── src/
│   └── wcs_analyzer/
│       ├── __init__.py
│       ├── cli.py              # CLI entry point (click/typer)
│       ├── video.py            # Video frame extraction (OpenCV)
│       ├── audio.py            # Audio extraction + beat detection (librosa)
│       ├── analyzer.py         # Vision LLM analysis orchestration
│       ├── prompts.py          # WCS-specific prompts for Claude
│       ├── scoring.py          # Scoring engine & aggregation
│       └── report.py           # CLI report formatting (rich)
```

## Implementation Steps

### Step 1: Project Setup
- `pyproject.toml` with dependencies: `anthropic`, `opencv-python`, `librosa`, `click`, `rich`, `ffmpeg-python`
- Basic CLI skeleton with `click`

### Step 2: Video Processing (`video.py`)
- Extract frames from video using OpenCV
- Smart frame sampling: ~2-4 frames per second (WCS is ~28-34 BPM, need enough to catch each beat)
- Resize frames for LLM input (keep under token limits)
- Group frames into segments (e.g., 8-count phrases)

### Step 3: Audio Processing (`audio.py`)
- Extract audio track from video using ffmpeg
- Beat detection using librosa (`librosa.beat.beat_track`)
- Tempo estimation (BPM)
- Generate beat timestamps to correlate with frames

### Step 4: WCS Analysis Prompts (`prompts.py`)
- System prompt establishing Claude as a WCS judge with WSDC criteria
- Per-segment analysis prompt: send ~8-16 frames (one 8-count phrase) with beat timestamps
- Category-specific evaluation criteria embedded in prompts
- Structured output format (JSON) for consistent scoring

### Step 5: LLM Orchestration (`analyzer.py`)
- Send frame batches to Claude API (claude-sonnet-4-6 for cost efficiency, opus for quality)
- Include beat timing context with each batch
- Parse structured JSON responses
- Handle rate limiting and retries
- Aggregate segment-level analysis into full-dance analysis

### Step 6: Scoring Engine (`scoring.py`)
- Aggregate per-segment scores into category scores
- Apply WSDC weighting (30/30/20/20)
- Calculate overall score
- Identify top strengths and areas for improvement
- Track timing deviations (off-beat moments count)

### Step 7: Report Output (`report.py`)
- Rich CLI output with colored tables
- Overall score with letter grade
- Per-category breakdown with scores and feedback
- Timeline of notable moments (off-beat, great extension, posture breaks)
- Actionable improvement suggestions

## Example CLI Usage

```bash
# Basic analysis
wcs-analyzer analyze dance_video.mp4

# With options
wcs-analyzer analyze dance_video.mp4 --model claude-sonnet-4-6 --detail high --output report.json

# Quick timing-only check
wcs-analyzer timing dance_video.mp4
```

## Example Output

```
╔══════════════════════════════════════════╗
║     WCS Dance Analysis Report            ║
╠══════════════════════════════════════════╣
║  Overall Score: 7.2 / 10  (B+)           ║
╠══════════════════════════════════════════╣
║  Timing & Rhythm:   7.5 / 10  ████████░░ ║
║  Technique:         7.0 / 10  ███████░░░ ║
║  Teamwork:          7.5 / 10  ████████░░ ║
║  Presentation:      6.8 / 10  ███████░░░ ║
╠══════════════════════════════════════════╣
║  Off-beat moments: 4 detected            ║
║  - 0:12 (count 3&4, rushed)              ║
║  - 0:28 (anchor, cut short)              ║
║  - 0:45 (syncopation, late)              ║
║  - 1:02 (whip, early on 5&6)            ║
╠══════════════════════════════════════════╣
║  Strengths:                              ║
║  • Good slot maintenance                 ║
║  • Clean sugar push technique            ║
║  • Musical interpretation on breaks      ║
║                                          ║
║  Areas to Improve:                       ║
║  • Anchor steps getting rushed           ║
║  • Left arm extension incomplete         ║
║  • Posture drops during turns            ║
╚══════════════════════════════════════════╝
```

## Dependencies

- `anthropic` — Claude API for vision analysis
- `opencv-python` — Video frame extraction
- `librosa` — Audio beat detection and tempo analysis
- `ffmpeg-python` — Audio extraction from video
- `click` — CLI framework
- `rich` — Beautiful terminal output
- `pydantic` — Structured data models for scores/responses

## Verification

1. Run with a sample WCS video: `wcs-analyzer analyze sample.mp4`
2. Verify frames are extracted and sampled correctly
3. Verify beat detection produces reasonable BPM (28-34 for WCS music)
4. Verify Claude returns structured JSON scores
5. Verify report renders correctly in terminal
6. Test with different video lengths/qualities
