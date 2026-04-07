# WCS Analyzer

AI-powered West Coast Swing dance video analyzer that evaluates performances according to WSDC competition scoring standards.

Drop in a video of your WCS dancing, get back detailed scores, technique breakdowns, partner-specific feedback, and actionable improvement tips — like having a judge review your practice sessions.

## How It Works

### Gemini (default) — Native Video + Audio

```
Video File (.mp4, .mov, etc.)
       |
       v
  Gemini API ────────> Uploads full video with audio
       |                Model SEES motion and HEARS the music
       v
  Scoring Engine ────> WSDC-weighted aggregation
       |
       v
  Report ────────────> Terminal, JSON, or CSV output
```

Gemini processes the entire video natively — it can see continuous movement and hear the music, so timing judgments are based on actual beat alignment, not guesswork from still frames.

### Claude (fallback) — Frame-Based

```
Video File
       |
       v
  Frame Extraction ──> Samples frames at configurable FPS (OpenCV)
  Audio Processing ──> Beat detection, BPM estimation (librosa)
       |
       v
  Claude Vision ─────> Analyzes each segment from still images
       |
       v
  Scoring Engine ────> WSDC-weighted aggregation
       |
       v
  Report
```

Claude analyzes individual frames with beat context from audio analysis. Works well but can't see motion between frames or hear the music directly.

### Why Gemini is the default

For WCS analysis, seeing motion is critical — you can't judge timing, connection, or slot maintenance from snapshots. Gemini is the only major AI API that accepts video files directly with audio, making it far better suited for dance evaluation.

## Scoring Categories

| Category | Weight | What It Evaluates |
|---|---|---|
| **Timing & Rhythm** | 30% | On-beat dancing, syncopation accuracy, anchor steps, rhythm consistency |
| **Technique** | 30% | Posture, extension, footwork, slot maintenance, connection frame |
| **Teamwork** | 20% | Lead/follow connection, shared weight, responsiveness, matched energy |
| **Presentation** | 20% | Musicality, styling, confidence, performance quality |

Scores are 1-10 per category, with letter grades from A+ to F. Lead and follow get individual technique and presentation scores.

## Quick Start

### Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) installed and on your PATH
- A [Google Gemini API key](https://aistudio.google.com/apikey) (default) and/or an [Anthropic API key](https://console.anthropic.com/) (for Claude fallback)

### Install

```bash
pip install -e .
```

### Set your API key

```bash
# For Gemini (default provider)
export GEMINI_API_KEY="your-key-here"

# For Claude (fallback)
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Analyze a video

```bash
wcs-analyzer analyze my_dance.mp4
```

That's it. You'll get a full terminal report with scores, technique breakdown, partner feedback, and improvement suggestions.

## Commands

### `analyze` — Full analysis

```bash
wcs-analyzer analyze video.mp4 [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--provider` | `gemini` | AI provider: `gemini` (native video+audio) or `claude` (frame-based) |
| `--model` | auto | Model ID (defaults to `gemini-2.5-flash` or `claude-sonnet-4-6`) |
| `--detail` | `medium` | Analysis granularity: `low`, `medium`, `high` |
| `--fps` | `3.0` | Frames per second to sample — Claude only (1-30) |
| `--format` | `terminal` | Output format: `terminal`, `json`, `csv` |
| `-o`, `--output` | — | Save JSON report to a file path |
| `--no-cache` | — | Force re-analysis, skip cache |
| `-v`, `--verbose` | — | Enable debug logging |

**Examples:**

```bash
# Default: Gemini with native video analysis
wcs-analyzer analyze competition.mp4

# Use Claude instead
wcs-analyzer analyze competition.mp4 --provider claude

# High detail with Gemini Pro
wcs-analyzer analyze competition.mp4 --model gemini-2.5-pro --detail high

# Export as CSV
wcs-analyzer analyze competition.mp4 --format csv -o scores.csv
```

### `timing` — Quick timing check

A faster, cheaper analysis focused only on beat alignment:

```bash
wcs-analyzer timing video.mp4
wcs-analyzer timing video.mp4 --provider claude
```

### `compare` — Compare multiple analyses

Compare scores across multiple JSON reports to track progress:

```bash
wcs-analyzer compare session1.json session2.json session3.json
```

Shows side-by-side scores with trend indicators.

## Output

### Terminal Report

The default terminal output includes:
- Overall score with letter grade
- Category score table with visual bars
- Technique sub-scores (posture, extension, footwork, slot)
- Partner breakdown (lead vs follow scores)
- Off-beat moment timeline
- Identified patterns (sugar push, whip, etc.)
- Top strengths and areas to improve
- Judge's notes

### JSON Export

```bash
wcs-analyzer analyze video.mp4 --format json -o report.json
```

Structured JSON with all scores, partner breakdown, and detailed feedback — useful for building dashboards or tracking progress programmatically.

### CSV Export

```bash
wcs-analyzer analyze video.mp4 --format csv
```

Spreadsheet-friendly format with summary rows and per-segment scores.

## Caching

Results are automatically cached in `~/.wcs-analyzer/cache/` based on a hash of the video file, provider, and analysis parameters. Re-running the same video with the same settings will use the cached result instantly.

```bash
# Force a fresh analysis
wcs-analyzer analyze video.mp4 --no-cache
```

## Architecture

```
src/wcs_analyzer/
  cli.py              CLI with analyze, timing, compare commands
  gemini_analyzer.py  Gemini native video+audio analysis
  analyzer.py         Claude frame-based analysis with retries
  video.py            OpenCV frame extraction and phrase grouping
  audio.py            ffmpeg audio extraction + librosa beat detection
  prompts.py          WSDC-specific prompts (Gemini video + Claude segment)
  scoring.py          Weighted score aggregation and grade assignment
  report.py           Rich terminal output + JSON/CSV export + comparison
  cache.py            File-hash based LLM response caching
  exceptions.py       Custom exception hierarchy
```

## Development

### Setup

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Or with uv
uv sync
```

### Run checks

```bash
ruff check src/ tests/    # Lint
pyright src/               # Type check
pytest tests/ -v           # Tests (69 tests, no API key needed)
```

### Run all CI checks locally

```bash
ruff check src/ && pyright src/ && pytest tests/ -v
```

## Tips for Best Results

- **Video length**: 30 seconds to 5 minutes works best. Trim to the dance portion.
- **Camera angle**: A wide shot showing both partners' full bodies gives the best analysis.
- **Audio quality**: Clear music helps — Gemini hears it directly, Claude relies on librosa beat detection.
- **Detail level**: Use `--detail low` for quick feedback, `--detail high` for thorough analysis (costs more).
- **Provider**: Use Gemini (default) for best results. Fall back to Claude if you don't have a Gemini API key.

## Tech Stack

- **[Google Gemini](https://ai.google.dev/)** — Native video + audio understanding (default provider)
- **[Anthropic Claude](https://www.anthropic.com/)** — Vision model for frame-based analysis (fallback)
- **[OpenCV](https://opencv.org/)** — Video frame extraction (Claude path)
- **[librosa](https://librosa.org/)** — Audio beat detection (Claude path)
- **[Click](https://click.palletsprojects.com/)** — CLI framework
- **[Rich](https://rich.readthedocs.io/)** — Terminal formatting
