# WCS Analyzer

AI-powered West Coast Swing dance video analyzer that uses Claude's vision capabilities to evaluate performances according to WSDC competition scoring standards.

Drop in a video of your WCS dancing, get back detailed scores, technique breakdowns, partner-specific feedback, and actionable improvement tips — like having a judge review your practice sessions.

## How It Works

```
Video File (.mp4, .mov, etc.)
       |
       v
  Frame Extraction ──> Samples frames at configurable FPS (OpenCV)
  Audio Processing ──> Beat detection, BPM estimation (librosa)
       |
       v
  Frame Grouping ────> Organizes into 8-count musical phrases
       |
       v
  Claude Vision ─────> Analyzes each segment for technique, timing, etc.
       |
       v
  Scoring Engine ────> WSDC-weighted aggregation across 4 categories
       |
       v
  Report ────────────> Terminal, JSON, or CSV output
```

## Scoring Categories

| Category | Weight | What It Evaluates |
|---|---|---|
| **Timing & Rhythm** | 30% | On-beat dancing, syncopation accuracy, anchor steps, rhythm consistency |
| **Technique** | 30% | Posture, extension, footwork, slot maintenance, connection frame |
| **Teamwork** | 20% | Lead/follow connection, shared weight, responsiveness, matched energy |
| **Presentation** | 20% | Musicality, styling, confidence, performance quality |

Scores are 1-10 per category, with letter grades from A+ to F.

## Quick Start

### Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) installed and on your PATH
- An [Anthropic API key](https://console.anthropic.com/)

### Install

```bash
pip install -e .
```

### Set your API key

```bash
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
| `--model` | `claude-sonnet-4-6` | Claude model to use |
| `--detail` | `medium` | Analysis granularity: `low`, `medium`, `high` |
| `--fps` | `3.0` | Frames per second to sample (1-30) |
| `--format` | `terminal` | Output format: `terminal`, `json`, `csv` |
| `-o`, `--output` | — | Save JSON report to a file path |
| `--no-cache` | — | Force re-analysis, skip cache |
| `-v`, `--verbose` | — | Enable debug logging |

### `timing` — Quick timing check

A faster, cheaper analysis focused only on beat alignment:

```bash
wcs-analyzer timing video.mp4
```

### `compare` — Compare multiple analyses

Compare scores across multiple JSON reports to track progress:

```bash
wcs-analyzer compare session1.json session2.json session3.json
```

Shows side-by-side scores with trend indicators.

## Output Examples

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

Structured JSON with all scores, per-segment data, partner breakdown, and detailed feedback — useful for building dashboards or tracking progress programmatically.

### CSV Export

```bash
wcs-analyzer analyze video.mp4 --format csv
```

Spreadsheet-friendly format with summary rows and per-segment scores.

## Caching

Results are automatically cached in `~/.wcs-analyzer/cache/` based on a hash of the video file and analysis parameters. Re-running the same video with the same settings will use the cached result instantly.

```bash
# Force a fresh analysis
wcs-analyzer analyze video.mp4 --no-cache
```

## Architecture

```
src/wcs_analyzer/
  cli.py          Click-based CLI with analyze, timing, compare commands
  video.py        OpenCV frame extraction and phrase grouping
  audio.py        ffmpeg audio extraction + librosa beat detection
  analyzer.py     Claude API orchestration with retries and token awareness
  prompts.py      WSDC-specific system and analysis prompts
  scoring.py      Weighted score aggregation and grade assignment
  report.py       Rich terminal output + JSON/CSV export
  cache.py        File-hash based LLM response caching
  exceptions.py   Custom exception hierarchy
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
pytest tests/ -v           # Tests (61 tests, no API key needed)
```

### Run all CI checks locally

```bash
ruff check src/ && pyright src/ && pytest tests/ -v
```

## Tips for Best Results

- **Video length**: 30 seconds to 5 minutes works best. Trim to the dance portion.
- **Camera angle**: A wide shot showing both partners' full bodies gives the best analysis.
- **Audio quality**: Clear music helps with beat detection. Noisy audio may reduce timing accuracy.
- **Detail level**: Use `--detail low` for quick feedback, `--detail high` for frame-by-frame analysis (costs more).
- **FPS**: Default 3 fps is a good balance. Increase for faster songs, decrease for longer videos.

## Tech Stack

- **[Anthropic Claude](https://www.anthropic.com/)** — Vision + language model for dance analysis
- **[OpenCV](https://opencv.org/)** — Video frame extraction
- **[librosa](https://librosa.org/)** — Audio beat detection and tempo estimation
- **[Click](https://click.palletsprojects.com/)** — CLI framework
- **[Rich](https://rich.readthedocs.io/)** — Terminal formatting
