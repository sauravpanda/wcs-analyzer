# WCS Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/sauravpanda/wcs-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravpanda/wcs-analyzer/actions)

AI-powered West Coast Swing dance video analyzer that evaluates performances according to WSDC competition scoring standards.

Drop in a video of your WCS dancing, get back detailed scores, technique breakdowns, partner-specific feedback, and actionable improvement tips — like having a judge review your practice sessions.

**Beyond LLM vibes.** Pose estimation (MediaPipe) can measure posture, extension, footwork, and slot linearity directly from the video, and a beat-sync verifier cross-correlates detected footfalls with librosa-detected beats to produce an *objective* timing score. Multi-provider ensemble mode runs several models on the same clip and flags categories where they disagreed. Longitudinal tracking answers "am I getting better?" across runs.

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

Every score now comes with a **confidence interval** (`score_low` / `score_high`). Wide intervals mean the model was uncertain — a shaky camera angle, ambiguous technique, or a cropped frame. The report shows the interval next to the score and raises a ⚠ *low confidence* warning when any category's range exceeds 2.0 points. Scores are also **clamped to [1, 10]** on parse, and the Claude path automatically **retries once with a corrective hint** if the response can't be parsed as JSON — so a malformed response never silently collapses to 5.0s.

The segment prompts require the model to fill a **`reasoning` field before each score**, a lightweight chain-of-thought pattern that improves rubric-grading consistency. Three few-shot exemplars (novice, intermediate, champion) are embedded in the system prompt to anchor the 1-10 scale.

## Quick Start

### Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) installed and on your PATH
- A [Google Gemini API key](https://aistudio.google.com/apikey) (default) and/or an [Anthropic API key](https://console.anthropic.com/) (for Claude fallback)

### Install

```bash
pip install -e .

# Optional: MediaPipe for pose-driven metrics (see --pose below)
pip install -e ".[pose]"
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
| `--provider` | `gemini` | AI provider: `gemini` (native video+audio), `claude` (API, frame-based), or `claude-code` (local CLI) |
| `--providers` | — | Comma-separated list for **ensemble mode** (e.g. `gemini,claude-code`). Each runs independently; results are aggregated with disagreement flagging |
| `--model` | auto | Model ID (defaults to `gemini-2.5-flash` or `claude-sonnet-4-6`) |
| `--detail` | `medium` | Analysis granularity: `low`, `medium`, `high` |
| `--fps` | `3.0` | Frames per second to sample — Claude only (1-30) |
| `--dancers` | — | Describe which dancers to focus on (for crowded floors) |
| `--pose` | — | Run MediaPipe pose estimation + beat-sync and feed the measured metrics to the LLM (Gemini only; needs `[pose]` extra) |
| `--save-history` | — | Persist this run's scores under the given dancer name for longitudinal tracking |
| `--format` | `terminal` | Output format: `terminal`, `json`, `csv` |
| `-o`, `--output` | — | Save JSON report to a file path |
| `-j`, `--parallel` | `1` | Number of videos to analyze in parallel (batch mode) |
| `--no-cache` | — | Force re-analysis, skip cache |
| `-v`, `--verbose` | — | Enable debug logging |

**Examples:**

```bash
# Default: Gemini with native video analysis
wcs-analyzer analyze competition.mp4

# Focus on a specific couple on a crowded floor
wcs-analyzer analyze competition.mp4 --dancers "lead in blue shirt, follow in red dress"

# Use Claude instead
wcs-analyzer analyze competition.mp4 --provider claude

# High detail with Gemini Pro
wcs-analyzer analyze competition.mp4 --model gemini-2.5-pro --detail high

# Export as CSV
wcs-analyzer analyze competition.mp4 --format csv -o scores.csv

# Pose-driven objective metrics (MediaPipe) + beat-sync verification
wcs-analyzer analyze competition.mp4 --pose

# Ensemble: run two providers and flag categories where they disagreed
wcs-analyzer analyze competition.mp4 --providers gemini,claude-code

# Save this run to the longitudinal history under "alice"
wcs-analyzer analyze competition.mp4 --save-history alice
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

### `patterns` — WCS pattern timeline

Detect and list the patterns (sugar push, whip, side pass, tuck turn, …) on the video's timeline in a single cheap LLM call. Useful as a sanity check before a full analysis.

```bash
wcs-analyzer patterns video.mp4
```

Renders a colored table of pattern segments with start/end times, duration, and per-pattern confidence (green/yellow/red).

### `progress` — Longitudinal dancer tracking

Reads the history DB populated by `analyze --save-history <name>` and renders a run timeline plus linear-fit trajectories per category:

```bash
wcs-analyzer analyze jan.mp4 --save-history alice
wcs-analyzer analyze feb.mp4 --save-history alice
wcs-analyzer analyze mar.mp4 --save-history alice

wcs-analyzer progress alice
```

Each trajectory is labeled **improving**, **flat**, or **regressing** based on the per-run slope. History lives at `~/.wcs-analyzer/history.db` (override with `WCS_ANALYZER_HISTORY_DB`).

### `dancers` — List tracked dancers

```bash
wcs-analyzer dancers
```

Shows every dancer in the longitudinal history store with their run count.

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

## Advanced Features

### Pose-driven objective metrics (`--pose`)

Technique sub-scores (posture, extension, footwork, slot) are normally inferred by the LLM from still frames. With `--pose` and the `[pose]` extra installed, the tool runs MediaPipe BlazePose over the video to compute:

- **Posture** — spine deviation from vertical, mean and stddev across frames
- **Extension** — arm-to-torso length ratio (mean and peak)
- **Footwork** — footfall rate from ankle-y zero-crossings
- **Slot** — hip-trajectory linearity as R², via PCA on the 2×2 covariance matrix (works for any line direction)

The measured values are formatted as a context block and prepended to the Gemini prompt, with an explicit instruction to the model to treat them as objective measurements. **Known limitation:** BlazePose is single-person, so the metrics reflect whichever dancer the detector locked onto — typically the lead. Multi-person tracking is a future improvement.

### Beat-sync verification (pairs with `--pose`)

When `--pose` is enabled, the tool also cross-correlates pose-detected footfalls with librosa-detected audio beats and computes an **objective timing score** in `[0, 10]`:

- 10 = every footfall lands exactly on a beat
- 0 = mean offset is half a beat interval (on the off-beat)

The result is surfaced in the prompt with the instruction: *keep `timing.score` within ±1.0 of the measured value unless there's a strong musical-interpretation reason, and explain any deviation in `timing.reasoning`*. This is the "measured timing, LLM-judged artistry" pattern.

### Ensemble mode (`--providers`)

Run multiple providers on the same clip and aggregate:

```bash
wcs-analyzer analyze video.mp4 --providers gemini,claude-code
```

- Consensus is computed as the **median** across providers per category (more robust to a single wild outlier than the mean)
- Per-category stddev is reported; any category where stddev > 1.0 is flagged as ⚠ **contested** — a signal that the models disagreed and a human should review
- Graceful degradation: if a provider errors out, the ensemble continues with the rest (requires at least two successful runs)

## Caching

Results are automatically cached in `~/.wcs-analyzer/cache/` based on a hash of the video file, provider, and analysis parameters. Re-running the same video with the same settings will use the cached result instantly.

```bash
# Force a fresh analysis
wcs-analyzer analyze video.mp4 --no-cache
```

## Architecture

```
src/wcs_analyzer/
  cli.py                  CLI — analyze, timing, compare, patterns,
                          progress, dancers commands
  gemini_analyzer.py      Gemini native video+audio analysis
  analyzer.py             Claude frame-based analysis, JSON parse + retry,
                          score clamping, shared parse_segment_data helper,
                          pattern-timeline detection
  claude_code_analyzer.py Local Claude Code CLI provider
  video.py                OpenCV frame extraction and phrase grouping
  audio.py                ffmpeg audio extraction + librosa beat detection
  pose.py                 MediaPipe BlazePose extraction + posture/
                          extension/footwork/slot metrics + beat-sync
                          verification (optional [pose] extra)
  history.py              SQLite longitudinal tracking + linear-fit
                          trajectories per category
  prompts.py              WSDC prompts with few-shot exemplars,
                          chain-of-thought reasoning fields, and
                          confidence-interval schema
  scoring.py              Weighted score aggregation, confidence-interval
                          aggregation, ensemble median + disagreement
                          flagging, grade assignment
  report.py               Rich terminal output + JSON/CSV export, ensemble
                          report, progress report
  cache.py                File-hash based LLM response caching
  exceptions.py           Custom exception hierarchy
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
pytest tests/ -v           # Tests (156+ tests, all mocked, no API key needed)
```

### Run all CI checks locally

```bash
ruff check src/ && pyright src/ && pytest tests/ -v
```

## Tips for Best Results

- **Video length**: 30 seconds to 5 minutes works best. Trim to the dance portion.
- **Camera angle**: A wide shot showing both partners' full bodies gives the best analysis. Pose estimation (`--pose`) needs a reasonably unobstructed view.
- **Audio quality**: Clear music helps — Gemini hears it directly, Claude relies on librosa beat detection.
- **Detail level**: Use `--detail low` for quick feedback, `--detail high` for thorough analysis (costs more).
- **Provider**: Use Gemini (default) for best results. Fall back to Claude if you don't have a Gemini API key.
- **Objective technique scoring**: Add `--pose` when technique really matters — spine angles, extension ratios, footfall rate, and slot linearity become measured instead of inferred.
- **Disagreement check**: Use `--providers gemini,claude-code` on contested or edge-case routines. Categories with stddev > 1.0 are flagged for human review.
- **Progress tracking**: Always pass `--save-history <your-name>` so `wcs-analyzer progress <your-name>` can render your trajectory over time.

## Tech Stack

- **[Google Gemini](https://ai.google.dev/)** — Native video + audio understanding (default provider)
- **[Anthropic Claude](https://www.anthropic.com/)** — Vision model for frame-based analysis
- **[MediaPipe](https://developers.google.com/mediapipe)** — BlazePose landmark detection for objective technique metrics (optional)
- **[OpenCV](https://opencv.org/)** — Video frame extraction
- **[librosa](https://librosa.org/)** — Audio beat detection and BPM estimation
- **[SQLite](https://www.sqlite.org/)** — Longitudinal history store (stdlib)
- **[Click](https://click.palletsprojects.com/)** — CLI framework
- **[Rich](https://rich.readthedocs.io/)** — Terminal formatting
