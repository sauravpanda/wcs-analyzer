# WCS Analyzer

## Quick Analysis

When the user asks to "analyze" a dance video, follow these steps:

1. Find video files in the current directory (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`)
2. Extract ~12 frames using ffmpeg: `mkdir -p /tmp/wcs_frames && ffmpeg -i "<VIDEO>" -vf "fps=1/$(ffprobe -v error -show_entries format=duration -of csv=p=0 "<VIDEO>" | awk '{printf "%.0f", $1/12}')" -q:v 2 -frames:v 12 /tmp/wcs_frames/frame_%03d.jpg -y`
3. Read all frame images with the Read tool
4. Score the dance on WSDC categories: Timing (30%), Technique (30%), Teamwork (20%), Presentation (20%)
5. Give separate lead/follow scores for technique and presentation
6. Report: overall score, letter grade, category breakdown, technique sub-scores (posture/extension/footwork/slot), patterns identified, top 3 strengths, top 3 improvements
7. Clean up: `rm -rf /tmp/wcs_frames`

Scoring: 1-3 novice, 4-5 intermediate, 6-7 advanced, 8-9 champion, 10 exceptional.
Overall = Timing×0.30 + Technique×0.30 + Teamwork×0.20 + Presentation×0.20

## Development

- Run checks: `uv run ruff check src/ tests/ && uv run pyright src/ && uv run pytest tests/ -v`
- 77 tests, all mocked (no API key needed)
- Three providers: gemini (default, native video), claude-code (local CLI), claude (API)
