# Contributing to WCS Analyzer

Thanks for your interest in contributing! Whether you're a developer, a WCS dancer with feedback, or both — all contributions are welcome.

## Getting Started

### Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Setup

```bash
git clone https://github.com/sauravpanda/wcs-analyzer.git
cd wcs-analyzer
uv sync        # installs all dependencies including dev tools
```

### Running Checks

```bash
uv run ruff check src/ tests/    # Lint
uv run pyright src/               # Type check
uv run pytest tests/ -v           # Tests (no API key needed)
```

All three must pass before submitting a PR.

## How to Contribute

### Bug Reports

Open an issue with:
- What you expected vs what happened
- Video details if relevant (length, format, number of people in frame)
- Full error output (with `--verbose` flag)

### Feature Requests

Open an issue describing:
- What you want and why
- How you'd use it in your WCS practice/competition workflow

### Pull Requests

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Add tests for new functionality
4. Run all checks (`ruff`, `pyright`, `pytest`)
5. Open a PR with a clear description of what and why

### Code Style

- We use `ruff` for linting and `pyright` for type checking
- Keep functions focused and well-typed
- Add tests for new features and bug fixes
- Don't add unnecessary dependencies

### WCS Domain Knowledge

If you're a WCS dancer or judge, we'd especially value feedback on:
- Scoring rubric accuracy vs real WSDC judging
- Prompt improvements for better analysis quality
- Missing patterns or technique aspects
- How the tool could better support your practice workflow

## Development Tips

- Use `--no-cache` when testing prompt changes
- Use `--verbose` to see what's happening under the hood
- Mock tests don't need API keys — all 72 tests run locally
- The Gemini path (`gemini_analyzer.py`) and Claude path (`analyzer.py`) are independent

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
