"""CLI entry point for WCS Analyzer."""

import logging
import sys

import click
from pathlib import Path

from rich.console import Console

from .exceptions import WCSAnalyzerError

console = Console()

# Duration limits (seconds)
MIN_VIDEO_DURATION = 10
MAX_VIDEO_DURATION = 600


def _validate_fps(ctx: click.Context, param: click.Parameter, value: float) -> float:
    if value <= 0 or value > 30:
        raise click.BadParameter("FPS must be between 0 and 30.")
    return value


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(name)s — %(levelname)s — %(message)s",
        stream=sys.stderr,
    )


@click.group()
@click.version_option(package_name="wcs-analyzer")
def main():
    """WCS Analyzer — AI-powered West Coast Swing dance video analysis."""
    pass


# Default models per provider
_DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "claude": "claude-sonnet-4-6",
    "claude-code": "claude-sonnet-4-6",
}


@main.command()
@click.argument("video_paths", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--provider", type=click.Choice(["gemini", "claude", "claude-code"]), default="gemini", help="AI provider: gemini (native video), claude (API), claude-code (uses local CLI).")
@click.option("--model", default=None, help="Model to use (defaults to provider's best).")
@click.option("--detail", type=click.Choice(["low", "medium", "high"]), default="medium", help="Analysis detail level.")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Save report as JSON to this path (single video only).")
@click.option("--fps", type=float, default=3.0, callback=_validate_fps, help="Frames per second to sample — Claude only (0-30).")
@click.option("--dancers", default=None, help='Describe which dancers to analyze, e.g. "lead in blue shirt, follow in red dress".')
@click.option("--no-cache", is_flag=True, default=False, help="Skip cache and force re-analysis.")
@click.option("--format", "fmt", type=click.Choice(["terminal", "json", "csv"]), default="terminal", help="Output format.")
@click.option("--parallel", "-j", type=int, default=1, help="Number of videos to analyze in parallel.")
@click.option("--pose", "use_pose", is_flag=True, default=False, help="Extract MediaPipe pose metrics and feed them as context to the LLM (Gemini only). Requires `pip install 'wcs-analyzer[pose]'`.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable verbose logging output.")
def analyze(video_paths: tuple[Path, ...], provider: str, model: str | None, detail: str, output: Path | None, fps: float, dancers: str | None, no_cache: bool, fmt: str, parallel: int, use_pose: bool, verbose: bool):
    """Analyze one or more West Coast Swing dance videos.

    Pass multiple video files to analyze them all. Use -j to run in parallel.

    \b
    Examples:
      wcs-analyzer analyze video.mp4
      wcs-analyzer analyze *.MOV -j 3 --provider claude-code
      wcs-analyzer analyze vid1.mp4 vid2.mp4 --format json
    """
    _setup_logging(verbose)
    model = model or _DEFAULT_MODELS[provider]

    if use_pose and provider != "gemini":
        console.print(
            "  [yellow]--pose currently only affects the gemini provider; "
            "ignoring for this run.[/yellow]"
        )
        use_pose = False

    if len(video_paths) == 1:
        _analyze_single(video_paths[0], provider, model, detail, output, fps, dancers, no_cache, fmt, use_pose)
    else:
        _analyze_batch(video_paths, provider, model, detail, fps, dancers, no_cache, fmt, parallel, use_pose)


def _analyze_single(
    video_path: Path, provider: str, model: str, detail: str,
    output: Path | None, fps: float, dancers: str | None, no_cache: bool, fmt: str,
    use_pose: bool = False,
) -> None:
    """Analyze a single video with full reporting."""
    from .scoring import compute_final_scores
    from .report import print_report, save_report_json, save_report_csv
    from .cache import get_cached_result, save_to_cache, segments_to_dicts, dicts_to_segments

    console.print(f"\n[bold]WCS Analyzer[/bold] — analyzing [cyan]{video_path.name}[/cyan]")
    console.print(f"  Provider: [bold]{provider}[/bold] ({model})")
    if dancers:
        console.print(f"  Dancers: [cyan]{dancers}[/cyan]")
    console.print()

    try:
        cache_key_model = f"{provider}:{model}:{dancers or ''}"
        cached = None
        if not no_cache:
            cached = get_cached_result(video_path, fps, detail, cache_key_model)

        pose_context = _compute_pose_context(video_path) if use_pose else None

        if cached is not None:
            console.print("  [cyan]Using cached analysis result[/cyan]")
            segments = dicts_to_segments(cached)
        elif provider == "gemini":
            segments = _analyze_with_gemini(video_path, model, detail, dancers, pose_context)
        elif provider == "claude-code":
            segments = _analyze_with_claude_code(video_path, detail, fps, dancers)
        else:
            segments = _analyze_with_claude(video_path, model, detail, fps, dancers)

        if cached is None and not no_cache:
            save_to_cache(video_path, fps, detail, cache_key_model, segments_to_dicts(segments))

        scores = compute_final_scores(segments)

        if fmt == "csv":
            out_path = output or Path(video_path.stem + "_report.csv")
            save_report_csv(scores, out_path)
            console.print(f"\n  CSV report saved to [cyan]{out_path}[/cyan]")
        elif fmt == "json":
            out_path = output or Path(video_path.stem + "_report.json")
            save_report_json(scores, out_path)
            console.print(f"\n  JSON report saved to [cyan]{out_path}[/cyan]")
        else:
            print_report(scores, video_path.name)
            if output:
                save_report_json(scores, output)
                console.print(f"\n  Report saved to [cyan]{output}[/cyan]")

    except WCSAnalyzerError as e:
        console.print(f"\n  [red]Error:[/red] {e}")
        raise SystemExit(1)


def _analyze_batch(
    video_paths: tuple[Path, ...], provider: str, model: str, detail: str,
    fps: float, dancers: str | None, no_cache: bool, fmt: str, parallel: int,
    use_pose: bool = False,
) -> None:
    """Analyze multiple videos, optionally in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .scoring import compute_final_scores
    from .report import print_report, save_report_json, save_report_csv
    from .cache import get_cached_result, save_to_cache, segments_to_dicts, dicts_to_segments

    console.print(f"\n[bold]WCS Analyzer[/bold] — batch analysis of [cyan]{len(video_paths)}[/cyan] videos")
    console.print(f"  Provider: [bold]{provider}[/bold] ({model})")
    if parallel > 1:
        console.print(f"  Parallel: [green]{parallel}[/green] concurrent")
    if dancers:
        console.print(f"  Dancers: [cyan]{dancers}[/cyan]")
    console.print()

    cache_key_model = f"{provider}:{model}:{dancers or ''}"

    def _process_one(video_path: Path) -> tuple[Path, object | None, str | None]:
        """Process a single video, return (path, scores, error)."""
        try:
            cached = None
            if not no_cache:
                cached = get_cached_result(video_path, fps, detail, cache_key_model)

            pose_context = _compute_pose_context(video_path) if use_pose else None

            if cached is not None:
                segments = dicts_to_segments(cached)
            elif provider == "gemini":
                segments = _analyze_with_gemini(video_path, model, detail, dancers, pose_context)
            elif provider == "claude-code":
                segments = _analyze_with_claude_code(video_path, detail, fps, dancers)
            else:
                segments = _analyze_with_claude(video_path, model, detail, fps, dancers)

            if cached is None and not no_cache:
                save_to_cache(video_path, fps, detail, cache_key_model, segments_to_dicts(segments))

            scores = compute_final_scores(segments)
            return (video_path, scores, None)
        except Exception as e:
            return (video_path, None, str(e))

    results = []
    workers = min(parallel, len(video_paths))

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one, vp): vp for vp in video_paths}
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for vp in video_paths:
            console.print(f"  Analyzing [cyan]{vp.name}[/cyan]...")
            results.append(_process_one(vp))

    # Sort by original order
    path_order = {vp: i for i, vp in enumerate(video_paths)}
    results.sort(key=lambda r: path_order[r[0]])

    # Print results
    console.print(f"\n{'=' * 60}")
    errors = []
    for video_path, scores, error in results:
        if error:
            console.print(f"\n  [red]Failed:[/red] {video_path.name} — {error}")
            errors.append(video_path.name)
            continue

        if fmt == "csv":
            out_path = Path(video_path.stem + "_report.csv")
            save_report_csv(scores, out_path)  # type: ignore[arg-type]
            console.print(f"\n  [cyan]{video_path.name}[/cyan] → {out_path}")
        elif fmt == "json":
            out_path = Path(video_path.stem + "_report.json")
            save_report_json(scores, out_path)  # type: ignore[arg-type]
            console.print(f"\n  [cyan]{video_path.name}[/cyan] → {out_path}")
        else:
            print_report(scores, video_path.name)  # type: ignore[arg-type]

    # Summary
    console.print(f"\n{'=' * 60}")
    ok = len(video_paths) - len(errors)
    console.print(f"  [bold]Done:[/bold] {ok}/{len(video_paths)} videos analyzed")
    if errors:
        console.print(f"  [red]Failed:[/red] {', '.join(errors)}")


def _analyze_with_claude_code(video_path: Path, detail: str, fps: float = 3.0, dancers: str | None = None) -> list:
    """Run analysis via the locally installed Claude Code CLI."""
    from .claude_code_analyzer import analyze_dance_claude_code

    with console.status("Analyzing with Claude Code CLI (reading frames)..."):
        segments = analyze_dance_claude_code(video_path, detail=detail, dancers=dancers, fps=fps)
    console.print("  Claude Code analyzed video frames")
    return segments


def _analyze_with_gemini(
    video_path: Path, model: str, detail: str,
    dancers: str | None = None, pose_context: str | None = None,
) -> list:
    """Run analysis via Gemini's native video understanding."""
    from .gemini_analyzer import analyze_dance_gemini

    with console.status("Uploading video to Gemini..."):
        segments = analyze_dance_gemini(
            video_path, model=model, detail=detail,
            dancers=dancers, pose_context=pose_context,
        )
    console.print("  Gemini analyzed full video (native video + audio)")
    return segments


def _compute_pose_context(video_path: Path) -> str | None:
    """Extract pose landmarks, compute metrics, and format for the LLM prompt.

    Returns None (with a warning) if MediaPipe is not installed so the
    analysis can still proceed without pose guidance.
    """
    from .pose import PoseUnavailableError, compute_all_metrics, extract_poses, format_pose_context

    try:
        with console.status("Extracting pose landmarks (MediaPipe)..."):
            pose_data = extract_poses(video_path)
    except PoseUnavailableError as e:
        console.print(f"  [yellow]Pose skipped:[/yellow] {e}")
        return None

    metrics = compute_all_metrics(pose_data)
    console.print(
        f"  Pose metrics: posture {metrics['posture'].get('mean_deg', 0):.1f}° "
        f"deviation, coverage {pose_data.coverage * 100:.0f}%"
    )
    return format_pose_context(metrics)


def _analyze_with_claude(video_path: Path, model: str, detail: str, fps: float, dancers: str | None = None) -> list:
    """Run analysis via Claude's frame-based approach."""
    from .video import extract_frames
    from .audio import extract_audio_features
    from .analyzer import analyze_dance

    with console.status("Extracting video frames..."):
        frames = extract_frames(video_path, fps=fps)
    console.print(f"  Extracted [green]{len(frames.images)}[/green] frames ({frames.duration:.1f}s video)")

    if frames.duration < MIN_VIDEO_DURATION:
        console.print(f"  [yellow]Warning: Video is very short ({frames.duration:.0f}s). Results may be limited.[/yellow]")
    elif frames.duration > MAX_VIDEO_DURATION:
        console.print(f"  [yellow]Warning: Video is long ({frames.duration:.0f}s). Consider trimming to the key section.[/yellow]")

    with console.status("Analyzing audio & detecting beats..."):
        audio = extract_audio_features(video_path)

    if audio.bpm > 0:
        console.print(f"  Detected tempo: [green]{audio.bpm:.0f} BPM[/green] ({len(audio.beat_times)} beats)")
    else:
        console.print("  [yellow]No audio detected — analysis will be visual only.[/yellow]")

    with console.status("Sending to Claude for analysis..."):
        segments = analyze_dance(frames, audio, model=model, detail=detail, dancers=dancers)
    console.print(f"  Analyzed [green]{len(segments)}[/green] segments")
    return segments


@main.command()
@click.argument("json_files", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
def compare(json_files: tuple[Path, ...]):
    """Compare scores across multiple analysis JSON reports.

    Pass 2 or more JSON report files (created via --output or --format json).

    Example: wcs-analyzer compare run1.json run2.json run3.json
    """
    from .report import print_comparison
    import json as json_mod

    if len(json_files) < 2:
        console.print("[red]Error:[/red] Need at least 2 JSON reports to compare.")
        raise SystemExit(1)

    reports = []
    for p in json_files:
        try:
            reports.append((p.stem, json_mod.loads(p.read_text())))
        except (json_mod.JSONDecodeError, OSError) as e:
            console.print(f"[red]Error reading {p}:[/red] {e}")
            raise SystemExit(1)

    print_comparison(reports)


@main.command()
@click.argument("video_path", type=click.Path(exists=True, path_type=Path))
@click.option("--provider", type=click.Choice(["gemini", "claude"]), default="gemini", help="AI provider.")
@click.option("--dancers", default=None, help='Describe which dancers to analyze.')
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable verbose logging output.")
def timing(video_path: Path, provider: str, dancers: str | None, verbose: bool):
    """Quick timing-only analysis of a WCS video.

    Focuses only on beat alignment and timing accuracy.
    """
    from .scoring import compute_final_scores
    from .report import print_timing_report

    _setup_logging(verbose)

    console.print(f"\n[bold]WCS Analyzer[/bold] — timing check for [cyan]{video_path.name}[/cyan]\n")

    try:
        model = _DEFAULT_MODELS[provider]
        if provider == "gemini":
            segments = _analyze_with_gemini(video_path, model, detail="low", dancers=dancers)
        elif provider == "claude-code":
            segments = _analyze_with_claude_code(video_path, detail="low", fps=2.0, dancers=dancers)
        else:
            segments = _analyze_with_claude(video_path, model, detail="low", fps=2.0, dancers=dancers)

        scores = compute_final_scores(segments)
        print_timing_report(scores, video_path.name)

    except WCSAnalyzerError as e:
        console.print(f"\n  [red]Error:[/red] {e}")
        raise SystemExit(1)
