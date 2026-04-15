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
@click.option("--providers", "providers_list", default=None, help="Comma-separated list of providers for ensemble mode (e.g. 'gemini,claude-code'). Each runs independently and results are aggregated with disagreement flagging.")
@click.option("--save-history", "save_history", default=None, help="Persist this run's scores to the longitudinal history DB under the given dancer name. Use `wcs-analyzer progress <name>` to review.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable verbose logging output.")
def analyze(video_paths: tuple[Path, ...], provider: str, model: str | None, detail: str, output: Path | None, fps: float, dancers: str | None, no_cache: bool, fmt: str, parallel: int, use_pose: bool, providers_list: str | None, save_history: str | None, verbose: bool):
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

    if providers_list:
        providers = [p.strip() for p in providers_list.split(",") if p.strip()]
        unknown = [p for p in providers if p not in _DEFAULT_MODELS]
        if unknown:
            console.print(f"[red]Unknown provider(s) in --providers:[/red] {', '.join(unknown)}")
            raise SystemExit(1)
        if len(providers) < 2:
            console.print("[red]--providers needs at least two provider names.[/red]")
            raise SystemExit(1)
        if len(video_paths) != 1:
            console.print("[red]--providers currently only supports a single video at a time.[/red]")
            raise SystemExit(1)
        _analyze_ensemble(video_paths[0], providers, detail, output, fps, dancers, no_cache, fmt, use_pose)
        return

    if use_pose and provider != "gemini":
        console.print(
            "  [yellow]--pose currently only affects the gemini provider; "
            "ignoring for this run.[/yellow]"
        )
        use_pose = False

    if len(video_paths) == 1:
        _analyze_single(video_paths[0], provider, model, detail, output, fps, dancers, no_cache, fmt, use_pose, save_history)
    else:
        _analyze_batch(video_paths, provider, model, detail, fps, dancers, no_cache, fmt, parallel, use_pose, save_history)


def _analyze_single(
    video_path: Path, provider: str, model: str, detail: str,
    output: Path | None, fps: float, dancers: str | None, no_cache: bool, fmt: str,
    use_pose: bool = False, save_history: str | None = None,
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

        if save_history:
            from .history import save_run
            save_run(save_history, video_path.name, provider, scores)
            console.print(f"  History saved for [cyan]{save_history}[/cyan]")

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


def _analyze_ensemble(
    video_path: Path, providers: list[str], detail: str,
    output: Path | None, fps: float, dancers: str | None,
    no_cache: bool, fmt: str, use_pose: bool,
) -> None:
    """Run multiple providers on a single video and aggregate their scores."""
    from .scoring import aggregate_ensemble, compute_final_scores
    from .report import print_ensemble_report, save_ensemble_json
    from .cache import get_cached_result, save_to_cache, segments_to_dicts, dicts_to_segments

    console.print(f"\n[bold]WCS Analyzer[/bold] — ensemble analysis of [cyan]{video_path.name}[/cyan]")
    console.print(f"  Providers: [bold]{', '.join(providers)}[/bold]")
    if dancers:
        console.print(f"  Dancers: [cyan]{dancers}[/cyan]")
    console.print()

    pose_context = _compute_pose_context(video_path) if use_pose else None
    per_provider = {}

    for provider in providers:
        model = _DEFAULT_MODELS[provider]
        console.print(f"  [bold]\u25b8[/bold] Running [cyan]{provider}[/cyan] ({model})...")
        cache_key_model = f"{provider}:{model}:{dancers or ''}"
        cached = None
        if not no_cache:
            cached = get_cached_result(video_path, fps, detail, cache_key_model)

        try:
            if cached is not None:
                console.print("    [cyan]Using cached result[/cyan]")
                segments = dicts_to_segments(cached)
            elif provider == "gemini":
                segments = _analyze_with_gemini(video_path, model, detail, dancers, pose_context)
            elif provider == "claude-code":
                segments = _analyze_with_claude_code(video_path, detail, fps, dancers)
            else:
                segments = _analyze_with_claude(video_path, model, detail, fps, dancers)

            if cached is None and not no_cache:
                save_to_cache(video_path, fps, detail, cache_key_model, segments_to_dicts(segments))
        except WCSAnalyzerError as e:
            console.print(f"    [red]{provider} failed:[/red] {e}")
            continue

        per_provider[provider] = compute_final_scores(segments)

    if len(per_provider) < 2:
        console.print(
            "\n[red]Ensemble needs at least 2 successful providers. "
            f"Got {len(per_provider)}.[/red]"
        )
        raise SystemExit(1)

    ensemble = aggregate_ensemble(per_provider)

    if fmt == "json":
        out_path = output or Path(video_path.stem + "_ensemble.json")
        save_ensemble_json(ensemble, out_path)
        console.print(f"\n  Ensemble JSON saved to [cyan]{out_path}[/cyan]")
    else:
        print_ensemble_report(ensemble, video_path.name)
        if output:
            save_ensemble_json(ensemble, output)
            console.print(f"\n  Ensemble JSON saved to [cyan]{output}[/cyan]")


def _analyze_batch(
    video_paths: tuple[Path, ...], provider: str, model: str, detail: str,
    fps: float, dancers: str | None, no_cache: bool, fmt: str, parallel: int,
    use_pose: bool = False, save_history: str | None = None,
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
            if save_history:
                from .history import save_run
                save_run(save_history, video_path.name, provider, scores)
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
    """Extract pose landmarks, compute metrics (including audio-synced
    beat verification), and format for the LLM prompt.

    Returns None (with a warning) if MediaPipe is not installed so the
    analysis can still proceed without pose guidance.
    """
    from .audio import extract_audio_features
    from .exceptions import AudioProcessingError
    from .pose import PoseUnavailableError, compute_all_metrics, extract_poses, format_pose_context

    try:
        with console.status("Extracting pose landmarks (MediaPipe)..."):
            pose_data = extract_poses(video_path)
    except PoseUnavailableError as e:
        console.print(f"  [yellow]Pose skipped:[/yellow] {e}")
        return None

    beat_times: list[float] = []
    bpm = 0.0
    try:
        with console.status("Extracting audio beats for beat-sync..."):
            audio = extract_audio_features(video_path)
        beat_times = audio.beat_times
        bpm = audio.bpm
    except AudioProcessingError as e:
        console.print(f"  [yellow]Beat-sync skipped:[/yellow] {e}")

    metrics = compute_all_metrics(pose_data, beat_times=beat_times, bpm=bpm)
    summary = (
        f"  Pose metrics: posture {metrics['posture'].get('mean_deg', 0):.1f}° "
        f"deviation, coverage {pose_data.coverage * 100:.0f}%"
    )
    if "beat_sync" in metrics:
        summary += (
            f", beat-sync {metrics['beat_sync'].get('timing_score', 0):.1f}/10 "
            f"({metrics['beat_sync'].get('mean_offset_ms', 0):.0f}ms mean offset)"
        )
    console.print(summary)
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


@main.command()
@click.argument("video_path", type=click.Path(exists=True, path_type=Path))
@click.option("--model", default="claude-sonnet-4-6", help="Claude model for pattern detection.")
@click.option("--fps", type=float, default=3.0, callback=_validate_fps, help="Frame sampling rate.")
def patterns(video_path: Path, model: str, fps: float):
    """Detect and list the WCS patterns on the video's timeline.

    Runs a single cheap LLM call to pre-segment the dance into named
    patterns (sugar push, whip, side pass, ...) with approximate time
    ranges. Useful before a full analysis so you know what the scorer
    is actually looking at.
    """
    import anthropic
    from rich.table import Table
    from .analyzer import detect_pattern_timeline
    from .video import extract_frames

    console.print(f"\n[bold]WCS Analyzer[/bold] — pattern timeline for [cyan]{video_path.name}[/cyan]\n")

    try:
        with console.status("Extracting frames..."):
            frames = extract_frames(video_path, fps=fps)
        console.print(f"  Extracted [green]{len(frames.images)}[/green] frames")

        client = anthropic.Anthropic()
        with console.status("Detecting patterns..."):
            timeline = detect_pattern_timeline(client, model, frames)
    except WCSAnalyzerError as e:
        console.print(f"\n  [red]Error:[/red] {e}")
        raise SystemExit(1)

    if not timeline:
        console.print("  [yellow]No patterns detected.[/yellow]")
        return

    table = Table(title="Pattern Timeline", show_header=True, header_style="bold cyan")
    table.add_column("#", justify="right", width=4)
    table.add_column("Start", justify="right", width=8)
    table.add_column("End", justify="right", width=8)
    table.add_column("Duration", justify="right", width=10)
    table.add_column("Pattern", width=24)
    table.add_column("Confidence", justify="center", width=12)

    for i, seg in enumerate(timeline, 1):
        dur = seg.end_time - seg.start_time
        conf_color = "green" if seg.confidence >= 0.7 else "yellow" if seg.confidence >= 0.4 else "red"
        table.add_row(
            str(i),
            f"{seg.start_time:.1f}s",
            f"{seg.end_time:.1f}s",
            f"{dur:.1f}s",
            seg.name,
            f"[{conf_color}]{seg.confidence:.2f}[/{conf_color}]",
        )
    console.print(table)
    console.print()


@main.command()
@click.argument("dancer", required=True)
def progress(dancer: str):
    """Show longitudinal progress for a tracked dancer.

    Reads the history DB populated by `analyze --save-history <name>` and
    renders a timeline plus linear trajectory fits for each category.
    """
    from .history import fit_trajectories, load_history
    from .report import print_progress_report

    rows = load_history(dancer)
    if not rows:
        console.print(
            f"\n  [yellow]No history for '{dancer}'.[/yellow] "
            f"Run `wcs-analyzer analyze <video> --save-history {dancer}` first."
        )
        return

    trajectories = fit_trajectories(rows)
    print_progress_report(dancer, rows, trajectories)


@main.command(name="dancers")
def list_dancers_cmd():
    """List every dancer tracked in the longitudinal history store."""
    from .history import list_dancers

    entries = list_dancers()
    if not entries:
        console.print(
            "\n  [yellow]No dancers tracked yet.[/yellow] "
            "Use `analyze --save-history <name>` to start."
        )
        return

    console.print("\n  [bold]Tracked dancers:[/bold]")
    for name, count in entries:
        console.print(f"    [cyan]{name}[/cyan] — {count} run{'s' if count != 1 else ''}")
