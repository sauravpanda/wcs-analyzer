"""CLI entry point for wcs-analyzer."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .audio import analyze_audio_from_video
from .analyzer import run_analysis
from .report import print_report, save_json_report
from .scoring import aggregate_scores
from .video import assign_beats_to_frames, build_segments, extract_frames

console = Console()


@click.group()
def cli():
    """WCS Dance Analyzer — WSDC-style scoring from video."""


WSDC_LEVELS = [
    "newcomer", "novice", "intermediate", "advanced", "all-star", "champions", "sophisticated",
]


@cli.command()
@click.argument("video", type=click.Path(exists=True, path_type=Path))
@click.option("--model", default="claude-sonnet-4-6", show_default=True, help="Claude model to use.")
@click.option("--fps", default=3.0, show_default=True, help="Frames per second to extract.")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Save JSON report to this path.")
@click.option("--save-report", "-r", type=click.Path(path_type=Path), default=None, help="Save formatted text report to this path.")
@click.option("--hd", is_flag=True, default=False, help="Use higher-resolution frames (1080px) for better detail. Increases token usage.")
@click.option("--max-dimension", default=None, type=int, help="Max frame dimension in pixels (default 768, --hd sets 1080).")
@click.option("--competition-date", default=None, help="Date of competition (e.g. 2026-03-15).")
@click.option("--competition-level", default=None, type=click.Choice(WSDC_LEVELS, case_sensitive=False), help="WSDC competition level.")
@click.option("--api-key", envvar="ANTHROPIC_API_KEY", default=None, help="Anthropic API key.")
def analyze(
    video: Path,
    model: str,
    fps: float,
    output: Path | None,
    save_report: Path | None,
    hd: bool,
    max_dimension: int | None,
    competition_date: str | None,
    competition_level: str | None,
    api_key: str | None,
):
    """Analyze a WCS dance video and produce WSDC-style scoring."""
    if max_dimension is None:
        max_dimension = 1080 if hd else 768

    meta = {
        "competition_date": competition_date,
        "competition_level": competition_level.title() if competition_level else None,
    }

    console.print(f"[bold cyan]WCS Analyzer[/bold cyan] — [dim]{video.name}[/dim]")
    if competition_date or competition_level:
        parts = []
        if competition_level:
            parts.append(f"[magenta]{competition_level.title()}[/magenta]")
        if competition_date:
            parts.append(f"[dim]{competition_date}[/dim]")
        console.print("  " + "  |  ".join(parts))
    if hd or max_dimension != 768:
        console.print(f"  Frame resolution: [bold]{max_dimension}px[/bold] (HD mode)")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Extracting frames…", total=None)

        progress.update(task, description="Extracting frames…")
        frames, video_fps = extract_frames(video, fps=fps, max_dimension=max_dimension)
        console.print(f"  Extracted [bold]{len(frames)}[/bold] frames @ {fps}fps")

        progress.update(task, description="Analyzing audio / beat detection…")
        audio = analyze_audio_from_video(video)
        wcs_note = " [green](WCS tempo ✓)[/green]" if audio.is_wcs_tempo else " [yellow](non-standard tempo)[/yellow]"
        console.print(f"  BPM: [bold]{audio.bpm:.1f}[/bold]{wcs_note}  |  Beats detected: {len(audio.beat_times)}")

        progress.update(task, description="Building segments…")
        assign_beats_to_frames(frames, audio.beat_times)
        segments = build_segments(frames, bpm=audio.bpm)
        console.print(f"  Segments: [bold]{len(segments)}[/bold] (8-count phrases)")

        def update_progress(current, total, description):
            progress.update(task, description=description)

        progress.update(task, description="Sending to Claude…")
        segment_scores, summary = run_analysis(
            video_path=video,
            audio=audio,
            segments=segments,
            model=model,
            api_key=api_key,
            progress_callback=update_progress,
        )

        progress.update(task, description="Computing final scores…")
        result = aggregate_scores(segment_scores, summary)

    print_report(result, video, meta=meta, save_report=save_report)

    if output:
        save_json_report(result, output, meta=meta)


@cli.command()
@click.argument("video", type=click.Path(exists=True, path_type=Path))
def timing(video: Path):
    """Quick timing check — BPM and beat timestamps only (no LLM calls)."""
    console.print(f"[bold cyan]WCS Timing Check[/bold cyan] — [dim]{video.name}[/dim]")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Analyzing audio…", total=None)
        audio = analyze_audio_from_video(video)

    wcs_note = "(WCS tempo ✓)" if audio.is_wcs_tempo else "(non-standard tempo)"
    console.print(f"\n  BPM: [bold]{audio.bpm:.1f}[/bold] {wcs_note}")
    console.print(f"  Duration: {audio.duration:.1f}s")
    console.print(f"  Beats detected: {len(audio.beat_times)}")
    console.print(f"  First 10 beat times: {[f'{t:.2f}s' for t in audio.beat_times[:10]]}")
