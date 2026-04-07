"""CLI entry point for WCS Analyzer."""

import click
from pathlib import Path

from rich.console import Console

console = Console()


@click.group()
@click.version_option(package_name="wcs-analyzer")
def main():
    """WCS Analyzer — AI-powered West Coast Swing dance video analysis."""
    pass


@main.command()
@click.argument("video_path", type=click.Path(exists=True, path_type=Path))
@click.option("--model", default="claude-sonnet-4-6", help="Claude model to use for analysis.")
@click.option("--detail", type=click.Choice(["low", "medium", "high"]), default="medium", help="Analysis detail level.")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Save report as JSON to this path.")
@click.option("--fps", type=float, default=3.0, help="Frames per second to sample from video.")
def analyze(video_path: Path, model: str, detail: str, output: Path | None, fps: float):
    """Analyze a West Coast Swing dance video.

    Extracts frames and audio, detects beats, sends to Claude for
    WSDC-style scoring across timing, technique, teamwork, and presentation.
    """
    from .video import extract_frames
    from .audio import extract_audio_features
    from .analyzer import analyze_dance
    from .scoring import compute_final_scores
    from .report import print_report, save_report_json

    console.print(f"\n[bold]WCS Analyzer[/bold] — analyzing [cyan]{video_path.name}[/cyan]\n")

    with console.status("Extracting video frames..."):
        frames = extract_frames(video_path, fps=fps)
    console.print(f"  Extracted [green]{len(frames.images)}[/green] frames ({frames.duration:.1f}s video)")

    with console.status("Analyzing audio & detecting beats..."):
        audio = extract_audio_features(video_path)
    console.print(f"  Detected tempo: [green]{audio.bpm:.0f} BPM[/green] ({len(audio.beat_times)} beats)")

    with console.status("Sending to Claude for analysis..."):
        segments = analyze_dance(frames, audio, model=model, detail=detail)
    console.print(f"  Analyzed [green]{len(segments)}[/green] segments")

    scores = compute_final_scores(segments)

    print_report(scores, video_path.name)

    if output:
        save_report_json(scores, output)
        console.print(f"\n  Report saved to [cyan]{output}[/cyan]")


@main.command()
@click.argument("video_path", type=click.Path(exists=True, path_type=Path))
def timing(video_path: Path):
    """Quick timing-only analysis of a WCS video.

    Focuses only on beat alignment and timing accuracy.
    """
    from .video import extract_frames
    from .audio import extract_audio_features
    from .analyzer import analyze_dance
    from .scoring import compute_final_scores
    from .report import print_timing_report

    console.print(f"\n[bold]WCS Analyzer[/bold] — timing check for [cyan]{video_path.name}[/cyan]\n")

    with console.status("Extracting video frames..."):
        frames = extract_frames(video_path, fps=2.0)

    with console.status("Analyzing audio & detecting beats..."):
        audio = extract_audio_features(video_path)
    console.print(f"  Tempo: [green]{audio.bpm:.0f} BPM[/green]")

    with console.status("Analyzing timing..."):
        segments = analyze_dance(frames, audio, model="claude-sonnet-4-6", detail="low")

    scores = compute_final_scores(segments)
    print_timing_report(scores, video_path.name)
