"""CLI report rendering using Rich."""

from __future__ import annotations

import json
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .scoring import FinalScore

console = Console()


def _bar(score: float, width: int = 10) -> str:
    filled = round(score / 10 * width)
    return "█" * filled + "░" * (width - filled)


def _score_color(score: float) -> str:
    if score >= 8.5:
        return "bright_green"
    if score >= 7.0:
        return "green"
    if score >= 5.5:
        return "yellow"
    return "red"


def print_report(
    result: FinalScore,
    video_path: Path,
    meta: dict | None = None,
    save_report: Path | None = None,
) -> None:
    meta = meta or {}

    outputs: list[Console] = [console]
    _file = None
    if save_report:
        _file = open(save_report, "w", encoding="utf-8")
        outputs.append(Console(file=_file, highlight=False, markup=True))

    def _print(*args, **kwargs):
        for c in outputs:
            c.print(*args, **kwargs)

    _print()

    # Header
    header_lines = ["[bold cyan]WCS Dance Analysis Report[/bold cyan]", f"[dim]{video_path.name}[/dim]"]
    if meta.get("competition_level"):
        header_lines.append(f"Level: [magenta]{meta['competition_level']}[/magenta]")
    if meta.get("competition_date"):
        header_lines.append(f"Date:  [dim]{meta['competition_date']}[/dim]")
    _print(Panel("\n".join(header_lines), box=box.DOUBLE, expand=False))

    # Overall score
    overall_color = _score_color(result.overall)
    _print(Panel(
        f"[bold {overall_color}]Overall Score: {result.overall:.1f} / 10  ({result.grade})[/bold {overall_color}]",
        box=box.ROUNDED,
        expand=False,
    ))

    # Category breakdown
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
    table.add_column("Category", style="bold", min_width=22)
    table.add_column("Score", justify="right", min_width=8)
    table.add_column("Visual", min_width=12)
    table.add_column("Weight", justify="right", min_width=7)

    categories = [
        ("Timing & Rhythm", result.timing, "30%"),
        ("Technique", result.technique, "30%"),
        ("Teamwork", result.teamwork, "20%"),
        ("Presentation", result.presentation, "20%"),
    ]
    for name, score, weight in categories:
        color = _score_color(score)
        table.add_row(
            name,
            f"[{color}]{score:.1f}[/{color}]",
            f"[{color}]{_bar(score)}[/{color}]",
            f"[dim]{weight}[/dim]",
        )
    _print(table)

    # Off-beat moments
    if result.off_beat_moments:
        obt = Table(title="Off-Beat Moments", box=box.SIMPLE, show_header=True, header_style="bold red")
        obt.add_column("Time", style="red", min_width=8)
        obt.add_column("Description")
        for m in result.off_beat_moments:
            ts = m.get("timestamp", 0)
            mins = int(ts // 60)
            secs = ts % 60
            obt.add_row(f"{mins}:{secs:04.1f}", m.get("description", ""))
        _print(obt)
    else:
        _print("[green]No significant off-beat moments detected.[/green]")

    # Highlights
    if result.highlight_moments:
        ht = Table(title="Highlight Moments", box=box.SIMPLE, show_header=True, header_style="bold green")
        ht.add_column("Time", style="green", min_width=8)
        ht.add_column("Description")
        for m in result.highlight_moments:
            ts = m.get("timestamp", 0)
            mins = int(ts // 60)
            secs = ts % 60
            ht.add_row(f"{mins}:{secs:04.1f}", m.get("description", ""))
        _print(ht)

    # Strengths & improvements
    if result.top_strengths:
        _print("\n[bold green]Strengths:[/bold green]")
        for s in result.top_strengths:
            _print(f"  [green]•[/green] {s}")

    if result.areas_to_improve:
        _print("\n[bold yellow]Areas to Improve:[/bold yellow]")
        for a in result.areas_to_improve:
            _print(f"  [yellow]•[/yellow] {a}")

    # Judge commentary
    if result.judge_commentary:
        _print(Panel(
            f"[italic]{result.judge_commentary}[/italic]",
            title="[bold]Judge Commentary[/bold]",
            box=box.ROUNDED,
        ))

    _print()

    if _file:
        _file.close()
        console.print(f"[dim]Report saved to {save_report}[/dim]")


def save_json_report(result: FinalScore, output_path: Path, meta: dict | None = None) -> None:
    meta = meta or {}
    data: dict = {}
    if meta.get("competition_date"):
        data["competition_date"] = meta["competition_date"]
    if meta.get("competition_level"):
        data["competition_level"] = meta["competition_level"]
    data.update({
        "overall": result.overall,
        "grade": result.grade,
        "timing": result.timing,
        "technique": result.technique,
        "teamwork": result.teamwork,
        "presentation": result.presentation,
        "top_strengths": result.top_strengths,
        "areas_to_improve": result.areas_to_improve,
        "judge_commentary": result.judge_commentary,
        "off_beat_moments": result.off_beat_moments,
        "highlight_moments": result.highlight_moments,
        "segments": [
            {
                "index": s.segment_index,
                "start": s.start_time,
                "end": s.end_time,
                "timing": s.timing,
                "technique": s.technique,
                "teamwork": s.teamwork,
                "presentation": s.presentation,
            }
            for s in result.segment_scores
        ],
    })
    output_path.write_text(json.dumps(data, indent=2))
    console.print(f"[dim]JSON report saved to {output_path}[/dim]")
