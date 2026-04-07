"""CLI report formatting with rich."""

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .scoring import FinalScores

console = Console()


def _score_bar(score: float, width: int = 10) -> str:
    """Create a visual bar for a score."""
    filled = round(score * width / 10)
    return "\u2588" * filled + "\u2591" * (width - filled)


def _score_color(score: float) -> str:
    """Get color based on score."""
    if score >= 8:
        return "green"
    if score >= 6:
        return "yellow"
    if score >= 4:
        return "dark_orange"
    return "red"


def _format_time(seconds: float) -> str:
    """Format seconds as M:SS."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def print_report(scores: FinalScores, video_name: str) -> None:
    """Print the full analysis report to the terminal."""
    console.print()

    # Header
    header = Table.grid(padding=1)
    header.add_column(justify="center")
    header.add_row(
        Text(f"WCS Dance Analysis Report", style="bold white"),
    )
    header.add_row(Text(video_name, style="dim"))

    console.print(Panel(header, style="blue", padding=(1, 2)))

    # Overall score
    color = _score_color(scores.overall)
    overall_text = Text()
    overall_text.append("  Overall Score: ", style="bold")
    overall_text.append(f"{scores.overall} / 10", style=f"bold {color}")
    overall_text.append(f"  ({scores.grade})", style=f"bold {color}")
    console.print(Panel(overall_text, style=color))

    # Category scores table
    table = Table(title="Category Scores", show_header=True, header_style="bold cyan")
    table.add_column("Category", style="bold", width=20)
    table.add_column("Score", justify="center", width=10)
    table.add_column("", width=12)
    table.add_column("Weight", justify="center", width=8)

    categories = [
        ("Timing & Rhythm", scores.timing, "30%"),
        ("Technique", scores.technique, "30%"),
        ("Teamwork", scores.teamwork, "20%"),
        ("Presentation", scores.presentation, "20%"),
    ]

    for name, score, weight in categories:
        color = _score_color(score)
        table.add_row(
            name,
            f"[{color}]{score}[/{color}]",
            f"[{color}]{_score_bar(score)}[/{color}]",
            weight,
        )

    console.print(table)
    console.print()

    # Technique sub-scores
    tech_table = Table(title="Technique Breakdown", show_header=True, header_style="bold cyan")
    tech_table.add_column("Area", style="bold", width=16)
    tech_table.add_column("Score", justify="center", width=10)
    tech_table.add_column("", width=12)

    sub_scores = [
        ("Posture", scores.posture),
        ("Extension", scores.extension),
        ("Footwork", scores.footwork),
        ("Slot", scores.slot),
    ]

    for name, score in sub_scores:
        color = _score_color(score)
        tech_table.add_row(
            name,
            f"[{color}]{score}[/{color}]",
            f"[{color}]{_score_bar(score)}[/{color}]",
        )

    console.print(tech_table)
    console.print()

    # Off-beat moments
    if scores.off_beat_moments:
        console.print(f"  [bold]Off-beat moments:[/bold] [red]{scores.total_off_beat} detected[/red]")
        for moment in scores.off_beat_moments:
            time_str = moment.get("timestamp_approx", moment.get("time", "?"))
            desc = moment.get("description", "")
            beat = moment.get("beat_count", "")
            beat_str = f" ({beat})" if beat else ""
            console.print(f"    [red]-[/red] {time_str}{beat_str}: {desc}")
        console.print()
    else:
        console.print("  [bold]Off-beat moments:[/bold] [green]None detected[/green]\n")

    # Patterns identified
    if scores.all_patterns:
        console.print(f"  [bold]Patterns identified:[/bold] {', '.join(scores.all_patterns)}")
        console.print()

    # Strengths
    if scores.top_strengths:
        console.print("  [bold green]Strengths:[/bold green]")
        for s in scores.top_strengths:
            console.print(f"    [green]\u2022[/green] {s}")
        console.print()

    # Improvements
    if scores.top_improvements:
        console.print("  [bold yellow]Areas to Improve:[/bold yellow]")
        for s in scores.top_improvements:
            console.print(f"    [yellow]\u2022[/yellow] {s}")
        console.print()

    # Overall impression
    if scores.overall_impression:
        console.print(Panel(scores.overall_impression, title="Judge's Notes", style="dim"))


def print_timing_report(scores: FinalScores, video_name: str) -> None:
    """Print a timing-focused report."""
    console.print()

    color = _score_color(scores.timing)
    console.print(Panel(
        f"  [bold]Timing Score: [{color}]{scores.timing} / 10[/{color}][/bold]  "
        f"({scores.grade})",
        title=f"Timing Check \u2014 {video_name}",
        style=color,
    ))

    if scores.off_beat_moments:
        console.print(f"\n  [bold]Off-beat moments:[/bold] [red]{scores.total_off_beat}[/red]\n")
        for moment in scores.off_beat_moments:
            time_str = moment.get("timestamp_approx", moment.get("time", "?"))
            desc = moment.get("description", "")
            beat = moment.get("beat_count", "")
            beat_str = f" ({beat})" if beat else ""
            console.print(f"    [red]\u2022[/red] {time_str}{beat_str}: {desc}")
    else:
        console.print("\n  [green]No off-beat moments detected![/green]")

    console.print()


def save_report_json(scores: FinalScores, path: Path) -> None:
    """Save the full report as JSON."""
    data = {
        "scores": {
            "overall": scores.overall,
            "grade": scores.grade,
            "timing": scores.timing,
            "technique": scores.technique,
            "teamwork": scores.teamwork,
            "presentation": scores.presentation,
        },
        "technique_breakdown": {
            "posture": scores.posture,
            "extension": scores.extension,
            "footwork": scores.footwork,
            "slot": scores.slot,
        },
        "off_beat_moments": scores.off_beat_moments,
        "total_off_beat": scores.total_off_beat,
        "patterns": scores.all_patterns,
        "strengths": scores.top_strengths,
        "improvements": scores.top_improvements,
        "overall_impression": scores.overall_impression,
        "segments": [
            {
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "timing": seg.timing_score,
                "technique": seg.technique_score,
                "teamwork": seg.teamwork_score,
                "presentation": seg.presentation_score,
            }
            for seg in scores.segments
        ],
    }
    path.write_text(json.dumps(data, indent=2))
