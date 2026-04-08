"""CLI report formatting with rich."""

import csv
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .scoring import FinalScores

_COMPARE_CATEGORIES = ["timing", "technique", "teamwork", "presentation"]

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
        Text("WCS Dance Analysis Report", style="bold white"),
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

    # Technique sub-scores with notes
    tech_table = Table(title="Technique Breakdown", show_header=True, header_style="bold cyan")
    tech_table.add_column("Area", style="bold", width=12)
    tech_table.add_column("Score", justify="center", width=8)
    tech_table.add_column("", width=12)
    tech_table.add_column("Notes", width=50)

    # Extract technique notes from raw data (summary or first segment)
    tech_raw = {}
    for seg in scores.segments:
        if seg.raw_data and "technique" in seg.raw_data:
            tech_raw = seg.raw_data["technique"]
            break

    sub_scores = [
        ("Posture", scores.posture, tech_raw.get("posture", {}).get("notes", "")),
        ("Extension", scores.extension, tech_raw.get("extension", {}).get("notes", "")),
        ("Footwork", scores.footwork, tech_raw.get("footwork", {}).get("notes", "")),
        ("Slot", scores.slot, tech_raw.get("slot", {}).get("notes", "")),
    ]

    for name, score, notes in sub_scores:
        color = _score_color(score)
        tech_table.add_row(
            name,
            f"[{color}]{score}[/{color}]",
            f"[{color}]{_score_bar(score)}[/{color}]",
            f"[dim]{notes}[/dim]" if notes else "",
        )

    console.print(tech_table)
    console.print()

    # Partner breakdown (only if data was detected)
    if scores.lead_technique > 0 or scores.follow_technique > 0:
        partner_table = Table(title="Partner Breakdown", show_header=True, header_style="bold cyan")
        partner_table.add_column("", style="bold", width=16)
        partner_table.add_column("Lead", justify="center", width=12)
        partner_table.add_column("Follow", justify="center", width=12)

        for label, lead_s, follow_s in [
            ("Technique", scores.lead_technique, scores.follow_technique),
            ("Presentation", scores.lead_presentation, scores.follow_presentation),
        ]:
            lc = _score_color(lead_s)
            fc = _score_color(follow_s)
            partner_table.add_row(
                label,
                f"[{lc}]{lead_s}[/{lc}]",
                f"[{fc}]{follow_s}[/{fc}]",
            )

        console.print(partner_table)

        if scores.lead_notes:
            console.print(f"\n  [bold]Lead notes:[/bold] {scores.lead_notes}")
        if scores.follow_notes:
            console.print(f"  [bold]Follow notes:[/bold] {scores.follow_notes}")
        console.print()

    # Off-beat moments
    if scores.off_beat_moments:
        console.print(f"  [bold]Off-beat moments:[/bold] [red]{scores.total_off_beat} detected[/red]")
        for moment in scores.off_beat_moments:
            if isinstance(moment, str):
                console.print(f"    [red]-[/red] {moment}")
            else:
                time_str = moment.get("timestamp_approx", moment.get("time", "?"))
                desc = moment.get("description", "")
                beat = moment.get("beat_count", "")
                beat_str = f" ({beat})" if beat else ""
                console.print(f"    [red]-[/red] {time_str}{beat_str}: {desc}")
        console.print()
    else:
        console.print("  [bold]Off-beat moments:[/bold] [green]None detected[/green]\n")

    # Patterns — detailed table if available, otherwise simple list
    if scores.pattern_details:
        pat_table = Table(title="Pattern Analysis", show_header=True, header_style="bold cyan")
        pat_table.add_column("Pattern", style="bold", width=22)
        pat_table.add_column("Quality", justify="center", width=12)
        pat_table.add_column("Timing", justify="center", width=12)
        pat_table.add_column("Notes", width=40)

        quality_colors = {"strong": "green", "solid": "yellow", "needs_work": "dark_orange", "weak": "red"}
        timing_colors = {"on_beat": "green", "slightly_off": "yellow", "off_beat": "red"}

        for pd in scores.pattern_details:
            q = pd.get("quality", "solid")
            t = pd.get("timing", "on_beat")
            qc = quality_colors.get(q, "white")
            tc = timing_colors.get(t, "white")
            pat_table.add_row(
                pd.get("name", "?"),
                f"[{qc}]{q.replace('_', ' ')}[/{qc}]",
                f"[{tc}]{t.replace('_', ' ')}[/{tc}]",
                pd.get("notes", ""),
            )

        console.print(pat_table)
        console.print()
    elif scores.all_patterns:
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
            if isinstance(moment, str):
                console.print(f"    [red]\u2022[/red] {moment}")
            else:
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
        "partner_breakdown": {
            "lead": {
                "technique": scores.lead_technique,
                "presentation": scores.lead_presentation,
                "notes": scores.lead_notes,
            },
            "follow": {
                "technique": scores.follow_technique,
                "presentation": scores.follow_presentation,
                "notes": scores.follow_notes,
            },
        },
        "off_beat_moments": scores.off_beat_moments,
        "total_off_beat": scores.total_off_beat,
        "patterns": scores.all_patterns,
        "pattern_details": scores.pattern_details,
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


def save_report_csv(scores: FinalScores, path: Path) -> None:
    """Save the report as CSV with a summary row and per-segment rows."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        # Summary section
        writer.writerow(["Section", "Category", "Score", "Grade", "Details"])
        writer.writerow(["Overall", "", scores.overall, scores.grade, scores.overall_impression])
        writer.writerow(["Category", "Timing", scores.timing, "", "Weight: 30%"])
        writer.writerow(["Category", "Technique", scores.technique, "", "Weight: 30%"])
        writer.writerow(["Category", "Teamwork", scores.teamwork, "", "Weight: 20%"])
        writer.writerow(["Category", "Presentation", scores.presentation, "", "Weight: 20%"])
        writer.writerow(["Technique", "Posture", scores.posture, "", ""])
        writer.writerow(["Technique", "Extension", scores.extension, "", ""])
        writer.writerow(["Technique", "Footwork", scores.footwork, "", ""])
        writer.writerow(["Technique", "Slot", scores.slot, "", ""])
        writer.writerow([])
        # Strengths / improvements
        for s in scores.top_strengths:
            writer.writerow(["Strength", "", "", "", s])
        for s in scores.top_improvements:
            writer.writerow(["Improvement", "", "", "", s])
        if scores.all_patterns:
            writer.writerow(["Patterns", "", "", "", "; ".join(scores.all_patterns)])
        writer.writerow([])
        # Per-segment scores
        writer.writerow(["Segment", "Start", "End", "Timing", "Technique", "Teamwork", "Presentation"])
        for i, seg in enumerate(scores.segments):
            writer.writerow([
                i + 1,
                f"{seg.start_time:.1f}",
                f"{seg.end_time:.1f}",
                seg.timing_score,
                seg.technique_score,
                seg.teamwork_score,
                seg.presentation_score,
            ])


def _trend(current: float, previous: float) -> str:
    """Return a trend indicator comparing two scores."""
    diff = current - previous
    if diff > 0.5:
        return f"[green]+{diff:.1f} \u2191[/green]"
    if diff < -0.5:
        return f"[red]{diff:.1f} \u2193[/red]"
    return "[dim]\u2192[/dim]"


def print_comparison(reports: list[tuple[str, dict]]) -> None:
    """Print a side-by-side comparison of multiple analysis reports."""
    console.print()
    console.print(Panel(
        Text("WCS Score Comparison", style="bold white", justify="center"),
        style="blue", padding=(1, 2),
    ))

    # Overall scores table
    table = Table(title="Overall Scores", show_header=True, header_style="bold cyan")
    table.add_column("Report", style="bold", width=20)
    table.add_column("Overall", justify="center", width=10)
    table.add_column("Grade", justify="center", width=8)
    table.add_column("Trend", justify="center", width=10)

    prev_overall = None
    for name, data in reports:
        scores = data.get("scores", {})
        overall = scores.get("overall", 0)
        grade = scores.get("grade", "?")
        color = _score_color(overall)
        trend_str = _trend(overall, prev_overall) if prev_overall is not None else ""
        table.add_row(name, f"[{color}]{overall}[/{color}]", f"[{color}]{grade}[/{color}]", trend_str)
        prev_overall = overall

    console.print(table)
    console.print()

    # Category breakdown
    cat_table = Table(title="Category Comparison", show_header=True, header_style="bold cyan")
    cat_table.add_column("Category", style="bold", width=16)
    for name, _ in reports:
        cat_table.add_column(name, justify="center", width=12)

    for cat in _COMPARE_CATEGORIES:
        row = [cat.title()]
        for _, data in reports:
            score = data.get("scores", {}).get(cat, 0)
            color = _score_color(score)
            row.append(f"[{color}]{score}[/{color}]")
        cat_table.add_row(*row)

    console.print(cat_table)
    console.print()

    # Technique sub-scores
    tech_table = Table(title="Technique Breakdown", show_header=True, header_style="bold cyan")
    tech_table.add_column("Area", style="bold", width=16)
    for name, _ in reports:
        tech_table.add_column(name, justify="center", width=12)

    for area in ["posture", "extension", "footwork", "slot"]:
        row = [area.title()]
        for _, data in reports:
            score = data.get("technique_breakdown", {}).get(area, 0)
            color = _score_color(score)
            row.append(f"[{color}]{score}[/{color}]")
        tech_table.add_row(*row)

    console.print(tech_table)
    console.print()
