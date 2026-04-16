"""CLI report formatting with rich."""

import csv
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .history import HistoryRow, Trajectory
from .pricing import pricing_updated_on
from .scoring import EnsembleScores, FinalScores

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


def _ci_str(low: float, high: float) -> str:
    """Format a confidence interval, or empty if it's effectively a point."""
    if high - low < 0.2:
        return ""
    return f"[{low:.1f}\u2013{high:.1f}]"


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
    ci = _ci_str(scores.overall_low, scores.overall_high)
    if ci:
        overall_text.append(f"  {ci}", style=f"dim {color}")
    overall_text.append(f"  ({scores.grade})", style=f"bold {color}")
    if scores.low_confidence:
        overall_text.append("  \u26a0 low confidence", style="bold yellow")
    console.print(Panel(overall_text, style=color))

    # Category scores table
    table = Table(title="Category Scores", show_header=True, header_style="bold cyan")
    table.add_column("Category", style="bold", width=20)
    table.add_column("Score", justify="center", width=10)
    table.add_column("", width=12)
    table.add_column("Range", justify="center", width=12)
    table.add_column("Weight", justify="center", width=8)

    categories = [
        ("Timing & Rhythm", scores.timing, scores.timing_low, scores.timing_high, "30%"),
        ("Technique", scores.technique, scores.technique_low, scores.technique_high, "30%"),
        ("Teamwork", scores.teamwork, scores.teamwork_low, scores.teamwork_high, "20%"),
        ("Presentation", scores.presentation, scores.presentation_low, scores.presentation_high, "20%"),
    ]

    for name, score, lo, hi, weight in categories:
        color = _score_color(score)
        table.add_row(
            name,
            f"[{color}]{score}[/{color}]",
            f"[{color}]{_score_bar(score)}[/{color}]",
            f"[dim]{_ci_str(lo, hi)}[/dim]",
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
        pat_table.add_column("Seen", justify="center", width=6)
        pat_table.add_column("Quality", justify="center", width=12)
        pat_table.add_column("Timing", justify="center", width=12)
        pat_table.add_column("Notes", width=36)

        quality_colors = {"strong": "green", "solid": "yellow", "needs_work": "dark_orange", "weak": "red"}
        timing_colors = {"on_beat": "green", "slightly_off": "yellow", "off_beat": "red"}

        for pd in scores.pattern_details:
            q = pd.get("quality")
            t = pd.get("timing")
            qc = quality_colors.get(q or "", "dim")
            tc = timing_colors.get(t or "", "dim")
            q_str = q.replace("_", " ") if q else "?"
            t_str = t.replace("_", " ") if t else "?"
            name = pd.get("name", "?")
            count = scores.pattern_counts.get(name, 1)
            count_str = f"[cyan]{count}×[/cyan]" if count > 1 else f"[dim]{count}×[/dim]"
            pat_table.add_row(
                name,
                count_str,
                f"[{qc}]{q_str}[/{qc}]",
                f"[{tc}]{t_str}[/{tc}]",
                pd.get("notes", ""),
            )

        console.print(pat_table)
        console.print()
    elif scores.all_patterns:
        console.print(f"  [bold]Patterns identified:[/bold] {', '.join(scores.all_patterns)}")
        console.print()

    # Pattern timeline — shown only when at least two time windows had
    # patterns, otherwise it's redundant with the pattern analysis table.
    if len(scores.pattern_timeline) >= 2:
        tl_table = Table(title="Pattern Timeline", show_header=True, header_style="bold cyan")
        tl_table.add_column("Time", justify="center", width=14)
        tl_table.add_column("Patterns", width=60)

        for entry in scores.pattern_timeline:
            start = _format_time(entry["start_time"])
            end = _format_time(entry["end_time"])
            names = entry.get("patterns", [])
            tl_table.add_row(f"{start} \u2192 {end}", "  \u00b7  ".join(names) if names else "\u2014")

        console.print(tl_table)
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

    # Per-category reasoning (chain-of-thought) if present
    if scores.reasoning:
        reason_table = Table(title="Judge's Reasoning", show_header=True, header_style="bold cyan")
        reason_table.add_column("Category", style="bold", width=14)
        reason_table.add_column("Reasoning", width=70)
        for cat in ("timing", "technique", "teamwork", "presentation"):
            text = scores.reasoning.get(cat)
            if text:
                reason_table.add_row(cat.title(), f"[dim]{text}[/dim]")
        console.print(reason_table)
        console.print()

    # Overall impression
    if scores.overall_impression:
        console.print(Panel(scores.overall_impression, title="Judge's Notes", style="dim"))

    # API usage and estimated cost
    usage = scores.usage
    if usage.input_tokens or usage.output_tokens:
        total_tokens = usage.input_tokens + usage.output_tokens
        cost_str = (
            f"~${usage.estimated_cost:.4f}"
            if usage.pricing_known else
            f"~${usage.estimated_cost:.4f} [yellow](unknown model {usage.model})[/yellow]"
        )
        console.print(
            f"\n  [bold]API usage:[/bold] "
            f"{usage.input_tokens:,} in + {usage.output_tokens:,} out "
            f"= {total_tokens:,} tokens"
        )
        console.print(
            f"  [bold]Estimated cost:[/bold] {cost_str}  "
            f"[dim](pricing as of {pricing_updated_on()}; set WCS_PRICING_FILE to override)[/dim]"
        )


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
            "overall_low": scores.overall_low,
            "overall_high": scores.overall_high,
            "grade": scores.grade,
            "low_confidence": scores.low_confidence,
            "timing": scores.timing,
            "timing_low": scores.timing_low,
            "timing_high": scores.timing_high,
            "technique": scores.technique,
            "technique_low": scores.technique_low,
            "technique_high": scores.technique_high,
            "teamwork": scores.teamwork,
            "teamwork_low": scores.teamwork_low,
            "teamwork_high": scores.teamwork_high,
            "presentation": scores.presentation,
            "presentation_low": scores.presentation_low,
            "presentation_high": scores.presentation_high,
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
        "reasoning": scores.reasoning,
        "usage": {
            "input_tokens": scores.usage.input_tokens,
            "output_tokens": scores.usage.output_tokens,
            "estimated_cost_usd": scores.usage.estimated_cost,
            "model": scores.usage.model,
            "pricing_known": scores.usage.pricing_known,
            "pricing_updated_on": pricing_updated_on(),
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


def print_progress_report(
    dancer: str,
    rows: list[HistoryRow],
    trajectories: dict[str, Trajectory],
) -> None:
    """Render a longitudinal trajectory view for one dancer."""
    console.print()
    header = Table.grid(padding=1)
    header.add_column(justify="center")
    header.add_row(Text(f"Progress for {dancer}", style="bold white"))
    header.add_row(Text(f"{len(rows)} runs tracked", style="dim"))
    console.print(Panel(header, style="blue", padding=(1, 2)))

    if not rows:
        console.print("  [yellow]No history for this dancer yet. "
                      "Run `analyze --save-history <name>` first.[/yellow]")
        return

    # Timeline of runs
    timeline = Table(title="Run Timeline", show_header=True, header_style="bold cyan")
    timeline.add_column("#", justify="right", width=4)
    timeline.add_column("Date", style="dim", width=20)
    timeline.add_column("Video", width=22)
    timeline.add_column("Provider", width=12)
    timeline.add_column("Overall", justify="center", width=9)
    timeline.add_column("Grade", justify="center", width=6)
    timeline.add_column("Trend", justify="center", width=8)
    timeline.add_column("Cost", justify="right", width=10)

    prev = None
    for i, row in enumerate(rows, 1):
        color = _score_color(row.overall)
        trend_str = ""
        if prev is not None:
            delta = row.overall - prev
            if delta > 0.3:
                trend_str = f"[green]+{delta:.1f}\u2191[/green]"
            elif delta < -0.3:
                trend_str = f"[red]{delta:.1f}\u2193[/red]"
            else:
                trend_str = "[dim]\u2192[/dim]"
        prev = row.overall
        cost_str = f"${row.estimated_cost:.4f}" if row.estimated_cost > 0 else "[dim]—[/dim]"
        timeline.add_row(
            str(i),
            row.created_at.replace("T", " ").split("+")[0],
            row.video_name[:20] + ("…" if len(row.video_name) > 20 else ""),
            row.provider,
            f"[{color}]{row.overall:.1f}[/{color}]",
            f"[{color}]{row.grade}[/{color}]",
            trend_str,
            cost_str,
        )
    console.print(timeline)

    total_spend = sum(r.estimated_cost for r in rows)
    if total_spend > 0:
        console.print(
            f"  [bold]Total spent on {dancer}:[/bold] "
            f"[yellow]~${total_spend:.4f}[/yellow] across {len(rows)} runs"
        )
    console.print()

    # Trajectory fits
    traj_table = Table(title="Linear Trajectories", show_header=True, header_style="bold cyan")
    traj_table.add_column("Category", style="bold", width=14)
    traj_table.add_column("First", justify="center", width=8)
    traj_table.add_column("Latest", justify="center", width=8)
    traj_table.add_column("Total \u0394", justify="center", width=10)
    traj_table.add_column("Per-run \u0394", justify="center", width=11)
    traj_table.add_column("Direction", width=14)

    for cat in ("overall", "timing", "technique", "teamwork", "presentation"):
        t = trajectories[cat]
        total_delta = t.latest - t.first
        if t.n < 2:
            direction = "[dim]need \u2265 2 runs[/dim]"
        elif t.slope_per_run > 0.2:
            direction = "[green]improving[/green]"
        elif t.slope_per_run < -0.2:
            direction = "[red]regressing[/red]"
        else:
            direction = "[yellow]flat[/yellow]"
        total_color = "green" if total_delta > 0.3 else "red" if total_delta < -0.3 else "dim"
        traj_table.add_row(
            cat.title(),
            f"{t.first:.1f}",
            f"{t.latest:.1f}",
            f"[{total_color}]{total_delta:+.1f}[/{total_color}]",
            f"{t.slope_per_run:+.2f}",
            direction,
        )
    console.print(traj_table)
    console.print()


def print_ensemble_report(ensemble: EnsembleScores, video_name: str) -> None:
    """Render a side-by-side multi-provider ensemble report."""
    console.print()
    header = Table.grid(padding=1)
    header.add_column(justify="center")
    header.add_row(Text("WCS Ensemble Analysis Report", style="bold white"))
    header.add_row(Text(f"{video_name} — {', '.join(ensemble.providers)}", style="dim"))
    console.print(Panel(header, style="blue", padding=(1, 2)))

    # Consensus overall
    color = _score_color(ensemble.overall)
    overall_text = Text()
    overall_text.append("  Consensus Score: ", style="bold")
    overall_text.append(f"{ensemble.overall} / 10", style=f"bold {color}")
    overall_text.append(f"  ({ensemble.grade})", style=f"bold {color}")
    if ensemble.contested:
        overall_text.append(
            f"  \u26a0 contested: {', '.join(ensemble.contested)}",
            style="bold yellow",
        )
    console.print(Panel(overall_text, style=color))

    # Per-category comparison
    table = Table(title="Category Scores by Provider", show_header=True, header_style="bold cyan")
    table.add_column("Category", style="bold", width=18)
    for p in ensemble.providers:
        table.add_column(p, justify="center", width=12)
    table.add_column("Consensus", justify="center", width=12)
    table.add_column("StdDev", justify="center", width=10)

    categories = [
        ("Timing & Rhythm", "timing", ensemble.timing),
        ("Technique", "technique", ensemble.technique),
        ("Teamwork", "teamwork", ensemble.teamwork),
        ("Presentation", "presentation", ensemble.presentation),
    ]

    for label, key, consensus in categories:
        row = [label]
        for p in ensemble.providers:
            v = getattr(ensemble.per_provider[p], key)
            row.append(f"[{_score_color(v)}]{v}[/{_score_color(v)}]")
        row.append(f"[{_score_color(consensus)}]{consensus}[/{_score_color(consensus)}]")
        sd = ensemble.stddev.get(key, 0.0)
        sd_color = "yellow" if key in ensemble.contested else "dim"
        row.append(f"[{sd_color}]{sd:.2f}[/{sd_color}]")
        table.add_row(*row)

    console.print(table)
    console.print()

    # Technique sub-scores consensus
    tech_table = Table(title="Technique Breakdown (consensus)", show_header=True, header_style="bold cyan")
    tech_table.add_column("Area", style="bold", width=12)
    tech_table.add_column("Score", justify="center", width=8)
    tech_table.add_column("", width=12)
    for name, score in [
        ("Posture", ensemble.posture),
        ("Extension", ensemble.extension),
        ("Footwork", ensemble.footwork),
        ("Slot", ensemble.slot),
    ]:
        c = _score_color(score)
        tech_table.add_row(name, f"[{c}]{score}[/{c}]", f"[{c}]{_score_bar(score)}[/{c}]")
    console.print(tech_table)
    console.print()

    if ensemble.contested:
        console.print(
            f"  [bold yellow]\u26a0 Contested categories[/bold yellow] — "
            f"stddev > 1.0 means the models disagreed. Review: "
            f"{', '.join(ensemble.contested)}"
        )
        console.print()


def save_ensemble_json(ensemble: EnsembleScores, path: Path) -> None:
    """Save the ensemble report as JSON."""
    data = {
        "providers": ensemble.providers,
        "consensus": {
            "overall": ensemble.overall,
            "grade": ensemble.grade,
            "timing": ensemble.timing,
            "technique": ensemble.technique,
            "teamwork": ensemble.teamwork,
            "presentation": ensemble.presentation,
            "posture": ensemble.posture,
            "extension": ensemble.extension,
            "footwork": ensemble.footwork,
            "slot": ensemble.slot,
        },
        "stddev": ensemble.stddev,
        "contested": ensemble.contested,
        "per_provider": {
            name: {
                "overall": fs.overall,
                "grade": fs.grade,
                "timing": fs.timing,
                "technique": fs.technique,
                "teamwork": fs.teamwork,
                "presentation": fs.presentation,
            }
            for name, fs in ensemble.per_provider.items()
        },
    }
    path.write_text(json.dumps(data, indent=2))


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
