"""CLI report formatting with rich."""

import csv
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

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


def _score_line(label: str, score: float, weight: str = "", lo: float = 0.0, hi: float = 0.0, notes: str = "") -> str:
    """Format a single score line with bar, optional CI, weight, and notes."""
    c = _score_color(score)
    parts = [f"    [bold]{label:<20}[/bold]", f"[{c}]{score:>4.1f}[/{c}]", f"[{c}]{_score_bar(score)}[/{c}]"]
    ci = _ci_str(lo, hi)
    if ci:
        parts.append(f"[dim]{ci}[/dim]")
    if weight:
        parts.append(f"[dim]{weight}[/dim]")
    line = "  ".join(parts)
    if notes:
        line += f"\n{'':>30}[dim]{notes}[/dim]"
    return line


def print_report(scores: FinalScores, video_name: str) -> None:
    """Print the full analysis report to the terminal.

    Uses plain Rich-formatted text instead of ASCII tables so content
    never gets swallowed by column-width miscalculations. Every line
    prints left-to-right and wraps naturally.
    """
    console.print()
    console.print(Panel(
        f"[bold white]WCS Dance Analysis Report[/bold white]\n[dim]{video_name}[/dim]",
        style="blue", padding=(1, 2),
    ))

    # Overall score
    color = _score_color(scores.overall)
    ci = _ci_str(scores.overall_low, scores.overall_high)
    ci_part = f"  [dim]{ci}[/dim]" if ci else ""
    warn = "  [bold yellow]\u26a0 low confidence[/bold yellow]" if scores.low_confidence else ""
    console.print(Panel(
        f"  [bold]Overall Score:[/bold] [{color}]{scores.overall} / 10[/{color}]"
        f"{ci_part}  [{color}]({scores.grade})[/{color}]{warn}",
        style=color,
    ))

    # Category scores
    console.print("\n  [bold cyan]Category Scores[/bold cyan]")
    for name, score, lo, hi, weight in [
        ("Timing & Rhythm", scores.timing, scores.timing_low, scores.timing_high, "30%"),
        ("Technique", scores.technique, scores.technique_low, scores.technique_high, "30%"),
        ("Teamwork", scores.teamwork, scores.teamwork_low, scores.teamwork_high, "20%"),
        ("Presentation", scores.presentation, scores.presentation_low, scores.presentation_high, "20%"),
    ]:
        console.print(_score_line(name, score, weight, lo, hi))
    console.print()

    # Technique breakdown
    tech_raw: dict = {}
    for seg in scores.segments:
        if seg.raw_data and "technique" in seg.raw_data:
            tech_raw = seg.raw_data["technique"]
            break
    console.print("  [bold cyan]Technique Breakdown[/bold cyan]")
    for area, score, note_key in [
        ("Posture", scores.posture, "posture"),
        ("Extension", scores.extension, "extension"),
        ("Footwork", scores.footwork, "footwork"),
        ("Slot", scores.slot, "slot"),
    ]:
        notes = tech_raw.get(note_key, {}).get("notes", "") if isinstance(tech_raw.get(note_key), dict) else ""
        console.print(_score_line(area, score, notes=notes))
    console.print()

    # Partner breakdown
    if scores.lead_technique > 0 or scores.follow_technique > 0:
        console.print("  [bold cyan]Partner Breakdown[/bold cyan]")
        lc = _score_color(scores.lead_technique)
        fc = _score_color(scores.follow_technique)
        lpc = _score_color(scores.lead_presentation)
        fpc = _score_color(scores.follow_presentation)
        console.print(
            f"    [bold]Lead[/bold]   — Technique: [{lc}]{scores.lead_technique}[/{lc}]"
            f"  Presentation: [{lpc}]{scores.lead_presentation}[/{lpc}]"
        )
        console.print(
            f"    [bold]Follow[/bold] — Technique: [{fc}]{scores.follow_technique}[/{fc}]"
            f"  Presentation: [{fpc}]{scores.follow_presentation}[/{fpc}]"
        )
        if scores.lead_notes:
            console.print(f"    [dim]Lead:[/dim] {scores.lead_notes}")
        if scores.follow_notes:
            console.print(f"    [dim]Follow:[/dim] {scores.follow_notes}")
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

    # Patterns
    quality_colors = {"strong": "green", "solid": "yellow", "needs_work": "dark_orange", "weak": "red"}
    timing_colors = {"on_beat": "green", "slightly_off": "yellow", "off_beat": "red"}
    if scores.pattern_details:
        n_unique = len(scores.pattern_details)
        n_total = sum(scores.pattern_counts.values()) if scores.pattern_counts else n_unique
        console.print(f"  [bold cyan]Patterns[/bold cyan] ({n_unique} unique, {n_total} total)")
        for pd in scores.pattern_details:
            name = pd.get("name", "?")
            q = pd.get("quality")
            t = pd.get("timing")
            qc = quality_colors.get(q or "", "dim")
            tc = timing_colors.get(t or "", "dim")
            q_str = q.replace("_", " ") if q else "?"
            t_str = t.replace("_", " ") if t else "?"
            count = scores.pattern_counts.get(name, 1)
            count_str = f"[cyan]{count}\u00d7[/cyan]" if count > 1 else "[dim]1\u00d7[/dim]"
            notes = pd.get("notes", "")
            console.print(
                f"    [bold]{name:<28}[/bold] {count_str}  [{qc}]{q_str:<11}[/{qc}]"
                f"  [{tc}]{t_str:<13}[/{tc}]"
            )
            if notes:
                console.print(f"{'':>6}[dim]{notes}[/dim]")
        console.print()
    elif scores.all_patterns:
        console.print(f"  [bold]Patterns:[/bold] {', '.join(scores.all_patterns)}\n")

    # Pattern timeline
    if len(scores.pattern_timeline) >= 2:
        console.print("  [bold cyan]Pattern Timeline[/bold cyan]")
        for entry in scores.pattern_timeline:
            start = _format_time(entry["start_time"])
            end = _format_time(entry["end_time"])
            names = entry.get("patterns", [])
            pat_str = ", ".join(names) if names else "\u2014"
            console.print(f"    [dim]{start} \u2192 {end}[/dim]  {pat_str}")
        console.print()

    # Strengths
    if scores.top_strengths:
        console.print("  [bold green]Strengths[/bold green]")
        for s in scores.top_strengths:
            console.print(f"    [green]\u2022[/green] {s}")
        console.print()

    # Improvements
    if scores.top_improvements:
        console.print("  [bold yellow]Areas to Improve[/bold yellow]")
        for s in scores.top_improvements:
            console.print(f"    [yellow]\u2022[/yellow] {s}")
        console.print()

    # Reasoning
    if scores.reasoning:
        console.print("  [bold cyan]Judge's Reasoning[/bold cyan]")
        for cat in ("timing", "technique", "teamwork", "presentation"):
            text = scores.reasoning.get(cat)
            if text:
                console.print(f"    [bold]{cat.title():<14}[/bold] [dim]{text}[/dim]")
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
        "metadata": {
            "video_recorded_at": scores.video_recorded_at,
            "competition": scores.competition,
            "comp_date": scores.comp_date,
            "comp_mode": scores.comp_mode,
            "comp_stage": scores.comp_stage,
        },
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
        "pattern_counts": scores.pattern_counts,
        "pattern_timeline": scores.pattern_timeline,
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
    console.print(Panel(
        f"[bold white]Progress for {dancer}[/bold white]\n[dim]{len(rows)} runs tracked[/dim]",
        style="blue", padding=(1, 2),
    ))

    if not rows:
        console.print("  [yellow]No history for this dancer yet. "
                      "Run `analyze --save-history <name>` first.[/yellow]")
        return

    # Timeline of runs
    console.print("\n  [bold cyan]Run Timeline[/bold cyan]")
    prev = None
    for i, row in enumerate(rows, 1):
        color = _score_color(row.overall)
        trend_str = ""
        if prev is not None:
            delta = row.overall - prev
            if delta > 0.3:
                trend_str = f" [green]+{delta:.1f}\u2191[/green]"
            elif delta < -0.3:
                trend_str = f" [red]{delta:.1f}\u2193[/red]"
            else:
                trend_str = " [dim]\u2192[/dim]"
        prev = row.overall
        cost_str = f"  ${row.estimated_cost:.4f}" if row.estimated_cost > 0 else ""
        date = row.created_at.replace("T", " ").split("+")[0]
        vid = row.video_name[:22] + ("\u2026" if len(row.video_name) > 22 else "")
        comp_parts = []
        if row.competition:
            comp_parts.append(row.competition)
        if row.comp_mode:
            comp_parts.append(row.comp_mode)
        if row.comp_stage:
            comp_parts.append(row.comp_stage)
        comp_str = f"  [cyan]{'  '.join(comp_parts)}[/cyan]" if comp_parts else ""
        console.print(
            f"    [dim]{i:>2}.[/dim] [{color}]{row.overall:>4.1f} ({row.grade})[/{color}]"
            f"{trend_str}  [dim]{date}[/dim]  {vid}  [dim]{row.provider}{cost_str}[/dim]{comp_str}"
        )

    total_spend = sum(r.estimated_cost for r in rows)
    if total_spend > 0:
        console.print(
            f"\n  [bold]Total spent on {dancer}:[/bold] "
            f"[yellow]~${total_spend:.4f}[/yellow] across {len(rows)} runs"
        )
    console.print()

    # Trajectory fits
    console.print("  [bold cyan]Linear Trajectories[/bold cyan]")
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
        tc = "green" if total_delta > 0.3 else "red" if total_delta < -0.3 else "dim"
        console.print(
            f"    [bold]{cat.title():<14}[/bold] {t.first:.1f} \u2192 {t.latest:.1f}"
            f"  [{tc}]{total_delta:+.1f}[/{tc}]  ({t.slope_per_run:+.2f}/run)  {direction}"
        )
    console.print()


def print_ensemble_report(ensemble: EnsembleScores, video_name: str) -> None:
    """Render a multi-provider ensemble report using plain text."""
    console.print()
    console.print(Panel(
        f"[bold white]WCS Ensemble Analysis Report[/bold white]\n"
        f"[dim]{video_name} \u2014 {', '.join(ensemble.providers)}[/dim]",
        style="blue", padding=(1, 2),
    ))

    # Consensus overall
    color = _score_color(ensemble.overall)
    warn = (
        f"  [bold yellow]\u26a0 contested: {', '.join(ensemble.contested)}[/bold yellow]"
        if ensemble.contested else ""
    )
    console.print(Panel(
        f"  [bold]Consensus Score:[/bold] [{color}]{ensemble.overall} / 10  "
        f"({ensemble.grade})[/{color}]{warn}",
        style=color,
    ))

    # Per-category comparison
    console.print("\n  [bold cyan]Category Scores by Provider[/bold cyan]")
    for label, key, consensus in [
        ("Timing & Rhythm", "timing", ensemble.timing),
        ("Technique", "technique", ensemble.technique),
        ("Teamwork", "teamwork", ensemble.teamwork),
        ("Presentation", "presentation", ensemble.presentation),
    ]:
        parts = [f"    [bold]{label:<18}[/bold]"]
        for p in ensemble.providers:
            v = getattr(ensemble.per_provider[p], key)
            c = _score_color(v)
            parts.append(f"{p}: [{c}]{v}[/{c}]")
        cc = _score_color(consensus)
        parts.append(f"\u2192 [{cc}]{consensus}[/{cc}]")
        sd = ensemble.stddev.get(key, 0.0)
        sd_color = "yellow" if key in ensemble.contested else "dim"
        parts.append(f"[{sd_color}](\u03c3 {sd:.2f})[/{sd_color}]")
        console.print("  ".join(parts))
    console.print()

    # Technique consensus
    console.print("  [bold cyan]Technique Breakdown (consensus)[/bold cyan]")
    for name, score in [
        ("Posture", ensemble.posture),
        ("Extension", ensemble.extension),
        ("Footwork", ensemble.footwork),
        ("Slot", ensemble.slot),
    ]:
        console.print(_score_line(name, score))
    console.print()

    if ensemble.contested:
        console.print(
            f"  [bold yellow]\u26a0 Contested categories[/bold yellow] \u2014 "
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
        "[bold white]WCS Score Comparison[/bold white]",
        style="blue", padding=(1, 2),
    ))

    # Overall scores
    console.print("\n  [bold cyan]Overall Scores[/bold cyan]")
    prev_overall = None
    for name, data in reports:
        scores = data.get("scores", {})
        overall = scores.get("overall", 0)
        grade = scores.get("grade", "?")
        color = _score_color(overall)
        trend_str = "  " + _trend(overall, prev_overall) if prev_overall is not None else ""
        console.print(
            f"    [bold]{name:<22}[/bold] [{color}]{overall:>4.1f} ({grade})[/{color}]{trend_str}"
        )
        prev_overall = overall
    console.print()

    # Category breakdown
    console.print("  [bold cyan]Category Comparison[/bold cyan]")
    for cat in _COMPARE_CATEGORIES:
        parts = [f"    [bold]{cat.title():<16}[/bold]"]
        for name, data in reports:
            score = data.get("scores", {}).get(cat, 0)
            c = _score_color(score)
            parts.append(f"{name}: [{c}]{score}[/{c}]")
        console.print("  ".join(parts))
    console.print()

    # Technique sub-scores
    console.print("  [bold cyan]Technique Breakdown[/bold cyan]")
    for area in ["posture", "extension", "footwork", "slot"]:
        parts = [f"    [bold]{area.title():<16}[/bold]"]
        for name, data in reports:
            score = data.get("technique_breakdown", {}).get(area, 0)
            c = _score_color(score)
            parts.append(f"{name}: [{c}]{score}[/{c}]")
        console.print("  ".join(parts))
    console.print()
