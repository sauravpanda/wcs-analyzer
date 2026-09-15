"""Render a CoachReport as Markdown and as a self-contained HTML page.

The layout mirrors a hand-written judge's review: overall impression, the
themes a judge would notice, a music/phrasing check, a table of timestamped
notes each with a strip of frames, then slow-motion count-by-count detail.
"""

from __future__ import annotations

import base64
import html
import subprocess
import tempfile
from pathlib import Path

from .coach import CoachReport, fmt_time, load_report
from .pricing import pricing_updated_on

_KIND_LABEL = {"refine": "refine", "keep": "keep it", "question": "check"}
# Markdown cannot color text; use a consistent glyph per verdict instead.
_KIND_GLYPH = {"keep": "🟢", "refine": "🔴", "question": "🔵"}
_KIND_HEX = {"keep": "#2e7d32", "refine": "#c62828", "question": "#1565c0"}
_KIND_BG = {"keep": "#e8f5e9", "refine": "#fdecea", "question": "#e8f0fe"}
LEGEND_MD = "🟢 keep doing this · 🔴 refine this · 🔵 check this yourself (the model was unsure)"


def _img_md(rel: str) -> str:
    return f"![frames]({rel})" if rel else ""


def write_markdown(report: CoachReport, out_dir: Path) -> Path:
    L: list[str] = []
    L.append(f"# Coaching notes: {report.video_name}")
    L.append("")
    L.append(
        f"_Model {report.model} via Claude Code · {report.duration:.0f}s clip · survey {report.survey_fps:.1f} fps"
        + (f" · zoom {report.zoom_fps:g} fps" if report.zoom_fps else "")
        + f" · {len(report.notes)} notes · {len(report.zooms)} slow-motion looks_"
    )
    L.append("")
    for w in report.warnings:
        L.append(f"> ⚠ {w}")
    if report.warnings:
        L.append("")
    L.append(f"_{LEGEND_MD}_")
    L.append("")

    f = report.focus or {}
    if f:
        conf = f.get("confidence")
        L.append("## Who I watched")
        L.append("")
        L.append(
            f"{f.get('description') or 'not stated'}"
            + (f" (confidence {float(conf):.0%})" if isinstance(conf, (int, float)) else "")
            + (f". Blocked: {', '.join(map(str, f.get('occluded') or []))}" if f.get("occluded") else "")
        )
        L.append("")

    L.append("## Overall impression")
    L.append("")
    if report.summary:
        L.append(report.summary)
        L.append("")
    if report.doing_well:
        L.append("**🟢 Doing well**")
        L.append("")
        L += [f"- 🟢 {x}" for x in report.doing_well]
        L.append("")
    if report.work_on:
        L.append(f"**🔴 Work on:** {report.work_on}")
        L.append("")

    if report.themes:
        L.append("## Themes a judge would notice")
        L.append("")
        for i, t in enumerate(report.themes, 1):
            ex = ", ".join(fmt_time(x) for x in t.get("examples") or [])
            L.append(f"{i}. **{t.get('title')}**" + (f" (at {ex})" if ex else ""))
            if t.get("detail"):
                L.append(f"   {t['detail']}")
        L.append("")

    m = report.music or {}
    if m.get("bpm") or report.phrases:
        L.append("## Music and phrasing")
        L.append("")
        if m.get("bpm"):
            L.append(
                f"Tempo {m['bpm']:.0f} BPM; music starts around {fmt_time(m['music_start'])}."
                if m.get("music_start") is not None else f"Tempo {m['bpm']:.0f} BPM."
            )
            L.append("")
        if report.phrases:
            L.append("| Phrase change | Acknowledged | What happened |")
            L.append("|---|---|---|")
            for p in report.phrases:
                mark = "⚪ not judged" if p["acknowledged"] is None else ("🟢 yes" if p["acknowledged"] else "🔴 **no**")
                L.append(f"| {fmt_time(p['time'])} | {mark} | {p.get('how', '')} |")
            L.append("")

    if report.notes:
        L.append("## Notes, every few seconds")
        L.append("")
        L.append("| Time | Frames | Note |")
        L.append("|---|---|---|")
        for n in report.notes:
            span = fmt_time(n.time) + (f"–{fmt_time(n.end_time)}" if n.end_time else "")
            tag = _KIND_LABEL.get(n.kind, n.kind)
            glyph = _KIND_GLYPH.get(n.kind, "")
            counts = f" _({n.counts})_" if n.counts else ""
            L.append(f"| {glyph} {span}<br>`{tag}` | {_img_md(('strips/' + n.strip) if n.strip else '')} | {n.note}{counts} |")
        L.append("")

    if report.zooms:
        L.append("## Slow motion, count by count")
        L.append("")
        for z in report.zooms:
            L.append(f"### {_KIND_GLYPH.get(z.kind, '')} {fmt_time(z.time)}: {z.reason}")
            L.append("")
            if z.strip:
                L.append(_img_md("strips/" + z.strip))
                L.append("")
            if z.count_notes:
                L.append("| Time | Count | Observation |")
                L.append("|---|---|---|")
                for c in z.count_notes:
                    L.append(f"| {fmt_time(float(c.get('time', z.time) or z.time))} | {c.get('count', '?')} | {c.get('observation', '')} |")
                L.append("")
            if z.diagnosis:
                L.append(f"**🔴 What happened:** {z.diagnosis}")
                L.append("")
            if z.fix:
                L.append(f"**🟢 Fix:** {z.fix}")
                L.append("")
            meta = []
            if z.visibility:
                meta.append(f"visibility {z.visibility}")
            if z.confidence:
                meta.append(f"confidence {z.confidence:.0%}")
            if meta:
                L.append(f"_{'; '.join(meta)}_")
                L.append("")

    if report.patterns:
        L.append("## Patterns seen")
        L.append("")
        L.append(", ".join(report.patterns))
        L.append("")

    u = report.usage
    if u.input_tokens or u.output_tokens or u.estimated_cost:
        L.append("---")
        L.append(
            f"_API usage: {u.input_tokens:,} in + {u.output_tokens:,} out; estimated cost "
            f"${u.estimated_cost:.2f} (pricing as of {pricing_updated_on()})._"
        )
        L.append("")

    path = out_dir / "coach_report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def _data_uri(path: Path) -> str:
    try:
        return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return ""


_CSS = (
    "body{font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1100px;"
    "margin:32px auto;padding:0 20px;color:#1a1a1a;background:#fff}h1{font-size:26px}h2{margin-top:36px;"
    "border-bottom:1px solid #ddd;padding-bottom:4px}table{border-collapse:collapse;width:100%}"
    "td,th{border-top:1px solid #e5e5e5;padding:10px 8px;vertical-align:top;text-align:left}"
    "img{max-width:100%;height:auto;border-radius:4px}.meta{color:#666;font-size:13px}"
    ".warn{background:#fff7d6;border-left:4px solid #e0b000;padding:8px 12px;margin:8px 0}"
    ".tag{display:inline-block;font-size:11px;padding:2px 8px;border-radius:3px;color:#fff;font-weight:600}"
    ".tag.keep{background:#2e7d32}.tag.refine{background:#c62828}.tag.question{background:#1565c0}"
    ".time{white-space:nowrap;font-weight:600}"
    "tr.row td{border-left:6px solid transparent}"
    "tr.row.keep td:first-child{border-left-color:#2e7d32}tr.row.keep{background:#f3faf3}"
    "tr.row.refine td:first-child{border-left-color:#c62828}tr.row.refine{background:#fff6f5}"
    "tr.row.question td:first-child{border-left-color:#1565c0}tr.row.question{background:#f4f7fe}"
    "tr.ok{background:#f3faf3}tr.ok td.verdict{color:#2e7d32;font-weight:700}"
    "tr.miss{background:#fff6f5}tr.miss td.verdict{color:#c62828;font-weight:700}"
    "tr.unjudged td{color:#777}tr.unjudged td.verdict{font-style:italic}"
    ".box{border-left:6px solid;padding:10px 14px;margin:10px 0;border-radius:4px}"
    ".box.good{border-color:#2e7d32;background:#f3faf3}.box.bad{border-color:#c62828;background:#fff6f5}"
    ".box.check{border-color:#1565c0;background:#f4f7fe}"
    ".legend span{display:inline-block;margin-right:14px}.legend i{display:inline-block;width:12px;height:12px;"
    "border-radius:2px;margin-right:6px;vertical-align:-1px}"
    ".zoom{margin:18px 0 28px;padding:12px 16px;border-radius:6px;border:1px solid #e5e5e5}"
    ".zoom.keep{border-color:#2e7d32}.zoom.refine{border-color:#c62828}.zoom.question{border-color:#1565c0}"
    ".zoom h3{margin-top:0}.zoom h3 .tag{margin-right:8px}"
    "video{max-width:100%;border-radius:4px;margin:6px 0;display:block}"
    "ol.toc li{margin:4px 0}section{scroll-margin-top:12px}"
)


def _document(title: str, body: list[str]) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>"
        + "".join(body) + "</body></html>"
    )


def _html_sections(
    report: CoachReport, out_dir: Path, *, heading: str | None = None, anchor: str = "top",
    clips: dict[str, str] | None = None,
) -> list[str]:
    """The HTML for one report, from its heading to its usage footer, with strips inline.

    `clips` maps a zoom strip filename to a video data URI to embed next to that strip.
    """
    e = html.escape
    strips = out_dir / "strips"

    def img(rel: str) -> str:
        if not rel:
            return ""
        uri = _data_uri(strips / rel)
        return f'<img src="{uri}" alt="frames">' if uri else ""

    H: list[str] = []
    H.append(f"<h1 id='{e(anchor)}'>{e(heading or 'Coaching notes: ' + report.video_name)}</h1>")
    H.append(
        f"<p class='meta'>Model {e(report.model)} via Claude Code · {report.duration:.0f}s clip · "
        f"survey {report.survey_fps:.1f} fps"
        + (f" · zoom {report.zoom_fps:g} fps" if report.zoom_fps else "")
        + f" · {len(report.notes)} notes · {len(report.zooms)} slow-motion looks</p>"
    )
    for w in report.warnings:
        H.append(f"<div class='warn'>⚠ {e(w)}</div>")
    H.append(
        "<p class='legend meta'>"
        "<span><i style='background:#2e7d32'></i>keep doing this</span>"
        "<span><i style='background:#c62828'></i>refine this</span>"
        "<span><i style='background:#1565c0'></i>check this yourself (the model was unsure)</span></p>"
    )

    f = report.focus or {}
    if f:
        conf = f.get("confidence")
        H.append("<h2>Who I watched</h2><p>" + e(str(f.get("description") or "not stated")))
        if isinstance(conf, (int, float)):
            H.append(f" <span class='meta'>(confidence {float(conf):.0%})</span>")
        if f.get("occluded"):
            H.append(f"<br><span class='meta'>Blocked: {e(', '.join(map(str, f['occluded'])))}</span>")
        H.append("</p>")

    H.append("<h2>Overall impression</h2>")
    if report.summary:
        H.append(f"<p>{e(report.summary)}</p>")
    if report.doing_well:
        H.append("<div class='box good'><strong>Doing well</strong><ul>"
                 + "".join(f"<li>{e(x)}</li>" for x in report.doing_well) + "</ul></div>")
    if report.work_on:
        H.append(f"<div class='box bad'><strong>Work on:</strong> {e(report.work_on)}</div>")

    if report.themes:
        H.append("<h2>Themes a judge would notice</h2><ol>")
        for t in report.themes:
            ex = ", ".join(fmt_time(x) for x in t.get("examples") or [])
            H.append(f"<li><strong>{e(str(t.get('title') or ''))}</strong>"
                     + (f" <span class='meta'>(at {e(ex)})</span>" if ex else "")
                     + (f"<br>{e(str(t.get('detail') or ''))}" if t.get("detail") else "") + "</li>")
        H.append("</ol>")

    m = report.music or {}
    if m.get("bpm") or report.phrases:
        H.append("<h2>Music and phrasing</h2>")
        if m.get("bpm"):
            start = f"; music starts around {fmt_time(m['music_start'])}" if m.get("music_start") is not None else ""
            H.append(f"<p>Tempo {m['bpm']:.0f} BPM{start}.</p>")
        song_map = out_dir / "song_map.svg"
        if song_map.exists():
            H.append("<p class='meta'>Song map from the audio: energy in amber, novelty in blue, ticks on count 1 of "
                     "each 8, phrase boundaries labelled with the counts of the section that ends there. Confidence "
                     "is heuristic; check the boundaries by ear.</p>")
            H.append(f"<div style='overflow-x:auto'>{song_map.read_text()}</div>")
        pmap = m.get("phrase_map") or {}
        kinds = {round(b["time"], 1): b for b in pmap.get("boundaries", [])}
        if report.phrases:
            H.append("<table><tr><th>Phrase change</th><th>Section</th><th>Acknowledged</th><th>What happened</th></tr>")
            for p in report.phrases:
                if p["acknowledged"] is None:
                    cls, ack = "unjudged", "not judged"
                else:
                    cls = "ok" if p["acknowledged"] else "miss"
                    ack = "yes" if p["acknowledged"] else "missed"
                b = kinds.get(round(float(p["time"]), 1))
                section = (f"{b['counts']} counts · {b['kind']}" if b and b["kind"] != "start" else ("first phrase" if b else "32-count grid"))
                H.append(f"<tr class='{cls}'><td class='time'>{fmt_time(p['time'])}</td><td class='meta'>{e(section)}</td>"
                         f"<td class='verdict'>{ack}</td><td>{e(str(p.get('how') or ''))}</td></tr>")
            H.append("</table>")

    if report.notes:
        H.append("<h2>Notes, every few seconds</h2><table><tr><th>Time</th><th>Frames</th><th>Note</th></tr>")
        for n in report.notes:
            span = fmt_time(n.time) + (f"–{fmt_time(n.end_time)}" if n.end_time else "")
            tag = _KIND_LABEL.get(n.kind, n.kind)
            counts = f" <span class='meta'>({e(n.counts)})</span>" if n.counts else ""
            H.append(
                f"<tr class='row {e(n.kind)}'><td class='time'>{span}<br><span class='tag {e(n.kind)}'>{e(tag)}</span></td>"
                f"<td style='width:46%'>{img(n.strip)}</td><td>{e(n.note)}{counts}</td></tr>"
            )
        H.append("</table>")

    if report.zooms:
        H.append("<h2>Slow motion, count by count</h2>")
        for z in report.zooms:
            ztag = _KIND_LABEL.get(z.kind, z.kind)
            H.append(f"<div class='zoom {e(z.kind)}'><h3><span class='tag {e(z.kind)}'>{e(ztag)}</span>{fmt_time(z.time)}: {e(z.reason)}</h3>")
            if z.strip:
                H.append(img(z.strip))
                if clips and z.strip in clips:
                    H.append(f"<video controls loop playsinline preload='metadata' src='{clips[z.strip]}'></video>")
            if z.count_notes:
                H.append("<table><tr><th>Time</th><th>Count</th><th>Observation</th></tr>")
                for c in z.count_notes:
                    try:
                        ct = float(c.get("time", z.time) or z.time)
                    except (TypeError, ValueError):
                        ct = z.time
                    H.append(f"<tr><td class='time'>{fmt_time(ct)}</td><td>{e(str(c.get('count', '?')))}</td><td>{e(str(c.get('observation') or ''))}</td></tr>")
                H.append("</table>")
            if z.diagnosis:
                H.append(f"<div class='box bad'><strong>What happened:</strong> {e(z.diagnosis)}</div>")
            if z.fix:
                H.append(f"<div class='box good'><strong>Fix:</strong> {e(z.fix)}</div>")
            meta = []
            if z.visibility:
                meta.append(f"visibility {e(z.visibility)}")
            if z.confidence:
                meta.append(f"confidence {z.confidence:.0%}")
            if meta:
                H.append(f"<p class='meta'>{'; '.join(meta)}</p>")
            H.append("</div>")

    if report.patterns:
        H.append("<h2>Patterns seen</h2><p>" + e(", ".join(report.patterns)) + "</p>")

    u = report.usage
    if u.input_tokens or u.output_tokens or u.estimated_cost:
        H.append(
            f"<hr><p class='meta'>API usage: {u.input_tokens:,} in + {u.output_tokens:,} out; "
            f"estimated cost ${u.estimated_cost:.2f} (pricing as of {e(pricing_updated_on())}).</p>"
        )
    return H


def write_html(report: CoachReport, out_dir: Path) -> Path:
    path = out_dir / "coach_report.html"
    path.write_text(_document(f"Coaching notes: {report.video_name}", _html_sections(report, out_dir)), encoding="utf-8")
    return path


def _video_snippet_uri(video: Path, start: float, end: float, height: int = 480) -> str | None:
    """A small H.264 clip of [start, end] as a data URI, or None when ffmpeg cannot make one."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cmd = [
            "ffmpeg", "-loglevel", "error", "-y", "-ss", f"{max(0.0, start):.2f}", "-t", f"{max(0.1, end - start):.2f}",
            "-i", str(video), "-vf", f"scale=-2:{height}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", str(tmp_path),
        ]
        try:
            ok = subprocess.run(cmd, capture_output=True, timeout=120).returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        if not ok or tmp_path.stat().st_size == 0:
            return None
        return "data:video/mp4;base64," + base64.b64encode(tmp_path.read_bytes()).decode()
    finally:
        tmp_path.unlink(missing_ok=True)


BUNDLE_INTRO = (
    "These notes were produced by an AI coach (Claude Opus 5) that watched the footage frame by frame, "
    "with the tempo and phrase changes taken from the audio. Green means keep doing this, red means refine "
    "this, blue means the model was unsure and it should be checked by eye. Every note carries a strip of "
    "frames from that moment; the slow-motion sections go count by count and end with one drill. Treat "
    "timings as accurate to about a beat and the confidence figures as rough."
)


def _pretty_name(video_name: str) -> str:
    stem = Path(video_name).stem
    if "_" in stem:
        base, _, n = stem.rpartition("_")
        return f"{base.replace('-', ' ')} · song {n}"
    return stem.replace("-", " ")


def write_bundle(
    dirs: list[Path], out_path: Path, *, title: str = "Coaching notes", intro: str | None = None,
    labels: list[str] | None = None, clips: bool = False, videos_dir: Path | None = None, clip_window: float = 3.0,
) -> Path:
    """One self-contained HTML file for several coached songs: intro, contents, then each report in full.

    Frame strips are always inline, so the file stands on its own. With clips=True a short
    video snippet around every slow-motion moment is embedded next to its strip (needs ffmpeg
    and the original videos, looked up in videos_dir or two levels above each report folder).
    """
    e = html.escape
    reports = [(d, load_report(d)) for d in dirs]
    names = list(labels) if labels else [_pretty_name(r.video_name) for _, r in reports]
    body: list[str] = [
        f"<h1 id='top'>{e(title)}</h1>",
        f"<p>{e(intro if intro is not None else BUNDLE_INTRO)}</p>",
        "<p class='legend meta'><span><i style='background:#2e7d32'></i>keep doing this</span>"
        "<span><i style='background:#c62828'></i>refine this</span>"
        "<span><i style='background:#1565c0'></i>check this yourself</span></p>",
        "<h2>Contents</h2><ol class='toc'>",
    ]
    for i, ((_, r), name) in enumerate(zip(reports, names), 1):
        keep = sum(1 for n in r.notes if n.kind == "keep")
        refine = sum(1 for n in r.notes if n.kind == "refine")
        body.append(
            f"<li><a href='#song-{i}'>{e(name)}</a> <span class='meta'>· {len(r.notes)} notes "
            f"({keep} keep, {refine} refine) · {len(r.zooms)} slow-motion looks</span></li>"
        )
    body.append("</ol>")
    for i, ((d, r), name) in enumerate(zip(reports, names), 1):
        clip_uris: dict[str, str] = {}
        if clips:
            video = (videos_dir or d.parent.parent) / r.video_name
            if video.exists():
                for z in r.zooms:
                    if z.strip:
                        uri = _video_snippet_uri(video, z.time - clip_window / 2, z.time + clip_window / 2)
                        if uri:
                            clip_uris[z.strip] = uri
        body.append("<hr style='margin:48px 0'><section>")
        body += _html_sections(r, d, heading=name, anchor=f"song-{i}", clips=clip_uris)
        body.append("<p class='meta'><a href='#top'>Back to contents</a></p></section>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_document(title, body), encoding="utf-8")
    return out_path


def write_reports(report: CoachReport, out_dir: Path) -> tuple[Path, Path]:
    """Write both formats; return (markdown_path, html_path)."""
    return write_markdown(report, out_dir), write_html(report, out_dir)
