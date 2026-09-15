"""Coach mode: judge-style notes with timestamps, frame strips, a phrase check,
and count-by-count zoom on the moments that matter.

Modelled on how an experienced dancer writes feedback for a friend: an overall
impression, the two-to-four themes a judge would notice, a note every few
seconds with a screenshot, and slow-motion detail on the handful of moments
where the footwork or connection went wrong. Two model passes:

1. Survey pass: the whole clip at a few frames per second, with tempo, music
   start, and estimated phrase boundaries from the audio track.
2. Zoom pass: short bursts at ~10 fps around the moments the survey flagged,
   with the beats inside each window, for count-level footwork notes.

Runs through the local Claude Code CLI; Opus 5 by default.
"""

from __future__ import annotations

import base64
import json
import logging
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .audio import AudioFeatures, estimate_phrase_starts, extract_audio_features, load_audio
from .phrases import PhraseBoundary, PhraseMap, analyze_structure, phrase_context, song_map_svg
from .claude_code_analyzer import _call_claude_cli, _check_claude_cli, _cli_budget
from .exceptions import AnalysisError, AudioProcessingError
from .pricing import UsageTotals
from .prompts import (
    COACH_PHRASE_JUDGE_PROMPT,
    COACH_SURVEY_PROMPT,
    COACH_ZOOM_PROMPT,
    DANCER_CONTEXT_TEMPLATE,
    PATTERN_VOCABULARY_INSTRUCTION,
)
from .video import FrameData, extract_frames, extract_frames_between, get_video_duration

logger = logging.getLogger(__name__)

DEFAULT_COACH_MODEL = "claude-opus-5"
# Survey frames are capped so a long clip cannot balloon the CLI call; the
# rate is lowered to fit instead.
# Upper bound on frames in one survey call. Every frame is read into the CLI's
# context as an image, so this is really a context budget: at 360 frames a
# 168 s clip once failed outright and, on retry, the model reported the first
# 55 s of frames as empty (consistent with the context being compacted). 300
# leaves headroom; the zoom pass supplies the fine detail anyway.
MAX_SURVEY_FRAMES = 300
# If the first survey note lands later than this fraction of the clip, the
# model most likely never saw the opening frames.
COVERAGE_GAP_FRACTION = 0.25
STRIP_HEIGHT = 240
# Ignore focus claims below this confidence: the notes may be about the wrong couple.
FOCUS_CONFIDENCE_FLOOR = 0.5

Progress = Callable[[str], None]

# One color per verdict, used for strip borders (BGR) and the HTML report.
# keep = did it right, refine = did it wrong, question = check it yourself.
KIND_COLORS_BGR = {
    "keep": (50, 125, 46),
    "refine": (40, 40, 198),
    "question": (192, 101, 21),
}
STRIP_BORDER = 8


@dataclass
class CoachNote:
    """One timestamped observation from the survey pass."""

    time: float
    note: str
    kind: str = "refine"          # refine | keep | question
    counts: str = ""
    end_time: float | None = None
    strip: str = ""               # path relative to the report folder


@dataclass
class ZoomResult:
    """Count-by-count detail for one flagged moment."""

    time: float
    reason: str
    count_notes: list[dict] = field(default_factory=list)
    diagnosis: str = ""
    fix: str = ""
    confidence: float = 0.0
    visibility: str = ""
    strip: str = ""
    kind: str = "refine"          # verdict of the nearest survey note, drives the color


@dataclass
class CoachReport:
    video_name: str
    duration: float
    model: str
    focus: dict = field(default_factory=dict)
    doing_well: list[str] = field(default_factory=list)
    work_on: str = ""
    summary: str = ""
    themes: list[dict] = field(default_factory=list)
    notes: list[CoachNote] = field(default_factory=list)
    phrases: list[dict] = field(default_factory=list)
    zooms: list[ZoomResult] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    music: dict = field(default_factory=dict)
    usage: UsageTotals = field(default_factory=UsageTotals)
    warnings: list[str] = field(default_factory=list)
    survey_fps: float = 0.0
    zoom_fps: float = 0.0

    @property
    def focus_uncertain(self) -> bool:
        f = self.focus or {}
        if f.get("identified") is False:
            return True
        return _coerce_float(f.get("confidence"), 1.0) < FOCUS_CONFIDENCE_FLOOR


# --------------------------------------------------------------------------- helpers


def fmt_time(seconds: float) -> str:
    """Format seconds as m:ss.s for report text."""
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = seconds - 60 * m
    return f"{m}:{s:04.1f}"


def _coerce_float(value: object, default: float) -> float:
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return default if v != v else v


def _write_frames(tmp: Path, images: list[str], prefix: str = "frame") -> list[str]:
    paths = []
    for i, img_b64 in enumerate(images):
        p = tmp / f"{prefix}_{i:03d}.jpg"
        p.write_bytes(base64.b64decode(img_b64))
        paths.append(str(p))
    return paths


def _frame_list(paths: list[str], timestamps: list[float]) -> str:
    return "\n".join(f"- {p}  (t={t:.2f}s)" for p, t in zip(paths, timestamps))


def build_phrase_map(video_path: Path, audio: AudioFeatures) -> PhraseMap | None:
    """Structure analysis (8-count grid + real phrase boundaries) for a video's audio.

    Returns None when the audio cannot be decoded or is too short; callers fall back to
    the fixed 32-count estimate and say so.
    """
    try:
        y, sr = load_audio(video_path)
    except AudioProcessingError as e:
        logger.warning("Phrase structure skipped: %s", e)
        return None
    if len(y) == 0 or len(audio.beat_times) < 16:
        return None
    pm = analyze_structure(y, sr, audio.beat_times, audio.bpm)
    return pm if pm.boundaries else None


def _parse_phrase_judgments(data: dict, bounds: list[PhraseBoundary]) -> list[dict]:
    """One entry per boundary, in order; a boundary the model skipped or could not see is 'not judged'."""
    raw = [p for p in (data.get("phrases") or []) if isinstance(p, dict)]
    out: list[dict] = []
    for b in bounds:
        near = min(raw, key=lambda p: abs(_coerce_float(p.get("time"), -1e9) - b.time), default=None)
        entry: dict = {
            "time": round(b.time, 2), "counts": b.counts, "kind": b.kind, "acknowledged": None,
            "how": "Not returned by the model.", "response": "", "timing": "", "offset_beats": 0.0,
            "confidence": 0.0, "visibility": "",
        }
        if near is not None and abs(_coerce_float(near.get("time"), -1e9) - b.time) <= 0.75:
            vis = str(near.get("visibility") or "").lower()
            ack = near.get("acknowledged")
            entry.update(
                acknowledged=None if vis == "blocked" or ack is None else bool(ack),
                how=str(near.get("how") or "").strip() or ("Couple not visible in this burst." if vis == "blocked" else ""),
                response=str(near.get("response") or ""), timing=str(near.get("timing") or ""),
                offset_beats=_coerce_float(near.get("offset_beats"), 0.0),
                confidence=max(0.0, min(1.0, _coerce_float(near.get("confidence"), 0.0))), visibility=vis,
            )
        out.append(entry)
    return out


def judge_phrases(
    video_path: Path, out_dir: Path, *, model: str = DEFAULT_COACH_MODEL, fps: float = 4.0, window: float = 3.0,
    max_dimension: int = 768, dancers: str | None = None, progress: Progress | None = None,
) -> dict:
    """Second-opinion pass on phrase acknowledgment against the structure-analysis boundaries.

    Runs the audio analysis, extracts a short burst around every boundary, asks the model
    once for all of them, and writes the verdicts into out_dir/coach.json (backing the
    previous file up as coach.json.prejudge.bak) plus phrase_map.json and song_map.svg.
    Returns {"phrases": judged list, "usage": UsageTotals, "phrase_map": PhraseMap}.
    """
    say: Progress = progress or (lambda msg: logger.info(msg))
    claude_path = _check_claude_cli()
    out_dir.mkdir(parents=True, exist_ok=True)

    say("Analysing audio for tempo, the 8-count grid and phrase changes...")
    audio = extract_audio_features(video_path)
    if not audio.beat_times:
        raise AnalysisError("No beats found in the audio; cannot place phrase changes.")
    pm = build_phrase_map(video_path, audio)
    if pm is None:
        raise AnalysisError("Too little audio to analyse the song's structure.")
    duration = get_video_duration(video_path) or audio.beat_times[-1]
    (out_dir / "phrase_map.json").write_text(json.dumps(pm.to_dict(), indent=1))
    (out_dir / "song_map.svg").write_text(song_map_svg(pm, duration, old_grid=estimate_phrase_starts(audio)))

    coach_path = out_dir / "coach.json"
    existing: dict = json.loads(coach_path.read_text()) if coach_path.exists() else {}
    focus_desc = str((existing.get("focus") or {}).get("description") or "").strip()
    dancer_context = DANCER_CONTEXT_TEMPLATE.format(dancer_description=dancers) + "\n" if dancers else ""
    focus_context = f"An earlier review of this clip identified the couple as: {focus_desc}\n" if focus_desc else ""

    bounds = [b for b in pm.boundaries if b.time + 0.5 < duration]
    bursts: list[PhraseBoundary] = []
    with tempfile.TemporaryDirectory(prefix="wcs_phrase_") as tmp:
        lines: list[str] = []
        n_frames = 0
        for k, b in enumerate(bounds, 1):
            start, end = max(0.0, b.time - window), min(duration, b.time + window)
            fr = extract_frames_between(video_path, start, end, fps=fps, max_dimension=max_dimension)
            if not fr.images:
                continue
            paths = _write_frames(Path(tmp), fr.images, prefix=f"p{k:02d}")
            n_frames += len(paths)
            what = "first phrase starts" if b.kind == "start" else f"{b.counts} counts end here; {b.kind}"
            lines.append(f"Phrase change {k} at {b.time:.1f}s ({what}):\n" + _frame_list(paths, fr.timestamps))
            bursts.append(b)
        if not bursts:
            raise AnalysisError("No frames could be extracted around the phrase changes.")
        prompt = COACH_PHRASE_JUDGE_PROMPT.format(
            dancer_context=dancer_context, focus_context=focus_context, n_bounds=len(bursts), fps=fps,
            window=window, gap=1.0 / fps, burst_list="\n".join(lines), bpm=audio.bpm,
            eights=", ".join(fmt_time(t) for t in pm.eights[:4]),
        )
        timeout, max_turns = _cli_budget(n_frames)
        say(f"Judging {len(bursts)} phrase changes from {n_frames} frames through {model} (budget {timeout // 60} min)...")
        data, usage = _call_claude_cli(claude_path, prompt, timeout=timeout, max_turns=max_turns, model=model)

    judged = _parse_phrase_judgments(data, bursts)
    if coach_path.exists():
        backup = out_dir / "coach.json.prejudge.bak"
        if not backup.exists():
            shutil.copy(coach_path, backup)
        existing["phrases"] = judged
        music = existing.setdefault("music", {})
        d = pm.to_dict()
        music.update(
            bpm=round(audio.bpm, 1), music_start=round(audio.beat_times[0], 2),
            phrase_starts=[round(b.time, 2) for b in bursts], method="structure-v2+judge",
            phrase_map={k: d[k] for k in ("phase", "eights", "boundaries", "method")},
        )
        existing.setdefault("warnings", []).append(
            f"Phrase changes re-judged on {date.today().isoformat()} against the structure analysis "
            f"({len(judged)} boundaries, ${usage.estimated_cost:.2f})."
        )
        existing["phrase_judge_usage"] = asdict(usage)
        coach_path.write_text(json.dumps(existing, indent=1))
    return {"phrases": judged, "usage": usage, "phrase_map": pm}


def _music_context(audio: AudioFeatures | None, phrase_starts: list[float], phrase_map: PhraseMap | None = None) -> str:
    if audio is None or not audio.beat_times:
        return (
            "MUSIC: no audio track could be analysed. Do not penalise the couple for waiting "
            "before the music starts, and treat every timing judgment as a visual estimate.\n"
        )
    lines = [
        f"MUSIC (from the audio track): tempo {audio.bpm:.0f} BPM; the first beat is at "
        f"{audio.beat_times[0]:.1f}s, so anything before that is pre-music and not a fault."
    ]
    if phrase_map is not None and phrase_map.boundaries:
        lines.append(phrase_context(phrase_map) + " Judges expect the couple to acknowledge these.")
    elif phrase_starts:
        lines.append(
            "Estimated phrase changes (32-count phrases; the estimate may be off by a few counts): "
            + ", ".join(fmt_time(t) for t in phrase_starts)
            + ". Judges expect the couple to acknowledge these."
        )
    return "\n".join(lines) + "\n"


def _beat_context(audio: AudioFeatures | None, start: float, end: float, phrase_starts: list[float]) -> str:
    if audio is None or not audio.beat_times:
        return "BEATS: no audio available; infer the pulse from the movement and say so."
    beats = [t for t in audio.beat_times if start <= t <= end]
    if not beats:
        return f"BEATS: none detected inside this window (tempo {audio.bpm:.0f} BPM)."
    marks = []
    for t in beats:
        tag = " (phrase start)" if any(abs(t - p) < 0.05 for p in phrase_starts) else ""
        marks.append(f"{t:.2f}s{tag}")
    return f"BEATS in this window (from audio, {audio.bpm:.0f} BPM): " + ", ".join(marks)


def _decode(img_b64: str) -> np.ndarray | None:
    try:
        buf = np.frombuffer(base64.b64decode(img_b64), dtype=np.uint8)
    except (ValueError, TypeError):
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def _labelled(img: np.ndarray, label: str) -> np.ndarray:
    h, w = img.shape[:2]
    scale = STRIP_HEIGHT / max(h, 1)
    img = cv2.resize(img, (max(1, int(w * scale)), STRIP_HEIGHT), interpolation=cv2.INTER_AREA)
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _strip(images: list[str], labels: list[str], path: Path, kind: str = "") -> str:
    """Write a horizontal strip of labelled frames; return the path or '' on failure.

    A colored border encodes the verdict so the screenshot itself says
    keep (green), refine (red), or check yourself (blue).
    """
    tiles = []
    for img_b64, label in zip(images, labels):
        img = _decode(img_b64)
        if img is not None:
            tiles.append(_labelled(img, label))
    if not tiles:
        return ""
    sep = np.full((STRIP_HEIGHT, 4, 3), 40, dtype=np.uint8)
    parts: list[np.ndarray] = []
    for i, t in enumerate(tiles):
        if i:
            parts.append(sep)
        parts.append(t)
    strip = np.hstack(parts)
    color = KIND_COLORS_BGR.get(kind)
    if color:
        strip = cv2.copyMakeBorder(
            strip, STRIP_BORDER, STRIP_BORDER, STRIP_BORDER, STRIP_BORDER,
            cv2.BORDER_CONSTANT, value=color,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), strip, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return path.name


def _nearest(timestamps: list[float], t: float) -> int:
    return min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - t))


def note_strip(frames: FrameData, t: float, path: Path, spread: float = 0.5, kind: str = "") -> str:
    """Three survey frames around t (t-spread, t, t+spread), de-duplicated."""
    if not frames.images:
        return ""
    idx: list[int] = []
    for offset in (-spread, 0.0, spread):
        i = _nearest(frames.timestamps, t + offset)
        if i not in idx:
            idx.append(i)
    return _strip([frames.images[i] for i in idx], [fmt_time(frames.timestamps[i]) for i in idx], path, kind)


def burst_strip(burst: FrameData, path: Path, count: int = 6, kind: str = "") -> str:
    """Up to `count` evenly spaced frames from a zoom burst."""
    n = len(burst.images)
    if n == 0:
        return ""
    step = max(1, n // count)
    idx = list(range(0, n, step))[:count]
    return _strip([burst.images[i] for i in idx], [fmt_time(burst.timestamps[i]) for i in idx], path, kind)


def nearest_kind(notes: list[CoachNote], t: float, within: float = 3.0) -> str:
    """Verdict of the survey note closest to t, or 'refine' if none is near."""
    near = min(notes, key=lambda n: abs(n.time - t), default=None)
    return near.kind if near is not None and abs(near.time - t) <= within else "refine"


def load_report(out_dir: Path) -> CoachReport:
    """Rebuild a CoachReport from the coach.json a previous run saved.

    Lets the report be re-rendered (new styling, new layout) without
    paying for the model passes again.
    """
    data = json.loads((out_dir / "coach.json").read_text())
    note_fields = {f for f in CoachNote.__dataclass_fields__}
    zoom_fields = {f for f in ZoomResult.__dataclass_fields__}
    usage_fields = {f for f in UsageTotals.__dataclass_fields__}
    report_fields = {f for f in CoachReport.__dataclass_fields__}
    notes = [CoachNote(**{k: v for k, v in n.items() if k in note_fields}) for n in data.get("notes", [])]
    zooms = [ZoomResult(**{k: v for k, v in z.items() if k in zoom_fields}) for z in data.get("zooms", [])]
    usage_raw = data.get("usage") or {}
    usage = UsageTotals(**{k: v for k, v in usage_raw.items() if k in usage_fields})
    rest = {k: v for k, v in data.items() if k in report_fields and k not in ("notes", "zooms", "usage")}
    return CoachReport(notes=notes, zooms=zooms, usage=usage, **rest)


# --------------------------------------------------------------------------- parsing


def _parse_survey(data: dict, duration: float) -> tuple[dict, list[CoachNote], list[dict], list[dict], list[str], dict]:
    """Return (focus, notes, themes, phrases, patterns, impression)."""
    raw_focus = data.get("focus")
    focus: dict = raw_focus if isinstance(raw_focus, dict) else {}

    notes: list[CoachNote] = []
    for raw in data.get("notes") or []:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("note") or "").strip()
        if not text:
            continue
        t = _coerce_float(raw.get("time"), -1.0)
        if t < 0:
            continue
        end = raw.get("end_time")
        kind = str(raw.get("kind") or "refine").lower()
        notes.append(CoachNote(
            time=min(t, duration) if duration else t,
            end_time=_coerce_float(end, -1.0) if end is not None else None,
            kind=kind if kind in ("refine", "keep", "question") else "refine",
            counts=str(raw.get("counts") or ""),
            note=text,
        ))
    notes.sort(key=lambda n: n.time)
    for n in notes:
        if n.end_time is not None and n.end_time < 0:
            n.end_time = None

    themes = [
        {
            "title": str(t.get("title") or "").strip(),
            "detail": str(t.get("detail") or "").strip(),
            "examples": [_coerce_float(x, -1.0) for x in (t.get("examples") or []) if _coerce_float(x, -1.0) >= 0],
        }
        for t in (data.get("themes") or [])
        if isinstance(t, dict) and (t.get("title") or t.get("detail"))
    ]

    phrases = [
        {
            "time": _coerce_float(p.get("time"), -1.0),
            "acknowledged": bool(p.get("acknowledged")),
            "how": str(p.get("how") or ""),
        }
        for p in (data.get("phrases") or [])
        if isinstance(p, dict) and _coerce_float(p.get("time"), -1.0) >= 0
    ]

    patterns = [str(p) for p in (data.get("patterns_identified") or []) if p]
    raw_impression = data.get("overall_impression")
    impression: dict = raw_impression if isinstance(raw_impression, dict) else {}
    return focus, notes, themes, phrases, patterns, impression


def coverage_warning(notes: list[CoachNote], duration: float, fps: float) -> str | None:
    """Warn when the survey notes only start well into the clip.

    The CLI reads every survey frame into its context; on long clips it can
    drop the earliest images and the model then reports them as empty. The
    tell is a first note that arrives a quarter or more of the way through.
    """
    if not notes or duration <= 0:
        return None
    first = min(n.time for n in notes)
    if first < COVERAGE_GAP_FRACTION * duration:
        return None
    return (
        f"Survey notes start at {fmt_time(first)} of a {duration:.0f}s clip; the model reported "
        f"no usable frames before then, so the opening is unreviewed. Re-run with "
        f"--fps {max(0.5, round(fps * 0.75, 1))} (fewer frames) to cover the whole clip."
    )


def _zoom_candidates(data: dict, notes: list[CoachNote], limit: int, duration: float) -> list[tuple[float, str, str]]:
    """(time, reason, nearest survey note) for the moments to slow down on."""
    picked: list[tuple[float, str]] = []
    for raw in data.get("moments_to_zoom") or []:
        if not isinstance(raw, dict):
            continue
        t = _coerce_float(raw.get("time"), -1.0)
        if t < 0 or (duration and t > duration):
            continue
        if any(abs(t - p) < 2.0 for p, _ in picked):
            continue
        picked.append((t, str(raw.get("reason") or "")))
    if len(picked) < limit:
        for n in notes:
            if n.kind != "refine" or len(picked) >= limit:
                continue
            if any(abs(n.time - p) < 2.0 for p, _ in picked):
                continue
            picked.append((n.time, n.note[:120]))
    picked = picked[:limit]
    out = []
    for t, reason in picked:
        near = min(notes, key=lambda n: abs(n.time - t), default=None)
        survey_note = near.note if near and abs(near.time - t) <= 3.0 else reason
        out.append((t, reason, survey_note))
    return out


# --------------------------------------------------------------------------- main entry


def run_coach(
    video_path: Path,
    out_dir: Path,
    *,
    model: str = DEFAULT_COACH_MODEL,
    fps: float = 3.0,
    max_dimension: int = 768,
    dancers: str | None = None,
    division: str | None = None,
    zoom: bool = True,
    zoom_moments: int = 8,
    zoom_fps: float = 10.0,
    zoom_window: float = 2.4,
    progress: Progress | None = None,
) -> CoachReport:
    """Produce coach-style notes for one video. Writes strips and coach.json to out_dir."""
    say: Progress = progress or (lambda msg: logger.info(msg))
    claude_path = _check_claude_cli()
    out_dir.mkdir(parents=True, exist_ok=True)
    strips_dir = out_dir / "strips"
    warnings: list[str] = []

    # ---- audio: tempo, music start, phrase boundaries
    say("Analysing audio for tempo, music start, and phrase changes...")
    audio: AudioFeatures | None = None
    try:
        audio = extract_audio_features(video_path)
    except AudioProcessingError as e:
        warnings.append(f"Audio analysis skipped: {e}")
    phrase_map: PhraseMap | None = build_phrase_map(video_path, audio) if audio and audio.beat_times else None
    if phrase_map is not None:
        phrase_starts = phrase_map.starts
    else:
        phrase_starts = estimate_phrase_starts(audio) if audio and audio.beat_times else []
    if not (audio and audio.beat_times):
        warnings.append("No usable audio: timing and phrasing notes are visual estimates only.")
    music = {
        "bpm": round(audio.bpm, 1) if audio else 0.0,
        "music_start": round(audio.beat_times[0], 2) if audio and audio.beat_times else None,
        "phrase_starts": [round(t, 2) for t in phrase_starts],
        "method": phrase_map.method if phrase_map else "fixed-32",
    }
    if phrase_map is not None:
        d = phrase_map.to_dict()
        music["phrase_map"] = {k: d[k] for k in ("phase", "eights", "boundaries", "method")}
        duration_for_map = get_video_duration(video_path) or (audio.beat_times[-1] if audio else 0.0)
        (out_dir / "song_map.svg").write_text(
            song_map_svg(phrase_map, duration_for_map, old_grid=estimate_phrase_starts(audio) if audio else None)
        )

    # ---- survey frames
    duration = get_video_duration(video_path)
    eff_fps = fps
    if duration > 0 and duration * fps > MAX_SURVEY_FRAMES:
        eff_fps = MAX_SURVEY_FRAMES / duration
        warnings.append(
            f"Survey rate reduced to {eff_fps:.1f} fps to stay under {MAX_SURVEY_FRAMES} frames "
            f"for a {duration:.0f}s clip; use --zoom for detail."
        )
    say(f"Extracting survey frames at {eff_fps:.1f} fps...")
    frames = extract_frames(video_path, fps=eff_fps, max_dimension=max_dimension)
    if not frames.images:
        raise AnalysisError("No frames extracted from video")
    duration = duration or frames.duration

    dancer_context = DANCER_CONTEXT_TEMPLATE.format(dancer_description=dancers) + "\n" if dancers else ""
    division_context = (
        f"DIVISION: this dancer competes in {division} Jack & Jill. Frame the notes for what "
        f"judges expect at that level: say what would raise a question mark on a judge's card.\n"
        if division else ""
    )

    usage = UsageTotals()
    with tempfile.TemporaryDirectory(prefix="wcs_coach_") as tmp:
        paths = _write_frames(Path(tmp), frames.images)
        prompt = COACH_SURVEY_PROMPT.format(
            dancer_context=dancer_context,
            division_context=division_context,
            music_context=_music_context(audio, phrase_starts, phrase_map),
            n_frames=len(paths),
            duration=duration,
            fps=eff_fps,
            gap=1.0 / eff_fps,
            frame_list=_frame_list(paths, frames.timestamps),
            max_zoom=zoom_moments,
            vocab_instruction=PATTERN_VOCABULARY_INSTRUCTION,
        )
        timeout, max_turns = _cli_budget(len(paths))
        say(f"Survey pass: {len(paths)} frames through {model} (budget {timeout // 60} min)...")
        survey, u = _call_claude_cli(claude_path, prompt, timeout=timeout, max_turns=max_turns, model=model)
        usage = usage.add(u)

    focus, notes, themes, phrases, patterns, impression = _parse_survey(survey, duration)
    report = CoachReport(
        video_name=video_path.name,
        duration=duration,
        model=model,
        focus=focus,
        doing_well=[str(x) for x in (impression.get("doing_well") or []) if x],
        work_on=str(impression.get("work_on") or ""),
        summary=str(impression.get("summary") or ""),
        themes=themes,
        notes=notes,
        phrases=phrases,
        patterns=patterns,
        music=music,
        warnings=warnings,
        survey_fps=eff_fps,
    )
    if not notes:
        report.warnings.append("The survey pass returned no timestamped notes; the response was incomplete.")
    gap = coverage_warning(notes, duration, eff_fps)
    if gap:
        report.warnings.append(gap)
        say(f"  ⚠ {gap}")

    say(f"Building frame strips for {len(notes)} notes...")
    for i, n in enumerate(notes):
        n.strip = note_strip(frames, n.time, strips_dir / f"note_{i:02d}.jpg", kind=n.kind)

    # ---- zoom pass
    if zoom:
        candidates = _zoom_candidates(survey, notes, zoom_moments, duration)
        report.zoom_fps = zoom_fps
        for j, (t, reason, survey_note) in enumerate(candidates):
            start = max(0.0, t - zoom_window / 2)
            end = min(duration, t + zoom_window / 2) if duration else t + zoom_window / 2
            say(f"Zoom {j + 1}/{len(candidates)}: {fmt_time(t)} at {zoom_fps:g} fps...")
            burst = extract_frames_between(video_path, start, end, fps=zoom_fps, max_dimension=max_dimension)
            if not burst.images:
                report.warnings.append(f"Zoom at {fmt_time(t)}: no frames could be extracted.")
                continue
            with tempfile.TemporaryDirectory(prefix="wcs_zoom_") as tmp:
                paths = _write_frames(Path(tmp), burst.images, prefix="zoom")
                prompt = COACH_ZOOM_PROMPT.format(
                    dancer_context=dancer_context,
                    start=start,
                    end=end,
                    n_frames=len(paths),
                    fps=zoom_fps,
                    gap=1.0 / zoom_fps,
                    frame_list=_frame_list(paths, burst.timestamps),
                    beat_context=_beat_context(audio, start, end, phrase_starts),
                    survey_note=survey_note.replace('"', "'"),
                )
                timeout, max_turns = _cli_budget(len(paths))
                try:
                    data, u = _call_claude_cli(
                        claude_path, prompt, timeout=timeout, max_turns=max_turns, model=model,
                    )
                except AnalysisError as e:
                    report.warnings.append(f"Zoom at {fmt_time(t)} failed: {e}")
                    continue
                usage = usage.add(u)
            z = ZoomResult(
                time=t,
                reason=reason,
                count_notes=[c for c in (data.get("count_notes") or []) if isinstance(c, dict)],
                diagnosis=str(data.get("diagnosis") or ""),
                fix=str(data.get("fix") or ""),
                confidence=_coerce_float(data.get("confidence"), 0.0),
                visibility=str(data.get("visibility") or ""),
                kind=nearest_kind(notes, t),
            )
            z.strip = burst_strip(burst, strips_dir / f"zoom_{j:02d}.jpg", kind=z.kind)
            report.zooms.append(z)

    report.usage = usage
    if report.focus_uncertain:
        report.warnings.insert(
            0,
            "The model was not confident it followed the right couple. Check the frame strips "
            "before acting on any note below.",
        )
    (out_dir / "coach.json").write_text(json.dumps(asdict(report), indent=2, default=str))
    return report
