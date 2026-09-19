"""Build <bench>/progress.json: per-song and per-event metrics from every coach report.

Feeds the progress dashboard (build_progress.py). Re-run after new coach runs or phrase
judging land in <bench>/coach_*/coach.json. Every keyword tally is rough by design; the
regexes live here so the numbers stay auditable.

Events come from the manifest (videos.csv): songs are grouped by event + comp_mode +
comp_stage, ordered by date then stage. Optional manifest columns: `short` (label for
charts), `marks` (judge marks, shown on the event card).

Rules:
  * a song whose focus confidence is under MIN_FOCUS is kept in the table but left out of
    every event mean (the model could not confirm the couple);
  * a song whose first note lands later than PARTIAL_FRACTION of the way through is marked
    partial (the model never saw the opening) and is also left out of the means.

Usage:
  python tools/season/progress_data.py --manifest videos.csv --bench bench [--plan bench/plan.json]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

MIN_FOCUS = 0.5
PARTIAL_FRACTION = 0.25
STAGE_ORDER = {"prelims": 0, "semis": 1, "finals": 2, "social": 3}
STAGE_LABEL = {"prelims": "Prelim", "semis": "Semi", "finals": "Final", "social": "Social"}

FAMILIES = [
    dict(id="slot", label="Slot and drive on 1",
         keys=["slot", "count 1", "drive", "travel", "toward her", "toward each other", "shrink"]),
    dict(id="anchor", label="Anchor not settled", keys=["anchor", "5&6", "settle"]),
    dict(id="connection", label="Connection height",
         keys=["connection height", "hand rides", "above the shoulder", "overhead", "arm-led", "elbow",
               "hand climbs", "hand goes up", "shoulder height", "eyeline", "hairline"]),
    dict(id="closed", label="Closed position or stalled",
         keys=["closed position", "hug", "chest-to-chest", "in place", "stall", "small footprint",
               "square meter", "huddle", "park", "static", "embrace", "standing", "stand still"]),
    dict(id="phrasing", label="Phrase changes unmarked", keys=["phrase", "musical", "hit", "the change"]),
    dict(id="early", label="Led before the follower was ready",
         keys=["not ready", "before she", "before he", "early", "rushed", "hesitat", "guess", "wait for her", "wait for him"]),
    dict(id="posture", label="Posture, eyes, wide base",
         keys=["looking down", "chin", "eyes drop", "look at the floor", "posture", "folds", "hunch",
               "squat", "crouch", "straddle", "wide base", "wide stance", "sink", "center drops", "centre drops"]),
]

OFF_BEAT = re.compile(
    r"(a beat|half a beat|beat and a half|two beats) (late|early|behind|ahead)|\blate\b|behind the (beat|music|count)"
    r"|ahead of the (beat|music|count)|off[- ]time|off the beat|\brush(ed|ing)?\b|catching up|catch(es)? up"
    r"|not on the beat|drag(s|ging)? behind|lands? after the beat", re.I)
ON_BEAT = re.compile(
    r"\bon the beat\b|\bon time\b|\bon the (1|one)\b|right on (the )?(beat|count|phrase|1|one)|square on"
    r"|lands? on the beat|in time with|on the downbeat|dead on|on the phrase", re.I)
OPENING = re.compile(
    r"standing|stationary|not (yet )?moving|before (you|the dance)|waiting|flat on both feet"
    r"|nothing that reads as dancing|no visible weight change|first (weight change|real figure|figure)", re.I)


def stem_of(video_name: str) -> str:
    """Song id: the file name without its video extension (.MOV and .MP4 alike)."""
    return re.sub(r"\.(mov|mp4|m4v|avi|mkv|webm)$", "", Path(video_name).name, flags=re.I)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def auto_short(event: str, mode: str, stage: str, multi_mode: bool, multi_stage: bool) -> str:
    """A chart label when the manifest has no `short` column: initials or the single word,
    plus a mode/stage suffix only when the event has several contests."""
    name = re.sub(r"\b(19|20)\d{2}\b", "", event).replace("·", " ").strip()
    words = [w for w in re.split(r"[\s\-/]+", name) if w]
    base = words[0] if len(words) == 1 else "".join(w[0] for w in words if w[0].isalnum()).upper()
    parts = [base or event]
    m = (mode or "").lower()
    if multi_mode and m and m not in ("j&j", "jj", "jack & jill", "jack and jill"):
        parts.append({"strictly": "Str", "classic": "Cls", "showcase": "Show", "routine": "Rtn"}.get(m, m[:3].title()))
    if multi_stage and stage:
        parts.append(STAGE_LABEL.get(stage.lower(), stage.title())[0])
    return " ".join(parts)


def load_events(manifest: Path) -> tuple[list[dict], dict[str, str]]:
    """Events (one per event + mode + stage) in season order, and song id -> event key."""
    rows = list(csv.DictReader(manifest.open(newline="")))
    groups: dict[str, dict] = {}
    for r in rows:
        event, mode, stage = r.get("event", ""), r.get("comp_mode", ""), r.get("comp_stage", "")
        key = slug(f"{event}-{mode}-{stage}")
        g = groups.setdefault(key, dict(
            key=key, label=event, date=r.get("comp_date", ""), division=(r.get("division") or "").title(),
            mode=mode, stage=stage, round=STAGE_LABEL.get(stage.lower(), stage.title()),
            result=(r.get("result") or "").strip(), marks=(r.get("marks") or "").strip(),
            bib=(r.get("bib") or "").strip(), short=(r.get("short") or "").strip(), files=[],
        ))
        g["files"].append(r["file"])
        for col in ("short", "marks", "result"):          # first non-empty value wins
            if not g[col] and (r.get(col) or "").strip():
                g[col] = r[col].strip()
    modes_per_event: dict[str, set] = {}
    stages_per_event: dict[str, set] = {}
    for g in groups.values():
        modes_per_event.setdefault(g["label"], set()).add(g["mode"])
        stages_per_event.setdefault(g["label"], set()).add(g["stage"])
    for g in groups.values():
        if not g["short"]:
            g["short"] = auto_short(g["label"], g["mode"], g["stage"], len(modes_per_event[g["label"]]) > 1,
                                    len(stages_per_event[g["label"]]) > 1)
        g["result_short"] = g["result"].split(";")[0].strip()[:48]
    events = sorted(groups.values(), key=lambda g: (g["date"], STAGE_ORDER.get(g["stage"].lower(), 9), g["mode"]))
    stem_to_key = {stem_of(f): g["key"] for g in events for f in g["files"]}
    return events, stem_to_key


def family_counts(texts: list[str]) -> dict[str, int]:
    return {f["id"]: sum(1 for t in texts if any(k in t for k in f["keys"])) for f in FAMILIES}


def family_of(text: str) -> str:
    t = text.lower()
    best, best_n = "other", 0
    for f in FAMILIES:
        n = sum(t.count(k) for k in f["keys"])
        if n > best_n:
            best, best_n = f["id"], n
    return best


def stall_seconds(notes: list[dict]) -> float:
    keys = next(f for f in FAMILIES if f["id"] == "closed")["keys"]
    s = 0.0
    for n in notes:
        if n["kind"] == "refine" and any(k in n["note"].lower() for k in keys):
            end = n.get("end_time") or (n["time"] + 3.0)
            s += max(0.0, end - n["time"])
    return round(s, 1)


def opening_seconds(notes: list[dict]) -> float:
    """Seconds the coach flagged as standing or not yet dancing at the top of the song.

    Zero when no note complained about the opening (any verdict counts: the slow
    starts are sometimes filed as questions rather than refines).
    """
    best = 0.0
    for n in notes:
        if n["time"] > 15 or n["kind"] == "keep":
            continue
        counts = (n.get("counts") or "").lower()
        if "before count 1" in counts or "pre-pattern" in counts or OPENING.search(n["note"]):
            span = (n.get("end_time") or (n["time"] + 3.0)) - n["time"]
            if span > 20:   # a note spanning the whole clip is a caveat, not an opening
                continue
            best = max(best, span)
    return round(best, 1)


def phrase_map_for(r: dict, folder: Path) -> dict | None:
    """The structure analysis for a song: from coach.json when the run produced it, else
    from a phrase_map.json written afterwards by `wcs-analyzer phrases`."""
    pm = (r.get("music") or {}).get("phrase_map")
    if not pm and (folder / "phrase_map.json").exists():
        pm = json.loads((folder / "phrase_map.json").read_text())
    if not pm:
        return None
    judged = {round(float(p["time"]), 1): p for p in r["phrases"]}
    bounds = []
    for b in pm.get("boundaries", []):
        near = min(judged.values(), key=lambda p: abs(float(p["time"]) - b["time"]), default=None)
        verdict = None
        if near is not None and abs(float(near["time"]) - b["time"]) <= 0.75:
            verdict = near.get("acknowledged")
        bounds.append(dict(t=round(b["time"], 2), counts=b["counts"], kind=b["kind"], confidence=b.get("confidence", 0),
                           judged=verdict))
    return dict(phase=pm.get("phase", 0), eights=[round(t, 2) for t in pm.get("eights", [])], boundaries=bounds,
                energy=pm.get("energy", []), novelty=pm.get("novelty", []),
                beat_times=pm.get("beat_times", []), method=pm.get("method", "structure-v2"))


def clip_metrics(r: dict, event_key: str, folder: Path) -> dict:
    notes = r["notes"]
    kinds = Counter(n["kind"] for n in notes)
    focus = (r.get("focus") or {}).get("confidence") or 0.0
    duration = float(r.get("duration") or 0)
    first = min((n["time"] for n in notes), default=0.0)
    partial = bool(notes) and duration > 0 and first > PARTIAL_FRACTION * duration
    refine_txt = [n["note"].lower() for n in notes if n["kind"] == "refine"]
    zoom_obs = [c.get("observation", "") for z in r["zooms"] for c in (z.get("count_notes") or [])]
    survey_txt = [n["note"] for n in notes]
    # Boundaries added by a post-run tempo correction carry acknowledged=None: not judged.
    phrases = [p for p in r["phrases"] if p.get("acknowledged") is not None]
    m = re.search(r"_(\d+)$", stem_of(r["video_name"]))
    return dict(
        id=stem_of(r["video_name"]),
        event=event_key, clip=int(m.group(1)) if m else 0,
        focus=round(focus, 2), used=focus >= MIN_FOCUS and not partial, partial=partial,
        exclude_reason=("couple not confirmed on camera" if focus < MIN_FOCUS else
                        (f"model saw nothing before {first:.0f}s" if partial else "")),
        duration=round(duration, 1), bpm=round(float((r.get("music") or {}).get("bpm") or 0)),
        notes=len(notes), keep=kinds.get("keep", 0), refine=kinds.get("refine", 0), check=kinds.get("question", 0),
        keep_share=round(kinds.get("keep", 0) / len(notes), 3) if notes else 0.0,
        phrases=len(phrases), phrase_hits=sum(1 for p in phrases if p["acknowledged"]),
        phrase_rate=round(sum(1 for p in phrases if p["acknowledged"]) / len(phrases), 3) if phrases else None,
        # decoy windows (count 1 of an 8 that is not a phrase change) measure the judge's false-positive rate
        decoys=len(r.get("phrase_decoys") or []),
        decoy_hits=sum(1 for p in (r.get("phrase_decoys") or []) if p.get("acknowledged")),
        phrase_judge=("strict" if r.get("phrase_judge_calibration") else
                      ("in-run" if (r.get("music") or {}).get("method") == "structure-v2" else "old-grid")),
        stall_s=stall_seconds(notes),
        opening_s=opening_seconds(notes),
        families=family_counts(refine_txt),
        off_beat=sum(len(OFF_BEAT.findall(t)) for t in zoom_obs + survey_txt),
        on_beat=sum(len(ON_BEAT.findall(t)) for t in zoom_obs + survey_txt),
        count_obs=len(zoom_obs),
        patterns=sorted({p for p in (r.get("patterns") or [])}),
        themes=[t["title"] for t in r["themes"]],
        themes_full=[dict(title=t["title"], detail=t.get("detail", "")) for t in r["themes"]],
        video=r["video_name"],
        folder=folder.name,
        phrase_method=(r.get("music") or {}).get("method", "fixed-32"),
        old_grid=[round(float(t), 2) for t in (r.get("music") or {}).get("phrase_starts", [])],
        phrase_map=phrase_map_for(r, folder),
        moments=sorted(
            [dict(t=round(float(n["time"]), 1), end=round(float(n["end_time"]), 1) if n.get("end_time") else None,
                  kind=n["kind"], counts=n.get("counts") or "", text=n["note"], src="note") for n in notes]
            + [dict(t=round(float(z["time"]), 1), end=None, kind=z.get("kind") or "refine", counts="slow motion",
                    text=(z.get("diagnosis") or z.get("reason") or "").strip(), fix=(z.get("fix") or "").strip(),
                    src="zoom", confidence=round(float(z.get("confidence") or 0), 2)) for z in r["zooms"]],
            key=lambda m: m["t"]),
        doing_well=list(r.get("doing_well") or []),
        work_on=r.get("work_on") or "",
        summary=r.get("summary") or "",
        cost=round(float((r.get("usage") or {}).get("estimated_cost") or 0), 2),
    )


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def build(manifest: Path, bench: Path, plan: Path | None = None) -> dict:
    events_meta, stem_to_key = load_events(manifest)
    reports: list[tuple[dict, Path]] = []
    skipped: list[str] = []
    for p in sorted(glob.glob(str(bench / "coach_*" / "coach.json"))):
        r = json.load(open(p))
        if stem_of(r["video_name"]) in stem_to_key:
            reports.append((r, Path(p).parent))
        else:
            skipped.append(Path(p).parent.name)
    clips = [clip_metrics(r, stem_to_key[stem_of(r["video_name"])], folder) for r, folder in reports]
    order = {e["key"]: i for i, e in enumerate(events_meta)}
    clips.sort(key=lambda c: (order[c["event"]], c["clip"]))

    drills = []
    for r, _ in reports:
        for z in r["zooms"]:
            fix = (z.get("fix") or "").strip()
            if not fix:
                continue
            drills.append(dict(
                clip=stem_of(r["video_name"]), event=stem_to_key[stem_of(r["video_name"])], time=round(float(z["time"]), 1),
                kind=z.get("kind") or "refine", confidence=round(float(z.get("confidence") or 0), 2),
                family=family_of(" ".join([z.get("reason") or "", z.get("diagnosis") or "", fix])),
                reason=(z.get("reason") or "").strip(), fix=fix,
            ))

    events = []
    for e in events_meta:
        mine = [c for c in clips if c["event"] == e["key"]]
        used = [c for c in mine if c["used"]]
        if not mine:
            continue
        meta = {k: v for k, v in e.items() if k != "files"}
        judged = sum(c["phrases"] for c in used)
        events.append(dict(
            **meta, n_clips=len(mine), n_used=len(used), cost=round(sum(c["cost"] for c in mine), 2),
            keep_share=mean([c["keep_share"] for c in used]),
            # pooled over the event's judged boundaries, like decoy_rate, so the two can be subtracted
            phrase_rate=round(sum(c["phrase_hits"] for c in used) / judged, 3) if judged else None,
            phrase_hits=sum(c["phrase_hits"] for c in used), phrases=judged,
            decoys=sum(c["decoys"] for c in used), decoy_hits=sum(c["decoy_hits"] for c in used),
            decoy_rate=(round(sum(c["decoy_hits"] for c in used) / sum(c["decoys"] for c in used), 3)
                        if sum(c["decoys"] for c in used) else None),
            stall_s=mean([c["stall_s"] for c in used]),
            opening_s=mean([c["opening_s"] for c in used]),
            off_beat=mean([c["off_beat"] for c in used]), on_beat=mean([c["on_beat"] for c in used]),
            refine=mean([c["refine"] for c in used]), notes=mean([c["notes"] for c in used]),
            patterns=mean([len(c["patterns"]) for c in used]),
            families={f["id"]: mean([c["families"][f["id"]] for c in used]) for f in FAMILIES},
            themes=[t for c in used for t in c["themes"]],
            doing_well=[d for c in used for d in c["doing_well"]],
        ))

    all_decoys = sum(e["decoys"] for e in events)
    out = dict(
        generated=date.today().isoformat(),
        rules=dict(min_focus=MIN_FOCUS, partial_fraction=PARTIAL_FRACTION),
        families=[dict(id=f["id"], label=f["label"]) for f in FAMILIES],
        # the season-pooled decoy rate is the chance level to read every phrase rate against;
        # six decoys per event are too few to trust on their own
        chance_level=dict(decoys=all_decoys, hits=sum(e["decoy_hits"] for e in events),
                          rate=round(sum(e["decoy_hits"] for e in events) / all_decoys, 3) if all_decoys else None),
        events=events, clips=clips, drills=drills, skipped=skipped,
    )
    if plan and plan.exists():
        out["plan"] = json.loads(plan.read_text())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("videos.csv"))
    ap.add_argument("--bench", type=Path, default=Path("bench"))
    ap.add_argument("--plan", type=Path, default=None, help="plan.json for the dashboard's practice plan (default: <bench>/plan.json)")
    args = ap.parse_args()
    plan = args.plan or (args.bench / "plan.json")
    out = build(args.manifest, args.bench, plan)
    (args.bench / "progress.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {args.bench / 'progress.json'}: {len(out['events'])} events, {len(out['clips'])} songs, "
          f"{len(out['drills'])} drills" + (f"; skipped {len(out['skipped'])} report folders not in the manifest: "
                                            f"{', '.join(out['skipped'])}" if out["skipped"] else ""))
    ch = out["chance_level"]
    if ch["rate"] is not None:
        print(f"  chance level (season-pooled decoys): {ch['hits']}/{ch['decoys']} = {ch['rate']:.0%}")
    for e in out["events"]:
        pr = f"{e['phrase_rate']:.0%}" if e["phrase_rate"] is not None else "-"
        lift = (f" ({(e['phrase_rate'] - ch['rate']) * 100:+.0f} pts)"
                if e["phrase_rate"] is not None and ch["rate"] is not None else "")
        print(f"  {e['short']:14s} used {e['n_used']}/{e['n_clips']} keep {e['keep_share']:.0%} phrases {pr}{lift} "
              f"decoys {e['decoy_hits']}/{e['decoys']} stall {e['stall_s']:.0f}s open {e['opening_s']} off {e['off_beat']}")


if __name__ == "__main__":
    main()
