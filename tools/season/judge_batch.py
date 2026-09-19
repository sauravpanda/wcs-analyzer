"""Run `wcs-analyzer phrases --judge` over every coached song that has not been strictly judged.

A song counts as judged when its coach.json carries `phrase_judge_calibration` (written by
the judge with decoy windows). Logs to <bench>/logs/judge_<stem>.log, summarises in
<bench>/judge_runs.csv, retries once after a minute on failure. Budget about four minutes
and $3 to $4 of Opus usage per song.

Usage (inside the project environment):
  uv run python tools/season/judge_batch.py --manifest videos.csv --bench bench [--clips substr] [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("videos.csv"))
    ap.add_argument("--videos-dir", type=Path, default=None)
    ap.add_argument("--bench", type=Path, default=Path("bench"))
    ap.add_argument("--clips", default="")
    ap.add_argument("--priority", default="", help="comma-separated substrings; matching songs run first")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--cmd", default="wcs-analyzer")
    args = ap.parse_args()

    videos_dir = args.videos_dir or args.manifest.resolve().parent
    selectors = [s for s in args.clips.split(",") if s]
    priority = [s for s in args.priority.split(",") if s]
    hints = {Path(r["file"]).stem: r.get("dancers_hint", "") for r in csv.DictReader(args.manifest.open(newline=""))}

    todo = []
    for cj in sorted(args.bench.glob("coach_*/coach.json")):
        r = json.loads(cj.read_text())
        stem = Path(r["video_name"]).stem
        if selectors and not any(s in stem for s in selectors):
            continue
        if r.get("phrase_judge_calibration"):
            continue
        video = videos_dir / r["video_name"]
        if video.exists():
            todo.append((stem, video, cj.parent))
    todo.sort(key=lambda t: (next((i for i, k in enumerate(priority) if k in t[0]), len(priority)), t[0]))
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} songs to judge", flush=True)

    logs = args.bench / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    runs = args.bench / "judge_runs.csv"
    new = not runs.exists()
    with runs.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["started_utc", "clip", "model", "exit_code", "seconds", "attempt"])
        for stem, video, out in todo:
            for attempt in (1, 2):
                cmd = [*shlex.split(args.cmd), "phrases", str(video), "-o", str(out), "--judge", "--model", args.model]
                if hints.get(stem):
                    cmd += ["--dancers", hints[stem]]
                started = datetime.now(timezone.utc).isoformat(timespec="seconds")
                t0 = time.time()
                print(f"[{started}] judge {stem} (attempt {attempt}) ...", flush=True)
                with (logs / f"judge_{stem}.log").open("a") as lf:
                    try:
                        code = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=1800).returncode
                    except subprocess.TimeoutExpired:
                        code = 124
                secs = round(time.time() - t0)
                print(f"    exit={code} {secs // 60}m{secs % 60:02d}s", flush=True)
                w.writerow([started, stem, args.model, code, secs, attempt])
                f.flush()
                if code == 0:
                    break
                time.sleep(60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
