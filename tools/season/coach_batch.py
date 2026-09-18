"""Run `wcs-analyzer coach` over every clip in a manifest, sequentially and resumably.

A clip is skipped when <bench>/coach_<stem>/coach.json already exists, so the batch can be
stopped and restarted. Each run logs to <bench>/logs/coach_<stem>.log and is summarised in
<bench>/coach_runs.csv. Budget roughly 20 to 35 minutes and $8 to $14 of Opus usage per
two-minute clip.

Usage (inside the project environment so `wcs-analyzer` is on PATH):
  uv run python tools/season/coach_batch.py --manifest videos.csv --bench bench
  uv run python tools/season/coach_batch.py --clips swingtime --priority finals,prelims
"""
from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("videos.csv"))
    ap.add_argument("--videos-dir", type=Path, default=None, help="where the manifest's files live (default: the manifest's folder)")
    ap.add_argument("--bench", type=Path, default=Path("bench"))
    ap.add_argument("--clips", default="", help="comma-separated substrings; only matching file names run")
    ap.add_argument("--priority", default="", help="comma-separated substrings; matching clips run first, in this order")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--zoom-moments", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=5400, help="per-clip wall clock limit, seconds")
    ap.add_argument("--cmd", default="wcs-analyzer", help="command to run (e.g. 'uv run wcs-analyzer')")
    args = ap.parse_args()

    videos_dir = args.videos_dir or args.manifest.resolve().parent
    selectors = [s for s in args.clips.split(",") if s]
    priority = [s for s in args.priority.split(",") if s]
    rows = list(csv.DictReader(args.manifest.open(newline="")))
    rows = [r for r in rows if not selectors or any(s in r["file"] for s in selectors)]
    rows.sort(key=lambda r: (next((i for i, k in enumerate(priority) if k in r["file"]), len(priority)), r["file"]))

    logs = args.bench / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    runs_csv = args.bench / "coach_runs.csv"
    new_file = not runs_csv.exists()
    with runs_csv.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["started_utc", "clip", "model", "exit_code", "seconds", "out_dir"])
        for r in rows:
            stem = Path(r["file"]).stem
            out = args.bench / f"coach_{stem}"
            if (out / "coach.json").exists():
                print(f"skip (done): {stem}", flush=True)
                continue
            video = videos_dir / r["file"]
            if not video.exists():
                print(f"skip (missing video): {video}", flush=True)
                continue
            cmd = [*shlex.split(args.cmd), "coach", str(video), "--model", args.model,
                   "--zoom-moments", str(args.zoom_moments), "-o", str(out)]
            if r.get("dancers_hint"):
                cmd += ["--dancers", r["dancers_hint"]]
            if r.get("division"):
                cmd += ["--division", r["division"]]
            started = datetime.now(timezone.utc).isoformat(timespec="seconds")
            t0 = time.time()
            print(f"[{started}] coach {stem} ...", flush=True)
            with (logs / f"coach_{stem}.log").open("w") as lf:
                try:
                    code = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=args.timeout).returncode
                except subprocess.TimeoutExpired:
                    code = 124
            secs = round(time.time() - t0)
            print(f"    exit={code} {secs // 60}m{secs % 60:02d}s -> {out.name}", flush=True)
            w.writerow([started, stem, args.model, code, secs, out.name])
            f.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
