"""Make browser-playable copies of the competition clips for the local dashboard.

Phones record 10-bit HEVC, which browsers often refuse to play. This writes
<out>/<stem>.mp4 as H.264 720p at 30 fps (same timestamps as the original). Clips that
are already H.264 at 720p or smaller are hard-linked instead of re-encoded. Tries the
Apple hardware encoder first, then libx264. Needs ffmpeg and ffprobe on PATH.

Usage:
  python tools/season/make_proxies.py --videos-dir . --out bench/proxies
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

VIDEO_EXT = {".mov", ".mp4", ".m4v"}


def probe(path: Path, field: str) -> str:
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", f"stream={field}",
                           "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip().strip(",")


def encode(src: Path, dst: Path, hw: bool) -> bool:
    vcodec = ["-c:v", "h264_videotoolbox", "-b:v", "3000k", "-profile:v", "high"] if hw else \
             ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
    part = dst.with_suffix(".part.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", "scale=-2:720", "-r", "30", *vcodec,
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(part)]
    ok = subprocess.run(cmd).returncode == 0 and part.exists() and part.stat().st_size > 1_000_000
    if ok:
        os.replace(part, dst)
    elif part.exists():
        part.unlink()
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos-dir", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("bench/proxies"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done = skipped = linked = failed = 0
    for src in sorted(p for p in args.videos_dir.iterdir() if p.suffix.lower() in VIDEO_EXT):
        dst = args.out / (src.stem + ".mp4")
        if dst.exists():
            skipped += 1
            continue
        if probe(src, "codec_name") == "h264" and int(probe(src, "height") or 0) <= 720:
            try:
                os.link(src, dst)
            except OSError:
                import shutil
                shutil.copy(src, dst)
            linked += 1
            print(f"{src.stem}: linked (already H.264 <= 720p)", flush=True)
            continue
        t1 = time.time()
        ok = encode(src, dst, hw=True) or encode(src, dst, hw=False)
        done += ok
        failed += not ok
        size = dst.stat().st_size // 1_000_000 if ok else 0
        print(f"{src.stem}: {'ok' if ok else 'FAILED'} {size} MB in {round(time.time() - t1)}s", flush=True)
    print(f"transcoded {done}, linked {linked}, skipped {skipped}, failed {failed}, "
          f"{round(time.time() - t0)}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
