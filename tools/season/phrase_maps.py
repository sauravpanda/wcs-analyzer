"""Write phrase_map.json and song_map.svg into every coach folder that lacks one.

Audio only, no model call: gives older reports (judged on the fixed 32-count grid) a song
map so the dashboard can show where the music really changes. New coach runs and
`phrases --judge` write these themselves.

Usage (inside the project environment):
  uv run python tools/season/phrase_maps.py --videos-dir . --bench bench
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from wcs_analyzer.audio import estimate_phrase_starts, extract_audio_features
from wcs_analyzer.coach import build_phrase_map
from wcs_analyzer.phrases import song_map_svg
from wcs_analyzer.video import get_video_duration


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos-dir", type=Path, default=Path("."))
    ap.add_argument("--bench", type=Path, default=Path("bench"))
    args = ap.parse_args()
    t0 = time.time()
    done = 0
    for cj in sorted(args.bench.glob("coach_*/coach.json")):
        out = cj.parent
        if (out / "phrase_map.json").exists():
            continue
        r = json.loads(cj.read_text())
        video = args.videos_dir / r["video_name"]
        if not video.exists():
            print(out.name, "video missing")
            continue
        audio = extract_audio_features(video)
        pm = build_phrase_map(video, audio)
        if pm is None:
            print(out.name, "too little audio")
            continue
        dur = get_video_duration(video) or audio.beat_times[-1]
        (out / "phrase_map.json").write_text(json.dumps(pm.to_dict(), indent=1))
        (out / "song_map.svg").write_text(song_map_svg(pm, dur, old_grid=estimate_phrase_starts(audio)))
        done += 1
        print(f"{out.name}: {audio.bpm:.0f} bpm, sections " + ", ".join(str(b.counts) for b in pm.boundaries[1:]), flush=True)
    print(f"wrote {done} maps in {round(time.time() - t0)}s")


if __name__ == "__main__":
    main()
