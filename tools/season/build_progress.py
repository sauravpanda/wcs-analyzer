"""Inject <bench>/progress.json into the dashboard template and write two pages.

  <bench>/progress_dashboard.html        share or host it; videos load through the folder picker
  <bench>/progress_dashboard_local.html  open from disk; plays <bench>/proxies/<song>.mp4 directly

Run progress_data.py first. Usage:
  python tools/season/build_progress.py --bench bench
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", type=Path, default=Path("bench"))
    ap.add_argument("--template", type=Path, default=HERE / "progress_template.html")
    ap.add_argument("--videos", type=Path, default=None,
                    help="folder of browser-playable .mp4 copies for the local page (default: <bench>/proxies)")
    args = ap.parse_args()
    bench = args.bench.resolve()
    videos = (args.videos or bench / "proxies").resolve()

    data = json.loads((bench / "progress.json").read_text())
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    template = args.template.read_text().replace("__DATA_JSON__", payload)

    hosted = template.replace("__LOCAL_BASE__", "null")
    (bench / "progress_dashboard.html").write_text(hosted)

    # The hosted page is wrapped in a document by whatever serves it; the local file needs
    # its own skeleton so browsers render it in standards mode with the right charset.
    body = template.replace("__LOCAL_BASE__", json.dumps(videos.as_uri() + "/"))
    head_end = body.index("</style>") + len("</style>")
    local = ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
             "<meta name='viewport' content='width=device-width, initial-scale=1'>"
             + body[:head_end] + "</head><body>" + body[head_end:] + "</body></html>")
    (bench / "progress_dashboard_local.html").write_text(local)

    missing = [c["id"] for c in data["clips"] if not (videos / (c["id"] + ".mp4")).exists()]
    print(f"wrote {bench / 'progress_dashboard.html'} ({len(hosted) // 1024} KB) and progress_dashboard_local.html; "
          f"local page plays from {videos}; {len(missing)} songs have no .mp4 there"
          + (f": {', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}" if missing else ""))


if __name__ == "__main__":
    main()
