# The season pipeline

Everything here lives outside git: the videos, the manifest, and the `bench/` folder of
reports. The scripts are in `tools/season/` and only orchestrate the CLI.

## 1. Name the clips and fill the manifest

Name files so the event, division, round and bib are in the stem, one number per song:

```
2026-09-sbdf-intermediate-finals-bib285_1.MOV
2026-09-sbdf-intermediate-finals-bib285_2.MOV
```

`videos.csv` sits next to them, one row per file:

| column | meaning |
|---|---|
| `file` | file name (the stem is the song id everywhere downstream) |
| `event` | event name; give different contests at one event different names ("SwingTime 2026 · All-In") |
| `comp_date` | ISO date; orders the season |
| `division` | newcomer, novice, intermediate, advanced, all levels |
| `comp_mode` | j&j, strictly, classic, showcase, routine |
| `comp_stage` | prelims, semis, finals, social |
| `bib` | the user's bib; shown on the event card |
| `role` | lead or follow |
| `dancers_hint` | passed to `--dancers`, e.g. `lead wearing bib 285` |
| `result` | free text; the part before the first `;` is shown on the card |
| `judge_rank_hint` | optional 1 = best event, for correlating with the tool |
| `notes` | anything, e.g. "camera on another couple" |
| `short` | optional chart label ("SBDF F"); auto-generated from initials when absent |
| `marks` | optional judge marks ("Y A3 Y Y") shown under the result |

Songs are grouped into an event by `event + comp_mode + comp_stage`. Six clips from one
weekend are often more than one contest; confirm with the user before filling rows.

## 2. Coach every clip

```bash
uv run python tools/season/coach_batch.py --manifest videos.csv --bench bench \
    [--clips sbdf] [--priority finals,prelims] [--model claude-opus-5]
```

Sequential and resumable: a song with `bench/coach_<stem>/coach.json` is skipped, so the
batch can be stopped and restarted. Logs in `bench/logs/coach_<stem>.log`, summary rows in
`bench/coach_runs.csv`. Budget 25 to 35 minutes and $8 to $14 per clip. Run it in the
background and check `coach_runs.csv` rather than waiting.

When a run fails: read the log tail. "Claude Code CLI failed" with a subtype is the CLI
(sign-in, rate limit, context); "No frames" is the video. A report whose first note is far
into the song means the CLI dropped the early frames; rerun that clip alone with a lower
`--fps` via `wcs-analyzer coach` directly.

## 3. Judge the phrase changes strictly

```bash
uv run python tools/season/judge_batch.py --manifest videos.csv --bench bench [--priority sbdf]
```

Runs `phrases --judge` on every coached song that lacks a `phrase_judge_calibration`
block. About 4 minutes and $3 to $4 per song. It rewrites `phrases` in `coach.json`, keeps
the previous file as `coach.json.prejudge.bak`, and re-renders the report with the song
map. Older reports without a song map can get one for free with
`uv run python tools/season/phrase_maps.py --videos-dir . --bench bench`.

## 4. Proxies for playback

```bash
python tools/season/make_proxies.py --videos-dir . --out bench/proxies
```

H.264 720p copies (hard links when the original already qualifies). Roughly 40 MB per
song. The local dashboard plays these directly.

## 5. Build the dashboard

```bash
python tools/season/progress_data.py --manifest videos.csv --bench bench
python tools/season/build_progress.py --bench bench
open bench/progress_dashboard_local.html
```

`progress.json` holds per-song and per-event metrics (definitions in
[reading-reports.md](reading-reports.md)); the build writes two pages:

- `progress_dashboard_local.html`: open from disk; every play button plays `bench/proxies/<song>.mp4`. This is the one the user should use day to day.
- `progress_dashboard.html`: the same page for hosting or sending; it cannot open local paths, so it offers a folder picker (Chrome and Edge remember the folder; the page walks subfolders and prefers `proxies`).

Both pages carry a practice plan. Without `bench/plan.json` it shows generic items; with
one, the user's own. The file is a JSON list of items shaped like:

```json
{"id": "anchor", "title": "Finish the anchor before every lead", "metric": "families.anchor",
 "up": false, "target": 4, "why": "one or two sentences grounded in this season's numbers",
 "drills": ["one concrete drill", "another"]}
```

`metric` is one of `phrase_rate`, `keep_share`, `stall_s`, `opening_s`, `off_beat`, or
`families.<slot|anchor|connection|closed|phrasing|early|posture>`; `up` says whether higher
is better. Rewrite `plan.json` when the read-out changes; the checkboxes and notes people
tick on the page are saved separately and survive a rebuild.

## Refresh after a new event

1. Add the rows to `videos.csv` (ask which contest each clip is).
2. `coach_batch.py`, then `judge_batch.py`, then `make_proxies.py`.
3. `progress_data.py`, `build_progress.py`.
4. Reread the newest reports in full before writing the read-out; the dashboard numbers point, the prose proves.
5. Update `bench/plan.json` if the priorities moved, rebuild once more.
