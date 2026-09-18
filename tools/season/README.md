# Season tools

Scripts for coaching a whole season of competition clips and watching the trend. They sit
outside the package on purpose: they orchestrate the `wcs-analyzer` CLI and read its
output folders, nothing more. The Claude Code skill in `.claude/skills/wcs-coach` explains
when to use which one; `references/season-pipeline.md` there is the full walkthrough.

| Script | What it does |
|---|---|
| `coach_batch.py` | `wcs-analyzer coach` over every clip in the manifest, resumable, with logs |
| `judge_batch.py` | `wcs-analyzer phrases --judge` over every coached song not yet strictly judged |
| `phrase_maps.py` | song maps for older reports (audio only, no model call) |
| `make_proxies.py` | browser-playable H.264 copies for the local dashboard |
| `progress_data.py` | every `coach.json` into one `progress.json` of per-song and per-event metrics |
| `build_progress.py` | `progress.json` + `progress_template.html` into the two dashboard pages |

Layout the scripts expect (all of it gitignored):

```
videos.csv                 manifest: file,event,comp_date,division,comp_mode,comp_stage,bib,role,
                           dancers_hint,result,judge_rank_hint,notes   (+ optional short, marks)
<clips>.MOV / .MP4         next to the manifest, or --videos-dir
bench/coach_<stem>/        one folder per song: coach.json, coach_report.html, strips/, song_map.svg
bench/proxies/<stem>.mp4   from make_proxies.py
bench/plan.json            optional: the dashboard's practice plan (see the skill reference)
bench/progress.json        from progress_data.py
bench/progress_dashboard.html, progress_dashboard_local.html   from build_progress.py
```

Refresh after new clips:

```bash
uv run python tools/season/coach_batch.py  --manifest videos.csv --bench bench
uv run python tools/season/judge_batch.py  --manifest videos.csv --bench bench
python tools/season/make_proxies.py        --videos-dir . --out bench/proxies
python tools/season/progress_data.py       --manifest videos.csv --bench bench
python tools/season/build_progress.py      --bench bench
```
