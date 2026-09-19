---
name: wcs-coach
description: Coach West Coast Swing competition video with this repo's wcs-analyzer CLI and turn the results into an honest read of what to practise. Use this whenever the user mentions a dance video, a Jack & Jill, Strictly, prelim, semi or final, a bib number, a judge's card, phrasing or musicality, anchors, slot, connection, or asks how they danced, what improved, what to work on, or wants notes they can share with a coach or friend. Also use it for the season pipeline (a folder of clips plus a videos.csv manifest, the bench/ folder of reports, the progress dashboard) and whenever coach.json, coach_report.html, phrase_map.json or progress.json come up. Prefer it over ad-hoc frame extraction even when the user just says "analyze this clip".
---

# Coaching West Coast Swing video

This repo turns competition footage into what an experienced judge would write in the
margin: timestamped notes with frame strips, slow-motion count-by-count looks, a check of
whether the couple answered the music's phrase changes, and drills. The model never sees
the whole video at once; it sees frames plus tempo and phrase boundaries taken from the
audio. Keep that in mind when reading its notes: timing claims are good to about a beat,
and anything about a moment when the couple was hidden is a guess the report should flag.

## Which command answers which question

| The user wants | Run | Cost and time per two-minute clip |
|---|---|---|
| Notes on one clip, like a friend's video review | `wcs-analyzer coach CLIP --dancers "lead wearing bib 42" --division intermediate` | $8 to $14, 20 to 35 min |
| Where the music actually changes, no model call | `wcs-analyzer phrases CLIP -o CLIP_coach` | free, seconds |
| Did they acknowledge each phrase change, strictly judged | `wcs-analyzer phrases CLIP -o CLIP_coach --judge --dancers "..."` | $3 to $4, 4 min |
| One file to send to a coach or friend | `wcs-analyzer coach-bundle DIR1 DIR2 -o notes.html --title "..." --clips` | free |
| A whole season and the trend | the batch scripts in `tools/season/`, see [references/season-pipeline.md](references/season-pipeline.md) | coach and judge as above; the rest is free |
| Rebuild the dashboard from existing reports | `uv run python tools/season/progress_data.py ...` then `build_progress.py` | free, seconds, safe to rerun |
| A WSDC-style score, not coaching | `wcs-analyzer analyze CLIP --provider claude-code` | $3 to $7 |

Only `coach` and `phrases --judge` (and the two batch scripts that call them) spend money;
`phrases` without `--judge`, `coach-bundle`, `phrase_maps.py`, `make_proxies.py`,
`progress_data.py` and `build_progress.py` are local and free, so run them whenever useful.
Use `uv run ...` for every command so the project environment is the one that runs.

The coaching commands run through the local Claude Code CLI (`claude`), so they need it
installed and signed in; no API key. Opus 5 is the default and the only model these
prompts have been tuned on. Cheaper models produce vaguer notes; do not switch without
telling the user what changes.

## Before the first run

- `uv sync` in the repo, then run commands as `uv run wcs-analyzer ...` (or activate the venv).
- `ffmpeg` and `ffprobe` on PATH (audio extraction, frame bursts, proxies).
- `claude --version` works and the user is logged in. A run that fails with "Claude Code CLI failed" plus an exit code is usually a sign-in or rate-limit problem, not a bug.
- Tell the user the cost before a batch. A season of 40 clips is roughly $500 of coaching plus $150 of phrase judging.
- Videos, manifests and report folders stay out of git. `.gitignore` already covers `*.mov/*.mp4`, `bench/`, `proxies/`, `videos.csv`, `*_coach/`. Never `git add -f` any of them.

## Coaching one clip

1. Ask which couple to follow if it is not obvious. A bib number is best: `--dancers "lead wearing bib 285"`. Add `--division` so the notes are pitched at that level.
2. Run `coach`. It writes `<clip>_coach/` (or `-o DIR`) with `coach.json`, `coach_report.html` (self-contained, strips inline), `coach_report.md`, `strips/`, `song_map.svg`.
3. Read the **focus check first**. Confidence under 50% means the model could not confirm the couple; the notes may be about someone else. Say so and do not build conclusions on that clip. Between 50 and 80% read the strips before trusting a note.
4. Check the warnings block. "Survey notes start at 0:55" means the model never saw the opening; rerun with a lower `--fps` (the warning says which). A tempo that looks halved or doubled is folded automatically now, but older reports may carry one.
5. Summarise for the user in this order: what is working (the reports always find real strengths; lead with them), the causal chain the themes describe, the two or three moments worth watching with timestamps, then the drills. Quote the report's timestamps so the user can scrub to them.

Details of every field and how to read the numbers: [references/reading-reports.md](references/reading-reports.md).

## Phrase changes: how to judge musicality honestly

Competition songs are not built from perfect 32-count phrases. `phrases` finds the real
boundaries from the audio (harmony and texture novelty, energy changes, the 8-count grid,
a preference for 32- and 16-count sections) and writes a `song_map.svg` the user can check
by ear. `coach` uses the same boundaries.

`--judge` is the strict second opinion: one model call reviews a short frame burst around
each boundary and credits only a deliberate musical choice (a hit, a stop of two counts or
more, a level change, a release, a styling accent, a sharp change of direction or speed).
Two **decoy windows** per song, count 1 of an 8 that is not a phrase change, are mixed in
unlabelled. The share of decoys the judge credits is its chance level. Report acknowledgment
as pooled counts and a lift over the **season-pooled** chance level in points ("8 of 17,
47%, against 16% chance"): six decoys per event are too few to use alone, under 15 points
above chance is "at chance", and events within 10 points of each other are a tie. Never
compare a strictly judged song with one judged the old way. `coach.json` says which:
`phrase_judge_calibration` present means strict; `music.method` `structure-v2` means judged
in the coach run without decoys; `fixed-32` means the old grid. Boundaries the judge could
not see come back `acknowledged: null` and are left out of every rate.

## Sharing

`coach-bundle` combines any report folders into one HTML file: an introduction for the
reader, contents, every report with strips inline, and with `--clips` a three-second video
snippet next to each slow-motion look (needs the original videos, found two levels above
each report folder or via `--videos-dir`). Expect 10 to 25 MB for three songs; offer the
no-clips version too when the user will email it.

## A season of clips

For more than a handful of clips use the manifest and the batch scripts rather than
running `coach` by hand: they resume, log, and feed the progress dashboard. Follow
[references/season-pipeline.md](references/season-pipeline.md) step by step. The dashboard
(`bench/progress_dashboard_local.html`, open from disk) shows per-event trends, recurring
theme families, the drill library, every note with a play button into the video, and a
practice plan that can be personalised in `bench/plan.json`.

## Writing the read-out

The user usually wants three things: what improved, what has not moved, what to do next.
Ground each claim in the numbers and the report prose together. Rules that keep it honest:

- Three songs per event is a small sample. Differences under about one refine note per song, five percentage points of keep share, or ten points of phrase-rate lift over chance are noise; say "flat" or "a tie" rather than inventing a trend or a winner.
- Compare like with like: prelim with prelim, final with final, and only strictly judged phrase rates with each other. A final with one partner for three songs behaves differently from a prelim with three partners; name that.
- Put strengths first and keep them specific (the reports name timestamps; use them). Then the causal chain: in this dance the faults are usually one chain (anchor not settled → no slot rebuilt → count 1 steps toward the partner → hand climbs to compensate → closed position as a resting state), and saying so is more useful than seven separate complaints.
- When the tool and the judges' marks disagree, say what the tool cannot see: partner context, the rest of the floor, count-level footwork, genre-specific rhythm.
- Retract cleanly. If a metric changes because the method changed (as the phrase rate did when the fixed grid was replaced), say the earlier reading was an artifact.
- End with an ordered plan of at most five items, each with one concrete drill, and say which items can change before the next event and which are the year's work.

## Gotchas

- Phones record 10-bit HEVC. The reports and bundles are fine, but browsers may refuse to play the originals; `tools/season/make_proxies.py` makes H.264 copies the dashboard uses.
- Six clips from one weekend can be several contests. Ask which is which before labelling the manifest; contact sheets (`ffmpeg -vf fps=1/N,tile=4x2`) help the user recognise partners.
- A clip longer than about 150 seconds is surveyed at a reduced frame rate to fit the CLI's context; the report says so.
- The keyword tallies behind the dashboard (theme families, off-beat mentions, stalled seconds) are rough by design. Use them for direction, quote the report prose for evidence.
- Never put the user's name, event names or bib numbers into code, tests or docs that get committed; the examples use bib 42.
