# Reading the reports and the numbers

## coach.json

| field | what it is |
|---|---|
| `focus` | `description`, `confidence` (0..1), `occluded` time ranges. Under 0.5: do not trust the notes. |
| `summary`, `doing_well`, `work_on` | the overall impression; `work_on` is the single highest-value change |
| `themes` | 2 to 4 items with `title`, `detail`, `examples` (seconds); what a judge would write on the card |
| `notes` | every few seconds: `time`, `end_time`, `kind` (`keep` / `refine` / `question`), `counts`, `note`, `strip` |
| `phrases` | one per boundary: `time`, `acknowledged` (true / false / null = not judged), `how`, and after strict judging `counts`, `kind`, `response`, `timing`, `offset_beats`, `confidence`, `visibility` |
| `phrase_decoys`, `phrase_judge_calibration` | strict judge only: decoy verdicts and `{real, real_hit, decoys, decoys_hit}` |
| `zooms` | slow-motion looks: `time`, `reason`, `count_notes` (per beat), `diagnosis`, `fix`, `confidence`, `visibility`, `kind`, `strip` |
| `music` | `bpm`, `music_start`, `phrase_starts`, `method` (`fixed-32`, `structure-v2`, `structure-v2+judge`), `phrase_map` |
| `patterns` | canonical pattern names seen |
| `warnings` | reduced survey rate, coverage gaps, tempo repairs, judging notes |
| `usage` | tokens and estimated cost |

Verdict colours are consistent everywhere: green `keep`, red `refine`, blue `question`
(the model was unsure; check by eye).

## Metrics in progress.json

Per song, then averaged per event over songs marked `used` (focus at least 0.5 and the
survey saw the opening):

| metric | definition | reading |
|---|---|---|
| `keep_share` | keep notes over all notes | higher is better; 30 to 40% is typical |
| `phrase_rate` | acknowledged over judged boundaries | read against the decoy rate; `null` when nothing was judged |
| `decoys`, `decoy_hits`, `decoy_rate` | strict judge's false positives | the chance level; 15 to 25% is normal |
| `phrase_judge` | `strict`, `in-run`, `old-grid` | only compare `strict` with `strict` |
| `stall_s` | seconds covered by refine notes about closed position, hugging, standing | lower is better; finals run higher than prelims |
| `opening_s` | length of the earliest note about standing or not yet dancing (first 15 s) | under 2 s is the target |
| `off_beat`, `on_beat` | mentions of late / behind / rushing and of on-the-beat in the count notes and survey notes | rough; direction only |
| `families.<id>` | refine notes matching each theme family's keywords | a note can land in several families |
| `patterns` | distinct pattern names per song | vocabulary size, not quality |

Theme families and what they mean on the floor:

- `slot`: count 1 does not travel; the slot shrinks or rotates; the lead posts and the follower shuttles.
- `anchor`: the next lead starts before 5&6 settles; no stretch to lead from.
- `connection`: hand height jumps (belt to overhead), arm-led turns, elbows locked.
- `closed`: closed position, hug, or standing as a resting state.
- `phrasing`: phrase changes pass unmarked.
- `early`: led before the follower was ready; hesitations, guessing.
- `posture`: eyes down, chin drops, wide low base, sinking.

## What counts as a difference

With three songs per event, treat as noise: under one refine note per song in a family,
under five points of keep share, under ten points of phrase rate, under three seconds of
stall time. Two events differing by more than that in the same direction as the report
prose is a finding; one number alone is not.

Prelims and finals are different tasks: a final is three songs with one partner, so
partner comfort shows (more closed position, more stalling) and the vocabulary is the same
as the prelim. Mixed-level contests with a strong partner run cleaner on every metric
because the partner keeps the slot alive; say so rather than crediting the lead.

## A read-out that holds up

Structure that has worked:

1. **What is working**, specific, with timestamps from `doing_well` and `keep` notes. These are the same across events (partnership, safe turns, floorcraft, eyes up); name them every time so the user knows what not to break.
2. **The chain**, from the themes: which fault causes which. Usually anchor → slot → count 1 → hand height → closed position.
3. **A per-event table** of the two or three metrics that moved, with the judges' result alongside. When the tool disagrees with the judges, say what the tool cannot see.
4. **What is new** at the latest event, with the report phrases that say it.
5. **The plan**: at most five items, ordered by cost on a judge's card, one drill each, split into what can change before the next event and what is the year's work.

Phrases to avoid: "most improved" from a metric whose method changed; "consistently"
from three songs; any percentage of phrase acknowledgment without its chance level.
