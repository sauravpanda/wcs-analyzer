# Reading the reports and the numbers

## coach.json

| field | what it is |
|---|---|
| `focus` | `description`, `confidence` (0..1), `occluded` time ranges. Under 0.5: do not trust the notes. |
| `summary`, `doing_well`, `work_on` | the overall impression; `work_on` is the single highest-value change |
| `themes` | 2 to 4 items with `title`, `detail`, `examples` (seconds); what a judge would write on the card |
| `notes` | every few seconds: `time`, `end_time`, `kind` (`keep` / `refine` / `question`), `counts`, `note`, `strip`, `tags` (theme families: `count1`, `rhythm`, `body`, `free_arm`, `pulse`, `connection`, `anchor`, `slot`, `closed`, `phrasing`, `early`, `posture`, `face`) |
| `movement` | the survey's quality-of-movement read: `count1` (`strike-and-transfer` / `falls back` / `mixed` / `not visible`), `rhythm` (`matches the song` / `straight on a swung song` / `swung on a straight song` / `not visible`), `body` (one sentence on the torso and the free arm) |
| `phrases` | one per boundary: `time`, `acknowledged` (true / false / null = not judged), `how`, and after strict judging `counts`, `kind`, `response`, `timing`, `offset_beats`, `confidence`, `visibility` |
| `phrase_decoys`, `phrase_judge_calibration` | strict judge only: decoy verdicts and `{real, real_hit, decoys, decoys_hit}` |
| `zooms` | slow-motion looks: `time`, `reason`, `count_notes` (per beat), `diagnosis`, `fix`, `confidence`, `visibility`, `kind`, `strip`, plus `count1` (strike-and-transfer or falls back), `rhythm` (straight or swung triples) and `tags` |
| `music` | `bpm`, `music_start`, `phrase_starts`, `method` (`fixed-32`, `structure-v2`, `structure-v2+judge`), `phrase_map`, `feel` (`straight` / `light swing` / `swung` / empty when unmeasurable) and `swing_ratio` (where the off-beat sits in the beat: 0.50 straight, about 0.67 triplet swing) |
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
| `phrase_rate` | acknowledged over judged boundaries, pooled over the event's used songs (not a mean of song rates) | read against the season chance level, see below; `null` when nothing was judged |
| `decoys`, `decoy_hits`, `decoy_rate` | the strict judge's verdicts on windows that were not phrase changes | per event these are 6 or 8 windows: too few to trust alone |
| `chance_level` (top level) | decoys pooled over the whole season: `{decoys, hits, rate}` | the number to quote as chance |
| `phrase_judge` | `strict`, `in-run`, `old-grid` | only compare `strict` with `strict` |

### Reading the phrase rate

1. Chance is the **season-pooled** decoy rate (`chance_level.rate`, typically 15 to 25%). A single event's 0 of 6 or 3 of 6 is sampling noise; mention it only as a sanity check.
2. Lift is the difference in points: event `phrase_rate` minus `chance_level.rate`. Under 15 points above chance, call the event "at chance". Between events, treat differences under 10 points of lift as a tie.
3. Quote pooled counts with the percentage ("8 of 17, 47%, against a 16% chance level"). With 15 to 25 judged boundaries per event, two events are rarely separable; say "a tie at the top" when they are within 10 points rather than naming a winner.
4. `phrases` entries with `acknowledged: null` were sent to the judge but came back blocked (couple hidden) or unanswered; they are excluded from `phrases` and `phrase_hits`, which is why `phrase_judge_calibration.real` can exceed the judged count. Do not treat a null as a miss.
5. Song-level rates rest on 3 to 8 boundaries; never rank songs by them.
| `stall_s` | seconds covered by refine notes about closed position, hugging, standing | lower is better; finals run higher than prelims |
| `opening_s` | length of the earliest note about standing or not yet dancing (first 15 s) | under 2 s is the target |
| `off_beat`, `on_beat` | mentions of late / behind / rushing and of on-the-beat in the count notes and survey notes | rough; direction only |
| `families.<id>` | refine notes matching each theme family's keywords | a note can land in several families |
| `patterns` | distinct pattern names per song | vocabulary size, not quality |
| `feel`, `swing_ratio` | the song's rhythm feel from the audio | read the triples against it; an empty feel means the song had no audible subdivision or the off-beat landed too early to trust (ratio under 0.45) |
| `count1_fallback`, `count1_obs` | slow-motion looks where count 1 fell back, over looks that could see count 1; per event pooled as `count1_rate` | under about five looks per event is noise; direction only |
| `rhythm_mismatch`, `rhythm_obs` | looks whose triples contradict the song's feel (straight on swung, swung on straight) | direction only |
| `count1`, `rhythm`, `body` (per song) | the survey's `movement` read | quote the words; do not average them |

Theme families and what they mean on the floor:

- `slot`: count 1 does not travel; the slot shrinks or rotates; the lead posts and the follower shuttles.
- `anchor`: the next lead starts before 5&6 settles; no stretch to lead from.
- `connection`: hand height jumps (belt to overhead), arm-led turns, elbows locked.
- `closed`: closed position, hug, or standing as a resting state.
- `phrasing`: phrase changes pass unmarked.
- `early`: led before the follower was ready; hesitations, guessing.
- `posture`: eyes down, chin drops, wide low base, sinking.
- `count1`: the lead falls back onto the count-1 foot before the beat instead of striking and transferring, so the 1 looks rushed and the pattern starts before the follower is sent.
- `rhythm`: the triples do not match the song (straight, even triples on a swung song read as anti-swung; the middle step should be late).
- `body`: the torso is a rigid frame carried over the footwork; no contra-body rotation, wave through the spine, or head arriving last.
- `free_arm`: the free arm hangs, parks on the hip or in a pocket, instead of adding energy, slowing a moment, or finishing a rotation.
- `pulse`: no visible pulse on the main beats, or a stiff one.

Reports from before these families existed carry no tags; their counts come from keywords
alone and run low, so compare tagged events with tagged events.

## Floor and ceiling

A judge's mark is set by the floor of the dance: what happens in every ordinary pattern. The
count-1 step (strike and transfer on the beat, not a fall back before it), the rhythm against
the song's feel, whether the torso and free arm take part, connection height and the slot are
floor items and belong at the top of a read-out. Phrase hits, held pictures and tricks raise
the ceiling; report them, but after the floor. A tool read that leads with the phrase rate has
the priorities upside down for novice and intermediate.

## What counts as a difference

With three songs per event, treat as noise: under one refine note per song in a family,
under five points of keep share, under ten points of phrase-rate lift over chance, under
three seconds of stall time. Two events differing by more than that in the same direction as the report
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
