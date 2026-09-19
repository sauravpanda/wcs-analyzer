"""Tests for the season tools: events from the manifest, per-song metrics, dashboard build."""
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "season"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


pd = _load("progress_data")


def _report(video: str, *, focus: float = 0.9, keep: int = 2, refine: int = 3, strict: bool = True,
            first_note: float = 3.0, duration: float = 100.0) -> dict:
    notes = [{"time": first_note, "end_time": first_note + 5.0, "kind": "refine", "counts": "before count 1",
              "note": "You are standing before the dance starts and the slot never forms."}]
    # every other note follows the first one, so a late first note means the opening was never seen
    notes += [{"time": first_note + 17.0 + i * 10, "end_time": None, "kind": "refine", "counts": "",
               "note": "The anchor is skipped and the hand rides overhead."} for i in range(refine - 1)]
    notes += [{"time": first_note + 12.0 + i * 10, "end_time": None, "kind": "keep", "counts": "", "note": "Clean turn on time."}
              for i in range(keep)]
    r = {
        "video_name": video, "duration": duration, "model": "claude-opus-5",
        "focus": {"identified": True, "description": "lead in white shirt, bib 42", "confidence": focus, "occluded": []},
        "doing_well": ["Calm partnership."], "work_on": "Drive count 1.", "summary": "A warm dance with no slot.",
        "themes": [{"title": "The slot keeps collapsing", "detail": "Count 1 steps in.", "examples": [20.0]}],
        "notes": notes,
        "phrases": [{"time": 10.0, "acknowledged": True, "how": "hit"}, {"time": 26.0, "acknowledged": False, "how": "none"},
                    {"time": 42.0, "acknowledged": None, "how": "not judged"}],
        "zooms": [{"time": 20.0, "reason": "no anchor", "count_notes": [{"time": 20.1, "count": "5", "observation": "late"}],
                   "diagnosis": "The anchor is cut short.", "fix": "Tape a lane and anchor in it.", "confidence": 0.6,
                   "visibility": "clear", "kind": "refine", "strip": "zoom_00.jpg"}],
        "patterns": ["sugar push", "whip"],
        "music": {"bpm": 100.0, "music_start": 1.0, "phrase_starts": [10.0, 26.0, 42.0],
                  "method": "structure-v2+judge" if strict else "structure-v2"},
        "usage": {"input_tokens": 1, "output_tokens": 1, "estimated_cost": 2.5},
        "warnings": [],
    }
    if strict:
        r["phrase_decoys"] = [{"time": 18.0, "acknowledged": False}, {"time": 34.0, "acknowledged": True}]
        r["phrase_judge_calibration"] = {"real": 2, "real_hit": 1, "decoys": 2, "decoys_hit": 1}
    return r


def _season(tmp_path: Path) -> tuple[Path, Path]:
    rows = [
        # a multi-word event name with no `short`: label comes from initials
        dict(file="2025-10-boogie-by-the-bay-newcomer-finals-bib42_1.MOV", event="Boogie by the Bay 2025", comp_date="2025-10-19",
             division="newcomer", comp_mode="j&j", comp_stage="finals", bib="42", role="lead", dancers_hint="lead wearing bib 42",
             result="3rd of 8; prelim Y Y Y N N", judge_rank_hint="1", notes="", short="", marks=""),
        dict(file="2025-10-boogie-by-the-bay-newcomer-finals-bib42_2.MOV", event="Boogie by the Bay 2025", comp_date="2025-10-19",
             division="newcomer", comp_mode="j&j", comp_stage="finals", bib="42", role="lead", dancers_hint="lead wearing bib 42",
             result="3rd of 8; prelim Y Y Y N N", judge_rank_hint="1", notes="", short="", marks=""),
        # the same event, a different contest, with an explicit short label
        dict(file="2025-10-boogie-by-the-bay-strictly-prelims-bib42_1.MOV", event="Boogie by the Bay 2025", comp_date="2025-10-19",
             division="novice", comp_mode="strictly", comp_stage="prelims", bib="42", role="lead", dancers_hint="lead wearing bib 42",
             result="DNQ", judge_rank_hint="2", notes="", short="Boogie Str", marks="N N Y"),
        # an earlier-dated event listed later in the file: ordering must come from the date
        dict(file="2025-08-swingtacular-novice-prelims-bib7_1.MP4", event="Swingtacular 2025", comp_date="2025-08-08",
             division="novice", comp_mode="j&j", comp_stage="prelims", bib="7", role="lead", dancers_hint="lead wearing bib 7",
             result="Advanced", judge_rank_hint="3", notes="", short="", marks=""),
    ]
    manifest = tmp_path / "videos.csv"
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    bench = tmp_path / "bench"
    reports = {
        "2025-10-boogie-by-the-bay-newcomer-finals-bib42_1": _report(rows[0]["file"], keep=3, refine=3),
        "2025-10-boogie-by-the-bay-newcomer-finals-bib42_2": _report(rows[1]["file"], focus=0.2),          # excluded
        "2025-10-boogie-by-the-bay-strictly-prelims-bib42_1": _report(rows[2]["file"], strict=False),
        "2025-08-swingtacular-novice-prelims-bib7_1": _report(rows[3]["file"], first_note=60.0),           # partial coverage
        "old-clip-not-in-manifest": _report("old-clip-not-in-manifest.MOV"),
    }
    for stem, r in reports.items():
        d = bench / f"coach_{stem}"
        d.mkdir(parents=True)
        (d / "coach.json").write_text(json.dumps(r))
    (bench / "plan.json").write_text(json.dumps([{"id": "x", "title": "Custom item", "metric": None, "why": "w", "drills": ["d"]}]))
    return manifest, bench


def test_events_come_from_the_manifest(tmp_path: Path):
    manifest, bench = _season(tmp_path)
    out = pd.build(manifest, bench, bench / "plan.json")

    # date, then stage order; the event with two contests gets a stage suffix on its auto label
    assert [e["short"] for e in out["events"]] == ["Swingtacular", "Boogie Str", "BBTB F"]
    boogie = out["events"][2]
    assert boogie["label"] == "Boogie by the Bay 2025" and boogie["round"] == "Final" and boogie["division"] == "Newcomer"
    assert boogie["result_short"] == "3rd of 8" and boogie["bib"] == "42"
    assert boogie["n_clips"] == 2 and boogie["n_used"] == 1            # the 20%-focus song is kept but not averaged
    assert boogie["keep_share"] == 0.5                                  # 3 keep of 6 notes on the used song
    assert boogie["decoys"] == 2 and boogie["decoy_hits"] == 1 and boogie["decoy_rate"] == 0.5
    assert boogie["phrase_rate"] == 0.5 and boogie["phrases"] == 2      # the null verdict is not judged
    strictly = out["events"][1]
    assert strictly["marks"] == "N N Y" and strictly["decoys"] == 0
    assert out["skipped"] == ["coach_old-clip-not-in-manifest"]
    assert out["plan"][0]["title"] == "Custom item"
    assert out["chance_level"] == {"decoys": 2, "hits": 1, "rate": 0.5}   # pooled over the season's used songs


def test_song_metrics_and_exclusions(tmp_path: Path):
    manifest, bench = _season(tmp_path)
    out = pd.build(manifest, bench, None)
    by_id = {c["id"]: c for c in out["clips"]}
    used = by_id["2025-10-boogie-by-the-bay-newcomer-finals-bib42_1"]
    assert used["used"] and used["clip"] == 1 and used["phrase_judge"] == "strict"
    assert used["opening_s"] == 5.0 and used["families"]["anchor"] == 2 and used["families"]["slot"] == 1
    assert used["stall_s"] == 5.0                                        # "standing" note counts as stalled time
    assert by_id["2025-10-boogie-by-the-bay-newcomer-finals-bib42_2"]["exclude_reason"] == "couple not confirmed on camera"
    partial = by_id["2025-08-swingtacular-novice-prelims-bib7_1"]
    assert partial["partial"] and not partial["used"] and "60s" in partial["exclude_reason"]
    assert by_id["2025-10-boogie-by-the-bay-strictly-prelims-bib42_1"]["phrase_judge"] == "in-run"
    assert len(out["drills"]) == 4 and out["drills"][0]["family"] == "anchor"
    assert "plan" not in out


def test_build_writes_hosted_and_local_pages(tmp_path: Path):
    manifest, bench = _season(tmp_path)
    (bench / "progress.json").write_text(json.dumps(pd.build(manifest, bench, None)))
    (bench / "proxies").mkdir()
    (bench / "proxies" / "2025-10-boogie-by-the-bay-newcomer-finals-bib42_1.mp4").write_bytes(b"x")
    res = subprocess.run([sys.executable, str(TOOLS / "build_progress.py"), "--bench", str(bench)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    hosted = (bench / "progress_dashboard.html").read_text()
    local = (bench / "progress_dashboard_local.html").read_text()
    assert "__DATA_JSON__" not in hosted and "const LOCAL_BASE = null" in hosted
    assert local.startswith("<!doctype html>") and (bench / "proxies").as_uri() in local
    assert "3 songs have no .mp4" in res.stdout
