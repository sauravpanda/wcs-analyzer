"""Tests for coach mode: phrase estimation, burst extraction, orchestration, and rendering."""

import base64
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from wcs_analyzer.audio import AudioFeatures, estimate_phrase_starts
from wcs_analyzer.coach import (
    CoachNote,
    CoachReport,
    ZoomResult,
    _zoom_candidates,
    coverage_warning,
    fmt_time,
    judge_phrases,
    load_report,
    nearest_kind,
    note_strip,
    run_coach,
)
from wcs_analyzer.coach_report import write_bundle, write_reports
from wcs_analyzer.phrases import PhraseBoundary, PhraseMap
from wcs_analyzer.exceptions import AudioProcessingError
from wcs_analyzer.pricing import UsageTotals
from wcs_analyzer.video import FrameData, extract_frames_between


def _jpeg_b64(color=(30, 60, 90), size=(48, 64)) -> str:
    img = np.zeros((size[0], size[1], 3), np.uint8)
    img[:] = color
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def _frames(n: int = 30, fps: float = 3.0) -> FrameData:
    return FrameData(
        images=[_jpeg_b64(((i * 8) % 255, 100, 150)) for i in range(n)],
        timestamps=[i / fps for i in range(n)],
        fps_original=30.0, fps_sampled=fps, duration=n / fps, width=64, height=48,
    )


def _audio(n_beats: int = 96) -> AudioFeatures:
    beats = [0.5 * i for i in range(n_beats)]
    strengths = [1.0 if i % 32 == 5 else 0.2 for i in range(n_beats)]
    return AudioFeatures(bpm=120.0, beat_times=beats, beat_strengths=strengths, duration=beats[-1] + 0.5)


SURVEY = {
    "focus": {"identified": True, "description": "lead in white shirt, bib 42", "confidence": 0.8, "occluded": ["0:41-0:44"]},
    "overall_impression": {"doing_well": ["natural pulse", "clean foot strikes"], "work_on": "drive down the slot on 1",
                           "summary": "Solid basics with a strong groove."},
    "themes": [
        {"title": "Look up", "detail": "Eyes drop before each lead.", "examples": [6.0, 18.0]},
        {"title": "Drive on 1", "detail": "First step lands sideways.", "examples": [13.0]},
    ],
    "notes": [
        {"time": 6.0, "end_time": None, "kind": "refine", "counts": "", "note": "Looking down before the lead."},
        {"time": 13.0, "end_time": 14.0, "kind": "refine", "counts": "1-2 of pattern", "note": "Left foot points at the follow; direction ambiguous."},
        {"time": 4.0, "end_time": None, "kind": "keep", "counts": "", "note": "Cool sweep, keep that."},
        {"time": -3, "note": "negative time, dropped"},
        {"time": 8.0, "note": ""},
    ],
    "phrases": [{"time": 5.0, "acknowledged": False, "how": "no change of energy"},
                {"time": 21.0, "acknowledged": True, "how": "hit with a spin"}],
    "moments_to_zoom": [{"time": 13.0, "reason": "ambiguous 1"}, {"time": 13.5, "reason": "duplicate within 2s"},
                        {"time": 6.0, "reason": "looking down"}],
    "patterns_identified": ["sugar push", "whip"],
}

ZOOM = {
    "count_notes": [{"time": 12.0, "count": "1", "observation": "left foot steps toward the follow"},
                    {"time": 12.5, "count": "2", "observation": "pigeon-toe stance"}],
    "diagnosis": "Direction on 1 was unclear, so 2 could not follow through.",
    "fix": "Drive straight down the slot on 1.",
    "confidence": 0.7,
    "visibility": "clear",
}


class TestPhraseStarts:
    def test_picks_offset_with_strongest_onsets(self):
        starts = estimate_phrase_starts(_audio(96))
        assert starts == [2.5, 18.5, 34.5]  # beats 5, 37, 69 at 0.5 s per beat

    def test_too_few_beats_returns_empty(self):
        assert estimate_phrase_starts(_audio(20)) == []

    def test_without_strengths_uses_first_beat(self):
        audio = AudioFeatures(bpm=120.0, beat_times=[0.5 * i for i in range(64)], beat_strengths=[], duration=32.0)
        assert estimate_phrase_starts(audio) == [0.0, 16.0]


class TestBurstExtraction:
    def _write_video(self, path: Path, seconds: int = 3, fps: int = 30) -> None:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
        if not writer.isOpened():
            pytest.skip("cv2 VideoWriter (mp4v) unavailable in this environment")
        for i in range(seconds * fps):
            frame = np.zeros((48, 64, 3), np.uint8)
            frame[:] = (i % 255, 0, 0)
            writer.write(frame)
        writer.release()

    def test_window_at_10fps(self, tmp_path: Path):
        clip = tmp_path / "clip.mp4"
        self._write_video(clip)
        burst = extract_frames_between(clip, 1.0, 2.0, fps=10.0)
        assert 9 <= len(burst.images) <= 12
        assert all(0.9 <= t <= 2.05 for t in burst.timestamps)
        assert burst.timestamps == sorted(burst.timestamps)
        assert burst.fps_sampled == 10.0

    def test_empty_window(self, tmp_path: Path):
        clip = tmp_path / "clip.mp4"
        self._write_video(clip)
        assert extract_frames_between(clip, 2.0, 1.0, fps=10.0).images == []


class TestHelpers:
    def test_fmt_time(self):
        assert fmt_time(0) == "0:00.0"
        assert fmt_time(65.25) == "1:05.2"

    def test_coverage_warning_when_notes_start_late(self):
        late = [CoachNote(time=55.6, note="a"), CoachNote(time=90.0, note="b")]
        msg = coverage_warning(late, duration=168.0, fps=2.1)
        assert msg is not None
        assert "0:55.6" in msg and "168s" in msg and "--fps 1.6" in msg

    def test_coverage_warning_quiet_for_normal_runs(self):
        early = [CoachNote(time=3.0, note="a"), CoachNote(time=90.0, note="b")]
        assert coverage_warning(early, duration=168.0, fps=2.1) is None
        assert coverage_warning([], duration=168.0, fps=2.1) is None
        assert coverage_warning(early, duration=0.0, fps=2.1) is None

    def test_zoom_candidates_dedupe_and_fill(self):
        notes = [CoachNote(time=6.0, note="look down"), CoachNote(time=13.0, note="ambiguous 1"),
                 CoachNote(time=25.0, note="late anchor")]
        picked = _zoom_candidates(SURVEY, notes, limit=8, duration=30.0)
        times = [t for t, _, _ in picked]
        assert times[:2] == [13.0, 6.0]      # 13.5 dropped as a duplicate of 13.0
        assert 25.0 in times                  # filled from remaining refine notes
        assert picked[0][2] == "ambiguous 1"  # nearest survey note attached

    def test_note_strip_writes_file(self, tmp_path: Path):
        rel = note_strip(_frames(10, 2.0), 2.0, tmp_path / "strips" / "n.jpg")
        assert rel == "n.jpg"
        img = cv2.imread(str(tmp_path / "strips" / "n.jpg"))
        assert img is not None and img.shape[1] > img.shape[0]  # three tiles side by side

    def test_strip_border_color_encodes_verdict(self, tmp_path: Path):
        note_strip(_frames(10, 2.0), 2.0, tmp_path / "keep.jpg", kind="keep")
        note_strip(_frames(10, 2.0), 2.0, tmp_path / "refine.jpg", kind="refine")
        keep = cv2.imread(str(tmp_path / "keep.jpg"))[2, 2]      # BGR corner pixel inside the border
        refine = cv2.imread(str(tmp_path / "refine.jpg"))[2, 2]
        assert keep[1] > 100 and keep[2] < 90        # green dominant
        assert refine[2] > 150 and refine[1] < 90    # red dominant

    def test_nearest_kind(self):
        notes = [CoachNote(time=10.0, note="x", kind="keep"), CoachNote(time=30.0, note="y", kind="question")]
        assert nearest_kind(notes, 11.0) == "keep"
        assert nearest_kind(notes, 29.0) == "question"
        assert nearest_kind(notes, 20.0) == "refine"   # nothing within 3 s
        assert nearest_kind([], 5.0) == "refine"


class TestRunCoach:
    def _run(self, tmp_path: Path, survey: dict = SURVEY, audio_side_effect=None, **kwargs) -> CoachReport:
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"x")
        out = tmp_path / "out"
        usage = UsageTotals(input_tokens=100, output_tokens=50, estimated_cost=0.5, model="claude-opus-5", pricing_known=True)
        with patch("wcs_analyzer.coach._check_claude_cli", return_value="/bin/claude"), \
             patch("wcs_analyzer.coach.extract_audio_features", side_effect=audio_side_effect or (lambda p: _audio())), \
             patch("wcs_analyzer.coach.get_video_duration", return_value=30.0), \
             patch("wcs_analyzer.coach.extract_frames", return_value=_frames(90, 3.0)), \
             patch("wcs_analyzer.coach.extract_frames_between", return_value=_frames(24, 10.0)), \
             patch("wcs_analyzer.coach._call_claude_cli",
                   side_effect=[(survey, usage), (ZOOM, usage), (ZOOM, usage)]) as cli:
            report = run_coach(video, out, zoom_moments=2, **kwargs)
        self.cli = cli
        self.out = out
        return report

    def test_full_run(self, tmp_path: Path):
        report = self._run(tmp_path)
        assert report.model == "claude-opus-5"
        assert [n.time for n in report.notes] == [4.0, 6.0, 13.0]       # sorted, invalid ones dropped
        assert report.notes[0].kind == "keep"
        assert all((self.out / "strips" / n.strip).exists() for n in report.notes)
        assert [t["title"] for t in report.themes] == ["Look up", "Drive on 1"]
        assert len(report.phrases) == 2 and report.phrases[0]["acknowledged"] is False
        assert len(report.zooms) == 2
        assert report.zooms[0].time == 13.0 and report.zooms[0].count_notes[0]["count"] == "1"
        assert (self.out / "strips" / report.zooms[0].strip).exists()
        assert report.usage.input_tokens == 300                           # survey + two zooms
        assert report.music["bpm"] == 120.0 and report.music["music_start"] == 0.0
        assert report.music["phrase_starts"] == [2.5, 18.5, 34.5]
        assert report.patterns == ["sugar push", "whip"]
        assert not report.focus_uncertain and report.warnings == []
        assert self.cli.call_count == 3
        # The survey prompt carries the music context and the focus instruction
        survey_prompt = self.cli.call_args_list[0][0][1]
        assert "120 BPM" in survey_prompt and "phrase changes" in survey_prompt
        assert "confirm which couple" in survey_prompt
        # The zoom prompt carries the beats inside its window
        zoom_prompt = self.cli.call_args_list[1][0][1]
        assert "BEATS in this window" in zoom_prompt
        assert (self.out / "coach.json").exists()

    def test_low_focus_confidence_is_flagged_first(self, tmp_path: Path):
        survey = json.loads(json.dumps(SURVEY))
        survey["focus"]["confidence"] = 0.3
        report = self._run(tmp_path, survey=survey)
        assert report.focus_uncertain
        assert "not confident" in report.warnings[0]

    def test_no_audio_still_runs(self, tmp_path: Path):
        def boom(_p):
            raise AudioProcessingError("ffmpeg missing")
        report = self._run(tmp_path, audio_side_effect=boom)
        assert any("Audio analysis skipped" in w for w in report.warnings)
        assert report.music["music_start"] is None
        survey_prompt = self.cli.call_args_list[0][0][1]
        assert "no audio track" in survey_prompt

    def test_no_zoom(self, tmp_path: Path):
        report = self._run(tmp_path, zoom=False)
        assert report.zooms == [] and report.zoom_fps == 0.0
        assert self.cli.call_count == 1

    def test_zoom_inherits_nearest_note_verdict(self, tmp_path: Path):
        report = self._run(tmp_path)
        # zooms at 13.0 (refine note) and 6.0 (refine note)
        assert [z.kind for z in report.zooms] == ["refine", "refine"]

    def test_load_report_round_trip(self, tmp_path: Path):
        report = self._run(tmp_path)
        loaded = load_report(self.out)
        assert loaded.video_name == report.video_name
        assert [n.note for n in loaded.notes] == [n.note for n in report.notes]
        assert loaded.notes[0].strip == report.notes[0].strip
        assert isinstance(loaded.zooms[0], ZoomResult) and loaded.zooms[0].fix == ZOOM["fix"]
        assert loaded.usage.input_tokens == report.usage.input_tokens
        assert loaded.music == report.music

    def test_division_and_dancers_reach_prompt(self, tmp_path: Path):
        self._run(tmp_path, dancers="lead wearing bib 42", division="intermediate")
        prompt = self.cli.call_args_list[0][0][1]
        assert "lead wearing bib 42" in prompt
        assert "intermediate Jack & Jill" in prompt


JUDGE = {"focus_confirmed": True, "phrases": [
    {"time": 2.5, "acknowledged": True, "how": "hands-up hit on the 1", "response": "hit", "timing": "on",
     "offset_beats": 0, "confidence": 0.8, "visibility": "clear"},
    {"time": 18.6, "acknowledged": False, "how": "the side pass just continues", "response": "none", "timing": "none",
     "offset_beats": 0, "confidence": 0.7, "visibility": "clear"},
]}


class TestJudgePhrases:
    def test_writes_verdicts_into_coach_json(self, tmp_path: Path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"x")
        out = tmp_path / "out"
        out.mkdir()
        (out / "coach.json").write_text(json.dumps({
            "video_name": "clip.mp4", "focus": {"description": "lead in white shirt, bib 42"},
            "phrases": [{"time": 2.5, "acknowledged": False, "how": "old verdict"}],
            "music": {"bpm": 120.0, "method": "fixed-32", "phrase_starts": [2.5, 18.5]}, "warnings": [],
        }))
        beats = [0.5 * i for i in range(96)]
        pm = PhraseMap(bpm=120.0, beat_times=beats, phase=5, eights=beats[5::8], boundaries=[
            PhraseBoundary(time=2.5, beat=5, counts=0, kind="start", confidence=0.5),
            PhraseBoundary(time=18.5, beat=37, counts=32, kind="section", confidence=0.9),
            PhraseBoundary(time=34.5, beat=69, counts=32, kind="regular", confidence=0.4),
        ], energy=[0.5] * 96, novelty=[0.1] * 96)
        usage = UsageTotals(input_tokens=10, output_tokens=5, estimated_cost=1.25, model="claude-opus-5", pricing_known=True)
        with patch("wcs_analyzer.coach._check_claude_cli", return_value="/bin/claude"), \
             patch("wcs_analyzer.coach.extract_audio_features", return_value=_audio()), \
             patch("wcs_analyzer.coach.build_phrase_map", return_value=pm), \
             patch("wcs_analyzer.coach.get_video_duration", return_value=40.0), \
             patch("wcs_analyzer.coach.extract_frames_between", return_value=_frames(24, 4.0)), \
             patch("wcs_analyzer.coach._call_claude_cli", return_value=(JUDGE, usage)) as cli:
            result = judge_phrases(video, out, dancers="lead wearing bib 42")
        judged = result["phrases"]
        assert [p["time"] for p in judged] == [2.5, 18.5, 34.5]
        assert judged[0]["acknowledged"] is True and judged[1]["acknowledged"] is False
        assert judged[2]["acknowledged"] is None and "Not returned" in judged[2]["how"]   # model skipped it
        assert judged[1]["counts"] == 32 and judged[1]["kind"] == "section" and judged[1]["response"] == "none"
        # two decoy windows on the 8-count grid, away from the real boundaries, presented the same way
        decoys = result["decoys"]
        assert len(decoys) == 2 and all(d["decoy"] for d in decoys)
        assert all(min(abs(d["time"] - t) for t in (2.5, 18.5, 34.5)) >= 4.0 for d in decoys)
        assert result["calibration"] == {"real": 3, "real_hit": 1, "decoys": 2, "decoys_hit": 0}
        prompt = cli.call_args[0][1]
        assert "at 18.5s (32 counts end here; section)" in prompt and prompt.count("Window ") == 5
        assert "lead in white shirt" in prompt and "bib 42" in prompt and "p01_000.jpg" in prompt
        saved = json.loads((out / "coach.json").read_text())
        assert saved["music"]["method"] == "structure-v2+judge"
        assert saved["music"]["phrase_starts"] == [2.5, 18.5, 34.5]
        assert saved["phrases"][0]["acknowledged"] is True and len(saved["phrase_decoys"]) == 2
        assert saved["phrase_judge_calibration"]["decoys"] == 2
        assert saved["phrase_judge_usage"]["estimated_cost"] == 1.25
        assert any("re-judged" in w and "decoy" in w for w in saved["warnings"])
        assert (out / "coach.json.prejudge.bak").exists()
        assert (out / "song_map.svg").exists() and (out / "phrase_map.json").exists()
        # the backup still holds the old verdict
        assert json.loads((out / "coach.json.prejudge.bak").read_text())["phrases"][0]["how"] == "old verdict"

    def test_blocked_burst_is_not_judged(self, tmp_path: Path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"x")
        out = tmp_path / "out"
        pm = PhraseMap(bpm=120.0, beat_times=[0.5 * i for i in range(40)], phase=0, eights=[0.0, 4.0],
                       boundaries=[PhraseBoundary(time=4.0, beat=8, counts=0, kind="start", confidence=0.5)])
        blocked = {"phrases": [{"time": 4.0, "acknowledged": True, "how": "", "visibility": "blocked", "confidence": 0.2}]}
        with patch("wcs_analyzer.coach._check_claude_cli", return_value="/bin/claude"), \
             patch("wcs_analyzer.coach.extract_audio_features", return_value=_audio(40)), \
             patch("wcs_analyzer.coach.build_phrase_map", return_value=pm), \
             patch("wcs_analyzer.coach.get_video_duration", return_value=20.0), \
             patch("wcs_analyzer.coach.extract_frames_between", return_value=_frames(24, 4.0)), \
             patch("wcs_analyzer.coach._call_claude_cli", return_value=(blocked, UsageTotals())):
            result = judge_phrases(video, out)
        assert result["phrases"][0]["acknowledged"] is None
        assert "not visible" in result["phrases"][0]["how"]
        assert not (out / "coach.json").exists()      # nothing to write back into


class TestReports:
    def test_bundle_is_one_file_with_every_song(self, tmp_path: Path):
        import shutil

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"x")
        out1 = tmp_path / "song1_coach"
        usage = UsageTotals(input_tokens=1, output_tokens=1, estimated_cost=0.25, model="claude-opus-5", pricing_known=True)
        with patch("wcs_analyzer.coach._check_claude_cli", return_value="/bin/claude"), \
             patch("wcs_analyzer.coach.extract_audio_features", return_value=_audio()), \
             patch("wcs_analyzer.coach.get_video_duration", return_value=30.0), \
             patch("wcs_analyzer.coach.extract_frames", return_value=_frames(90, 3.0)), \
             patch("wcs_analyzer.coach.extract_frames_between", return_value=_frames(24, 10.0)), \
             patch("wcs_analyzer.coach._call_claude_cli", side_effect=[(SURVEY, usage), (ZOOM, usage), (ZOOM, usage)]):
            run_coach(video, out1, zoom_moments=2)
        out2 = tmp_path / "song2_coach"
        shutil.copytree(out1, out2)

        path = write_bundle([out1, out2], tmp_path / "share" / "notes.html", title="Two songs", labels=["Song A", "Song B"])
        html = path.read_text()
        assert path.parent.name == "share"
        assert html.count("<!doctype html>") == 1 and html.count("</html>") == 1
        assert "<title>Two songs</title>" in html and "<h1 id='top'>Two songs</h1>" in html
        assert "href='#song-1'>Song A" in html and "href='#song-2'>Song B" in html
        assert "<h1 id='song-1'>Song A</h1>" in html and "<h1 id='song-2'>Song B</h1>" in html
        assert html.count("data:image/jpeg;base64,") >= 2 * (3 + 2)        # both songs' note and zoom strips inline
        assert html.count("Back to contents") == 2
        assert "<video" not in html                                        # no clips requested
        assert "AI coach" in html                                          # default intro for the reader

    def test_markdown_and_html(self, tmp_path: Path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"x")
        out = tmp_path / "out"
        usage = UsageTotals(input_tokens=1, output_tokens=1, estimated_cost=0.25, model="claude-opus-5", pricing_known=True)
        with patch("wcs_analyzer.coach._check_claude_cli", return_value="/bin/claude"), \
             patch("wcs_analyzer.coach.extract_audio_features", return_value=_audio()), \
             patch("wcs_analyzer.coach.get_video_duration", return_value=30.0), \
             patch("wcs_analyzer.coach.extract_frames", return_value=_frames(90, 3.0)), \
             patch("wcs_analyzer.coach.extract_frames_between", return_value=_frames(24, 10.0)), \
             patch("wcs_analyzer.coach._call_claude_cli", side_effect=[(SURVEY, usage), (ZOOM, usage), (ZOOM, usage)]):
            report = run_coach(video, out, zoom_moments=2)
        md_path, html_path = write_reports(report, out)

        md = md_path.read_text()
        assert "# Coaching notes: clip.mp4" in md
        assert "🟢 keep doing this" in md                       # legend
        assert "| 🟢 0:04.0<br>`keep it` |" in md              # verdict glyph on notes
        assert "| 0:05.0 | 🔴 **no** |" in md                  # missed phrase
        assert "**🟢 Fix:**" in md
        assert "## Themes a judge would notice" in md and "**Look up**" in md
        assert "| 0:05.0 | 🔴 **no** | no change of energy |" in md
        assert "Cool sweep, keep that." in md and "![frames](strips/note_00.jpg)" in md
        assert "## Slow motion, count by count" in md and "Drive straight down the slot on 1." in md
        assert "estimated cost $0.75" in md

        html = html_path.read_text()
        assert "data:image/jpeg;base64," in html
        assert "<tr class='row keep'>" in html and "<tr class='row refine'>" in html
        assert "<tr class='miss'>" in html and "<tr class='ok'>" in html
        assert "class='box good'" in html and "class='box bad'" in html
        assert "keep doing this" in html                        # legend
        assert "Who I watched" in html and "bib 42" in html
        assert "<td class='time'>0:13.0" in html

    def test_report_with_no_content_does_not_crash(self, tmp_path: Path):
        report = CoachReport(video_name="x.mp4", duration=0.0, model="claude-opus-5")
        md_path, html_path = write_reports(report, tmp_path)
        assert md_path.exists() and html_path.exists()
