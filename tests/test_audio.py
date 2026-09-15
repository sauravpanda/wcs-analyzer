"""Tests for audio beat context formatting."""

import pytest

from wcs_analyzer.audio import AudioFeatures, fold_tempo, format_beat_context


@pytest.mark.parametrize("raw, expected", [
    (50.7, 101.4),   # halved by the tracker
    (57.4, 114.8),
    (184.6, 92.3),   # doubled by the tracker
    (172.3, 86.15),
    (103.4, 103.4),  # already in the danced band
    (72.0, 72.0),
    (150.0, 150.0),
    (30.0, 120.0),   # two octaves off still folds in
    (0.0, 0.0),      # no tempo stays no tempo
])
def test_fold_tempo(raw, expected):
    assert fold_tempo(raw) == pytest.approx(expected)


def test_format_beat_context():
    audio = AudioFeatures(
        bpm=100.0,
        beat_times=[1.0, 1.5, 2.0, 2.5, 3.0, 5.0],
        beat_strengths=[0.9, 0.3, 0.8, 0.5, 0.2, 0.7],
        duration=10.0,
    )

    result = format_beat_context(audio, start_time=1.0, end_time=3.5)

    assert "100 BPM" in result
    assert "1.0s - 3.5s" in result
    assert "Beats in segment: 5" in result  # beats at 1.0, 1.5, 2.0, 2.5, 3.0 (3.0 < 3.5)
    assert "strong" in result  # 0.9 > 0.7
    assert "light" in result  # 0.3 < 0.4


def test_format_beat_context_no_beats():
    audio = AudioFeatures(bpm=120.0, beat_times=[], beat_strengths=[], duration=5.0)
    result = format_beat_context(audio, 0.0, 5.0)
    assert "Beats in segment: 0" in result
