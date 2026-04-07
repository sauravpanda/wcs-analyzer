"""Tests for audio beat context formatting."""

from wcs_analyzer.audio import AudioFeatures, format_beat_context


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
