"""Tests for audio beat context formatting."""

import numpy as np
import pytest

from wcs_analyzer.audio import AudioFeatures, describe_feel, fold_tempo, format_beat_context, swing_ratio


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

# --------------------------------------------------------------------------- rhythm feel



def _onsets(offbeat, n_beats: int = 24, beat: float = 0.5, dt: float = 0.01):
    """An onset envelope with a spike on every beat and, optionally, a weaker one at `offbeat`
    of the way to the next beat (0.5 = straight eighths, 2/3 = triplet swing)."""
    times = np.arange(0.0, n_beats * beat, dt)
    env = np.zeros_like(times)
    beats = [i * beat for i in range(n_beats)]
    for b in beats:
        env[int(round(b / dt))] = 1.0
        if offbeat is not None:
            k = int(round((b + offbeat * beat) / dt))
            if k < len(env):
                env[k] = 0.6
    return env, times, beats


def test_swing_ratio_reads_straight_and_swung_offbeats():
    env, times, beats = _onsets(0.5)
    assert swing_ratio(env, times, beats) == pytest.approx(0.5, abs=0.03)
    env, times, beats = _onsets(2 / 3)
    assert swing_ratio(env, times, beats) == pytest.approx(0.667, abs=0.03)


def test_swing_ratio_is_unknown_without_offbeats_or_enough_beats():
    env, times, beats = _onsets(None)
    assert swing_ratio(env, times, beats) == 0.0          # nothing between the beats
    env, times, beats = _onsets(0.5, n_beats=6)
    assert swing_ratio(env, times, beats) == 0.0          # too few beats to trust
    assert swing_ratio(env[:-5], times, beats) == 0.0     # mismatched arrays


@pytest.mark.parametrize("ratio, feel", [
    (0.0, ""), (0.35, ""), (0.5, "straight"), (0.54, "straight"), (0.57, "light swing"), (0.6, "swung"), (0.67, "swung"),
])
def test_describe_feel(ratio, feel):
    assert describe_feel(ratio) == feel
