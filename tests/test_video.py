"""Tests for video frame grouping logic."""

from wcs_analyzer.video import FrameData, group_frames_by_phrase


def test_group_frames_by_phrase():
    """Frames should be grouped into 8-count phrases based on BPM."""
    frames = FrameData(
        images=[f"img_{i}" for i in range(12)],
        timestamps=[i * 0.5 for i in range(12)],  # 0.0, 0.5, 1.0, ... 5.5
        fps_original=30.0,
        fps_sampled=2.0,
        duration=6.0,
        width=640,
        height=480,
    )

    # At 120 BPM, 8 beats = 4 seconds per phrase
    phrases = group_frames_by_phrase(frames, beats_per_phrase=8, bpm=120.0)

    assert len(phrases) == 2  # 6s / 4s = 1.5 -> 2 phrases
    assert phrases[0]["phrase_index"] == 0
    assert phrases[1]["phrase_index"] == 1
    # First phrase: 0-4s should have frames at 0.0, 0.5, 1.0, ..., 3.5 = 8 frames
    assert len(phrases[0]["images"]) == 8
    # Second phrase: 4-6s should have frames at 4.0, 4.5, 5.0, 5.5 = 4 frames
    assert len(phrases[1]["images"]) == 4


def test_group_frames_empty():
    frames = FrameData(duration=0.0)
    phrases = group_frames_by_phrase(frames, beats_per_phrase=8, bpm=120.0)
    assert phrases == []
