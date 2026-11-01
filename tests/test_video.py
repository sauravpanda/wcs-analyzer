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


def test_get_video_duration_returns_seconds():
    """get_video_duration should return a positive float for a real video."""
    from wcs_analyzer.video import get_video_duration
    from pathlib import Path
    # Use IMG_8665.MOV if present in the repo, otherwise skip
    sample = Path("IMG_8665.MOV")
    if not sample.exists():
        import pytest
        pytest.skip("No sample video present")
    duration = get_video_duration(sample)
    assert duration > 0


def test_get_video_duration_raises_on_missing_file():
    """Nonexistent file should raise VideoProcessingError."""
    from pathlib import Path
    import pytest
    from wcs_analyzer.exceptions import VideoProcessingError
    from wcs_analyzer.video import get_video_duration
    with pytest.raises(VideoProcessingError):
        get_video_duration(Path("/nonexistent/nope.mp4"))


def test_group_frames_zero_bpm_falls_back_instead_of_crashing():
    """Silent clips yield bpm=0 from audio extraction; grouping must not divide by zero."""
    frames = FrameData(images=["a", "b"], timestamps=[0.0, 3.0], duration=6.0)
    phrases = group_frames_by_phrase(frames, beats_per_phrase=8, bpm=0.0)
    assert phrases  # grouped at the fallback tempo (120 BPM -> 4s phrases)
    assert all(p["end_time"] > p["start_time"] for p in phrases)
    assert sum(len(p["images"]) for p in phrases) == 2


class _FakeCapture:
    """Stands in for cv2.VideoCapture on a clip whose header advertises one
    frame rate while its frames are spaced at another (the iPhone HEVC case)."""

    def __init__(self, header_fps=24.95, true_fps=30.0, total=3006, clock=True, seek_early=0.0):
        self.header_fps, self.true_fps, self.total, self.clock = header_fps, true_fps, total, clock
        self.seek_early = seek_early
        self.idx = -1  # index of the last frame handed out
        self.seeks: list[int] = []

    def isOpened(self):
        return True

    def grab(self):
        if self.idx + 1 >= self.total:
            return False
        self.idx += 1
        return True

    def read(self):
        import numpy as np
        if not self.grab():
            return False, None
        return True, np.zeros((8, 8, 3), np.uint8)

    def get(self, prop):
        import cv2
        if prop == cv2.CAP_PROP_FPS:
            return self.header_fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.total)
        if prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
            return 8.0
        if prop == cv2.CAP_PROP_POS_MSEC:
            return max(self.idx, 0) / self.true_fps * 1000.0 if self.clock else 0.0
        return 0.0

    def set(self, prop, value):
        import cv2
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.seeks.append(int(value))
            self.idx = int(value) - 1
        if prop == cv2.CAP_PROP_POS_MSEC:
            self.seeks.append(float(value))
            # a real decoder lands on a keyframe at or before the request
            self.idx = round((value / 1000.0 - self.seek_early) * self.true_fps) - 1
        return True

    def release(self):
        pass


def _fake_video(monkeypatch, **kwargs):
    import cv2
    created: list[_FakeCapture] = []

    def factory(_path):
        cap = _FakeCapture(**kwargs)
        created.append(cap)
        return cap

    monkeypatch.setattr(cv2, "VideoCapture", factory)
    return created


def test_measure_frame_rate_uses_decoder_clock(monkeypatch):
    """A header that says 24.95 fps loses to frames spaced at 30 fps."""
    import pytest
    from pathlib import Path
    from wcs_analyzer.video import measure_frame_rate
    _fake_video(monkeypatch)
    assert measure_frame_rate(Path("clip.mov"), 24.95) == pytest.approx(30.0)


def test_measure_frame_rate_keeps_header_without_a_clock(monkeypatch):
    """When the decoder reports no timestamps, the header rate stands."""
    from pathlib import Path
    from wcs_analyzer.video import measure_frame_rate
    _fake_video(monkeypatch, clock=False)
    assert measure_frame_rate(Path("clip.mov"), 24.95) == 24.95


def test_measure_frame_rate_keeps_exact_header_when_it_agrees(monkeypatch):
    from pathlib import Path
    from wcs_analyzer.video import measure_frame_rate
    _fake_video(monkeypatch, header_fps=29.97, true_fps=30.0)
    assert measure_frame_rate(Path("clip.mov"), 29.97) == 29.97


def test_extract_frames_times_frames_by_measured_rate(monkeypatch):
    """3006 frames at 30 fps is a 100 s clip, not 120 s, whatever the header says."""
    import pytest
    from pathlib import Path
    from wcs_analyzer.video import extract_frames
    _fake_video(monkeypatch)
    frames = extract_frames(Path("clip.mov"), fps=3.0)
    assert frames.duration == pytest.approx(100.2)
    assert frames.fps_original == pytest.approx(30.0)
    assert len(frames.images) == 301
    assert frames.timestamps[:3] == pytest.approx([0.0, 1 / 3, 2 / 3])
    assert frames.timestamps[-1] == pytest.approx(100.0)


def test_get_video_duration_uses_measured_rate(monkeypatch):
    import pytest
    from pathlib import Path
    from wcs_analyzer.video import get_video_duration
    _fake_video(monkeypatch)
    assert get_video_duration(Path("clip.mov")) == pytest.approx(100.2)


def test_extract_frames_between_seeks_by_time(monkeypatch):
    """A zoom at 10 s seeks to 10 000 ms and labels frames from the decoder clock."""
    import pytest
    from pathlib import Path
    from wcs_analyzer.video import extract_frames_between
    caps = _fake_video(monkeypatch)
    burst = extract_frames_between(Path("clip.mov"), 10.0, 10.5, fps=10.0)
    assert [c.seeks for c in caps if c.seeks] == [[10000.0]]
    assert burst.timestamps == pytest.approx([10.0 + i * 0.1 for i in range(6)])


def test_extract_frames_between_walks_up_from_an_early_landing(monkeypatch):
    """When the seek lands on a keyframe before the window, frames before it are skipped."""
    import pytest
    from pathlib import Path
    from wcs_analyzer.video import extract_frames_between
    _fake_video(monkeypatch, seek_early=0.27)
    burst = extract_frames_between(Path("clip.mov"), 10.0, 10.5, fps=10.0)
    assert burst.timestamps[0] == pytest.approx(10.0, abs=0.02)
    assert len(burst.timestamps) == 6


def test_extract_frames_between_counts_frames_without_a_clock(monkeypatch):
    """A decoder with no position clock still yields a window counted from the seek."""
    from pathlib import Path
    from wcs_analyzer.video import extract_frames_between
    _fake_video(monkeypatch, clock=False, header_fps=30.0)
    burst = extract_frames_between(Path("clip.mov"), 10.0, 10.5, fps=10.0)
    assert 5 <= len(burst.images) <= 6
    assert all(10.0 <= t <= 10.51 for t in burst.timestamps)
