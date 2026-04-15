"""Tests for the pose metric functions.

These use synthetic landmark data so they run without MediaPipe installed.
"""

from wcs_analyzer.pose import (
    LEFT_ANKLE, LEFT_HIP, LEFT_SHOULDER, LEFT_WRIST,
    RIGHT_ANKLE, RIGHT_HIP, RIGHT_SHOULDER, RIGHT_WRIST,
    FramePose, PoseData,
    compute_all_metrics,
    compute_beat_sync,
    compute_extension_metric,
    compute_footfall_times,
    compute_footwork_metric,
    compute_posture_metric,
    compute_slot_metric,
    format_pose_context,
)


def _blank_landmarks() -> list[tuple[float, float, float]]:
    return [(0.0, 0.0, 0.0)] * 33


def _upright_frame(t: float, lean_x: float = 0.0, hip_y: float = 0.8) -> FramePose:
    """Build a frame with shoulders at y=0.3 and hips at y=0.8; lean_x shifts shoulders left/right."""
    lms = _blank_landmarks()
    lms[LEFT_SHOULDER] = (0.45 + lean_x, 0.30, 0.0)
    lms[RIGHT_SHOULDER] = (0.55 + lean_x, 0.30, 0.0)
    lms[LEFT_HIP] = (0.47, hip_y, 0.0)
    lms[RIGHT_HIP] = (0.53, hip_y, 0.0)
    lms[LEFT_WRIST] = (0.35, 0.45, 0.0)
    lms[RIGHT_WRIST] = (0.65, 0.45, 0.0)
    lms[LEFT_ANKLE] = (0.48, 0.95, 0.0)
    lms[RIGHT_ANKLE] = (0.52, 0.95, 0.0)
    return FramePose(
        timestamp=t, detected=True,
        landmarks=lms, visibility=[1.0] * 33,
    )


class TestPosture:
    def test_upright_is_zero_degrees(self):
        frames = [_upright_frame(t * 0.1) for t in range(10)]
        m = compute_posture_metric(frames)
        assert m["mean_deg"] == 0.0
        assert m["samples"] == 10

    def test_leaning_shoulders_increase_angle(self):
        # Shoulders shifted 0.1 right of hips over a torso of 0.5 → atan2(0.1, 0.5)
        frames = [_upright_frame(t * 0.1, lean_x=0.1) for t in range(5)]
        m = compute_posture_metric(frames)
        assert m["mean_deg"] > 5.0
        assert m["stddev_deg"] == 0.0  # constant lean

    def test_undetected_frames_skipped(self):
        frames = [
            _upright_frame(0.0),
            FramePose(timestamp=0.1, detected=False),
            _upright_frame(0.2, lean_x=0.1),
        ]
        m = compute_posture_metric(frames)
        assert m["samples"] == 2

    def test_empty_frames_returns_zero(self):
        assert compute_posture_metric([])["samples"] == 0


class TestExtension:
    def test_arm_ratio_computed(self):
        # Torso = 0.5, wrists are |shoulder.x - wrist.x| + |y-diff| → 0.10 + 0.15 = ~0.18
        frames = [_upright_frame(t * 0.1) for t in range(5)]
        m = compute_extension_metric(frames)
        assert m["mean_ratio"] > 0
        assert m["peak_ratio"] >= m["mean_ratio"]
        assert m["samples"] == 5

    def test_degenerate_torso_skipped(self):
        lms = _blank_landmarks()
        # Shoulders and hips all at same point → torso = 0, skip
        lms[LEFT_SHOULDER] = lms[RIGHT_SHOULDER] = (0.5, 0.5, 0.0)
        lms[LEFT_HIP] = lms[RIGHT_HIP] = (0.5, 0.5, 0.0)
        lms[LEFT_WRIST] = lms[RIGHT_WRIST] = (0.6, 0.6, 0.0)
        frame = FramePose(timestamp=0.0, detected=True, landmarks=lms, visibility=[1.0] * 33)
        m = compute_extension_metric([frame])
        assert m["samples"] == 0


class TestFootwork:
    def test_constant_ankle_has_zero_steps(self):
        frames = [_upright_frame(t * 0.1) for t in range(30)]
        m = compute_footwork_metric(frames)
        assert m["steps_per_second"] == 0.0

    def test_oscillating_ankle_detects_steps(self):
        frames = []
        for i in range(30):
            f = _upright_frame(i * 0.1)
            # Sinusoidal-like oscillation → zero-crossings every ~half period
            ankle_y = 0.95 + (0.03 if i % 2 == 0 else -0.03)
            lms = list(f.landmarks)
            lms[LEFT_ANKLE] = (lms[LEFT_ANKLE][0], ankle_y, 0.0)
            lms[RIGHT_ANKLE] = (lms[RIGHT_ANKLE][0], ankle_y, 0.0)
            f.landmarks = lms
            frames.append(f)
        m = compute_footwork_metric(frames)
        assert m["steps_per_second"] > 0

    def test_too_few_frames_returns_zero(self):
        m = compute_footwork_metric([_upright_frame(0.0), _upright_frame(0.1)])
        assert m["steps_per_second"] == 0.0


class TestSlot:
    def test_straight_line_high_r2(self):
        # Hip midpoint moves linearly along x while y stays constant
        frames = []
        for i in range(10):
            lms = _blank_landmarks()
            lms[LEFT_HIP] = (0.1 + i * 0.05, 0.8, 0.0)
            lms[RIGHT_HIP] = (0.12 + i * 0.05, 0.8, 0.0)
            frames.append(FramePose(
                timestamp=i * 0.1, detected=True, landmarks=lms, visibility=[1.0] * 33,
            ))
        m = compute_slot_metric(frames)
        assert m["linearity_r2"] > 0.9

    def test_stationary_dancer_is_perfectly_linear(self):
        frames = [_upright_frame(i * 0.1) for i in range(10)]
        m = compute_slot_metric(frames)
        assert m["linearity_r2"] == 1.0

    def test_too_few_samples(self):
        m = compute_slot_metric([_upright_frame(0.0)])
        assert m["samples"] == 1
        assert m["linearity_r2"] == 0.0


def _oscillating_ankle_frames(n: int, fps: float, period_frames: int, ankle_y_amp: float = 0.03) -> list[FramePose]:
    """Build n frames with ankles oscillating on a regular period.

    Rising-to-falling crossings happen once per period, at the peak
    (minimum y). This generates predictable footfall timestamps for
    beat-sync tests.
    """
    import math as _math
    frames = []
    for i in range(n):
        f = _upright_frame(i / fps)
        y = 0.95 + ankle_y_amp * _math.sin(2 * _math.pi * i / period_frames)
        lms = list(f.landmarks)
        lms[LEFT_ANKLE] = (lms[LEFT_ANKLE][0], y, 0.0)
        lms[RIGHT_ANKLE] = (lms[RIGHT_ANKLE][0], y, 0.0)
        f.landmarks = lms
        frames.append(f)
    return frames


class TestFootfallTimes:
    def test_detects_regular_footfalls(self):
        # 4 sec @ 20 fps, ankle oscillates with period 10 frames (0.5s)
        # → expect ~8 footfalls total
        frames = _oscillating_ankle_frames(n=80, fps=20.0, period_frames=10)
        falls = compute_footfall_times(frames)
        assert 6 <= len(falls) <= 10
        # Times should be monotonic and within the video duration
        assert falls == sorted(falls)
        assert all(0.0 <= t <= 4.0 for t in falls)

    def test_empty_without_motion(self):
        frames = [_upright_frame(t * 0.1) for t in range(20)]
        assert compute_footfall_times(frames) == []

    def test_too_few_frames(self):
        assert compute_footfall_times([_upright_frame(0.0)]) == []


class TestBeatSync:
    def test_perfect_alignment_scores_ten(self):
        beat_times = [0.5, 1.0, 1.5, 2.0, 2.5]
        footfalls = [0.5, 1.0, 1.5, 2.0, 2.5]
        result = compute_beat_sync(footfalls, beat_times, bpm=120.0)
        assert result["timing_score"] == 10.0
        assert result["mean_offset_ms"] == 0.0
        assert result["on_beat_fraction"] == 1.0

    def test_worst_case_off_beat_scores_zero(self):
        # Footfalls land exactly half a beat off → score should hit 0
        beat_times = [0.5, 1.0, 1.5, 2.0]
        footfalls = [0.75, 1.25, 1.75]  # 250ms off at 120 BPM (half interval)
        result = compute_beat_sync(footfalls, beat_times, bpm=120.0)
        assert result["timing_score"] == 0.0
        assert result["on_beat_fraction"] == 0.0

    def test_partial_offset_interpolates(self):
        beat_times = [0.5, 1.0, 1.5, 2.0]
        # 125 ms off at 120 BPM (beat_interval = 500 ms, half = 250 ms)
        # → normalized = 1 - 125/250 = 0.5 → score 5.0
        footfalls = [0.625, 1.125, 1.625]
        result = compute_beat_sync(footfalls, beat_times, bpm=120.0)
        assert 4.5 <= result["timing_score"] <= 5.5

    def test_empty_inputs(self):
        assert compute_beat_sync([], [1.0, 2.0], bpm=120.0)["timing_score"] == 0.0
        assert compute_beat_sync([1.0], [], bpm=120.0)["timing_score"] == 0.0
        assert compute_beat_sync([1.0], [1.0], bpm=0.0)["timing_score"] == 0.0


class TestPoseContext:
    def test_format_includes_all_metrics(self):
        data = PoseData(
            frames=[_upright_frame(t * 0.1) for t in range(10)],
            fps=5.0, duration=2.0,
        )
        metrics = compute_all_metrics(data)
        ctx = format_pose_context(metrics)
        assert "Spine deviation" in ctx
        assert "Arm extension" in ctx
        assert "Footfall" in ctx
        assert "Hip trajectory" in ctx
        assert "coverage" in ctx.lower()
        # Without beat_times, beat-sync section should be absent
        assert "BEAT SYNCHRONIZATION" not in ctx

    def test_format_includes_beat_sync_when_audio_provided(self):
        data = PoseData(
            frames=_oscillating_ankle_frames(n=80, fps=20.0, period_frames=10),
            fps=20.0, duration=4.0,
        )
        # 120 BPM → beats every 0.5 s; line up with the oscillation period
        beat_times = [i * 0.5 for i in range(9)]
        metrics = compute_all_metrics(data, beat_times=beat_times, bpm=120.0)
        assert "beat_sync" in metrics
        ctx = format_pose_context(metrics)
        assert "BEAT SYNCHRONIZATION" in ctx
        assert "Objective timing score" in ctx
        assert "timing.score" in ctx  # model instruction is present

    def test_coverage_fraction(self):
        data = PoseData(
            frames=[
                _upright_frame(0.0),
                FramePose(timestamp=0.1, detected=False),
                _upright_frame(0.2),
                _upright_frame(0.3),
            ],
            fps=10.0, duration=0.4,
        )
        assert data.coverage == 0.75


class TestPoseUnavailable:
    def test_extract_poses_raises_helpful_error_without_mediapipe(self, monkeypatch, tmp_path):
        """If mediapipe is not installed, extract_poses should raise PoseUnavailableError."""
        import sys
        from wcs_analyzer.pose import PoseUnavailableError, extract_poses

        # Simulate mediapipe missing by injecting a broken import
        monkeypatch.setitem(sys.modules, "mediapipe", None)

        video = tmp_path / "fake.mp4"
        video.write_bytes(b"not a real video")
        try:
            extract_poses(video)
            assert False, "expected PoseUnavailableError"
        except PoseUnavailableError as e:
            assert "mediapipe" in str(e).lower() or "MediaPipe" in str(e)
