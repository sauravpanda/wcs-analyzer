"""Tests for the pose metric functions.

These use synthetic landmark data so they run without MediaPipe installed.
"""

from wcs_analyzer.pose import (
    LEFT_ANKLE, LEFT_HIP, LEFT_SHOULDER, LEFT_WRIST,
    RIGHT_ANKLE, RIGHT_HIP, RIGHT_SHOULDER, RIGHT_WRIST,
    FramePose, PoseData,
    compute_all_metrics,
    compute_extension_metric,
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
