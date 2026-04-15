"""Pose estimation for objective technique metrics.

Uses MediaPipe's BlazePose (single-person) to extract 33 body landmarks
per frame, then computes numeric metrics that the LLM scorers can use
as context instead of inferring technique purely from pixels.

MediaPipe is an optional dependency installed via `pip install
'wcs-analyzer[pose]'`. Metric computation functions operate on plain
landmark arrays, so they are importable and testable without MediaPipe.

Current limitation: BlazePose detects one person per frame. In couple
dances the detector locks onto whichever dancer is most prominent —
usually the lead. Posture / extension / footwork metrics are still
meaningful; slot metrics (which need the follower's trajectory) are
approximate. Multi-person tracking is a follow-up.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

from .exceptions import WCSAnalyzerError

logger = logging.getLogger(__name__)


# BlazePose landmark indices (subset we use).
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


Landmark = tuple[float, float, float]  # x, y, z in normalized [0, 1] image coords


@dataclass
class FramePose:
    """Pose landmarks extracted from a single frame.

    `landmarks` is a list of 33 (x, y, z) tuples in normalized image
    coordinates. When `detected` is False the landmarks list is empty.
    """

    timestamp: float
    detected: bool = False
    landmarks: list[Landmark] = field(default_factory=list)
    visibility: list[float] = field(default_factory=list)


@dataclass
class PoseData:
    """Pose data for a whole video."""

    frames: list[FramePose] = field(default_factory=list)
    fps: float = 0.0
    duration: float = 0.0

    @property
    def coverage(self) -> float:
        """Fraction of frames where a pose was detected."""
        if not self.frames:
            return 0.0
        detected = sum(1 for f in self.frames if f.detected)
        return detected / len(self.frames)


class PoseUnavailableError(WCSAnalyzerError):
    """Raised when MediaPipe is needed but not installed."""


def extract_poses(video_path: Path, fps: float = 5.0, max_dimension: int = 640) -> PoseData:
    """Extract pose landmarks from a video using MediaPipe BlazePose.

    Lazy-imports mediapipe so the rest of the package works without it.
    Raises PoseUnavailableError with install instructions if missing.
    """
    try:
        import cv2
        import mediapipe as mp  # type: ignore[import-not-found]
    except ImportError as e:
        raise PoseUnavailableError(
            "MediaPipe is required for pose estimation. Install with "
            "`pip install 'wcs-analyzer[pose]'` or `pip install mediapipe`."
        ) from e

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise WCSAnalyzerError(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps if original_fps > 0 else 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_interval = max(1, int(original_fps / fps))
    scale = min(max_dimension / max(width, 1), max_dimension / max(height, 1), 1.0)
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))

    pose_data = PoseData(fps=fps, duration=duration)

    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                if scale < 1.0:
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                timestamp = frame_idx / original_fps

                if result.pose_landmarks:
                    lms = [(lm.x, lm.y, lm.z) for lm in result.pose_landmarks.landmark]
                    vis = [lm.visibility for lm in result.pose_landmarks.landmark]
                    pose_data.frames.append(FramePose(
                        timestamp=timestamp,
                        detected=True,
                        landmarks=lms,
                        visibility=vis,
                    ))
                else:
                    pose_data.frames.append(FramePose(timestamp=timestamp, detected=False))

            frame_idx += 1

    cap.release()
    logger.info(
        "Extracted poses from %s: %d/%d frames detected (%.0f%% coverage)",
        video_path.name,
        sum(1 for f in pose_data.frames if f.detected),
        len(pose_data.frames),
        pose_data.coverage * 100,
    )
    return pose_data


# ---- Metric helpers (pure math, no mediapipe dependency) -------------------


def _midpoint(a: Landmark, b: Landmark) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _dist2d(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle_from_vertical(top: tuple[float, float], bottom: tuple[float, float]) -> float:
    """Angle in degrees between the line (bottom→top) and the vertical axis.

    Image Y increases downward, so "vertical" in image coords means the
    point above (smaller y) is "top". A perfectly upright spine returns 0°.
    """
    dx = top[0] - bottom[0]
    dy = bottom[1] - top[1]  # invert so positive dy = upright
    if dy == 0 and dx == 0:
        return 0.0
    return math.degrees(math.atan2(abs(dx), max(dy, 1e-9)))


def compute_posture_metric(frames: list[FramePose]) -> dict[str, float]:
    """Mean and stddev of spine deviation from vertical across detected frames."""
    angles: list[float] = []
    for f in frames:
        if not f.detected:
            continue
        lms = f.landmarks
        shoulder_mid = _midpoint(lms[LEFT_SHOULDER], lms[RIGHT_SHOULDER])
        hip_mid = _midpoint(lms[LEFT_HIP], lms[RIGHT_HIP])
        angles.append(_angle_from_vertical(shoulder_mid, hip_mid))

    if not angles:
        return {"mean_deg": 0.0, "stddev_deg": 0.0, "samples": 0}

    mean = sum(angles) / len(angles)
    var = sum((a - mean) ** 2 for a in angles) / len(angles)
    return {"mean_deg": mean, "stddev_deg": math.sqrt(var), "samples": len(angles)}


def compute_extension_metric(frames: list[FramePose]) -> dict[str, float]:
    """Mean arm extension ratio (wrist-to-shoulder / torso length) across frames."""
    ratios: list[float] = []
    for f in frames:
        if not f.detected:
            continue
        lms = f.landmarks
        shoulder_mid = _midpoint(lms[LEFT_SHOULDER], lms[RIGHT_SHOULDER])
        hip_mid = _midpoint(lms[LEFT_HIP], lms[RIGHT_HIP])
        torso = _dist2d(shoulder_mid, hip_mid)
        if torso < 1e-6:
            continue
        l_arm = _dist2d((lms[LEFT_WRIST][0], lms[LEFT_WRIST][1]),
                        (lms[LEFT_SHOULDER][0], lms[LEFT_SHOULDER][1]))
        r_arm = _dist2d((lms[RIGHT_WRIST][0], lms[RIGHT_WRIST][1]),
                        (lms[RIGHT_SHOULDER][0], lms[RIGHT_SHOULDER][1]))
        ratios.append((l_arm + r_arm) / (2 * torso))

    if not ratios:
        return {"mean_ratio": 0.0, "peak_ratio": 0.0, "samples": 0}

    return {
        "mean_ratio": sum(ratios) / len(ratios),
        "peak_ratio": max(ratios),
        "samples": len(ratios),
    }


def _ankle_footfalls(detected: list[FramePose], idx: int) -> list[float]:
    """Return timestamps where the given ankle's y-velocity crosses from
    rising (negative vy) to descending (positive vy) — i.e. a footfall.

    The returned times are interpolated between the two sample times to
    reduce quantization error from the pose sample rate.
    """
    if len(detected) < 3:
        return []
    times: list[float] = []
    ys = [f.landmarks[idx][1] for f in detected]
    for i in range(len(ys) - 2):
        v0 = ys[i + 1] - ys[i]
        v1 = ys[i + 2] - ys[i + 1]
        if v0 < 0 <= v1:
            t0 = detected[i + 1].timestamp
            t1 = detected[i + 2].timestamp
            denom = v1 - v0
            alpha = (-v0) / denom if denom > 0 else 0.0
            times.append(t0 + alpha * (t1 - t0))
    return times


def compute_footfall_times(frames: list[FramePose]) -> list[float]:
    """Detect footfall timestamps by merging left/right ankle events.

    Returns a sorted list of times (in seconds) that represent
    approximate footfall moments — useful as input to beat-sync
    verification.
    """
    detected = [f for f in frames if f.detected]
    if len(detected) < 3:
        return []
    left = _ankle_footfalls(detected, LEFT_ANKLE)
    right = _ankle_footfalls(detected, RIGHT_ANKLE)
    merged = sorted(left + right)
    # Collapse near-duplicate events (left+right hitting within 50 ms)
    collapsed: list[float] = []
    for t in merged:
        if collapsed and t - collapsed[-1] < 0.05:
            collapsed[-1] = (collapsed[-1] + t) / 2
        else:
            collapsed.append(t)
    return collapsed


def compute_footwork_metric(frames: list[FramePose]) -> dict[str, float]:
    """Approximate footfall rate by counting ankle-y zero-crossings.

    A footfall is detected when the ankle's vertical velocity crosses
    zero from negative (rising) to positive (descending). We average
    across left and right ankles and express as steps per second.
    """
    detected = [f for f in frames if f.detected]
    if len(detected) < 3:
        return {"steps_per_second": 0.0, "samples": len(detected)}

    times = [f.timestamp for f in detected]
    duration = times[-1] - times[0]

    def step_rate(idx: int) -> float:
        events = _ankle_footfalls(detected, idx)
        return len(events) / duration if duration > 0 else 0.0

    left = step_rate(LEFT_ANKLE)
    right = step_rate(RIGHT_ANKLE)
    return {
        "steps_per_second": (left + right) / 2,
        "left_steps_per_second": left,
        "right_steps_per_second": right,
        "samples": len(detected),
    }


def compute_beat_sync(
    footfalls: list[float],
    beat_times: list[float],
    bpm: float,
) -> dict[str, float]:
    """Cross-correlate footfall times with audio beats and produce a
    timing score in [0, 10].

    For each footfall we find the nearest beat and compute the absolute
    offset. The score is 10 when every footfall lands exactly on a beat
    and 0 when the mean offset reaches half a beat interval (i.e. the
    worst possible case — on the off-beat). Footfalls that sit further
    than `beat_interval` from any beat are counted at the cap.
    """
    if not footfalls or not beat_times or bpm <= 0:
        return {
            "timing_score": 0.0,
            "mean_offset_ms": 0.0,
            "beat_interval_ms": 0.0,
            "on_beat_fraction": 0.0,
            "footfalls": float(len(footfalls)),
        }

    beat_interval = 60.0 / bpm
    half_interval = beat_interval / 2

    offsets: list[float] = []
    on_beat = 0
    sorted_beats = sorted(beat_times)

    for ft in footfalls:
        # Binary search would be nice, but the lists are small — use the
        # linear scan to keep the pure-math module dependency-free.
        nearest = min(sorted_beats, key=lambda b: abs(b - ft))
        offset = abs(ft - nearest)
        capped = min(offset, beat_interval)
        offsets.append(capped)
        if capped <= 0.10:  # within 100 ms counts as on-beat
            on_beat += 1

    mean_offset = sum(offsets) / len(offsets)
    # Normalize: 0 offset -> 10, half_interval offset -> 0
    normalized = max(0.0, 1.0 - (mean_offset / half_interval)) if half_interval > 0 else 0.0
    return {
        "timing_score": round(normalized * 10, 2),
        "mean_offset_ms": round(mean_offset * 1000, 1),
        "beat_interval_ms": round(beat_interval * 1000, 1),
        "on_beat_fraction": round(on_beat / len(footfalls), 3),
        "footfalls": float(len(footfalls)),
    }


def compute_slot_metric(frames: list[FramePose]) -> dict[str, float]:
    """Linearity (R²) of hip-midpoint trajectory across the video.

    In WCS the follower travels along a "slot" — roughly a straight
    line. With single-person pose we can only measure whichever dancer
    the detector locked onto, so treat this as indicative, not
    definitive. A perfectly straight path yields R² = 1.
    """
    points: list[tuple[float, float]] = []
    for f in frames:
        if not f.detected:
            continue
        mid = _midpoint(f.landmarks[LEFT_HIP], f.landmarks[RIGHT_HIP])
        points.append(mid)

    n = len(points)
    if n < 3:
        return {"linearity_r2": 0.0, "samples": n}

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cxx = sum((x - mean_x) ** 2 for x in xs) / n
    cyy = sum((y - mean_y) ** 2 for y in ys) / n
    cxy = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n)) / n

    trace = cxx + cyy
    if trace <= 1e-12:
        # Dancer didn't move — degenerate, treat as perfectly linear.
        return {"linearity_r2": 1.0, "samples": n}

    # PCA on the 2x2 covariance matrix; R² = largest eigenvalue / trace.
    disc = max(0.0, (cxx - cyy) ** 2 + 4 * cxy * cxy)
    lambda1 = (trace + math.sqrt(disc)) / 2
    r2 = min(1.0, max(0.0, lambda1 / trace))
    return {"linearity_r2": r2, "samples": n}


def compute_all_metrics(
    pose_data: PoseData,
    beat_times: list[float] | None = None,
    bpm: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Run every metric and return a single dict for prompt formatting.

    If `beat_times` and `bpm` are provided, also computes a beat-sync
    score by cross-correlating detected footfalls with the audio beats.
    """
    metrics: dict[str, dict[str, float]] = {
        "posture": compute_posture_metric(pose_data.frames),
        "extension": compute_extension_metric(pose_data.frames),
        "footwork": compute_footwork_metric(pose_data.frames),
        "slot": compute_slot_metric(pose_data.frames),
        "coverage": {"fraction": pose_data.coverage, "frames": float(len(pose_data.frames))},
    }
    if beat_times and bpm > 0:
        footfalls = compute_footfall_times(pose_data.frames)
        metrics["beat_sync"] = compute_beat_sync(footfalls, beat_times, bpm)
    return metrics


def format_pose_context(metrics: dict[str, dict[str, float]]) -> str:
    """Format pose metrics as a prompt context block for the LLM.

    The LLM treats these as measured ground truth for technique and
    should weight them when scoring posture / extension / footwork / slot.
    """
    posture = metrics.get("posture", {})
    extension = metrics.get("extension", {})
    footwork = metrics.get("footwork", {})
    slot = metrics.get("slot", {})
    coverage = metrics.get("coverage", {})
    beat_sync = metrics.get("beat_sync")

    lines = [
        "MEASURED POSE METRICS (MediaPipe BlazePose, single-person detection):",
        f"- Spine deviation from vertical: mean {posture.get('mean_deg', 0):.1f}°, "
        f"stddev {posture.get('stddev_deg', 0):.1f}° — lower is more upright",
        f"- Arm extension ratio (arm length / torso length): mean "
        f"{extension.get('mean_ratio', 0):.2f}, peak {extension.get('peak_ratio', 0):.2f} "
        f"— higher means fuller extension",
        f"- Footfall rate: {footwork.get('steps_per_second', 0):.2f} steps/sec "
        f"(WCS triple-step target ~2.5 at 120 BPM)",
        f"- Hip trajectory linearity (R²): {slot.get('linearity_r2', 0):.2f} "
        f"— slot discipline; 1.0 is a perfect line",
        f"- Pose detection coverage: {coverage.get('fraction', 0) * 100:.0f}% of sampled frames",
    ]

    if beat_sync:
        lines.extend([
            "",
            "MEASURED BEAT SYNCHRONIZATION "
            "(footfalls cross-correlated with librosa-detected beats):",
            f"- Objective timing score: {beat_sync.get('timing_score', 0):.1f} / 10",
            f"- Mean footfall-to-nearest-beat offset: "
            f"{beat_sync.get('mean_offset_ms', 0):.0f} ms "
            f"(beat interval {beat_sync.get('beat_interval_ms', 0):.0f} ms)",
            f"- On-beat fraction (within ±100 ms): "
            f"{beat_sync.get('on_beat_fraction', 0) * 100:.0f}%",
            f"- Detected footfalls: {int(beat_sync.get('footfalls', 0))}",
            "",
            "The objective timing score above is MEASURED from the audio and the "
            "dancer's footfalls — your `timing.score` should stay within ±1.0 of it "
            "unless you have a strong musical-interpretation reason to differ. "
            "Explain any deviation in `timing.reasoning`.",
        ])

    lines.extend([
        "",
        "Treat these as objective measurements. Use them to inform your posture, "
        "extension, footwork, and slot scores rather than inferring from pixels alone. "
        "Note: detection is single-person, so metrics reflect whichever dancer the "
        "detector locked onto (typically the lead).",
    ])
    return "\n".join(lines)
