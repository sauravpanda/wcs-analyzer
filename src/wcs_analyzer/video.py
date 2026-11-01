"""Video frame extraction and sampling."""

import base64
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2

from .exceptions import VideoProcessingError

logger = logging.getLogger(__name__)

# Safety limit: max total frames to extract. At ~1600 tokens each, 500
# frames ≈ 800k tokens which far exceeds a single call but caps memory.
_MAX_TOTAL_FRAMES = 500

# Assumed tempo when beat detection finds nothing (silent clip, no audio
# stream). Typical WCS music sits around 90-120 BPM.
_FALLBACK_BPM = 120.0

# Frames read from the start of a clip to measure the real frame spacing.
_CLOCK_PROBE_FRAMES = 30


def measure_frame_rate(video_path: Path, reported_fps: float) -> float:
    """Frame rate measured from the decoder's own frame timestamps.

    Some iPhone HEVC files advertise an average frame rate in their header
    (24.95 fps) that does not match the spacing of the frames themselves
    (30 fps). Dividing a frame index by the header rate then stretches every
    timestamp and the duration by 20% and puts the frames out of step with
    the audio, which is timed from the real clock. So measure the spacing of
    the first frames and fall back to the header rate only when the decoder
    gives no usable clock.
    """
    cap = cv2.VideoCapture(str(video_path))
    first: float | None = None
    last: float | None = None
    n = 0
    try:
        while n < _CLOCK_PROBE_FRAMES and cap.grab():
            t = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            if first is None:
                first = t
            last = t
            n += 1
    finally:
        cap.release()
    if first is None or last is None or n < 2 or last <= first:
        return reported_fps
    measured = (n - 1) * 1000.0 / (last - first)
    if reported_fps > 0 and abs(measured - reported_fps) / reported_fps < 0.01:
        return reported_fps  # keep the exact header value when it agrees
    logger.info(
        "%s: header says %.2f fps but frames are spaced at %.2f fps; using the measured rate",
        video_path.name, reported_fps, measured,
    )
    return measured


@dataclass
class FrameData:
    """Extracted frames from a video with metadata."""

    images: list[str] = field(default_factory=list)  # base64-encoded JPEGs
    timestamps: list[float] = field(default_factory=list)  # seconds
    fps_original: float = 0.0
    fps_sampled: float = 0.0
    duration: float = 0.0
    width: int = 0
    height: int = 0


def extract_frames(video_path: Path, fps: float = 3.0, max_dimension: int = 768) -> FrameData:
    """Extract frames from video at the given sample rate.

    Args:
        video_path: Path to the video file.
        fps: Frames per second to sample.
        max_dimension: Max width/height for resized frames (keeps aspect ratio).

    Returns:
        FrameData with base64-encoded frames and timestamps.
    """
    logger.debug("Opening video: %s", video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoProcessingError(f"Cannot open video: {video_path}")

    original_fps = measure_frame_rate(video_path, cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps if original_fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate frame interval for desired sample rate
    frame_interval = max(1, int(original_fps / fps))

    data = FrameData(
        fps_original=original_fps,
        fps_sampled=fps,
        duration=duration,
        width=width,
        height=height,
    )

    # Calculate resize dimensions
    scale = min(max_dimension / width, max_dimension / height, 1.0)
    new_w = int(width * scale)
    new_h = int(height * scale)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Resize
            if scale < 1.0:
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Encode as JPEG
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

            data.images.append(b64)
            data.timestamps.append(frame_idx / original_fps)

            if len(data.images) >= _MAX_TOTAL_FRAMES:
                logger.warning(
                    "Reached max frame limit (%d). Consider lowering --fps or trimming the video.",
                    _MAX_TOTAL_FRAMES,
                )
                break

        frame_idx += 1

    cap.release()
    logger.info(
        "Extracted %d frames from %s (%.1fs, %dx%d, sampled at %.1f fps)",
        len(data.images), video_path, duration, width, height, fps,
    )
    return data


def get_video_recorded_at(video_path: Path) -> str:
    """Best-effort extraction of the video's recording timestamp.

    Strategy:
    1. Ask ffprobe for the container's `creation_time` tag (most
       accurate for iPhone MOV / MP4 files).
    2. Fall back to the file's birth time on macOS (`st_birthtime`)
       or modification time elsewhere.

    Returns an ISO 8601 UTC string, or "" if nothing could be determined.
    """
    # 1. Try ffprobe
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format_tags=creation_time",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=15,
        )
        ts = result.stdout.strip()
        if ts:
            return ts
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 2. Fall back to OS file metadata
    try:
        stat = os.stat(video_path)
        # macOS exposes birth time; Linux doesn't always
        birth = getattr(stat, "st_birthtime", None)
        t = birth if birth else stat.st_mtime
        return datetime.fromtimestamp(t, tz=timezone.utc).isoformat(timespec="seconds")
    except OSError:
        return ""


def get_video_duration(video_path: Path) -> float:
    """Return the duration of a video in seconds without extracting frames.

    Used by the Gemini path (which hands the whole video to the model
    and doesn't otherwise need a frame decode) to validate the clip
    length before paying for an upload.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoProcessingError(f"Cannot open video: {video_path}")
    fps = measure_frame_rate(video_path, cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count / fps if fps > 0 else 0.0


def group_frames_by_phrase(
    frames: FrameData, beats_per_phrase: int = 8, bpm: float = 120.0
) -> list[dict]:
    """Group frames into musical phrases (e.g., 8-count segments).

    Args:
        frames: Extracted frame data.
        beats_per_phrase: Beats per phrase (8 for WCS 8-count patterns).
        bpm: Tempo in beats per minute.

    Returns:
        List of dicts with 'images', 'timestamps', 'phrase_index', 'start_time', 'end_time'.
    """
    if bpm <= 0:
        # No tempo detected (silent video or failed beat tracking). Fall
        # back to a typical WCS tempo so grouping still works instead of
        # dividing by zero after the CLI promised a visual-only analysis.
        logger.warning("No tempo available; grouping phrases at an assumed %.0f BPM", _FALLBACK_BPM)
        bpm = _FALLBACK_BPM
    phrase_duration = (beats_per_phrase / bpm) * 60  # seconds per phrase
    phrases = []

    phrase_start = 0.0
    phrase_idx = 0

    while phrase_start < frames.duration:
        phrase_end = phrase_start + phrase_duration
        phrase_images = []
        phrase_timestamps = []

        for img, ts in zip(frames.images, frames.timestamps):
            if phrase_start <= ts < phrase_end:
                phrase_images.append(img)
                phrase_timestamps.append(ts)

        if phrase_images:
            phrases.append({
                "images": phrase_images,
                "timestamps": phrase_timestamps,
                "phrase_index": phrase_idx,
                "start_time": phrase_start,
                "end_time": min(phrase_end, frames.duration),
            })

        phrase_start = phrase_end
        phrase_idx += 1

    return phrases


def extract_frames_between(
    video_path: Path, start: float, end: float, fps: float = 10.0, max_dimension: int = 768,
) -> FrameData:
    """Extract frames from the window [start, end] seconds at `fps`.

    Used by coach mode's zoom pass to look at a flagged moment at a rate
    high enough to see individual counts (10 fps at 120 BPM is five frames
    per beat), without paying for the whole clip at that rate.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoProcessingError(f"Cannot open video: {video_path}")

    original_fps = measure_frame_rate(video_path, cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps if original_fps > 0 else 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    data = FrameData(fps_original=original_fps, fps_sampled=fps, duration=duration, width=width, height=height)
    if original_fps <= 0:
        cap.release()
        return data

    start = max(0.0, start)
    if duration:
        end = min(end, duration)
    if end <= start:
        cap.release()
        return data

    # Seek and label by time, not by frame number. OpenCV converts a frame
    # number to a position with the header frame rate, so on a clip whose
    # header is wrong a frame-number seek lands in the wrong place or past the
    # end of the file; the decoder's own clock is the only reliable position.
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    scale = min(max_dimension / max(width, 1), max_dimension / max(height, 1), 1.0)
    new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))

    step = 1.0 / fps
    next_t = start
    idx = 0
    clock_ok = True
    first_pos = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if idx == 0:
            first_pos = pos
        elif idx == 1 and pos <= first_pos:
            clock_ok = False  # the decoder gives no usable clock; count frames from the seek
        t = pos if clock_ok else start + idx / original_fps
        idx += 1
        if t + 1e-3 < next_t:
            continue  # the seek landed a few frames early; walk up to the window
        if t > end + 1e-3:
            break
        if scale < 1.0:
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        data.images.append(base64.b64encode(buffer.tobytes()).decode("utf-8"))
        data.timestamps.append(t)
        while next_t <= t + 1e-3:
            next_t += step

    cap.release()
    logger.debug("Extracted %d burst frames from %s (%.1f-%.1fs at %.1f fps)", len(data.images), video_path.name, start, end, fps)
    return data
