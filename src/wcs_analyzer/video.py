"""Video frame extraction and sampling."""

import base64
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .exceptions import VideoProcessingError

logger = logging.getLogger(__name__)

# Safety limit: max total frames to extract. At ~1600 tokens each, 500
# frames ≈ 800k tokens which far exceeds a single call but caps memory.
_MAX_TOTAL_FRAMES = 500


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

    original_fps = cap.get(cv2.CAP_PROP_FPS)
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
