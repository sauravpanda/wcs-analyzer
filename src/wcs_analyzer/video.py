"""Video frame extraction and sampling for WCS analysis."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Frame:
    timestamp: float  # seconds
    index: int
    data: bytes  # JPEG bytes
    beat_number: int | None = None

    def as_base64(self) -> str:
        return base64.b64encode(self.data).decode()


@dataclass
class Segment:
    """~8-count phrase of frames."""

    segment_index: int
    start_time: float
    end_time: float
    frames: list[Frame] = field(default_factory=list)


def extract_frames(
    video_path: Path,
    fps: float = 3.0,
    max_dimension: int = 768,
) -> tuple[list[Frame], float]:
    """Extract frames from video at given fps. Returns (frames, video_fps)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    frame_interval = int(video_fps / fps)
    frames: list[Frame] = []
    frame_index = 0
    extracted_index = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

            _, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames.append(Frame(
                timestamp=frame_index / video_fps,
                index=extracted_index,
                data=buf.tobytes(),
            ))
            extracted_index += 1

        frame_index += 1

    cap.release()
    return frames, video_fps


def assign_beats_to_frames(frames: list[Frame], beat_times: list[float]) -> None:
    """Tag each frame with the closest beat number (1-indexed)."""
    beat_arr = np.array(beat_times)
    for frame in frames:
        if len(beat_arr) == 0:
            break
        idx = int(np.argmin(np.abs(beat_arr - frame.timestamp)))
        frame.beat_number = idx + 1


def build_segments(frames: list[Frame], beats_per_segment: int = 8, bpm: float = 120.0) -> list[Segment]:
    """Group frames into segments roughly aligned to 8-count phrases."""
    if not frames:
        return []

    segment_duration = (beats_per_segment / bpm) * 60.0
    total_duration = frames[-1].timestamp + 0.1
    num_segments = max(1, int(total_duration / segment_duration))

    segments: list[Segment] = []
    for i in range(num_segments):
        start = i * segment_duration
        end = (i + 1) * segment_duration
        seg_frames = [f for f in frames if start <= f.timestamp < end]
        if seg_frames:
            segments.append(Segment(
                segment_index=i,
                start_time=start,
                end_time=end,
                frames=seg_frames,
            ))

    return segments
