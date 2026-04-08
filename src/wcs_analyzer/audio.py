"""Audio extraction and beat detection for WCS analysis."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import ffmpeg
import librosa
import numpy as np


@dataclass
class AudioAnalysis:
    bpm: float
    beat_times: list[float]  # seconds
    duration: float  # seconds

    @property
    def is_wcs_tempo(self) -> bool:
        """WCS music is typically 28–34 BPM (slow) or 84–102 BPM (fast)."""
        return (28 <= self.bpm <= 38) or (80 <= self.bpm <= 108)

    def beats_in_range(self, start: float, end: float) -> list[float]:
        return [t for t in self.beat_times if start <= t <= end]


def extract_audio(video_path: Path) -> Path:
    """Extract audio from video to a temp WAV file. Caller owns the tempfile."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    (
        ffmpeg
        .input(str(video_path))
        .output(tmp.name, acodec="pcm_s16le", ac=1, ar="22050", loglevel="quiet")
        .overwrite_output()
        .run()
    )
    return Path(tmp.name)


def analyze_beats(audio_path: Path) -> AudioAnalysis:
    """Run librosa beat tracking and return timing data."""
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    bpm = float(np.mean(tempo_arr)) if hasattr(tempo_arr, "__len__") else float(tempo_arr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    return AudioAnalysis(bpm=bpm, beat_times=beat_times, duration=duration)


def analyze_audio_from_video(video_path: Path) -> AudioAnalysis:
    """Convenience: extract audio from video then analyze it."""
    audio_path = extract_audio(video_path)
    try:
        return analyze_beats(audio_path)
    finally:
        audio_path.unlink(missing_ok=True)
