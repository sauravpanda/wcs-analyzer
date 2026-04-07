"""Audio extraction and beat detection."""

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np


@dataclass
class AudioFeatures:
    """Extracted audio features from a video."""

    bpm: float = 0.0
    beat_times: list[float] = field(default_factory=list)  # seconds
    beat_strengths: list[float] = field(default_factory=list)
    duration: float = 0.0
    downbeat_times: list[float] = field(default_factory=list)  # phrase starts


def extract_audio_features(video_path: Path) -> AudioFeatures:
    """Extract audio from video and detect beats.

    Uses ffmpeg to extract audio, then librosa for beat detection.

    Args:
        video_path: Path to the video file.

    Returns:
        AudioFeatures with BPM, beat timestamps, and downbeats.
    """
    # Extract audio to temp WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        subprocess.run(
            [
                "ffmpeg", "-i", str(video_path),
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "22050", "-ac", "1",
                "-y", tmp_path,
            ],
            capture_output=True,
            check=True,
        )

        # Load audio
        y, sr = librosa.load(tmp_path, sr=22050)
        duration = librosa.get_duration(y=y, sr=sr)

        # Beat detection
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

        # Get beat strengths from onset envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        beat_strengths = []
        for bf in beat_frames:
            if bf < len(onset_env):
                beat_strengths.append(float(onset_env[bf]))
            else:
                beat_strengths.append(0.0)

        # Normalize strengths to 0-1
        if beat_strengths:
            max_s = max(beat_strengths)
            if max_s > 0:
                beat_strengths = [s / max_s for s in beat_strengths]

        # Estimate downbeats (every 4 beats for common time)
        downbeat_times = beat_times[::4] if len(beat_times) >= 4 else beat_times[:1]

        # Handle tempo as scalar
        bpm = float(np.atleast_1d(tempo)[0])

        return AudioFeatures(
            bpm=bpm,
            beat_times=beat_times,
            beat_strengths=beat_strengths,
            duration=duration,
            downbeat_times=downbeat_times,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def format_beat_context(audio: AudioFeatures, start_time: float, end_time: float) -> str:
    """Format beat information for a time segment as text context for the LLM.

    Args:
        audio: Full audio features.
        start_time: Segment start in seconds.
        end_time: Segment end in seconds.

    Returns:
        Human-readable beat context string.
    """
    segment_beats = [
        (t, s) for t, s in zip(audio.beat_times, audio.beat_strengths)
        if start_time <= t < end_time
    ]

    lines = [
        f"Tempo: {audio.bpm:.0f} BPM",
        f"Segment: {start_time:.1f}s - {end_time:.1f}s",
        f"Beats in segment: {len(segment_beats)}",
    ]

    if segment_beats:
        lines.append("Beat timestamps (seconds):")
        for i, (t, s) in enumerate(segment_beats, 1):
            strength = "strong" if s > 0.7 else "medium" if s > 0.4 else "light"
            lines.append(f"  Beat {i}: {t:.2f}s ({strength})")

    return "\n".join(lines)
