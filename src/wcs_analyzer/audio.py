"""Audio extraction and beat detection."""

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np

from .exceptions import AudioProcessingError

logger = logging.getLogger(__name__)

# Expected BPM range for WCS music
WCS_BPM_MIN = 60
WCS_BPM_MAX = 200


@dataclass
class AudioFeatures:
    """Extracted audio features from a video."""

    bpm: float = 0.0
    beat_times: list[float] = field(default_factory=list)  # seconds
    beat_strengths: list[float] = field(default_factory=list)
    duration: float = 0.0
    downbeat_times: list[float] = field(default_factory=list)  # phrase starts


def _check_audio_stream(video_path: Path) -> bool:
    """Check if the video file contains an audio stream."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return bool(result.stdout.strip())
    except FileNotFoundError:
        raise AudioProcessingError(
            "ffprobe not found. Please install ffmpeg: https://ffmpeg.org/download.html"
        )
    except subprocess.TimeoutExpired:
        raise AudioProcessingError(f"Timed out checking audio streams in {video_path}")


def extract_audio_features(video_path: Path) -> AudioFeatures:
    """Extract audio from video and detect beats.

    Uses ffmpeg to extract audio, then librosa for beat detection.

    Args:
        video_path: Path to the video file.

    Returns:
        AudioFeatures with BPM, beat timestamps, and downbeats.

    Raises:
        AudioProcessingError: If audio extraction or processing fails.
    """
    if not _check_audio_stream(video_path):
        logger.warning("No audio stream found in %s — returning empty audio features", video_path)
        return AudioFeatures()

    # Extract audio to temp WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        try:
            subprocess.run(
                [
                    "ffmpeg", "-i", str(video_path),
                    "-vn", "-acodec", "pcm_s16le",
                    "-ar", "22050", "-ac", "1",
                    "-y", tmp_path,
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
            )
        except FileNotFoundError:
            raise AudioProcessingError(
                "ffmpeg not found. Please install ffmpeg: https://ffmpeg.org/download.html"
            )
        except subprocess.CalledProcessError as e:
            raise AudioProcessingError(
                f"ffmpeg failed to extract audio: {e.stderr.strip()}"
            )
        except subprocess.TimeoutExpired:
            raise AudioProcessingError(
                f"ffmpeg timed out extracting audio from {video_path}"
            )

        # Load audio
        y, sr = librosa.load(tmp_path, sr=22050)

        if len(y) == 0:
            logger.warning("Audio track is empty in %s", video_path)
            return AudioFeatures()

        duration = librosa.get_duration(y=y, sr=sr)

        # Beat detection
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        if len(beat_frames) == 0:
            logger.warning("No beats detected in %s — audio may be too quiet or non-musical", video_path)
            bpm = float(np.atleast_1d(tempo)[0])
            return AudioFeatures(bpm=bpm, duration=duration)

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

        if bpm < WCS_BPM_MIN or bpm > WCS_BPM_MAX:
            logger.warning(
                "Detected BPM (%.0f) is outside typical WCS range (%d-%d). "
                "The music may not be West Coast Swing.",
                bpm, WCS_BPM_MIN, WCS_BPM_MAX,
            )

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
