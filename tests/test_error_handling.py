"""Tests for error handling and validation."""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from wcs_analyzer.audio import extract_audio_features, _check_audio_stream
from wcs_analyzer.exceptions import AudioProcessingError, VideoProcessingError
from wcs_analyzer.video import extract_frames
from wcs_analyzer.cli import main, _validate_fps


class TestAudioErrorHandling:
    def test_ffmpeg_not_found_raises(self, tmp_path: Path):
        video = tmp_path / "test.mp4"
        video.touch()
        with patch("wcs_analyzer.audio.subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(AudioProcessingError, match="ffprobe not found"):
                _check_audio_stream(video)

    def test_no_audio_stream_returns_empty(self, tmp_path: Path):
        video = tmp_path / "test.mp4"
        video.touch()
        with patch("wcs_analyzer.audio._check_audio_stream", return_value=False):
            result = extract_audio_features(video)
            assert result.bpm == 0.0
            assert result.beat_times == []

    def test_ffmpeg_extraction_failure(self, tmp_path: Path):
        video = tmp_path / "test.mp4"
        video.touch()
        with (
            patch("wcs_analyzer.audio._check_audio_stream", return_value=True),
            patch(
                "wcs_analyzer.audio.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "ffmpeg", stderr="codec error"),
            ),
        ):
            with pytest.raises(AudioProcessingError, match="ffmpeg failed"):
                extract_audio_features(video)

    def test_ffmpeg_timeout(self, tmp_path: Path):
        video = tmp_path / "test.mp4"
        video.touch()
        with (
            patch("wcs_analyzer.audio._check_audio_stream", return_value=True),
            patch(
                "wcs_analyzer.audio.subprocess.run",
                side_effect=subprocess.TimeoutExpired("ffmpeg", 120),
            ),
        ):
            with pytest.raises(AudioProcessingError, match="timed out"):
                extract_audio_features(video)

    def test_check_audio_stream_timeout(self, tmp_path: Path):
        video = tmp_path / "test.mp4"
        video.touch()
        with patch(
            "wcs_analyzer.audio.subprocess.run",
            side_effect=subprocess.TimeoutExpired("ffprobe", 30),
        ):
            with pytest.raises(AudioProcessingError, match="Timed out"):
                _check_audio_stream(video)


class TestVideoErrorHandling:
    def test_cannot_open_video_raises(self, tmp_path: Path):
        fake = tmp_path / "nonexistent.mp4"
        fake.touch()  # empty file, cv2 can't open it
        with pytest.raises(VideoProcessingError, match="Cannot open video"):
            extract_frames(fake)


class TestCLIValidation:
    def test_fps_zero_rejected(self):
        with pytest.raises(Exception):
            _validate_fps(None, None, 0)  # type: ignore[arg-type]

    def test_fps_negative_rejected(self):
        with pytest.raises(Exception):
            _validate_fps(None, None, -1)  # type: ignore[arg-type]

    def test_fps_too_high_rejected(self):
        with pytest.raises(Exception):
            _validate_fps(None, None, 31)  # type: ignore[arg-type]

    def test_fps_valid_passes(self):
        assert _validate_fps(None, None, 3.0) == 3.0  # type: ignore[arg-type]
        assert _validate_fps(None, None, 1.0) == 1.0  # type: ignore[arg-type]
        assert _validate_fps(None, None, 30.0) == 30.0  # type: ignore[arg-type]

    def test_cli_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.3.1" in result.output


class TestAnalyzerErrorHandling:
    def test_assert_replaced_with_exception(self):
        """Verify that analyzer.py no longer uses assert for validation."""
        import inspect
        from wcs_analyzer import analyzer

        source = inspect.getsource(analyzer._call_claude)
        assert "assert " not in source, "assert should be replaced with proper exception"
