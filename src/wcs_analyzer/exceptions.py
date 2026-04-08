"""Custom exceptions for WCS Analyzer."""


class WCSAnalyzerError(Exception):
    """Base exception for WCS Analyzer."""


class VideoProcessingError(WCSAnalyzerError):
    """Error during video frame extraction."""


class AudioProcessingError(WCSAnalyzerError):
    """Error during audio extraction or beat detection."""


class AnalysisError(WCSAnalyzerError):
    """Error during LLM analysis."""
