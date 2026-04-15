"""LLM response caching for avoiding redundant API calls."""

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".wcs-analyzer" / "cache"


def _cache_key(video_path: Path, fps: float, detail: str, model: str) -> str:
    """Generate a cache key from video file hash + analysis parameters."""
    h = hashlib.sha256()
    # Hash first 10MB + file size for speed (avoids reading entire large file)
    with open(video_path, "rb") as f:
        chunk = f.read(10 * 1024 * 1024)
        h.update(chunk)
    h.update(str(video_path.stat().st_size).encode())
    h.update(f"{fps}:{detail}:{model}".encode())
    return h.hexdigest()[:16]


def get_cached_result(
    video_path: Path, fps: float, detail: str, model: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> list[dict] | None:
    """Load cached analysis result if it exists.

    Returns:
        List of segment dicts (raw JSON) or None if no cache hit.
    """
    key = _cache_key(video_path, fps, detail, model)
    cache_file = cache_dir / f"{key}.json"

    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            logger.info("Cache hit: %s", cache_file.name)
            return data  # type: ignore[no-any-return]
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt cache file %s, ignoring", cache_file)
            return None
    return None


def save_to_cache(
    video_path: Path, fps: float, detail: str, model: str,
    segments: list[dict],
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> None:
    """Save analysis result to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(video_path, fps, detail, model)
    cache_file = cache_dir / f"{key}.json"

    try:
        cache_file.write_text(json.dumps(segments, indent=2))
        logger.info("Cached result to %s", cache_file.name)
    except OSError as e:
        logger.warning("Failed to write cache: %s", e)


def segments_to_dicts(segments: list) -> list[dict]:
    """Convert SegmentAnalysis list to serializable dicts."""
    from dataclasses import asdict
    return [asdict(s) for s in segments]


def dicts_to_segments(data: list[dict]) -> list:
    """Restore SegmentAnalysis list from cached dicts."""
    from .scoring import SegmentAnalysis
    results = []
    for d in data:
        results.append(SegmentAnalysis(
            start_time=d.get("start_time", 0.0),
            end_time=d.get("end_time", 0.0),
            timing_score=d.get("timing_score", 5.0),
            technique_score=d.get("technique_score", 5.0),
            teamwork_score=d.get("teamwork_score", 5.0),
            presentation_score=d.get("presentation_score", 5.0),
            posture_score=d.get("posture_score", 5.0),
            extension_score=d.get("extension_score", 5.0),
            footwork_score=d.get("footwork_score", 5.0),
            slot_score=d.get("slot_score", 5.0),
            off_beat_moments=d.get("off_beat_moments", []),
            patterns=d.get("patterns", []),
            pattern_details=d.get("pattern_details", []),
            is_summary=d.get("is_summary", False),
            highlights=d.get("highlights", []),
            improvements=d.get("improvements", []),
            raw_data=d.get("raw_data", {}),
        ))
    return results
