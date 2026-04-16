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
    """Restore SegmentAnalysis list from cached dicts.

    Rehydrates every dataclass field present in the cached dict, so
    fields added over time (pattern_details, confidence intervals,
    reasoning, usage, is_summary, etc.) aren't silently dropped on a
    cache hit. Unknown fields are ignored so old caches stay readable.
    The nested `usage` dict is re-inflated to a UsageTotals instance.
    """
    from dataclasses import fields

    from .pricing import UsageTotals
    from .scoring import SegmentAnalysis

    valid = {f.name for f in fields(SegmentAnalysis)}
    results = []
    for d in data:
        kwargs = {k: v for k, v in d.items() if k in valid}
        # Re-inflate nested dataclasses serialized by asdict()
        if isinstance(kwargs.get("usage"), dict):
            usage_dict = kwargs["usage"]
            usage_valid = {f.name for f in fields(UsageTotals)}
            kwargs["usage"] = UsageTotals(**{k: v for k, v in usage_dict.items() if k in usage_valid})
        results.append(SegmentAnalysis(**kwargs))
    return results
