"""Tests for LLM response caching."""

from pathlib import Path

from wcs_analyzer.cache import (
    _cache_key,
    get_cached_result,
    save_to_cache,
    segments_to_dicts,
    dicts_to_segments,
)
from wcs_analyzer.scoring import SegmentAnalysis


def _make_video_file(tmp_path: Path) -> Path:
    """Create a small fake video file for cache key generation."""
    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake video content " * 100)
    return video


class TestCacheKey:
    def test_same_params_same_key(self, tmp_path: Path):
        video = _make_video_file(tmp_path)
        k1 = _cache_key(video, fps=3.0, detail="medium", model="claude-sonnet-4-6")
        k2 = _cache_key(video, fps=3.0, detail="medium", model="claude-sonnet-4-6")
        assert k1 == k2

    def test_different_fps_different_key(self, tmp_path: Path):
        video = _make_video_file(tmp_path)
        k1 = _cache_key(video, fps=3.0, detail="medium", model="claude-sonnet-4-6")
        k2 = _cache_key(video, fps=5.0, detail="medium", model="claude-sonnet-4-6")
        assert k1 != k2

    def test_different_detail_different_key(self, tmp_path: Path):
        video = _make_video_file(tmp_path)
        k1 = _cache_key(video, fps=3.0, detail="low", model="claude-sonnet-4-6")
        k2 = _cache_key(video, fps=3.0, detail="high", model="claude-sonnet-4-6")
        assert k1 != k2


class TestCacheRoundTrip:
    def test_save_and_load(self, tmp_path: Path):
        video = _make_video_file(tmp_path)
        cache_dir = tmp_path / "cache"
        data = [{"timing_score": 7.5, "start_time": 0.0, "end_time": 4.0}]

        save_to_cache(video, 3.0, "medium", "claude-sonnet-4-6", data, cache_dir=cache_dir)
        result = get_cached_result(video, 3.0, "medium", "claude-sonnet-4-6", cache_dir=cache_dir)

        assert result is not None
        assert result[0]["timing_score"] == 7.5

    def test_cache_miss_returns_none(self, tmp_path: Path):
        video = _make_video_file(tmp_path)
        cache_dir = tmp_path / "cache"
        result = get_cached_result(video, 3.0, "medium", "claude-sonnet-4-6", cache_dir=cache_dir)
        assert result is None

    def test_corrupt_cache_returns_none(self, tmp_path: Path):
        video = _make_video_file(tmp_path)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        key = _cache_key(video, 3.0, "medium", "claude-sonnet-4-6")
        (cache_dir / f"{key}.json").write_text("not valid json{{{")
        result = get_cached_result(video, 3.0, "medium", "claude-sonnet-4-6", cache_dir=cache_dir)
        assert result is None


class TestSegmentSerialization:
    def test_roundtrip(self):
        segments = [
            SegmentAnalysis(
                start_time=0.0, end_time=4.0,
                timing_score=8.0, technique_score=7.0,
                teamwork_score=6.0, presentation_score=9.0,
                patterns=["Sugar Push"],
                highlights=["Good anchor"],
            ),
        ]
        dicts = segments_to_dicts(segments)
        restored = dicts_to_segments(dicts)
        assert len(restored) == 1
        assert restored[0].timing_score == 8.0
        assert restored[0].patterns == ["Sugar Push"]
        assert restored[0].highlights == ["Good anchor"]

    def test_roundtrip_preserves_all_fields(self):
        """Cache rehydration should preserve every dataclass field.

        Previously dicts_to_segments handpicked a subset, silently
        dropping pattern_details, confidence intervals, reasoning,
        is_summary, usage, and partner scores on every cache hit.
        """
        from wcs_analyzer.pricing import UsageTotals
        segments = [
            SegmentAnalysis(
                start_time=0.0, end_time=8.0,
                timing_score=8.5, technique_score=7.2,
                teamwork_score=7.8, presentation_score=8.0,
                timing_low=7.5, timing_high=9.0,
                posture_score=8.0, extension_score=7.5,
                pattern_details=[{"name": "Sugar Push", "quality": "strong"}],
                reasoning={"timing": "clean anchors on 3&4"},
                is_summary=True,
                lead_technique=7.5, follow_technique=8.0,
                usage=UsageTotals.from_counts("claude-sonnet-4-6", 5000, 2500),
            ),
        ]
        dicts = segments_to_dicts(segments)
        restored = dicts_to_segments(dicts)
        r = restored[0]
        assert r.pattern_details == [{"name": "Sugar Push", "quality": "strong"}]
        assert r.reasoning == {"timing": "clean anchors on 3&4"}
        assert r.is_summary is True
        assert r.timing_low == 7.5
        assert r.timing_high == 9.0
        assert r.lead_technique == 7.5
        assert r.follow_technique == 8.0
        assert r.usage.input_tokens == 5000
        assert r.usage.output_tokens == 2500
        assert r.usage.estimated_cost > 0

    def test_old_cache_missing_new_fields_still_loads(self):
        """Legacy cache dicts (from before this session) should load fine."""
        old_dict = {
            "start_time": 0.0, "end_time": 4.0,
            "timing_score": 7.0, "technique_score": 7.0,
            "teamwork_score": 7.0, "presentation_score": 7.0,
            "patterns": ["Whip"],
        }
        restored = dicts_to_segments([old_dict])
        assert len(restored) == 1
        assert restored[0].timing_score == 7.0
        assert restored[0].patterns == ["Whip"]
        # New fields default cleanly
        assert restored[0].is_summary is False
        assert restored[0].pattern_details == []
        assert restored[0].reasoning == {}
