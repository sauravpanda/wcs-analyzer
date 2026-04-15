"""Tests for the longitudinal history store."""

from pathlib import Path

from wcs_analyzer.history import (
    HistoryRow,
    _fit_linear,
    fit_trajectories,
    list_dancers,
    load_history,
    save_run,
)
from wcs_analyzer.scoring import FinalScores


def _fs(overall=7.0, timing=7.0, technique=7.0, teamwork=7.0, presentation=7.0,
        grade="B-") -> FinalScores:
    return FinalScores(
        timing=timing, technique=technique, teamwork=teamwork,
        presentation=presentation, overall=overall, grade=grade,
    )


def test_save_and_load_roundtrip(tmp_path: Path):
    db = tmp_path / "history.db"
    save_run("alice", "clip1.mp4", "gemini", _fs(overall=7.2), db_path=db)
    save_run("alice", "clip2.mp4", "gemini", _fs(overall=7.8), db_path=db)

    rows = load_history("alice", db_path=db)
    assert len(rows) == 2
    assert rows[0].video_name == "clip1.mp4"
    assert rows[1].overall == 7.8
    # Should be sorted oldest first
    assert rows[0].created_at <= rows[1].created_at


def test_load_missing_dancer_returns_empty(tmp_path: Path):
    db = tmp_path / "history.db"
    save_run("alice", "clip1.mp4", "gemini", _fs(), db_path=db)
    assert load_history("bob", db_path=db) == []


def test_load_nonexistent_db_returns_empty(tmp_path: Path):
    assert load_history("alice", db_path=tmp_path / "nothing.db") == []


def test_list_dancers_counts_runs(tmp_path: Path):
    db = tmp_path / "history.db"
    save_run("alice", "c1.mp4", "gemini", _fs(), db_path=db)
    save_run("alice", "c2.mp4", "gemini", _fs(), db_path=db)
    save_run("bob", "c3.mp4", "claude", _fs(), db_path=db)

    dancers = list_dancers(db_path=db)
    assert ("alice", 2) in dancers
    assert ("bob", 1) in dancers


def test_list_dancers_empty_db(tmp_path: Path):
    assert list_dancers(db_path=tmp_path / "empty.db") == []


class TestLinearFit:
    def test_improving_trajectory_positive_slope(self):
        t = _fit_linear([5.0, 6.0, 7.0, 8.0])
        assert t.slope_per_run > 0.9
        assert t.first == 5.0
        assert t.latest == 8.0
        assert t.n == 4

    def test_flat_trajectory_zero_slope(self):
        t = _fit_linear([7.0, 7.0, 7.0])
        assert t.slope_per_run == 0.0

    def test_regressing_trajectory_negative_slope(self):
        t = _fit_linear([8.0, 7.0, 6.0])
        assert t.slope_per_run < 0

    def test_single_value_returns_latest(self):
        t = _fit_linear([7.5])
        assert t.slope_per_run == 0.0
        assert t.latest == 7.5
        assert t.n == 1

    def test_empty_returns_zeros(self):
        t = _fit_linear([])
        assert t.n == 0
        assert t.slope_per_run == 0.0


def test_fit_trajectories_covers_all_categories(tmp_path: Path):
    db = tmp_path / "history.db"
    save_run("alice", "c1.mp4", "gemini", _fs(overall=6.0, timing=5.0, technique=6.0), db_path=db)
    save_run("alice", "c2.mp4", "gemini", _fs(overall=7.0, timing=7.0, technique=7.0), db_path=db)
    save_run("alice", "c3.mp4", "gemini", _fs(overall=8.0, timing=8.0, technique=8.0), db_path=db)

    rows = load_history("alice", db_path=db)
    traj = fit_trajectories(rows)
    assert set(traj.keys()) == {"overall", "timing", "technique", "teamwork", "presentation"}
    assert traj["overall"].slope_per_run > 0.8
    assert traj["timing"].first == 5.0
    assert traj["timing"].latest == 8.0


def test_history_row_fields():
    row = HistoryRow(
        dancer="alice", video_name="c.mp4", created_at="2026-04-15T00:00:00+00:00",
        provider="gemini", overall=7.0, grade="B-",
        timing=7.0, technique=7.0, teamwork=7.0, presentation=7.0,
    )
    assert row.dancer == "alice"
    assert row.overall == 7.0
