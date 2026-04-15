"""Longitudinal history store for dancer progression tracking.

A small SQLite-backed log of past analysis runs, keyed by dancer name.
Used by the `progress` command to answer "am I getting better?" and to
fit a simple linear trajectory across runs.

The default database location is `~/.wcs-analyzer/history.db`; override
with the `WCS_ANALYZER_HISTORY_DB` environment variable. Both `analyze
--save-history <name>` and `progress <name>` share this path.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .scoring import FinalScores


def default_db_path() -> Path:
    """Return the on-disk location of the history database."""
    override = os.environ.get("WCS_ANALYZER_HISTORY_DB")
    if override:
        return Path(override)
    return Path.home() / ".wcs-analyzer" / "history.db"


@dataclass
class HistoryRow:
    """A single stored analysis row."""

    dancer: str
    video_name: str
    created_at: str  # ISO 8601 UTC
    provider: str
    overall: float
    grade: str
    timing: float
    technique: float
    teamwork: float
    presentation: float


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    dancer      TEXT NOT NULL,
    video_name  TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    provider    TEXT NOT NULL,
    overall     REAL NOT NULL,
    grade       TEXT NOT NULL,
    timing      REAL NOT NULL,
    technique   REAL NOT NULL,
    teamwork    REAL NOT NULL,
    presentation REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_dancer_created
    ON runs(dancer, created_at);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def save_run(
    dancer: str,
    video_name: str,
    provider: str,
    scores: FinalScores,
    db_path: Path | None = None,
) -> None:
    """Persist a completed analysis run for the given dancer."""
    db = db_path or default_db_path()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect(db) as conn:
        conn.execute(
            "INSERT INTO runs (dancer, video_name, created_at, provider, "
            "overall, grade, timing, technique, teamwork, presentation) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                dancer, video_name, now, provider,
                scores.overall, scores.grade,
                scores.timing, scores.technique,
                scores.teamwork, scores.presentation,
            ),
        )
        conn.commit()


def load_history(dancer: str, db_path: Path | None = None) -> list[HistoryRow]:
    """Return all runs for `dancer`, oldest first."""
    db = db_path or default_db_path()
    if not db.exists():
        return []
    with _connect(db) as conn:
        rows = conn.execute(
            "SELECT dancer, video_name, created_at, provider, overall, grade, "
            "timing, technique, teamwork, presentation "
            "FROM runs WHERE dancer = ? ORDER BY created_at ASC",
            (dancer,),
        ).fetchall()
    return [HistoryRow(**dict(r)) for r in rows]


def list_dancers(db_path: Path | None = None) -> list[tuple[str, int]]:
    """Return `(dancer, run_count)` tuples for every tracked dancer."""
    db = db_path or default_db_path()
    if not db.exists():
        return []
    with _connect(db) as conn:
        rows = conn.execute(
            "SELECT dancer, COUNT(*) as n FROM runs GROUP BY dancer ORDER BY dancer"
        ).fetchall()
    return [(r["dancer"], r["n"]) for r in rows]


@dataclass
class Trajectory:
    """Linear fit over a sequence of runs for one dancer and one metric."""

    slope_per_run: float  # score delta per run (not per day — simpler)
    intercept: float
    latest: float
    first: float
    n: int


def _fit_linear(values: list[float]) -> Trajectory:
    n = len(values)
    if n == 0:
        return Trajectory(0.0, 0.0, 0.0, 0.0, 0)
    if n == 1:
        return Trajectory(0.0, values[0], values[0], values[0], 1)
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    num = sum((xs[i] - mean_x) * (values[i] - mean_y) for i in range(n))
    den = sum((x - mean_x) ** 2 for x in xs) or 1e-9
    slope = num / den
    intercept = mean_y - slope * mean_x
    return Trajectory(
        slope_per_run=slope,
        intercept=intercept,
        latest=values[-1],
        first=values[0],
        n=n,
    )


def fit_trajectories(rows: list[HistoryRow]) -> dict[str, Trajectory]:
    """Fit a linear trajectory per category across runs."""
    categories = ("overall", "timing", "technique", "teamwork", "presentation")
    result: dict[str, Trajectory] = {}
    for cat in categories:
        values = [getattr(r, cat) for r in rows]
        result[cat] = _fit_linear(values)
    return result
