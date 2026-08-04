"""SQLite-backed request metrics (replaces the in-memory Python list).

Persists every /query request so /metrics stays accurate across restarts and
multiple worker processes — the scope document's SQLite persistence requirement.
"""

from __future__ import annotations

import sqlite3
import statistics
import time
from pathlib import Path
from typing import Optional

from backend import config


def _build_metric_row(
    latency_ms: float,
    refused: bool,
    chunks_retrieved: int,
    detected_db: Optional[str],
    error: Optional[str],
) -> tuple:
    return (
        round(latency_ms, 2),
        1 if refused else 0,
        chunks_retrieved,
        detected_db,
        error,
        time.time(),
    )


class MetricsStore:
    SCHEMA = """
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latency_ms REAL NOT NULL,
            refused INTEGER NOT NULL DEFAULT 0,
            chunks_retrieved INTEGER NOT NULL DEFAULT 0,
            detected_db TEXT,
            error TEXT,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_requests_created ON requests(created_at);
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = str(db_path or config.DB_FILE)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(self.SCHEMA)

    def record_request(
        self,
        latency_ms: float,
        refused: bool = False,
        chunks_retrieved: int = 0,
        detected_db: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO requests (latency_ms, refused, chunks_retrieved, detected_db, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                _build_metric_row(latency_ms, refused, chunks_retrieved, detected_db, error),
            )

    def get_metrics(self) -> dict:
        """P50/P95 latency, refusal rate, per-database breakdown."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT latency_ms, refused, detected_db FROM requests"
            ).fetchall()

        if not rows:
            return {"message": "No requests yet", "total_requests": 0}

        latencies = sorted(r["latency_ms"] for r in rows)
        n = len(latencies)
        refused = sum(1 for r in rows if r["refused"])

        db_counts: dict[str, int] = {}
        for r in rows:
            db = r["detected_db"] or "ambiguous"
            db_counts[db] = db_counts.get(db, 0) + 1

        return {
            "total_requests": n,
            "latency_p50_ms": round(latencies[(n - 1) // 2], 1),
            "latency_p95_ms": round(latencies[min(int(n * 0.95), n - 1)], 1),
            "latency_mean_ms": round(statistics.mean(latencies), 1),
            "latency_max_ms": round(max(latencies), 1),
            "refusal_rate_pct": round((refused / n) * 100, 1),
            "requests_by_database": db_counts,
        }

    def count(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM requests").fetchone()
        return int(row[0])