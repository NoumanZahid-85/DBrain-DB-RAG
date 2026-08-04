"""Improvement Tracker — before/after metric comparison.

After a ChangeProposal is applied, the ImprovementTracker:
  1. Re-runs the appropriate inspector
  2. Compares the key metrics against a pre-stored baseline snapshot
  3. Returns an ImprovementReport with a clear verdict

Snapshots are stored in SQLite so the before/after comparison
survives a process restart between apply() and check().
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from backend import config
from backend.diagnostics.models import DiagnosticMetrics, ImprovementReport
from backend.diagnostics.dispatcher import DiagnosticDispatcher
from backend.diagnostics.models import DiagnosticPlan

LOG = logging.getLogger("dbrain.improvement_tracker")


class ImprovementTracker:
    """Snapshot and compare key database metrics across a change boundary."""

    def __init__(self) -> None:
        self._db_path = config.DB_FILE
        self._dispatcher = DiagnosticDispatcher()
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def snapshot_before(self, proposal_id: str, engine: str) -> str:
        """Run the inspector and persist the baseline metrics snapshot.

        Returns the snapshot_id (same as proposal_id for simplicity).
        """
        metrics = self._inspect(engine)
        self._save_snapshot(proposal_id, "before", engine, metrics)
        LOG.info("Before-snapshot saved for proposal %s (%s)", proposal_id, engine)
        return proposal_id

    def snapshot_after(self, proposal_id: str, engine: str, wait_s: int = 5) -> str:
        """Wait briefly (index builds need a moment), then snapshot post-change metrics."""
        if wait_s > 0:
            LOG.info("Waiting %ds before after-snapshot…", wait_s)
            time.sleep(wait_s)
        metrics = self._inspect(engine)
        self._save_snapshot(proposal_id, "after", engine, metrics)
        LOG.info("After-snapshot saved for proposal %s (%s)", proposal_id, engine)
        return proposal_id

    def compare(self, proposal_id: str) -> list[ImprovementReport]:
        """Load before/after snapshots and produce ImprovementReports."""
        before_row = self._load_snapshot(proposal_id, "before")
        after_row = self._load_snapshot(proposal_id, "after")

        if not before_row or not after_row:
            LOG.warning("Missing snapshots for proposal %s", proposal_id)
            return []

        before = json.loads(before_row["metrics_json"])
        after = json.loads(after_row["metrics_json"])
        engine = before_row["engine"]

        reports: list[ImprovementReport] = []

        # ── Mean query latency ──────────────────────────────────────────
        b_mean = self._top_mean_ms(before)
        a_mean = self._top_mean_ms(after)
        if b_mean is not None and a_mean is not None and b_mean > 0:
            pct = round((b_mean - a_mean) / b_mean * 100, 1)
            reports.append(ImprovementReport(
                change_proposal_id=proposal_id, engine=engine,
                metric_name="top_query_mean_time_ms",
                before_value=b_mean, after_value=a_mean,
                improvement_pct=pct,
                verdict=self._verdict(pct),
                detail=(
                    f"Slowest query mean time: {b_mean:.1f}ms → {a_mean:.1f}ms "
                    f"({'↓' if pct > 0 else '↑'}{abs(pct):.1f}%)"
                ),
            ))

        # ── Sequential scan count ──────────────────────────────────────
        b_seq = self._total_seq_scans(before)
        a_seq = self._total_seq_scans(after)
        if b_seq > 0:
            pct = round((b_seq - a_seq) / b_seq * 100, 1) if b_seq else 0
            reports.append(ImprovementReport(
                change_proposal_id=proposal_id, engine=engine,
                metric_name="seq_scan_total",
                before_value=float(b_seq), after_value=float(a_seq),
                improvement_pct=pct,
                verdict=self._verdict(pct),
                detail=f"Total sequential scans: {b_seq} → {a_seq}",
            ))

        # ── Active lock waits ──────────────────────────────────────────
        b_locks = self._lock_count(before)
        a_locks = self._lock_count(after)
        if b_locks > 0 or a_locks > 0:
            pct = round((b_locks - a_locks) / max(b_locks, 1) * 100, 1)
            reports.append(ImprovementReport(
                change_proposal_id=proposal_id, engine=engine,
                metric_name="blocking_lock_count",
                before_value=float(b_locks), after_value=float(a_locks),
                improvement_pct=pct,
                verdict=self._verdict(pct),
                detail=f"Blocking lock waits: {b_locks} → {a_locks}",
            ))

        return reports

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _inspect(self, engine: str) -> DiagnosticMetrics:
        plan = DiagnosticPlan(
            engines=[engine], checks=["slow_query", "seq_scan", "lock"],
            symptom_summary="improvement tracking snapshot",
            raw_symptom="",
        )
        result = self._dispatcher.dispatch(plan)
        return result.get(engine, DiagnosticMetrics(engine=engine, dsn_label="unknown"))

    def _save_snapshot(
        self, proposal_id: str, phase: str, engine: str, metrics: DiagnosticMetrics
    ) -> None:
        metrics_dict = {
            "slow_queries": [dataclasses.asdict(q) for q in metrics.slow_queries],
            "table_stats":  [dataclasses.asdict(t) for t in metrics.table_stats],
            "active_locks": [dataclasses.asdict(lk) for lk in metrics.active_locks],
            "connection_count": metrics.connection_count,
            "raw": metrics.raw,
            "error": metrics.error,
        }
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO improvement_snapshots
                   (proposal_id, phase, engine, metrics_json, ts)
                   VALUES (?, ?, ?, ?, ?)
                """,
                (proposal_id, phase, engine, json.dumps(metrics_dict, default=str), int(time.time())),
            )

    def _load_snapshot(self, proposal_id: str, phase: str):
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM improvement_snapshots WHERE proposal_id=? AND phase=?",
                (proposal_id, phase),
            ).fetchone()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS improvement_snapshots (
                    proposal_id TEXT NOT NULL,
                    phase       TEXT NOT NULL,
                    engine      TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    ts          INTEGER,
                    PRIMARY KEY (proposal_id, phase)
                )
            """)

    @staticmethod
    def _top_mean_ms(m: dict) -> float | None:
        queries = m.get("slow_queries", [])
        if not queries:
            return None
        return max((q.get("mean_time_ms", 0) for q in queries), default=None)

    @staticmethod
    def _total_seq_scans(m: dict) -> int:
        return sum(t.get("seq_scans", 0) for t in m.get("table_stats", []))

    @staticmethod
    def _lock_count(m: dict) -> int:
        return sum(1 for lk in m.get("active_locks", []) if lk.get("blocked_by"))

    @staticmethod
    def _verdict(pct: float) -> str:
        if pct >= 20:
            return "✅ Improved"
        if pct >= 0:
            return "⚠ Marginal improvement"
        return "❌ No improvement (or regressed)"
