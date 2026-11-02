"""PostgreSQL live diagnostic inspector.

Connects with a read-only DSN and queries these system views:
  • pg_stat_statements  — top slow queries by mean execution time
  • pg_stat_user_tables — seq scan vs index scan ratio per table
  • pg_stat_activity    — active connections + lock waits
  • pg_locks            — blocking lock chains
  • pg_stat_bgwriter   — checkpoint / buffer write pressure

The inspector never writes to the database.  The apply DSN
(used by change_manager) is a separate credential.

Usage::
    insp = PgInspector(dsn="postgresql://user:pass@host:5432/db")
    metrics = insp.inspect()
    for q in metrics.top_slow_queries(5):
        print(q.mean_time_ms, q.query[:80])
"""

from __future__ import annotations

import logging
import re
import time
from contextlib import contextmanager
from typing import Generator

from backend.diagnostics.models import (
    DiagnosticMetrics, SlowQuery, TableStat, LockInfo,
)

LOG = logging.getLogger("dbrain.diagnostics.pg")

# Minimum number of sequential scans before we flag a table.
_SEQ_SCAN_THRESHOLD = 100


class PgInspector:
    """Read-only PostgreSQL diagnostic inspector."""

    def __init__(self, dsn: str, statement_timeout_ms: int = 5000) -> None:
        """
        Args:
            dsn: psycopg2-compatible DSN string.
                 Example: ``postgresql://readonly:pass@localhost:5432/mydb``
            statement_timeout_ms: Per-query timeout to prevent the inspector
                from hanging on a locked or overloaded instance.
        """
        self._dsn = dsn
        self._timeout = statement_timeout_ms
        self._dsn_label = self._safe_label(dsn)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inspect(self) -> DiagnosticMetrics:
        """Run all diagnostic checks and return a unified Metrics snapshot."""
        metrics = DiagnosticMetrics(engine="postgresql", dsn_label=self._dsn_label)
        try:
            with self._connect() as conn:
                metrics.slow_queries = self._slow_queries(conn)
                metrics.table_stats = self._table_stats(conn)
                metrics.active_locks = self._active_locks(conn)
                metrics.connection_count = self._connection_count(conn)
                metrics.raw["bgwriter"] = self._bgwriter(conn)
                metrics.raw["pg_version"] = self._pg_version(conn)
        except Exception as exc:  # noqa: BLE001
            LOG.error("PgInspector failed for %s: %s", self._dsn_label, exc)
            metrics.error = str(exc)
        return metrics

    def test_connection(self) -> bool:
        """Return True if the DSN is reachable, False otherwise."""
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Generator:
        import psycopg2
        conn = psycopg2.connect(self._dsn, connect_timeout=8)
        conn.set_session(readonly=True, autocommit=True)
        try:
            with conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = {self._timeout}")
            yield conn
        finally:
            conn.close()

    def _slow_queries(self, conn) -> list[SlowQuery]:
        """Query pg_stat_statements for the top slowest queries."""
        sql = """
            SELECT query,
                   round(mean_exec_time::numeric, 2)   AS mean_ms,
                   calls,
                   round(total_exec_time::numeric, 2)  AS total_ms,
                   round(
                       CASE WHEN calls > 0 THEN rows::numeric / calls ELSE 0 END,
                       2
                   ) AS rows_per_call
            FROM pg_stat_statements
            ORDER BY mean_exec_time DESC
            LIMIT 10;
        """
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
            return [
                SlowQuery(
                    query=r[0], mean_time_ms=float(r[1]),
                    calls=int(r[2]), total_time_ms=float(r[3]),
                    rows_per_call=float(r[4]),
                )
                for r in rows
            ]
        except Exception as exc:  # pg_stat_statements not installed
            LOG.warning("pg_stat_statements unavailable: %s", exc)
            return self._slow_queries_fallback(conn)

    def _slow_queries_fallback(self, conn) -> list[SlowQuery]:
        """Fallback when pg_stat_statements extension is absent — uses pg_stat_activity."""
        sql = """
            SELECT query,
                   extract(epoch FROM (now() - query_start)) * 1000 AS running_ms,
                   1                                                 AS calls,
                   extract(epoch FROM (now() - query_start)) * 1000 AS total_ms
            FROM pg_stat_activity
            WHERE state = 'active'
              AND query NOT LIKE '%pg_stat_activity%'
            ORDER BY query_start
            LIMIT 10;
        """
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        return [
            SlowQuery(
                query=r[0], mean_time_ms=float(r[1] or 0),
                calls=int(r[2]), total_time_ms=float(r[3] or 0),
                extra={"source": "pg_stat_activity"},
            )
            for r in rows
        ]

    def _table_stats(self, conn) -> list[TableStat]:
        """Detect tables with high sequential scan ratios."""
        sql = """
            SELECT relname,
                   seq_scan,
                   idx_scan,
                   n_live_tup,
                   n_dead_tup,
                   to_char(last_vacuum, 'YYYY-MM-DD HH24:MI') AS last_vacuum
            FROM pg_stat_user_tables
            WHERE seq_scan > %s
            ORDER BY seq_scan DESC
            LIMIT 20;
        """
        with conn.cursor() as cur:
            cur.execute(sql, (_SEQ_SCAN_THRESHOLD,))
            rows = cur.fetchall()
        return [
            TableStat(
                name=r[0], seq_scans=int(r[1]), idx_scans=int(r[2] or 0),
                live_rows=int(r[3] or 0), dead_rows=int(r[4] or 0),
                last_vacuum=r[5],
            )
            for r in rows
        ]

    def _active_locks(self, conn) -> list[LockInfo]:
        """Find active queries waiting on locks."""
        sql = """
            SELECT a.pid,
                   a.query,
                   a.wait_event,
                   a.state,
                   a.wait_event_type,
                   blocking.pid AS blocked_by
            FROM pg_stat_activity a
            LEFT JOIN pg_stat_activity blocking
                   ON blocking.pid = ANY(pg_blocking_pids(a.pid))
            WHERE a.state != 'idle'
              AND a.query NOT LIKE '%pg_stat_activity%'
            ORDER BY a.query_start
            LIMIT 15;
        """
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        return [
            LockInfo(
                pid=r[0], query=(r[1] or "")[:200],
                wait_event=r[2], state=r[3], blocked_by=r[5],
            )
            for r in rows
        ]

    def _connection_count(self, conn) -> int:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM pg_stat_activity WHERE state != 'idle'")
            return int(cur.fetchone()[0])

    def _bgwriter(self, conn) -> dict:
        """Checkpoint pressure. PG17 moved these counters to pg_stat_checkpointer."""
        queries = (
            "SELECT num_timed, num_requested FROM pg_stat_checkpointer",
            "SELECT checkpoints_timed, checkpoints_req FROM pg_stat_bgwriter",
        )
        for sql in queries:
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    r = cur.fetchone()
                if r:
                    return {"checkpoints_timed": r[0], "checkpoints_req": r[1]}
            except Exception:  # noqa: BLE001 - view absent on this version
                continue
        return {}

    def _pg_version(self, conn) -> str:
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            return cur.fetchone()[0]

    @staticmethod
    def _safe_label(dsn: str) -> str:
        """Strip password from DSN for safe logging."""
        return re.sub(r":[^@/]+@", ":***@", dsn)
