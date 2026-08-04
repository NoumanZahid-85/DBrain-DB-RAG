"""MySQL live diagnostic inspector.

Connects with a read-only DSN and queries:
  • performance_schema.events_statements_summary_by_digest — slow statements
  • information_schema.TABLES — table row counts / engine
  • performance_schema.events_waits_summary_global_by_event_name — wait events
  • SHOW PROCESSLIST           — active connections & locks
  • SHOW ENGINE INNODB STATUS  — InnoDB lock graph (parsed)

The inspector never writes to the database.
"""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from typing import Generator

from backend.diagnostics.models import (
    DiagnosticMetrics, SlowQuery, TableStat, LockInfo,
)

LOG = logging.getLogger("dbrain.diagnostics.mysql")


class MysqlInspector:
    """Read-only MySQL / MariaDB diagnostic inspector."""

    def __init__(self, dsn: str, statement_timeout_s: int = 5) -> None:
        """
        Args:
            dsn: SQLAlchemy-style DSN.
                 Example: ``mysql+pymysql://readonly:pass@localhost:3306/mydb``
            statement_timeout_s: Per-query timeout in seconds.
        """
        self._dsn = dsn
        self._timeout = statement_timeout_s
        self._dsn_label = self._safe_label(dsn)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inspect(self) -> DiagnosticMetrics:
        metrics = DiagnosticMetrics(engine="mysql", dsn_label=self._dsn_label)
        try:
            with self._connect() as conn:
                metrics.slow_queries = self._slow_queries(conn)
                metrics.table_stats = self._table_stats(conn, conn.database)
                metrics.active_locks = self._active_locks(conn)
                metrics.connection_count = self._connection_count(conn)
                metrics.raw["innodb_status"] = self._innodb_status_summary(conn)
                metrics.raw["mysql_version"] = self._mysql_version(conn)
        except Exception as exc:  # noqa: BLE001
            LOG.error("MysqlInspector failed for %s: %s", self._dsn_label, exc)
            metrics.error = str(exc)
        return metrics

    def test_connection(self) -> bool:
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Generator:
        import pymysql
        # Parse host/port/db from DSN (simple regex; handles standard form)
        m = re.match(
            r"mysql\+pymysql://(?P<user>[^:]+):(?P<pwd>[^@]+)@(?P<host>[^:/]+):?(?P<port>\d+)?/(?P<db>\w+)",
            self._dsn,
        )
        if not m:
            raise ValueError(f"Cannot parse MySQL DSN: {self._safe_label(self._dsn)}")
        conn = pymysql.connect(
            host=m["host"], port=int(m["port"] or 3306),
            user=m["user"], password=m["pwd"], database=m["db"],
            read_timeout=self._timeout, write_timeout=self._timeout,
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
        )
        conn.database = m["db"]
        try:
            yield conn
        finally:
            conn.close()

    def _slow_queries(self, conn) -> list[SlowQuery]:
        sql = """
            SELECT DIGEST_TEXT                                         AS query,
                   ROUND(AVG_TIMER_WAIT / 1e9, 2)                     AS mean_ms,
                   COUNT_STAR                                          AS calls,
                   ROUND(SUM_TIMER_WAIT / 1e9, 2)                     AS total_ms,
                   ROUND(
                       CASE WHEN COUNT_STAR > 0
                            THEN SUM_ROWS_SENT / COUNT_STAR ELSE 0 END,
                       2
                   ) AS rows_per_call
            FROM performance_schema.events_statements_summary_by_digest
            WHERE DIGEST_TEXT IS NOT NULL
            ORDER BY AVG_TIMER_WAIT DESC
            LIMIT 10;
        """
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
            return [
                SlowQuery(
                    query=r["query"], mean_time_ms=float(r["mean_ms"]),
                    calls=int(r["calls"]), total_time_ms=float(r["total_ms"]),
                    rows_per_call=float(r["rows_per_call"]),
                )
                for r in rows
            ]
        except Exception as exc:
            LOG.warning("performance_schema unavailable: %s", exc)
            return self._slow_query_processlist(conn)

    def _slow_query_processlist(self, conn) -> list[SlowQuery]:
        """Fallback: use SHOW PROCESSLIST."""
        with conn.cursor() as cur:
            cur.execute("SHOW FULL PROCESSLIST")
            rows = cur.fetchall()
        return [
            SlowQuery(
                query=r.get("Info") or "", mean_time_ms=float(r.get("Time", 0)) * 1000,
                calls=1, total_time_ms=float(r.get("Time", 0)) * 1000,
                extra={"source": "processlist", "state": r.get("State")},
            )
            for r in rows if r.get("Command") == "Query"
        ]

    def _table_stats(self, conn, database: str) -> list[TableStat]:
        sql = """
            SELECT TABLE_NAME     AS name,
                   TABLE_ROWS     AS live_rows,
                   DATA_LENGTH    AS data_len,
                   INDEX_LENGTH   AS idx_len
            FROM information_schema.TABLES
            WHERE TABLE_SCHEMA = %s
              AND TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_ROWS DESC
            LIMIT 20;
        """
        with conn.cursor() as cur:
            cur.execute(sql, (database,))
            rows = cur.fetchall()
        return [
            TableStat(
                name=r["name"],
                seq_scans=0,      # MySQL doesn't expose per-table seq_scan without perf_schema
                idx_scans=0,
                live_rows=int(r.get("live_rows") or 0),
                extra={"data_len": r.get("data_len"), "idx_len": r.get("idx_len")} if hasattr(r, "get") else {},
            )
            for r in rows
        ]

    def _active_locks(self, conn) -> list[LockInfo]:
        sql = """
            SELECT PROCESSLIST_ID   AS pid,
                   SQL_TEXT         AS query,
                   EVENT_NAME       AS wait_event,
                   PROCESSLIST_STATE AS state
            FROM performance_schema.events_statements_current esc
            JOIN performance_schema.threads t
              ON esc.THREAD_ID = t.THREAD_ID
            WHERE PROCESSLIST_COMMAND = 'Query'
              AND PROCESSLIST_STATE != 'Sleep'
            ORDER BY TIMER_WAIT DESC
            LIMIT 10;
        """
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
            return [
                LockInfo(
                    pid=r.get("pid"), query=(r.get("query") or "")[:200],
                    wait_event=r.get("wait_event"), state=r.get("state") or "",
                )
                for r in rows
            ]
        except Exception:
            return []

    def _connection_count(self, conn) -> int:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM information_schema.PROCESSLIST WHERE COMMAND != 'Sleep'")
            r = cur.fetchone()
        return int(r["cnt"]) if r else 0

    def _innodb_status_summary(self, conn) -> dict:
        """Parse key InnoDB metrics from SHOW ENGINE INNODB STATUS."""
        try:
            with conn.cursor() as cur:
                cur.execute("SHOW ENGINE INNODB STATUS")
                r = cur.fetchone()
            status = r.get("Status", "") if r else ""
            # Extract transaction count and deadlocks
            trx = re.search(r"(\d+) lock struct", status)
            dead = re.search(r"(\d+) deadlocks", status)
            return {
                "lock_structs": int(trx.group(1)) if trx else 0,
                "deadlocks_total": int(dead.group(1)) if dead else 0,
                "status_snippet": status[:500],
            }
        except Exception:
            return {}

    def _mysql_version(self, conn) -> str:
        with conn.cursor() as cur:
            cur.execute("SELECT VERSION()")
            r = cur.fetchone()
        return list(r.values())[0] if r else "unknown"

    @staticmethod
    def _safe_label(dsn: str) -> str:
        return re.sub(r":[^@/]+@", ":***@", dsn)
