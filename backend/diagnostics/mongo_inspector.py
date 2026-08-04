"""MongoDB live diagnostic inspector.

Connects with a read-only URI and queries:
  • db.currentOp()            — active operations + long-running queries
  • db.stats()                — database-level stats (dataSize, storageSize)
  • serverStatus              — WiredTiger cache, connections, opcounters
  • system.profile            — profiler slow operation log (if enabled)
  • db.collection.stats()     — per-collection index stats

The inspector never writes to the database.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from backend.diagnostics.models import (
    DiagnosticMetrics, SlowQuery, TableStat, LockInfo,
)

LOG = logging.getLogger("dbrain.diagnostics.mongo")

# Milliseconds: operations longer than this are treated as "slow"
_SLOW_OP_THRESHOLD_MS = 100


class MongoInspector:
    """Read-only MongoDB diagnostic inspector."""

    def __init__(self, uri: str, db_name: str = "admin", server_timeout_ms: int = 5000) -> None:
        """
        Args:
            uri: PyMongo URI.
                 Example: ``mongodb://readonly:pass@localhost:27017/mydb``
            db_name: Which database to inspect (defaults to admin for serverStatus).
            server_timeout_ms: Connection + server selection timeout.
        """
        self._uri = uri
        self._db_name = db_name
        self._timeout = server_timeout_ms
        self._dsn_label = self._safe_label(uri)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inspect(self) -> DiagnosticMetrics:
        metrics = DiagnosticMetrics(engine="mongodb", dsn_label=self._dsn_label)
        try:
            client, db = self._connect()
            try:
                metrics.slow_queries = self._slow_queries(db)
                metrics.table_stats = self._collection_stats(db)
                metrics.active_locks = self._current_ops(client)
                metrics.connection_count = self._connection_count(client)
                metrics.raw["server_status"] = self._server_status_summary(client)
                metrics.raw["db_stats"] = self._db_stats(db)
            finally:
                client.close()
        except Exception as exc:  # noqa: BLE001
            LOG.error("MongoInspector failed for %s: %s", self._dsn_label, exc)
            metrics.error = str(exc)
        return metrics

    def test_connection(self) -> bool:
        try:
            client, _ = self._connect()
            client.admin.command("ping")
            client.close()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self):
        import pymongo
        client = pymongo.MongoClient(
            self._uri,
            serverSelectionTimeoutMS=self._timeout,
            connectTimeoutMS=self._timeout,
        )
        client.admin.command("ping")          # fast fail if unreachable
        db = client[self._db_name]
        return client, db

    def _slow_queries(self, db) -> list[SlowQuery]:
        """Read from system.profile (must be enabled: db.setProfilingLevel(1, 100))."""
        try:
            profile_col = db["system.profile"]
            docs = list(
                profile_col.find(
                    {"millis": {"$gte": _SLOW_OP_THRESHOLD_MS}},
                    {"ns": 1, "op": 1, "millis": 1, "keysExamined": 1, "nreturned": 1, "command": 1},
                ).sort("millis", -1).limit(10)
            )
            return [
                SlowQuery(
                    query=f"{d.get('op','?')} on {d.get('ns','?')}: {str(d.get('command',{}))[:120]}",
                    mean_time_ms=float(d.get("millis", 0)),
                    calls=1,
                    total_time_ms=float(d.get("millis", 0)),
                    rows_per_call=float(d.get("nreturned", 0)),
                    extra={"keysExamined": d.get("keysExamined", 0)},
                )
                for d in docs
            ]
        except Exception as exc:
            LOG.info("system.profile unreadable (profiler may be off): %s", exc)
            return []

    def _collection_stats(self, db) -> list[TableStat]:
        """Collect-level stats: document count + index usage."""
        try:
            names = db.list_collection_names()
            stats = []
            for name in names[:20]:
                col = db[name]
                try:
                    s = db.command("collStats", name)
                    stats.append(TableStat(
                        name=name,
                        seq_scans=0,   # MongoDB doesn't expose seq_scan count easily
                        idx_scans=0,
                        live_rows=int(s.get("count", 0)),
                        extra={
                            "size_bytes": s.get("size", 0),
                            "index_count": s.get("nindexes", 0),
                            "avg_obj_size": s.get("avgObjSize", 0),
                        },
                    ))
                except Exception:
                    pass
            return stats
        except Exception as exc:
            LOG.warning("Collection stats failed: %s", exc)
            return []

    def _current_ops(self, client) -> list[LockInfo]:
        """currentOp — long-running operations."""
        try:
            result = client.admin.command(
                "currentOp",
                {"active": True, "secs_running": {"$gt": 1}},
            )
            ops = result.get("inprog", [])
            return [
                LockInfo(
                    pid=op.get("opid"),
                    query=str(op.get("command", {}))[:200],
                    wait_event="Lock" if op.get("waitingForLock") else None,
                    state="running" if op.get("active") else "idle",
                )
                for op in ops[:15]
            ]
        except Exception as exc:
            LOG.warning("currentOp failed: %s", exc)
            return []

    def _connection_count(self, client) -> int:
        try:
            status = client.admin.command("serverStatus", {"connections": 1})
            return int(status.get("connections", {}).get("current", 0))
        except Exception:
            return 0

    def _server_status_summary(self, client) -> dict[str, Any]:
        try:
            s = client.admin.command("serverStatus", {
                "connections": 1, "opcounters": 1, "wiredTiger": 1, "mem": 1,
            })
            wt = s.get("wiredTiger", {}).get("cache", {})
            return {
                "connections_current": s.get("connections", {}).get("current"),
                "connections_available": s.get("connections", {}).get("available"),
                "ops_insert": s.get("opcounters", {}).get("insert"),
                "ops_query": s.get("opcounters", {}).get("query"),
                "ops_update": s.get("opcounters", {}).get("update"),
                "wt_cache_bytes_in_use": wt.get("bytes currently in the cache"),
                "wt_cache_bytes_max": wt.get("maximum bytes configured"),
                "wt_cache_dirty_pct": round(
                    (wt.get("tracked dirty bytes in the cache", 0) /
                     max(wt.get("maximum bytes configured", 1), 1)) * 100, 2
                ),
                "mem_resident_mb": s.get("mem", {}).get("resident"),
            }
        except Exception as exc:
            LOG.warning("serverStatus failed: %s", exc)
            return {}

    def _db_stats(self, db) -> dict[str, Any]:
        try:
            return db.command("dbStats", {"scale": 1048576})  # bytes → MB
        except Exception:
            return {}

    @staticmethod
    def _safe_label(uri: str) -> str:
        return re.sub(r":[^@/]+@", ":***@", uri)
