"""Diagnostic Dispatcher — routes a DiagnosticPlan to the right inspector(s).

Given a``DiagnosticPlan(engines=["postgresql","mysql"])`` it builds the
appropriate inspectors, runs them (each in its own try/except so one failing
engine doesn't abort the others), and returns a dict of engine → Metrics.

Connection strings are read from config (env vars).  The dispatcher never
requires all three engines to be configured — it silently skips unconfigured
engines with a clear log message so a PostgreSQL-only installation still works.
"""

from __future__ import annotations

import logging
from typing import TypeAlias

from backend import config
from backend.diagnostics.models import DiagnosticMetrics, DiagnosticPlan
from backend.diagnostics.pg_inspector import PgInspector
from backend.diagnostics.mysql_inspector import MysqlInspector
from backend.diagnostics.mongo_inspector import MongoInspector

LOG = logging.getLogger("dbrain.diagnostics.dispatcher")

MetricsMap: TypeAlias = dict[str, DiagnosticMetrics]


class DiagnosticDispatcher:
    """Route a DiagnosticPlan to the matching live inspector(s)."""

    def dispatch(self, plan: DiagnosticPlan) -> MetricsMap:
        """Run every inspector requested by the plan.

        Returns:
            Dict mapping engine name → DiagnosticMetrics.
            Engines that are not configured or fail are still present in the
            dict but have ``metrics.error`` set.
        """
        results: MetricsMap = {}
        for engine in plan.engines:
            results[engine] = self._run_inspector(engine)
        return results

    def available_engines(self) -> list[str]:
        """Return engines that have a DSN configured (not necessarily reachable)."""
        available = []
        if config.PG_DIAG_DSN:
            available.append("postgresql")
        if config.MYSQL_DIAG_DSN:
            available.append("mysql")
        if config.MONGO_DIAG_URI:
            available.append("mongodb")
        return available

    def test_all(self) -> dict[str, bool]:
        """Test connectivity for every configured engine. Used by /health."""
        results: dict[str, bool] = {}
        if config.PG_DIAG_DSN:
            results["postgresql"] = PgInspector(config.PG_DIAG_DSN).test_connection()
        if config.MYSQL_DIAG_DSN:
            results["mysql"] = MysqlInspector(config.MYSQL_DIAG_DSN).test_connection()
        if config.MONGO_DIAG_URI:
            results["mongodb"] = MongoInspector(config.MONGO_DIAG_URI, config.MONGO_DIAG_DB).test_connection()
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_inspector(self, engine: str) -> DiagnosticMetrics:
        try:
            if engine == "postgresql":
                if not config.PG_DIAG_DSN:
                    return DiagnosticMetrics(
                        engine=engine, dsn_label="not configured",
                        error="PG_DIAG_DSN not set in .env",
                    )
                return PgInspector(config.PG_DIAG_DSN).inspect()

            if engine == "mysql":
                if not config.MYSQL_DIAG_DSN:
                    return DiagnosticMetrics(
                        engine=engine, dsn_label="not configured",
                        error="MYSQL_DIAG_DSN not set in .env",
                    )
                return MysqlInspector(config.MYSQL_DIAG_DSN).inspect()

            if engine == "mongodb":
                if not config.MONGO_DIAG_URI:
                    return DiagnosticMetrics(
                        engine=engine, dsn_label="not configured",
                        error="MONGO_DIAG_URI not set in .env",
                    )
                return MongoInspector(config.MONGO_DIAG_URI, config.MONGO_DIAG_DB).inspect()

            return DiagnosticMetrics(
                engine=engine, dsn_label="unknown",
                error=f"Unknown engine: {engine}",
            )

        except Exception as exc:  # noqa: BLE001
            LOG.error("Dispatcher failed for %s: %s", engine, exc)
            return DiagnosticMetrics(
                engine=engine, dsn_label="error",
                error=str(exc),
            )
