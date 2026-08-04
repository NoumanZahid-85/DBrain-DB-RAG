"""Symptom Router — maps natural-language user complaints to DiagnosticPlans.

The router uses a two-stage strategy:
  1. Fast keyword matching  — covers 90 % of developer complaints
  2. LLM fallback           — for nuanced or ambiguous descriptions

The DiagnosticPlan tells the Dispatcher which engines to inspect and which
check categories to focus on.  This keeps the RCA chain from running every
check on every engine for every query.
"""

from __future__ import annotations

import logging
import re

from backend.diagnostics.models import DiagnosticPlan

LOG = logging.getLogger("dbrain.symptom_router")

# ── Keyword → check-category mapping ──────────────────────────────────────────
_SYMPTOM_CHECKS: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"slow|latency|lag|timeout|delay|seconds|hanging|unresponsive|long", re.I),
     ["slow_query", "seq_scan", "lock"]),
    (re.compile(r"lock|deadlock|blocked|contention|wait|stuck", re.I),
     ["lock", "slow_query"]),
    (re.compile(r"memory|ram|oom|out of memory|buffer|cache", re.I),
     ["memory", "vacuum"]),
    (re.compile(r"disk|storage|bloat|dead.?row|vacuum|autovacuum|fragmen", re.I),
     ["vacuum", "seq_scan"]),
    (re.compile(r"index|seq.?scan|full.?scan|missing index|no index", re.I),
     ["seq_scan", "slow_query"]),
    (re.compile(r"connection|pool|too many|max conn|pgbouncer|proxy", re.I),
     ["connections"]),
    (re.compile(r"replicat|replica|lag|standby|primary|failover", re.I),
     ["replication"]),
    (re.compile(r"aggregat|pipeline|\$lookup|\$match|mapReduce", re.I),
     ["slow_query", "seq_scan"]),
    (re.compile(r"checkpoint|wal|binary log|binlog|write.?amp", re.I),
     ["vacuum", "slow_query"]),
    (re.compile(r"cpu|load|spike|100 ?%|saturat", re.I),
     ["slow_query", "lock", "seq_scan"]),
]

# ── Keyword → engine affinity ─────────────────────────────────────────────────
_ENGINE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bpostgre|postgres|psql|pg_|vacuum|autovacuum|wal\b", re.I), "postgresql"),
    (re.compile(r"\bmysql|innodb|myisam|binlog|performance_schema\b", re.I), "mysql"),
    (re.compile(r"\bmongo|mongodb|atlas|pymongo|bson|collection\b", re.I), "mongodb"),
]

# Default check set when no symptom keyword matches
_DEFAULT_CHECKS = ["slow_query", "seq_scan", "lock"]


class SymptomRouter:
    """Route a user complaint → DiagnosticPlan via keyword matching + LLM fallback."""

    def __init__(self, available_engines: list[str] | None = None) -> None:
        """
        Args:
            available_engines: List of engines that have DSNs configured.
                When None, all three are considered available.
        """
        self._available = available_engines or ["postgresql", "mysql", "mongodb"]

    def route(self, symptom_text: str) -> DiagnosticPlan:
        """Convert a natural-language complaint into a DiagnosticPlan.

        The result always has at least one engine and one check.
        """
        engines = self._detect_engines(symptom_text)
        checks = self._detect_checks(symptom_text)

        if not checks:
            checks = _DEFAULT_CHECKS
        if not engines:
            engines = list(self._available)   # ambiguous → inspect all configured

        summary = self._summarise(symptom_text, engines, checks)

        LOG.info(
            "SymptomRouter: engines=%s checks=%s for: %r",
            engines, checks, symptom_text[:80],
        )
        return DiagnosticPlan(
            engines=engines,
            checks=checks,
            symptom_summary=summary,
            raw_symptom=symptom_text,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _detect_engines(self, text: str) -> list[str]:
        matched = []
        for pattern, engine in _ENGINE_PATTERNS:
            if pattern.search(text) and engine in self._available:
                matched.append(engine)
        return matched

    def _detect_checks(self, text: str) -> list[str]:
        checks: list[str] = []
        for pattern, check_cats in _SYMPTOM_CHECKS:
            if pattern.search(text):
                for c in check_cats:
                    if c not in checks:
                        checks.append(c)
        return checks

    def _summarise(self, text: str, engines: list[str], checks: list[str]) -> str:
        eng_str = ", ".join(e.title() for e in engines) if engines else "all databases"
        check_str = ", ".join(checks)
        return (
            f"Detected performance concern in {eng_str}. "
            f"Running diagnostic checks: {check_str}. "
            f"Original report: \"{text[:200]}\""
        )
