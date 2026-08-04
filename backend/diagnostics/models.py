"""Shared data contracts for the diagnostic sub-system.

All inspectors return a ``Metrics`` TypedDict.  The RCA chain and
improvement tracker consume only this type — they never import from
a specific inspector, which keeps them DB-engine agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SlowQuery:
    """One entry from the slow-query or statement-stats view."""
    query: str
    mean_time_ms: float
    calls: int
    total_time_ms: float
    rows_per_call: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TableStat:
    """Table-level access statistics (seq vs index scans)."""
    name: str
    seq_scans: int
    idx_scans: int
    live_rows: int = 0
    dead_rows: int = 0
    last_vacuum: str | None = None


@dataclass
class LockInfo:
    """An active lock or blocking event."""
    pid: int | None
    query: str
    wait_event: str | None
    state: str
    blocked_by: int | None = None


@dataclass
class DiagnosticMetrics:
    """Canonical diagnostic snapshot returned by every inspector.

    Fields are always populated (empty lists when nothing found).
    The ``raw`` dict carries engine-specific extras the RCA chain
    may surface verbatim inside citations.
    """
    engine: str                             # "postgresql" | "mysql" | "mongodb"
    dsn_label: str                          # host:port/db — safe for logging
    slow_queries: list[SlowQuery] = field(default_factory=list)
    table_stats: list[TableStat] = field(default_factory=list)
    active_locks: list[LockInfo] = field(default_factory=list)
    connection_count: int = 0
    error: str | None = None               # non-None when inspection failed
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    def top_slow_queries(self, n: int = 5) -> list[SlowQuery]:
        return sorted(self.slow_queries, key=lambda q: q.mean_time_ms, reverse=True)[:n]

    def seq_scan_tables(self, min_scans: int = 500) -> list[TableStat]:
        """Tables that are being sequentially scanned far more than indexed."""
        return [
            t for t in self.table_stats
            if t.seq_scans >= min_scans and t.seq_scans > (t.idx_scans or 0) * 2
        ]


@dataclass
class DiagnosticPlan:
    """Output of the SymptomRouter — what to check and where."""
    engines: list[str]                     # ["postgresql"] or ["postgresql","mysql"]
    checks: list[str]                      # ["slow_query","lock","seq_scan","vacuum"]
    symptom_summary: str                   # human-readable rephrasing of the complaint
    raw_symptom: str                       # original user text


@dataclass
class ChangeProposal:
    """A candidate fix proposed by the RCA chain, awaiting HITL approval."""
    id: str
    engine: str
    sql_or_command: str
    rationale: str
    risk_level: str                        # "low" | "medium" | "high"
    rollback_sql: str | None
    estimated_improvement: str            # e.g. "75-90% reduction in mean query time"
    status: str = "pending"               # "pending" | "approved" | "rejected" | "applied"
    session_id: str | None = None


@dataclass
class ImprovementReport:
    """Before/after comparison produced by ImprovementTracker."""
    change_proposal_id: str
    engine: str
    metric_name: str
    before_value: float
    after_value: float
    improvement_pct: float
    verdict: str                           # "✅ Improved" | "⚠ No change" | "❌ Worse"
    detail: str
