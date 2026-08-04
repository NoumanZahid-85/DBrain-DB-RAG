"""Root Cause Analysis (RCA) chain.

Fuses live DiagnosticMetrics from the inspectors with doc-RAG retrieval to
produce a pinpoint root-cause diagnosis, ranked findings, and a structured
ChangeProposal for each finding.

Architecture seam:
    rca_chain.analyze(metrics_map, symptom) → RCAResult
        ├── formats live metrics into a context block
        ├── queries the existing rag_chain (doc retrieval) for each DB
        ├── sends [system + metrics context + docs context + question] to LLM
        └── parses the LLM's structured response into RCAResult

The RCA chain reuses the existing ``query_rag`` retrieval path so the doc
corpus benefits automatically from any future corpus expansion.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from backend.diagnostics.models import (
    DiagnosticMetrics, DiagnosticPlan, ChangeProposal,
)
from backend.hybrid_retrieval import hybrid_search
from backend.llm import get_llm

LOG = logging.getLogger("dbrain.rca_chain")


@dataclass
class RCAFinding:
    """One root-cause finding with evidence + a proposed fix."""
    rank: int
    engine: str
    title: str                          # short headline, e.g. "Missing index on orders.customer_id"
    evidence: str                       # what the metrics/docs show
    estimated_improvement: str          # "75-90% query time reduction"
    confidence: str                     # "high" | "medium" | "low"
    change_proposal: ChangeProposal | None = None


@dataclass
class RCAResult:
    """Full output of the RCA chain."""
    symptom_summary: str
    findings: list[RCAFinding] = field(default_factory=list)
    refusal_reason: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.findings)


# ── System prompt ─────────────────────────────────────────────────────────────
_RCA_SYSTEM = """You are DBrain, an expert database performance diagnostics engine.

You will be given:
1. LIVE METRICS — real data from the running database (slow queries, table scan ratios, locks, etc.)
2. DOCUMENTATION CONTEXT — relevant excerpts from official database documentation

Your job: pinpoint the root cause of the performance problem and propose a concrete, safe fix.

OUTPUT FORMAT — respond with valid JSON only, no prose outside the JSON:
{
  "findings": [
    {
      "rank": 1,
      "engine": "postgresql",
      "title": "short headline (< 80 chars)",
      "evidence": "what in the metrics + docs proves this",
      "estimated_improvement": "percentage or qualitative improvement if fix applied",
      "confidence": "high|medium|low",
      "fix_sql": "the exact SQL or command to apply (safe, non-destructive)",
      "fix_rationale": "why this fixes it, citing the source",
      "rollback_sql": "how to undo if needed",
      "risk_level": "low|medium|high"
    }
  ],
  "refusal_reason": null
}

RULES:
- If the metrics provide no clear signal, set findings=[] and explain in refusal_reason.
- Only propose fixes that are safe to apply to a running system: CREATE INDEX CONCURRENTLY, ANALYZE, ALTER TABLE SET ... (config params), not DROP or destructive DDL.
- Rank findings by impact (highest first).
- Always cite the documentation source for every recommendation.
- Be specific: name the exact table, query, or parameter. Never say "consider optimizing queries".
"""


def _format_metrics_context(metrics_map: dict[str, DiagnosticMetrics]) -> str:
    """Render live metrics as a readable context block for the LLM."""
    parts = []
    for engine, m in metrics_map.items():
        block = [f"=== LIVE METRICS: {engine.upper()} ({m.dsn_label}) ==="]
        if not m.ok:
            block.append(f"  ⚠ Inspector error: {m.error}")
            parts.append("\n".join(block))
            continue

        block.append(f"  Active connections: {m.connection_count}")

        if m.slow_queries:
            block.append("\nSlowest queries (by mean execution time):")
            for i, q in enumerate(m.top_slow_queries(5), 1):
                block.append(
                    f"  [{i}] mean={q.mean_time_ms:.1f}ms  calls={q.calls}  "
                    f"rows/call={q.rows_per_call:.0f}\n"
                    f"      {q.query[:200]}"
                )
        else:
            block.append("\nNo slow query data (pg_stat_statements / profiler may be disabled).")

        if m.table_stats:
            high_seq = m.seq_scan_tables(min_scans=500)
            if high_seq:
                block.append("\nTables with high sequential scan ratio (possible missing index):")
                for t in high_seq[:5]:
                    block.append(
                        f"  {t.name}: seq_scans={t.seq_scans}  idx_scans={t.idx_scans}  "
                        f"live_rows={t.live_rows}"
                    )

        if m.active_locks:
            blocking = [lk for lk in m.active_locks if lk.blocked_by]
            if blocking:
                block.append(f"\nBlocking lock chains detected: {len(blocking)} queries waiting")
                for lk in blocking[:3]:
                    block.append(f"  PID {lk.pid} blocked by {lk.blocked_by}: {lk.query[:100]}")

        if m.raw.get("bgwriter"):
            bg = m.raw["bgwriter"]
            if bg.get("checkpoints_req", 0) > bg.get("checkpoints_timed", 1):
                block.append(
                    f"\n⚠ Forced checkpoints: {bg['checkpoints_req']} required "
                    f"vs {bg['checkpoints_timed']} timed — consider increasing max_wal_size"
                )

        if m.raw.get("server_status"):
            ss = m.raw["server_status"]
            dirty = ss.get("wt_cache_dirty_pct", 0)
            if dirty and dirty > 20:
                block.append(f"\n⚠ WiredTiger cache dirty: {dirty:.1f}% — possible eviction pressure")

        parts.append("\n".join(block))
    return "\n\n".join(parts)


def _retrieve_docs(symptom: str, engines: list[str], top_k: int = 6) -> str:
    """Pull relevant doc chunks for each engine using the existing hybrid search."""
    parts = []
    for engine in engines:
        try:
            chunks = hybrid_search(symptom, top_k=top_k, db_filter=engine)
            if chunks:
                parts.append(f"=== DOCUMENTATION: {engine.upper()} ===")
                for c in chunks[:top_k]:
                    parts.append(
                        f"[{c['doc_name']} | score={c.get('normalized_score', 0):.2f}]\n{c['text']}"
                    )
        except Exception as exc:
            LOG.warning("Doc retrieval failed for %s: %s", engine, exc)
    return "\n\n---\n\n".join(parts) if parts else "No documentation retrieved."


def analyze(
    metrics_map: dict[str, DiagnosticMetrics],
    plan: DiagnosticPlan,
    session_id: str | None = None,
) -> RCAResult:
    """Core RCA function — fuses live metrics + docs → root cause findings.

    Args:
        metrics_map: engine → DiagnosticMetrics from the dispatcher.
        plan:        DiagnosticPlan with symptom summary and engine list.
        session_id:  Optional session for context (unused in v1).

    Returns:
        RCAResult with ranked findings and optional ChangeProposals.
    """
    metrics_ctx = _format_metrics_context(metrics_map)
    docs_ctx = _retrieve_docs(plan.symptom_summary, plan.engines)

    user_message = (
        f"SYMPTOM REPORTED BY USER:\n{plan.raw_symptom}\n\n"
        f"LIVE DATABASE METRICS:\n{metrics_ctx}\n\n"
        f"RELEVANT DOCUMENTATION:\n{docs_ctx}\n\n"
        f"Diagnose the root cause(s) and propose safe, specific fixes. "
        f"Return only the JSON object described in the system prompt."
    )

    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [SystemMessage(content=_RCA_SYSTEM), HumanMessage(content=user_message)]

    try:
        llm = get_llm()
        response = llm.invoke(messages)
        raw_text = str(response.content).strip()

        # Strip markdown code fences if the LLM wraps the JSON
        raw_text = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        LOG.warning("RCA chain: LLM did not return valid JSON: %s", exc)
        return RCAResult(
            symptom_summary=plan.symptom_summary,
            refusal_reason=f"Diagnosis could not be parsed: {exc}",
        )
    except Exception as exc:  # noqa: BLE001
        LOG.error("RCA chain LLM call failed: %s", exc)
        return RCAResult(
            symptom_summary=plan.symptom_summary,
            refusal_reason=f"LLM error: {exc}",
        )

    if parsed.get("refusal_reason"):
        return RCAResult(
            symptom_summary=plan.symptom_summary,
            refusal_reason=parsed["refusal_reason"],
        )

    findings = []
    for f in parsed.get("findings", []):
        proposal = None
        if f.get("fix_sql"):
            proposal = ChangeProposal(
                id=uuid.uuid4().hex[:12],
                engine=f.get("engine", "unknown"),
                sql_or_command=f["fix_sql"],
                rationale=f.get("fix_rationale", ""),
                risk_level=f.get("risk_level", "medium"),
                rollback_sql=f.get("rollback_sql"),
                estimated_improvement=f.get("estimated_improvement", ""),
                session_id=session_id,
            )
        findings.append(RCAFinding(
            rank=int(f.get("rank", 99)),
            engine=f.get("engine", "unknown"),
            title=f.get("title", "Unnamed finding"),
            evidence=f.get("evidence", ""),
            estimated_improvement=f.get("estimated_improvement", ""),
            confidence=f.get("confidence", "medium"),
            change_proposal=proposal,
        ))

    findings.sort(key=lambda x: x.rank)
    return RCAResult(symptom_summary=plan.symptom_summary, findings=findings)
