"""FastAPI entrypoint for the DBrain Module 7 RAG backend.

Production-safe by default:
  - CORS allowlist from CORS_ORIGINS (no wildcard)
  - optional API-key auth (API_KEY) with a pluggable JWT hook for Module 10
  - X-Request-ID propagation
  - SQLite-backed /metrics and FE-7.1 session endpoints
  - versioned routes under /api/v1 with legacy aliases for the current UI
"""

from __future__ import annotations

import logging
import time
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional

from backend import config
from backend.change_manager import ChangeManager
from backend.chat import SessionStore
from backend.diagnostics.dispatcher import DiagnosticDispatcher
from backend.guardrails import check_command_guardrail, check_prompt_guardrail
from backend.improvement_tracker import ImprovementTracker
from backend.llm import llm_label
from backend.metrics_store import MetricsStore
from backend.rag_chain import query_rag
from backend.rca_chain import analyze as rca_analyze
from backend.symptom_router import SymptomRouter
from backend.vectorstore import VALID_DB_FILTERS, get_embedding_model, get_index_stats

LOG = logging.getLogger("dbrain.api")
load_dotenv(config.PROJECT_ROOT / ".env", override=False)

app = FastAPI(
    title="Multi-Database RAG API",
    description="Production RAG over PostgreSQL, MySQL, and MongoDB documentation",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

_session_store = SessionStore()
_metrics_store = MetricsStore()


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Optional API-key auth + X-Request-ID plumbing (JWT hook for Module 10)."""
    request_id = request.headers.get(config.REQUEST_ID_HEADER) or uuid.uuid4().hex
    request.state.request_id = request_id
    request.state.t0 = time.perf_counter()

    if config.API_KEY and request.url.path not in config.AUTH_EXEMPT_PATHS:
        provided = request.headers.get("X-API-Key") or ""
        bearer = request.headers.get("Authorization", "")
        if bearer.startswith("Bearer "):
            provided = bearer[len("Bearer "):]
        if provided != config.API_KEY:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )

    response = await call_next(request)
    response.headers[config.REQUEST_ID_HEADER] = request_id
    return response


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    response = await call_next(request)
    elapsed = round(
        (time.perf_counter() - getattr(request.state, "t0", time.perf_counter())) * 1000,
        1,
    )
    LOG.info(
        "%s %s -> %s (%.1fms) rid=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
        getattr(request.state, "request_id", "-"),
    )
    return response


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(10, ge=1, le=50)
    db_filter: Optional[str] = Field(None, description="postgresql | mysql | mongodb")
    session_id: Optional[str] = Field(None, description="FE-7.1 chat session id")

    @field_validator("db_filter")
    @classmethod
    def validate_db_filter(cls, v):
        if v is not None and v not in VALID_DB_FILTERS:
            raise ValueError(f"db_filter must be one of {sorted(VALID_DB_FILTERS)} or null")
        return v

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty")
        return v


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    dbs_used: list[str]
    detected_db: Optional[str]
    chunks_retrieved: int
    chunks_used: int
    refused: bool
    top_chunk_score: float
    latency_ms: float
    request_id: str = ""
    session_id: Optional[str] = None


class SessionCreate(BaseModel):
    db_filter: Optional[str] = None


class ActionRequest(BaseModel):
    session_id: str
    query: str
    action: str
    risk_level: str = "medium"


def _resolve_session(session_id: Optional[str]) -> str:
    """Create a fresh session when none supplied, else validate an existing one."""
    if session_id:
        if _session_store.get_session(session_id):
            return session_id
        raise HTTPException(status_code=404, detail="Session not found")
    return _session_store.create_session()


@app.get("/")
def root():
    return {"status": "ok", "service": "multi-db-rag", "version": app.version}


@app.get("/health")
def health():
    return {"status": "healthy", "version": app.version}


@app.post("/api/v1/query", response_model=QueryResponse)
@app.post("/query", response_model=QueryResponse, include_in_schema=False)
def query_endpoint(req: QueryRequest, request: Request):
    start = time.perf_counter()

    try:
        session_id = _resolve_session(req.session_id)
    except HTTPException as exc:
        raise exc

    # FE-7.4 command guardrail runs before any retrieval.
    blocked, reason = check_command_guardrail(req.query)
    if blocked:
        _metrics_store.record_request(
            latency_ms=0,
            refused=True,
            chunks_retrieved=0,
            detected_db=req.db_filter,
        )
        if session_id:
            _session_store.add_message(session_id, "user", req.query)
            _session_store.add_message(session_id, "assistant", reason)
        return QueryResponse(
            query=req.query,
            answer=reason,
            sources=[],
            dbs_used=[],
            detected_db=req.db_filter,
            chunks_retrieved=0,
            chunks_used=0,
            refused=True,
            top_chunk_score=0.0,
            latency_ms=0.0,
            request_id=getattr(request.state, "request_id", ""),
            session_id=session_id,
        )

    # Prompt-security guardrail (injection / jailbreak / leakage / extraction).
    blocked, reason = check_prompt_guardrail(req.query)
    if blocked:
        _metrics_store.record_request(
            latency_ms=0,
            refused=True,
            chunks_retrieved=0,
            detected_db=req.db_filter,
        )
        if session_id:
            _session_store.add_message(session_id, "user", req.query)
            _session_store.add_message(session_id, "assistant", reason)
        return QueryResponse(
            query=req.query,
            answer=reason,
            sources=[],
            dbs_used=[],
            detected_db=req.db_filter,
            chunks_retrieved=0,
            chunks_used=0,
            refused=True,
            top_chunk_score=0.0,
            latency_ms=0.0,
            request_id=getattr(request.state, "request_id", ""),
            session_id=session_id,
        )

    try:
        result = query_rag(
            query=req.query,
            top_k=req.top_k,
            db_filter=req.db_filter,
            session_id=session_id,
        )
    except Exception as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        _metrics_store.record_request(
            latency_ms=latency_ms, refused=True, chunks_retrieved=0,
            detected_db=req.db_filter, error=str(e),
        )
        raise HTTPException(status_code=500, detail=f"RAG query execution error: {str(e)}")

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    _metrics_store.record_request(
        latency_ms=latency_ms,
        refused=result["refused"],
        chunks_retrieved=result["chunks_retrieved"],
        detected_db=result["detected_db"],
    )

    return QueryResponse(
        **result,
        latency_ms=latency_ms,
        request_id=getattr(request.state, "request_id", ""),
        session_id=session_id,
    )


@app.post("/api/v1/sessions", status_code=201)
def create_session(req: SessionCreate):
    session_id = _session_store.create_session(db_filter=req.db_filter)
    return {"session_id": session_id}


@app.get("/api/v1/sessions")
def list_sessions():
    return {"sessions": _session_store.list_sessions()}


@app.get("/api/v1/sessions/{session_id}/messages")
def get_session_messages(session_id: str, limit: int = 20):
    if not _session_store.get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"messages": _session_store.get_messages(session_id, limit=limit)}


@app.post("/api/v1/actions", status_code=201)
def create_action(req: ActionRequest):
    """FE-7.5 Propose Action -> pending recommendation (Module 5/6 approval)."""
    if not _session_store.get_session(req.session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    action_id = _session_store.create_action(
        session_id=req.session_id,
        query=req.query,
        action=req.action,
        risk_level=req.risk_level,
    )
    return {"action_id": action_id, "status": "pending"}


@app.get("/api/v1/actions")
def list_actions(session_id: Optional[str] = None, status: Optional[str] = None):
    return {"actions": _session_store.list_actions(session_id=session_id, status=status)}


@app.get("/stats")
def get_stats():
    """Index stats + runtime info. Useful for portfolio demos."""
    try:
        index_stats = get_index_stats()
    except Exception as e:
        index_stats = {"error": f"Failed to retrieve index stats: {str(e)}"}

    return {
        "index": index_stats,
        "embedding_model": get_embedding_model(),
        "llm": llm_label(),
        "databases_supported": sorted(VALID_DB_FILTERS),
        "reranker_available": _reranker_status(),
    }


def _reranker_status() -> dict:
    from backend import reranker
    return {
        "enabled": reranker.is_available(),
        "model": config.RERANKER_MODEL,
        "package_installed": reranker.HAS_RERANKER,
    }


@app.get("/metrics")
def get_metrics():
    """P50/P95 latency, refusal rate, per-database breakdown (SQLite-backed)."""
    return _metrics_store.get_metrics()


# =============================================================================
# Diagnostic Agent endpoints  (DBrain v2)
# =============================================================================

_dispatcher      = DiagnosticDispatcher()
_change_manager  = ChangeManager()
_tracker         = ImprovementTracker()


class DiagnoseRequest(BaseModel):
    symptom: str = Field(..., min_length=5, description="Natural-language description of the problem")
    session_id: Optional[str] = None
    engines: Optional[list[str]] = None  # override auto-detection


@app.post("/api/v1/diagnose")
async def diagnose(body: DiagnoseRequest, request: Request):
    """Full diagnostic pipeline: symptom → inspect → RCA → ChangeProposals.

    The agent:
    1. Routes the symptom to a DiagnosticPlan (which engines + checks).
    2. Runs the appropriate inspector(s) against live DB(s).
    3. Fuses live metrics + doc-RAG to produce ranked root-cause findings.
    4. Persists each ChangeProposal (status=pending) and returns them.

    No changes are applied until the human approves via /proposals/{id}/approve.
    """
    router  = SymptomRouter(available_engines=_dispatcher.available_engines())
    plan    = router.route(body.symptom)

    if body.engines:  # manual override
        plan.engines = [e for e in body.engines if e in ("postgresql", "mysql", "mongodb")]

    if not plan.engines:
        raise HTTPException(
            status_code=400,
            detail="No database engines are configured. Add PG_DIAG_DSN / MYSQL_DIAG_DSN / MONGO_DIAG_URI to .env",
        )

    metrics_map = _dispatcher.dispatch(plan)
    rca_result  = rca_analyze(metrics_map, plan, session_id=body.session_id)

    proposals = []
    for finding in rca_result.findings:
        if finding.change_proposal:
            cp = finding.change_proposal
            # Save before-snapshot for improvement tracking
            try:
                _tracker.snapshot_before(cp.id, cp.engine)
            except Exception:
                pass
            _change_manager.save_proposal(cp)
            proposals.append({
                "id": cp.id, "engine": cp.engine,
                "sql_or_command": cp.sql_or_command,
                "rationale": cp.rationale,
                "risk_level": cp.risk_level,
                "rollback_sql": cp.rollback_sql,
                "estimated_improvement": cp.estimated_improvement,
                "status": cp.status,
            })

    return {
        "symptom_summary": rca_result.symptom_summary,
        "engines_inspected": list(metrics_map.keys()),
        "findings": [
            {
                "rank":                  f.rank,
                "engine":               f.engine,
                "title":                f.title,
                "evidence":             f.evidence,
                "estimated_improvement": f.estimated_improvement,
                "confidence":           f.confidence,
                "proposal_id":          f.change_proposal.id if f.change_proposal else None,
            }
            for f in rca_result.findings
        ],
        "proposals": proposals,
        "refusal_reason": rca_result.refusal_reason,
        "metrics_snapshot": {
            engine: {"ok": m.ok, "error": m.error, "connection_count": m.connection_count}
            for engine, m in metrics_map.items()
        },
    }


@app.get("/api/v1/diagnose/engines")
async def list_engines():
    """Return which DB engines are configured and reachable."""
    configured = _dispatcher.available_engines()
    return {"configured": configured, "total": len(configured)}


@app.post("/api/v1/proposals/{proposal_id}/approve")
async def approve_proposal(proposal_id: str):
    """Human approves a ChangeProposal. Does NOT apply it — use /apply next."""
    proposal = _change_manager.get_proposal(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found")
    if proposal.status not in ("pending", "rejected"):
        raise HTTPException(status_code=409, detail=f"Cannot approve: current status is '{proposal.status}'")
    updated = _change_manager.approve(proposal_id)
    return {"id": updated.id, "status": updated.status, "message": "Approved. Call /apply to execute."}


@app.post("/api/v1/proposals/{proposal_id}/reject")
async def reject_proposal(proposal_id: str, reason: str = ""):
    """Human rejects a ChangeProposal."""
    proposal = _change_manager.get_proposal(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found")
    updated = _change_manager.reject(proposal_id, reason)
    return {"id": updated.id, "status": updated.status}


@app.post("/api/v1/proposals/{proposal_id}/apply")
async def apply_proposal(proposal_id: str):
    """Apply an approved ChangeProposal to the live database, then snapshot metrics."""
    proposal = _change_manager.get_proposal(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found")
    if proposal.status != "approved":
        raise HTTPException(
            status_code=409,
            detail=f"Proposal must be 'approved' before applying (current: '{proposal.status}')."
                   " Call /approve first.",
        )
    result = _change_manager.apply(proposal_id)
    # Snapshot after-metrics for the improvement tracker (5s wait for index build)
    if result["ok"]:
        try:
            _tracker.snapshot_after(proposal_id, proposal.engine, wait_s=5)
        except Exception:
            pass
    return result


@app.get("/api/v1/proposals/{proposal_id}/improvement")
async def get_improvement(proposal_id: str):
    """Compare before/after metrics for an applied ChangeProposal."""
    reports = _tracker.compare(proposal_id)
    if not reports:
        return {"proposal_id": proposal_id, "reports": [], "message": "No snapshots found yet"}
    return {
        "proposal_id": proposal_id,
        "reports": [
            {
                "metric":           r.metric_name,
                "engine":           r.engine,
                "before":           r.before_value,
                "after":            r.after_value,
                "improvement_pct":  r.improvement_pct,
                "verdict":          r.verdict,
                "detail":           r.detail,
            }
            for r in reports
        ],
    }


@app.get("/api/v1/proposals")
async def list_proposals(status: Optional[str] = None, session_id: Optional[str] = None):
    """List ChangeProposals, optionally filtered by status (pending/approved/applied/rejected)."""
    proposals = _change_manager.list_proposals(status=status, session_id=session_id)
    return {
        "proposals": [
            {
                "id": p.id, "engine": p.engine, "status": p.status,
                "risk_level": p.risk_level, "estimated_improvement": p.estimated_improvement,
                "sql_or_command": p.sql_or_command[:120],
            }
            for p in proposals
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)