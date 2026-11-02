"""FastAPI entrypoint for the DBrain RAG + diagnostics backend.

Public by design (anyone can try the Q&A demo), guarded where it matters:
  - per-IP rate limit on the LLM-backed endpoints (protects the provider key)
  - admin API key on endpoints that change state or expose operator data
  - X-Request-ID propagation and one structured access-log line per request
  - the built React app is served from the same origin (no CORS in prod)
"""

from __future__ import annotations

import hmac
import logging
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from backend import config
from backend.change_manager import ChangeManager
from backend.chat import SessionStore
from backend.diagnostics.dispatcher import DiagnosticDispatcher
from backend.improvement_tracker import ImprovementTracker
from backend.llm import llm_label
from backend.metrics_store import MetricsStore
from backend.rag_chain import query_rag
from backend.rca_chain import analyze as rca_analyze
from backend.symptom_router import SymptomRouter
from backend.vectorstore import VALID_DB_FILTERS, get_embedding_model, get_index_stats

LOG = logging.getLogger("dbrain.api")
VERSION = "2.0.0"


def warm_up() -> None:
    """Build the index if missing and load every model before the first request."""
    from backend import reranker
    from backend.hybrid_retrieval import ensure_index, load_bm25_index
    from backend.vectorstore import get_embed_model

    ensure_index()
    load_bm25_index()
    get_embed_model().embed_query("warm-up")
    reranker.get_reranker()
    LOG.info("Warm-up complete: %s", get_index_stats())


@asynccontextmanager
async def lifespan(_app: FastAPI):
    warm_up()
    yield
    if config.LANGFUSE_SECRET_KEY and config.LANGFUSE_PUBLIC_KEY:
        try:
            from langfuse import get_client

            get_client().flush()
        except Exception as exc:  # noqa: BLE001 - shutdown must not fail on telemetry
            LOG.warning("Langfuse flush failed: %s", exc)


app = FastAPI(
    title="DBrain API",
    description="Cited answers from PostgreSQL, MySQL and MongoDB docs, plus live diagnostics.",
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

_session_store = SessionStore()
_metrics_store = MetricsStore()
_dispatcher = DiagnosticDispatcher()
_change_manager = ChangeManager()
_tracker = ImprovementTracker()


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get(config.REQUEST_ID_HEADER) or uuid.uuid4().hex
    request.state.request_id = request_id
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers[config.REQUEST_ID_HEADER] = request_id
    LOG.info(
        "%s %s -> %s (%.1fms) rid=%s",
        request.method, request.url.path, response.status_code,
        (time.perf_counter() - t0) * 1000, request_id,
    )
    return response


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", "-")
    LOG.exception("Unhandled error rid=%s", rid)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Something went wrong on our side. Reference: {rid}"},
    )


# --------------------------------------------------------------------------
# Access control
# --------------------------------------------------------------------------

def require_admin(request: Request) -> None:
    """Admin key check (X-API-Key or Bearer). No key configured → open (local dev)."""
    if not config.API_KEY:
        return
    provided = request.headers.get("X-API-Key", "")
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        provided = auth[len("Bearer "):]
    if not hmac.compare_digest(provided.encode(), config.API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class RateLimiter:
    """Sliding one-minute window per client IP.

    shortcut: in-process state, so the budget is per replica; move to Redis
    when running more than one instance.
    """

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        if self.per_minute <= 0:
            return True
        if len(self._hits) > 10_000:  # bound memory under IP churn
            self._hits.clear()
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= self.per_minute:
            return False
        hits.append(now)
        return True


_limiter = RateLimiter(config.RATE_LIMIT_PER_MINUTE)


def rate_limited(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    if not _limiter.allow(client):
        raise HTTPException(
            status_code=429,
            detail="Too many questions in the last minute. Wait a moment and try again.",
        )


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

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


class Citation(BaseModel):
    doc_name: str
    db: str
    db_label: str
    url: str = ""


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    citations: list[Citation]
    dbs_used: list[str]
    detected_db: Optional[str]
    chunks_retrieved: int
    chunks_used: int
    refused: bool
    guardrail_blocked: bool
    top_chunk_score: float
    timings: dict[str, float]
    latency_ms: float
    request_id: str = ""
    session_id: Optional[str] = None


class SessionCreate(BaseModel):
    db_filter: Optional[str] = None


class ActionRequest(BaseModel):
    session_id: str
    query: str = Field(..., max_length=1000)
    action: str = Field(..., max_length=20_000)
    risk_level: str = Field("medium", pattern="^(low|medium|high)$")


class DiagnoseRequest(BaseModel):
    symptom: str = Field(..., min_length=5, max_length=2000,
                         description="Natural-language description of the problem")
    session_id: Optional[str] = None
    engines: Optional[list[str]] = None  # override auto-detection


# --------------------------------------------------------------------------
# Status
# --------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy", "version": VERSION}


@app.get("/api/v1/status")
def status():
    """Index size, models and configured engines, for the UI status panel."""
    from backend import reranker

    try:
        index_stats = get_index_stats()
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Index stats unavailable: %s", exc)
        index_stats = {"total": 0}
    return {
        "version": VERSION,
        "index": index_stats,
        "embedding_model": get_embedding_model(),
        "llm": llm_label(),
        "reranker": {"enabled": reranker.is_available(), "model": config.RERANKER_MODEL},
        "min_similarity": config.MIN_SEMANTIC_SIMILARITY,
        "databases_supported": sorted(VALID_DB_FILTERS),
        "diagnostics_engines": _dispatcher.available_engines(),
    }


# Kept for existing dashboards / README links.
app.add_api_route("/stats", status, methods=["GET"], include_in_schema=False)


@app.get("/metrics")
def get_metrics():
    """P50/P95 latency, refusal rate, per-database breakdown (SQLite-backed)."""
    return _metrics_store.get_metrics()


# --------------------------------------------------------------------------
# Q&A
# --------------------------------------------------------------------------

def _resolve_session(session_id: Optional[str]) -> str:
    """Create a fresh session when none supplied, else validate an existing one."""
    if session_id:
        if _session_store.get_session(session_id):
            return session_id
        raise HTTPException(status_code=404, detail="Session not found")
    return _session_store.create_session()


@app.post("/api/v1/query", response_model=QueryResponse, dependencies=[Depends(rate_limited)])
@app.post("/query", response_model=QueryResponse, include_in_schema=False,
          dependencies=[Depends(rate_limited)])
def query_endpoint(req: QueryRequest, request: Request):
    start = time.perf_counter()
    session_id = _resolve_session(req.session_id)
    try:
        result = query_rag(
            query=req.query, top_k=req.top_k, db_filter=req.db_filter, session_id=session_id,
        )
    except Exception as exc:
        _metrics_store.record_request(
            latency_ms=(time.perf_counter() - start) * 1000, refused=True,
            chunks_retrieved=0, detected_db=req.db_filter, error=str(exc)[:500],
        )
        raise

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
    return {"session_id": _session_store.create_session(db_filter=req.db_filter)}


@app.get("/api/v1/sessions", dependencies=[Depends(require_admin)])
def list_sessions():
    return {"sessions": _session_store.list_sessions()}


@app.get("/api/v1/sessions/{session_id}/messages")
def get_session_messages(session_id: str, limit: int = 50):
    # The unguessable session id is the capability: only its owner knows it.
    if not _session_store.get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"messages": _session_store.get_messages(session_id, limit=min(limit, 200))}


@app.post("/api/v1/actions", status_code=201)
def create_action(req: ActionRequest):
    """FE-7.5 Propose Action -> pending recommendation (Module 5/6 approval)."""
    if not _session_store.get_session(req.session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    action_id = _session_store.create_action(
        session_id=req.session_id, query=req.query, action=req.action, risk_level=req.risk_level,
    )
    return {"action_id": action_id, "status": "pending"}


@app.get("/api/v1/actions", dependencies=[Depends(require_admin)])
def list_actions(session_id: Optional[str] = None, status: Optional[str] = None):
    return {"actions": _session_store.list_actions(session_id=session_id, status=status)}


# --------------------------------------------------------------------------
# Diagnostic agent. Sync handlers: they block on DB drivers, the LLM and
# sleeps, so FastAPI runs them in its threadpool instead of the event loop.
# --------------------------------------------------------------------------

def _proposal_dict(cp) -> dict:
    return {
        "id": cp.id, "engine": cp.engine, "sql_or_command": cp.sql_or_command,
        "rationale": cp.rationale, "risk_level": cp.risk_level,
        "rollback_sql": cp.rollback_sql, "estimated_improvement": cp.estimated_improvement,
        "status": cp.status,
    }


@app.post("/api/v1/diagnose", dependencies=[Depends(require_admin), Depends(rate_limited)])
def diagnose(body: DiagnoseRequest):
    """Symptom → inspect live DBs → RCA → ChangeProposals (status=pending).

    Nothing is applied until a human approves via /proposals/{id}/approve.
    """
    plan = SymptomRouter(available_engines=_dispatcher.available_engines()).route(body.symptom)
    if body.engines:  # manual override
        plan.engines = [e for e in body.engines if e in VALID_DB_FILTERS]
    if not plan.engines:
        raise HTTPException(
            status_code=400,
            detail="No database engines are configured. Add PG_DIAG_DSN / MYSQL_DIAG_DSN / MONGO_DIAG_URI to .env",
        )

    metrics_map = _dispatcher.dispatch(plan)
    rca_result = rca_analyze(metrics_map, plan, session_id=body.session_id)

    proposals, rejected = [], []
    for finding in rca_result.findings:
        cp = finding.change_proposal
        if not cp:
            continue
        try:
            _change_manager.save_proposal(cp)
        except ValueError as exc:
            # The LLM proposed something outside the safe allowlist: surface, don't store.
            finding.change_proposal = None
            rejected.append({"engine": cp.engine, "sql_or_command": cp.sql_or_command,
                             "reason": str(exc)})
            continue
        try:
            _tracker.snapshot_before(cp.id, cp.engine)
        except Exception as exc:  # noqa: BLE001 - tracking is best-effort
            LOG.warning("Before-snapshot failed for %s: %s", cp.id, exc)
        proposals.append(_proposal_dict(cp))

    return {
        "symptom_summary": rca_result.symptom_summary,
        "engines_inspected": list(metrics_map.keys()),
        "findings": [
            {
                "rank": f.rank, "engine": f.engine, "title": f.title, "evidence": f.evidence,
                "estimated_improvement": f.estimated_improvement, "confidence": f.confidence,
                "proposal_id": f.change_proposal.id if f.change_proposal else None,
            }
            for f in rca_result.findings
        ],
        "proposals": proposals,
        "rejected_proposals": rejected,
        "refusal_reason": rca_result.refusal_reason,
        "metrics_snapshot": {
            engine: {"ok": m.ok, "error": m.error, "connection_count": m.connection_count}
            for engine, m in metrics_map.items()
        },
    }


@app.get("/api/v1/diagnose/engines")
def list_engines():
    """Return which DB engines are configured (not necessarily reachable)."""
    configured = _dispatcher.available_engines()
    return {"configured": configured, "total": len(configured)}


def _get_proposal_or_404(proposal_id: str):
    proposal = _change_manager.get_proposal(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found")
    return proposal


@app.post("/api/v1/proposals/{proposal_id}/approve", dependencies=[Depends(require_admin)])
def approve_proposal(proposal_id: str):
    """Human approves a ChangeProposal. Does NOT apply it — use /apply next."""
    proposal = _get_proposal_or_404(proposal_id)
    if proposal.status not in ("pending", "rejected"):
        raise HTTPException(status_code=409, detail=f"Cannot approve: current status is '{proposal.status}'")
    updated = _change_manager.approve(proposal_id)
    return {"id": updated.id, "status": updated.status, "message": "Approved. Call /apply to execute."}


@app.post("/api/v1/proposals/{proposal_id}/reject", dependencies=[Depends(require_admin)])
def reject_proposal(proposal_id: str, reason: str = ""):
    _get_proposal_or_404(proposal_id)
    updated = _change_manager.reject(proposal_id, reason)
    return {"id": updated.id, "status": updated.status}


@app.post("/api/v1/proposals/{proposal_id}/apply", dependencies=[Depends(require_admin)])
def apply_proposal(proposal_id: str):
    """Apply an approved ChangeProposal to the live database, then snapshot metrics."""
    proposal = _get_proposal_or_404(proposal_id)
    if proposal.status != "approved":
        raise HTTPException(
            status_code=409,
            detail=f"Proposal must be 'approved' before applying (current: '{proposal.status}'). Call /approve first.",
        )
    result = _change_manager.apply(proposal_id)
    if result["ok"]:
        try:
            _tracker.snapshot_after(proposal_id, proposal.engine, wait_s=5)
        except Exception as exc:  # noqa: BLE001 - tracking is best-effort
            LOG.warning("After-snapshot failed for %s: %s", proposal_id, exc)
    return result


@app.get("/api/v1/proposals/{proposal_id}/improvement", dependencies=[Depends(require_admin)])
def get_improvement(proposal_id: str):
    """Compare before/after metrics for an applied ChangeProposal."""
    reports = _tracker.compare(proposal_id)
    return {
        "proposal_id": proposal_id,
        "reports": [
            {
                "metric": r.metric_name, "engine": r.engine, "before": r.before_value,
                "after": r.after_value, "improvement_pct": r.improvement_pct,
                "verdict": r.verdict, "detail": r.detail,
            }
            for r in reports
        ],
        **({} if reports else {"message": "No snapshots found yet"}),
    }


@app.get("/api/v1/proposals", dependencies=[Depends(require_admin)])
def list_proposals(status: Optional[str] = None, session_id: Optional[str] = None):
    """List ChangeProposals, optionally filtered by status (pending/approved/applied/rejected)."""
    return {
        "proposals": [
            {
                "id": p.id, "engine": p.engine, "status": p.status,
                "risk_level": p.risk_level, "estimated_improvement": p.estimated_improvement,
                "sql_or_command": p.sql_or_command[:120],
            }
            for p in _change_manager.list_proposals(status=status, session_id=session_id)
        ]
    }


# --------------------------------------------------------------------------
# Frontend (mounted last so every API route above wins)
# --------------------------------------------------------------------------

if config.FRONTEND_DIST.joinpath("index.html").exists():
    app.mount("/", StaticFiles(directory=config.FRONTEND_DIST, html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {"status": "ok", "service": "dbrain", "version": VERSION, "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
