import os
import time
import statistics
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
from rag_chain import query_rag
from vectorstore import get_index_stats, VALID_DB_FILTERS, get_embedding_model

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

app = FastAPI(
    title="Multi-Database RAG API",
    description="Production RAG over PostgreSQL, MySQL, and MongoDB documentation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Lock down to specific domains in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory request log for /metrics endpoint
request_log: list[dict] = []


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    db_filter: str | None = None   # "postgresql" | "mysql" | "mongodb" | None

    @field_validator("db_filter")
    @classmethod
    def validate_db_filter(cls, v):
        if v is not None and v not in VALID_DB_FILTERS:
            raise ValueError(f"db_filter must be one of {VALID_DB_FILTERS} or null")
        return v

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        if len(v) > 1000:
            raise ValueError("Query too long (max 1000 chars)")
        return v.strip()


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    dbs_used: list[str]
    detected_db: str | None
    chunks_retrieved: int
    chunks_used: int
    refused: bool
    top_chunk_score: float
    latency_ms: float


@app.get("/")
def root():
    return {"status": "ok", "service": "multi-db-rag", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    start = time.perf_counter()

    try:
        result = query_rag(
            query=req.query,
            top_k=req.top_k,
            db_filter=req.db_filter,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query execution error: {str(e)}")

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    request_log.append({
        "latency_ms": latency_ms,
        "refused": result["refused"],
        "chunks_retrieved": result["chunks_retrieved"],
        "detected_db": result["detected_db"],
        "timestamp": time.time(),
    })

    return QueryResponse(**result, latency_ms=latency_ms)


@app.get("/stats")
def get_stats():
    """Index stats + runtime info. Useful for portfolio demos."""
    try:
        index_stats = get_index_stats()
    except Exception as e:
        index_stats = {"error": f"Failed to retrieve index stats: {str(e)}"}
        
    # Report which LLM will be used based on environment/config
    llm_used = "unknown"
    groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    google_model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
    if os.getenv("GROQ_API_KEY") and os.getenv("USE_OLLAMA", "false").lower() != "true":
        llm_used = f"groq ({groq_model})"
    elif os.getenv("USE_OLLAMA", "false").lower() == "true":
        llm_used = "ollama (gemma4:12b-it-qat)"
    elif os.getenv("GOOGLE_API_KEY"):
        llm_used = f"google-gemini ({google_model})"
    else:
        llm_used = "gemini (fallback)"

    return {
        "index": index_stats,
        "embedding_model": get_embedding_model(),
        "llm": llm_used,
        "databases_supported": list(VALID_DB_FILTERS),
    }


@app.get("/metrics")
def get_metrics():
    """P50/P95 latency, refusal rate, per-database breakdown."""
    if not request_log:
        return {"message": "No requests yet"}

    latencies = sorted(r["latency_ms"] for r in request_log)
    n = len(latencies)
    refused = sum(1 for r in request_log if r["refused"])

    # Per-database request counts
    db_counts: dict = {}
    for r in request_log:
        db = r["detected_db"] or "ambiguous"
        db_counts[db] = db_counts.get(db, 0) + 1

    return {
        "total_requests": n,
        "latency_p50_ms": round(latencies[int(n * 0.50)], 1),
        "latency_p95_ms": round(latencies[min(int(n * 0.95), n - 1)], 1),
        "latency_mean_ms": round(statistics.mean(latencies), 1),
        "latency_max_ms": round(max(latencies), 1),
        "refusal_rate_pct": round((refused / n) * 100, 1),
        "requests_by_database": db_counts,
    }
@app.get("/debug/embed")
async def debug_embed(query: str):
    from vectorstore import _embed_single, get_chroma_client, get_or_create_collection
    import numpy as np

    q_emb = _embed_single(query)
    client = get_chroma_client()
    coll = get_or_create_collection(client)
    # Get one stored chunk with its embedding
    result = coll.get(limit=1, include=["embeddings", "documents"])
    if not result or not result['embeddings']:
        return {"error": "No embeddings found in DB"}
    stored_emb = result['embeddings'][0]
    # Cosine similarity
    sim = np.dot(q_emb, stored_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(stored_emb))
    return {
        "query_preview": query,
        "query_embedding_first5": [float(x) for x in q_emb[:5]],
        "stored_embedding_first5": [float(x) for x in stored_emb[:5]],
        "cosine_similarity": float(sim),
        "stored_doc_preview": result['documents'][0][:100]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
