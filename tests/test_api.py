"""Hermetic API integration tests.

The LLM factory and retrieval are monkeypatched so no network, API key, or
RAG index is required. Verifies routing, guardrail refusal, metrics
persistence, session creation, and the /api/v1/actions endpoint.
"""

import sys
import os
import statistics

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
from fastapi.testclient import TestClient

from backend import config, llm


class FakeLLM:
    """Drop-in chat model answering with the number of chunks it received."""

    def invoke(self, messages):
        last = messages[-1].content
        n_context = last.count("[Chunk ") if "[Chunk " in last else 0
        n = max(1, n_context)
        return type("Resp", (), {"content": (
            f"Based on the documentation, there are {n} relevant chunks. "
            "[Source: pg_indexes.txt (PostgreSQL)]"
        )})()


def _fake_hybrid_search(query, top_k=15, db_filter=None):
    return [
        {
            "text": "A B-tree index handles equality and range queries.",
            "source": "pg_indexes.txt",
            "db": "postgresql",
            "db_label": "PostgreSQL",
            "doc_name": "pg_indexes.txt",
            "source_url": "",
            "chunk_id": 1,
            "score": 0.9,
            "semantic_score": 0.62,
            "fused_score": 0.03,
            "normalized_score": 1.0,
            "rerank_score": 0.9,
        },
        {
            "text": "Indexes speed up indexed column lookups.",
            "source": "pg_indexes.txt",
            "db": "postgresql",
            "db_label": "PostgreSQL",
            "doc_name": "pg_indexes.txt",
            "source_url": "",
            "chunk_id": 2,
            "score": 0.6,
            "fused_score": 0.02,
            "normalized_score": 0.5,
            "rerank_score": 0.4,
        },
    ]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """App wired to a throwaway SQLite + fake LLM/retrieval."""
    # Point SQLite stores at a temp file so tests are hermetic.
    monkeypatch.setattr("backend.chat.config.DB_FILE", tmp_path / "test.db")
    monkeypatch.setattr("backend.metrics_store.config.DB_FILE", tmp_path / "test.db")

    # Never touch the real LLM or the real index.
    monkeypatch.setattr("backend.llm.get_llm", lambda: FakeLLM())
    monkeypatch.setattr("backend.rag_chain.get_llm", lambda: FakeLLM())
    monkeypatch.setattr("backend.rag_chain.hybrid_search", _fake_hybrid_search)
    # Disable the reranker for hermetic tests.
    monkeypatch.setattr("backend.reranker.is_available", lambda: False)

    # Rebuild the app's stores against the temp DB (fresh import state).
    from backend import chat as chat_mod
    from backend import metrics_store as ms_mod
    from backend import main as main_mod
    import backend.rag_chain as rc

    monkeypatch.setattr(main_mod, "warm_up", lambda: None)  # no real index/models
    monkeypatch.setattr(main_mod, "_limiter", main_mod.RateLimiter(0))
    rc._session_store = None
    # main.py builds its stores at import time; rebuild against temp DB.
    main_mod._session_store = chat_mod.SessionStore()
    main_mod._metrics_store = ms_mod.MetricsStore()

    with TestClient(main_mod.app) as c:
        yield c


def test_health_and_status(client, monkeypatch):
    monkeypatch.setattr("backend.main.get_index_stats", lambda: {"total": 3})
    assert client.get("/health").json()["status"] == "healthy"
    assert client.get("/api/v1/status").json()["index"]["total"] == 3


def test_query_flow(client):
    resp = client.post(
        "/api/v1/query",
        json={"query": "What is a B-tree index?", "db_filter": "postgresql"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is False
    assert len(body["sources"]) >= 1
    assert body["detected_db"] == "postgresql"
    assert body["top_chunk_score"] == 0.62
    assert body["citations"][0]["doc_name"] == "pg_indexes.txt"
    assert set(body["timings"]) == {"retrieval_ms", "rerank_ms", "generation_ms"}
    assert body["latency_ms"] >= 0


def test_off_topic_refused_without_llm_call(client, monkeypatch):
    """Weak dense similarity refuses even though min-max scores the top chunk 1.0."""
    weak = [dict(c, semantic_score=0.08) for c in _fake_hybrid_search("x")]
    monkeypatch.setattr("backend.rag_chain.hybrid_search", lambda *a, **k: weak)
    monkeypatch.setattr("backend.rag_chain._invoke_llm", lambda m: pytest.fail("LLM called"))
    body = client.post("/api/v1/query", json={"query": "best pizza in Naples?"}).json()
    assert body["refused"] is True
    assert body["citations"] == []


def test_legacy_query_alias(client):
    resp = client.post("/query", json={"query": "What is VACUUM?"})
    assert resp.status_code == 200
    assert resp.json()["session_id"]


def test_command_guardrail_refusal(client):
    resp = client.post(
        "/api/v1/query",
        json={"query": "DROP TABLE users"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is True
    assert "Destructive" in body["answer"]
    assert body["guardrail_blocked"] is True


def test_query_validation(client):
    assert client.post("/api/v1/query", json={"query": ""}).status_code == 422
    assert client.post(
        "/api/v1/query", json={"query": "x", "db_filter": "oracle"}
    ).status_code == 422


def test_metrics_persisted(client):
    """Two queries -> metrics reflect them (SQLite, not in-memory)."""
    client.post("/api/v1/query", json={"query": "B-tree index?"})
    client.post("/api/v1/query", json={"query": "DROP TABLE x"})

    metrics = client.get("/metrics").json()
    assert metrics["total_requests"] == 2


def test_session_and_messages(client):
    created = client.post("/api/v1/sessions", json={}).json()
    session_id = created["session_id"]
    assert session_id

    client.post(
        "/api/v1/query",
        json={"query": "What is vacuum?", "session_id": session_id},
    )
    msgs = client.get(f"/api/v1/sessions/{session_id}/messages").json()["messages"]
    assert any(m["role"] == "user" for m in msgs)
    assert any(m["role"] == "assistant" for m in msgs)


def test_propose_action(client):
    created = client.post("/api/v1/sessions", json={}).json()
    session_id = created["session_id"]
    resp = client.post(
        "/api/v1/actions",
        json={
            "session_id": session_id,
            "query": "index bloat",
            "action": "Run VACUUM ANALYZE",
            "risk_level": "low",
        },
    )
    assert resp.status_code == 201
    assert resp.json()["status"] == "pending"

    actions = client.get("/api/v1/actions", params={"session_id": session_id}).json()
    assert len(actions["actions"]) == 1


def test_admin_key_guards_operator_endpoints(client, monkeypatch):
    """Public Q&A stays open; state-changing and listing endpoints need the key."""
    monkeypatch.setattr("backend.config.API_KEY", "secret-demo")
    assert client.post("/api/v1/query", json={"query": "What is VACUUM?"}).status_code == 200
    assert client.get("/api/v1/sessions").status_code == 401
    assert client.post("/api/v1/proposals/x/apply").status_code == 401
    assert client.post("/api/v1/diagnose", json={"symptom": "db is slow"}).status_code == 401
    assert client.get("/api/v1/sessions", headers={"X-API-Key": "secret-demo"}).status_code == 200
    bearer = client.get("/api/v1/actions", headers={"Authorization": "Bearer secret-demo"})
    assert bearer.status_code == 200


def test_rate_limit(client, monkeypatch):
    from backend import main as main_mod

    monkeypatch.setattr(main_mod, "_limiter", main_mod.RateLimiter(2))
    codes = [
        client.post("/api/v1/query", json={"query": "What is VACUUM?"}).status_code
        for _ in range(3)
    ]
    assert codes == [200, 200, 429]
