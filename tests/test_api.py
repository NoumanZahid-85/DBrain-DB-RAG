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

    chat_mod._session_store = None
    ms_mod._metrics_store = None
    # Reset module-level singletons used by rag_chain.stores()
    import backend.rag_chain as rc

    rc._session_store = None
    rc._metrics_store = None
    # main.py builds its stores at import time; rebuild against temp DB.
    main_mod._session_store = chat_mod.SessionStore()
    main_mod._metrics_store = ms_mod.MetricsStore()

    with TestClient(main_mod.app) as c:
        yield c


def test_root_and_health(client):
    assert client.get("/").json()["status"] == "ok"
    assert client.get("/health").json()["status"] == "healthy"


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
    assert body["top_chunk_score"] == 1.0
    assert body["latency_ms"] >= 0


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
    assert body["context_used"] if "context_used" in body else True


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


def test_api_key_enforced(monkeypatch, tmp_path):
    monkeypatch.setenv("API_KEY", "secret-demo")
    import importlib

    from backend import config as config_mod

    monkeypatch.setattr(config_mod, "API_KEY", "secret-demo")

    # Keep the reloaded main on the hermetic stack (fake LLM/retrieval).
    monkeypatch.setattr("backend.llm.get_llm", lambda: FakeLLM())
    monkeypatch.setattr("backend.rag_chain.get_llm", lambda: FakeLLM())
    monkeypatch.setattr("backend.rag_chain.hybrid_search", _fake_hybrid_search)
    monkeypatch.setattr("backend.reranker.is_available", lambda: False)
    from backend import chat as chat_mod
    from backend import metrics_store as ms_mod

    monkeypatch.setattr(config_mod, "DB_FILE", tmp_path / "keytest.db")
    monkeypatch.setattr("backend.chat.config.DB_FILE", tmp_path / "keytest.db")
    monkeypatch.setattr("backend.metrics_store.config.DB_FILE", tmp_path / "keytest.db")

    # Reload main so middlewares rebuild with the temp key value.
    from backend import main as main_mod

    main_mod = importlib.reload(main_mod)
    main_mod._session_store = chat_mod.SessionStore()
    main_mod._metrics_store = ms_mod.MetricsStore()

    with TestClient(main_mod.app) as c:
        ok = c.get("/")
        assert ok.status_code == 200  # exempt path
        denied = c.post("/api/v1/query", json={"query": "vacuum"})
        assert denied.status_code == 401
        allowed = c.post(
            "/api/v1/query",
            json={"query": "vacuum"},
            headers={"X-API-Key": "secret-demo"},
        )
        assert allowed.status_code == 200