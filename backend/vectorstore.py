"""Vector store access with a pluggable embedding provider.

Provider abstraction (``EMBEDDING_PROVIDER`` env):
  - ``local``   (default) all-MiniLM-L6-v2 — bundled ONNX engine shipped
                with ChromaDB. Fully offline, no API key, no torch install.
  - ``jina``    jina-embeddings-v4 via the Jina API (requires a funded
                JINA_API_KEY).
  - ``gemini``  Google Generative AI embeddings (requires GOOGLE_API_KEY).

Switching providers changes the vector dimensionality, so the Chroma
collection must be reindexed — run ``python scripts/reindex.py``.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from dotenv import load_dotenv

from backend import config

LOG = logging.getLogger("dbrain.vectorstore")

PROJECT_ROOT = config.PROJECT_ROOT
CHROMA_PERSIST_DIR = config.CHROMA_PERSIST_DIR
COLLECTION_NAME = config.COLLECTION_NAME
VALID_DB_FILTERS = config.VALID_DB_FILTERS

_EMBED_MODEL: "EmbeddingProvider | None" = None


class EmbeddingProvider(Protocol):
    """Minimal embedding interface shared by every provider."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    @property
    def model_name(self) -> str: ...


class LocalMiniLMEmbeddings:
    """all-MiniML-L6-v2 via ChromaDB's bundled ONNX runtime (offline)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._fn = DefaultEmbeddingFunction()
        LOG.info("Local embeddings initialised: %s (ONNX, offline)", model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._fn(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._fn([text])[0]

    @property
    def model_name(self) -> str:
        return self._model_name


def get_embed_model() -> EmbeddingProvider:
    """Return the configured embedding provider (cached)."""
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL

    provider = config.EMBEDDING_PROVIDER
    model = config.resolved_embedding_model()

    if provider == "jina":
        from langchain_community.embeddings import JinaEmbeddings

        api_key = config.JINA_API_KEY
        if not api_key:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=jina but JINA_API_KEY is not set in .env"
            )
        _EMBED_MODEL = JinaEmbeddings(
            jina_api_key=api_key, model_name=model, session=None
        )
        LOG.info("Jina embeddings initialised: %s", model)
    elif provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        api_key = config.GOOGLE_API_KEY
        if not api_key:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=gemini but GOOGLE_API_KEY is not set in .env"
            )
        _EMBED_MODEL = GoogleGenerativeAIEmbeddings(
            model=model, google_api_key=api_key
        )
        LOG.info("Gemini embeddings initialised: %s", model)
    else:
        _EMBED_MODEL = LocalMiniLMEmbeddings(model_name=model)

    return _EMBED_MODEL


def get_embedding_model() -> str:
    """Human-readable label of the active embedding model."""
    provider = config.EMBEDDING_PROVIDER
    model = config.resolved_embedding_model()
    if provider == "local":
        return f"local/{model}"
    return f"{provider}/{model}"


def _embed_batch(texts: list[str]) -> list[list[float]]:
    return get_embed_model().embed_documents(texts)


def _embed_single(text: str) -> list[float]:
    return get_embed_model().embed_query(text)


def get_chroma_client() -> chromadb.PersistentClient:
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))


def get_or_create_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks, force: bool = False) -> chromadb.Collection:
    """Index chunks with the active provider. ``force=True`` rebuilds from
    scratch (required after switching embedding models)."""
    chroma = get_chroma_client()

    if force:
        try:
            chroma.delete_collection(COLLECTION_NAME)
            LOG.info("Deleted collection %s for full reindex", COLLECTION_NAME)
        except Exception:  # noqa: BLE001 — collection may not exist
            pass

    collection = get_or_create_collection(chroma)

    existing_ids = set(collection.get(include=[])["ids"]) if collection.count() > 0 else set()
    pending = [
        (i, c) for i, c in enumerate(chunks)
        if f"chunk_{c.metadata['chunk_id']}" not in existing_ids
    ]

    if not pending:
        LOG.info("All %d chunks already indexed.", len(chunks))
        return collection

    LOG.info("%d already indexed, %d remaining. Embedding with %s...",
             len(existing_ids), len(pending), get_embedding_model())

    BATCH_SIZE = 32
    total = len(pending)
    for i in range(0, total, BATCH_SIZE):
        batch_items = pending[i:i + BATCH_SIZE]
        batch_texts = [c.page_content for _, c in batch_items]
        embeddings = _embed_batch(batch_texts)
        collection.add(
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=[c.metadata for _, c in batch_items],
            ids=[f"chunk_{c.metadata['chunk_id']}" for _, c in batch_items],
        )
        done = min(i + BATCH_SIZE, total)
        LOG.info("  [%d/%d] chunks embedded", done, total)

    LOG.info("Done. %d total chunks in ChromaDB.", collection.count())
    return collection


def semantic_search(
    query: str, top_k: int = 10, db_filter: str | None = None
) -> list[dict]:
    chroma = get_chroma_client()
    collection = get_or_create_collection(chroma)
    query_embedding = _embed_single(query)

    where_clause = None
    if db_filter and db_filter in VALID_DB_FILTERS:
        where_clause = {"db": {"$eq": db_filter}}

    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    if where_clause:
        kwargs["where"] = where_clause

    results = collection.query(**kwargs)

    chunks = []
    if results and results.get("documents") and len(results["documents"]) > 0:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # cosine similarity in [-1, 1]; clamp to [0,1] for the relevance gate
            cosine = max(0.0, min(1.0, 1 - dist))
            chunks.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "db": meta.get("db", "unknown"),
                "db_label": meta.get("db_label", "Unknown"),
                "doc_name": meta.get("doc_name", ""),
                "source_url": meta.get("source_url", ""),
                "chunk_id": meta.get("chunk_id", -1),
                "score": round(cosine, 4),
            })
    return chunks


def get_index_stats() -> dict:
    chroma = get_chroma_client()
    collection = get_or_create_collection(chroma)
    total = collection.count()
    stats = {"total": total}
    for db in VALID_DB_FILTERS:
        result = collection.get(where={"db": {"$eq": db}}, include=[])
        stats[db] = len(result["ids"]) if result else 0
    return stats


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    from backend.ingest import load_documents, chunk_documents

    docs = load_documents(str(config.DOCS_DIR))
    chunks = chunk_documents(docs)
    index_chunks(chunks, force=True)

    print("\n--- Test Searches ---")
    tests = [
        ("how to create a B-tree index", None),
        ("EXPLAIN plan for slow query", "postgresql"),
        ("InnoDB transaction isolation level", "mysql"),
        ("aggregation pipeline performance", "mongodb"),
    ]
    for query, db_filter in tests:
        label = db_filter or "all DBs"
        results = semantic_search(query, top_k=2, db_filter=db_filter)
        print(f"\nQuery: '{query}' [{label}]")
        for r in results:
            print(f"  [{r['db']:<12}] score={r['score']:.3f} | {r['doc_name']} | {r['text'][:80]}...")