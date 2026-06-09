import os
import chromadb
from langchain_community.embeddings import JinaEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

_raw_chroma = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db_local")
CHROMA_PERSIST_DIR = os.path.join(PROJECT_ROOT, _raw_chroma) if not os.path.isabs(_raw_chroma) else _raw_chroma
COLLECTION_NAME = "multi_db_docs"
VALID_DB_FILTERS = {"postgresql", "mysql", "mongodb"}

# Jina Embeddings (free tier: 1M tokens)
_JINA_MODEL = None

def get_embed_model() -> JinaEmbeddings:
    global _JINA_MODEL
    if _JINA_MODEL is None:
        api_key = os.getenv("JINA_API_KEY")
        if not api_key:
            raise ValueError("JINA_API_KEY not set in environment. Get one from https://jina.ai/embeddings")
        print("Initialising Jina Embeddings (jina-embeddings-v4)...")
        _JINA_MODEL = JinaEmbeddings(
            jina_api_key=api_key,
            model_name="jina-embeddings-v4",
            session=None
        )
    return _JINA_MODEL

def get_embedding_model() -> str:
    return "jina-embeddings-v4"

def _embed_batch(texts: list[str]) -> list[list[float]]:
    model = get_embed_model()
    return model.embed_documents(texts)

def _embed_single(text: str) -> list[float]:
    model = get_embed_model()
    return model.embed_query(text)

def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

def get_or_create_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

def index_chunks(chunks: list[Document]) -> chromadb.Collection:
    """
    Index chunks using Jina embeddings. Skips already indexed chunks.
    Run this once after changing the embedding model.
    """
    chroma = get_chroma_client()
    collection = get_or_create_collection(chroma)

    existing_ids = set(collection.get(include=[])["ids"]) if collection.count() > 0 else set()
    pending = [(i, c) for i, c in enumerate(chunks) if f"chunk_{c.metadata['chunk_id']}" not in existing_ids]

    if not pending:
        print(f"All {len(chunks)} chunks already indexed.")
        return collection

    print(f"  {len(existing_ids)} already indexed, {len(pending)} remaining.")
    print("  Embedding with Jina (jina-embeddings-v4) in batches of 64...")

    BATCH_SIZE = 32
    total = len(pending)

    for i in range(0, total, BATCH_SIZE):
        batch_items = pending[i:i + BATCH_SIZE]
        batch_texts = [c.page_content for _, c in batch_items]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        embeddings = _embed_batch(batch_texts)

        collection.add(
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=[c.metadata for _, c in batch_items],
            ids=[f"chunk_{c.metadata['chunk_id']}" for _, c in batch_items],
        )

        done = min(i + BATCH_SIZE, total)
        print(f"    [{done}/{total}] batch {batch_num}/{total_batches} done")

    print(f"\nDone. {collection.count()} total chunks in ChromaDB.")
    return collection

def semantic_search(query: str, top_k: int = 10, db_filter: str | None = None) -> list[dict]:
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
            chunks.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "db": meta.get("db", "unknown"),
                "db_label": meta.get("db_label", "Unknown"),
                "doc_name": meta.get("doc_name", ""),
                "source_url": meta.get("source_url", ""),
                "chunk_id": meta.get("chunk_id", -1),
                "score": round(1 - dist, 4),
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

if __name__ == "__main__":
    from ingest import load_documents, chunk_documents

    docs_dir = os.getenv("DOCS_DIR", "./data/db_docs")
    docs = load_documents(docs_dir)
    chunks = chunk_documents(docs)
    index_chunks(chunks)

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