import os
import pickle
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
# pyrefly: ignore [missing-import]
from vectorstore import semantic_search, get_chroma_client, get_or_create_collection, VALID_DB_FILTERS

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BM25_INDEX_PATH = os.path.join(PROJECT_ROOT, "bm25_index.pkl")


def build_bm25_index() -> dict:
    """
    Build a per-database BM25 index from all ChromaDB chunks.
    Saves to disk. Run once after indexing.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    all_docs = collection.get(include=["documents", "metadatas"])
    texts = all_docs["documents"]
    metadatas = all_docs["metadatas"]
    ids = all_docs["ids"]

    if not texts:
        raise RuntimeError(
            "ChromaDB collection is empty. Run 'python backend/vectorstore.py' from the project root first."
        )

    # Build one global BM25 index and per-db indexes
    tokenized_all = [t.lower().split() for t in texts]
    bm25_all = BM25Okapi(tokenized_all)

    # Per-database indexes for filtered search
    bm25_per_db = {}
    for db in VALID_DB_FILTERS:
        db_texts = [texts[i] for i, m in enumerate(metadatas) if m.get("db") == db]
        db_meta  = [metadatas[i] for i, m in enumerate(metadatas) if m.get("db") == db]
        db_ids   = [ids[i] for i, m in enumerate(metadatas) if m.get("db") == db]
        if db_texts:
            bm25_per_db[db] = {
                "bm25": BM25Okapi([t.lower().split() for t in db_texts]),
                "texts": db_texts,
                "metadatas": db_meta,
                "ids": db_ids,
            }
            print(f"  BM25 [{db}]: {len(db_texts)} docs indexed")

    index_data = {
        "bm25_all": bm25_all,
        "bm25_per_db": bm25_per_db,
        "texts": texts,
        "metadatas": metadatas,
        "ids": ids,
    }

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(index_data, f)

    print(f"\nBM25 index saved. Total: {len(texts)} chunks across {len(bm25_per_db)} databases.")
    return index_data


def load_bm25_index() -> dict:
    if not Path(BM25_INDEX_PATH).exists():
        print("BM25 index not found — building now...")
        return build_bm25_index()
    with open(BM25_INDEX_PATH, "rb") as f:
        return pickle.load(f)


def bm25_search(query: str, top_k: int = 10, db_filter: str | None = None) -> list[dict]:
    """BM25 keyword search, optionally filtered to one database."""
    index_data = load_bm25_index()
    tokens = query.lower().split()

    if db_filter and db_filter in index_data["bm25_per_db"]:
        db_data = index_data["bm25_per_db"][db_filter]
        scores = db_data["bm25"].get_scores(tokens)
        texts = db_data["texts"]
        metadatas = db_data["metadatas"]
    else:
        scores = index_data["bm25_all"].get_scores(tokens)
        texts = index_data["texts"]
        metadatas = index_data["metadatas"]

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "text": texts[idx],
                "source": metadatas[idx].get("source", "unknown"),
                "db": metadatas[idx].get("db", "unknown"),
                "db_label": metadatas[idx].get("db_label", "Unknown"),
                "doc_name": metadatas[idx].get("doc_name", ""),
                "source_url": metadatas[idx].get("source_url", ""),
                "chunk_id": metadatas[idx].get("chunk_id", int(idx)),
                "score": float(scores[idx]),
                "retrieval_method": "bm25",
            })

    return results


def reciprocal_rank_fusion(
    semantic: list[dict],
    bm25: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    RRF: score = sum(1 / (k + rank)) across both ranked lists.
    k=60 is the standard constant from the original RRF paper.
    """
    scores: dict[int, float] = {}
    data: dict[int, dict] = {}

    for rank, r in enumerate(semantic):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        data[cid] = {**r, "retrieval_method": "semantic"}

    for rank, r in enumerate(bm25):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        if cid in data:
            data[cid]["retrieval_method"] = "hybrid"
        else:
            data[cid] = {**r, "retrieval_method": "bm25"}

    sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    for cid, fused_score in sorted_chunks:
        chunk = data[cid].copy()
        chunk["fused_score"] = round(fused_score, 6)
        result.append(chunk)

    return result


def hybrid_search(
    query: str,
    top_k: int = 10,
    db_filter: str | None = None,
) -> list[dict]:
    """
    Main hybrid retrieval: BM25 + semantic → RRF fusion.
    db_filter applied to both retrieval methods independently.
    """
    sem = semantic_search(query, top_k=top_k, db_filter=db_filter)
    bm  = bm25_search(query, top_k=top_k, db_filter=db_filter)
    return reciprocal_rank_fusion(sem, bm)[:top_k]


if __name__ == "__main__":
    build_bm25_index()

    query = "VACUUM ANALYZE performance bloat"
    print(f"\nTest query: '{query}'\n")

    sem_results = semantic_search(query, top_k=3)
    bm25_results = bm25_search(query, top_k=3)
    hybrid_results = hybrid_search(query, top_k=3)

    print("Semantic:")
    for r in sem_results:
        print(f"  [{r['db']:<12}] score={r['score']:.3f} | {r['text'][:70]}...")

    print("\nBM25:")
    for r in bm25_results:
        print(f"  [{r['db']:<12}] score={r['score']:.2f}  | {r['text'][:70]}...")

    print("\nHybrid (RRF):")
    for r in hybrid_results:
        print(f"  [{r['db']:<12}] fused={r['fused_score']:.4f} method={r['retrieval_method']:<8} | {r['text'][:70]}...")
