from sentence_transformers import CrossEncoder
import time

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model = None


def get_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        print(f"Loading cross-encoder: {MODEL_NAME} (first load ~30s)")
        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Score (query, chunk) pairs with cross-encoder.
    Unlike bi-encoder similarity, the cross-encoder sees both together —
    dramatically better at judging relevance, especially for technical queries.

    Performance on CPU: ~50-200ms for 10 chunks. Acceptable for a web app.
    """
    if not chunks:
        return chunks

    model = get_reranker()
    t0 = time.perf_counter()

    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)

    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = round(float(scores[i]), 4)

    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    print(f"Reranker: {len(chunks)} chunks -> top {top_k} in {elapsed}ms")

    return reranked[:top_k]


if __name__ == "__main__":
    from hybrid_retrieval import hybrid_search

    query = "How do I create a partial index in PostgreSQL?"
    candidates = hybrid_search(query, top_k=10)

    print("Before reranking (top 5 by RRF):")
    for c in candidates[:5]:
        print(f"  [{c['db']:<12}] fused={c.get('fused_score', 0):.4f} | {c['text'][:70]}...")

    reranked = rerank(query, candidates, top_k=5)
    print("\nAfter reranking (top 5 by cross-encoder):")
    for c in reranked:
        print(f"  [{c['db']:<12}] rerank={c['rerank_score']:.3f} | {c['text'][:70]}...")
