"""Cross-encoder reranker (optional, lazy-loaded).

FE-7.4/feature #6: cross-encoder re-scoring of the hybrid candidates. The
module never hard-fails the pipeline — if sentence-transformers is missing the
rerank stage degrades gracefully to RRF ordering (``RERANKER_ENABLED=false``
or a missing package both bypass the stage).
"""

from __future__ import annotations

import logging
import os
import time

from backend import config

LOG = logging.getLogger("dbrain.reranker")

try:
    from sentence_transformers import CrossEncoder

    HAS_RERANKER = True
except ImportError:  # pragma: no cover - env dependent
    CrossEncoder = None
    HAS_RERANKER = False

MODEL_NAME = config.RERANKER_MODEL
_model = None


def is_available() -> bool:
    return HAS_RERANKER and config.RERANKER_ENABLED


def get_reranker():
    """Return the cross-encoder model or None when unavailable."""
    global _model
    if not is_available():
        return None
    if _model is None:
        LOG.info("Loading cross-encoder: %s (first load ~30s)", MODEL_NAME)
        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Score (query, chunk) pairs with the cross-encoder.

    Falls back to plain RRF order when the model is unavailable.
    """
    if not chunks:
        return chunks

    model = get_reranker()
    if model is None:
        LOG.info(
            "Reranker unavailable (package %s, enabled=%s). Using RRF order.",
            "installed" if HAS_RERANKER else "missing",
            config.RERANKER_ENABLED,
        )
        for chunk in chunks:
            chunk["rerank_score"] = chunk.get(
                "rerank_score", chunk.get("normalized_score", chunk.get("fused_score", 0.0))
            )
        return chunks[:top_k]

    t0 = time.perf_counter()
    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)

    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = round(float(scores[i]), 4)

    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    LOG.info("Reranker: %d chunks -> top %d in %sms", len(chunks), top_k, elapsed)

    return reranked[:top_k]