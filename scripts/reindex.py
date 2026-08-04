"""Rebuild the entire retrieval index with the active embedding provider.

- Re-embeds the corpus (data/db_docs) into the Chroma collection (force=True,
  so a provider/model switch that changes vector dimensionality cannot poison
  the store).
- Rebuilds the per-database BM25 index.

Usage:
    python scripts/reindex.py
    DOCS_DIR=<alt> CHROMA_PERSIST_DIR=<alt> python scripts/reindex.py
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend import config
from backend.hybrid_retrieval import build_bm25_index
from backend.ingest import chunk_documents, load_documents
from backend.vectorstore import get_embedding_model, index_chunks


def main() -> int:
    docs_dir = config.DOCS_DIR
    print(f"Corpus: {docs_dir}")
    docs = load_documents(str(docs_dir))
    if not docs:
        print("ERROR: no documents found. Run scripts/download_corpus.py first.")
        return 1

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks.")
    print(f"Embedding model: {get_embedding_model()}")
    print("Re-embedding into Chroma (force rebuild)...")
    index_chunks(chunks, force=True)

    print("Rebuilding BM25 index...")
    build_bm25_index()
    print("Reindex complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())