# Production-Grade Multi-Database RAG System — Complete Build Roadmap
**Stack:** Python · FastAPI · LangChain · ChromaDB · Google Gemini · React · Langfuse · RAGAS · GitHub Actions  
**Corpus:** PostgreSQL + MySQL + MongoDB Official Documentation  
**Deploy:** Render (backend) · Vercel (frontend)  
**FYP Alignment:** DBrain — AI-powered performance co-pilot (Module 7: Chat Interface & RAG)  
**Target:** AIML Internship Portfolio — fully deployable, metrics-driven, CI-gated

---

## Table of Contents

1. [Project Overview & Architecture](#1-project-overview--architecture)
2. [Environment Setup](#2-environment-setup)
3. [Week 1 — Core RAG MVP](#3-week-1--core-rag-mvp)
   - [Phase 1.1 — Multi-Database Corpus Download](#phase-11--multi-database-corpus-download)
   - [Phase 1.2 — Document Ingestion Pipeline](#phase-12--document-ingestion-pipeline)
   - [Phase 1.3 — Embedding + Vector Store](#phase-13--embedding--vector-store)
   - [Phase 1.4 — Retrieval + Generation with Citations](#phase-14--retrieval--generation-with-citations)
   - [Phase 1.5 — FastAPI Backend](#phase-15--fastapi-backend)
   - [Phase 1.6 — React Frontend](#phase-16--react-frontend)
   - [Phase 1.7 — Deploy MVP](#phase-17--deploy-mvp)
4. [Week 2 — Production Layer](#4-week-2--production-layer)
   - [Phase 2.1 — Hybrid Retrieval (BM25 + Semantic)](#phase-21--hybrid-retrieval-bm25--semantic)
   - [Phase 2.2 — Cross-Encoder Reranker](#phase-22--cross-encoder-reranker)
   - [Phase 2.3 — Langfuse Observability Tracing](#phase-23--langfuse-observability-tracing)
   - [Phase 2.4 — Latency & Metrics Endpoint](#phase-24--latency--metrics-endpoint)
   - [Phase 2.5 — Golden Eval Dataset + RAGAS Scoring](#phase-25--golden-eval-dataset--ragas-scoring)
   - [Phase 2.6 — GitHub Actions CI Eval Gate](#phase-26--github-actions-ci-eval-gate)
5. [Folder Structure](#5-folder-structure)
6. [Environment Variables Reference](#6-environment-variables-reference)
7. [Interview Talking Points](#7-interview-talking-points)
8. [FYP Overlap Map (DBrain)](#8-fyp-overlap-map-dbrain)
9. [Common Bugs & Fixes](#9-common-bugs--fixes)

---

## 1. Project Overview & Architecture

### What you're building

A multi-database Q&A system over **PostgreSQL, MySQL, and MongoDB** official documentation that:

- Answers questions about any of the three databases — grounded in their actual docs
- Tags every answer with which database it relates to and which doc page it came from
- Refuses to answer when retrieved evidence doesn't support a response (zero hallucination policy)
- Runs hybrid retrieval (BM25 + semantic) and a cross-encoder reranker for production-quality results
- Traces every request end-to-end with Langfuse — latency, chunks, tokens, refusal rate
- Enforces quality gates via RAGAS in GitHub Actions CI

### Why three databases and not one

Your FYP (DBrain) supports PostgreSQL, MySQL, and MongoDB in its chat interface (Module 7). The RAG corpus you build now **is** the corpus DBrain will use. You're not building a practice project — you're building FYP Module 7 three weeks early. When DBrain's RAG needs an update, you already know every line of this codebase.

### Why this beats a tutorial chatbot

| Tutorial Chatbot | Your System |
|---|---|
| One database | Three databases, routing by query |
| Basic vector search | Hybrid BM25 + semantic + RRF fusion |
| No reranking | Cross-encoder reranker |
| No source metadata | Database-tagged chunks (pg/mysql/mongo) |
| No observability | Full Langfuse tracing per request |
| No evals | RAGAS per database + aggregate score |
| No CI | GitHub Actions eval gate, build fails on regression |
| No refusal | Refuses explicitly when evidence is weak |

### Architecture

```
User Query (with optional database filter: pg / mysql / mongo / auto)
    │
    ▼
React Frontend (Vercel)
    │  POST /query  {query, db_filter}
    ▼
FastAPI Backend (Render)
    │
    ├── BM25 Search (filtered by db tag) ──────────┐
    │                                               │
    ├── ChromaDB Semantic Search (where db=X) ──────┤
    │                                               │
    │                                    Reciprocal Rank Fusion
    │                                               │
    │                                    Cross-Encoder Reranker
    │                                               │
    │                                    Top-K Chunks + Sources
    │                                               │
    └── Gemini Flash (grounded prompt) ◄────────────┘
              │
              ▼
    Answer + Citations (source: db, page, section)
              │
    ├──► Langfuse trace log
    │
    ▼
React Frontend — answer card + source chips (color coded by DB)
```

---

## 2. Environment Setup

### 2.1 Prerequisites

```bash
python --version   # need 3.10+
node --version     # need 18+
git --version
```

### 2.2 Create project

```bash
mkdir multi-db-rag && cd multi-db-rag
git init
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2.3 Install all backend dependencies

```bash
pip install \
  fastapi==0.111.0 \
  uvicorn==0.30.1 \
  langchain==0.2.6 \
  langchain-google-genai==1.0.6 \
  langchain-community==0.2.6 \
  chromadb==0.5.3 \
  sentence-transformers==3.0.1 \
  rank-bm25==0.2.2 \
  pypdf==4.2.0 \
  python-dotenv==1.0.1 \
  pydantic==2.7.4 \
  langfuse==2.36.2 \
  ragas==0.1.14 \
  httpx==0.27.0 \
  requests==2.32.3 \
  beautifulsoup4==4.12.3

pip freeze > requirements.txt
```

### 2.4 Get API keys (all free)

| Service | URL | Purpose |
|---|---|---|
| Google Gemini | aistudio.google.com → Get API key | LLM generation |
| Langfuse | cloud.langfuse.com → Sign up free | Request tracing |

### 2.5 Create `.env`

```env
GOOGLE_API_KEY=your_gemini_key_here
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
CHROMA_PERSIST_DIR=./chroma_db
DOCS_DIR=./data/db_docs
```

```bash
echo ".env"          >> .gitignore
echo "chroma_db/"    >> .gitignore
echo "data/"         >> .gitignore
echo "__pycache__/"  >> .gitignore
echo "venv/"         >> .gitignore
echo "bm25_index.pkl" >> .gitignore
```

---

## 3. Week 1 — Core RAG MVP

### Phase 1.1 — Multi-Database Corpus Download

**Goal:** Build a rich, tagged corpus from the official docs of all three databases. Each document must carry a `db` metadata tag so the retriever can filter by database.

Create `scripts/download_corpus.py`:

```python
"""
Downloads official documentation pages for PostgreSQL, MySQL, and MongoDB.
Each file is saved with a db prefix so metadata tagging is automatic.
Run once. Re-run to refresh.
"""

import os
import time
import requests
from bs4 import BeautifulSoup

DOCS_DIR = "./data/db_docs"
os.makedirs(DOCS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# CORPUS DEFINITION
# key = filename prefix  |  value = (db_tag, url)
# These pages map directly to DBrain's FYP scope:
#   - indexes, queries, performance → Module 6 (Index Intelligence)
#   - transactions, locking         → Module 3 (Anomaly Detection)
#   - replication                   → Module 1 (Metrics Collection)
#   - vacuum/optimize               → Module 6 (Bloat Management)
# ─────────────────────────────────────────────

PAGES = {
    # ── PostgreSQL ──────────────────────────────────────────────────────
    "pg_indexes":        ("postgresql", "https://www.postgresql.org/docs/16/indexes.html"),
    "pg_queries":        ("postgresql", "https://www.postgresql.org/docs/16/queries.html"),
    "pg_performance":    ("postgresql", "https://www.postgresql.org/docs/16/performance-tips.html"),
    "pg_vacuum":         ("postgresql", "https://www.postgresql.org/docs/16/routine-vacuuming.html"),
    "pg_transactions":   ("postgresql", "https://www.postgresql.org/docs/16/mvcc.html"),
    "pg_locking":        ("postgresql", "https://www.postgresql.org/docs/16/explicit-locking.html"),
    "pg_replication":    ("postgresql", "https://www.postgresql.org/docs/16/high-availability.html"),
    "pg_explain":        ("postgresql", "https://www.postgresql.org/docs/16/using-explain.html"),
    "pg_stat_stmts":     ("postgresql", "https://www.postgresql.org/docs/16/pgstatstatements.html"),
    "pg_connections":    ("postgresql", "https://www.postgresql.org/docs/16/runtime-config-connection.html"),

    # ── MySQL ────────────────────────────────────────────────────────────
    "mysql_indexes":     ("mysql", "https://dev.mysql.com/doc/refman/8.0/en/optimization-indexes.html"),
    "mysql_explain":     ("mysql", "https://dev.mysql.com/doc/refman/8.0/en/explain.html"),
    "mysql_slow_query":  ("mysql", "https://dev.mysql.com/doc/refman/8.0/en/slow-query-log.html"),
    "mysql_perf_schema": ("mysql", "https://dev.mysql.com/doc/refman/8.0/en/performance-schema.html"),
    "mysql_transactions":("mysql", "https://dev.mysql.com/doc/refman/8.0/en/innodb-transaction-model.html"),
    "mysql_locking":     ("mysql", "https://dev.mysql.com/doc/refman/8.0/en/innodb-locking.html"),
    "mysql_replication": ("mysql", "https://dev.mysql.com/doc/refman/8.0/en/replication.html"),
    "mysql_optimize":    ("mysql", "https://dev.mysql.com/doc/refman/8.0/en/optimize-table.html"),
    "mysql_partitions":  ("mysql", "https://dev.mysql.com/doc/refman/8.0/en/partitioning.html"),
    "mysql_buffer_pool": ("mysql", "https://dev.mysql.com/doc/refman/8.0/en/innodb-buffer-pool.html"),

    # ── MongoDB ──────────────────────────────────────────────────────────
    "mongo_indexes":     ("mongodb", "https://www.mongodb.com/docs/manual/indexes/"),
    "mongo_explain":     ("mongodb", "https://www.mongodb.com/docs/manual/reference/explain-results/"),
    "mongo_profiler":    ("mongodb", "https://www.mongodb.com/docs/manual/tutorial/manage-the-database-profiler/"),
    "mongo_aggregation": ("mongodb", "https://www.mongodb.com/docs/manual/aggregation/"),
    "mongo_transactions":("mongodb", "https://www.mongodb.com/docs/manual/core/transactions/"),
    "mongo_replication": ("mongodb", "https://www.mongodb.com/docs/manual/replication/"),
    "mongo_sharding":    ("mongodb", "https://www.mongodb.com/docs/manual/sharding/"),
    "mongo_schema":      ("mongodb", "https://www.mongodb.com/docs/manual/data-modeling/"),
    "mongo_currentop":   ("mongodb", "https://www.mongodb.com/docs/manual/reference/method/db.currentOp/"),
    "mongo_performance": ("mongodb", "https://www.mongodb.com/docs/manual/administration/analyzing-mongodb-performance/"),
}

HEADERS = {"User-Agent": "Mozilla/5.0 (research bot; educational use)"}


def download_page(name: str, db_tag: str, url: str) -> bool:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Try to grab the main content area (varies by doc site)
        content = (
            soup.find("div", {"class": "chapter"})
            or soup.find("div", {"id": "content"})
            or soup.find("main")
            or soup.find("article")
            or soup.body
        )
        text = content.get_text(separator="\n") if content else soup.get_text(separator="\n")

        # Clean up excessive whitespace
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        clean_text = "\n".join(lines)

        # Save with db prefix in filename for easy identification
        filepath = os.path.join(DOCS_DIR, f"{name}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            # Write metadata header at top of file — used by ingest.py
            f.write(f"DB_TAG: {db_tag}\n")
            f.write(f"SOURCE_URL: {url}\n")
            f.write(f"DOC_NAME: {name}\n")
            f.write("---\n")
            f.write(clean_text)

        char_count = len(clean_text)
        print(f"  ✓ [{db_tag:12s}] {name:<25s} {char_count:>8,} chars")
        return True

    except Exception as e:
        print(f"  ✗ [{db_tag:12s}] {name:<25s} FAILED: {e}")
        return False


if __name__ == "__main__":
    print(f"Downloading {len(PAGES)} documentation pages...\n")

    stats = {"postgresql": 0, "mysql": 0, "mongodb": 0}
    failed = []

    for name, (db_tag, url) in PAGES.items():
        success = download_page(name, db_tag, url)
        if success:
            stats[db_tag] += 1
        else:
            failed.append(name)
        time.sleep(0.5)  # be polite to doc servers

    print(f"\n{'='*50}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*50}")
    for db, count in stats.items():
        print(f"  {db:<15s}: {count} pages")
    if failed:
        print(f"\n  Failed ({len(failed)}): {', '.join(failed)}")
        print("  Re-run script to retry failed pages.")
    print(f"\nCorpus saved to: {DOCS_DIR}")
```

**Run it:**
```bash
python scripts/download_corpus.py
# Takes ~2-3 minutes. Downloads 30 pages total.
# Expected output: ~10 pages per database
```

---

### Phase 1.2 — Document Ingestion Pipeline

**Goal:** Load all docs with proper `db` metadata tagging, split into chunks preserving which database each chunk belongs to.

Create `backend/ingest.py`:

```python
import os
import re
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = os.getenv("DOCS_DIR", "./data/db_docs")

# db tag → human-readable label
DB_LABELS = {
    "postgresql": "PostgreSQL",
    "mysql": "MySQL",
    "mongodb": "MongoDB",
}


def parse_metadata_header(content: str) -> tuple[dict, str]:
    """
    Parse the metadata header written by download_corpus.py.
    Returns (metadata_dict, remaining_content).
    """
    metadata = {}
    lines = content.split("\n")
    body_start = 0

    for i, line in enumerate(lines):
        if line == "---":
            body_start = i + 1
            break
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip().lower()] = value.strip()

    body = "\n".join(lines[body_start:])
    return metadata, body


def load_documents(docs_dir: str) -> list[Document]:
    """
    Load all .txt files, parse metadata headers, return LangChain Documents.
    Each document gets: db_tag, source_url, doc_name, source (filename).
    """
    documents = []
    docs_path = Path(docs_dir)

    for file_path in sorted(docs_path.rglob("*.txt")):
        try:
            raw = file_path.read_text(encoding="utf-8")
            file_meta, body = parse_metadata_header(raw)

            db_tag = file_meta.get("db_tag", "unknown")
            doc_name = file_meta.get("doc_name", file_path.stem)
            source_url = file_meta.get("source_url", "")

            if len(body.strip()) < 100:
                print(f"  Skipping {file_path.name}: too short after parsing")
                continue

            doc = Document(
                page_content=body,
                metadata={
                    "db": db_tag,
                    "db_label": DB_LABELS.get(db_tag, db_tag),
                    "doc_name": doc_name,
                    "source": file_path.name,
                    "source_url": source_url,
                },
            )
            documents.append(doc)

        except Exception as e:
            print(f"  Error loading {file_path.name}: {e}")

    # Print summary by database
    from collections import Counter
    db_counts = Counter(d.metadata["db"] for d in documents)
    print(f"\nLoaded {len(documents)} documents:")
    for db, count in sorted(db_counts.items()):
        print(f"  {db:<15s}: {count} docs")

    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Chunk documents. Metadata (db, source, etc.) is copied to each chunk.

    Chunk sizing rationale:
    - 700 tokens: enough context for one complete concept/explanation
    - 100 overlap: prevents cutting across important sentence boundaries
    - separators: tries double-newline first (section breaks), then single, then sentence
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks = []
    chunk_id = 0

    for doc in documents:
        chunks = splitter.split_documents([doc])
        for chunk in chunks:
            chunk.metadata["chunk_id"] = chunk_id
            chunk.metadata["chunk_preview"] = chunk.page_content[:80].replace("\n", " ")
            all_chunks.append(chunk)
            chunk_id += 1

    # Print summary by database
    from collections import Counter
    db_counts = Counter(c.metadata["db"] for c in all_chunks)
    print(f"\nCreated {len(all_chunks)} total chunks:")
    for db, count in sorted(db_counts.items()):
        print(f"  {db:<15s}: {count} chunks")

    return all_chunks


if __name__ == "__main__":
    docs = load_documents(DOCS_DIR)
    chunks = chunk_documents(docs)

    print(f"\nSample chunks:")
    # Show one chunk per database
    seen_dbs = set()
    for c in chunks:
        db = c.metadata["db"]
        if db not in seen_dbs:
            print(f"\n  [{db}] {c.metadata['doc_name']}")
            print(f"  Preview: {c.metadata['chunk_preview']}")
            print(f"  Length:  {len(c.page_content)} chars")
            seen_dbs.add(db)
        if len(seen_dbs) == 3:
            break
```

**Run and verify:**
```bash
cd backend
python ingest.py
# Expected:
# Loaded 30 documents: ~10 per database
# Created ~3000-5000 total chunks: roughly split across 3 DBs
```

---

### Phase 1.3 — Embedding + Vector Store

**Goal:** Embed all chunks into ChromaDB with `db` metadata so you can filter by database at query time.

Create `backend/vectorstore.py`:

```python
import os
import chromadb
from chromadb.utils import embedding_functions
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "multi_db_docs"

# CPU-friendly embedding model — 384 dims, fast, good quality
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Valid database filter values
VALID_DB_FILTERS = {"postgresql", "mysql", "mongodb"}


def get_chroma_client() -> chromadb.Client:
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def get_or_create_collection(client: chromadb.Client) -> chromadb.Collection:
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: list[Document]) -> chromadb.Collection:
    """
    Embed and persist all chunks. Skip if already indexed.
    Chunks are tagged with 'db' metadata for per-database filtering.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    existing = collection.count()
    if existing > 0:
        print(f"Already indexed {existing} chunks. Delete chroma_db/ to re-index.")
        return collection

    print(f"Indexing {len(chunks)} chunks...")
    print("First run downloads embedding model (~90MB). Takes 2-5 min on CPU.")

    BATCH_SIZE = 500
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        collection.add(
            documents=[c.page_content for c in batch],
            metadatas=[c.metadata for c in batch],
            ids=[f"chunk_{c.metadata['chunk_id']}" for c in batch],
        )
        done = min(i + BATCH_SIZE, len(chunks))
        print(f"  {done}/{len(chunks)} chunks indexed...")

    print(f"\nDone. {collection.count()} chunks stored in ChromaDB.")
    return collection


def semantic_search(
    query: str,
    top_k: int = 10,
    db_filter: str | None = None,
) -> list[dict]:
    """
    Semantic search with optional database filter.

    db_filter: "postgresql" | "mysql" | "mongodb" | None (search all)
    Returns list of {text, source, db, doc_name, chunk_id, score, source_url}
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    # Build ChromaDB where clause if filtering by database
    where_clause = None
    if db_filter and db_filter in VALID_DB_FILTERS:
        where_clause = {"db": {"$eq": db_filter}}

    kwargs = dict(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    if where_clause:
        kwargs["where"] = where_clause

    results = collection.query(**kwargs)

    chunks = []
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
    """Return per-database chunk counts. Used by /stats endpoint."""
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    total = collection.count()

    stats = {"total": total}
    for db in VALID_DB_FILTERS:
        result = collection.get(where={"db": {"$eq": db}}, include=[])
        stats[db] = len(result["ids"])

    return stats


if __name__ == "__main__":
    from ingest import load_documents, chunk_documents
    import os

    docs = load_documents(os.getenv("DOCS_DIR", "./data/db_docs"))
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
```

**Run:**
```bash
python vectorstore.py
# First run: downloads model, indexes all chunks (~5 min)
# Second run: skips indexing, runs test searches immediately
```

---

### Phase 1.4 — Retrieval + Generation with Citations

**Goal:** Build the RAG chain that detects which database a query is about, retrieves the right chunks, and generates a cited, grounded answer.

Create `backend/rag_chain.py`:

```python
import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
from vectorstore import semantic_search, VALID_DB_FILTERS

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1,
    max_output_tokens=1024,
)

# ─────────────────────────────────────────────────────────────────
# DATABASE AUTO-DETECTION KEYWORDS
# When user doesn't specify a db, we scan the query for keywords
# to route to the right database's documentation.
# ─────────────────────────────────────────────────────────────────
DB_KEYWORDS = {
    "postgresql": [
        "postgresql", "postgres", "pg_", "psql", "vacuum", "analyze",
        "pg_stat", "pg_locks", "tablespace", "wal", "autovacuum",
        "pg_stat_statements", "sequence", "schema public",
    ],
    "mysql": [
        "mysql", "innodb", "myisam", "mysqld", "optimize table",
        "mysql workbench", "binlog", "relay log", "mysql router",
        "performance_schema", "information_schema mysql",
    ],
    "mongodb": [
        "mongodb", "mongo", "document", "collection", "bson",
        "aggregation pipeline", "replica set", "mongod", "mongos",
        "pymongo", "mongoose", "$lookup", "sharding", "atlas",
    ],
}


def detect_database(query: str) -> str | None:
    """
    Scan query for database-specific keywords.
    Returns db tag if confident, None if ambiguous.
    """
    query_lower = query.lower()
    scores = {}
    for db, keywords in DB_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[db] = score

    if not scores:
        return None  # ambiguous — search all databases

    # If one database scores significantly higher, use it
    top_db = max(scores, key=scores.get)
    if scores[top_db] >= 1:
        return top_db
    return None


# ─────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — versioned here, treated as system config
# ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a precise database documentation assistant covering PostgreSQL, MySQL, and MongoDB.

RULES YOU MUST FOLLOW:
1. Answer ONLY using information from the provided context chunks.
2. After every factual claim, cite the source using [Source: doc_name (database)].
   Example: [Source: pg_indexes.txt (PostgreSQL)]
3. If the question involves multiple databases, clearly separate answers per database.
4. If the context does not contain enough information, respond EXACTLY with:
   "I cannot answer this question based on the available documentation. The retrieved context does not contain sufficient information."
5. Do NOT add information from your training data. Strictly use the context.
6. Be concise and technical. This is for developers and DBAs.

FORMAT:
- For single-database questions: direct answer with citations.
- For cross-database comparison questions: use headers (## PostgreSQL / ## MySQL / ## MongoDB).
- Always end with a "Sources:" section listing doc names used.
"""


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into context block, grouped by database."""
    # Group by db for cleaner context
    by_db: dict[str, list] = {}
    for c in chunks:
        by_db.setdefault(c["db"], []).append(c)

    parts = []
    for db, db_chunks in by_db.items():
        parts.append(f"=== {db.upper()} DOCUMENTATION ===")
        for i, chunk in enumerate(db_chunks):
            parts.append(
                f"[Chunk {i+1} | {chunk['doc_name']} | score={chunk['score']:.2f}]\n"
                f"{chunk['text']}"
            )
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, chunks: list[dict]) -> dict:
    """Generate grounded answer from retrieved chunks."""
    relevant = [c for c in chunks if c["score"] > 0.25]

    if not relevant:
        return {
            "answer": "I cannot answer this question based on the available documentation. No relevant chunks found.",
            "sources": [],
            "context_used": [],
            "refused": True,
        }

    context = format_context(relevant)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ]

    response = llm.invoke(messages)
    answer = response.content
    refused = "cannot answer" in answer.lower() and "documentation" in answer.lower()

    # Extract unique sources with database labels
    sources = list({f"{c['doc_name']} ({c['db_label']})" for c in relevant})

    # Which databases were actually used in the answer
    dbs_used = list({c["db"] for c in relevant})

    return {
        "answer": answer,
        "sources": sources,
        "dbs_used": dbs_used,
        "context_used": relevant,
        "refused": refused,
    }


def query_rag(
    query: str,
    top_k: int = 10,
    db_filter: str | None = None,
) -> dict:
    """
    Full RAG pipeline: detect DB → retrieve → generate.

    db_filter: explicit override. If None, auto-detects from query.
    """
    # Auto-detect database if not specified
    detected_db = db_filter or detect_database(query)

    # If still None (truly ambiguous), search all — the model will handle it
    chunks = semantic_search(query, top_k=top_k, db_filter=detected_db)

    result = generate_answer(query, chunks)

    return {
        "query": query,
        "answer": result["answer"],
        "sources": result["sources"],
        "dbs_used": result["dbs_used"],
        "detected_db": detected_db,
        "chunks_retrieved": len(chunks),
        "chunks_used": len(result["context_used"]),
        "refused": result["refused"],
        "top_chunk_score": chunks[0]["score"] if chunks else 0.0,
    }


if __name__ == "__main__":
    tests = [
        ("What is a B-tree index in PostgreSQL?", None),
        ("How does InnoDB handle deadlocks?", None),
        ("What is the aggregation pipeline in MongoDB?", None),
        ("How do I compare index creation in PostgreSQL vs MySQL?", None),
        ("How do I cure a headache?", None),   # should be refused
    ]

    for query, db_filter in tests:
        print(f"\nQ: {query}")
        r = query_rag(query, db_filter=db_filter)
        print(f"  Detected DB : {r['detected_db']}")
        print(f"  DBs Used    : {r['dbs_used']}")
        print(f"  Refused     : {r['refused']}")
        print(f"  Top Score   : {r['top_chunk_score']:.3f}")
        print(f"  Sources     : {r['sources']}")
        print(f"  Answer      : {r['answer'][:200]}...")
```

**Run:**
```bash
python rag_chain.py
# The cross-database comparison query should use both pg and mysql docs
# The headache query should be refused
```

---

### Phase 1.5 — FastAPI Backend

Create `backend/main.py`:

```python
import os
import time
import statistics
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
from rag_chain import query_rag
from vectorstore import get_index_stats, VALID_DB_FILTERS

load_dotenv()

app = FastAPI(
    title="Multi-Database RAG API",
    description="Production RAG over PostgreSQL, MySQL, and MongoDB documentation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock to Vercel URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory request log for /metrics endpoint
request_log: list[dict] = []


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    db_filter: str | None = None   # "postgresql" | "mysql" | "mongodb" | null

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

    result = query_rag(
        query=req.query,
        top_k=req.top_k,
        db_filter=req.db_filter,
    )

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
    index_stats = get_index_stats()
    return {
        "index": index_stats,
        "embedding_model": "all-MiniLM-L6-v2",
        "llm": "gemini-1.5-flash",
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
```

**Test locally:**
```bash
cd backend
uvicorn main:app --reload --port 8000

# In another terminal:
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is a B-tree index?", "db_filter": "postgresql"}'

# Open: http://localhost:8000/docs  →  Swagger UI (screenshot this)
# Open: http://localhost:8000/stats →  Index stats per database
```

---

### Phase 1.6 — React Frontend

```bash
cd ..   # project root
npx create-react-app frontend --template typescript
cd frontend
npm install axios react-markdown
```

Replace `frontend/src/App.tsx`:

```tsx
import { useState } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

// Color coding per database — matches DBrain's dashboard style
const DB_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  postgresql: { bg: "#dbeafe", text: "#1e40af", border: "#3b82f6" },
  mysql:      { bg: "#dcfce7", text: "#166534", border: "#22c55e" },
  mongodb:    { bg: "#fef9c3", text: "#854d0e", border: "#eab308" },
  ambiguous:  { bg: "#f3f4f6", text: "#374151", border: "#9ca3af" },
};

type DBFilter = "postgresql" | "mysql" | "mongodb" | null;

interface QueryResponse {
  query: string;
  answer: string;
  sources: string[];
  dbs_used: string[];
  detected_db: string | null;
  chunks_retrieved: number;
  chunks_used: number;
  refused: boolean;
  top_chunk_score: number;
  latency_ms: number;
}

const SAMPLE_QUERIES: { q: string; db: DBFilter }[] = [
  { q: "What types of indexes does PostgreSQL support?", db: "postgresql" },
  { q: "How does InnoDB handle deadlocks in MySQL?", db: "mysql" },
  { q: "How does the MongoDB aggregation pipeline work?", db: "mongodb" },
  { q: "Compare index creation syntax: PostgreSQL vs MySQL vs MongoDB", db: null },
  { q: "How does VACUUM work and when should I run it?", db: "postgresql" },
  { q: "What is replication lag and how do I monitor it?", db: null },
];

export default function App() {
  const [query, setQuery] = useState("");
  const [dbFilter, setDbFilter] = useState<DBFilter>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleQuery = async (q?: string, db?: DBFilter) => {
    const activeQuery = q ?? query;
    const activeDb = db !== undefined ? db : dbFilter;
    if (!activeQuery.trim()) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await axios.post<QueryResponse>(`${API_URL}/query`, {
        query: activeQuery,
        db_filter: activeDb,
      });
      setResult(res.data);
      if (q) setQuery(q);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const detectedColor = result
    ? DB_COLORS[result.detected_db || "ambiguous"]
    : DB_COLORS.ambiguous;

  return (
    <div style={{ maxWidth: 860, margin: "40px auto", padding: "0 20px", fontFamily: "system-ui, sans-serif" }}>

      {/* Header */}
      <div style={{ marginBottom: 32 }}>
        <h1 style={{ fontSize: 26, fontWeight: 700, margin: 0 }}>
          Multi-Database RAG
        </h1>
        <p style={{ color: "#6b7280", margin: "6px 0 0" }}>
          Q&A over PostgreSQL · MySQL · MongoDB official documentation
        </p>
      </div>

      {/* DB Filter Tabs */}
      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        {([null, "postgresql", "mysql", "mongodb"] as (DBFilter)[]).map((db) => {
          const label = db ?? "All Databases";
          const colors = DB_COLORS[db || "ambiguous"];
          const active = dbFilter === db;
          return (
            <button
              key={label}
              onClick={() => setDbFilter(db)}
              style={{
                padding: "6px 14px", borderRadius: 20, fontSize: 13, cursor: "pointer",
                border: `1px solid ${active ? colors.border : "#e5e7eb"}`,
                background: active ? colors.bg : "white",
                color: active ? colors.text : "#6b7280",
                fontWeight: active ? 600 : 400,
              }}
            >
              {label}
            </button>
          );
        })}
      </div>

      {/* Query input */}
      <div style={{ display: "flex", gap: 8, marginBottom: 28 }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleQuery()}
          placeholder="Ask anything about PostgreSQL, MySQL, or MongoDB..."
          style={{
            flex: 1, padding: "10px 14px", fontSize: 15,
            border: "1px solid #d1d5db", borderRadius: 8, outline: "none",
          }}
        />
        <button
          onClick={() => handleQuery()}
          disabled={loading}
          style={{
            padding: "10px 22px", background: loading ? "#93c5fd" : "#2563eb",
            color: "white", border: "none", borderRadius: 8,
            cursor: loading ? "not-allowed" : "pointer", fontSize: 15, fontWeight: 600,
          }}
        >
          {loading ? "Searching..." : "Ask"}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div style={{ background: "#fef2f2", border: "1px solid #fca5a5", borderRadius: 8, padding: 14, color: "#dc2626", marginBottom: 20 }}>
          {error}
        </div>
      )}

      {/* Result */}
      {result && (
        <div>
          {/* Metrics bar */}
          <div style={{
            display: "flex", flexWrap: "wrap", gap: 14, padding: "10px 14px",
            background: "#f8fafc", border: "1px solid #e2e8f0",
            borderRadius: 8, marginBottom: 16, fontSize: 13, color: "#64748b",
          }}>
            <span>⏱ {result.latency_ms}ms</span>
            <span>📄 {result.chunks_retrieved} retrieved · {result.chunks_used} used</span>
            <span>🎯 Top score: {(result.top_chunk_score * 100).toFixed(0)}%</span>
            {result.detected_db && (
              <span style={{ color: detectedColor.text, fontWeight: 600 }}>
                🗃 Auto-detected: {result.detected_db}
              </span>
            )}
            {result.refused && (
              <span style={{ color: "#ef4444", fontWeight: 600 }}>⚠️ Refused — insufficient evidence</span>
            )}
          </div>

          {/* Answer */}
          <div style={{
            padding: "18px 22px",
            background: result.refused ? "#fef2f2" : "#f0fdf4",
            border: `1px solid ${result.refused ? "#fca5a5" : "#86efac"}`,
            borderRadius: 10, marginBottom: 18, lineHeight: 1.75, fontSize: 15,
          }}>
            <ReactMarkdown>{result.answer}</ReactMarkdown>
          </div>

          {/* Sources — color coded by database */}
          {result.sources.length > 0 && (
            <div>
              <span style={{ fontSize: 13, color: "#64748b", fontWeight: 600 }}>Sources: </span>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 6 }}>
                {result.sources.map((s) => {
                  const db = result.dbs_used.find((d) => s.toLowerCase().includes(d)) || "ambiguous";
                  const c = DB_COLORS[db] || DB_COLORS.ambiguous;
                  return (
                    <span key={s} style={{
                      padding: "3px 12px", borderRadius: 20, fontSize: 12,
                      background: c.bg, color: c.text,
                      border: `1px solid ${c.border}`,
                    }}>
                      {s}
                    </span>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Sample queries */}
      <div style={{ marginTop: 36 }}>
        <p style={{ fontSize: 13, color: "#94a3b8", marginBottom: 10, fontWeight: 600 }}>
          TRY THESE QUERIES
        </p>
        {SAMPLE_QUERIES.map(({ q, db }) => {
          const c = DB_COLORS[db || "ambiguous"];
          return (
            <button
              key={q}
              onClick={() => handleQuery(q, db)}
              style={{
                display: "flex", alignItems: "center", gap: 10,
                width: "100%", padding: "9px 14px", marginBottom: 6,
                background: "white", border: "1px solid #e5e7eb",
                borderRadius: 8, cursor: "pointer", textAlign: "left",
                fontSize: 14, color: "#374151",
              }}
            >
              <span style={{
                padding: "2px 8px", borderRadius: 12, fontSize: 11, fontWeight: 600,
                background: c.bg, color: c.text, whiteSpace: "nowrap",
              }}>
                {db ?? "all DBs"}
              </span>
              {q}
            </button>
          );
        })}
      </div>
    </div>
  );
}
```

**Run frontend:**
```bash
cd frontend && npm start
# Opens at http://localhost:3000
```

---

### Phase 1.7 — Deploy MVP

#### Backend → Render

Create `render.yaml` in project root:

```yaml
services:
  - type: web
    name: multi-db-rag-api
    env: python
    buildCommand: |
      pip install -r requirements.txt &&
      python scripts/download_corpus.py &&
      cd backend && python vectorstore.py && python hybrid_retrieval.py
    startCommand: cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: GOOGLE_API_KEY
        sync: false
      - key: LANGFUSE_SECRET_KEY
        sync: false
      - key: LANGFUSE_PUBLIC_KEY
        sync: false
      - key: LANGFUSE_HOST
        value: https://cloud.langfuse.com
      - key: CHROMA_PERSIST_DIR
        value: /opt/render/project/src/chroma_db
      - key: DOCS_DIR
        value: /opt/render/project/src/data/db_docs
```

```bash
git add .
git commit -m "feat: multi-db RAG MVP — PostgreSQL + MySQL + MongoDB"
git push origin main
# Connect repo on render.com → New Web Service
```

> **Render free tier note:** Disk is ephemeral. The `buildCommand` above re-downloads and re-indexes on every deploy. That's ~5 min but it works reliably. If you want instant deploys, commit the `chroma_db/` folder (~100-150MB total for 3 databases).

#### Frontend → Vercel

```bash
cd frontend
echo "REACT_APP_API_URL=https://your-render-service.onrender.com" > .env.production
npx vercel --prod
```

**You now have a live multi-database RAG system.** Screenshot the UI with the three colored source chips. That's your LinkedIn banner.

---

## 4. Week 2 — Production Layer

### Phase 2.1 — Hybrid Retrieval (BM25 + Semantic)

BM25 catches exact keyword matches (e.g., "pg_stat_statements") that vector search misses because semantically similar embeddings don't always surface exact terms.

Create `backend/hybrid_retrieval.py`:

```python
import os
import pickle
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from vectorstore import semantic_search, get_chroma_client, get_or_create_collection, VALID_DB_FILTERS

BM25_INDEX_PATH = "./bm25_index.pkl"


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
```

**Update `rag_chain.py`** — one line change:
```python
# Replace:
from vectorstore import semantic_search, VALID_DB_FILTERS
# With:
from hybrid_retrieval import hybrid_search as semantic_search
from vectorstore import VALID_DB_FILTERS
```

---

### Phase 2.2 — Cross-Encoder Reranker

Create `backend/reranker.py`:

```python
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
    print(f"Reranker: {len(chunks)} chunks → top {top_k} in {elapsed}ms")

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
```

**Update `rag_chain.py`** to use reranker:

```python
# Add import at top of rag_chain.py:
from reranker import rerank

# Update query_rag():
def query_rag(
    query: str,
    top_k: int = 10,
    rerank_top_k: int = 5,
    db_filter: str | None = None,
) -> dict:
    detected_db = db_filter or detect_database(query)

    # Step 1: Hybrid retrieval — cast a wide net
    chunks = hybrid_search(query, top_k=top_k, db_filter=detected_db)

    # Step 2: Rerank — cross-encoder picks the best
    reranked = rerank(query, chunks, top_k=rerank_top_k)

    # Step 3: Generate
    result = generate_answer(query, reranked)

    return {
        "query": query,
        "answer": result["answer"],
        "sources": result["sources"],
        "dbs_used": result["dbs_used"],
        "detected_db": detected_db,
        "chunks_retrieved": len(chunks),
        "chunks_used": len(result["context_used"]),
        "refused": result["refused"],
        "top_chunk_score": reranked[0]["rerank_score"] if reranked else 0.0,
    }
```

---

### Phase 2.3 — Langfuse Observability Tracing

Update `rag_chain.py` to wrap `query_rag` with tracing:

```python
import os
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)


@observe()
def query_rag(
    query: str,
    top_k: int = 10,
    rerank_top_k: int = 5,
    db_filter: str | None = None,
) -> dict:
    import time
    t0 = time.perf_counter()

    detected_db = db_filter or detect_database(query)
    chunks = hybrid_search(query, top_k=top_k, db_filter=detected_db)
    t_retrieval = (time.perf_counter() - t0) * 1000

    reranked = rerank(query, chunks, top_k=rerank_top_k)
    t_rerank = (time.perf_counter() - t0) * 1000 - t_retrieval

    result = generate_answer(query, reranked)
    t_generation = (time.perf_counter() - t0) * 1000 - t_retrieval - t_rerank

    langfuse_context.update_current_observation(
        metadata={
            "detected_db": detected_db,
            "chunks_retrieved": len(chunks),
            "chunks_after_rerank": len(reranked),
            "refused": result["refused"],
            "sources": result["sources"],
            "dbs_used": result["dbs_used"],
            "top_rerank_score": reranked[0]["rerank_score"] if reranked else 0,
            "latency_retrieval_ms": round(t_retrieval, 1),
            "latency_rerank_ms": round(t_rerank, 1),
            "latency_generation_ms": round(t_generation, 1),
        }
    )

    return {
        "query": query,
        "answer": result["answer"],
        "sources": result["sources"],
        "dbs_used": result["dbs_used"],
        "detected_db": detected_db,
        "chunks_retrieved": len(chunks),
        "chunks_used": len(result["context_used"]),
        "refused": result["refused"],
        "top_chunk_score": reranked[0]["rerank_score"] if reranked else 0.0,
    }
```

After 5-10 queries, go to **cloud.langfuse.com → your project → Traces**. You'll see every request broken down. Screenshot the trace detail view. It's your most impressive portfolio asset.

---

### Phase 2.4 — Latency & Metrics Endpoint

Already implemented in `main.py` Phase 1.5. Just verify it works after running some queries:

```bash
curl http://localhost:8000/metrics
# Should show p50, p95, per-database request breakdown
```

---

### Phase 2.5 — Golden Eval Dataset + RAGAS Scoring

Create `eval/golden_dataset.json` with **50 questions across all three databases** — roughly 17 per database:

```json
[
  {
    "question": "What is a B-tree index and when should I use it in PostgreSQL?",
    "ground_truth": "A B-tree index is the default index type in PostgreSQL. It handles equality and range queries on sortable data. Use it when query conditions involve <, <=, =, >=, >, BETWEEN, IN, IS NULL, or LIKE patterns with a non-wildcard prefix.",
    "db": "postgresql"
  },
  {
    "question": "How does VACUUM prevent table bloat in PostgreSQL?",
    "ground_truth": "VACUUM reclaims space from dead tuples — rows deleted or updated. Without it, dead tuples accumulate and bloat tables. AUTOVACUUM runs automatically. VACUUM FULL rewrites the table entirely, reclaiming maximum space but requiring an exclusive lock.",
    "db": "postgresql"
  },
  {
    "question": "What is pg_stat_statements and how do I use it?",
    "ground_truth": "pg_stat_statements is a PostgreSQL extension that tracks execution statistics for all SQL statements. It shows call counts, total and mean execution time, and rows returned. Enable it in postgresql.conf with shared_preload_libraries = 'pg_stat_statements'.",
    "db": "postgresql"
  },
  {
    "question": "How do I use EXPLAIN ANALYZE in PostgreSQL?",
    "ground_truth": "EXPLAIN ANALYZE executes the query and shows the actual execution plan with real row counts and timing. Add BUFFERS to see cache hit statistics. Unlike EXPLAIN alone, ANALYZE runs the query so use it on read-only queries or inside a transaction you can roll back.",
    "db": "postgresql"
  },
  {
    "question": "What is MVCC in PostgreSQL?",
    "ground_truth": "Multi-Version Concurrency Control allows concurrent transactions without blocking by keeping multiple versions of rows. Each transaction sees a snapshot of the database at transaction start. Old row versions accumulate until VACUUM removes them.",
    "db": "postgresql"
  },

  {
    "question": "How does InnoDB handle deadlocks in MySQL?",
    "ground_truth": "InnoDB automatically detects deadlocks by examining the wait-for graph. When a deadlock is found, InnoDB rolls back the transaction that holds the fewest row-level locks. The error 1213 is returned to the rolled-back transaction. Applications should catch this error and retry.",
    "db": "mysql"
  },
  {
    "question": "What is the InnoDB buffer pool and how do I tune it?",
    "ground_truth": "The InnoDB buffer pool is an in-memory cache for table and index data. Set innodb_buffer_pool_size to 70-80% of available RAM on a dedicated database server. Monitor the buffer pool hit ratio; values below 99% indicate insufficient memory.",
    "db": "mysql"
  },
  {
    "question": "How do I use the MySQL slow query log?",
    "ground_truth": "Enable the slow query log by setting slow_query_log=ON and long_query_time to your threshold in seconds. Set slow_query_log_file to the output path. Use mysqldumpslow or pt-query-digest to analyze the log. The log captures queries exceeding the time threshold.",
    "db": "mysql"
  },
  {
    "question": "What does OPTIMIZE TABLE do in MySQL?",
    "ground_truth": "OPTIMIZE TABLE reorganizes the physical storage of table data and associated index data to reduce storage space and improve I/O efficiency. For InnoDB tables it rebuilds the table to update index statistics and compact clustered index space. It is equivalent to ALTER TABLE ... FORCE.",
    "db": "mysql"
  },
  {
    "question": "How does MySQL replication work?",
    "ground_truth": "MySQL replication uses a binary log on the source and a relay log on the replica. The source writes all changes to the binary log. The replica's I/O thread reads the binary log and writes it to the relay log. The replica's SQL thread replays the relay log events.",
    "db": "mysql"
  },

  {
    "question": "How do I create an index on a MongoDB collection?",
    "ground_truth": "Use db.collection.createIndex({field: 1}) for ascending or {field: -1} for descending. MongoDB supports single field, compound, multikey (array), text, geospatial, hashed, and wildcard indexes. Use the background option for non-blocking index builds on large collections.",
    "db": "mongodb"
  },
  {
    "question": "How does the MongoDB aggregation pipeline work?",
    "ground_truth": "The aggregation pipeline processes documents through a sequence of stages. Each stage transforms the documents. Common stages include $match (filter), $group (group and aggregate), $sort, $project (reshape), $lookup (join), and $unwind (flatten arrays). Stages are passed as an array to aggregate().",
    "db": "mongodb"
  },
  {
    "question": "How do I interpret MongoDB explain output?",
    "ground_truth": "MongoDB explain() returns execution stats. Check the winningPlan for the query plan chosen. A COLLSCAN means a full collection scan — add an index. An IXSCAN shows an index was used. Check totalDocsExamined vs nReturned; a large ratio means the index is not selective enough.",
    "db": "mongodb"
  },
  {
    "question": "What is the MongoDB database profiler?",
    "ground_truth": "The database profiler captures operation data for slow queries. Set the profiling level to 0 (off), 1 (slow ops only), or 2 (all ops) using db.setProfilingLevel(). Profiled operations are stored in the system.profile capped collection. Use slowOpThresholdMs to set the slow op threshold.",
    "db": "mongodb"
  },
  {
    "question": "How does MongoDB replication work?",
    "ground_truth": "MongoDB uses replica sets for replication. A replica set contains one primary and one or more secondaries. The primary accepts all write operations and records changes to its oplog. Secondaries replicate the oplog asynchronously. If the primary fails, an election selects a new primary.",
    "db": "mongodb"
  }
]
```

> **Your task:** Expand to 50 entries — roughly 17 per database, covering the topics in DBrain's Module 7 scope (indexes, locks, replication, vacuum/optimize, schema analysis, performance). Write ground truths yourself after reading the docs. Bad ground truth → useless eval scores.

Create `eval/run_eval.py`:

```python
"""
RAGAS evaluation script.
Run before adding reranker → record baseline.
Run after adding reranker → record improved score.
The delta is your interview story.
"""

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../backend"))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from rag_chain import query_rag

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "last_results.json")

# Quality thresholds — CI fails if below these
THRESHOLDS = {
    "faithfulness": 0.75,
    "answer_relevancy": 0.70,
}


def run_evaluation(db_filter_eval: str | None = None):
    """
    Run RAGAS evaluation on golden dataset.
    db_filter_eval: restrict to one database subset for targeted testing.
    """
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    if db_filter_eval:
        golden = [g for g in golden if g.get("db") == db_filter_eval]
        print(f"Evaluating {len(golden)} questions for: {db_filter_eval}")
    else:
        print(f"Evaluating all {len(golden)} questions across all databases")

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in golden:
        result = query_rag(item["question"])
        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append([c["text"] for c in result.get("context_used", [])])
        ground_truths.append(item["ground_truth"])
        print(f"  ✓ [{item.get('db','?'):<12}] {item['question'][:55]}...")

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    print("\nRunning RAGAS scoring...")
    scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])

    print("\n" + "=" * 55)
    print(f"EVALUATION RESULTS {f'[{db_filter_eval}]' if db_filter_eval else '[all databases]'}")
    print("=" * 55)
    print(f"  Faithfulness      : {scores['faithfulness']:.3f}  (threshold ≥ {THRESHOLDS['faithfulness']})")
    print(f"  Answer Relevancy  : {scores['answer_relevancy']:.3f}  (threshold ≥ {THRESHOLDS['answer_relevancy']})")
    print(f"  Context Precision : {scores['context_precision']:.3f}")

    failed = False
    for metric, threshold in THRESHOLDS.items():
        if scores[metric] < threshold:
            print(f"\n  ❌ FAIL: {metric} = {scores[metric]:.3f} < {threshold}")
            failed = True

    if not failed:
        print(f"\n  ✅ PASS: All metrics above thresholds")

    # Save results
    output = {
        "faithfulness": scores["faithfulness"],
        "answer_relevancy": scores["answer_relevancy"],
        "context_precision": scores["context_precision"],
        "db_filter": db_filter_eval,
        "n_questions": len(golden),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {RESULTS_PATH}")

    return 1 if failed else 0


if __name__ == "__main__":
    # Run full eval by default
    # Pass db name as arg for targeted: python run_eval.py postgresql
    db = sys.argv[1] if len(sys.argv) > 1 else None
    exit_code = run_evaluation(db_filter_eval=db)
    sys.exit(exit_code)
```

**Run your baseline BEFORE adding reranker (if you haven't):**
```bash
# First commit eval/last_results.json as your "before" baseline
python eval/run_eval.py
# Record: faithfulness=X, answer_relevancy=Y

# Then add the reranker changes to rag_chain.py
# Then run again:
python eval/run_eval.py
# Record: faithfulness=X+delta, answer_relevancy=Y+delta

# Per-database breakdown:
python eval/run_eval.py postgresql
python eval/run_eval.py mysql
python eval/run_eval.py mongodb
```

---

### Phase 2.6 — GitHub Actions CI Eval Gate

Create `.github/workflows/eval.yml`:

```yaml
name: RAG Quality Gate — Multi-Database

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    timeout-minutes: 30

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Download corpus (PostgreSQL + MySQL + MongoDB)
        run: python scripts/download_corpus.py

      - name: Build vector index
        env:
          GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
        run: cd backend && python vectorstore.py

      - name: Build BM25 index
        run: cd backend && python hybrid_retrieval.py

      - name: Run full RAGAS evaluation
        env:
          GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
          LANGFUSE_SECRET_KEY: ${{ secrets.LANGFUSE_SECRET_KEY }}
          LANGFUSE_PUBLIC_KEY: ${{ secrets.LANGFUSE_PUBLIC_KEY }}
          LANGFUSE_HOST: https://cloud.langfuse.com
        run: python eval/run_eval.py
        # Non-zero exit = quality regression = PR blocked

      - name: Run per-database eval (informational)
        env:
          GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
          LANGFUSE_SECRET_KEY: ${{ secrets.LANGFUSE_SECRET_KEY }}
          LANGFUSE_PUBLIC_KEY: ${{ secrets.LANGFUSE_PUBLIC_KEY }}
          LANGFUSE_HOST: https://cloud.langfuse.com
        run: |
          python eval/run_eval.py postgresql || true
          python eval/run_eval.py mysql       || true
          python eval/run_eval.py mongodb     || true
        continue-on-error: true   # informational only, doesn't fail build

      - name: Upload eval results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: eval-results-${{ github.sha }}
          path: eval/last_results.json
```

**Add GitHub Secrets:**
Repo → Settings → Secrets → Actions → New secret:
- `GOOGLE_API_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_PUBLIC_KEY`

**Now every push triggers a full eval run with per-database breakdown.** A PR that drops faithfulness below 0.75 is blocked from merging. Screenshot this in your README.

---

## 5. Folder Structure

```
multi-db-rag/
│
├── backend/
│   ├── main.py                  # FastAPI app + /query /stats /metrics
│   ├── ingest.py                # Multi-database doc loading + chunking
│   ├── vectorstore.py           # ChromaDB + semantic search (db-filtered)
│   ├── hybrid_retrieval.py      # BM25 + RRF per database
│   ├── reranker.py              # Cross-encoder reranker
│   └── rag_chain.py             # Full pipeline + auto-detection + Langfuse
│
├── frontend/
│   ├── src/
│   │   └── App.tsx              # React chat UI with DB filter tabs
│   ├── package.json
│   └── .env.production
│
├── eval/
│   ├── golden_dataset.json      # 50 Q&A pairs (≥15 per database)
│   ├── run_eval.py              # RAGAS eval + CI gate
│   └── last_results.json        # Latest scores (committed)
│
├── scripts/
│   └── download_corpus.py       # Downloads 30 pages: pg + mysql + mongo
│
├── data/
│   └── db_docs/                 # Downloaded docs (30 .txt files)
│
├── .github/
│   └── workflows/
│       └── eval.yml             # CI quality gate
│
├── chroma_db/                   # Persisted vector index (~150MB)
├── bm25_index.pkl               # Persisted BM25 index
├── requirements.txt
├── render.yaml
├── .gitignore
└── README.md
```

---

## 6. Environment Variables Reference

| Variable | Where used | How to get |
|---|---|---|
| `GOOGLE_API_KEY` | Gemini LLM calls | aistudio.google.com |
| `LANGFUSE_SECRET_KEY` | Request tracing | cloud.langfuse.com |
| `LANGFUSE_PUBLIC_KEY` | Request tracing | cloud.langfuse.com |
| `LANGFUSE_HOST` | Tracing endpoint | `https://cloud.langfuse.com` |
| `CHROMA_PERSIST_DIR` | Vector store location | `./chroma_db` locally |
| `DOCS_DIR` | Corpus location | `./data/db_docs` locally |
| `REACT_APP_API_URL` | Frontend → backend | Your Render URL |

---

## 7. Interview Talking Points

These are the exact things you say in an interview. Have a number for every claim.

**1. The production gap story**
> "Most RAG demos do basic vector search. I built hybrid retrieval — BM25 for keyword matching, semantic search for intent, fused with Reciprocal Rank Fusion, then passed through a cross-encoder reranker. My faithfulness score went from [X%] to [Y%]. I have the RAGAS eval run in my CI pipeline to prove the delta."

**2. Multi-database routing**
> "The system covers PostgreSQL, MySQL, and MongoDB — the same three databases my FYP monitors. I built a keyword-based DB detector that routes queries to the right documentation subset automatically. A query mentioning 'InnoDB' routes to MySQL docs. 'pg_stat_statements' routes to PostgreSQL. Cross-database comparisons search all three and the model separates the answers by database with citations."

**3. The observability angle**
> "Every single request is traced in Langfuse. I can see which chunks were retrieved, what the reranker scored each one, the latency breakdown at P50 and P95 per stage, and how often the system refused to answer due to insufficient evidence. That's how production AI teams operate."

**4. The CI gate (this surprises most interviewers)**
> "I have a GitHub Actions pipeline that runs RAGAS evaluation on every pull request — full suite plus per-database breakdowns. If faithfulness drops below 75%, the build fails and the merge is blocked. No human has to remember to run evals — it's enforced."

**5. The refusal mechanism**
> "The system doesn't hallucinate. Chunks below 0.25 cosine similarity are filtered before the LLM sees them. The system prompt forces citation per claim and provides an exact refusal phrase when evidence is insufficient. I track the refusal rate in the /metrics endpoint."

**6. The FYP connection**
> "This is literally Module 7 of my FYP done three weeks early. DBrain's chat interface uses an identical RAG pipeline over the same three databases. The corpus I built here — 30 pages from PostgreSQL, MySQL, and MongoDB docs — is the same corpus DBrain will use. I didn't build a portfolio project — I built a FYP module."

---

## 8. FYP Overlap Map (DBrain)

| What you build here | DBrain module | Your responsibility |
|---|---|---|
| FastAPI `/query` endpoint | Module 10: Backend API (FE-10.1) | Direct reuse |
| ChromaDB + hybrid retrieval | Module 7: Chat & RAG (FE-7.2) | Direct reuse |
| Multi-database corpus (pg/mysql/mongo) | Module 7: RAG pipeline (FE-7.2) | Same 3 databases |
| Context-aware prompt + live metrics | Module 7: Structured prompt (FE-7.3) | Add live metrics snapshot |
| Command guardrail concept | Module 7: Guardrail (FE-7.4) | Extend to block DROP/DELETE |
| Langfuse traces | Module 5: Explainable AI Traces (FE-5.3) | Same trace format |
| React chat UI component | Module 7: Chat UI (FE-7.1) | Port App.tsx → DBrain |
| RAGAS evaluation | DBrain eval framework | Adapt golden dataset |
| GitHub Actions CI | DBrain pipeline | Reuse workflow |
| Pydantic request models | Module 10: API contracts (FE-10.1) | Same pattern |

**You are not doing two separate projects.** Every line of code here is a direct deposit into your FYP bank.

---

## 9. Common Bugs & Fixes

| Bug | Cause | Fix |
|---|---|---|
| `ChromaDB dimension mismatch` | Re-indexed with different embedding model | Delete `chroma_db/` and re-run `vectorstore.py` |
| `where clause returns 0 results` | `db_filter` value doesn't match stored metadata | Run `vectorstore.py` test — check actual metadata values stored |
| `BM25 returns all zero scores` | Query tokens not in any document after tokenization | Lower-case tokenization mismatch — check `query.lower().split()` |
| `Gemini 429 rate limit` | Free tier = 15 req/min | Add `time.sleep(4)` between RAGAS eval calls |
| `RAGAS faithfulness = 0.0` | RAGAS can't parse Gemini response format | Set `RAGAS_EXPERIMENTAL=true` and configure Gemini as RAGAS LLM explicitly |
| `Cross-encoder slow on first query` | Model loading on first call | Run `python reranker.py` once after deploy to warm the model |
| `download_corpus.py partial failure` | Rate limiting or connection timeout | Re-run — script skips already-downloaded files with `os.path.exists` check |
| `Render deploy fails — disk full` | `chroma_db/` + models exceed free tier | Commit `chroma_db/` to git OR use Render's Disk add-on ($7/mo) |
| `CORS error from Vercel frontend` | `allow_origins=["*"]` not matching | Lock `allow_origins` to `["https://your-app.vercel.app"]` in production |
| `React build fails on Vercel` | `REACT_APP_API_URL` not set | Add env var in Vercel project settings dashboard |
| `DB auto-detection wrong DB` | Query has overlapping keywords | Override with explicit `db_filter` in the UI or expand `DB_KEYWORDS` |
| `Langfuse traces not appearing` | Wrong secret/public key pair | Double-check both keys — they're easy to swap |

---

*Built by: Sahibzada Hasanat Ahmad — AIML Internship Portfolio 2026*  
*Stack: Python · FastAPI · LangChain · ChromaDB · Gemini Flash · React · Langfuse · RAGAS · GitHub Actions*  
*Corpus: PostgreSQL 16 · MySQL 8.0 · MongoDB 7.x official documentation*  
*FYP: DBrain — AI-powered multi-database performance co-pilot (COMSATS University Islamabad)*
