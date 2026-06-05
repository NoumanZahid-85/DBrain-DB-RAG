# Production-Grade RAG System — Complete Build Roadmap
**Stack:** Python · FastAPI · LangChain · ChromaDB · Google Gemini · React · Langfuse · RAGAS · GitHub Actions  
**Corpus:** PostgreSQL Official Documentation  
**Deploy:** Render (backend) · Vercel (frontend)  
**Target:** AIML Internship Portfolio — fully deployable, metrics-driven, CI-gated

---

## Table of Contents

1. [Project Overview & Architecture](#1-project-overview--architecture)
2. [Environment Setup](#2-environment-setup)
3. [Week 1 — Core RAG MVP](#3-week-1--core-rag-mvp)
   - [Phase 1.1 — Document Ingestion Pipeline](#phase-11--document-ingestion-pipeline)
   - [Phase 1.2 — Embedding + Vector Store](#phase-12--embedding--vector-store)
   - [Phase 1.3 — Retrieval + Generation with Citations](#phase-13--retrieval--generation-with-citations)
   - [Phase 1.4 — FastAPI Backend](#phase-14--fastapi-backend)
   - [Phase 1.5 — React Frontend](#phase-15--react-frontend)
   - [Phase 1.6 — Deploy MVP](#phase-16--deploy-mvp)
4. [Week 2 — Production Layer](#4-week-2--production-layer)
   - [Phase 2.1 — Hybrid Retrieval (BM25 + Semantic)](#phase-21--hybrid-retrieval-bm25--semantic)
   - [Phase 2.2 — Cross-Encoder Reranker](#phase-22--cross-encoder-reranker)
   - [Phase 2.3 — Langfuse Observability Tracing](#phase-23--langfuse-observability-tracing)
   - [Phase 2.4 — Latency & Cost Metrics Dashboard](#phase-24--latency--cost-metrics-dashboard)
   - [Phase 2.5 — Golden Eval Dataset + RAGAS Scoring](#phase-25--golden-eval-dataset--ragas-scoring)
   - [Phase 2.6 — GitHub Actions CI Eval Gate](#phase-26--github-actions-ci-eval-gate)
5. [Folder Structure](#5-folder-structure)
6. [Environment Variables Reference](#6-environment-variables-reference)
7. [Interview Talking Points](#7-interview-talking-points)
8. [FYP Overlap Map](#8-fyp-overlap-map)
9. [Common Bugs & Fixes](#9-common-bugs--fixes)

---

## 1. Project Overview & Architecture

### What you're building

A domain-specific Q&A system over PostgreSQL documentation that:
- Retrieves the most relevant doc chunks for any query
- Generates answers grounded in those chunks with explicit citations
- Refuses to answer when retrieved evidence doesn't support it (no hallucination)
- Tracks every request end-to-end with latency, cost, and quality metrics
- Runs automated quality regression on every git push

### Why this is not a tutorial chatbot

| Tutorial Chatbot | Your System |
|---|---|
| Single vector search | Hybrid BM25 + semantic search |
| No reranking | Cross-encoder reranker |
| No observability | Full Langfuse tracing per request |
| No evals | RAGAS faithfulness + relevance scoring |
| No CI | GitHub Actions eval gate, build fails on quality drop |
| No citations | Every answer pinpoints source paragraph |

### Architecture Diagram

```
User Query
    │
    ▼
React Frontend (Vercel)
    │  HTTP POST /query
    ▼
FastAPI Backend (Render)
    │
    ├──► BM25 Keyword Search ──────────────┐
    │                                       │
    ├──► ChromaDB Semantic Search ──────────┤
    │                                       │
    │                              Reciprocal Rank Fusion
    │                                       │
    │                              Cross-Encoder Reranker
    │                                       │
    │                              Top-K Chunks + Sources
    │                                       │
    └──► Gemini Flash API ◄─────────────────┘
              │
              ▼
         Answer + Citations
              │
    ├──► Langfuse (trace log)
    │
    ▼
React Frontend renders answer with source cards
```

---

## 2. Environment Setup

### 2.1 Prerequisites

```bash
# Verify versions
python --version     # Need 3.10+
node --version       # Need 18+
git --version
```

### 2.2 Create project structure

```bash
mkdir pg-rag-system && cd pg-rag-system
git init
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2.3 Install backend dependencies

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
  httpx==0.27.0

pip freeze > requirements.txt
```

### 2.4 Get API keys (all free tier)

| Service | Where to get | What it's for |
|---|---|---|
| Google Gemini | [aistudio.google.com](https://aistudio.google.com) → Get API key | LLM generation |
| Langfuse | [cloud.langfuse.com](https://cloud.langfuse.com) → Sign up free | Observability tracing |

### 2.5 Create `.env` file

```env
GOOGLE_API_KEY=your_gemini_api_key_here
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
CHROMA_PERSIST_DIR=./chroma_db
DOCS_DIR=./data/postgres_docs
```

```bash
echo ".env" >> .gitignore
echo "chroma_db/" >> .gitignore
echo "data/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "venv/" >> .gitignore
```

### 2.6 Download PostgreSQL docs corpus

```bash
mkdir -p data/postgres_docs

# Option A: Download the official PostgreSQL 16 HTML docs
# Go to: https://www.postgresql.org/docs/16/postgres-A4.pdf
# Save to: data/postgres_docs/postgres16.pdf

# Option B (better for chunking): Download specific HTML pages as text
pip install requests beautifulsoup4

python - <<'EOF'
import requests
from bs4 import BeautifulSoup
import os

# Key PostgreSQL doc pages to download
pages = {
    "queries": "https://www.postgresql.org/docs/16/queries.html",
    "indexes": "https://www.postgresql.org/docs/16/indexes.html",
    "performance": "https://www.postgresql.org/docs/16/performance-tips.html",
    "sql_commands": "https://www.postgresql.org/docs/16/sql-commands.html",
    "functions": "https://www.postgresql.org/docs/16/functions.html",
    "transactions": "https://www.postgresql.org/docs/16/mvcc.html",
    "replication": "https://www.postgresql.org/docs/16/high-availability.html",
    "vacuum": "https://www.postgresql.org/docs/16/routine-vacuuming.html",
}

os.makedirs("data/postgres_docs", exist_ok=True)

for name, url in pages.items():
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    # Extract just the main content div
    content = soup.find("div", {"class": "chapter"}) or soup.find("div", {"id": "content"})
    text = content.get_text(separator="\n") if content else soup.get_text(separator="\n")
    with open(f"data/postgres_docs/{name}.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Downloaded: {name}.txt ({len(text)} chars)")

print("Done. Corpus ready.")
EOF
```

---

## 3. Week 1 — Core RAG MVP

### Phase 1.1 — Document Ingestion Pipeline

**Goal:** Load all docs, split into chunks, ready for embedding.

Create `backend/ingest.py`:

```python
import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = os.getenv("DOCS_DIR", "./data/postgres_docs")

def load_documents(docs_dir: str) -> list[Document]:
    """Load all .txt and .pdf files from the docs directory."""
    documents = []
    docs_path = Path(docs_dir)

    for file_path in docs_path.rglob("*"):
        if file_path.suffix == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
            # Tag each doc with its source filename
            for doc in docs:
                doc.metadata["source"] = file_path.name
                doc.metadata["file_type"] = "txt"
            documents.extend(docs)

        elif file_path.suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file_path.name
                doc.metadata["file_type"] = "pdf"
            documents.extend(docs)

    print(f"Loaded {len(documents)} raw documents from {docs_dir}")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into chunks.
    - chunk_size=700: sweet spot between context and precision
    - chunk_overlap=100: prevents slicing important sentences at boundaries
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],  # tries larger breaks first
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index to metadata for citation tracking
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_preview"] = chunk.page_content[:80].replace("\n", " ")

    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


if __name__ == "__main__":
    docs = load_documents(DOCS_DIR)
    chunks = chunk_documents(docs)
    print(f"\nSample chunk:")
    print(f"  Source: {chunks[0].metadata['source']}")
    print(f"  Preview: {chunks[0].metadata['chunk_preview']}")
    print(f"  Length: {len(chunks[0].page_content)} chars")
```

**Run and verify:**
```bash
python backend/ingest.py
# Expected output:
# Loaded 8 raw documents from ./data/postgres_docs
# Created ~1200-1800 chunks from 8 documents
```

---

### Phase 1.2 — Embedding + Vector Store

**Goal:** Embed all chunks and persist them in ChromaDB.

Create `backend/vectorstore.py`:

```python
import os
import chromadb
from chromadb.utils import embedding_functions
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "postgres_docs"

# Use sentence-transformers for embeddings — free, runs on CPU, good quality
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dims, fast, CPU-friendly


def get_chroma_client():
    """Get persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def get_or_create_collection(client: chromadb.Client):
    """Get or create the vector collection with embedding function."""
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for text
    )
    return collection


def index_chunks(chunks: list[Document]):
    """Embed and store all chunks in ChromaDB."""
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    # Check if already indexed
    existing = collection.count()
    if existing > 0:
        print(f"Collection already has {existing} chunks. Skipping re-indexing.")
        print("To re-index, delete the chroma_db/ directory and run again.")
        return collection

    print(f"Indexing {len(chunks)} chunks into ChromaDB...")
    print("This will take 1-3 minutes on CPU. Do it once, it persists.")

    # ChromaDB batch insert (max 5461 per batch to avoid memory issues)
    BATCH_SIZE = 500
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        collection.add(
            documents=[c.page_content for c in batch],
            metadatas=[c.metadata for c in batch],
            ids=[f"chunk_{c.metadata['chunk_id']}" for c in batch],
        )
        print(f"  Indexed batch {i//BATCH_SIZE + 1}/{(len(chunks)//BATCH_SIZE) + 1}")

    print(f"Done. {collection.count()} chunks indexed.")
    return collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search ChromaDB for semantically similar chunks.
    Returns list of {text, source, score, chunk_id}
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "chunk_id": meta.get("chunk_id", -1),
            "score": 1 - dist,  # convert distance to similarity score
        })

    return chunks


if __name__ == "__main__":
    from ingest import load_documents, chunk_documents
    import os

    docs = load_documents(os.getenv("DOCS_DIR", "./data/postgres_docs"))
    chunks = chunk_documents(docs)
    collection = index_chunks(chunks)

    # Test search
    results = semantic_search("how to create an index in postgresql", top_k=3)
    print("\nTest search results:")
    for r in results:
        print(f"  [{r['score']:.3f}] {r['source']}: {r['text'][:100]}...")
```

**Run and verify:**
```bash
cd backend
python vectorstore.py
# First run will download the embedding model (~90MB) and index all chunks
# Second run will skip re-indexing (cached in chroma_db/)
```

---

### Phase 1.3 — Retrieval + Generation with Citations

**Goal:** Build the core RAG chain — retrieves chunks, generates a grounded answer with citations, refuses if evidence is weak.

Create `backend/rag_chain.py`:

```python
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
from vectorstore import semantic_search

load_dotenv()

# Gemini Flash — free tier, fast, good for RAG
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1,  # low temp = more factual, less creative
    max_output_tokens=1024,
)

# ─────────────────────────────────────────────
# SYSTEM PROMPT — versioned here, treated as config
# Change this and commit — it IS part of your system
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a precise PostgreSQL documentation assistant.

RULES YOU MUST FOLLOW:
1. Answer ONLY using information from the provided context chunks.
2. After every factual claim, cite the source using [Source: filename.txt].
3. If the context does not contain enough information to answer the question, respond with exactly:
   "I cannot answer this question based on the available documentation. The retrieved context does not contain sufficient information."
4. Do NOT add information from your training data. Stick strictly to the context.
5. Be concise and technical. This is for developers and DBAs.

FORMAT:
- Answer in 2-5 sentences maximum for simple questions.
- For multi-step explanations, use numbered steps.
- Always end with the citations list under "Sources:" if you used any.
"""


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block."""
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Chunk {i+1} | Source: {chunk['source']} | Relevance: {chunk['score']:.2f}]\n"
            f"{chunk['text']}\n"
        )
    return "\n---\n".join(context_parts)


def generate_answer(query: str, chunks: list[dict]) -> dict:
    """
    Generate a grounded answer from retrieved chunks.
    Returns: {answer, sources, context_used, refused}
    """
    # Filter out very low relevance chunks (below 0.3 cosine similarity)
    relevant_chunks = [c for c in chunks if c["score"] > 0.3]

    if not relevant_chunks:
        return {
            "answer": "I cannot answer this question based on the available documentation. No relevant chunks found.",
            "sources": [],
            "context_used": [],
            "refused": True,
        }

    context = format_context(relevant_chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    answer = response.content

    # Check if model refused to answer
    refused = "cannot answer" in answer.lower() and "documentation" in answer.lower()

    # Extract unique sources used
    sources = list({c["source"] for c in relevant_chunks})

    return {
        "answer": answer,
        "sources": sources,
        "context_used": relevant_chunks,
        "refused": refused,
    }


def query_rag(query: str, top_k: int = 6) -> dict:
    """
    Full RAG pipeline: retrieve → generate → return result.
    This is what the API endpoint calls.
    """
    # Step 1: Retrieve
    chunks = semantic_search(query, top_k=top_k)

    # Step 2: Generate
    result = generate_answer(query, chunks)

    return {
        "query": query,
        "answer": result["answer"],
        "sources": result["sources"],
        "chunks_retrieved": len(chunks),
        "chunks_used": len(result["context_used"]),
        "refused": result["refused"],
        "top_chunk_score": chunks[0]["score"] if chunks else 0,
    }


if __name__ == "__main__":
    test_queries = [
        "What is a B-tree index in PostgreSQL?",
        "How does VACUUM work and when should I run it?",
        "What is the difference between INNER JOIN and LEFT JOIN?",
        "How do I cure a headache?",  # off-topic — should be refused
    ]

    for q in test_queries:
        print(f"\nQ: {q}")
        result = query_rag(q)
        print(f"Refused: {result['refused']}")
        print(f"Sources: {result['sources']}")
        print(f"Answer: {result['answer'][:200]}...")
        print("-" * 60)
```

**Run and verify:**
```bash
python rag_chain.py
# The off-topic query "How do I cure a headache?" should be refused
# The PostgreSQL queries should return cited answers
```

---

### Phase 1.4 — FastAPI Backend

Create `backend/main.py`:

```python
import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from rag_chain import query_rag

load_dotenv()

app = FastAPI(
    title="PostgreSQL RAG API",
    description="Production-grade RAG over PostgreSQL documentation",
    version="1.0.0",
)

# CORS — allow your Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock this down to your Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 6  # number of chunks to retrieve


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    chunks_retrieved: int
    chunks_used: int
    refused: bool
    top_chunk_score: float
    latency_ms: float


@app.get("/")
def health_check():
    return {"status": "ok", "service": "pg-rag-api", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if len(request.query) > 1000:
        raise HTTPException(status_code=400, detail="Query too long (max 1000 chars)")

    start = time.perf_counter()
    result = query_rag(request.query, top_k=request.top_k)
    latency_ms = (time.perf_counter() - start) * 1000

    return QueryResponse(**result, latency_ms=round(latency_ms, 2))


@app.get("/stats")
def get_stats():
    """Quick stats endpoint — useful for demos."""
    from vectorstore import get_chroma_client, get_or_create_collection
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    return {
        "total_chunks_indexed": collection.count(),
        "embedding_model": "all-MiniLM-L6-v2",
        "llm": "gemini-1.5-flash",
    }
```

**Run locally:**
```bash
cd backend
uvicorn main:app --reload --port 8000

# Test it:
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is a PostgreSQL index?"}'
```

Open `http://localhost:8000/docs` — this is your Swagger UI. Screenshot this for your portfolio.

---

### Phase 1.5 — React Frontend

```bash
# From project root
npx create-react-app frontend --template typescript
cd frontend
npm install axios react-markdown
```

Replace `frontend/src/App.tsx` with:

```tsx
import { useState } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

interface QueryResponse {
  query: string;
  answer: string;
  sources: string[];
  chunks_retrieved: number;
  chunks_used: number;
  refused: boolean;
  top_chunk_score: number;
  latency_ms: number;
}

function App() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleQuery = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await axios.post<QueryResponse>(`${API_URL}/query`, { query });
      setResult(res.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 800, margin: "40px auto", padding: "0 20px", fontFamily: "system-ui" }}>
      <h1 style={{ fontSize: 24, fontWeight: 700 }}>PostgreSQL Docs RAG</h1>
      <p style={{ color: "#666", marginBottom: 24 }}>
        Ask any question about PostgreSQL. Answers are grounded in official documentation.
      </p>

      <div style={{ display: "flex", gap: 8, marginBottom: 24 }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleQuery()}
          placeholder="e.g. How does VACUUM work in PostgreSQL?"
          style={{
            flex: 1, padding: "10px 14px", fontSize: 15,
            border: "1px solid #ddd", borderRadius: 6, outline: "none",
          }}
        />
        <button
          onClick={handleQuery}
          disabled={loading}
          style={{
            padding: "10px 20px", background: "#2563eb", color: "white",
            border: "none", borderRadius: 6, cursor: "pointer", fontSize: 15,
          }}
        >
          {loading ? "Searching..." : "Ask"}
        </button>
      </div>

      {error && (
        <div style={{ background: "#fef2f2", border: "1px solid #fca5a5", borderRadius: 6, padding: 12, color: "#dc2626" }}>
          {error}
        </div>
      )}

      {result && (
        <div>
          {/* Metrics bar */}
          <div style={{
            display: "flex", gap: 16, padding: "8px 12px",
            background: "#f8fafc", border: "1px solid #e2e8f0",
            borderRadius: 6, marginBottom: 16, fontSize: 13, color: "#64748b"
          }}>
            <span>⏱ {result.latency_ms}ms</span>
            <span>📄 {result.chunks_retrieved} chunks retrieved</span>
            <span>✅ {result.chunks_used} chunks used</span>
            <span>🎯 Top score: {(result.top_chunk_score * 100).toFixed(0)}%</span>
            {result.refused && <span style={{ color: "#ef4444" }}>⚠️ Refused (no evidence)</span>}
          </div>

          {/* Answer */}
          <div style={{
            padding: 20, background: result.refused ? "#fef2f2" : "#f0fdf4",
            border: `1px solid ${result.refused ? "#fca5a5" : "#86efac"}`,
            borderRadius: 8, marginBottom: 16, lineHeight: 1.7
          }}>
            <ReactMarkdown>{result.answer}</ReactMarkdown>
          </div>

          {/* Sources */}
          {result.sources.length > 0 && (
            <div>
              <strong style={{ fontSize: 13, color: "#475569" }}>Sources:</strong>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 6 }}>
                {result.sources.map((s) => (
                  <span key={s} style={{
                    padding: "3px 10px", background: "#dbeafe", color: "#1e40af",
                    borderRadius: 20, fontSize: 12, fontFamily: "monospace"
                  }}>
                    {s}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Sample queries */}
      <div style={{ marginTop: 32 }}>
        <p style={{ fontSize: 13, color: "#94a3b8", marginBottom: 8 }}>Try these:</p>
        {[
          "What is the difference between B-tree and Hash indexes?",
          "How do I use EXPLAIN ANALYZE to debug slow queries?",
          "What does VACUUM do and when should I use it?",
          "How does PostgreSQL handle transactions with MVCC?",
        ].map((q) => (
          <button
            key={q}
            onClick={() => { setQuery(q); }}
            style={{
              display: "block", textAlign: "left", width: "100%",
              padding: "8px 12px", marginBottom: 6, background: "none",
              border: "1px solid #e2e8f0", borderRadius: 6, cursor: "pointer",
              fontSize: 13, color: "#475569",
            }}
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

export default App;
```

**Run frontend:**
```bash
cd frontend
npm start
# Opens at http://localhost:3000
```

---

### Phase 1.6 — Deploy MVP

#### Deploy Backend to Render

1. Push to GitHub:
```bash
git add .
git commit -m "feat: core RAG MVP with FastAPI"
git push origin main
```

2. Create `render.yaml` in project root:
```yaml
services:
  - type: web
    name: pg-rag-api
    env: python
    buildCommand: pip install -r requirements.txt
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
        value: /opt/render/project/src/data/postgres_docs
```

> **Important:** On Render free tier, the disk is ephemeral. You have two options:
> - Option A (simple): Commit your `chroma_db/` folder to git (it's ~50-100MB, fine for demo)
> - Option B (proper): Use Render's persistent disk ($7/month) or switch ChromaDB to in-memory + re-index on startup

3. Go to [render.com](https://render.com) → New → Web Service → Connect your repo. Add env vars in the dashboard. Deploy.

#### Deploy Frontend to Vercel

```bash
cd frontend
# Create .env.production
echo "REACT_APP_API_URL=https://your-render-service.onrender.com" > .env.production

npx vercel
# Follow prompts, deploy
```

**You now have a live, publicly accessible RAG system. This is your MVP. Screenshot everything.**

---

## 4. Week 2 — Production Layer

### Phase 2.1 — Hybrid Retrieval (BM25 + Semantic)

**Why:** Vector search misses exact keyword matches. BM25 misses semantic meaning. You need both.

Create `backend/hybrid_retrieval.py`:

```python
import os
from rank_bm25 import BM25Okapi
from vectorstore import semantic_search, get_chroma_client, get_or_create_collection
import numpy as np
import pickle
from pathlib import Path

BM25_INDEX_PATH = "./bm25_index.pkl"


def build_bm25_index():
    """
    Build BM25 index from all chunks in ChromaDB.
    Run this once after indexing. Saves to disk.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    # Fetch all documents from ChromaDB
    all_docs = collection.get(include=["documents", "metadatas"])
    texts = all_docs["documents"]
    metadatas = all_docs["metadatas"]
    ids = all_docs["ids"]

    # Tokenize for BM25 (simple whitespace tokenization)
    tokenized = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized)

    # Save index + mapping
    index_data = {
        "bm25": bm25,
        "texts": texts,
        "metadatas": metadatas,
        "ids": ids,
    }
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(index_data, f)

    print(f"BM25 index built and saved. {len(texts)} documents indexed.")
    return index_data


def load_bm25_index():
    """Load BM25 index from disk."""
    if not Path(BM25_INDEX_PATH).exists():
        print("BM25 index not found. Building now...")
        return build_bm25_index()

    with open(BM25_INDEX_PATH, "rb") as f:
        return pickle.load(f)


def bm25_search(query: str, top_k: int = 10) -> list[dict]:
    """BM25 keyword search over all chunks."""
    index_data = load_bm25_index()
    bm25 = index_data["bm25"]
    texts = index_data["texts"]
    metadatas = index_data["metadatas"]

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # ignore zero-score results
            results.append({
                "text": texts[idx],
                "source": metadatas[idx].get("source", "unknown"),
                "chunk_id": metadatas[idx].get("chunk_id", idx),
                "score": float(scores[idx]),
                "retrieval_method": "bm25",
            })

    return results


def reciprocal_rank_fusion(
    semantic_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Combine semantic and BM25 results using Reciprocal Rank Fusion.
    RRF score = sum(1 / (k + rank)) for each result across both lists.
    k=60 is the standard constant that prevents high ranks from dominating.
    """
    chunk_scores: dict[int, float] = {}
    chunk_data: dict[int, dict] = {}

    # Score from semantic results
    for rank, result in enumerate(semantic_results):
        cid = result["chunk_id"]
        chunk_scores[cid] = chunk_scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_data[cid] = {**result, "retrieval_method": "semantic"}

    # Score from BM25 results
    for rank, result in enumerate(bm25_results):
        cid = result["chunk_id"]
        chunk_scores[cid] = chunk_scores.get(cid, 0) + 1 / (k + rank + 1)
        if cid not in chunk_data:
            chunk_data[cid] = {**result, "retrieval_method": "bm25"}
        else:
            chunk_data[cid]["retrieval_method"] = "hybrid"

    # Sort by fused score
    sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for cid, fused_score in sorted_chunks:
        chunk = chunk_data[cid].copy()
        chunk["fused_score"] = fused_score
        results.append(chunk)

    return results


def hybrid_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Main hybrid retrieval function.
    Combines semantic + BM25 with RRF, returns top_k results.
    """
    semantic_results = semantic_search(query, top_k=top_k)
    bm25_results = bm25_search(query, top_k=top_k)

    fused = reciprocal_rank_fusion(semantic_results, bm25_results)
    return fused[:top_k]


if __name__ == "__main__":
    # Build index first
    build_bm25_index()

    # Compare results
    query = "VACUUM ANALYZE performance tuning"
    print("\n=== Semantic only ===")
    for r in semantic_search(query, top_k=3):
        print(f"  [{r['score']:.3f}] {r['source']}: {r['text'][:80]}...")

    print("\n=== BM25 only ===")
    for r in bm25_search(query, top_k=3):
        print(f"  [{r['score']:.3f}] {r['source']}: {r['text'][:80]}...")

    print("\n=== Hybrid (RRF) ===")
    for r in hybrid_search(query, top_k=3):
        print(f"  [fused:{r['fused_score']:.4f}|method:{r['retrieval_method']}] {r['source']}: {r['text'][:80]}...")
```

**Update `rag_chain.py`** — replace the `semantic_search` import:

```python
# In rag_chain.py, change this line:
from vectorstore import semantic_search

# To this:
from hybrid_retrieval import hybrid_search as semantic_search
```

That's it. The rest of `rag_chain.py` stays the same. Now all queries use hybrid retrieval.

---

### Phase 2.2 — Cross-Encoder Reranker

**Why:** RRF gives you better candidates. The reranker scores each candidate against the query as a pair — much more accurate than vector similarity alone.

Create `backend/reranker.py`:

```python
from sentence_transformers import CrossEncoder
import time

# This model runs on CPU fine. Downloads ~300MB once, then cached.
# cross-encoder/ms-marco-MiniLM-L-6-v2 is fast + accurate
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Lazy load — don't load on import, load on first use
_model = None

def get_reranker():
    global _model
    if _model is None:
        print(f"Loading reranker model: {MODEL_NAME}")
        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank chunks using cross-encoder.
    Cross-encoder sees (query, chunk) together — far better than
    cosine similarity which encodes them independently.

    Returns top_k reranked chunks with updated scores.
    """
    if not chunks:
        return chunks

    model = get_reranker()
    start = time.perf_counter()

    # Build (query, passage) pairs
    pairs = [(query, chunk["text"]) for chunk in chunks]

    # Score all pairs
    scores = model.predict(pairs)

    # Attach scores and sort
    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])

    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    elapsed = (time.perf_counter() - start) * 1000

    print(f"Reranker: scored {len(chunks)} chunks in {elapsed:.1f}ms")

    return reranked[:top_k]


if __name__ == "__main__":
    from hybrid_retrieval import hybrid_search

    query = "How do I create a partial index in PostgreSQL?"
    candidates = hybrid_search(query, top_k=10)

    print("Before reranking:")
    for c in candidates[:5]:
        print(f"  [fused:{c.get('fused_score', 0):.4f}] {c['text'][:80]}...")

    reranked = rerank(query, candidates, top_k=5)
    print("\nAfter reranking:")
    for c in reranked:
        print(f"  [rerank:{c['rerank_score']:.3f}] {c['text'][:80]}...")
```

**Update `rag_chain.py`** to use the reranker:

```python
# In rag_chain.py, update the query_rag function:

from hybrid_retrieval import hybrid_search
from reranker import rerank

def query_rag(query: str, top_k: int = 10, rerank_top_k: int = 5) -> dict:
    """Full RAG pipeline: retrieve → rerank → generate."""

    # Step 1: Retrieve more candidates than needed
    chunks = hybrid_search(query, top_k=top_k)

    # Step 2: Rerank to get the best N
    reranked_chunks = rerank(query, chunks, top_k=rerank_top_k)

    # Step 3: Generate
    result = generate_answer(query, reranked_chunks)

    return {
        "query": query,
        "answer": result["answer"],
        "sources": result["sources"],
        "chunks_retrieved": len(chunks),
        "chunks_used": len(result["context_used"]),
        "refused": result["refused"],
        "top_chunk_score": reranked_chunks[0]["rerank_score"] if reranked_chunks else 0,
    }
```

---

### Phase 2.3 — Langfuse Observability Tracing

**Goal:** Every single request gets a trace in Langfuse — what was retrieved, what prompt was sent, what the response was, how much it cost.

Update `backend/rag_chain.py` to add tracing:

```python
import os
import time
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from dotenv import load_dotenv

load_dotenv()

langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)


@observe()  # This decorator automatically traces this function in Langfuse
def query_rag(query: str, top_k: int = 10, rerank_top_k: int = 5) -> dict:
    """Full RAG pipeline with observability tracing."""

    start = time.perf_counter()

    # Step 1: Retrieve
    chunks = hybrid_search(query, top_k=top_k)
    retrieval_ms = (time.perf_counter() - start) * 1000

    # Step 2: Rerank
    reranked = rerank(query, chunks, top_k=rerank_top_k)
    rerank_ms = (time.perf_counter() - start) * 1000 - retrieval_ms

    # Step 3: Generate
    result = generate_answer(query, reranked)
    generation_ms = (time.perf_counter() - start) * 1000 - retrieval_ms - rerank_ms

    total_ms = (time.perf_counter() - start) * 1000

    # Log metadata to current trace
    langfuse_context.update_current_observation(
        metadata={
            "chunks_retrieved": len(chunks),
            "chunks_after_rerank": len(reranked),
            "refused": result["refused"],
            "top_rerank_score": reranked[0]["rerank_score"] if reranked else 0,
            "sources": result["sources"],
            "latency_retrieval_ms": round(retrieval_ms, 1),
            "latency_rerank_ms": round(rerank_ms, 1),
            "latency_generation_ms": round(generation_ms, 1),
            "latency_total_ms": round(total_ms, 1),
        }
    )

    return {
        "query": query,
        "answer": result["answer"],
        "sources": result["sources"],
        "chunks_retrieved": len(chunks),
        "chunks_used": len(result["context_used"]),
        "refused": result["refused"],
        "top_chunk_score": reranked[0]["rerank_score"] if reranked else 0,
        "latency_ms": round(total_ms, 1),
    }
```

After running a few queries, go to [cloud.langfuse.com](https://cloud.langfuse.com) → your project → Traces. You'll see every request with full detail. **Screenshot this for your portfolio and your README.**

---

### Phase 2.4 — Latency & Cost Metrics Dashboard

Add a `/metrics` endpoint to `backend/main.py`:

```python
# Add this to main.py
import statistics

# In-memory store for this session (resets on restart — fine for demo)
request_log: list[dict] = []

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    # ... existing code ...
    result = query_rag(request.query, top_k=request.top_k)
    latency_ms = (time.perf_counter() - start) * 1000

    # Log to in-memory store
    request_log.append({
        "latency_ms": latency_ms,
        "refused": result["refused"],
        "chunks_retrieved": result["chunks_retrieved"],
        "timestamp": time.time(),
    })

    return QueryResponse(**result, latency_ms=round(latency_ms, 2))


@app.get("/metrics")
def get_metrics():
    """
    Returns P50, P95 latency and quality stats.
    This is what production AI monitoring looks like.
    """
    if not request_log:
        return {"message": "No requests logged yet"}

    latencies = [r["latency_ms"] for r in request_log]
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    refused_count = sum(1 for r in request_log if r["refused"])

    return {
        "total_requests": n,
        "latency_p50_ms": round(latencies_sorted[int(n * 0.50)], 1),
        "latency_p95_ms": round(latencies_sorted[int(n * 0.95)], 1),
        "latency_mean_ms": round(statistics.mean(latencies), 1),
        "latency_max_ms": round(max(latencies), 1),
        "refusal_rate_pct": round((refused_count / n) * 100, 1),
        "avg_chunks_retrieved": round(statistics.mean(r["chunks_retrieved"] for r in request_log), 1),
    }
```

---

### Phase 2.5 — Golden Eval Dataset + RAGAS Scoring

**Goal:** Create 50 manually verified Q&A pairs. Run RAGAS to score faithfulness and answer relevance. This is your "before and after" proof.

Create `eval/golden_dataset.json`:

```json
[
  {
    "question": "What is a B-tree index in PostgreSQL and when is it used?",
    "ground_truth": "A B-tree index is the default index type in PostgreSQL. It can handle equality and range queries on data that can be sorted. B-tree indexes are used when query conditions involve <, <=, =, >=, >, BETWEEN, IN, IS NULL, or LIKE patterns with a non-wildcard prefix."
  },
  {
    "question": "How does VACUUM prevent table bloat in PostgreSQL?",
    "ground_truth": "VACUUM reclaims storage occupied by dead tuples, which are rows that have been deleted or updated. Without VACUUM, dead tuples accumulate and bloat the table. AUTOVACUUM runs automatically, but manual VACUUM FULL reclaims all dead space and rewrites the table."
  },
  {
    "question": "What is the difference between INNER JOIN and LEFT JOIN?",
    "ground_truth": "INNER JOIN returns only rows where there is a matching row in both tables. LEFT JOIN returns all rows from the left table and matching rows from the right table; if there is no match, NULL values are returned for right table columns."
  },
  {
    "question": "How do I use EXPLAIN ANALYZE in PostgreSQL?",
    "ground_truth": "EXPLAIN ANALYZE executes the query and returns the actual execution plan with real timing and row counts. The syntax is: EXPLAIN ANALYZE SELECT ... You can add BUFFERS to see cache hit information. Unlike EXPLAIN alone, ANALYZE actually runs the query."
  },
  {
    "question": "What is MVCC in PostgreSQL?",
    "ground_truth": "Multi-Version Concurrency Control (MVCC) allows PostgreSQL to handle concurrent transactions without locking. Each transaction sees a snapshot of the database as of the start of the transaction. Old row versions are kept until no transaction needs them anymore, then VACUUM removes them."
  }
]
```

> **Your task:** Expand this to 50 Q&A pairs by reading the PostgreSQL docs yourself and writing verified answers. Quality > quantity. Bad ground truth = bad eval scores.

Create `eval/run_eval.py`:

```python
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../backend"))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from rag_chain import query_rag

GOLDEN_DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
THRESHOLD_FAITHFULNESS = 0.75  # CI fails if below this
THRESHOLD_ANSWER_RELEVANCY = 0.70


def run_evaluation():
    with open(GOLDEN_DATASET_PATH) as f:
        golden = json.load(f)

    print(f"Running evaluation on {len(golden)} questions...")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in golden:
        result = query_rag(item["question"])
        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append([c["text"] for c in result.get("context_used", [])])
        ground_truths.append(item["ground_truth"])
        print(f"  ✓ {item['question'][:50]}...")

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    print("\nRunning RAGAS metrics...")
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Faithfulness:      {scores['faithfulness']:.3f}  (threshold: {THRESHOLD_FAITHFULNESS})")
    print(f"Answer Relevancy:  {scores['answer_relevancy']:.3f}  (threshold: {THRESHOLD_ANSWER_RELEVANCY})")
    print(f"Context Precision: {scores['context_precision']:.3f}")

    # CI gate — exit with error code if below threshold
    failed = False
    if scores["faithfulness"] < THRESHOLD_FAITHFULNESS:
        print(f"\n❌ FAIL: Faithfulness {scores['faithfulness']:.3f} < {THRESHOLD_FAITHFULNESS}")
        failed = True
    if scores["answer_relevancy"] < THRESHOLD_ANSWER_RELEVANCY:
        print(f"\n❌ FAIL: Answer Relevancy {scores['answer_relevancy']:.3f} < {THRESHOLD_ANSWER_RELEVANCY}")
        failed = True

    if not failed:
        print("\n✅ PASS: All metrics above threshold")

    # Save results for tracking
    with open("eval/last_results.json", "w") as f:
        json.dump({
            "faithfulness": scores["faithfulness"],
            "answer_relevancy": scores["answer_relevancy"],
            "context_precision": scores["context_precision"],
        }, f, indent=2)

    return 1 if failed else 0


if __name__ == "__main__":
    exit_code = run_evaluation()
    sys.exit(exit_code)
```

**Run baseline eval:**
```bash
cd eval
python run_eval.py
# Record these numbers. This is your "before" baseline.
# After adding reranker, run again. This is your "after".
# The delta is your interview talking point.
```

---

### Phase 2.6 — GitHub Actions CI Eval Gate

Create `.github/workflows/eval.yml`:

```yaml
name: RAG Quality Gate

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Download PostgreSQL docs corpus
        run: |
          mkdir -p data/postgres_docs
          python scripts/download_corpus.py

      - name: Build vector index
        env:
          GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
        run: |
          cd backend
          python vectorstore.py

      - name: Build BM25 index
        run: |
          cd backend
          python hybrid_retrieval.py

      - name: Run RAGAS evaluation
        env:
          GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
          LANGFUSE_SECRET_KEY: ${{ secrets.LANGFUSE_SECRET_KEY }}
          LANGFUSE_PUBLIC_KEY: ${{ secrets.LANGFUSE_PUBLIC_KEY }}
          LANGFUSE_HOST: https://cloud.langfuse.com
        run: |
          python eval/run_eval.py
        # This step exits with code 1 if metrics below threshold
        # GitHub Actions treats non-zero exit as build failure

      - name: Upload eval results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: eval/last_results.json
```

**Add secrets to GitHub:**
- Go to your repo → Settings → Secrets → Actions
- Add: `GOOGLE_API_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`

**Now every PR triggers an eval run. If faithfulness drops below 0.75, the merge is blocked.** Screenshot this green check + blocked PR for your portfolio.

---

## 5. Folder Structure

```
pg-rag-system/
│
├── backend/
│   ├── main.py              # FastAPI app
│   ├── ingest.py            # Document loading + chunking
│   ├── vectorstore.py       # ChromaDB + semantic search
│   ├── hybrid_retrieval.py  # BM25 + RRF
│   ├── reranker.py          # Cross-encoder reranker
│   ├── rag_chain.py         # Main RAG pipeline + Langfuse
│   └── prompts/
│       └── system_prompt.txt  # Version-controlled prompt
│
├── frontend/
│   ├── src/
│   │   └── App.tsx
│   ├── package.json
│   └── .env.production
│
├── eval/
│   ├── golden_dataset.json  # 50 verified Q&A pairs
│   ├── run_eval.py          # RAGAS evaluation script
│   └── last_results.json    # Latest eval scores (committed)
│
├── data/
│   └── postgres_docs/       # Corpus (committed or downloaded in CI)
│
├── .github/
│   └── workflows/
│       └── eval.yml         # CI eval gate
│
├── chroma_db/               # Persisted vector index
├── bm25_index.pkl           # Persisted BM25 index
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
| `LANGFUSE_SECRET_KEY` | Tracing | cloud.langfuse.com |
| `LANGFUSE_PUBLIC_KEY` | Tracing | cloud.langfuse.com |
| `LANGFUSE_HOST` | Tracing | `https://cloud.langfuse.com` |
| `CHROMA_PERSIST_DIR` | Vector store | Set to `./chroma_db` locally |
| `DOCS_DIR` | Ingestion | Set to `./data/postgres_docs` |
| `REACT_APP_API_URL` | Frontend | Your Render URL |

---

## 7. Interview Talking Points

These are the exact things you say when asked "Tell me about this project."

**1. The production gap story**
> "Most RAG demos stop at basic vector search. I implemented hybrid retrieval combining BM25 and semantic search with Reciprocal Rank Fusion, then added a cross-encoder reranker. Faithfulness score went from X% to Y% — I have the eval run in my CI pipeline to prove it."

**2. The observability angle**
> "Every single query is traced in Langfuse — I can see exactly which chunks were retrieved, the reranker scores, latency breakdown per stage at P50 and P95, and how often the system refused to answer due to insufficient evidence."

**3. The CI angle (this will surprise interviewers)**
> "I have a GitHub Actions pipeline that runs RAGAS evaluation on every pull request. If faithfulness drops below 75%, the build fails and the merge is blocked. This is how production AI teams operate."

**4. The refusal mechanism**
> "The system doesn't hallucinate. If the retrieved chunks don't support an answer, it explicitly refuses. I enforce this with a score threshold on retrieved chunks and a citation enforcement instruction in the prompt."

**5. The FYP connection**
> "This directly fed into my FYP — DBrain uses a very similar RAG chat interface for natural language database queries. I reused the entire retrieval pipeline."

---

## 8. FYP Overlap Map

| What you build here | What it maps to in DBrain |
|---|---|
| FastAPI backend | DBrain's backend API (Module 10) |
| RAG chain + citations | Chat Interface with RAG (Module 7) |
| ChromaDB vector store | Same stack for DBrain's doc retrieval |
| Langfuse tracing | Explainable AI Traces (Module 5) |
| Pydantic request models | DBrain API contracts |
| React chat UI | DBrain Chat Interface (your responsibility) |
| RAGAS evaluation | Quality baseline for DBrain's LLM pipeline |
| GitHub Actions CI | DBrain's testing and deployment pipeline |

You're not doing two separate projects. You're doing Week 1-2 of DBrain right now.

---

## 9. Common Bugs & Fixes

| Bug | Likely cause | Fix |
|---|---|---|
| `ChromaDB dimension mismatch` | Re-indexed with different embedding model | Delete `chroma_db/` and re-run `vectorstore.py` |
| `Gemini 429 rate limit` | Free tier = 15 req/min | Add `time.sleep(1)` between eval calls in `run_eval.py` |
| `Cross-encoder very slow` | First load downloads model | Run `python reranker.py` once to cache the model |
| `BM25 returns zero scores` | Query tokens not in any document | Check tokenization — try stemming if needed |
| `RAGAS faithfulness is 0` | Gemini response format not matching | Make sure RAGAS is using same LLM, check `ragas` docs for Gemini setup |
| `Render deploy fails` | ChromaDB not persisted | Commit `chroma_db/` to git OR re-index on startup |
| `CORS error on frontend` | Backend not allowing Vercel origin | Update `allow_origins` in FastAPI CORS config to your Vercel URL |
| `React build fails on Vercel` | `REACT_APP_API_URL` not set | Add env var in Vercel project settings |

---

*Built for: Nouman Zahid — AIML Internship Portfolio 2026*  
*Stack: Python · FastAPI · LangChain · ChromaDB · Gemini · React · Langfuse · RAGAS · GitHub Actions*
