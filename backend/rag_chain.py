import os
import re
import time
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from hybrid_retrieval import hybrid_search
from reranker import rerank
from vectorstore import VALID_DB_FILTERS
from langfuse import Langfuse, observe

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

# Initialize Langfuse client
langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

# Initialize dynamic LLM provider (Groq first, then Ollama, then Gemini)
use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:12b-it-qat")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

if os.getenv("GROQ_API_KEY") and not use_ollama:
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0,
        max_tokens=1024,
    )
    print(f"Backend LLM: Using Groq ({GROQ_MODEL})")
elif use_ollama:
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0.0,
            num_predict=1024,
        )
        print(f"Backend LLM: Using local Ollama ({OLLAMA_MODEL})")
    except ImportError:
        # langchain_ollama not installed. Use local `ollama` CLI if available
        # to avoid depending on an extra Python package. This wrapper calls
        # `ollama generate` with a concatenated prompt and exposes an
        # `invoke(messages)` method that returns an object with `content`.
        import subprocess

        class LocalOllama:
            def __init__(self, model="gemma4:12b-it-qat", temperature=0.0, max_output_tokens=1024):
                self.model = model
                self.temperature = temperature
                self.max_output_tokens = max_output_tokens

            def invoke(self, messages):
                # Concatenate messages into a single prompt understood by the model
                prompt_parts = []
                for m in messages:
                    # messages are LangChain types with `content`
                    role = getattr(m, "type", None) or m.__class__.__name__
                    prompt_parts.append(f"[{role}]\n{m.content}")
                prompt = "\n\n".join(prompt_parts)

                # Use `ollama run MODEL PROMPT` (the correct CLI entrypoint).
                # Note: very long prompts may hit OS command-length limits; if
                # that becomes a problem we can write to a temp file or stream
                # via stdin. For most RAG prompts this should work.
                cmd = ["ollama", "run", self.model, prompt]
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120)
                    text = (proc.stdout or "").strip()
                except subprocess.CalledProcessError as e:
                    stderr = (e.stderr or "").strip()
                    text = f"Ollama invocation failed (exit {e.returncode}): {stderr or str(e)}"
                except Exception as e:
                    text = f"Ollama invocation failed: {e}"

                class Resp:
                    pass

                r = Resp()
                r.content = text
                return r

        llm = LocalOllama(model=OLLAMA_MODEL, temperature=0.0, max_output_tokens=1024)
        print(f"Backend LLM: Using local `ollama` CLI ({OLLAMA_MODEL}) as fallback")
else:
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model=GOOGLE_MODEL,
        temperature=0.0,
        max_output_tokens=1024,
    )
    print(f"Backend LLM: Using Google Gemini ({GOOGLE_MODEL})")

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
    by_db: dict[str, list] = {}
    for c in chunks:
        by_db.setdefault(c["db"], []).append(c)

    parts = []
    for db, db_chunks in by_db.items():
        parts.append(f"=== {db.upper()} DOCUMENTATION ===")
        for i, chunk in enumerate(db_chunks):
            display_score = chunk.get("rerank_score", chunk.get("fused_score", chunk.get("score", 0.0)))
            parts.append(
                f"[Chunk {i+1} | {chunk['doc_name']} | score={display_score:.2f}]\n"
                f"{chunk['text']}"
            )
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, chunks: list[dict]) -> dict:
    """Generate grounded answer from retrieved chunks."""
    # Filter out low relevance chunks — check rerank_score first, then fused_score, then original score
    relevant = [
        c for c in chunks
        if c.get("rerank_score", c.get("fused_score", c.get("score", 0.0))) > 0.15
    ]

    if not relevant:
        return {
            "answer": "I cannot answer this question based on the available documentation. No relevant chunks found.",
            "sources": [],
            "dbs_used": [],
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


@observe()
def query_rag(
    query: str,
    top_k: int = 10,
    rerank_top_k: int = 5,
    db_filter: str | None = None,
) -> dict:
    """
    Full RAG pipeline with Langfuse tracing: detect DB → retrieve → rerank → generate.
    db_filter: explicit override. If None, auto-detects from query.
    """
    t0 = time.perf_counter()

    detected_db = db_filter or detect_database(query)

    # 1. Hybrid search (falls back to semantic)
    chunks = hybrid_search(query, top_k=top_k, db_filter=detected_db)
    t_retrieval = (time.perf_counter() - t0) * 1000

    # 2. Rerank
    reranked = rerank(query, chunks, top_k=rerank_top_k)
    t_rerank = (time.perf_counter() - t0) * 1000 - t_retrieval

    # 3. Generate answer
    result = generate_answer(query, reranked)
    t_generation = (time.perf_counter() - t0) * 1000 - t_retrieval - t_rerank

    # Log rich metadata and spans to Langfuse using native OpenTelemetry APIs
    from opentelemetry import trace
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.set_attribute("detected_db", str(detected_db or "None"))
        current_span.set_attribute("chunks_retrieved", len(chunks))
        current_span.set_attribute("chunks_after_rerank", len(reranked))
        current_span.set_attribute("refused", bool(result["refused"]))
        current_span.set_attribute("top_rerank_score", float(reranked[0].get("rerank_score", 0.0) if reranked else 0.0))
        current_span.set_attribute("latency_retrieval_ms", round(t_retrieval, 1))
        current_span.set_attribute("latency_rerank_ms", round(t_rerank, 1))
        current_span.set_attribute("latency_generation_ms", round(t_generation, 1))

    return {
        "query": query,
        "answer": result["answer"],
        "sources": result["sources"],
        "dbs_used": result["dbs_used"],
        "detected_db": detected_db,
        "chunks_retrieved": len(chunks),
        "chunks_used": len(result["context_used"]),
        "context_used": result["context_used"],
        "refused": result["refused"],
        "top_chunk_score": reranked[0].get("rerank_score", 0.0) if reranked else 0.0,
    }


if __name__ == "__main__":
    print("RAG chain script executable")
