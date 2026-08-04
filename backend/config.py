"""Central configuration for the DBrain RAG backend.

Single source of truth for every runtime setting. Values are read from the
environment (``.env`` at the repository root) with sensible defaults so the
service boots with zero configuration and degrades gracefully per provider.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
LOG = logging.getLogger("dbrain")

load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key, "").strip().lower()
    if raw == "":
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

CHROMA_PERSIST_DIR = Path(_env("CHROMA_PERSIST_DIR", "./chroma_db_local"))
if not CHROMA_PERSIST_DIR.is_absolute():
    CHROMA_PERSIST_DIR = PROJECT_ROOT / CHROMA_PERSIST_DIR

DOCS_DIR = Path(_env("DOCS_DIR", "./data/db_docs"))
if not DOCS_DIR.is_absolute():
    DOCS_DIR = PROJECT_ROOT / DOCS_DIR
# Fall back to ./data when the db_docs sub-folder is missing or empty
# (legacy/production layout where the corpus lives one level up).
if not DOCS_DIR.exists() or not any(DOCS_DIR.rglob("*.txt")):
    fallback = PROJECT_ROOT / "data"
    if fallback.exists() and any(fallback.rglob("*.txt")):
        DOCS_DIR = fallback

DB_FILE = Path(_env("DB_FILE", "./data/dbrain.db"))
if not DB_FILE.is_absolute():
    DB_FILE = PROJECT_ROOT / DB_FILE

COLLECTION_NAME = "multi_db_docs"
VALID_DB_FILTERS = {"postgresql", "mysql", "mongodb"}
DB_LABELS = {"postgresql": "PostgreSQL", "mysql": "MySQL", "mongodb": "MongoDB"}

# --------------------------------------------------------------------------
# Embeddings — provider abstraction
#   local  (default) all-MiniLM-L6-v2 via ChromaDB's bundled ONNX engine (offline)
#   jina   jina-embeddings-v4 (cloud, requires a funded JINA_API_KEY)
#   gemini Google Generative AI embeddings (requires GOOGLE_API_KEY)
# --------------------------------------------------------------------------

EMBEDDING_PROVIDER = _env("EMBEDDING_PROVIDER", "local").lower()
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def resolved_embedding_model() -> str:
    """Provider-aware model name so defaults never silently mix scales."""
    if EMBEDDING_PROVIDER == "jina" and EMBEDDING_MODEL == "all-MiniLM-L6-v2":
        return "jina-embeddings-v4"
    if EMBEDDING_PROVIDER == "gemini" and EMBEDDING_MODEL == "all-MiniLM-L6-v2":
        return "models/text-embedding-004"
    return EMBEDDING_MODEL


# --------------------------------------------------------------------------
# LLM — standardised on Groq openai/gpt-oss-120b (chat + RAGAS evaluator)
# --------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = _env("GROQ_MODEL", "openai/gpt-oss-120b")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL = _env("GOOGLE_MODEL", "gemini-2.0-flash")
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
USE_OLLAMA = _env_bool("USE_OLLAMA", False)
OLLAMA_MODEL = _env("OLLAMA_MODEL", "")
MAX_OUTPUT_TOKENS = _env_int("MAX_OUTPUT_TOKENS", 2048)
TEMPERATURE = _env_float("TEMPERATURE", 0.0)

# --------------------------------------------------------------------------
# Retrieval / reranking
# --------------------------------------------------------------------------

TOP_K_DEFAULT = _env_int("TOP_K_DEFAULT", 10)
RERANK_TOP_K = _env_int("RERANK_TOP_K", 5)
# Gate applied to the normalised relevance score in [0, 1] (scale-independent).
RELEVANCE_THRESHOLD = _env_float("RELEVANCE_THRESHOLD", 0.15)
GUARDRAIL_ENABLED = _env_bool("GUARDRAIL_ENABLED", True)
# Reranker requires sentence-transformers; when missing the pipeline degrades
# gracefully to RRF ordering (keeps the 512MB free tier usable).
RERANKER_ENABLED = _env_bool("RERANKER_ENABLED", True)
RERANKER_MODEL = _env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --------------------------------------------------------------------------
# API / security (Module 10 owns full JWT+RBAC; this is the pluggable hook)
# --------------------------------------------------------------------------

CORS_ORIGINS = [o.strip() for o in _env("CORS_ORIGINS", "").split(",") if o.strip()]
if not CORS_ORIGINS:
    CORS_ORIGINS = ["http://localhost:3000"]
API_KEY = os.getenv("API_KEY", "")  # empty disables authentication
REQUEST_ID_HEADER = "X-Request-ID"
AUTH_EXEMPT_PATHS = {"/", "/health"}

# --------------------------------------------------------------------------
# Observability
# --------------------------------------------------------------------------

LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# --------------------------------------------------------------------------
# Live DB Diagnostics  (Phase 2 — Diagnostic Agent, DBrain v2)
# --------------------------------------------------------------------------
# Read-only credentials — used by inspectors (never write to DB)
PG_DIAG_DSN    = os.getenv("PG_DIAG_DSN", "")     # postgresql://readonly:pass@host:5432/db
MYSQL_DIAG_DSN = os.getenv("MYSQL_DIAG_DSN", "")   # mysql+pymysql://readonly:pass@host:3306/db
MONGO_DIAG_URI = os.getenv("MONGO_DIAG_URI", "")   # mongodb://readonly:pass@host:27017/db
MONGO_DIAG_DB  = os.getenv("MONGO_DIAG_DB", "admin")

# Schema-owner credentials — used ONLY after HITL approval, to apply safe changes
PG_APPLY_DSN    = os.getenv("PG_APPLY_DSN", "")
MYSQL_APPLY_DSN = os.getenv("MYSQL_APPLY_DSN", "")
MONGO_APPLY_URI = os.getenv("MONGO_APPLY_URI", "")

# SQLite file shared between SessionStore, MetricsStore, ChangeManager, ImprovementTracker
DB_FILE = str(PROJECT_ROOT / "data" / "dbrain.db")