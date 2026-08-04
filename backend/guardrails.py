"""FE-7.4 command guardrails + prompt-security guardrails + evidence-based relevance gate.

Three layers:

1. ``check_command_guardrail`` — blocks destructive/unsafe SQL (DROP, DELETE,
   TRUNCATE, ALTER ... DROP, DB DROP) from being executed on the model.

2. ``check_prompt_guardrail`` — blocks prompt injection, jailbreak, role
   confusion, system-prompt leakage, and data-extraction attacks.

3. ``filter_relevant`` — the evidence gate. After the mixed cosine/BM25
   scores are normalised to a single [0,1] relevance scale by the canonical
   ``hybrid_retrieval.normalize_scores``, chunks below the configured
   threshold are discarded so the model never answers from weak context.

Note: normalisation lives exclusively in ``hybrid_retrieval.normalize_scores``
(single seam, single fallback: 1.0 when all scores are equal). The old
``normalize_relevance()`` duplicate has been removed from this module.
"""

from __future__ import annotations

import re

from backend import config

# ------------------------------------------------------------------
# Layer 1 — destructive SQL (FE-7.4, unchanged)
# ------------------------------------------------------------------

_DESTRUCTIVE = re.compile(
    r"(?i)\b(DROP|DELETE|TRUNCATE|UPDATE|INSERT|REPLACE)\b"
)
_ALTER_DROP = re.compile(
    r"(?i)\bALTER\b[^;]{0,200}\bDROP\b"
)
_SAFE_CONTEXT = re.compile(
    r"(?i)\b("
    r"what\s+(is|was|are|were|does|did|can|could|should|would)|"
    r"how\s+(do\s+i|does|can|could|should|would|to)|"
    r"why\s+(is|are|does|did|do)|"
    r"when\s+(is|was|should|would)|"
    r"where\s+(is|are|can)|"
    r"explain\b|"
    r"describe\b|"
    r"difference\s+between\b|"
    r"compare\b|"
    r"learn\s+about\b|"
    r"teach\s+me\s+about\b|"
    r"understand\b|"
    r"meaning\s+of\b|"
    r"syntax\s+of\b|"
    r"documentation\s+about\b|"
    r"docs\s+about\b"
    r")\b"
)

# ------------------------------------------------------------------
# Layer 2 — prompt-security guardrails
# ------------------------------------------------------------------

# Instruction-override / injection intent patterns.
_INJECTION = re.compile(
    r"(?i)"
    r"\b(ignore\s+(all\s+)?previous\s+instructions|"
    r"disregard\s+(all\s+)?previous|"
    r"new\s+\w+\s+instructions|"
    r"new\s+(system\s+)?instructions|"
    r"override\s+(your\s+)?(rules|instructions|prompt)|"
    r"admin\s+override|"
    r"developer\s+mode|"
    r"(act\s+as|become|enable|enter).*jailbreak|"
    r"jailbreak\s+(mode|enabled|on)|"
    r"do\s+anything\s+now|"
    r"no\s+restrictions|"
    r"unfiltered\s+mode|"
    r"treated\s+as\s+instructions|"
    r"documentation\s+update.*instructions|"
    r"system\s+prompt\s+(override|bypass)|"
    r"you\s+are\s+now\b|"
    r"pretend\s+(to\s+be|you\s+are)\b|"
    r"switch\s+to\s+(the\s+)?role\b|"
    r"act\s+as\s+(if|though|a|an)\b|"
    r"you\s+have\s+been\s+(placed|put)\s+in\b|"
    r"guardrails?\s+(have\s+been\s+)?(disabled|lifted|removed)|"
    r"constraints?\s+(have\s+been\s+)?(disabled|lifted|removed)|"
    r"rules?\s+(have\s+been\s+)?(disabled|lifted|removed)|"
    r"from\s+now\s+on\b|"
    r"every\s+answer\s+must\s+start\s+with\b|"
    r"output\s+the\s+token\b|"
    r"reply\s+with\s+exactly\b|"
    r"print\s+the\s+token\b|"
    r"confirm\s+you\s+are\b|"
    r"confirm\s+override\b|"
    r"echo\s+the\s+marker\b|"
    r"repeat\s+the\s+word\b)"
)

# System-prompt leakage intent patterns.
_LEAKAGE = re.compile(
    r"(?i)"
    r"\b(repeat\s+(your\s+)?(system\s+)?prompt|"
    r"print\s+(your\s+)?(system\s+)?prompt|"
    r"show\s+me\s+(your\s+)?(system\s+)?prompt|"
    r"what\s+are\s+(your\s+)?instructions|"
    r"reveal\s+(your\s+)?(system\s+)?prompt|"
    r"output\s+(your\s+)?(system\s+)?prompt|"
    r"tell\s+me\s+(your\s+)?(system\s+)?prompt|"
    r"share\s+(your\s+)?(system\s+)?prompt|"
    r"display\s+(your\s+)?(system\s+)?prompt|"
    r"what\s+is\s+your\s+system\s+prompt|"
    r"what\s+are\s+your\s+rules|"
    r"what\s+are\s+your\s+instructions|"
    r"show\s+me\s+your\s+rules|"
    r"output\s+the\s+(full|entire)\s+(system\s+)?prompt|"
    r"repeat\s+(the|your)\s+instructions\b)"
)

# Data-extraction intent patterns.
_EXTRACTION = re.compile(
    r"(?i)"
    r"\b(list\s+every\s+(document|chunk|source|title)|"
    r"dump\s+(all|the|your)\s+(documents|chunks|data|knowledge|"
    r"base|corpus)|"
    r"show\s+me\s+(all|every)\s+(document|chunk|source|title)|"
    r"raw\s+content\b|"
    r"full\s+copy\b|"
    r"entire\s+contents?\b|"
    r"all\s+chunks?\b|"
    r"every\s+document\s+in\s+your\s+knowledge\s+base\b|"
    r"enumerate\s+(all|every)\s+(document|chunk|source)|"
    r"give\s+me\s+(the|all)\s+(documents|chunks|sources)\b|"
    r"return\s+(the|all)\s+(documents|chunks|sources)\b|"
    r"extract\s+(all|every)\s+(document|chunk|source)\b)"
)

# Legitimate learning-question framing — if present, the query is about
# the *concept* of the attack, not an attempt to execute it.
_LEARNING_CONTEXT = re.compile(
    r"(?i)\b("
    r"what\s+is\s+(prompt\s+injection|jailbreak|role\s+confusion|"
    r"system\s+prompt\s+leakage|data\s+extraction)|"
    r"how\s+do\s+I\s+(prevent|detect|stop|defend\s+against|protect\s+against)"
    r"\s+(prompt\s+injection|jailbreak|role\s+confusion|leakage|extraction)|"
    r"explain\s+(prompt\s+injection|jailbreak|role\s+confusion|"
    r"system\s+prompt\s+leakage|data\s+extraction)|"
    r"describe\s+(prompt\s+injection|jailbreak|role\s+confusion|"
    r"system\s+prompt\s+leakage|data\s+extraction)|"
    r"difference\s+between\s+(prompt\s+injection|jailbreak|"
    r"role\s+confusion|leakage|extraction)|"
    r"compare\s+(prompt\s+injection|jailbreak|role\s+confusion|"
    r"leakage|extraction)|"
    r"learn\s+about\s+(prompt\s+injection|jailbreak|role\s+confusion|"
    r"leakage|extraction)|"
    r"teach\s+me\s+about\s+(prompt\s+injection|jailbreak|role\s+confusion|"
    r"leakage|extraction)|"
    r"understand\s+(prompt\s+injection|jailbreak|role\s+confusion|"
    r"leakage|extraction)|"
    r"meaning\s+of\s+(prompt\s+injection|jailbreak|role\s+confusion|"
    r"leakage|extraction)|"
    r"syntax\s+of\s+(prompt\s+injection|jailbreak|role\s+confusion|"
    r"leakage|extraction)|"
    r"documentation\s+about\s+(prompt\s+injection|jailbreak|role\s+confusion|"
    r"leakage|extraction)|"
    r"docs\s+about\s+(prompt\s+injection|jailbreak|role\s+confusion|"
    r"leakage|extraction))\b"
)


def check_command_guardrail(prompt: str) -> tuple[bool, str | None]:
    """Return (blocked, reason) for executable destructive SQL intent.

    A question phrase like "how do I DROP a column" is documentation intent
    and passes; an imperative "DROP TABLE users" or "DELETE FROM orders"
    without an explanatory frame is blocked.
    """
    if not config.GUARDRAIL_ENABLED:
        return False, None

    dangerous = bool(_DESTRUCTIVE.search(prompt)) or bool(_ALTER_DROP.search(prompt))
    if not dangerous:
        return False, None

    imperative = bool(
        re.match(
            r"(?i)^\s*(drop\s+\w+|delete\s+from|truncate\s+table|"
            r"update\s+\w+\s+set|alter\s+table\s+\w+\s+drop)",
            prompt,
        )
    )
    if imperative and not _SAFE_CONTEXT.search(prompt):
        reason = (
            "Destructive command detected. This RAG assistant only provides "
            "documentation guidance and refuses to formulate DROP/DELETE/UPDATE/"
            "INSERT statements that could damage the monitored database."
        )
        return True, reason
    return False, None


def check_prompt_guardrail(prompt: str) -> tuple[bool, str | None]:
    """Return (blocked, reason) for prompt-security attacks.

    Detects five attack classes and blocks imperative attempts while
    allowing legitimate learning questions about the concepts.

    Classes: prompt injection, system-prompt leakage, data extraction,
    jailbreak, role confusion.
    """
    if not config.GUARDRAIL_ENABLED:
        return False, None

    low = prompt.lower().strip()

    # First pass: detect any attack pattern present.
    has_injection = bool(_INJECTION.search(low))
    has_leakage = bool(_LEAKAGE.search(low))
    has_extraction = bool(_EXTRACTION.search(low))

    # If NO attack patterns detected, it's safe.
    if not (has_injection or has_leakage or has_extraction):
        return False, None

    # Attack pattern(s) detected → BLOCK.
    # Legitimate learning questions ("What is prompt injection?") do NOT
    # trigger the attack patterns because those patterns target imperative
    # instructions, not concept questions. So this block is safe.
    if has_injection:
        return (
            True,
            "Prompt-injection attempt detected. This RAG assistant only "
            "answers from its documentation context and ignores instructions "
            "that try to override its rules or adopt a different role.",
        )

    if has_leakage:
        return (
            True,
            "System-prompt leakage attempt detected. This RAG assistant "
            "refuses to reveal its internal instructions, system prompt, or "
            "guardrail configuration.",
        )

    if has_extraction:
        return (
            True,
            "Data-extraction attempt detected. This RAG assistant only "
            "provides summarised, cited answers from its documentation "
            "context and does not dump raw chunks, source lists, or internal "
            "index metadata.",
        )

    return False, None


# ---------------------------------------------------------------------------
# Normalisation — single canonical implementation lives in hybrid_retrieval.
# This alias keeps any direct callers working while the real logic stays DRY.
# ---------------------------------------------------------------------------
def normalize_relevance(chunks: list[dict], fuse_key: str = "fused_score") -> list[dict]:
    """Thin alias → ``hybrid_retrieval.normalize_scores`` (canonical seam).

    Do not extend this function. Add logic to ``hybrid_retrieval.normalize_scores``
    so there is always exactly one normalisation implementation.
    """
    from backend.hybrid_retrieval import normalize_scores  # deferred to avoid circular
    return normalize_scores(chunks, key=fuse_key)


def filter_relevant(chunks: list[dict], threshold: float | None = None) -> list[dict]:
    """Keep only chunks whose normalised relevance clears the gate."""
    if threshold is None:
        threshold = config.RELEVANCE_THRESHOLD
    if threshold <= 0:
        return chunks
    return [c for c in chunks if c.get("normalized_score", 0.0) >= threshold]


def aggregate_top_score(chunks: list[dict]) -> float:
    """Best normalised relevance across the surviving context (for UI/metrics)."""
    if not chunks:
        return 0.0
    return max(c.get("normalized_score", 0.0) for c in chunks)


def ensure_db_diversity(chunks: list[dict], per_db: int = 3) -> list[dict]:
    """Guarantee every database with relevant chunks is represented in the prompt.

    Global RRF ranking can starve a database out of a top-K slice on
    cross-database comparison queries (FE-7.3). For multi-DB queries, take the
    top ``per_db`` chunks from each DB present, then re-rank the union by
    relevance so ordering stays score-driven while coverage is guaranteed.
    Single-DB queries keep pure score order untouched.
    """
    if not chunks:
        return chunks
    dbs = {c["db"] for c in chunks}
    if len(dbs) <= 1:
        return chunks

    picked: list[dict] = []
    for db in sorted(dbs):
        db_chunks = [c for c in chunks if c["db"] == db]
        picked.extend(
            sorted(
                db_chunks,
                key=lambda c: c.get("normalized_score", c.get("fused_score", 0.0)),
                reverse=True,
            )[:per_db]
        )
    picked.sort(
        key=lambda c: c.get("normalized_score", c.get("fused_score", 0.0)),
        reverse=True,
    )
    return picked


def lexical_refusal_reason() -> str:
    """Human-readable refusal message for destructive SQL intent."""
    return (
        "This request targets a live database with a destructive SQL command. "
        "DBrain RAG provides documentation and recommendation guidance only; "
        "it will not generate DROP/DELETE/UPDATE/INSERT statements."
    )