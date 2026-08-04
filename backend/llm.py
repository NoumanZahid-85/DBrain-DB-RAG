"""LLM provider factory.

Standardises on Groq ``openai/gpt-oss-120b`` for both the RAG chat pipeline and
the RAGAS evaluator. Provider selection is driven entirely by ``backend.config``
so the rest of the codebase never branches on ``USE_OLLAMA``-style flags.

Kept as a tiny module (not scattered across rag_chain/eval) so swap-out is a
one-line change and unit tests can inject a fake LLM at this seam.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks import Callbacks

from backend import config

LOG = logging.getLogger("dbrain.llm")

_LLM: BaseChatModel | None = None


def get_llm() -> BaseChatModel:
    """Return the process-wide chat model instance (created once)."""
    global _LLM
    if _LLM is not None:
        return _LLM

    _LLM = _build_llm()
    LOG.info("LLM backend active: %s", llm_label())
    return _LLM


def _build_llm() -> BaseChatModel:
    if config.GROQ_API_KEY and not config.USE_OLLAMA:
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=config.GROQ_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_OUTPUT_TOKENS,
        )

    if config.USE_OLLAMA:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.OLLAMA_MODEL,
            temperature=config.TEMPERATURE,
            num_predict=config.MAX_OUTPUT_TOKENS,
        )

    if config.GOOGLE_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.GOOGLE_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=config.TEMPERATURE,
            max_output_tokens=config.MAX_OUTPUT_TOKENS,
        )

    raise RuntimeError(
        "No LLM configured. Set GROQ_API_KEY (recommended) or GOOGLE_API_KEY "
        "in .env, or enable USE_OLLAMA for a local model."
    )


def llm_label() -> str:
    """Human-readable label used by /stats and logs."""
    if config.GROQ_API_KEY and not config.USE_OLLAMA:
        return f"groq ({config.GROQ_MODEL})"
    if config.USE_OLLAMA:
        return f"ollama ({config.OLLAMA_MODEL})"
    if config.GOOGLE_API_KEY:
        return f"google-gemini ({config.GOOGLE_MODEL})"
    return "unconfigured"


def reset_llm() -> None:
    """Drop the cached instance (used by tests)."""
    global _LLM
    _LLM = None