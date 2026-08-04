"""FE-7.2/FE-7.3 pluggable context providers.

The scope document calls for RAG over documentation PLUS schema, past
recommendations and EXPLAIN guides, with the prompt optionally enriched by live
metrics / slow-query data. This module defines the ``ContextProvider`` contract
and ships an ``EmptyContextProvider`` that degrades gracefully, so Module 3/6
data slots can be wired in later without touching the rag pipeline.
"""

from __future__ import annotations

import logging
from typing import Protocol

LOG = logging.getLogger("dbrain.providers")


class ContextProvider(Protocol):
    """A source of additional context injected into the RAG prompt."""

    name: str

    def get_context(self, query: str, dbs: list[str]) -> list[dict]:
        """Return context blocks ``{title, body}`` for the given databases."""
        ...


class EmptyContextProvider:
    """Placeholder that returns no context (docs-only mode).

    Swap for a real provider (schema discovery, recommendation store, EXPLAIN
    guide index, live slow-query metrics) once the corresponding modules land.
    """

    name = "empty"

    def get_context(self, query: str, dbs: list[str]) -> list[dict]:
        return []


def build_context_providers(query: str, dbs: list[str]) -> list[dict]:
    """Collect context blocks from every registered provider (empty now).

    Future wiring:
        providers = [SchemaProvider(...), RecommendationProvider(...), ...]
    """
    blocks: list[dict] = []
    providers: list[ContextProvider] = [EmptyContextProvider()]
    for provider in providers:
        try:
            blocks.extend(provider.get_context(query, dbs))
        except Exception as exc:  # noqa: BLE001 - providers must not break RAG
            LOG.warning("Context provider %s failed: %s", provider.name, exc)
    return blocks