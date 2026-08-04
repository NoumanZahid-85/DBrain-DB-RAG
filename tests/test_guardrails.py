"""Unit tests for FE-7.4 command guardrails and the evidence-based relevance gate."""

import sys
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend import guardrails


class TestCommandGuardrail:
    def test_blocks_imperative_drop(self):
        blocked, reason = guardrails.check_command_guardrail("DROP TABLE users")
        assert blocked is True
        assert reason

    def test_blocks_delete_from(self):
        blocked, _ = guardrails.check_command_guardrail("DELETE FROM orders")
        assert blocked is True

    def test_blocks_alter_table_drop(self):
        blocked, _ = guardrails.check_command_guardrail("ALTER TABLE users DROP COLUMN email")
        assert blocked is True

    def test_blocks_update_set(self):
        blocked, _ = guardrails.check_command_guardrail("UPDATE users SET admin = true")
        assert blocked is True

    def test_allows_how_to_drop(self):
        blocked, _ = guardrails.check_command_guardrail(
            "How do I drop an index in PostgreSQL?"
        )
        assert blocked is False

    def test_allows_what_does_delete_do(self):
        blocked, _ = guardrails.check_command_guardrail(
            "What does DELETE do in MySQL and what are the risks?"
        )
        assert blocked is False

    def test_allows_explain_syntax(self):
        blocked, _ = guardrails.check_command_guardrail(
            "Explain the TRUNCATE TABLE syntax in PostgreSQL"
        )
        assert blocked is False


class TestNormalizeRelevance:
    def test_scales_to_unit_interval(self):
        chunks = [
            {"text": "a", "fused_score": 0.001},
            {"text": "b", "fused_score": 0.003},
            {"text": "c", "fused_score": 0.030},
        ]
        out = guardrails.normalize_relevance(chunks)
        assert max(c["normalized_score"] for c in out) == 1.0
        assert all(0.0 <= c["normalized_score"] <= 1.0 for c in out)

    def test_threshold_filters_weak_chunks(self):
        chunks = [
            {"text": "a", "fused_score": 0.001},
            {"text": "b", "fused_score": 0.003},
            {"text": "c", "fused_score": 0.030},
        ]
        out = guardrails.normalize_relevance(chunks)
        kept = guardrails.filter_relevant(out, threshold=0.2)
        assert len(kept) < len(chunks)
        assert all(c["normalized_score"] >= 0.2 for c in kept)

    def test_threshold_zero_keeps_all(self):
        chunks = guardrails.normalize_relevance(
            [{"text": "a", "fused_score": 0.0}, {"text": "b", "fused_score": 0.0}]
        )
        assert len(guardrails.filter_relevant(chunks, threshold=0)) == 2


class TestEnsureDbDiversity:
    def test_multi_db_keeps_every_db(self):
        chunks = []
        for db in ("postgresql", "mysql", "mongodb"):
            for i in range(4):
                chunks.append(
                    {
                        "text": f"{db}-{i}",
                        "db": db,
                        "doc_name": f"{db}.txt",
                        "fused_score": 0.01 * (i + 1),
                        "normalized_score": 0.25 * (i + 1),
                    }
                )
        out = guardrails.ensure_db_diversity(chunks, per_db=2)
        dbs = {c["db"] for c in out}
        assert dbs == {"postgresql", "mysql", "mongodb"}
        assert len(out) == 6  # 2 per DB

    def test_single_db_untouched(self):
        chunks = [
            {"text": "a", "db": "postgresql", "normalized_score": 1.0},
            {"text": "b", "db": "postgresql", "normalized_score": 0.5},
        ]
        assert guardrails.ensure_db_diversity(chunks) == chunks

    def test_empty_input(self):
        assert guardrails.ensure_db_diversity([]) == []