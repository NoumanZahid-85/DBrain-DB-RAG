"""Unit tests for hybrid retrieval primitives (RRF + score normalisation)
and database routing. Hermetic: no external services, no network."""

import sys
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.hybrid_retrieval import normalize_scores, reciprocal_rank_fusion
from backend.rag_chain import detect_database


class TestRRF:
    def test_fuses_shared_and_unique_docs(self):
        semantic = [
            {"chunk_id": 1, "text": "a", "score": 0.8, "db": "postgresql"},
            {"chunk_id": 2, "text": "b", "score": 0.7, "db": "postgresql"},
        ]
        bm25 = [
            {"chunk_id": 1, "text": "a", "score": 12.0, "db": "postgresql"},
            {"chunk_id": 3, "text": "c", "score": 8.0, "db": "mongodb"},
        ]
        fused = reciprocal_rank_fusion(semantic, bm25)
        assert fused[0]["chunk_id"] == 1  # present in both lists ranks highest
        methods = {c["chunk_id"]: c["retrieval_method"] for c in fused}
        assert methods[1] == "hybrid"
        assert methods[2] == "semantic"
        assert methods[3] == "bm25"

    def test_keeps_per_path_scores(self):
        semantic = [{"chunk_id": 1, "text": "a", "score": 0.66, "db": "pg"}]
        bm25 = [{"chunk_id": 1, "text": "a", "score": 21.0, "db": "pg"}]
        fused = reciprocal_rank_fusion(semantic, bm25)
        assert fused[0]["semantic_score"] == 0.66
        assert fused[0]["bm25_score"] == 21.0


class TestNormalizeScores:
    def test_maps_to_unit_interval(self):
        chunks = [
            {"chunk_id": 1, "fused_score": 0.0},
            {"chunk_id": 2, "fused_score": 0.031},
            {"chunk_id": 3, "fused_score": 0.033},
        ]
        out = normalize_scores(chunks)
        assert max(c["normalized_score"] for c in out) == 1.0
        assert all(0.0 <= c["normalized_score"] <= 1.0 for c in out)

    def test_empty_input(self):
        assert normalize_scores([]) == []


class TestDetectDatabase:
    def test_postgres_keywords(self):
        assert detect_database("How does VACUUM work in postgresql?") == "postgresql"

    def test_mysql_keywords(self):
        assert detect_database("innodb deadlock handling in mysql") == "mysql"

    def test_mongodb_keywords(self):
        assert detect_database("mongodb aggregation pipeline $lookup") == "mongodb"

    def test_ambiguous_returns_none(self):
        assert detect_database("how do databases index their columns?") is None

    def test_explicit_filter_wins(self):
        # detect_database is only a fallback; explicit filters win in query_rag.
        assert detect_database("general question") is None