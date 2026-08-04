import sys
import os

sys.path.insert(0, os.getcwd())

from backend.hybrid_retrieval import hybrid_search
from backend.guardrails import filter_relevant, ensure_db_diversity

q = "Compare index creation: PostgreSQL vs MySQL vs MongoDB"
chunks = hybrid_search(q, top_k=15, db_filter=None)
print("total candidates:", len(chunks))
for c in chunks:
    print(
        "  {:<11} norm={:.3f} m={:<8} | {}".format(
            c["db"], c.get("normalized_score", 0.0), c.get("retrieval_method", ""), c["doc_name"]
        )
    )
kept = filter_relevant(chunks)
print("after gate:", len(kept), "dbs:", sorted(set(c["db"] for c in kept)))
div = ensure_db_diversity(kept, per_db=3)
print("after diversity:", len(div), "dbs:", sorted(set(c["db"] for c in div)))
for c in div:
    print(
        "  {:<11} norm={:.3f} | {}".format(c["db"], c.get("normalized_score", 0.0), c["doc_name"])
    )