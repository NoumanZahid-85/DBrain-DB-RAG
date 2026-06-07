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
        time.sleep(0.5)  

    print(f"\n{'='*50}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*50}")
    for db, count in stats.items():
        print(f"  {db:<15s}: {count} pages")
    if failed:
        print(f"\n  Failed ({len(failed)}): {', '.join(failed)}")
        print("  Re-run script to retry failed pages.")
    print(f"\nCorpus saved to: {DOCS_DIR}")
