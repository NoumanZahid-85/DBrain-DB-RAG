"""
Downloads official documentation pages for PostgreSQL, MySQL, and MongoDB.
Each file is saved with a db prefix so metadata tagging is automatic.
"""

import os
import re
import time
from urllib.parse import urljoin
import requests
# pyrefly: ignore [missing-import]
import cloudscraper
from bs4 import BeautifulSoup

DOCS_DIR = "./data/db_docs"
os.makedirs(DOCS_DIR, exist_ok=True)

PG_BASE = "https://www.postgresql.org/docs/16/"

# ─────────────────────────────────────────────
# CORPUS DEFINITION
# key = filename prefix  |  value = (db_tag, url)
# ─────────────────────────────────────────────

PAGES = {
    # ── PostgreSQL ──────────────────────────────────────────────────────
    #
    # Each URL below is a *leaf* page with actual content, NOT a chapter
    # overview / TOC page.  Grouped by topic.
    #
    # --- Indexes (Chapter 11) ---
    "pg_indexes_intro":          ("postgresql", PG_BASE + "indexes-intro.html"),
    "pg_indexes_types":          ("postgresql", PG_BASE + "indexes-types.html"),
    "pg_indexes_multicolumn":    ("postgresql", PG_BASE + "indexes-multicolumn.html"),
    "pg_indexes_ordering":       ("postgresql", PG_BASE + "indexes-ordering.html"),
    "pg_indexes_bitmap_scans":   ("postgresql", PG_BASE + "indexes-bitmap-scans.html"),
    "pg_indexes_unique":         ("postgresql", PG_BASE + "indexes-unique.html"),
    "pg_indexes_expressional":   ("postgresql", PG_BASE + "indexes-expressional.html"),
    "pg_indexes_partial":        ("postgresql", PG_BASE + "indexes-partial.html"),
    "pg_indexes_index_only":     ("postgresql", PG_BASE + "indexes-index-only-scans.html"),
    "pg_indexes_opclass":        ("postgresql", PG_BASE + "indexes-opclass.html"),
    "pg_indexes_collations":     ("postgresql", PG_BASE + "indexes-collations.html"),
    "pg_indexes_examine":        ("postgresql", PG_BASE + "indexes-examine.html"),

    # --- Queries (Chapter 7) ---
    "pg_queries_overview":       ("postgresql", PG_BASE + "queries-overview.html"),
    "pg_queries_table_expr":     ("postgresql", PG_BASE + "queries-table-expressions.html"),
    "pg_queries_select":         ("postgresql", PG_BASE + "queries-select-lists.html"),
    "pg_queries_union":          ("postgresql", PG_BASE + "queries-union.html"),
    "pg_queries_order":          ("postgresql", PG_BASE + "queries-order.html"),
    "pg_queries_limit":          ("postgresql", PG_BASE + "queries-limit.html"),
    "pg_queries_values":         ("postgresql", PG_BASE + "queries-values.html"),
    "pg_queries_with":           ("postgresql", PG_BASE + "queries-with.html"),

    # --- Performance Tips (Chapter 14) ---
    "pg_explain":                ("postgresql", PG_BASE + "using-explain.html"),
    "pg_planner_stats":          ("postgresql", PG_BASE + "planner-stats.html"),
    "pg_explicit_joins":         ("postgresql", PG_BASE + "explicit-joins.html"),
    "pg_populate":               ("postgresql", PG_BASE + "populate.html"),
    "pg_non_durability":         ("postgresql", PG_BASE + "non-durability.html"),

    # --- MVCC / Transactions (Chapter 13) ---
    "pg_mvcc_intro":             ("postgresql", PG_BASE + "mvcc-intro.html"),
    "pg_transaction_iso":        ("postgresql", PG_BASE + "transaction-iso.html"),
    "pg_locking":                ("postgresql", PG_BASE + "explicit-locking.html"),
    "pg_applevel_checks":        ("postgresql", PG_BASE + "applevel-consistency.html"),
    "pg_locking_indexes":        ("postgresql", PG_BASE + "locking-indexes.html"),

    # --- Vacuuming (Chapter 25.1) — this is a leaf page ---
    "pg_vacuum":                 ("postgresql", PG_BASE + "routine-vacuuming.html"),

    # --- Replication / High Availability (Chapter 27) ---
    "pg_ha_standby_servers":     ("postgresql", PG_BASE + "warm-standby.html"),
    "pg_ha_failover":            ("postgresql", PG_BASE + "warm-standby-failover.html"),
    "pg_ha_hot_standby":         ("postgresql", PG_BASE + "hot-standby.html"),

    # --- Other important leaf pages ---
    "pg_stat_stmts":             ("postgresql", PG_BASE + "pgstatstatements.html"),
    "pg_connections":            ("postgresql", PG_BASE + "runtime-config-connection.html"),
    "pg_wal_config":             ("postgresql", PG_BASE + "wal-configuration.html"),
    "pg_parallel_query":         ("postgresql", PG_BASE + "parallel-query.html"),
    "pg_partition":              ("postgresql", PG_BASE + "ddl-partitioning.html"),
    "pg_constraints":            ("postgresql", PG_BASE + "ddl-constraints.html"),

    # ── MySQL ────────────────────────────────────────────────────────────
    "mysql_indexes":             ("mysql", "https://dev.mysql.com/doc/refman/8.0/en/optimization-indexes.html"),
    "mysql_explain":             ("mysql", "https://dev.mysql.com/doc/refman/8.0/en/explain.html"),
    "mysql_explain_output":      ("mysql", "https://dev.mysql.com/doc/refman/8.0/en/explain-output.html"),
    "mysql_slow_query":          ("mysql", "https://dev.mysql.com/doc/refman/8.0/en/slow-query-log.html"),
    "mysql_perf_schema":         ("mysql", "https://dev.mysql.com/doc/refman/8.0/en/performance-schema.html"),
    "mysql_transactions":        ("mysql", "https://dev.mysql.com/doc/refman/8.0/en/innodb-transaction-model.html"),
    "mysql_locking":             ("mysql", "https://dev.mysql.com/doc/refman/8.0/en/innodb-locking.html"),
    "mysql_replication":         ("mysql", "https://dev.mysql.com/doc/refman/8.0/en/replication.html"),
    "mysql_optimize":            ("mysql", "https://dev.mysql.com/doc/refman/8.0/en/optimize-table.html"),
    "mysql_partitions":          ("mysql", "https://dev.mysql.com/doc/refman/8.0/en/partitioning.html"),
    "mysql_buffer_pool":         ("mysql", "https://dev.mysql.com/doc/refman/8.0/en/innodb-buffer-pool.html"),
    "mysql_query_optimizer":     ("mysql", "https://dev.mysql.com/doc/refman/8.0/en/select-optimization.html"),
    "mysql_joins":               ("mysql", "https://dev.mysql.com/doc/refman/8.0/en/join.html"),
    "mysql_subqueries":          ("mysql", "https://dev.mysql.com/doc/refman/8.0/en/subqueries.html"),

    # ── MongoDB ──────────────────────────────────────────────────────────
    "mongo_indexes":             ("mongodb", "https://www.mongodb.com/docs/manual/indexes/"),
    "mongo_explain":             ("mongodb", "https://www.mongodb.com/docs/manual/reference/explain-results/"),
    "mongo_profiler":            ("mongodb", "https://www.mongodb.com/docs/manual/tutorial/manage-the-database-profiler/"),
    "mongo_aggregation":         ("mongodb", "https://www.mongodb.com/docs/manual/aggregation/"),
    "mongo_transactions":        ("mongodb", "https://www.mongodb.com/docs/manual/core/transactions/"),
    "mongo_replication":         ("mongodb", "https://www.mongodb.com/docs/manual/replication/"),
    "mongo_sharding":            ("mongodb", "https://www.mongodb.com/docs/manual/sharding/"),
    "mongo_schema":              ("mongodb", "https://www.mongodb.com/docs/manual/data-modeling/"),
    "mongo_currentop":           ("mongodb", "https://www.mongodb.com/docs/manual/reference/method/db.currentOp/"),
    "mongo_performance":         ("mongodb", "https://www.mongodb.com/docs/manual/administration/analyzing-mongodb-performance/"),
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )
}


# ─────────────────────────────────────────────
# CONTENT EXTRACTION HELPERS
# ─────────────────────────────────────────────

# Junk patterns found in PG site chrome that should be stripped.
_PG_NAV_MARKERS = [
    "Submit correction",
    "If you see anything in the documentation that is not correct",
    "please use",
    "this form",
    "to report a documentation issue.",
    "Policies",
    "Code of Conduct",
    "About PostgreSQL",
    "Copyright © 1996",
]


def _is_pg_toc_page(soup: BeautifulSoup) -> bool:
    """Detect PostgreSQL chapter-level TOC pages (almost no real content)."""
    content_div = soup.find("div", class_="chapter") or soup.find(
        "div", class_="part"
    )
    if not content_div:
        return False
    # TOC pages have a <div class="toc"> or lots of links but very little
    # paragraph text outside the TOC
    toc = content_div.find("div", class_="toc")
    if not toc:
        return False
    # Count non-TOC paragraph text
    for toc_div in content_div.find_all("div", class_="toc"):
        toc_div.decompose()
    remaining_text = content_div.get_text(strip=True)
    # If the remaining text (after removing TOC) is tiny, it's a TOC page.
    return len(remaining_text) < 500


def _extract_pg_subpage_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Extract subpage URLs from a PostgreSQL chapter TOC page."""
    links = []
    seen = set()
    toc = soup.find("div", class_="toc")
    if not toc:
        return links
    for a_tag in toc.find_all("a", href=True):
        href = str(a_tag.get("href", ""))
        # Skip anchor-only links (like #INDEXES-TYPES-BTREE)
        if href.startswith("#"):
            continue
        # Strip fragment
        href = href.split("#")[0]
        full_url = urljoin(base_url, href)
        if full_url not in seen and "postgresql.org/docs/16/" in full_url:
            seen.add(full_url)
            links.append(full_url)
    return links


def _extract_pg_content(soup: BeautifulSoup) -> str:
    """Extract clean content from a PostgreSQL documentation page,
    stripping navigation boilerplate."""
    # The real content lives in <div class="sect1">, <div class="sect2">,
    # or <div class="chapter">. Fall back through several selectors.
    content = (
        soup.find("div", class_="sect1")
        or soup.find("div", class_="sect2")
        or soup.find("div", class_="chapter")
        or soup.find("div", id="docContent")
        or soup.find("main")
        or soup.body
    )
    if not content:
        return soup.get_text(separator="\n")

    # Remove navigation elements within the content
    for nav in content.find_all("div", class_="navheader"):
        nav.decompose()
    for nav in content.find_all("div", class_="navfooter"):
        nav.decompose()
    for nav in content.find_all("table", class_="nav"):
        nav.decompose()

    text = content.get_text(separator="\n")
    return text


def _extract_mysql_content(soup: BeautifulSoup) -> str:
    """Extract content from a MySQL documentation page."""
    content = (
        soup.find("div", id="docs-body")
        or soup.find("div", id="content")
        or soup.find("main")
        or soup.find("article")
        or soup.body
    )
    return content.get_text(separator="\n") if content else soup.get_text(separator="\n")


def _extract_mongo_content(soup: BeautifulSoup) -> str:
    """Extract content from a MongoDB documentation page."""
    content = (
        soup.find("div", class_="body")
        or soup.find("section", class_="section")
        or soup.find("main")
        or soup.find("article")
        or soup.body
    )
    return content.get_text(separator="\n") if content else soup.get_text(separator="\n")


def _clean_text(raw_text: str, db_tag: str) -> str:
    """Clean up extracted text: strip whitespace, remove nav junk."""
    lines = [line.strip() for line in raw_text.splitlines()]
    cleaned = []
    skip_rest = False
    for line in lines:
        if not line:
            continue
        # For PostgreSQL, stop at the footer boilerplate
        if db_tag == "postgresql":
            if any(marker in line for marker in _PG_NAV_MARKERS):
                skip_rest = True
            if skip_rest:
                continue
            # Skip version-picker lines like "Supported Versions:", "13 / 12 / 11..."
            if re.match(r"^(Supported Versions|Development Versions|Unsupported versions):", line):
                skip_rest = True  # Will reset on next content section
                continue
            if re.match(r"^[\d.]+\s*$", line) or line in ("/", ")", "("):
                continue
            # Skip common nav-only lines
            if line in ("Prev", "Up", "Next", "Home", "#"):
                continue
            skip_rest = False
        cleaned.append(line)
    return "\n".join(cleaned)


def _extract_content(soup: BeautifulSoup, db_tag: str) -> str:
    """Route to the correct extractor based on database."""
    if db_tag == "postgresql":
        raw = _extract_pg_content(soup)
    elif db_tag == "mysql":
        raw = _extract_mysql_content(soup)
    elif db_tag == "mongodb":
        raw = _extract_mongo_content(soup)
    else:
        raw = (soup.find("main") or soup.find("article") or soup.body or soup).get_text(separator="\n")
    return _clean_text(raw, db_tag)


# ─────────────────────────────────────────────
# DOWNLOAD LOGIC
# ─────────────────────────────────────────────

def _fetch_soup(url: str, use_cloudscraper: bool = False) -> BeautifulSoup | None:
    """Fetch a URL and return parsed BeautifulSoup, or None on failure.
    Falls back to cloudscraper for sites that block standard requests."""
    try:
        if use_cloudscraper:
            scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "mobile": False}
            )
            resp = scraper.get(url, timeout=30)
        else:
            resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 403 and not use_cloudscraper:
            print(f"    [RETRY] 403 on {url} - retrying with cloudscraper...")
            return _fetch_soup(url, use_cloudscraper=True)
        print(f"    [WARN] Fetch failed: {url} - {e}")
        return None
    except Exception as e:
        if not use_cloudscraper:
            print(f"    [RETRY] Error on {url} - retrying with cloudscraper...")
            return _fetch_soup(url, use_cloudscraper=True)
        print(f"    [WARN] Fetch failed: {url} - {e}")
        return None


def _save_doc(name: str, db_tag: str, url: str, text: str) -> bool:
    """Save extracted text to a file with metadata header."""
    filepath = os.path.join(DOCS_DIR, f"{name}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"DB_TAG: {db_tag}\n")
        f.write(f"SOURCE_URL: {url}\n")
        f.write(f"DOC_NAME: {name}\n")
        f.write("---\n")
        f.write(text)
    return True


def download_page(name: str, db_tag: str, url: str, force: bool = False) -> list[str]:
    """Download a single doc page. Returns list of saved filenames.
    If the page is detected as a PG TOC page, it crawls child pages instead.
    """
    saved: list[str] = []

    # Skip if file already exists with good content (unless --force)
    if not force:
        filepath = os.path.join(DOCS_DIR, f"{name}.txt")
        if os.path.exists(filepath) and os.path.getsize(filepath) > 500:
            print(f"  [SKIP] [{db_tag:12s}] {name:<30s} already exists ({os.path.getsize(filepath):,} bytes)")
            saved.append(name)
            return saved

    soup = _fetch_soup(url)
    if not soup:
        print(f"  [FAIL] [{db_tag:12s}] {name:<30s} FAILED (fetch)")
        return saved

    # -- PostgreSQL TOC detection --
    if db_tag == "postgresql" and _is_pg_toc_page(soup):
        sublinks = _extract_pg_subpage_links(soup, url)
        if sublinks:
            print(f"  [TOC] [{db_tag:12s}] {name:<30s} TOC detected - crawling {len(sublinks)} subpages")
            for i, sub_url in enumerate(sublinks):
                # Derive a filename from the subpage URL
                slug = sub_url.rstrip("/").split("/")[-1].replace(".html", "")
                sub_name = f"{name}_{slug}"

                # Skip existing subpage files too
                if not force:
                    sub_path = os.path.join(DOCS_DIR, f"{sub_name}.txt")
                    if os.path.exists(sub_path) and os.path.getsize(sub_path) > 500:
                        print(f"    [SKIP] [{db_tag:12s}] {sub_name:<30s} already exists")
                        saved.append(sub_name)
                        continue

                sub_soup = _fetch_soup(sub_url)
                if not sub_soup:
                    continue
                text = _extract_content(sub_soup, db_tag)
                if len(text) < 100:
                    print(f"    [SKIP] [{db_tag:12s}] {sub_name:<30s} skipped (too short)")
                    continue
                _save_doc(sub_name, db_tag, sub_url, text)
                saved.append(sub_name)
                print(f"    [OK] [{db_tag:12s}] {sub_name:<30s} {len(text):>8,} chars")
                time.sleep(0.3)
            return saved

    # -- Normal leaf page --
    text = _extract_content(soup, db_tag)
    if len(text) < 100:
        print(f"  [FAIL] [{db_tag:12s}] {name:<30s} FAILED (content too short: {len(text)} chars)")
        return saved

    _save_doc(name, db_tag, url, text)
    saved.append(name)
    print(f"  [OK] [{db_tag:12s}] {name:<30s} {len(text):>8,} chars")
    return saved


# Stale files from v1 that are now replaced by proper leaf-page downloads.
# These are PG chapter-level TOC files that contain only a table of contents.
STALE_FILES = [
    "pg_indexes.txt",       # replaced by pg_indexes_intro, pg_indexes_types, etc.
    "pg_queries.txt",       # replaced by pg_queries_overview, pg_queries_table_expr, etc.
    "pg_performance.txt",   # replaced by pg_explain, pg_planner_stats, etc.
    "pg_transactions.txt",  # replaced by pg_mvcc_intro, pg_transaction_iso, etc.
    "pg_replication.txt",   # replaced by pg_ha_standby_servers, pg_ha_failover, etc.
]


def _cleanup_stale_files():
    """Remove old TOC-only files that have been replaced by content-rich subpages."""
    removed = 0
    for filename in STALE_FILES:
        filepath = os.path.join(DOCS_DIR, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            removed += 1
            print(f"  [CLEAN] Removed stale TOC file: {filename}")
    return removed


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv

    if force:
        print("Force mode: re-downloading all pages.\n")

    # Clean up stale v1 TOC files
    removed = _cleanup_stale_files()
    if removed:
        print(f"  Cleaned up {removed} stale file(s).\n")

    print(f"Downloading {len(PAGES)} documentation entries...\n")

    stats: dict[str, int] = {"postgresql": 0, "mysql": 0, "mongodb": 0}
    all_saved: list[str] = []
    failed: list[str] = []

    for name, (db_tag, url) in PAGES.items():
        result = download_page(name, db_tag, url, force=force)
        if result:
            stats[db_tag] += len(result)
            all_saved.extend(result)
        else:
            failed.append(name)
        time.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"  Total files: {len(all_saved)}")
    for db, count in stats.items():
        print(f"  {db:<15s}: {count} pages")
    if failed:
        print(f"\n  Failed ({len(failed)}): {', '.join(failed)}")
        print("  These will use previously-downloaded data if available.")
        print("  Re-run with --force to retry all pages.")
    print(f"\nCorpus saved to: {DOCS_DIR}")

