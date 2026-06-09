import os
import re
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# DOCS_DIR points to downloaded database docs
DOCS_DIR = os.getenv("DOCS_DIR", "./data/db_docs")

# Mapping of db_tag from headers to human-readable label
DB_LABELS = {
    "postgresql": "PostgreSQL",
    "mysql": "MySQL",
    "mongodb": "MongoDB",
}


def parse_metadata_header(content: str) -> tuple[dict, str]:
    """
    Parse the metadata header written by download_corpus.py.
    Format:
      DB_TAG: <tag>
      SOURCE_URL: <url>
      DOC_NAME: <name>
      ---
      <body>
    Returns (metadata_dict, remaining_content).
    """
    metadata = {}
    lines = content.split("\n")
    body_start = 0

    for i, line in enumerate(lines):
        if line.strip() == "---":
            body_start = i + 1
            break
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip().lower()] = value.strip()

    body = "\n".join(lines[body_start:])
    return metadata, body


def load_documents(docs_dir: str) -> list[Document]:
    """
    Load all .txt files, parse metadata headers, and return LangChain Documents.
    Each document is enriched with:
      - db: database identifier ("postgresql" | "mysql" | "mongodb")
      - db_label: human-readable label
      - doc_name: specific document page name
      - source: filename on disk
      - source_url: original url reference
    """
    documents = []
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        print(f"Directory {docs_dir} does not exist. Please run download_corpus.py first.")
        return []

    for file_path in sorted(docs_path.rglob("*.txt")):
        try:
            raw = file_path.read_text(encoding="utf-8")
            file_meta, body = parse_metadata_header(raw)

            db_tag = file_meta.get("db_tag", "unknown")
            doc_name = file_meta.get("doc_name", file_path.stem)
            source_url = file_meta.get("source_url", "")

            # Ensure we have a reasonable amount of content
            if len(body.strip()) < 100:
                print(f"  Skipping {file_path.name}: too short after parsing")
                continue

            doc = Document(
                page_content=body,
                metadata={
                    "db": db_tag,
                    "db_label": DB_LABELS.get(db_tag, db_tag),
                    "doc_name": doc_name,
                    "source": file_path.name,
                    "source_url": source_url,
                },
            )
            documents.append(doc)

        except Exception as e:
            print(f"  Error loading {file_path.name}: {e}")

    # Print summary by database
    from collections import Counter
    db_counts = Counter(d.metadata["db"] for d in documents)
    print(f"\nLoaded {len(documents)} documents:")
    for db, count in sorted(db_counts.items()):
        print(f"  {db:<15s}: {count} docs")

    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Chunk documents using RecursiveCharacterTextSplitter.
    Each chunk preserves the original document metadata.

    Chunk sizing rationale:
    - 700 characters: enough context for one complete concept/explanation
    - 100 character overlap: prevents cutting across important sentences
    - separators: double-newline, single-newline, sentence periods, spaces
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks = []
    chunk_id = 0

    for doc in documents:
        chunks = splitter.split_documents([doc])
        for chunk in chunks:
            chunk.metadata["chunk_id"] = chunk_id
            chunk.metadata["chunk_preview"] = chunk.page_content[:80].replace("\n", " ")
            all_chunks.append(chunk)
            chunk_id += 1

    # Print summary by database
    from collections import Counter
    db_counts = Counter(c.metadata["db"] for c in all_chunks)
    print(f"\nCreated {len(all_chunks)} total chunks:")
    for db, count in sorted(db_counts.items()):
        print(f"  {db:<15s}: {count} chunks")

    return all_chunks


if __name__ == "__main__":
    print(f"Starting document ingestion from: {DOCS_DIR}")
    docs = load_documents(DOCS_DIR)
    if docs:
        chunks = chunk_documents(docs)
        print(f"\nSample chunks:")
        # Show one chunk per database to verify
        seen_dbs = set()
        for c in chunks:
            db = c.metadata["db"]
            if db not in seen_dbs:
                print(f"\n  [{db}] {c.metadata['doc_name']}")
                print(f"  Preview: {c.metadata['chunk_preview']}")
                print(f"  Length:  {len(c.page_content)} chars")
                seen_dbs.add(db)
            if len(seen_dbs) == 3:
                break
