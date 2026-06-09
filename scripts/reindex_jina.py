# reindex_jina.py
import os
from dotenv import load_dotenv
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores import Chroma
from backend.ingest import load_documents, chunk_documents

load_dotenv()

# --- Configuration ---
PERSIST_DIR = "./chroma_db_local"  # make sure this matches your setup
COLLECTION_NAME = "multi_db_docs"

# 1. Initialize Jina Embeddings
embeddings = JinaEmbeddings(
    jina_api_key=os.getenv("JINA_API_KEY"),  # Your key from step 1
    model_name="jina-embeddings-v4",
    session=None
)

# 2. Load and chunk your documents
docs = load_documents("./data/db_docs")
chunks = chunk_documents(docs)
print(f"Loaded {len(chunks)} chunks from documents.")

# 3. Connect to Chroma and replace the collection
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)

# Delete existing collection to start fresh
vectorstore.delete_collection()
# Recreate it
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION_NAME
)
print("Re-indexing complete!")