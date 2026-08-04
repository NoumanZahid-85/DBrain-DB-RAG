# DBrain Module 7 RAG — backend image (fresh-demo scope requirement)
# Multi-stage: build deps first so runtime layer stays lean enough for the
# 512MB free tier while keeping the sentence-transformers reranker available.
FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Reindex is skipped at boot (data is baked in); render.yaml uses the same
# module path so local/cloud behaviour matches.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]