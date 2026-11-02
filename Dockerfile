# One image serves the API and the React UI from the same origin.
# Runs on Render, Hugging Face Spaces, Fly.io or any Docker host ($PORT aware).

# --- 1. frontend --------------------------------------------------------------
FROM node:24-alpine AS frontend
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
# Same-origin API: the UI calls /api/v1/* on whatever host serves it.
RUN REACT_APP_API_URL= npm run build

# --- 2. backend ---------------------------------------------------------------
FROM python:3.11-slim

# false → lean image without torch for 512MB hosts (Render passes env vars as build args).
ARG RERANKER_ENABLED=true

ENV RERANKER_ENABLED=${RERANKER_ENABLED} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DB_FILE=/app/data/state/dbrain.db \
    PORT=8000

RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY requirements.txt requirements-reranker.txt ./
RUN pip install -r requirements.txt
# CPU-only torch; the default Linux wheel drags in ~4GB of CUDA.
RUN if [ "$RERANKER_ENABLED" = "true" ]; then \
      pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
      && pip install -r requirements-reranker.txt; \
    fi

COPY backend/ backend/
COPY scripts/reindex.py scripts/reindex.py
COPY data/db_docs/ data/db_docs/
COPY --from=frontend /web/build frontend/build
# app writes the index, the BM25 pickle and SQLite state under /app.
RUN mkdir -p data/state && chown -R app:app /app

USER app
# Bake the index and models into the image: boot is fast and needs no network.
RUN python scripts/reindex.py \
 && if [ "$RERANKER_ENABLED" = "true" ]; then \
      python -c "from backend import reranker; reranker.get_reranker()"; \
    fi

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
