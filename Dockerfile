# ── Stage 1: build deps ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Non-root user
RUN adduser --disabled-password --gecos "" appuser

# Copy installed packages and source
COPY --from=builder /install /usr/local
COPY . .

# Persistent data lives in a Docker volume
VOLUME ["/app/data", "/app/cache"]

ENV DB_PATH=/app/data/retail.duckdb \
    CACHE_META=/app/cache/cache_meta.json \
    CACHE_INDEX=/app/cache/cache_index.faiss

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["python", "main.py"]
