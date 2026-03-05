# LegacyLens – production image for Cloud Run or any container host
FROM python:3.11-slim

WORKDIR /app

# Copy project and install dependencies (backend/ must end up as /app/backend/)
COPY pyproject.toml README.md ./
COPY backend/ backend/
RUN pip install --no-cache-dir -e .

# Copy frontend (static assets) – use production index (no Admin UI)
COPY frontend/ frontend/
COPY frontend/index.production.html frontend/index.html

# Copy BM25 index for hybrid search (built locally after ingestion)
COPY bm25_index.pkl ./

# Verify app imports (PYTHONPATH must include /app for backend)
RUN PYTHONPATH=/app python -c "from backend.api.main import app; print('Import OK')"

# Cloud Run sets PORT; default 8080 for local runs
ENV PORT=8080
ENV PYTHONPATH=/app
ENV FRONTEND_DIR=/app/frontend
EXPOSE 8080

# Run uvicorn (PYTHONPATH=/app required for backend import)
CMD ["sh", "-c", "exec python -m uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8080} --loop asyncio"]
