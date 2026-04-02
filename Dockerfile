# ============================================================
# ACD Framework — Multi-stage Production Dockerfile
# ============================================================
# Stage 1: Builder — installs deps and resolves wheels
# Stage 2: Runtime — slim image with only what's needed
# ============================================================

# ── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for building wheels (Neo4j driver, bcrypt, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into /install
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="ACD Framework <acd@msrit.edu>"
LABEL description="Autonomous Cyber Defence Framework — API Server"
LABEL version="1.0.0"

# Runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    libffi8 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd -r acd && useradd -r -g acd acd

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY --chown=acd:acd . .

# Create data directories with correct ownership
RUN mkdir -p \
    data/checkpoints \
    data/logs \
    data/experiences \
    data/kg_cache \
    data/embeddings \
    data/incidents \
    && chown -R acd:acd data/

# Switch to non-root user
USER acd

# Environment defaults (override via docker-compose or -e flags)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=1 \
    LOG_FORMAT=json \
    LOG_LEVEL=INFO \
    HF_HOME=/app/data/kg_cache/huggingface \
    MPLCONFIGDIR=/app/data/logs/matplotlib

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Production startup command
CMD ["uvicorn", "api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--no-access-log"]