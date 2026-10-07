# Multi-stage Dockerfile for Apex Fintech Platform
# Stage 1: Build dependencies
# Stage 2: Production runtime

# ── Build Stage ────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# ── Production Stage ───────────────────────────────────────────────
FROM python:3.12-slim AS production

WORKDIR /app

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy only necessary artifacts from builder
COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy source code
COPY src/ ./src/

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.devops.engine import DevOpsEngine; e = DevOpsEngine(); print(e.health_check())" || exit 1

# Default command
CMD ["python", "-c", "from src.devops.engine import DevOpsEngine; print('DevOps Engine ready')"]
