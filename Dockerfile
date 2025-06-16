# Multi-stage build for production-ready container
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Final stage
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r cognito && useradd -r -g cognito cognito

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy wheels and install
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/* && rm -rf /wheels

# Copy application
COPY . .
RUN pip install -e .

# Create directories and set permissions
RUN mkdir -p data logs models config && \
    chown -R cognito:cognito /app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    COGNITO_ENV=production \
    COGNITO_DATA_DIR=/app/data \
    COGNITO_LOG_FILE=/app/logs/cognito.log

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src.main; print('OK')" || exit 1

# Switch to non-root user
USER cognito

# Expose port
EXPOSE 8000

# Entry point
ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--help"]