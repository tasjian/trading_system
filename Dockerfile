# Multi-stage build for LLM-Enhanced Trading System
# Production-optimized container for AWS App Runner deployment

# Stage 1: Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_ENV=production
ARG APP_VERSION=1.0.0

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create build directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install dependencies
RUN pip install --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Production runtime
FROM python:3.11-slim as runtime

# Set production labels
LABEL maintainer="Trading System Team" \
      version="${APP_VERSION}" \
      description="LLM-Enhanced AI Trading System" \
      environment="production"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r trading && useradd -r -g trading -s /bin/bash trading

# Set up application directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=trading:trading . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/data /app/temp && \
    chown -R trading:trading /app && \
    chmod -R 755 /app

# Remove unnecessary files for production
RUN rm -rf tests/ docs/ *.md CLEANUP_SUMMARY.md requirements.txt setup.py && \
    find . -name "*.pyc" -delete && \
    find . -name "__pycache__" -type d -exec rm -rf {} + || true

# Set environment variables for production
ENV PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRADING_MODE=paper \
    LOG_LEVEL=INFO \
    PORT=8080

# Expose port for AWS App Runner
EXPOSE 8080

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Switch to non-root user
USER trading

# Create startup script
COPY --chown=trading:trading <<'EOF' /app/start.sh
#!/bin/bash
set -e

echo "🚀 Starting LLM-Enhanced Trading System in Production Mode"
echo "=================================================="
echo "Environment: ${TRADING_MODE:-paper}"
echo "Log Level: ${LOG_LEVEL:-INFO}"
echo "Port: ${PORT:-8080}"
echo "Timestamp: $(date)"

# Validate required environment variables
if [ -z "$ALPACA_API_KEY" ]; then
    echo "❌ ERROR: ALPACA_API_KEY environment variable is required"
    exit 1
fi

if [ -z "$ALPACA_SECRET_KEY" ]; then
    echo "❌ ERROR: ALPACA_SECRET_KEY environment variable is required"
    exit 1
fi

# Check if we have at least one LLM API key
if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "❌ ERROR: At least one LLM API key (OPENAI_API_KEY or ANTHROPIC_API_KEY) is required"
    exit 1
fi

echo "✅ Environment validation passed"

# Start the application
echo "🔄 Starting web server..."
exec python app.py
EOF

RUN chmod +x /app/start.sh

# Set the startup command
CMD ["/app/start.sh"]