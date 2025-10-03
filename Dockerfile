# Multi-stage Docker build for ML4T Trading System
# Optimized for production deployment with Redis and FinRL support

# =============================================================================
# BUILDER STAGE: Install dependencies and build wheels
# =============================================================================
FROM python:3.11-slim as builder

# Build arguments for customization
ARG BUILD_ENV=production
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    # Build essentials for compiling Python packages
    build-essential \
    gcc \
    g++ \
    # Common libraries for financial/data packages
    libffi-dev \
    libssl-dev \
    # For numerical computing (numpy, pandas, etc.)
    libblas-dev \
    liblapack-dev \
    gfortran \
    # For TA-Lib (technical analysis library)
    wget \
    # Git for installing from repositories
    git \
    # Package config
    pkg-config \
    # Clean up to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for better dependency isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Copy requirements files
COPY requirements.txt requirements_deployment.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_deployment.txt

# Install additional ML dependencies for FinRL
RUN pip install --no-cache-dir \
    stable-baselines3==2.0.0 \
    tensorboard==2.15.1 \
    gym==0.21.0 \
    stockstats==0.6.2

# =============================================================================
# RUNTIME STAGE: Production trading environment with Redis
# =============================================================================
FROM python:3.11-slim as runtime

# Runtime build arguments
ARG APP_ENV=production
ARG APP_VERSION=latest

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=${APP_ENV} \
    APP_VERSION=${APP_VERSION} \
    PYTHONPATH=/app

# Install runtime dependencies including Redis
RUN apt-get update && apt-get install -y \
    # Runtime libraries for numerical computing
    libblas3 \
    liblapack3 \
    # SSL support for API connections
    ca-certificates \
    # For timezone data (important for trading apps)
    tzdata \
    # Redis server for caching and data persistence
    redis-server \
    # Process management
    supervisor \
    # Networking tools for health checks
    curl \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd -g 1001 trading && \
    useradd -r -u 1001 -g trading trading

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=trading:trading . /app/

# Create directories for logs, data, and Redis with proper permissions
RUN mkdir -p /app/logs /app/data /app/models /app/cache /var/lib/redis && \
    chown -R trading:trading /app /var/lib/redis

# Create supervisor configuration for managing multiple services
RUN cat > /etc/supervisor/conf.d/trading-system.conf << 'EOF'
[supervisord]
nodaemon=true
user=root

[program:redis]
command=redis-server --bind 127.0.0.1 --port 6379 --dir /var/lib/redis
user=trading
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/app/logs/redis.log

[program:trading-system]
command=python continuous_rebalancer.py
directory=/app
user=trading
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/app/logs/trading-system.log
environment=PYTHONPATH="/app"
EOF

# Create startup script
RUN cat > /app/start.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Starting ML4T Trading System"

# Start supervisor to manage Redis and trading system
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/trading-system.conf
EOF

RUN chmod +x /app/start.sh

# Switch to non-root user for the main process
USER trading

# Expose ports (Redis internal, health check)
EXPOSE 6379 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD redis-cli ping && python -c "import sys; sys.exit(0)" || exit 1

# Set default environment variables for trading app
ENV TRADING_MODE=paper \
    LOG_LEVEL=INFO \
    REDIS_HOST=localhost \
    REDIS_PORT=6379

# Switch back to root for supervisor
USER root

# Default command to run the application
CMD ["/app/start.sh"]

# Alternative commands for different deployment scenarios:
# For development: CMD ["python", "main.py"]
# For uvicorn (FastAPI): CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
# For Flask: CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]

# =============================================================================
# LABELS for container metadata
# =============================================================================
LABEL maintainer="your-email@company.com" \
      version="${APP_VERSION}" \
      description="Quantitative Trading Application for Cloud Run" \
      environment="${APP_ENV}"

# =============================================================================
# CUSTOMIZATION EXAMPLES:
# 
# 1. Change Python version:
#    FROM python:3.12-slim-bullseye as builder
#    FROM python:3.12-slim-bullseye as runtime
#
# 2. Add TA-Lib (Technical Analysis Library):
#    In builder stage, after apt-get install:
#    RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
#        tar -xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib/ && \
#        ./configure --prefix=/usr && make && make install
#
# 3. Use different web server:
#    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
#
# 4. Add private package repository:
#    RUN pip install --find-links https://your-repo.com/simple your-package
#
# 5. Multi-environment build:
#    ARG TARGET_ENV=production
#    COPY requirements-${TARGET_ENV}.txt requirements.txt
# =============================================================================