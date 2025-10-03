# ML4T Trading System - Docker Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the ML4T Trading System using Docker and Docker Compose. The system includes automated trading with FinRL, sentiment analysis, and real-time market data processing.

## Architecture

The dockerized system consists of:
- **Trading System**: Main application with FinRL and sentiment analysis
- **Redis**: Caching and data persistence
- **Development Environment**: Jupyter notebooks for development
- **Monitoring Dashboard**: Performance monitoring (optional)

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB+ RAM
- 20GB+ storage
- Internet connection for market data

## Quick Start

1. **Clone and Setup**
   ```bash
   git clone <your-repo-url>
   cd trading_system
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. **Start Services**
   ```bash
   # Production deployment
   docker-compose up -d

   # Development with Jupyter
   docker-compose --profile dev up -d

   # With monitoring dashboard
   docker-compose --profile monitoring up -d
   ```

4. **Verify Deployment**
   ```bash
   docker-compose ps
   docker-compose logs trading-system
   ```

## Environment Configuration

### Required API Keys

1. **Alpaca Trading** (Required)
   - Get from: https://app.alpaca.markets/paper/dashboard/overview
   - Set: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`

2. **OpenAI API** (Optional, for enhanced sentiment)
   - Get from: https://platform.openai.com/api-keys
   - Set: `OPENAI_API_KEY`

3. **Reddit API** (Optional, for social sentiment)
   - Get from: https://www.reddit.com/prefs/apps
   - Set: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`

### Configuration Examples

**Paper Trading (Recommended for testing):**
```bash
TRADING_MODE=paper
ALPACA_BASE_URL=https://paper-api.alpaca.markets
MAX_POSITION_SIZE=0.1
```

**Live Trading (Use with extreme caution):**
```bash
TRADING_MODE=live
ALPACA_BASE_URL=https://api.alpaca.markets
MAX_POSITION_SIZE=0.05
```

## Docker Commands

### Basic Operations

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f trading-system

# Restart specific service
docker-compose restart trading-system

# Update and rebuild
docker-compose build --no-cache
docker-compose up -d
```

### Development Commands

```bash
# Start development environment with Jupyter
docker-compose --profile dev up -d

# Access Jupyter notebook
open http://localhost:8888

# Run commands inside container
docker-compose exec trading-system python continuous_rebalancer.py

# Debug shell access
docker-compose exec trading-system bash
```

### Monitoring Commands

```bash
# Check system health
docker-compose exec trading-system redis-cli ping

# Monitor resource usage
docker stats

# View trading logs
docker-compose exec trading-system tail -f /app/logs/trading-system.log

# Access performance dashboard
open http://localhost:8082
```

## Data Persistence

### Volume Mounts

- `./data:/app/data` - Trading data and state
- `./logs:/app/logs` - Application logs
- `./models:/app/models` - FinRL models
- `redis_data:/data` - Redis persistence

### Backup Strategy

```bash
# Backup trading data
docker-compose exec trading-system tar -czf /tmp/trading-backup.tar.gz /app/data
docker cp trading-system:/tmp/trading-backup.tar.gz ./backups/

# Backup Redis data
docker-compose exec redis redis-cli BGSAVE
docker cp trading-redis:/data/dump.rdb ./backups/
```

## Scaling and Performance

### Resource Limits

```yaml
# docker-compose.override.yml
services:
  trading-system:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
```

### Performance Tuning

1. **Memory Settings**
   - Increase Redis `maxmemory` for larger datasets
   - Adjust Python `MAX_WORKERS` based on CPU cores

2. **Network Optimization**
   - Use Redis clustering for high-frequency trading
   - Configure connection pooling

3. **Storage Optimization**
   - Use SSD storage for `/app/data` volume
   - Enable Redis persistence tuning

## Security Considerations

### Container Security

1. **API Keys**
   - Never commit `.env` file
   - Use Docker secrets in production
   - Rotate keys regularly

2. **Network Security**
   - Use internal Docker networks
   - Expose only necessary ports
   - Enable firewall rules

3. **Resource Isolation**
   - Run containers as non-root user
   - Use read-only filesystems where possible
   - Implement resource limits

### Production Hardening

```bash
# Use Docker secrets
echo "your_api_key" | docker secret create alpaca_api_key -

# Run with security options
docker-compose up -d --security-opt no-new-privileges:true
```

## Troubleshooting

### Common Issues

1. **Connection Errors**
   ```bash
   # Check network connectivity
   docker-compose exec trading-system curl -I https://paper-api.alpaca.markets
   
   # Verify API keys
   docker-compose exec trading-system python -c "from tools.alpaca_client import alpaca_client; print(alpaca_client.get_account_info())"
   ```

2. **Memory Issues**
   ```bash
   # Monitor memory usage
   docker stats trading-system
   
   # Increase memory limits
   # Edit docker-compose.yml deploy.resources.limits.memory
   ```

3. **Redis Connection Issues**
   ```bash
   # Test Redis connectivity
   docker-compose exec trading-system redis-cli -h redis ping
   
   # Check Redis logs
   docker-compose logs redis
   ```

### Log Analysis

```bash
# Trading system logs
docker-compose logs -f trading-system | grep ERROR

# Redis logs
docker-compose logs redis

# System performance
docker-compose exec trading-system htop
```

## Deployment Environments

### Development

```bash
# Start development stack
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Features:
# - Jupyter notebooks
# - Hot reloading
# - Debug logging
# - Development tools
```

### Staging

```bash
# Start staging environment
docker-compose -f docker-compose.yml -f docker-compose.staging.yml up -d

# Features:
# - Paper trading only
# - Monitoring enabled
# - Performance testing
# - Load testing
```

### Production

```bash
# Start production environment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Features:
# - Resource limits
# - Health checks
# - Backup automation
# - Security hardening
```

## Monitoring and Alerts

### Health Checks

The system includes built-in health checks:
- **Trading System**: API connectivity and Redis connection
- **Redis**: Service availability and memory usage

### Metrics Collection

```bash
# View system metrics
curl http://localhost:8080/metrics

# Trading performance
curl http://localhost:8080/api/performance

# System health
curl http://localhost:8080/health
```

### Alert Configuration

Configure alerts in `.env`:
```bash
EMAIL_ENABLED=true
SMTP_USER=alerts@yourcompany.com
NOTIFICATION_EMAIL=trading-team@yourcompany.com
```

## Support and Maintenance

### Regular Maintenance

1. **Daily**
   - Check logs for errors
   - Verify trading performance
   - Monitor resource usage

2. **Weekly**
   - Backup trading data
   - Update market data cache
   - Review trading metrics

3. **Monthly**
   - Update Docker images
   - Rotate API keys
   - Performance optimization

### Getting Help

1. **Logs**: Check container logs for detailed error information
2. **Health Checks**: Use built-in health endpoints
3. **Documentation**: Refer to inline code documentation
4. **Community**: Trading system community forums

---

**⚠️ Important Disclaimers:**
- This system is for educational purposes
- Always test in paper trading mode first
- Never invest more than you can afford to lose
- Past performance does not guarantee future results