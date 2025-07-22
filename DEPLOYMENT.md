# 🚀 AWS App Runner Deployment Guide
## LLM-Enhanced Trading System Production Deployment

### 📋 Overview

This guide provides step-by-step instructions for deploying the LLM-Enhanced Trading System to AWS App Runner, a fully managed service for containerized applications.

### 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GitHub Repo   │───▶│   AWS App       │───▶│   Trading       │
│   (Source Code) │    │   Runner        │    │   System        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │
                               ▼
                       ┌─────────────────┐
                       │   Secrets       │
                       │   Manager       │
                       └─────────────────┘
```

### 🔧 Prerequisites

Before deployment, ensure you have:

1. **AWS Account** with appropriate permissions
2. **GitHub Repository** with the trading system code
3. **API Keys** for required services:
   - Alpaca Trading API (Paper Trading)
   - OpenAI API Key (recommended)
   - Anthropic API Key (optional backup)
   - Additional data source APIs (optional)

### 📦 Container Configuration

The system is containerized with:
- **Base Image**: Python 3.11 slim
- **Web Framework**: FastAPI with Uvicorn
- **Port**: 8080 (configurable)
- **Health Checks**: Built-in `/health` endpoint
- **Security**: Non-root user, minimal attack surface

### 🚀 Deployment Steps

#### Step 1: Prepare GitHub Repository

1. Ensure your repository contains all required files:
   ```
   ├── Dockerfile
   ├── apprunner.yaml
   ├── app.py
   ├── requirements.txt
   ├── main.py
   ├── agents/
   ├── tools/
   ├── config/
   └── prompts/
   ```

2. Verify the repository is public or set up GitHub access for AWS

#### Step 2: Create AWS App Runner Service

1. **Log into AWS Console** and navigate to App Runner
2. **Create Service**:
   - Choose "Source code repository"
   - Connect to GitHub
   - Select your repository
   - Choose "Automatic deployments"

3. **Configure Build**:
   - Runtime: Python 3.11
   - Build command: `pip install -r requirements.txt`
   - Start command: `python app.py`
   - Port: `8080`

4. **Set Environment Variables**:
   ```bash
   TRADING_MODE=paper
   LOG_LEVEL=INFO
   PORT=8080
   MAX_PORTFOLIO_RISK=0.05
   MAX_POSITION_SIZE=0.15
   ```

#### Step 3: Configure Secrets

**⚠️ CRITICAL: Store sensitive API keys in AWS Secrets Manager or App Runner environment variables**

Required secrets:
```bash
ALPACA_API_KEY=your_alpaca_paper_api_key
ALPACA_SECRET_KEY=your_alpaca_paper_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
OPENAI_API_KEY=your_openai_api_key
```

Optional secrets:
```bash
ANTHROPIC_API_KEY=your_anthropic_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
FINNHUB_API_KEY=your_finnhub_key
NEWS_API_KEY=your_news_api_key
```

#### Step 4: Configure Service Settings

1. **Service Configuration**:
   - Service name: `llm-trading-system`
   - CPU: 1 vCPU
   - Memory: 2 GB
   - Auto scaling: 1-10 instances

2. **Health Check**:
   - Health check path: `/health`
   - Health check interval: 30 seconds
   - Health check timeout: 30 seconds
   - Healthy threshold: 2
   - Unhealthy threshold: 3

3. **Security**:
   - Custom domain (optional)
   - HTTPS only
   - WAF protection (recommended)

#### Step 5: Deploy and Monitor

1. **Deploy Service**: Click "Create & Deploy"
2. **Monitor Deployment**: Check build and deployment logs
3. **Verify Health**: Access `/health` endpoint
4. **Test System**: Use `/docs` for interactive API testing

### 📊 Monitoring and Observability

#### Built-in Endpoints

- **Health Check**: `GET /health`
- **System Status**: `GET /status`  
- **Metrics**: `GET /metrics`
- **API Documentation**: `GET /docs`

#### Key Metrics to Monitor

1. **Application Health**:
   - HTTP response times
   - Error rates
   - CPU and memory usage

2. **Trading System Metrics**:
   - Portfolio value
   - Number of positions
   - Trading cycle success rate
   - LLM API usage and costs

3. **Infrastructure Metrics**:
   - Container restarts
   - Auto-scaling events
   - Network latency

### 🔒 Security Best Practices

#### Production Security Checklist

- ✅ **Paper Trading Only**: Never use live trading keys
- ✅ **Secrets Management**: Store API keys in AWS Secrets Manager
- ✅ **HTTPS Only**: Force SSL/TLS encryption
- ✅ **Network Security**: Use VPC and security groups
- ✅ **WAF Protection**: Enable Web Application Firewall
- ✅ **Access Logging**: Enable CloudTrail and access logs
- ✅ **Regular Updates**: Keep dependencies updated
- ✅ **Monitoring**: Set up CloudWatch alarms

#### Environment-Specific Security

```bash
# Production Environment Variables
TRADING_MODE=paper              # ⚠️ NEVER set to 'live'
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
SECURE_SSL_REDIRECT=true
SECURE_HSTS_SECONDS=31536000
```

### 🚨 Risk Management

#### Built-in Safety Controls

1. **Paper Trading Lock**: Hard-coded to paper trading only
2. **Position Size Limits**: Maximum 15% per position
3. **Daily Loss Limits**: Maximum 5% daily loss
4. **Risk Circuit Breakers**: Automatic trading halt on violations
5. **Emergency Procedures**: Immediate position closure on critical alerts

#### Monitoring Alerts

Set up CloudWatch alarms for:
- High error rates (>5%)
- Memory usage (>80%)
- CPU usage (>80%)
- Failed health checks
- LLM API quota exceeded

### 💰 Cost Optimization

#### AWS App Runner Costs

- **Compute**: ~$0.064 per vCPU hour + $0.007 per GB RAM hour
- **Requests**: $0.40 per million requests
- **Data Transfer**: $0.09 per GB

#### Estimated Monthly Costs

For a basic deployment:
- **App Runner**: ~$50-100/month (1 vCPU, 2GB RAM)
- **LLM APIs**: Variable based on usage
- **Data Transfer**: ~$5-10/month

### 📈 Scaling and Performance

#### Horizontal Scaling

App Runner automatically scales based on:
- CPU utilization
- Memory usage
- Request volume
- Custom metrics

#### Performance Optimization

1. **Container Optimization**:
   - Multi-stage Docker build
   - Minimal base image
   - Efficient dependency management

2. **Application Performance**:
   - Async/await patterns
   - Connection pooling
   - Caching strategies
   - Background task processing

### 🛠️ Troubleshooting

#### Common Issues

1. **Build Failures**:
   ```bash
   # Check requirements.txt dependencies
   # Verify Dockerfile syntax
   # Check build logs in App Runner console
   ```

2. **Health Check Failures**:
   ```bash
   # Verify /health endpoint returns 200
   # Check application startup time
   # Review container logs
   ```

3. **API Key Issues**:
   ```bash
   # Verify secrets are properly set
   # Check API key permissions
   # Test connections independently
   ```

4. **Performance Issues**:
   ```bash
   # Monitor CPU/memory usage
   # Check for memory leaks
   # Review slow query logs
   ```

### 📞 Support and Maintenance

#### Regular Maintenance Tasks

1. **Weekly**:
   - Review system logs
   - Check portfolio performance
   - Monitor API usage costs

2. **Monthly**:
   - Update dependencies
   - Review security patches
   - Analyze performance metrics

3. **Quarterly**:
   - Security audit
   - Disaster recovery testing
   - Cost optimization review

#### Getting Help

- **AWS Support**: For infrastructure issues
- **Application Logs**: Check `/logs` endpoint or CloudWatch
- **Health Monitoring**: Use built-in `/health` and `/metrics` endpoints

### 🎯 Production Readiness Checklist

Before going live, verify:

- ✅ All API keys are properly configured
- ✅ Paper trading mode is enforced
- ✅ Health checks are passing
- ✅ Monitoring and alerting are set up
- ✅ Security best practices are implemented
- ✅ Backup and recovery procedures are documented
- ✅ Performance testing is completed
- ✅ Cost monitoring is configured

---

## 🚀 Ready for Production!

Your LLM-Enhanced Trading System is now ready for deployment to AWS App Runner. The containerized architecture provides scalability, reliability, and security for production use while maintaining all safety controls.

**Remember**: This system is designed for paper trading only. Never deploy with live trading credentials.