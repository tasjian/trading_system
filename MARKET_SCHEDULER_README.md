# Market-Aware Trading System

## Overview

The market-aware trading system automatically manages timing and execution of the trading pipeline based on market hours. It ensures optimal resource usage and trading execution by running different strategies during different market phases.

## System Architecture

### Trading Phases

1. **🌅 PRE-MARKET (9:00 AM - 9:30 AM ET)**
   - Runs **once** 30 minutes before market open
   - Performs initial portfolio preparation
   - Analyzes overnight news and market conditions
   - Sets up positions for the trading day

2. **📈 TRADING HOURS (9:30 AM - 4:00 PM ET)**
   - Runs **continuously** every 15 minutes
   - Full rebalancing pipeline with dynamic stock discovery
   - Real-time sentiment analysis and signal generation
   - Active order execution and portfolio management

3. **🌙 AFTER-HOURS (After 4:00 PM ET)**
   - Runs **once** for end-of-day review
   - Portfolio performance analysis
   - Risk assessment and position cleanup
   - Preparation for next trading day

4. **⏰ WAITING (All other times)**
   - System hibernates until next pre-market session
   - Minimal resource usage
   - Automatic restart before next trading day

## Quick Start

### Start the System
```bash
python start_trading_system.py
```

### Check Status
```bash
python check_trading_status.py
```

### Test Scheduler Logic
```bash
python test_market_scheduler.py
```

## Features

### ✅ Intelligent Timing
- Automatically starts 30 minutes before market open
- Continuous operation during trading hours only
- Hibernates during non-trading periods

### ✅ Dynamic Stock Discovery
- No hardcoded stock lists
- Universe filtering reduces 11,000+ stocks to ~240 actionable candidates
- LLM-driven portfolio construction with sentiment integration

### ✅ Comprehensive Data Sources
- News sentiment analysis
- Social media sentiment (Reddit, Twitter)
- Earnings data integration
- Real-time market data

### ✅ Risk Management
- Circuit breakers for market volatility
- Position size limits
- Diversification controls
- Stop-loss and target price setting

### ✅ Market Hours Awareness
- Stages orders when market is closed
- Adjusts execution strategy based on market status
- Handles pre-market and after-hours periods appropriately

## Configuration

### Key Parameters
- **Pre-market start**: 30 minutes before market open (9:00 AM ET)
- **Trading interval**: 15 minutes during market hours
- **After-hours runs**: Maximum 1 per day
- **Max positions**: 35 stocks maximum
- **Universe filtering**: 99.7% reduction (11k+ → 240 stocks)

### Environment Variables
Ensure these are set in your `.env` file:
```
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # for paper trading
```

## System Status Examples

### Pre-Market Phase
```
📅 Trading Phase: PRE_MARKET
   Current Time: 09:15 EDT
   ⏰ Market opens in: 15 minutes
💡 Recommendations: Run initial portfolio preparation
```

### Trading Phase
```
📅 Trading Phase: TRADING
   Current Time: 11:30 EDT
   ⏰ Market closes in: 270 minutes
💡 Recommendations: Continuous rebalancing should be active
```

### After-Hours Phase
```
📅 Trading Phase: AFTER_HOURS
   Current Time: 17:30 EDT
💡 Recommendations: Consider running end-of-day review if not done yet
```

## Monitoring

### Log Files
- `logs/market_scheduler.log` - Scheduler events and phase transitions
- `logs/continuous_rebalancer.log` - Rebalancing pipeline execution
- `data/scheduler_state.json` - Persistent scheduler state
- `data/continuous_rebalancer_state.json` - Rebalancer performance metrics

### Health Checks
The system automatically tracks:
- Uptime and run statistics
- Success/failure rates
- API rate limit monitoring
- Circuit breaker status
- Portfolio performance metrics

## Error Handling

### Graceful Degradation
- Falls back to Ollama LLM if Anthropic API fails
- Uses cached data when APIs are rate-limited
- Continues operation with reduced functionality if non-critical components fail

### Automatic Recovery
- Exponential backoff for failed operations
- Circuit breakers prevent cascade failures
- State persistence allows resumption after crashes

## Performance Metrics

### Efficiency Gains
- **99.7% processing reduction**: 11,332 → 240 stocks via universe filtering
- **Intelligent scheduling**: Runs only when needed based on market hours
- **Batch processing**: Parallel sentiment analysis and data collection
- **Caching**: Reduces API calls through intelligent data reuse

### Resource Usage
- **Memory**: Optimized for long-running operation
- **API calls**: Rate-limited and cached to prevent quota exhaustion
- **CPU**: Intensive operations scheduled during non-market hours when possible

## Troubleshooting

### Common Issues

1. **System not starting at expected time**
   - Check system timezone matches Eastern Time
   - Verify market calendar API is accessible
   - Check log files for timing calculation errors

2. **No trades being executed**
   - Verify Alpaca API credentials
   - Check if market is open
   - Review circuit breaker status
   - Examine risk limits and position sizing

3. **High API usage**
   - Review rate limiting settings
   - Check if universe filtering is working properly
   - Monitor sentiment analysis batch sizes

### Support Commands
```bash
# Check current system status
python check_trading_status.py

# Test scheduler logic without running trades  
python test_market_scheduler.py

# View recent logs
tail -f logs/market_scheduler.log
tail -f logs/continuous_rebalancer.log
```

## Integration with Existing System

The market-aware scheduler is fully compatible with existing system components:
- Uses the same `TradingWorkflow` and agent architecture
- Leverages existing `continuous_rebalancer.py` during trading hours
- Maintains all risk controls and portfolio management features
- Preserves dynamic stock discovery and LLM-driven decision making

Simply replace manual execution with `python start_trading_system.py` for fully automated, market-aware operation.