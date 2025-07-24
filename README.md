# ML4T Trading System

A comprehensive, AI-powered trading system built for algorithmic trading. The system provides professional-grade market analysis, automated trading execution, portfolio management, and strategy implementation.

## 🚀 Features

### Core Capabilities
- **Unified Market Intelligence**: Multi-source data aggregation with premium API integration
- **Automated Trading**: Smart order execution with risk controls
- **Portfolio Management**: Comprehensive portfolio analytics and rebalancing
- **Pairs Trading**: Statistical arbitrage with cointegration testing
- **Risk Management**: Multi-layer risk controls and position sizing
- **Performance Analytics**: Real-time performance tracking and reporting

### Data Sources
- **Premium APIs**: Alpha Vantage, Finnhub, Financial Modeling Prep, News API
- **Fallback Systems**: Yahoo Finance with multiple retrieval methods
- **Real-time Data**: Live market data with intelligent caching
- **News Sentiment**: Automated sentiment analysis from news sources

## 📋 Quick Start

### Installation
```bash
# Clone repository
git clone <repository-url>
cd trading_system

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Basic Usage
```bash
# Run full automated trading cycle
python simplified_main.py

# Run interactive mode
python simplified_main.py --interactive

# Run specific components
python -c "
import asyncio
from core import analyze_stock
result = asyncio.run(analyze_stock('AAPL'))
print(f'Signal: {result.signal.value}, Score: {result.score:.2f}')
"
```

## 🏗️ Architecture

### Simplified Structure
```
trading_system/
├── core/                    # Unified core components
│   ├── market_intelligence.py  # Market data & analysis
│   └── trading_engine.py       # Trading execution & portfolio management
├── simplified_main.py       # Main application
├── config/                  # Configuration
├── strategies/              # Trading strategies
│   └── pairs_trading.py
├── tools/                   # Trading tools & clients
│   └── alpaca_client.py
├── notifications/           # Email notifications
└── enhanced_market_screener.py  # Stock selection
```

### Core Components

#### Market Intelligence (`core/market_intelligence.py`)
Unified system for market data and analysis:
- **Multi-source Data**: Aggregates from premium APIs with fallbacks
- **Technical Analysis**: RSI, moving averages, volume analysis
- **Fundamental Analysis**: P/E ratios, growth metrics, financial health
- **Sentiment Analysis**: News-based market sentiment
- **Risk Assessment**: Volatility and risk scoring

#### Trading Engine (`core/trading_engine.py`)
Unified trading execution and portfolio management:
- **Order Execution**: Smart order routing with risk checks
- **Portfolio Management**: Position tracking and rebalancing
- **Strategy Execution**: Support for multiple trading strategies
- **Risk Controls**: Position sizing and risk limits
- **Performance Tracking**: Real-time performance metrics

## 📊 Usage Examples

### Market Analysis
```python
from core import analyze_stock, analyze_portfolio

# Analyze single stock
analysis = await analyze_stock('AAPL')
print(f"Signal: {analysis.signal.value}")
print(f"Confidence: {analysis.confidence.value}")
print(f"Target: ${analysis.target_price:.2f}")

# Analyze portfolio
symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
analyses = await analyze_portfolio(symbols)
for analysis in analyses:
    print(f"{analysis.symbol}: {analysis.signal.value} ({analysis.score:.2f})")
```

### Trading Execution
```python
from core import execute_signal, get_portfolio_metrics

# Execute trading signal
analysis = await analyze_stock('AAPL')
if analysis.confidence.value in ['high', 'very_high']:
    order = await execute_signal(analysis)
    print(f"Executed: {order.side} {order.quantity} {order.symbol}")

# Get portfolio metrics
metrics = await get_portfolio_metrics()
print(f"Portfolio Value: ${metrics.total_value:,.2f}")
print(f"Day P&L: ${metrics.day_pnl:,.2f}")
```

### Pairs Trading
```python
from strategies.pairs_trading import PairsTradingStrategy

strategy = PairsTradingStrategy()

# Find suitable pairs
pairs = strategy.find_pairs(['AAPL', 'MSFT', 'GOOGL', 'AMZN'])

# Generate trading signals
for pair in pairs:
    signals = strategy.generate_signals(pair)
    for signal in signals:
        print(f"Pair: {pair['symbols']} | Signal: {signal['action']} | Z-score: {signal['z_score']:.2f}")
```

### Portfolio Rebalancing
```python
from core import rebalance_portfolio

# Define target allocation
target_allocation = {
    'AAPL': 0.25,   # 25%
    'MSFT': 0.25,   # 25%
    'GOOGL': 0.25,  # 25%
    'SPY': 0.25     # 25%
}

# Execute rebalancing
orders = await rebalance_portfolio(target_allocation)
print(f"Rebalancing complete: {len(orders)} orders executed")
```

## ⚙️ Configuration

### Environment Variables
```bash
# Trading API (Alpaca - Paper Trading)
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Market Data APIs
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
FINNHUB_API_KEY=your_finnhub_key
FMP_API_KEY=your_fmp_key
NEWS_API_KEY=your_news_api_key

# Email Notifications
GMAIL_EMAIL=your_email@gmail.com
GMAIL_APP_PASSWORD=your_app_password

# Trading Parameters
MAX_POSITION_SIZE=0.15          # 15% max position size
STOP_LOSS_PERCENT=0.08          # 8% stop loss
MIN_REBALANCE_THRESHOLD=0.05    # 5% rebalance threshold
```

### Risk Parameters
The system includes comprehensive risk management:
- **Position Sizing**: Maximum 15% per position
- **Stop Losses**: Automatic 8% stop losses
- **Portfolio Risk**: Maximum 2% daily risk
- **Diversification**: Cross-industry allocation requirements
- **Buying Power**: Automatic buying power validation

## 🔧 API Integration

### Supported Data Providers
1. **Alpha Vantage**: Technical indicators, fundamentals
2. **Finnhub**: Real-time quotes, analyst data, insider trading
3. **Financial Modeling Prep**: Detailed financial metrics
4. **News API**: Real-time news sentiment analysis
5. **Yahoo Finance**: Fallback data provider
6. **Alpaca**: Trading execution and account management

### Fallback System
The system uses intelligent fallback ordering:
```
Market Data Request
    ↓
Premium APIs (Finnhub, FMP, Alpha Vantage)
    ↓ (if fails)
Yahoo Finance (Multiple methods)
    ↓ (if fails)
Cached/Mock Data
```

## 📈 Strategies

### Single Stock Analysis
- Technical analysis (RSI, moving averages, MACD)
- Fundamental screening (P/E, growth, margins)
- News sentiment integration
- Multi-factor scoring system

### Pairs Trading
- Cointegration testing (Engle-Granger)
- Z-score based entry/exit signals
- Hedge ratio calculation
- Mean reversion detection

### Portfolio Management
- Risk-adjusted position sizing
- Sector diversification requirements
- Automatic rebalancing triggers
- Performance optimization

## 🚨 Risk Management

### Pre-Trade Checks
- Position size validation
- Buying power verification
- Account status checks
- Market condition assessment

### Real-Time Monitoring
- Position exposure tracking
- Risk metric calculation
- Stop-loss management
- Performance monitoring

### Circuit Breakers
- Daily loss limits
- High volatility protection
- Correlation risk detection
- Account safety controls

## 📊 Performance Tracking

### Metrics Tracked
- Total return and P&L
- Win/loss ratios
- Sharpe ratio calculation
- Maximum drawdown
- Strategy-specific performance

### Reporting
- Real-time dashboard
- Daily summary emails
- Trade execution logs
- Performance analytics

## 🔍 Monitoring & Alerts

### Email Notifications
- Trade execution confirmations
- Daily performance summaries
- Risk alert notifications
- System status updates

### Logging
- Comprehensive trade logs
- Error tracking and alerts
- Performance monitoring
- System health checks

## 🐛 Troubleshooting

### Common Issues
1. **API Rate Limits**: System automatically handles rate limiting
2. **Market Data Issues**: Fallback systems ensure data availability
3. **Trading Errors**: Comprehensive error handling and recovery
4. **Performance Issues**: Optimized concurrent processing

### Support
- Check logs in `logs/` directory
- Review configuration in `.env` file
- Verify API key permissions
- Monitor system resource usage

- **Paper Trading**: Safe environment for learning
- **Strategy Development**: Framework for custom strategies
- **Risk Management**: Professional risk controls
- **Performance Analysis**: Comprehensive analytics


Educational use license. See LICENSE file for details.

---

*Built for ML4T - Machine Learning for Trading Education*
