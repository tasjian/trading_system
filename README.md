# 🤖 AI-Powered Trading System

An advanced autonomous trading system that combines reinforcement learning, sentiment analysis, and risk management for intelligent portfolio management. Built with Python and integrated with Alpaca Markets for paper and live trading.

## 🚀 Features

### Core Trading Intelligence
- **🧠 Reinforcement Learning**: Dual-agent RL system with stable policy and continuous learner
- **📊 Multi-Source Sentiment Analysis**: Real-time analysis from news, social media, SEC filings, and earnings
- **⚡ High-Frequency Trading**: 5-minute RL decision cycles with optimized performance
- **💡 Smart Position Sizing**: Automatic position sizing with margin account support
- **🛡️ Risk Management**: Pre-trade safety checks, drawdown limits, and volatility monitoring

### Advanced Architecture
- **🔄 Continuous Rebalancing**: Autonomous operation with intelligent scheduling
- **📈 Portfolio Optimization**: Multi-objective optimization with diversification constraints  
- **🎯 Signal Generation**: Advanced workflow orchestrating multiple AI agents
- **📧 Real-time Notifications**: Email alerts for trades and portfolio updates
- **📋 Compliance**: Regulatory compliance checks and audit trails

## 🆕 Recent Major Improvements

### 🔧 Enhanced Order Management (Latest)
- **Smart Quantity Adjustment**: Automatically adjusts sell orders to available shares
- **OCO Order Support**: Full One-Cancels-Other bracket and breakout order types
- **Improved Error Handling**: Graceful handling of insufficient quantities and API errors

### 📊 Optimized Sentiment Analysis
- **59% Performance Boost**: SEC filings analysis optimized (17.1s → 7.0s)
- **Comprehensive Timeout Management**: Increased from 60s to 240s for full analysis
- **Multi-source Integration**: News, social media, SEC filings, earnings, and market sentiment
- **Social Media Confirmed**: Reddit + Twitter collection operational (3 posts per symbol)

### 🎯 Pure Dynamic Discovery
- **Removed Hardcoded Symbols**: No more AAPL, GOOGL, MSFT, TSLA, NVDA preferences
- **Universe Filter Focus**: Pure 11,400+ stock scanning to 9 actionable candidates
- **ETF-Only Cache Warming**: SPY, QQQ, IWM, XLF, XLK for market context
- **Fresh Analysis**: Cleared cached sentiment for unbiased symbol discovery

### ✅ End-to-End Validation
- **Signal-to-Order Pipeline**: 6 signals → 3 orders successfully placed
- **Portfolio Rebalancing**: SELL/BUY execution operational with risk controls
- **99.9% Efficiency**: Universe filtering reduces 11,402 symbols to 9 candidates
- **50% Success Rate**: Strong signal conversion with quality filtering

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   AI Processing  │    │   Execution     │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • Market Data   │───▶│ • RL Agents      │───▶│ • Alpaca API    │
│ • News/Social   │    │ • Sentiment AI   │    │ • Order Mgmt    │
│ • SEC Filings   │    │ • Risk Models    │    │ • Position Mgmt │
│ • Earnings      │    │ • Portfolio Opt  │    │ • Notifications │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  ▼
                    ┌──────────────────────────┐
                    │   Continuous Rebalancer  │
                    │   • 5-min RL cycles      │
                    │   • 90-min sentiment     │
                    │   • Risk monitoring      │
                    │   • Performance tracking │
                    └──────────────────────────┘
```

### Key Components

- **`continuous_rebalancer.py`**: Main orchestrator with intelligent scheduling
- **`agents/workflow.py`**: Multi-agent workflow for signal generation  
- **`agents/rl_integration_bridge.py`**: RL system integration and fallback logic
- **`agents/sentiment_agent.py`**: Multi-source sentiment analysis
- **`tools/alpaca_client.py`**: Trading execution with safety checks
- **`core/universe_filter.py`**: Dynamic stock universe filtering
- **`compliance/regulatory_compliance.py`**: Compliance and audit framework

## 🛠️ Getting Started

### Prerequisites

```bash
# Python 3.8+
python --version

# Required packages
pip install -r requirements.txt
```

### Environment Setup

1. **Clone Repository**
```bash
git clone https://github.com/tasjian/trading_system.git
cd trading_system
```

2. **Create Environment File**
```bash
cp .env.example .env
```

3. **Configure API Keys** (edit `.env`):
```bash
# Alpaca Trading API
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2  # Paper trading

# LLM APIs (choose one)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Social Media APIs
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
```

### Quick Start

```bash
# Start the autonomous trading system
python continuous_rebalancer.py
```

The system will:
- Initialize all AI agents and data sources
- Begin 5-minute RL trading cycles
- Cache sentiment data every 90 minutes
- Execute trades based on RL decisions
- Send email notifications for all activities

## 📊 Trading Configuration

### Risk Parameters (`config/settings.py`)
```python
max_position_size = 0.35        # Max 35% per position
max_portfolio_risk = 0.05       # Max 5% portfolio risk
stop_loss_percent = 0.08        # 8% stop loss
target_portfolio_size = 35      # Target 35 stocks
```

### RL Configuration (`data/rl_config.json`)
```json
{
  "symbols": ["SPY", "QQQ", "IWM", "VXX", "XLF", "XLE", "XLV", "XLK"],
  "training_episodes": 1000,
  "learning_rate": 0.0001,
  "risk_tolerance": 0.4
}
```

### Timing Configuration
- **RL Decisions**: Every 5 minutes during market hours
- **Sentiment Analysis**: Every 90 minutes (cached)
- **After Hours**: 30-minute intervals
- **Portfolio Tracking**: Real-time

## 🔧 Advanced Usage

### Custom Agent Development

```python
from agents.workflow import TradingWorkflow
from agents.state import create_initial_state

# Create custom trading agent
workflow = TradingWorkflow()
state = create_initial_state()

# Run single cycle
results = await workflow.run_cycle(state, config)
```

### Backtesting

```python
from agents.rl_backtesting_framework import BacktestingFramework

# Run historical backtest
backtester = BacktestingFramework()
results = backtester.run_backtest(
    start_date='2024-01-01',
    end_date='2024-12-31',
    initial_capital=100000
)
```

### Custom Risk Rules

```python
from compliance.regulatory_compliance import RegulatoryCompliance

# Add custom compliance check
compliance = RegulatoryCompliance()
compliance.add_custom_rule("max_crypto_exposure", 0.1)
```

## 📈 Performance Monitoring

### Real-time Logs
```bash
tail -f logs/continuous_rebalancer.log
```

### Key Metrics
- **Signals Generated**: AI-driven buy/sell signals per cycle
- **Orders Executed**: Actual trades completed  
- **Portfolio Value**: Real-time portfolio tracking
- **Success Rate**: RL decision success percentage
- **Risk Metrics**: Drawdown, Sharpe ratio, volatility

### Email Notifications
- Trade executions with reasoning
- Risk limit breaches
- System status updates
- Daily performance summaries

## 🛡️ Safety Features

### Pre-trade Safety Checks
- **Buying Power**: Ensures sufficient capital
- **Position Limits**: Enforces maximum position sizes
- **Risk Limits**: Prevents excessive portfolio risk
- **Market Hours**: Validates trading window

### Margin Account Support
- Automatic detection of margin vs cash accounts
- Conservative position sizing for leveraged accounts
- Real-time buying power monitoring

### Error Handling
- Graceful fallback between RL systems
- Comprehensive logging and error tracking
- Automatic recovery from API failures

## 🔗 API Integrations

### Trading
- **Alpaca Markets**: Primary broker integration
- **Real-time Data**: Market data and portfolio updates

### Data Sources
- **News**: Financial news sentiment analysis
- **Social Media**: Reddit and Twitter sentiment
- **SEC Filings**: Corporate filing analysis
- **Earnings**: Earnings call transcripts

### AI Services
- **OpenAI**: GPT-5-nano for fast sentiment analysis (primary)
- **Anthropic**: Claude models for decision making
- **Local Models**: Llama 3.1 via Ollama (fallback)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`python -m pytest`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes. Trading involves significant financial risk. Always:
- Start with paper trading
- Understand the risks involved  
- Never invest more than you can afford to lose
- Consult with financial professionals
- Comply with all applicable regulations

## 🆘 Support

- **Documentation**: Check `CLAUDE.md` for detailed technical docs
- **Issues**: Report bugs on [GitHub Issues](https://github.com/tasjian/trading_system/issues)
- **Discussions**: Join [GitHub Discussions](https://github.com/tasjian/trading_system/discussions)

## 🎯 Roadmap

- [ ] **Portfolio Analytics Dashboard**: Web-based performance dashboard
- [ ] **Additional Brokers**: Support for Interactive Brokers, TD Ameritrade
- [ ] **Crypto Trading**: Cryptocurrency market support
- [ ] **Advanced RL**: Transformer-based decision models
- [ ] **Multi-timeframe**: Support for different trading timeframes
- [ ] **Backtesting UI**: Interactive backtesting interface

---

*Built with ❤️ by AI-powered automation. Trade responsibly.*