# 🤖 Agentic LLM Trading System

An advanced autonomous trading system using LangGraph multi-agent architecture with comprehensive risk management and Alpaca paper trading integration.

## 🏗️ System Architecture

Based on the architecture diagram, the system implements a multi-layered approach:

### 1. Data Ingestion Layer
- **Market Data**: Alpaca, Finnhub, Yahoo Finance APIs
- **News & Sentiment**: NewsAPI, Reddit, Twitter feeds  
- **Fundamentals**: SEC EDGAR filings integration

### 2. Perception & Memory Layer
- **Vector Database**: Pinecone/Weaviate/Chroma for market memory
- **Strategy Engine**: Real-time evaluation and feedback loops

### 3. Reasoning/Strategy Layer (LLM Agent Framework)
- **Market Analysis Agent**: Data ingestion and technical analysis
- **Risk Management Agent**: Portfolio risk assessment and limits  
- **Strategy Agent**: Signal generation and trade decision logic
- **Execution Agent**: Order management and portfolio tracking

### 4. Execution Layer
- **Alpaca Trading API**: Paper trading with full order management
- **Portfolio & Orders**: Real-time position tracking and P&L

## 🚀 Features

### Enhanced Multi-Agent Architecture
- **LangGraph State Machine**: Orchestrates agent workflows
- **Specialized Market Analysis Agents**: Technical, Fundamental, and Quantitative analysis
- **Risk-Aware Routing**: Conditional workflows based on comprehensive risk assessment
- **State Persistence**: Maintains trading session memory with market conditions

### Advanced Market Analysis Agents
- **Technical Analysis Agent**: Multi-strategy analysis (momentum, mean reversion, breakout)
  - RSI, MACD, Bollinger Bands, ATR, Support/Resistance
  - Ensemble decision making with confidence scoring
  - Dynamic price targets and stop losses
- **Fundamental Analysis Agent**: Market regime and sentiment analysis
  - Market trend detection (bullish/bearish/sideways)
  - Volatility regime assessment (low/medium/high)
  - Sector rotation analysis and fear/greed proxy
- **Quantitative Analysis Agent**: Statistical models and portfolio optimization
  - Risk parity portfolio weighting
  - Sharpe ratio optimization
  - Maximum drawdown analysis
  - Regime change detection

### Advanced Trading Strategies
- **Momentum Strategy**: Multi-timeframe momentum with volume confirmation
- **Mean Reversion Strategy**: Bollinger Band and statistical mean reversion
- **Pairs Trading**: Statistical arbitrage between correlated assets
- **Portfolio Optimization**: Modern Portfolio Theory implementation
- **Risk-Adjusted Sizing**: ATR-based position sizing with signal strength

### Comprehensive Risk Management
- **Multi-Layer Circuit Breakers**: Daily loss, concentration, volatility, correlation
- **Dynamic Position Sizing**: Fixed fractional risk model with ATR stops
- **Portfolio Heat Monitoring**: Total risk exposure tracking
- **Regime-Aware Adjustments**: Risk parameters adapt to market conditions
- **Advanced Stop Losses**: Volatility-based dynamic stops

### Safety Controls
- **Paper Trading Only**: All operations in simulation mode
- **Pre-Trade Validation**: Comprehensive risk checks before execution
- **Emergency Stops**: Manual and automatic halt mechanisms
- **Comprehensive Logging**: Full audit trail of all operations

## 📋 Prerequisites

- Python 3.8+
- Alpaca Paper Trading Account
- OpenAI or Anthropic API Key (for LLM agents)

## 🛠️ Installation

1. **Clone and Setup**
```bash
cd trading_system
pip install -r requirements.txt
```

2. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. **Required API Keys**
- `ALPACA_API_KEY`: Your Alpaca paper trading key
- `ALPACA_SECRET_KEY`: Your Alpaca paper trading secret
- `OPENAI_API_KEY`: OpenAI API key for LLM agents
- `ANTHROPIC_API_KEY`: Alternative Anthropic API key

## 🚦 Usage

### Enhanced Demo Mode
```bash
python enhanced_demo.py
```
Showcases all advanced features including:
- Multi-agent market analysis
- Quantitative strategy backtesting  
- Enhanced signal generation
- Comprehensive risk assessment
- Real-time performance analytics

### Interactive Mode
```bash
python main.py
```

Available commands:
- `cycle` - Run single trading cycle with enhanced analysis
- `auto` - Start continuous trading (5-minute intervals)
- `status` - Show comprehensive system status
- `buy SYMBOL QTY` - Manual buy order with risk validation
- `sell SYMBOL QTY` - Manual sell order with risk validation
- `close SYMBOL` - Close position
- `stop` - Stop system

### Programmatic Usage

#### Enhanced Market Analysis
```python
from agents.market_analysis import market_analysis_factory

# Get comprehensive portfolio recommendations
analysis = market_analysis_factory.get_portfolio_recommendations(
    symbols=["AAPL", "MSFT", "SPY"],
    portfolio_value=100000
)

# Access individual components
symbol_analysis = analysis["symbol_analysis"]  # Technical analysis per symbol
market_condition = analysis["market_condition"]  # Overall market regime
optimal_weights = analysis["optimal_weights"]   # Portfolio optimization
recommendations = analysis["recommendations"]   # Actionable insights
```

#### Quantitative Strategy Testing
```python
from tools.advanced_trading import algorithmic_engine, MomentumStrategy

# Backtest strategies across multiple symbols
results = algorithmic_engine.run_strategy_comparison(
    symbols=["AAPL", "MSFT"],
    lookback_days=252
)

# Get enhanced signals with risk management
signals = algorithmic_engine.generate_enhanced_signals(
    symbols=["AAPL", "MSFT", "SPY"],
    portfolio_value=100000
)
```

#### Advanced Risk Management
```python
from tools.risk_controls import risk_monitor
from tools.advanced_trading import RiskManagementEngine

# Comprehensive risk assessment
assessment = risk_monitor.assess_portfolio_risk()
print(f"Risk Level: {assessment['overall_risk']}")
print(f"Risk Score: {assessment['risk_score']:.1f}/100")

# Calculate optimal position size
risk_mgr = RiskManagementEngine()
position_size = risk_mgr.calculate_position_size(
    symbol="AAPL",
    entry_price=150.0,
    stop_loss=147.0,
    portfolio_value=100000,
    signal_strength=0.8
)
```

#### Trading Workflow
```python
from agents.workflow import trading_workflow

# Run enhanced trading cycle
result = await trading_workflow.run_cycle(
    session_id="my_session",
    input_message="Enhanced market analysis request"
)

# Access enhanced results
print(f"Portfolio Value: ${result['portfolio_value']:.2f}")
print(f"Market Trend: {result['final_state']['market_conditions'].market_trend}")
print(f"Signals Generated: {result['signals_generated']}")
print(f"Orders Executed: {result['orders_executed']}")
```

## ⚙️ Configuration

### Risk Parameters (in `.env`)
```bash
MAX_PORTFOLIO_RISK=0.01    # 1% max risk per trade
MAX_POSITION_SIZE=0.05     # 5% max position size  
STOP_LOSS_PERCENT=0.03     # 3% stop loss
MAX_DAILY_LOSS=0.02        # 2% daily loss limit
MIN_CASH_RESERVE=0.1       # 10% cash reserve
```

### Trading Settings
```bash
TRADING_MODE=paper         # Always use paper mode
MAX_DAILY_TRADES=10        # Limit daily trades
LOG_LEVEL=INFO            # Logging verbosity
```

## 🔍 Monitoring

### System Status
```python
from main import TradingSystemApp

app = TradingSystemApp()
status = app.get_status()

print(f"Portfolio: ${status['account']['equity']:.2f}")
print(f"Positions: {status['portfolio']['positions']}")
print(f"Risk Score: {status.get('risk_score', 'N/A')}")
```

### Risk Assessment
```python
from tools.risk_controls import risk_monitor

assessment = risk_monitor.assess_portfolio_risk()
print(f"Overall Risk: {assessment['overall_risk']}")
print(f"Risk Score: {assessment['risk_score']:.1f}/100")

for alert in assessment['alerts']:
    print(f"⚠️ {alert.level}: {alert.message}")
```

## 📊 Trading Workflow

1. **Market Monitor**: Updates account info, positions, market data
2. **Risk Assessment**: Evaluates portfolio risk and checks limits  
3. **Signal Generation**: Analyzes market data for trading opportunities
4. **Strategy Optimization**: Prioritizes and sizes trading signals
5. **Order Management**: Executes approved trades with safety checks
6. **Portfolio Tracking**: Updates performance metrics and statistics

## 📋 Regulatory Compliance

This system complies with Alpaca Customer Agreement requirements:

- **✅ Paper Trading Only:** System enforces paper trading mode for safety
- **✅ Risk Management:** Comprehensive position and portfolio risk controls  
- **✅ Order Validation:** Pre-trade checks and authorization validation
- **✅ Record Keeping:** Complete audit trails and transaction logging
- **⚠️ Self-directed Trading:** Automated operations with user oversight required

See `COMPLIANCE.md` for detailed compliance documentation.

## 🛡️ Safety Features

### Circuit Breakers
- **Daily Loss Limit**: Halts trading on excessive losses
- **Position Concentration**: Prevents over-exposure to single assets
- **Volatility Spike**: Stops trading during high volatility
- **Liquidity Risk**: Ensures minimum cash reserves
- **Correlation Risk**: Limits exposure to correlated assets

### Validation Layers
1. **Pre-Trade Checks**: Buying power, position limits, asset tradability
2. **Risk Validation**: Portfolio impact assessment
3. **Market Status**: Ensures market is open for trading
4. **Order Validation**: Confirms order parameters are valid

### Emergency Controls
- **Manual Override**: Stop system anytime with Ctrl+C
- **Automatic Halt**: System stops on critical risk violations
- **Order Cancellation**: All pending orders cancelled on shutdown
- **Position Summary**: Final portfolio state logged on exit

## 📈 Performance Tracking

### Metrics Tracked
- Portfolio value and P&L
- Win rate and profit factor  
- Maximum drawdown
- Sharpe ratio
- Position concentration
- Trade frequency

### Logging
- All trades logged with timestamps
- Risk assessments archived
- Agent decisions recorded
- Error conditions tracked

## 🚨 Important Disclaimers

⚠️ **PAPER TRADING ONLY**: This system is configured for paper trading simulation only. Do not use with live trading accounts.

⚠️ **NOT FINANCIAL ADVICE**: This is an educational/research system. All trading decisions are algorithmic and should not be considered financial advice.

⚠️ **RISK WARNING**: Algorithmic trading carries significant risks. Even in paper trading mode, monitor the system actively.

⚠️ **COMPLIANCE**: Ensure compliance with your local financial regulations before any live trading modifications.

## 🔧 Development

### Adding New Strategies
1. Create strategy function in `agents/workflow.py`
2. Add to signal generation agent
3. Configure risk parameters
4. Test thoroughly in paper mode

### Custom Risk Rules
1. Modify `tools/risk_controls.py`
2. Add new risk metrics
3. Update circuit breaker logic
4. Test risk scenarios

### Integration Extensions
1. Add new data sources in `tools/`
2. Create new agent nodes
3. Update workflow routing
4. Extend tool collection

## 📚 References

- [Alpaca Trading API Documentation](https://alpaca.markets/docs/)
- [LangChain Agent Framework](https://langchain.readthedocs.io/)
- [LangGraph Multi-Agent Patterns](https://langchain-ai.github.io/langgraph/)

## 📝 License

This project is for educational purposes. See license terms for usage restrictions.

---

**Happy Trading! 🚀📈**

*Remember: Past performance does not guarantee future results. Trade responsibly.*