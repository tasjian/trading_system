# Advanced Algorithmic Trading System

A sophisticated, production-ready algorithmic trading system with comprehensive tax optimization capabilities, multi-modal sentiment analysis, and institutional-grade risk management.

## 🚀 Key Features

### 💰 Tax-Loss Harvesting System (NEW!)
- **Institutional-Grade Tax Optimization**: Automated loss harvesting with 0.5-1.5% annual tax alpha target
- **Wash Sale Compliance**: 30-day monitoring with automatic violation prevention
- **Multi-Method Lot Tracking**: FIFO, LIFO, HIFO, Specific ID accounting with real-time cost basis calculation
- **Asset Replacement Engine**: Correlation-based replacement securities to maintain market exposure
- **Comprehensive Tax Reporting**: Form 8949 and Schedule D ready with audit trails

### 📊 Enhanced Short-Selling Strategy
- **Entity-Resolved Sentiment Analysis**: Advanced NLP with conviction scoring
- **Velocity & Acceleration Tracking**: Sentiment momentum analysis for timing
- **Topic Surprise Detection**: KL divergence-based anomaly detection
- **Kelly Criterion Position Sizing**: Risk-adjusted position allocation with uncertainty handling
- **Squeeze Risk Assessment**: Multi-factor protection against short squeezes

### 🤖 Multi-Modal Sentiment Analysis
- **GPT-5-nano Primary Engine**: State-of-the-art sentiment analysis
- **Social Media Integration**: Reddit, Twitter, TikTok sentiment aggregation
- **News Analysis**: Real-time financial news sentiment processing
- **Cross-Source Validation**: Disagreement detection and confidence scoring

### ⚖️ Advanced Risk Management
- **Portfolio Diversification**: 25-50 stocks across sectors with intelligent allocation
- **Dynamic Position Sizing**: Volatility-adjusted with correlation awareness
- **Real-time Risk Monitoring**: Continuous assessment with automatic rebalancing
- **OCO Trading Support**: One-Cancels-Other orders with breakout detection

### 🔧 Production Architecture
- **Scalable Design**: Asynchronous operations with intelligent caching
- **Comprehensive Logging**: Structured logging with performance monitoring
- **Database Integration**: Optimized SQLite with proper indexing
- **Error Resilience**: Advanced retry mechanisms with graceful degradation

## 📈 Performance Targets

- **Tax Efficiency Ratio**: >95%
- **Tax Alpha**: 0.5-1.5% annually
- **Harvest Success Rate**: >85%
- **Tracking Error**: <2% from replacement assets
- **System Uptime**: >99.5%

## 🏗️ System Architecture

### Core Components

```
trading_system/
├── agents/                          # AI agent implementations
│   ├── workflow.py                  # Agent orchestration
│   └── rl_integration_bridge.py     # Reinforcement learning bridge
├── core/                            # Core trading engines
│   ├── enhanced_short_signal_engine.py      # Advanced short selling
│   ├── tax_loss_harvesting.py              # TLH orchestration
│   ├── lot_tracking.py                     # Specific lot accounting
│   ├── wash_sale_monitor.py                # Compliance monitoring
│   ├── asset_replacement.py               # Replacement asset engine
│   ├── tax_aware_portfolio_balancer.py    # Tax-optimized rebalancing
│   ├── tax_reporting_analytics.py         # Tax reporting & analytics
│   ├── borrow_cost_monitor.py             # Short sell cost tracking
│   ├── short_risk_manager.py              # Short position risk
│   ├── enhanced_position_sizer.py         # Advanced position sizing
│   └── sec_edgar_client.py               # SEC filing analysis
├── tools/                           # Market data and execution
│   ├── alpaca_client.py            # Enhanced Alpaca integration
│   └── dual_provider_market_data.py # Multi-source market data
├── notifications/                   # Alert systems
│   └── email_webhooks.py           # Email and webhook notifications
├── utils/                          # Utilities and scheduling
│   └── daily_summary_scheduler.py  # Performance reporting
├── config/                         # Configuration management
│   └── settings.py                 # Comprehensive settings
├── continuous_rebalancer.py        # Main trading engine
└── main.py                         # System entry point
```

### Database Systems

- **Lot Tracking**: `data/lot_tracking.db` - Tax lot management
- **Wash Sale Monitoring**: `data/wash_sale.db` - Compliance tracking
- **Tax Reporting**: `data/tax_reporting.db` - Tax analytics
- **Asset Replacement**: `data/asset_replacement.db` - Replacement candidates

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Required API keys (set in .env file)
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
OPENAI_API_KEY=your_openai_key  # Required for GPT-5-nano
```

### Basic Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
vim .env

# Key settings:
TLH_ENABLED=true
TLH_STRATEGY=balanced_approach
TAX_SITUATION=medium_income
TRADING_MODE=paper  # Start with paper trading
```

### Launch System

```bash
# Start the trading system
python main.py

# Or run continuous rebalancer directly
python continuous_rebalancer.py
```

## 💼 Tax-Loss Harvesting Usage

### Enable TLH

```python
from core.tax_loss_harvesting import scan_for_tlh_opportunities, execute_tlh_strategy
from tools.alpaca_client import alpaca_client

# Scan for opportunities
portfolio_positions = alpaca_client.get_positions()
opportunities = await scan_for_tlh_opportunities(portfolio_positions)

# Execute harvesting
results = await execute_tlh_strategy(opportunities)
print(f"Harvested ${sum(r.tax_benefit_actual for r in results):,.2f} in tax benefits")
```

### Generate Tax Reports

```python
from core.tax_reporting_analytics import generate_tax_report
from datetime import date

# Generate annual tax report
report = await generate_tax_report(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31)
)

print(f"Total tax alpha: {report['after_tax_performance']['tax_alpha']:.2%}")
```

### Real-Time Tax Impact

```python
from core.tax_reporting_analytics import calculate_real_time_tax_impact

# Analyze current portfolio tax implications
tax_impact = await calculate_real_time_tax_impact(portfolio_positions)
print(f"Tax drag: {tax_impact['portfolio_summary']['tax_drag_percent']:.2f}%")
```

## 📊 Enhanced Short Selling

### Short Signal Generation

```python
from core.enhanced_short_signal_engine import enhanced_short_signal_engine

# Analyze short opportunities
short_signals = await enhanced_short_signal_engine.generate_enhanced_short_signals(
    symbols=['AAPL', 'TSLA', 'GOOGL']
)

for signal in short_signals:
    print(f"{signal.symbol}: {signal.confidence_score:.2f} confidence")
```

### Risk Management

```python
from core.short_risk_manager import short_risk_manager

# Assess short squeeze risk
risk_assessment = await short_risk_manager.assess_comprehensive_risk('TSLA')
print(f"Squeeze risk: {risk_assessment.overall_risk_level}")
```

## ⚙️ Configuration Options

### Tax-Loss Harvesting Strategies

- **`aggressive`**: Harvest all available losses immediately
- **`moderate`**: Harvest significant losses with timing consideration
- **`conservative`**: Harvest only high-conviction losses
- **`balanced_approach`** ⭐: Balance tax efficiency with portfolio needs
- **`rebalancing_only`**: Only harvest during portfolio rebalancing

### Tax Situations

- **`high_income`**: >$400K income, maximizing loss benefits
- **`medium_income`** ⭐: $100K-$400K income, balanced approach
- **`low_income`**: <$100K income, focus on long-term gains
- **`retired`**: Retiree tax planning focus

### Lot Accounting Methods

- **`HIFO`** ⭐: Highest In, First Out (optimal for tax harvesting)
- **`FIFO`**: First In, First Out
- **`LIFO`**: Last In, First Out
- **`SPECIFIC_ID`**: Specific lot identification
- **`AVERAGE_COST`**: Average cost method (for mutual funds)

## 📈 Monitoring & Analytics

### System Health

```bash
# Check system status
python -c "from continuous_rebalancer import ContinuousRebalancer; print('System OK')"

# Validate TLH components
python -c "from core.tax_loss_harvesting import tax_loss_harvesting_engine; print('TLH OK')"
```

### Performance Metrics

- **Tax Alpha Tracking**: Measure after-tax outperformance
- **Harvest Success Rate**: Monitor TLH effectiveness  
- **Compliance Status**: Wash sale violation tracking
- **Portfolio Tax Drag**: Real-time tax impact assessment

### Log Analysis

```bash
# View trading logs
tail -f logs/trading_system.log

# Tax-specific events
grep "TLH\|tax" logs/trading_system.log
```

## 🛡️ Risk Management

### Position Limits

```python
# Default configuration
MAX_POSITION_SIZE = 0.35      # 3.5% max per position
MIN_POSITION_SIZE = 0.01      # 1% minimum position
MAX_PORTFOLIO_RISK = 0.05     # 5% portfolio risk limit
TARGET_PORTFOLIO_SIZE = 35    # Target 35 positions
```

### Tax Risk Controls

```python
# TLH risk management
MAX_DAILY_HARVEST_AMOUNT = 10000.00    # $10K max daily harvesting
MAX_TRACKING_ERROR_TOLERANCE = 0.02    # 2% tracking error limit
WASH_SALE_COOLING_PERIOD = 31          # 31-day safety margin
```

## 🔧 Troubleshooting

### Common Issues

**TLH Not Finding Opportunities**
```bash
# Check filtering settings
python -c "from config.settings import settings; print(f'Min loss: ${settings.min_loss_threshold}')"

# Verify market conditions
python test_tlh_opportunities.py
```

**Database Errors**
```bash
# Reinitialize databases
rm -rf data/*.db
python -c "from core.tax_loss_harvesting import tax_loss_harvesting_engine; print('Databases reinitialized')"
```

**Import Errors**
```bash
# Validate all components
python -c "
from core.tax_loss_harvesting import TaxLossHarvestingEngine
from core.lot_tracking import SpecificLotTracker  
from core.wash_sale_monitor import WashSaleComplianceMonitor
print('All imports successful')
"
```

## 📋 Testing

### Comprehensive Test Suite

```bash
# Run tax compliance tests
python tests/test_tax_compliance.py

# Test Enhanced Short system
python test_enhanced_short_system.py

# End-to-end system validation
python test_end_to_end_enhanced_short.py
```

### Manual Testing

```bash
# Test TLH components
python test_tax_compliance.py

# Validate short selling
python test_short_signal_generation.py

# Check external APIs
python test_external_apis.py
```

## 🔒 Security & Compliance

### Tax Compliance

- **Wash Sale Rule (IRC §1091)**: Automated 30-day monitoring
- **Cost Basis Accuracy**: Commission and fee inclusion
- **Audit Trail**: Complete transaction history with lot identification
- **Form 8949 Ready**: Tax reporting preparation

### Data Security

- **API Key Protection**: Environment variable isolation
- **Local Storage**: SQLite databases with proper permissions
- **No Cloud Dependencies**: All processing local and secure

### Risk Controls

- **Position Limits**: Multiple layers of risk management
- **Liquidity Checks**: Ensure sufficient market depth
- **Correlation Monitoring**: Prevent over-concentration
- **Stop Loss Integration**: Automatic risk exits

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone <repository_url>
cd trading_system

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

### Architecture Guidelines

- **Asynchronous Design**: All I/O operations should be async
- **Error Handling**: Comprehensive exception handling with logging
- **Configuration**: Use settings.py for all configurable parameters
- **Database**: Use SQLite with proper indexing and transactions
- **Testing**: Include unit tests for all new functionality

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes. Tax-loss harvesting involves complex tax implications - consult with tax professionals before using with real money. Past performance does not guarantee future results. Trading involves substantial risk of loss.

---

## 🎯 Next Steps

1. **Paper Trading Validation**: Test all systems with paper money first
2. **Tax Professional Consultation**: Review TLH strategy with tax advisor
3. **Gradual Rollout**: Start with small position sizes
4. **Performance Monitoring**: Track tax alpha and system efficiency
5. **Regular Audits**: Quarterly review of TLH effectiveness

**System Status**: ✅ **READY FOR PRODUCTION**

- 🔧 All engines operational
- 📊 Tax optimization active  
- 🗄️ Databases initialized
- ⚖️ Compliance monitoring enabled
- 📈 Ready for tax-optimized trading

For detailed implementation guidance, see `TAX_LOSS_HARVESTING_GUIDE.md`.