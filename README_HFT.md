# High Frequency Trading (HFT) System

This branch contains a specialized high-frequency trading system designed for accounts with **$25,000+ balance** that can perform unlimited day trading without Pattern Day Trader (PDT) restrictions.

## 🚀 Key Features

### Account Requirements
- **Minimum Balance**: $25,000 (PDT compliance)
- **Unlimited Day Trading**: No 3-trade limit restrictions
- **High Leverage**: Up to 4:1 day trading buying power
- **Real-Time Monitoring**: Continuous balance and risk monitoring

### Trading Capabilities
- **High Frequency**: Up to 1,000 trades per day
- **Advanced Order Types**: Stop-limit, trailing stops, OCO orders
- **Cross-Industry Diversification**: 12+ industry sectors
- **Real-Time Signals**: Momentum, reversal, and breakout detection
- **Risk Management**: Automatic stop loss and take profit

### Technical Features
- **Sub-Second Execution**: Optimized for speed
- **Multi-Symbol Monitoring**: 40+ liquid stocks and ETFs
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Volume analysis
- **Position Management**: Up to 100 concurrent positions
- **Compliance Monitoring**: Real-time regulatory compliance

## 📁 HFT Directory Structure

```
hft/
├── hft_engine.py         # Core HFT trading engine
├── hft_compliance.py     # HFT-specific compliance checks
└── __init__.py          # HFT module initialization

config/
├── hft_settings.py      # HFT configuration parameters

hft_main.py              # HFT application entry point
README_HFT.md           # This documentation
```

## ⚙️ Configuration

### HFT Settings (`config/hft_settings.py`)

```python
# Account Requirements
minimum_account_balance: $25,000
minimum_cash_reserve: $5,000
max_account_utilization: 95%

# Trading Frequency
max_trades_per_day: 1,000
max_trades_per_hour: 200
max_trades_per_minute: 10
min_time_between_trades: 2.0 seconds

# Position Management
max_positions: 100
max_position_size: 5% per position
position_hold_time: 1-60 minutes

# Risk Management
max_daily_loss: 2%
max_drawdown: 5%
stop_loss_percent: 1%
take_profit_percent: 2%
```

## 🎯 Trading Universe

The HFT system focuses on highly liquid securities:

### Mega Cap Technology
- AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA

### Financial Services
- JPM, BAC, WFC, GS, MS, C

### Consumer & Retail
- WMT, HD, COST, MCD, DIS, NKE

### Healthcare & Pharma
- JNJ, UNH, PFE, ABBV, TMO, DHR

### Industrial & Defense
- CAT, BA, HON, LMT, RTX, MMM

### ETFs for Diversification
- SPY, QQQ, IWM, EFA, VTI

## 🚀 Getting Started

### 1. Prerequisites
- Account balance ≥ $25,000
- Pattern Day Trader status (or eligible for it)
- Alpaca paper trading account configured
- All dependencies installed

### 2. Launch HFT System
```bash
python hft_main.py
```

### 3. Main Menu Options
1. **Start HFT Trading (1 hour)** - Quick 1-hour session
2. **Start HFT Trading (Custom)** - Custom duration (1-480 minutes)
3. **Check Account Status** - View account details and positions
4. **View Compliance Report** - Review regulatory compliance
5. **View HFT Settings** - Display configuration parameters
6. **Test HFT Signals** - Test signal generation without trading
7. **Exit** - Shutdown system

## 📊 HFT Strategy Types

### 1. Momentum Trading
- Detects 0.2%+ price movements with 2x+ volume spikes
- Executes in direction of momentum
- 5-30 minute holding periods
- Automatic stop loss and take profit

### 2. Mean Reversion
- RSI oversold (<25) and overbought (>75) conditions
- Contrarian positioning against extreme moves
- 10-45 minute holding periods
- Smaller profit targets for safer returns

### 3. Breakout Trading
- Volume and volatility spike detection
- Bollinger Band breakouts
- MACD signal confirmations
- Variable holding periods based on momentum

## 🛡️ Risk Management

### Real-Time Monitoring
- **Balance Monitoring**: Continuous $25K minimum check
- **Drawdown Limits**: 5% maximum drawdown protection
- **Daily Loss Limits**: 2% maximum daily loss
- **Position Limits**: 5% maximum per position

### Compliance Features
- **PDT Compliance**: Validates unlimited day trading eligibility
- **Position Limits**: Monitors concentration risks
- **Trading Frequency**: Enforces rate limiting
- **Wash Sale Prevention**: Prevents prohibited transactions
- **Risk Limit Monitoring**: Real-time risk assessment

### Emergency Protocols
- **Circuit Breakers**: Automatic trading halt on violations
- **Position Liquidation**: Emergency position closure
- **Balance Alerts**: Immediate notifications for minimum balance
- **Compliance Violations**: Automatic system shutdown on critical issues

## 📈 Performance Tracking

### Real-Time Metrics
- **Win Rate**: Percentage of profitable trades
- **Daily P&L**: Real-time profit/loss tracking
- **Drawdown**: Maximum peak-to-trough decline
- **Trade Statistics**: Volume, frequency, and success rates
- **Position Analytics**: Hold times, sizes, and performance

### Compliance History
- **Compliance Score**: Historical compliance percentage
- **Violation Tracking**: Frequency and types of violations
- **Trend Analysis**: Improving vs declining compliance
- **Recommendations**: Automated compliance suggestions

## 🔧 Advanced Features

### Order Management
- **Smart Order Types**: Automatic selection based on confidence
- **Slippage Control**: Maximum 0.1% slippage tolerance
- **Order Timeout**: 30-second automatic cancellation
- **Fill Optimization**: Best execution algorithms

### Market Data
- **Tick-Level Data**: Sub-second price updates
- **Level 2 Data**: Order book depth (when available)
- **News Integration**: Sentiment-weighted signals
- **Multi-Timeframe**: 1-minute to daily analysis

## ⚠️ Important Notes

### Regulatory Compliance
- This system is designed for **paper trading only**
- Real money trading requires additional regulatory approval
- Maintain proper records for tax and regulatory purposes
- Consult with financial advisors for live trading

### Risk Warnings
- High-frequency trading involves significant risks
- Past performance does not guarantee future results
- Market conditions can change rapidly
- Always maintain adequate capital reserves

### Technical Requirements
- Stable internet connection required
- Low-latency execution preferred
- Sufficient computational resources
- Backup systems recommended for live trading

## 🔧 Troubleshooting

### Common Issues

**Balance Below Minimum**
- Deposit additional funds
- Reduce position sizes
- Check account transfer status

**Compliance Violations**
- Review compliance report
- Adjust trading parameters
- Contact compliance team

**Signal Generation Issues**
- Check market hours
- Verify data connectivity
- Review symbol universe

**Order Execution Problems**
- Check buying power
- Verify market hours
- Review order parameters

## 📞 Support

For technical support and questions:
- Review logs in `logs/hft_trading.log`
- Check compliance reports for violations
- Verify account status and balance
- Contact development team for system issues

---

**Disclaimer**: This HFT system is for educational and paper trading purposes only. Real money trading carries significant risks and requires proper regulatory compliance, risk management, and professional oversight.