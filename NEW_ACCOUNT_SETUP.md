# New Alpaca Account Setup Checklist

## 🔧 **REQUIRED ACTIONS BEFORE RESTART:**

### 1. **Update API Credentials**
```bash
# Update .env file with new account credentials:
ALPACA_API_KEY="your_new_key"
ALPACA_SECRET_KEY="your_new_secret" 
ALPACA_BASE_URL="https://paper-api.alpaca.markets"  # Start with paper trading
```

### 2. **Enable Enhanced Safety Features**
The following safeguards have been implemented to prevent the previous issues:

#### A. **Fixed Critical Bugs** ✅
- [x] Portfolio balancer Decimal/float type error (tax_aware_portfolio_balancer.py:161)
- [x] Enhanced position size validation
- [x] Cash requirement validation

#### B. **Enhanced Risk Limits**
- Max position size: 5% (vs previous large positions)  
- Min cash buffer: $10,000
- Max margin usage: 20% (vs 71% we had)
- Daily loss limit: 2%
- Max drawdown: 5%

#### C. **Trade Safety Checks**
- Pre-trade cash validation
- Wash trade prevention (30min minimum between same-symbol trades)
- Buying power validation before orders
- Position concentration limits

### 3. **Integration Points to Update**

#### A. **Continuous Rebalancer** (`continuous_rebalancer.py`)
Add validation before trade execution:
```python
from enhanced_position_limits import validate_trading_safety

# Before executing trades
validation = validate_trading_safety(account_info, proposed_trades)
if not validation['passed']:
    logger.error("❌ TRADING BLOCKED by safety validator")
    for error in validation['errors']:
        logger.error(f"  {error}")
    return  # Skip this cycle
```

#### B. **Alpaca Client** (`tools/alpaca_client.py`)
Add enhanced pre-trade validation to place_order method.

#### C. **Portfolio Balancer** (`core/portfolio_balancer.py`)
Integrate position size limits into rebalancing logic.

### 4. **Conservative Restart Settings**

#### A. **Initial Capital Allocation**
- Start with smaller position sizes (2-3% per position)
- Keep 20-30% cash buffer initially
- Limit to 10-15 positions maximum

#### B. **Gradual Scaling**
- Week 1: Small positions, manual oversight
- Week 2-3: Increase position sizes gradually  
- Month 1: Full automation only after proven stability

#### C. **Performance Monitoring**
- Daily P&L alerts
- Cash balance monitoring
- Position size drift alerts
- Margin usage tracking

### 5. **Test Protocol**

#### A. **Paper Trading First** (Recommended)
```bash
# Set paper trading mode in .env
ALPACA_BASE_URL="https://paper-api.alpaca.markets"
```

#### B. **Validation Tests**
```bash
# Test the enhanced validator
python enhanced_position_limits.py

# Test fixed portfolio balancer  
python -c "from core.tax_aware_portfolio_balancer import tax_aware_portfolio_balancer; print('✅ No import errors')"

# Test account connection
python quick_status_check.py
```

#### C. **Limited Live Testing**
- Start with $1,000-5,000 real money
- Run for 1 week with manual oversight
- Scale up only after confirming stability

### 6. **Monitoring Setup**

#### A. **Daily Alerts**
- Cash balance < $5,000
- Any single position > 5% of portfolio
- Daily loss > 1%
- Margin usage > 10%

#### B. **Emergency Stops**
- Auto-stop if cash goes negative
- Auto-stop if daily loss > 2%
- Auto-stop if any position > 8% of portfolio

### 7. **Key Metrics to Track**
- Cash balance (never go negative)
- Position sizes (max 5% each)
- Daily P&L volatility
- Number of trades per day
- Win/loss ratio
- Sharpe ratio

## 🚨 **RED FLAGS TO WATCH FOR:**
1. Declining cash balance trend
2. Increasing position concentration
3. Rising trade frequency
4. Repeated wash trade warnings
5. RL system generating poor signals
6. Portfolio balancer fallback messages

## ✅ **SUCCESS CRITERIA:**
- Positive cash balance maintained
- No position > 5% of portfolio
- Daily volatility < 2%
- Positive 30-day returns
- No margin usage
- Clean trade execution (no wash trades)

---

**Once you have the new account credentials, update the .env file and we'll run through the full validation checklist before restart.**