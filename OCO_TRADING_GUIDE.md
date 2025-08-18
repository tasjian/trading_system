# 🎯 OCO (One-Cancels-Other) Trading System

## Overview

The OCO (One-Cancels-Other) trading system provides sophisticated risk management and trading strategies by allowing you to place two linked orders simultaneously. When one order executes, the other is automatically canceled.

## ✅ **System Status: PRODUCTION READY**

- **✅ Fully Integrated** with existing trading system
- **✅ GPT-5-nano Compatible** for intelligent signal generation
- **✅ Risk Management** with comprehensive safety checks
- **✅ Paper Trading Ready** with full Alpaca API integration

## 🚀 Key Features

### 1. **Take-Profit + Stop-Loss Combo (Bracket Orders)**
Automatically protect existing positions with profit targets and stop losses:
```python
# Example: Protect 100 shares of AAPL
result = alpaca_client.place_oco_order(
    symbol='AAPL',
    qty=100,
    side='sell',
    take_profit_price=165.00,  # Sell if price rises to $165
    stop_loss_price=135.00     # Sell if price falls to $135
)
```

### 2. **Breakout Trading Strategies**
Capture moves in either direction with bidirectional breakout orders:
```python
# Example: SPY breakout strategy
result = alpaca_client.place_breakout_oco_order(
    symbol='SPY',
    qty=50,
    upper_breakout_price=425.00,  # Buy if breaks above $425
    lower_breakout_price=415.00   # Sell short if breaks below $415
)
```

## 📋 **Current Configuration**

Based on your system's current settings:
- **Portfolio Value**: $49,240.60
- **Available Cash**: $4,338.44 
- **Buying Power**: $24,673.95

### OCO Settings:
- **OCO Enabled**: ✅ True
- **Auto-placement**: ✅ Enabled
- **Default Take Profit**: 15.0%
- **Default Stop Loss**: 8.0%
- **Max Position Size**: 10.0% of portfolio per OCO
- **Max Daily Orders**: 20 OCO orders per day
- **Min Risk/Reward Ratio**: 1.5:1

## 🛡️ Risk Management Features

### Built-in Safety Checks:
1. **Pre-trade Validation**: Account balance, position limits, price validation
2. **Position Size Limits**: Maximum 10% of portfolio per OCO order
3. **Daily Limits**: Maximum 20 OCO orders per day
4. **Portfolio Exposure**: Maximum 30% of portfolio in OCO positions
5. **Risk/Reward Validation**: Minimum 1.5:1 profit to loss ratio
6. **Price Validation**: Ensures logical price levels based on current market

### Example Risk Validation:
```python
# For a $49,240 portfolio:
max_oco_position = $4,924 (10% limit)
max_daily_oco_orders = 20
max_portfolio_oco_exposure = $14,772 (30% limit)
```

## 🔗 System Integration

### Works with Existing Components:
- **✅ GPT-5-nano Sentiment Analysis**: OCO orders can be triggered by sentiment signals
- **✅ Universe Filter**: OCO strategies applied to filtered stock universe
- **✅ Portfolio Balancer**: Automatic OCO placement for new positions
- **✅ Risk Manager**: All existing risk controls apply to OCO orders
- **✅ Notification System**: Email/webhook notifications for OCO events

### Integration Example:
```python
from core.oco_trading_system import OCOTradingSystem

# Initialize OCO system
oco_system = OCOTradingSystem()

# Automatically protect all current positions
await oco_system.place_portfolio_protection()

# Execute breakout strategies on filtered universe
symbols = ["SPY", "QQQ", "IWM", "VXX"]
await oco_system.execute_breakout_strategies(symbols)
```

## 📊 **How to Use OCO Orders**

### Method 1: Direct API Calls
```python
from tools.alpaca_client import AlpacaClient

alpaca_client = AlpacaClient()

# Bracket OCO (Take-profit + Stop-loss)
bracket_result = alpaca_client.place_oco_order(
    symbol='MSFT',
    qty=25,
    side='sell',  # Assuming you own 25 shares
    take_profit_price=450.00,
    stop_loss_price=380.00,
    time_in_force='gtc'  # Good Till Canceled
)

# Breakout OCO (Bidirectional)
breakout_result = alpaca_client.place_breakout_oco_order(
    symbol='NVDA',
    qty=10,
    upper_breakout_price=1200.00,
    lower_breakout_price=1000.00,
    limit_buffer_percent=0.002,  # 0.2% buffer for limit orders
    time_in_force='gtc'
)
```

### Method 2: High-Level OCO System
```python
from core.oco_order_manager import OCOOrderManager

oco_manager = OCOOrderManager()

# Place protective OCO for existing position
protective_order = await oco_manager.place_bracket_oco(
    symbol='TSLA',
    quantity=15,
    side='sell'  # Will calculate profit/loss levels automatically
)

# Monitor OCO orders
active_ocos = await oco_manager.get_active_oco_orders()
for oco in active_ocos:
    print(f"OCO {oco.id}: {oco.symbol} - {oco.status}")
```

## ⚙️ Configuration Options

### Environment Variables:
```bash
# Enable/disable OCO functionality
OCO_ENABLED=true
OCO_AUTO_PLACEMENT_ENABLED=true

# Risk management settings
OCO_DEFAULT_TAKE_PROFIT_PERCENT=0.15  # 15%
OCO_DEFAULT_STOP_LOSS_PERCENT=0.08    # 8%
OCO_MAX_POSITION_SIZE=0.10            # 10%
OCO_MAX_DAILY_ORDERS=20
OCO_MAX_PORTFOLIO_EXPOSURE=0.30       # 30%

# Trading parameters
OCO_MIN_POSITION_VALUE=100.0          # Minimum $100
OCO_RISK_REWARD_RATIO=1.5             # 1.5:1 minimum
OCO_BREAKOUT_BUFFER_PERCENT=0.002     # 0.2% buffer
```

### Programmatic Configuration:
```python
from config.settings import settings

# Check current settings
print(f"OCO Enabled: {settings.oco_enabled}")
print(f"Take Profit %: {settings.oco_default_take_profit_percent}")
print(f"Stop Loss %: {settings.oco_default_stop_loss_percent}")

# Validate risk settings
if settings.oco_default_take_profit_percent / settings.oco_default_stop_loss_percent < settings.oco_risk_reward_ratio:
    print("⚠️ Risk/reward ratio below minimum threshold")
```

## 🎯 **Use Cases & Examples**

### 1. **Position Protection** (Most Common)
```python
# You bought AAPL at $150, now protect it
result = alpaca_client.place_oco_order(
    symbol='AAPL',
    qty=100,
    side='sell',
    take_profit_price=172.50,  # +15% profit
    stop_loss_price=138.00     # -8% loss
)
# Result: Risk/Reward = 22.50/12.00 = 1.875:1 ✅
```

### 2. **Breakout Capture**
```python
# Capture SPY breakout in either direction
result = alpaca_client.place_breakout_oco_order(
    symbol='SPY',
    qty=25,
    upper_breakout_price=430.00,  # Buy if breaks up
    lower_breakout_price=410.00   # Short if breaks down
)
```

### 3. **Automated Portfolio Protection**
```python
from core.oco_trading_system import place_portfolio_protection

# Automatically protect all positions
protection_results = await place_portfolio_protection()
print(f"Protected {len(protection_results)} positions with OCO orders")
```

### 4. **Integration with Signal Generation**
```python
# In your signal generation workflow
if signal_strength > 0.8 and signal_type == 'BUY':
    # Place normal buy order
    buy_order = alpaca_client.place_order(symbol, qty, 'buy')
    
    # Automatically add OCO protection
    if settings.oco_auto_placement_enabled:
        current_price = alpaca_client.get_current_price(symbol)
        take_profit = current_price * (1 + settings.oco_default_take_profit_percent)
        stop_loss = current_price * (1 - settings.oco_default_stop_loss_percent)
        
        oco_order = alpaca_client.place_oco_order(
            symbol=symbol,
            qty=qty,
            side='sell',
            take_profit_price=take_profit,
            stop_loss_price=stop_loss
        )
```

## 📈 **Performance & Monitoring**

### Order Status Tracking:
```python
# Check OCO order status
order_result = alpaca_client.place_oco_order(...)
order_id = order_result['id']

# Monitor the order
orders = alpaca_client.get_orders(status='all')
for order in orders:
    if order['id'] == order_id:
        print(f"OCO Status: {order['status']}")
        if order['status'] == 'filled':
            print(f"Filled at: {order['filled_avg_price']}")
```

### Portfolio Impact:
```python
# Track OCO performance
oco_positions = await oco_manager.get_active_oco_orders()
total_oco_exposure = sum(pos.quantity * pos.current_price for pos in oco_positions)
portfolio_value = alpaca_client.get_account_info()['portfolio_value']
oco_exposure_percent = total_oco_exposure / portfolio_value

print(f"OCO Exposure: {oco_exposure_percent:.1%} of portfolio")
```

## 🔧 **Troubleshooting**

### Common Issues:

1. **"Invalid take_profit_price" Error**
   ```python
   # Ensure take profit is above current price for buy orders
   current_price = alpaca_client.get_current_price('AAPL')
   take_profit = current_price * 1.15  # 15% above current
   ```

2. **"Position size exceeds limit" Error**
   ```python
   # Check position size against limits
   max_position_value = portfolio_value * settings.oco_max_position_size
   order_value = qty * current_price
   assert order_value <= max_position_value
   ```

3. **"Daily OCO limit exceeded" Error**
   ```python
   # Check daily order count
   if daily_oco_count >= settings.oco_max_daily_orders:
       logger.warning("Daily OCO limit reached")
   ```

### Debug Mode:
```python
import logging
logging.getLogger('tools.alpaca_client').setLevel(logging.DEBUG)
logging.getLogger('core.oco_order_manager').setLevel(logging.DEBUG)

# This will show detailed OCO order processing logs
```

## 🎉 **Next Steps**

1. **Start with Paper Trading**: The system is already configured for safe paper trading
2. **Test Basic Functionality**: Run `python oco_demo.py` to see examples
3. **Configure Risk Parameters**: Adjust settings in `config/settings.py` or environment variables
4. **Integrate with Signals**: Add OCO logic to your signal generation workflow
5. **Monitor Performance**: Track OCO order success rates and portfolio impact

## ⚠️ **Important Notes**

- **Paper Trading Only**: Current configuration is safe for paper trading
- **Risk Management**: All orders respect existing risk controls
- **Market Hours**: OCO orders follow standard market hour restrictions
- **Order Types**: Uses Alpaca's native bracket order functionality for true OCO execution
- **Integration**: Works seamlessly with existing GPT-5-nano sentiment analysis

**The OCO trading system is now fully integrated and ready for use in your paper trading environment!** 🚀