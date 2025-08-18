# OCO (One-Cancels-Other) Trading System

## Overview

The OCO Trading System is a comprehensive implementation of advanced order management that enables sophisticated trading strategies through One-Cancels-Other orders. This system integrates seamlessly with the existing algorithmic trading infrastructure to provide automated risk management and position protection.

## Key Features

### 1. **Bracket OCO Orders**
- **Take-Profit + Stop-Loss**: Automatically place both profit target and loss limit orders for existing positions
- **Risk Management**: Ensures every position has built-in protection
- **Customizable Parameters**: Configure take-profit and stop-loss percentages per trade

### 2. **Breakout OCO Strategies** 
- **Bidirectional Trading**: Place buy and sell orders that trigger on price breakouts in either direction
- **Volatility-Based**: Use ATR (Average True Range) calculations for dynamic breakout levels
- **Momentum Capture**: Automatically enter positions when significant price movements occur

### 3. **Automatic Position Protection**
- **Portfolio Monitoring**: Continuously monitor all positions for OCO placement opportunities
- **Smart Placement**: Only place OCO orders for significant positions that warrant protection
- **Rate Limiting**: Prevent excessive order placement through intelligent timing controls

### 4. **Risk Management Integration**
- **Position Size Limits**: Respect existing risk management constraints
- **Exposure Monitoring**: Track total OCO exposure as percentage of portfolio
- **Daily Limits**: Configurable maximum OCO orders per day
- **Profit:Loss Ratio Validation**: Ensure minimum risk-reward ratios

## Architecture

### Core Components

```
OCO Trading System
├── AlpacaClient (Enhanced)
│   ├── place_oco_order()
│   ├── place_breakout_oco_order()
│   └── get_bracket_orders()
├── OCOOrderManager
│   ├── Bracket OCO Management
│   ├── Breakout OCO Management
│   ├── Position Monitoring
│   └── Order Status Tracking
├── OCOOrderExecutor
│   ├── Rebalancing Integration
│   ├── Order Execution
│   └── Performance Monitoring
└── OCOTradingSystem
    ├── System Coordination
    ├── Risk Monitoring
    └── Metrics Collection
```

### Integration Points

1. **Portfolio Balancer**: OCO orders are integrated into the rebalancing decision process
2. **Risk Management**: All OCO orders respect existing risk limits and controls
3. **Signal Generation**: Trading signals can trigger OCO order placement
4. **Monitoring Systems**: OCO orders are tracked and monitored continuously

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# OCO System Configuration
OCO_ENABLED=true
OCO_AUTO_PLACEMENT_ENABLED=true
OCO_DEFAULT_TAKE_PROFIT_PERCENT=0.03  # 3% take profit
OCO_DEFAULT_STOP_LOSS_PERCENT=0.02    # 2% stop loss
OCO_MAX_POSITION_SIZE=0.10            # 10% max position size
OCO_MAX_DAILY_ORDERS=20               # Max 20 OCO orders per day
OCO_MIN_PROFIT_RATIO=1.5              # Minimum 1.5:1 profit:loss ratio

# OCO Breakout Configuration  
OCO_BREAKOUT_ENABLED=true
OCO_BREAKOUT_BUFFER_PERCENT=0.002     # 0.2% buffer for limit orders
OCO_BREAKOUT_LOOKBACK_PERIOD=20       # Days for ATR calculation
OCO_BREAKOUT_VOLATILITY_MULTIPLIER=1.5 # ATR multiplier for breakouts

# OCO Monitoring Configuration
OCO_POSITION_CHECK_INTERVAL=30        # Seconds between position checks
OCO_STATUS_CHECK_INTERVAL=60          # Seconds between status checks
OCO_MAX_HOLDING_PERIOD_HOURS=72       # Max 72 hours per OCO order
```

## Usage Examples

### 1. Basic Bracket OCO Order

```python
from core.oco_order_manager import place_bracket_oco_order

# Place a bracket OCO for SPY
oco_order = await place_bracket_oco_order(
    symbol="SPY",
    quantity=10.0,
    side="buy",
    take_profit_percent=0.025,  # 2.5% profit target
    stop_loss_percent=0.015,    # 1.5% stop loss
    reasoning="Long SPY with OCO protection"
)

if oco_order:
    print(f"OCO Order placed: {oco_order.oco_id}")
    print(f"Take Profit: ${oco_order.take_profit_price:.2f}")
    print(f"Stop Loss: ${oco_order.stop_loss_price:.2f}")
```

### 2. Breakout OCO Strategy

```python
from core.oco_order_manager import place_breakout_oco_order

# Place a breakout OCO for QQQ
oco_order = await place_breakout_oco_order(
    symbol="QQQ", 
    quantity=5.0,
    upper_breakout_percent=0.02,  # 2% above current price
    lower_breakout_percent=0.02,  # 2% below current price
    reasoning="QQQ breakout strategy"
)

if oco_order:
    print(f"Breakout OCO placed: {oco_order.oco_id}")
    print(f"Upper breakout: ${oco_order.upper_breakout_price:.2f}")
    print(f"Lower breakout: ${oco_order.lower_breakout_price:.2f}")
```

### 3. Portfolio Protection

```python
from core.oco_trading_system import place_portfolio_protection

# Automatically protect all significant positions
results = await place_portfolio_protection()

print(f"Portfolio Protection Results:")
print(f"  OCO Orders Placed: {results['oco_placed']}")
print(f"  Positions Skipped: {results['oco_skipped']}")
print(f"  Total Positions: {results['total_positions']}")
```

### 4. Batch Breakout Strategies

```python
from core.oco_trading_system import execute_breakout_strategies

# Execute breakout strategies for multiple symbols
symbols = ["SPY", "QQQ", "IWM", "XLF", "XLK"]
results = await execute_breakout_strategies(symbols)

print(f"Breakout Strategy Results:")
print(f"  Breakouts Placed: {results['breakouts_placed']}")
print(f"  Symbols Processed: {results['total_symbols']}")
```

## System Integration

### With Portfolio Balancer

The OCO system integrates with the portfolio balancer to automatically add risk management to rebalancing decisions:

```python
from core.portfolio_balancer import generate_rebalancing_orders
from core.oco_order_executor import execute_rebalancing_with_oco

# Generate rebalancing decisions (may include OCO orders)
target_allocation = {"SPY": 0.4, "QQQ": 0.3, "IWM": 0.3}
analyses = await analyze_portfolio_balance(target_allocation)
decisions = await generate_rebalancing_orders(analyses)

# Execute with OCO support
results = await execute_rebalancing_with_oco(decisions)
```

### With Risk Management

All OCO orders respect existing risk management constraints:

- **Position Size Limits**: OCO orders cannot exceed `max_position_size` configuration
- **Daily Trade Limits**: OCO orders count toward `max_daily_trades` limit  
- **Portfolio Risk**: Total OCO exposure monitored and limited
- **Margin Requirements**: OCO orders validated against available buying power

## Monitoring and Metrics

### Real-Time Monitoring

The system provides continuous monitoring of:

- **Active OCO Orders**: Track all pending OCO orders
- **Order Status**: Monitor fills, cancellations, and expirations
- **Risk Exposure**: Calculate total OCO exposure vs portfolio
- **Performance Metrics**: Track success rates and P&L

### System Status

```python
from core.oco_trading_system import get_oco_system_status

status = await get_oco_system_status()
print(f"Active OCO Orders: {status['oco_orders']['total_active']}")
print(f"Success Rate: {status['system_metrics']['success_rate']:.1%}")
print(f"OCO Exposure: {status['risk_status']['oco_exposure_percent']:.1%}")
```

### Performance Analytics

- **Success Rate**: Percentage of OCO orders that execute successfully
- **Profit:Loss Ratio**: Average profit vs loss for completed OCO orders
- **Execution Time**: Average time for OCO order placement
- **Risk-Adjusted Returns**: Performance considering risk exposure

## Risk Management Features

### 1. **Position Size Controls**
- Maximum position size per OCO order
- Total OCO exposure limits as percentage of portfolio
- Minimum position size thresholds to avoid micro-positions

### 2. **Rate Limiting**
- Minimum time between OCO orders for same symbol
- Daily maximum OCO order limits
- Concurrent OCO order limits

### 3. **Profit:Loss Ratio Validation**
- Ensure minimum risk-reward ratios (default 1.5:1)
- Validate take-profit and stop-loss levels before placement
- Reject OCO orders with poor risk-reward profiles

### 4. **Expiration Management**
- Maximum holding period for OCO orders (default 72 hours)
- Automatic cancellation of expired OCO orders
- Prevent stale OCO orders from accumulating

## Error Handling

### Robust Error Management
- **Order Rejection Handling**: Graceful handling of broker order rejections
- **Network Issues**: Retry logic with exponential backoff
- **Partial Fills**: Proper handling of partially filled OCO orders
- **Market Hours**: Respect market hours and trading sessions

### Logging and Alerting
- **Comprehensive Logging**: All OCO operations logged with context
- **Error Classification**: Categorize errors by type and severity
- **Performance Tracking**: Log execution times and success rates
- **Alert Integration**: Integration with notification systems

## Testing and Validation

### Paper Trading
All OCO functionality is thoroughly tested in paper trading mode:

```bash
# Run the demonstration script
python examples/oco_trading_demo.py
```

### Unit Tests
Comprehensive test suite covers:
- OCO order placement logic
- Risk management validation
- Error handling scenarios
- Integration points with existing systems

### Backtesting Integration
OCO orders can be backtested using historical data to validate:
- Strategy performance
- Risk management effectiveness
- Execution costs and slippage

## Performance Considerations

### Efficient Order Management
- **Batch Processing**: Process multiple OCO orders efficiently
- **Connection Pooling**: Reuse API connections for better performance
- **Rate Limiting**: Respect broker API rate limits
- **Caching**: Cache market data and order status for better performance

### Scalability
- **Concurrent Processing**: Handle multiple OCO orders simultaneously
- **Memory Management**: Efficient storage of OCO order data
- **Database Integration**: Optional persistence for large-scale operations

## Best Practices

### 1. **Risk Management First**
- Always validate risk-reward ratios before placing OCO orders
- Monitor total OCO exposure continuously
- Respect position size and daily trade limits

### 2. **Market Awareness**
- Consider market conditions when placing OCO orders
- Adjust parameters for different volatility environments
- Account for market hours and liquidity constraints

### 3. **Performance Monitoring**
- Track OCO order success rates and profitability
- Monitor execution costs and slippage
- Regularly review and optimize OCO parameters

### 4. **Integration Discipline**
- Use OCO orders as part of a comprehensive trading strategy
- Coordinate with other risk management systems
- Maintain clear separation between OCO logic and core trading decisions

## Troubleshooting

### Common Issues

1. **OCO Orders Not Placing**
   - Check account buying power and position limits
   - Verify OCO is enabled in configuration
   - Ensure market is open for the asset being traded

2. **OCO Orders Not Executing**
   - Check price levels are reasonable vs current market
   - Verify order status through broker API
   - Monitor for market gaps or unusual volatility

3. **Performance Issues**
   - Review API rate limiting settings
   - Check for excessive OCO order creation
   - Monitor system resource usage

### Debug Mode

Enable debug logging for detailed OCO operation tracking:

```python
import logging
logging.getLogger('core.oco_order_manager').setLevel(logging.DEBUG)
logging.getLogger('core.oco_trading_system').setLevel(logging.DEBUG)
```

## Future Enhancements

### Planned Features
- **Dynamic OCO Parameters**: Adjust take-profit/stop-loss based on volatility
- **Machine Learning Integration**: Use ML to optimize OCO placement timing
- **Multi-Asset OCO**: Cross-asset OCO strategies (e.g., long stock, short sector ETF)
- **Options Integration**: OCO orders with options for enhanced strategies

### Advanced Strategies
- **Trailing OCO**: Combine trailing stops with OCO functionality
- **Time-Based OCO**: OCO orders with time-based expiration rules
- **Correlation OCO**: OCO orders based on asset correlation analysis
- **News-Driven OCO**: OCO placement triggered by news sentiment

## Support and Documentation

### Additional Resources
- **API Documentation**: See `tools/alpaca_client.py` for detailed method documentation
- **Configuration Guide**: Complete list of settings in `config/settings.py`
- **Examples**: Working examples in `examples/oco_trading_demo.py`
- **Unit Tests**: Test cases demonstrate expected behavior

### Getting Help
- Review logs for detailed error messages
- Use the demo script to validate setup
- Check Alpaca API documentation for broker-specific requirements
- Monitor system status through the built-in status reporting

---

**Important Notice**: This OCO trading system is designed for paper trading and educational purposes. Always thoroughly test in paper trading mode before considering live trading. Trading involves substantial risk and is not suitable for all investors.