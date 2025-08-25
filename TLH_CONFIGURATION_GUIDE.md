# Tax-Loss Harvesting Configuration Guide

## Overview

The Tax-Loss Harvesting (TLH) system can be completely enabled or disabled via configuration settings. When disabled, the system operates as a traditional portfolio balancer without any tax-related processing.

## Configuration

### Enable/Disable TLH

Set the following in your `.env` file or environment variables:

```bash
# Enable TLH (default: true)
TLH_ENABLED=true

# Disable TLH completely
TLH_ENABLED=false
```

### TLH Strategy Configuration

When TLH is enabled, you can configure the strategy:

```bash
# TLH Strategy options: aggressive, moderate, conservative, balanced_approach, ignore_tax
TLH_STRATEGY=balanced_approach

# Tax situation: high_income, medium_income, low_income, retired  
TAX_SITUATION=medium_income

# Minimum loss thresholds
MIN_LOSS_THRESHOLD=100.00      # Minimum $100 loss to harvest
MIN_LOSS_PERCENTAGE=0.05       # Minimum 5% loss percentage

# Daily limits
MAX_DAILY_HARVEST_AMOUNT=10000.00  # Max $10K harvested per day
MAX_POSITIONS_TO_HARVEST=10        # Max 10 positions per day

# Tax rates for calculations
ORDINARY_INCOME_TAX_RATE=0.32      # 32% marginal tax rate
CAPITAL_GAINS_TAX_RATE=0.20        # 20% long-term capital gains
NET_INVESTMENT_INCOME_TAX=0.038    # 3.8% NIIT

# Wash sale monitoring
WASH_SALE_LOOKBACK_DAYS=30         # 30-day wash sale period
WASH_SALE_COOLING_PERIOD_DAYS=31   # 31-day cooling period

# Lot tracking method
DEFAULT_LOT_ACCOUNTING_METHOD=HIFO  # FIFO, LIFO, HIFO, LOFO, SPECIFIC_ID, AVERAGE_COST
```

## Behavior When TLH is Enabled

✅ **Full Tax-Aware Operations:**
- Scans portfolio for loss harvesting opportunities
- Monitors wash sale rule compliance (30-day periods)
- Identifies replacement assets to maintain exposure
- Calculates after-tax performance impact
- Generates tax-optimized rebalancing recommendations
- Tracks specific lots for precise cost basis management
- Provides comprehensive tax reporting

## Behavior When TLH is Disabled

🚫 **Traditional Portfolio Management:**
- No loss harvesting opportunity scanning
- No wash sale rule monitoring  
- No replacement asset identification
- No tax-specific calculations
- Standard portfolio rebalancing only
- Strategy automatically set to `IGNORE_TAX`
- No TLH-related database operations

## Code Examples

### Programmatic Control

```python
from config.settings import settings

# Check current TLH status
if settings.tlh_enabled:
    print("TLH is enabled")
else:
    print("TLH is disabled - using traditional rebalancing")

# Use TLH engine (respects the setting)
from core.tax_loss_harvesting import tax_loss_harvesting_engine

opportunities = await tax_loss_harvesting_engine.scan_for_opportunities(portfolio)
# Returns empty list if TLH is disabled

# Use tax-aware portfolio balancer
from core.tax_aware_portfolio_balancer import tax_aware_portfolio_balancer

analysis = await tax_aware_portfolio_balancer.analyze_tax_aware_portfolio_balance(allocation)
# Uses IGNORE_TAX strategy if TLH is disabled
```

### Runtime Status Check

```python
# Check if TLH components are active
from core.tax_loss_harvesting import tax_loss_harvesting_engine

print(f"TLH Status: {'ENABLED' if tax_loss_harvesting_engine.tlh_enabled else 'DISABLED'}")
```

## Performance Impact

### TLH Enabled
- Additional database queries for lot tracking
- Wash sale rule compliance checks
- Replacement asset correlation calculations
- Tax benefit estimations
- ~15-25% additional processing time

### TLH Disabled  
- Minimal performance impact
- No additional database operations
- No tax-related calculations
- Standard portfolio processing only
- ~5% performance improvement

## Use Cases

### When to Enable TLH
- **Taxable Investment Accounts**: Maximum benefit for taxable accounts
- **High Tax Brackets**: Greater benefit for higher marginal tax rates
- **Large Portfolios**: More opportunities in diversified portfolios
- **Active Management**: Frequent rebalancing creates more opportunities
- **Year-End Planning**: Especially valuable in Q4 for tax planning

### When to Disable TLH
- **Tax-Advantaged Accounts**: No benefit in IRAs, 401(k)s, etc.
- **Low Tax Brackets**: Minimal benefit for low-income investors
- **Small Portfolios**: Limited opportunities in concentrated positions
- **Passive Management**: Buy-and-hold strategies have fewer opportunities
- **Regulatory Requirements**: Some institutional accounts prohibit TLH

## Migration Guide

### Enabling TLH on Existing System

1. Set `TLH_ENABLED=true` in `.env`
2. Configure tax rates and thresholds
3. Restart the trading system
4. System will begin monitoring for opportunities

### Disabling TLH on Existing System

1. Set `TLH_ENABLED=false` in `.env`  
2. Restart the trading system
3. System reverts to traditional rebalancing
4. Existing TLH data remains in database for reporting

## Monitoring and Alerts

The system logs TLH status on startup:

```
✅ Tax-Loss Harvesting ENABLED with strategy: balanced_approach
```

or

```
🚫 Tax-Loss Harvesting DISABLED via settings.tlh_enabled=False
```

## Integration with Existing Workflows

The TLH enable/disable functionality is fully integrated:

- **Portfolio Balancer**: Automatically switches between tax-aware and traditional modes
- **Order Generation**: Respects TLH settings for all trading decisions  
- **Reporting**: Includes appropriate metrics based on TLH status
- **Risk Management**: Maintains all safety checks regardless of TLH status

## Testing

Use the provided test scripts to verify functionality:

```bash
# Full comprehensive test
python test_tlh_configuration.py

# Quick enable/disable test
python test_tlh_simple.py
```

Both tests verify that the system works correctly in both enabled and disabled modes.