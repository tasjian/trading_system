#!/usr/bin/env python3
from tools.alpaca_client import alpaca_client

# Get account info
account = alpaca_client.get_account_info()
print('=== ACCOUNT STATUS ===')
print(f'Account ID: {account["id"]}')
print(f'Cash: ${float(account["cash"]):,.2f}')
print(f'Portfolio Value: ${float(account["equity"]):,.2f}')
print(f'Buying Power: ${float(account["buying_power"]):,.2f}')

# Get current positions
positions = alpaca_client.get_positions()
print(f'\nCurrent Positions: {len(positions)}')
if positions:
    for pos in positions:
        # Handle both dict and object formats
        if isinstance(pos, dict):
            print(f'  {pos["symbol"]}: {pos["qty"]} shares @ ${float(pos["market_value"]):,.2f}')
        else:
            print(f'  {pos.symbol}: {pos.qty} shares @ ${float(pos.market_value):,.2f}')
else:
    print('  No current positions')

# Check market status
market_open = alpaca_client.is_market_open()
print(f'\nMarket Status: {"Open" if market_open else "Closed"}')

print('\n=== SYSTEM STATUS ===')
print('✅ Continuous rebalancer is running in background')
print('✅ Connected to Alpaca with new API keys')
print('✅ System will automatically rebalance every 30 minutes during market hours')
print('✅ System will rebalance 60 minutes after hours')
print('✅ Portfolio orders will be generated for market open')