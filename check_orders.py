#!/usr/bin/env python3
from tools.alpaca_client import alpaca_client
from datetime import datetime, timedelta

print('=== ORDER STATUS CHECK ===')
print(f'Current Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

# Check recent orders
try:
    orders = alpaca_client.api.list_orders(status='all', limit=20)
    print(f'\nRecent Orders (last 20): {len(orders)}')
    
    if orders:
        for order in orders[-10:]:  # Show last 10
            print(f'  {order.submitted_at.strftime("%m-%d %H:%M")} | {order.symbol} | {order.side} {order.qty} @ {order.order_type} | {order.status}')
    else:
        print('  No recent orders found')
        
    # Check today's orders specifically
    today_orders = [o for o in orders if o.submitted_at.date() == datetime.now().date()]
    print(f'\nToday\'s Orders: {len(today_orders)}')
    
except Exception as e:
    print(f'Error checking orders: {e}')

# Check current positions
try:
    positions = alpaca_client.get_positions()
    print(f'\nCurrent Positions: {len(positions)}')
    if positions:
        for pos in positions:
            print(f'  {pos.symbol}: {pos.qty} shares @ ${float(pos.market_value):,.2f}')
except Exception as e:
    print(f'Error checking positions: {e}')

# Check account activity
try:
    account = alpaca_client.get_account_info()
    print(f'\nAccount Status:')
    print(f'  Cash: ${float(account["cash"]):,.2f}')
    print(f'  Equity: ${float(account["equity"]):,.2f}')
    print(f'  Day Trade Count: {account.get("daytrade_count", "N/A")}')
except Exception as e:
    print(f'Error checking account: {e}')

# Check market status
try:
    market_open = alpaca_client.is_market_open()
    print(f'\nMarket Status: {"Open" if market_open else "Closed"}')
except Exception as e:
    print(f'Error checking market status: {e}')