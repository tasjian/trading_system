#!/usr/bin/env python3
import sys
sys.path.append('.')
from tools.alpaca_client import AlpacaClient

client = AlpacaClient()

print("📋 CHECKING ORDERS AND ACCOUNT STATUS")
print("=" * 50)

# Get pending orders
try:
    orders = client.api.list_orders(status='open')
    print(f"📝 Open orders: {len(orders)}")
    for order in orders:
        print(f"  {order.symbol}: {order.side} {order.qty} at ${order.limit_price or 'market'} - {order.status}")
except Exception as e:
    print(f"Error getting orders: {e}")

# Get filled orders from today
try:
    from datetime import datetime, timedelta
    start_date = datetime.now() - timedelta(days=1)
    orders = client.api.list_orders(status='filled', after=start_date)
    print(f"\n💰 Filled orders (last 24h): {len(orders)}")
    for order in orders:
        print(f"  {order.symbol}: {order.side} {order.qty} at ${order.filled_avg_price} - {order.filled_at}")
except Exception as e:
    print(f"Error getting filled orders: {e}")

# Check account
account = client.get_account_info()
print(f"\n📊 Account Status:")
print(f"  Status: {account['status']}")
print(f"  Portfolio Value: ${account['portfolio_value']:.2f}")
print(f"  Cash: ${account['cash']:.2f}")
print(f"  Buying Power: ${account['buying_power']:.2f}")
print(f"  Equity: ${account['equity']:.2f}")
print(f"  Day Trade Count: {account['day_trade_count']}")
print(f"  Trading Blocked: {account['trading_blocked']}")
print(f"  Account Blocked: {account['account_blocked']}")