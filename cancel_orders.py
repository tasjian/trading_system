#!/usr/bin/env python3
import sys
sys.path.append('.')
from tools.alpaca_client import AlpacaClient

client = AlpacaClient()

print("🚫 CANCELING ALL OPEN ORDERS")
print("=" * 40)

try:
    # Get open orders first
    orders = client.api.list_orders(status='open')
    print(f"Found {len(orders)} open orders to cancel")
    
    if len(orders) > 0:
        # Cancel all orders
        result = client.cancel_all_orders()
        print(f"Cancel result: {result}")
        
        # Check if orders are canceled
        remaining_orders = client.api.list_orders(status='open')
        print(f"Remaining open orders: {len(remaining_orders)}")
        
        # Check buying power after cancellation
        account = client.get_account_info()
        print(f"New buying power: ${account['buying_power']:.2f}")
    else:
        print("No open orders to cancel")
        
except Exception as e:
    print(f"Error canceling orders: {e}")