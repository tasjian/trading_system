#!/usr/bin/env python3
"""Quick script to check Alpaca account status"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(__file__))

from tools.alpaca_client import alpaca_client

def main():
    try:
        # Get account info
        account = alpaca_client.get_account_info()
        
        print("=== ALPACA ACCOUNT STATUS ===")
        print(f"Account Status: {account.get('status', 'unknown')}")
        print(f"Cash: ${float(account.get('cash', 0)):,.2f}")
        print(f"Portfolio Value: ${float(account.get('portfolio_value', 0)):,.2f}")
        print(f"Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
        print(f"Equity: ${float(account.get('equity', 0)):,.2f}")
        print(f"Day P&L: ${float(account.get('unrealized_pl', 0)):,.2f}")
        
        # Get positions
        positions = alpaca_client.get_positions()
        print(f"\n=== POSITIONS ({len(positions)}) ===")
        for pos in positions:
            pnl = float(pos.get('unrealized_pnl', 0))
            pnl_emoji = "📈" if pnl >= 0 else "📉"
            print(f"{pos['symbol']:6} | {pos['quantity']:8.0f} shares | ${pos['market_value']:8,.2f} | {pnl_emoji} ${pnl:6.2f}")
        
        # Get recent orders
        orders = alpaca_client.get_orders(status="all", limit=5)
        print(f"\n=== RECENT ORDERS ({len(orders)}) ===")
        for order in orders:
            print(f"{order['symbol']:6} | {order['side'].upper():4} {order['qty']:6.0f} | {order['order_type']} | Status: {order.get('status', 'unknown')}")
        
    except Exception as e:
        print(f"Error checking account: {e}")

if __name__ == "__main__":
    main()