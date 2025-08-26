#!/usr/bin/env python3
"""Quick status check for trading system"""

import asyncio
import logging
from tools.alpaca_client import AlpacaClient

logging.basicConfig(level=logging.ERROR)  # Suppress debug logs

async def main():
    try:
        print("🔍 Checking account status...")
        client = AlpacaClient()
        
        # Get account info (not async)
        account_info = client.get_account_info()
        
        print(f"Account ID: {account_info['id']}")
        print(f"Status: {account_info['status']}")
        print(f"Cash: ${float(account_info['cash']):,.2f}")
        print(f"Buying Power: ${float(account_info['buying_power']):,.2f}")
        print(f"Equity: ${float(account_info['equity']):,.2f}")
        print(f"Portfolio Value: ${float(account_info['portfolio_value']):,.2f}")
        
        # Check if system can trade
        if float(account_info['buying_power']) <= 0:
            print("❌ TRADING SUSPENDED: No buying power available")
            print("   System would halt on startup due to cash validation")
        else:
            print("✅ TRADING AVAILABLE: Sufficient buying power")
            
        # Check positions (async)
        positions = await client.get_positions()
        print(f"\nPositions: {len(positions)} open")
        if len(positions) > 0:
            total_unrealized = sum(float(p['unrealized_pl']) for p in positions)
            print(f"Total Unrealized P&L: ${total_unrealized:,.2f}")
        
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())