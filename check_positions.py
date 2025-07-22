#!/usr/bin/env python3
import sys
sys.path.append('.')
from tools.alpaca_client import AlpacaClient

client = AlpacaClient()
positions = client.get_positions()
print(f'Current positions: {len(positions)}')

for pos in positions:
    print(f"  {pos['symbol']}: {pos['qty']} shares, ${pos['market_value']:.2f} value")
    
account = client.get_account_info()
print(f"\nAccount summary:")
print(f"  Portfolio value: ${account['portfolio_value']:.2f}")
print(f"  Cash: ${account['cash']:.2f}")
print(f"  Buying power: ${account['buying_power']:.2f}")