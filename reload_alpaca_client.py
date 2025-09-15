#!/usr/bin/env python3
"""
Force reload Alpaca client with new credentials
"""

import sys
import importlib

# Force reload settings and alpaca client modules
if 'config.settings' in sys.modules:
    importlib.reload(sys.modules['config.settings'])

if 'tools.alpaca_client' in sys.modules:
    importlib.reload(sys.modules['tools.alpaca_client'])

# Import fresh instances
from config.settings import settings
from tools.alpaca_client import alpaca_client

print(f"🔄 Reloaded Alpaca client")
print(f"   Using API Key: {settings.alpaca_api_key}")

# Test the reloaded client
try:
    account_info = alpaca_client.get_account_info()
    print(f"✅ Alpaca client reloaded successfully")
    print(f"   Account ID: {account_info.get('id', 'N/A')}")
    print(f"   Cash: ${float(account_info.get('cash', 0)):,.2f}")
    print(f"   Buying Power: ${float(account_info.get('buying_power', 0)):,.2f}")
    
    # Check if this is the new account
    if account_info.get('id') == "76ebaa0c-49f6-48ac-91d8-24a1069bcf34":
        print(f"🎉 Successfully using NEW ACCOUNT!")
    else:
        print(f"⚠️  Still using old account: {account_info.get('id')}")
        
except Exception as e:
    print(f"❌ Reloaded client test failed: {e}")
    import traceback
    traceback.print_exc()