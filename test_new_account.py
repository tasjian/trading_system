#!/usr/bin/env python3
"""
Test New Account
Quick test of the new Alpaca API credentials.
"""

from tools.alpaca_client import alpaca_client

def test_new_account():
    """Test the new account setup."""
    print("🔍 TESTING NEW ALPACA ACCOUNT")
    print("=" * 50)
    
    try:
        # Test connection
        account = alpaca_client.get_account_info()
        print(f"✅ Connection successful!")
        print(f"📊 Account info:")
        for key, value in account.items():
            print(f"   {key}: {value}")
        
        # Test positions
        positions = alpaca_client.get_positions()
        print(f"\n📈 Current positions: {len(positions)}")
        
        # Test market status
        market_open = alpaca_client.is_market_open()
        print(f"🕐 Market open: {market_open}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_new_account()
    if success:
        print(f"\n🎉 New account is ready!")
    else:
        print(f"\n💥 Account setup needs fixing")