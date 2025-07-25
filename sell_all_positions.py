#!/usr/bin/env python3
"""
Sell all current positions to reset buying power for testing new sentiment system.
"""

import asyncio
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from tools.alpaca_client import alpaca_client

async def sell_all_positions():
    """Sell all current positions to free up buying power."""
    
    print("🔥 SELLING ALL POSITIONS TO RESET BUYING POWER")
    print("=" * 60)
    
    try:
        # Get current positions
        positions = alpaca_client.get_positions()
        
        if not positions:
            print("📭 No positions to sell")
            return True
        
        print(f"📊 Found {len(positions)} positions to sell:")
        print("-" * 40)
        
        total_value = 0
        for pos in positions:
            symbol = pos.get('symbol', '')
            qty = float(pos.get('qty', 0))
            market_value = float(pos.get('market_value', 0))
            unrealized_pl = float(pos.get('unrealized_pl', 0))
            
            total_value += market_value
            side = 'LONG' if qty > 0 else 'SHORT'
            
            print(f"  {symbol}: {abs(qty)} shares ({side})")
            print(f"    Value: ${market_value:,.2f}, P&L: ${unrealized_pl:,.2f}")
        
        print(f"\n💰 Total Portfolio Value: ${total_value:,.2f}")
        
        # Confirm action
        print(f"\n⚠️ CONFIRMATION REQUIRED")
        print(f"This will sell ALL {len(positions)} positions to free up buying power")
        print(f"for testing the new sentiment analysis system.")
        
        confirmation = input("\n🤔 Proceed with selling all positions? (yes/no): ").lower().strip()
        
        if confirmation in ['yes', 'y']:
            print(f"\n🔥 SELLING ALL POSITIONS")
            print("-" * 30)
            
            successful_sales = 0
            failed_sales = 0
            
            for i, pos in enumerate(positions):
                symbol = pos.get('symbol', '')
                qty = float(pos.get('qty', 0))
                
                try:
                    print(f"[{i+1}/{len(positions)}] Selling {symbol} ({abs(qty)} shares)...")
                    
                    # Place market sell order
                    result = alpaca_client.place_order(
                        symbol=symbol,
                        qty=abs(qty),  # Use absolute value
                        side='sell',
                        order_type='market',
                        time_in_force='day'
                    )
                    
                    if result:
                        successful_sales += 1
                        print(f"  ✅ Sell order placed successfully")
                    else:
                        failed_sales += 1
                        print(f"  ❌ Sell order failed")
                        
                except Exception as e:
                    failed_sales += 1
                    print(f"  ❌ Error selling {symbol}: {e}")
                
                # Small delay between orders
                await asyncio.sleep(0.5)
            
            print(f"\n📊 SELLING COMPLETE")
            print("-" * 30)
            print(f"✅ Successful sales: {successful_sales}")
            print(f"❌ Failed sales: {failed_sales}")
            print(f"📈 Success rate: {successful_sales/(successful_sales+failed_sales)*100:.1f}%")
            
            if successful_sales > 0:
                print(f"\n⏳ Waiting 5 seconds for orders to settle...")
                await asyncio.sleep(5)
                
                # Check new account status
                account_info = alpaca_client.get_account_info()
                new_positions = alpaca_client.get_positions()
                
                buying_power = float(account_info['buying_power'])
                portfolio_value = float(account_info['equity'])
                
                print(f"\n💰 UPDATED ACCOUNT STATUS:")
                print(f"  Portfolio Value: ${portfolio_value:,.2f}")
                print(f"  Buying Power: ${buying_power:,.2f}")
                print(f"  Active Positions: {len(new_positions)}")
                
                if buying_power > 500:
                    print(f"\n🎉 SUCCESS! Ready for new trades with enhanced sentiment analysis!")
                    print(f"🚀 You can now run trading cycles that will place orders")
                elif buying_power > 100:
                    print(f"\n✅ Good! Moderate buying power available for testing")
                else:
                    print(f"\n⚠️ Limited buying power, but should be enough for small test trades")
                
                return True
            else:
                print(f"\n❌ No positions were sold successfully")
                return False
        
        else:
            print(f"\n❌ Sale cancelled by user")
            print(f"💡 Positions remain unchanged")
            return False
        
    except Exception as e:
        print(f"❌ Error selling positions: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(sell_all_positions())