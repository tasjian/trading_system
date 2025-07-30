#!/usr/bin/env python3
"""
Check Trading System Status
Quick status check for the trading system and account.
"""

import asyncio
from datetime import datetime
import pytz
from tools.alpaca_client import alpaca_client
from market_aware_scheduler import MarketAwareScheduler

async def check_status():
    """Check current trading system status."""
    print("📊 TRADING SYSTEM STATUS CHECK")
    print("=" * 50)
    
    try:
        # Check account
        account = alpaca_client.get_account_info()
        print(f"💰 Account Status:")
        print(f"   Portfolio Value: ${float(account['equity']):,.2f}")
        print(f"   Available Cash: ${float(account['cash']):,.2f}")
        print(f"   Buying Power: ${float(account['buying_power']):,.2f}")
        
        # Check positions
        positions = alpaca_client.get_positions()
        print(f"\n📈 Current Positions: {len(positions)}")
        if positions:
            total_position_value = 0
            for pos in positions:
                qty = float(pos['qty'])
                market_value = float(pos['market_value'])
                total_position_value += abs(market_value)
                side = "LONG" if qty > 0 else "SHORT"
                print(f"   {pos['symbol']}: {qty} shares, ${market_value:,.2f} ({side})")
            print(f"   Total Position Value: ${total_position_value:,.2f}")
        
        # Check market status
        market_open = alpaca_client.is_market_open()
        print(f"\n🕐 Market Status: {'OPEN' if market_open else 'CLOSED'}")
        
        # Check trading phase
        scheduler = MarketAwareScheduler()
        schedule = await scheduler._get_market_schedule()
        
        now = datetime.now(pytz.timezone('US/Eastern'))
        current_phase = scheduler._get_trading_phase(now, schedule)
        
        print(f"📅 Trading Phase: {current_phase.upper()}")
        print(f"   Current Time: {now.strftime('%H:%M %Z')}")
        
        if schedule.is_trading_day:
            print(f"   Pre-Market: {schedule.pre_market_start.strftime('%H:%M %Z')}")
            print(f"   Market Open: {schedule.market_open.strftime('%H:%M %Z')}")
            print(f"   Market Close: {schedule.market_close.strftime('%H:%M %Z')}")
            
            # Time until next phase
            if current_phase == "waiting":
                time_until = (schedule.pre_market_start - now).total_seconds() / 3600
                print(f"   ⏰ Pre-market starts in: {time_until:.1f} hours")
            elif current_phase == "pre_market":
                time_until = (schedule.market_open - now).total_seconds() / 60
                print(f"   ⏰ Market opens in: {time_until:.0f} minutes")
            elif current_phase == "trading":
                time_until = (schedule.market_close - now).total_seconds() / 60
                print(f"   ⏰ Market closes in: {time_until:.0f} minutes")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if current_phase == "waiting":
            print("   System will automatically start 30 minutes before market open")
        elif current_phase == "pre_market":
            print("   Run initial portfolio preparation")
        elif current_phase == "trading":
            print("   Continuous rebalancing should be active")
        elif current_phase == "after_hours":
            print("   Consider running end-of-day review if not done yet")
        
        print(f"\n🚀 To start the automated system: python start_trading_system.py")
        
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_status())