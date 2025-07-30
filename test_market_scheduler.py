#!/usr/bin/env python3
"""
Test Market-Aware Scheduler
Quick test to show how the scheduler determines trading phases and timing.
"""

import asyncio
from datetime import datetime
import pytz
from market_aware_scheduler import MarketAwareScheduler

async def test_market_scheduler():
    """Test the market scheduler logic."""
    print("🔍 TESTING MARKET-AWARE SCHEDULER")
    print("=" * 50)
    
    scheduler = MarketAwareScheduler()
    
    # Get market schedule
    schedule = await scheduler._get_market_schedule()
    
    print(f"📅 Market Schedule for Today:")
    print(f"   Is Trading Day: {schedule.is_trading_day}")
    
    if schedule.is_trading_day:
        print(f"   Pre-Market Start: {schedule.pre_market_start.strftime('%H:%M %Z')}")
        print(f"   Market Open: {schedule.market_open.strftime('%H:%M %Z')}")
        print(f"   Market Close: {schedule.market_close.strftime('%H:%M %Z')}")
        
        # Show current trading phase
        now = datetime.now(scheduler.market_tz)
        current_phase = scheduler._get_trading_phase(now, schedule)
        print(f"\n🕐 Current Time: {now.strftime('%H:%M %Z')}")
        print(f"📊 Current Phase: {current_phase}")
        
        # Calculate sleep duration
        sleep_duration = scheduler._calculate_sleep_duration(current_phase, schedule)
        print(f"⏰ Next check in: {sleep_duration} minutes")
        
        # Show what would happen in each phase
        print(f"\n📋 Scheduler Behavior:")
        print(f"   🌅 PRE-MARKET ({schedule.pre_market_start.strftime('%H:%M')} - {schedule.market_open.strftime('%H:%M')}): Run once for portfolio prep")
        print(f"   📈 TRADING ({schedule.market_open.strftime('%H:%M')} - {schedule.market_close.strftime('%H:%M')}): Continuous rebalancing every 15 min")
        print(f"   🌙 AFTER-HOURS (after {schedule.market_close.strftime('%H:%M')}): Run once for end-of-day review")
        
    else:
        print(f"   Next Trading Day: {schedule.next_trading_day}")
    
    # Show scheduler status
    status = scheduler.get_status()
    print(f"\n📊 Scheduler Status:")
    for key, value in status.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    asyncio.run(test_market_scheduler())