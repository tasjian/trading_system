#!/usr/bin/env python3
"""
Market Open Scheduler

Handles cache flushing and pipeline refresh at market open (7:30 AM ET each trading day).
Ensures fresh data and analysis for each trading session.
"""

import asyncio
import logging
import redis
from datetime import datetime, time, timedelta
from typing import Optional
import pytz
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from config.settings import settings

logger = logging.getLogger(__name__)


class MarketOpenScheduler:
    """Scheduler for market open activities."""
    
    def __init__(self):
        """Initialize market open scheduler."""
        self.redis_client = None
        self.eastern_tz = pytz.timezone('US/Eastern')
        self.market_open_time = time(7, 30)  # 7:30 AM ET (30 min before market open)
        self.holiday_calendar = USFederalHolidayCalendar()
        self.last_flush_date = None
        self.pipeline_callback = None
        
        # Initialize Redis connection
        self._setup_redis()
        
        logger.info("🕐 Market Open Scheduler initialized for 7:30 AM ET cache flush")
    
    def _setup_redis(self):
        """Setup Redis connection for cache management."""
        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host if hasattr(settings, 'redis_host') else 'localhost',
                port=settings.redis_port if hasattr(settings, 'redis_port') else 6379,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Redis connection established for cache management")
            
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            logger.warning("Cache flushing will be skipped if Redis is unavailable")
            self.redis_client = None
    
    def is_trading_day(self, date: datetime) -> bool:
        """Check if given date is a trading day (not weekend or holiday)."""
        try:
            # Check if it's a weekend
            if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
            
            # Check if it's a federal holiday
            holidays = self.holiday_calendar.holidays(
                start=date - timedelta(days=1),
                end=date + timedelta(days=1),
                return_name=True
            )
            
            return date.date() not in holidays.index.date
            
        except Exception as e:
            logger.warning(f"Error checking trading day status: {e}")
            # Conservative approach - assume it's a trading day if check fails
            return date.weekday() < 5
    
    def get_next_market_open(self, current_time: Optional[datetime] = None) -> datetime:
        """Get the next market open time (7:30 AM ET on next trading day)."""
        if current_time is None:
            current_time = datetime.now(self.eastern_tz)
        
        # Start with today
        candidate_date = current_time.date()
        candidate_time = datetime.combine(candidate_date, self.market_open_time)
        candidate_time = self.eastern_tz.localize(candidate_time)
        
        # If we've already passed today's market open time, start with tomorrow
        if current_time >= candidate_time:
            candidate_date += timedelta(days=1)
            candidate_time = datetime.combine(candidate_date, self.market_open_time)
            candidate_time = self.eastern_tz.localize(candidate_time)
        
        # Find next trading day
        max_attempts = 10  # Prevent infinite loops
        attempts = 0
        
        while not self.is_trading_day(candidate_time) and attempts < max_attempts:
            candidate_date += timedelta(days=1)
            candidate_time = datetime.combine(candidate_date, self.market_open_time)
            candidate_time = self.eastern_tz.localize(candidate_time)
            attempts += 1
        
        if attempts >= max_attempts:
            logger.warning("Could not find next trading day within 10 days, using next weekday")
            # Fallback to next weekday
            while candidate_time.weekday() >= 5:
                candidate_date += timedelta(days=1)
                candidate_time = datetime.combine(candidate_date, self.market_open_time)
                candidate_time = self.eastern_tz.localize(candidate_time)
        
        return candidate_time
    
    def should_flush_cache(self, current_time: Optional[datetime] = None) -> bool:
        """Check if cache should be flushed based on market open schedule."""
        if current_time is None:
            current_time = datetime.now(self.eastern_tz)
        
        today = current_time.date()
        
        # Don't flush on non-trading days
        if not self.is_trading_day(current_time):
            return False
        
        # Don't flush if we already flushed today
        if self.last_flush_date == today:
            return False
        
        # Check if we're at or past the market open time
        market_open_today = datetime.combine(today, self.market_open_time)
        market_open_today = self.eastern_tz.localize(market_open_today)
        
        return current_time >= market_open_today
    
    async def flush_redis_cache(self) -> bool:
        """Flush Redis cache to ensure fresh data for the trading day."""
        if not self.redis_client:
            logger.warning("⚠️ Redis not available, skipping cache flush")
            return False
        
        try:
            logger.info("🧹 Flushing Redis cache for market open...")
            
            # Get cache statistics before flush
            cache_info = self.redis_client.info()
            keys_before = cache_info.get('db0', {}).get('keys', 0) if 'db0' in cache_info else 0
            
            # Flush all databases
            self.redis_client.flushall()
            
            # Verify flush
            cache_info_after = self.redis_client.info()
            keys_after = cache_info_after.get('db0', {}).get('keys', 0) if 'db0' in cache_info_after else 0
            
            logger.info(f"✅ Redis cache flushed successfully: {keys_before} → {keys_after} keys")
            
            # Update last flush date
            self.last_flush_date = datetime.now(self.eastern_tz).date()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to flush Redis cache: {e}")
            return False
    
    def set_pipeline_callback(self, callback):
        """Set callback function to trigger full pipeline refresh."""
        self.pipeline_callback = callback
        logger.info("🔄 Pipeline refresh callback registered")
    
    async def trigger_pipeline_refresh(self) -> bool:
        """Trigger full pipeline refresh including universe filter and sentiment analysis."""
        try:
            logger.info("🚀 Triggering full pipeline refresh for market open...")
            
            if self.pipeline_callback:
                # Call the registered callback function
                await self.pipeline_callback()
                logger.info("✅ Pipeline refresh completed successfully")
                return True
            else:
                logger.warning("⚠️ No pipeline callback registered, skipping refresh")
                return False
                
        except Exception as e:
            logger.error(f"❌ Pipeline refresh failed: {e}")
            return False
    
    async def run_market_open_routine(self) -> bool:
        """Run the complete market open routine: cache flush + pipeline refresh."""
        current_time = datetime.now(self.eastern_tz)
        
        logger.info(f"🌅 Running market open routine at {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # Check if we should run the routine
        if not self.should_flush_cache(current_time):
            logger.info("⏭️ Market open routine not needed (already ran today or not a trading day)")
            return False
        
        success = True
        
        # Step 1: Flush Redis cache
        cache_flushed = await self.flush_redis_cache()
        if not cache_flushed:
            logger.warning("⚠️ Cache flush failed, but continuing with pipeline refresh")
            success = False
        
        # Step 2: Trigger pipeline refresh
        pipeline_refreshed = await self.trigger_pipeline_refresh()
        if not pipeline_refreshed:
            logger.error("❌ Pipeline refresh failed")
            success = False
        
        if success:
            logger.info("🎉 Market open routine completed successfully!")
        else:
            logger.warning("⚠️ Market open routine completed with some issues")
        
        return success
    
    async def start_scheduler(self):
        """Start the market open scheduler loop."""
        logger.info("▶️ Starting market open scheduler...")
        
        while True:
            try:
                current_time = datetime.now(self.eastern_tz)
                next_market_open = self.get_next_market_open(current_time)
                
                time_until_open = next_market_open - current_time
                seconds_until_open = time_until_open.total_seconds()
                
                logger.info(f"⏰ Next market open: {next_market_open.strftime('%Y-%m-%d %H:%M:%S %Z')} "
                           f"(in {time_until_open})")
                
                if seconds_until_open > 0:
                    # Wait until next market open
                    logger.info(f"💤 Sleeping for {time_until_open} until next market open...")
                    await asyncio.sleep(seconds_until_open)
                
                # Run market open routine
                await self.run_market_open_routine()
                
                # Small delay to prevent immediate re-trigger
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"❌ Error in market open scheduler: {e}")
                # Wait 5 minutes before retrying
                await asyncio.sleep(300)
    
    async def manual_market_open_routine(self) -> bool:
        """Manually trigger market open routine (for testing/debugging)."""
        logger.info("🔧 Manually triggering market open routine...")
        
        # Temporarily override last_flush_date to force execution
        original_date = self.last_flush_date
        self.last_flush_date = None
        
        try:
            result = await self.run_market_open_routine()
            return result
        finally:
            # Restore original date if the routine failed
            if not result:
                self.last_flush_date = original_date
    
    def get_status(self) -> dict:
        """Get scheduler status information."""
        current_time = datetime.now(self.eastern_tz)
        next_market_open = self.get_next_market_open(current_time)
        
        return {
            "current_time": current_time.isoformat(),
            "next_market_open": next_market_open.isoformat(),
            "is_trading_day_today": self.is_trading_day(current_time),
            "last_flush_date": self.last_flush_date.isoformat() if self.last_flush_date else None,
            "redis_connected": self.redis_client is not None,
            "pipeline_callback_set": self.pipeline_callback is not None,
            "time_until_next_open": str(next_market_open - current_time)
        }


# Global scheduler instance
market_open_scheduler = MarketOpenScheduler()


async def start_market_open_scheduler():
    """Convenience function to start the market open scheduler."""
    await market_open_scheduler.start_scheduler()


if __name__ == "__main__":
    # Test the scheduler
    async def test_scheduler():
        scheduler = MarketOpenScheduler()
        
        print("📊 Scheduler Status:")
        status = scheduler.get_status()
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        print(f"\n🔍 Should flush cache now? {scheduler.should_flush_cache()}")
        
        # Test manual routine
        print("\n🧪 Testing manual market open routine...")
        success = await scheduler.manual_market_open_routine()
        print(f"Result: {'✅ Success' if success else '❌ Failed'}")
    
    asyncio.run(test_scheduler())