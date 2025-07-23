#!/usr/bin/env python3
"""
Daily Summary Email Scheduler

Handles scheduling and sending daily trading summary emails at market close.
Integrates with the existing email notification system and Alpaca trading client.
"""

import asyncio
import logging
from datetime import datetime, time
import pytz
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from notifications.email_webhooks import send_daily_summary_email, email_notifier
from tools.alpaca_client import AlpacaClient
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class TradingDayTracker:
    """Track trading activity for daily summary."""
    trades_executed: int = 0
    portfolio_start_value: float = 0.0
    last_summary_date: str = ""

class DailySummaryScheduler:
    """Scheduler for sending daily trading summary emails."""
    
    def __init__(self, alpaca_client: Optional[AlpacaClient] = None):
        self.alpaca_client = alpaca_client or AlpacaClient()
        self.trading_tracker = TradingDayTracker()
        self.is_running = False
        self.summary_sent_today = False
        
    async def start_scheduler(self):
        """Start the daily summary scheduler."""
        self.is_running = True
        logger.info("📧 Daily summary scheduler started")
        
        while self.is_running:
            try:
                # Check if we should send daily summary
                if await self.should_send_summary():
                    await self.send_daily_summary()
                
                # Check every 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in daily summary scheduler: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    def stop_scheduler(self):
        """Stop the daily summary scheduler."""
        self.is_running = False
        logger.info("📧 Daily summary scheduler stopped")
    
    async def should_send_summary(self) -> bool:
        """Check if we should send the daily summary email."""
        # Check if market just closed and we haven't sent today's summary
        if not email_notifier.is_market_closed():
            return False
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Reset flag for new day
        if self.trading_tracker.last_summary_date != today:
            self.summary_sent_today = False
        
        if self.summary_sent_today:
            return False
        
        # Check if it's within 30 minutes of market close (4:00-4:30 PM EST)
        est = pytz.timezone('US/Eastern')
        now = datetime.now(est)
        
        # Don't send on weekends
        if now.weekday() >= 5:
            return False
        
        current_time = now.time()
        send_window_start = time(16, 0)   # 4:00 PM EST
        send_window_end = time(16, 30)    # 4:30 PM EST
        
        return send_window_start <= current_time <= send_window_end
    
    async def send_daily_summary(self):
        """Generate and send the daily trading summary email."""
        try:
            logger.info("📊 Generating daily trading summary...")
            
            # Get portfolio data
            account_data = await self.get_account_data()
            positions_data = await self.get_positions_data()
            
            if not account_data:
                logger.error("Failed to get account data, skipping daily summary")
                return
            
            # Calculate day change
            current_portfolio_value = account_data['portfolio_value']
            if self.trading_tracker.portfolio_start_value == 0:
                day_change = 0.0
            else:
                day_change = current_portfolio_value - self.trading_tracker.portfolio_start_value
            
            # Determine portfolio performance description
            performance = self.get_performance_description(day_change, current_portfolio_value)
            
            # Send the daily summary email
            success = await send_daily_summary_email(
                portfolio_value=current_portfolio_value,
                cash_balance=account_data['cash_balance'],
                total_equity=account_data['total_equity'],
                day_change=day_change,
                positions_data=positions_data,
                trades_today=self.trading_tracker.trades_executed,
                portfolio_performance=performance
            )
            
            if success:
                logger.info("✅ Daily summary email sent successfully")
                self.summary_sent_today = True
                self.trading_tracker.last_summary_date = datetime.now().strftime('%Y-%m-%d')
            else:
                logger.error("❌ Failed to send daily summary email")
                
        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")
    
    async def get_account_data(self) -> Optional[Dict[str, float]]:
        """Get account data from Alpaca."""
        try:
            account = self.alpaca_client.get_account()
            
            return {
                'portfolio_value': float(account.portfolio_value),
                'cash_balance': float(account.cash),
                'total_equity': float(account.equity),
                'buying_power': float(account.buying_power)
            }
            
        except Exception as e:
            logger.error(f"Error getting account data: {e}")
            return None
    
    async def get_positions_data(self) -> List[Dict[str, Any]]:
        """Get positions data from Alpaca."""
        try:
            positions = self.alpaca_client.list_positions()
            positions_data = []
            
            for position in positions:
                pos_data = {
                    'symbol': position.symbol,
                    'quantity': float(position.qty),
                    'market_value': float(position.market_value),
                    'unrealized_pnl': float(position.unrealized_pl),
                    'percent_change': float(position.unrealized_plpc) * 100,
                    'side': 'long' if float(position.qty) > 0 else 'short'
                }
                positions_data.append(pos_data)
            
            return positions_data
            
        except Exception as e:
            logger.error(f"Error getting positions data: {e}")
            return []
    
    def get_performance_description(self, day_change: float, portfolio_value: float) -> str:
        """Get a descriptive text for portfolio performance."""
        if portfolio_value == 0:
            return "No positions"
        
        # Calculate change percentage properly
        if portfolio_value == day_change:  # Avoid division by zero
            change_percent = 0.0
        else:
            change_percent = (day_change / (portfolio_value - day_change)) * 100
        
        if change_percent > 2:
            return "Strong gains"
        elif change_percent > 0.5:
            return "Positive performance"
        elif change_percent > -0.5:
            return "Stable performance"
        elif change_percent > -2:
            return "Minor losses"
        else:
            return "Significant losses"
    
    def record_trade(self):
        """Record that a trade was executed today."""
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Reset counter for new day
        if self.trading_tracker.last_summary_date != today:
            self.trading_tracker.trades_executed = 0
        
        self.trading_tracker.trades_executed += 1
        logger.debug(f"Trade recorded. Total today: {self.trading_tracker.trades_executed}")
    
    async def initialize_day_tracking(self):
        """Initialize tracking for the trading day."""
        try:
            account_data = await self.get_account_data()
            if account_data:
                self.trading_tracker.portfolio_start_value = account_data['portfolio_value']
                logger.info(f"📊 Day tracking initialized. Starting portfolio value: ${self.trading_tracker.portfolio_start_value:,.2f}")
        except Exception as e:
            logger.error(f"Error initializing day tracking: {e}")
    
    async def test_daily_summary(self):
        """Test the daily summary email functionality."""
        logger.info("🧪 Testing daily summary email...")
        
        try:
            # Get real account data
            account_data = await self.get_account_data()
            positions_data = await self.get_positions_data()
            
            if not account_data:
                # Use mock data for testing
                logger.warning("Using mock data for testing")
                account_data = {
                    'portfolio_value': 100000.0,
                    'cash_balance': 25000.0,
                    'total_equity': 100000.0
                }
                positions_data = [
                    {
                        'symbol': 'AAPL',
                        'quantity': 50.0,
                        'market_value': 8750.0,
                        'unrealized_pnl': 250.0,
                        'percent_change': 2.9,
                        'side': 'long'
                    },
                    {
                        'symbol': 'TSLA',
                        'quantity': -25.0,
                        'market_value': 6250.0,
                        'unrealized_pnl': -125.0,
                        'percent_change': -2.0,
                        'side': 'short'
                    }
                ]
            
            # Send test summary
            success = await send_daily_summary_email(
                portfolio_value=account_data['portfolio_value'],
                cash_balance=account_data['cash_balance'],
                total_equity=account_data['total_equity'],
                day_change=500.0,  # Test day change
                positions_data=positions_data,
                trades_today=3,
                portfolio_performance="Test performance"
            )
            
            if success:
                logger.info("✅ Test daily summary email sent successfully")
                return True
            else:
                logger.error("❌ Test daily summary email failed")
                return False
                
        except Exception as e:
            logger.error(f"Error testing daily summary: {e}")
            return False

# Global scheduler instance
daily_scheduler = DailySummaryScheduler()

async def start_daily_summary_service():
    """Start the daily summary service."""
    await daily_scheduler.initialize_day_tracking()
    await daily_scheduler.start_scheduler()

def record_trade_for_summary():
    """Record a trade for the daily summary."""
    daily_scheduler.record_trade()

if __name__ == "__main__":
    # Test the daily summary functionality
    async def main():
        print("🧪 Testing Daily Summary Email System")
        print("=" * 50)
        
        scheduler = DailySummaryScheduler()
        
        # Test daily summary
        success = await scheduler.test_daily_summary()
        
        if success:
            print("🎉 Daily summary email system is working!")
        else:
            print("⚠️ Daily summary email system failed. Check configuration.")
        
        # Test market status detection
        print(f"\n📊 Current market status: {email_notifier.get_market_status()}")
        print(f"📅 Market is closed: {email_notifier.is_market_closed()}")
        print(f"📧 Should send summary: {await scheduler.should_send_summary()}")
    
    asyncio.run(main())