#!/usr/bin/env python3
"""
Market-Aware Scheduler for Continuous Trading System

Intelligently schedules trading system execution:
- Runs once 30 minutes before market open for pre-market preparation
- Runs continuously during trading hours with intelligent intervals
- Stops after market close with optional after-hours monitoring
"""

import asyncio
import logging
from datetime import datetime, time, timedelta, timezone
from typing import Optional, Dict, Any
from dataclasses import dataclass
import pytz
from pathlib import Path

from tools.alpaca_client import alpaca_client
from continuous_rebalancer import ContinuousRebalancer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/market_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MarketSchedule:
    """Market schedule information."""
    market_open: datetime
    market_close: datetime
    pre_market_start: datetime  # 30 minutes before open
    is_trading_day: bool
    next_trading_day: Optional[datetime] = None

class MarketAwareScheduler:
    """Scheduler that manages trading system based on market hours."""
    
    def __init__(self):
        """Initialize the market-aware scheduler."""
        self.rebalancer = None
        self.running = False
        self.current_mode = "stopped"  # stopped, pre_market, trading, after_hours
        
        # Scheduling parameters
        self.pre_market_minutes = 30  # Start 30 minutes before market open
        self.trading_interval_minutes = 15  # Run every 15 minutes during trading
        self.max_after_hours_runs = 1  # Maximum runs after market close
        self.after_hours_runs_today = 0
        
        # Timezone for market hours (US Eastern)
        self.market_tz = pytz.timezone('US/Eastern')
        
        # State persistence
        self.state_file = Path("data/scheduler_state.json")
        self.state_file.parent.mkdir(exist_ok=True)
        
        logger.info("Market-aware scheduler initialized")
    
    async def start_market_aware_trading(self):
        """Start market-aware trading system."""
        logger.info("🚀 STARTING MARKET-AWARE TRADING SYSTEM")
        logger.info("=" * 70)
        
        self.running = True
        
        try:
            while self.running:
                # Get current market schedule
                schedule = await self._get_market_schedule()
                
                if not schedule.is_trading_day:
                    logger.info(f"📅 Not a trading day. Next trading day: {schedule.next_trading_day}")
                    await self._wait_until_next_trading_day(schedule)
                    continue
                
                # Determine current trading phase
                now = datetime.now(self.market_tz)
                current_phase = self._get_trading_phase(now, schedule)
                
                if current_phase != self.current_mode:
                    logger.info(f"🔄 Trading phase changed: {self.current_mode} → {current_phase}")
                    self.current_mode = current_phase
                    await self._handle_phase_transition(current_phase, schedule)
                
                # Execute actions based on current phase
                await self._execute_phase_actions(current_phase, schedule)
                
                # Sleep until next check
                sleep_duration = self._calculate_sleep_duration(current_phase, schedule)
                logger.info(f"⏰ Next check in {sleep_duration} minutes")
                await asyncio.sleep(sleep_duration * 60)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            logger.error(f"Fatal error in market-aware scheduler: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self._shutdown()
    
    async def _get_market_schedule(self) -> MarketSchedule:
        """Get today's market schedule."""
        try:
            # Check if market is open using Alpaca's simple API
            market_open_now = alpaca_client.is_market_open()
            logger.info(f"Market currently open: {market_open_now}")
            
            # Use standard market hours (9:30 AM - 4:00 PM ET)
            now = datetime.now(self.market_tz)
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            pre_market_start = market_open - timedelta(minutes=self.pre_market_minutes)
            
            # Simple check: if it's a weekday, assume it's a trading day
            # (This could be enhanced to check for holidays)
            is_trading_day = now.weekday() < 5  # Monday=0, Friday=4
            
            return MarketSchedule(
                market_open=market_open,
                market_close=market_close,
                pre_market_start=pre_market_start,
                is_trading_day=is_trading_day
            )
                
        except Exception as e:
            logger.error(f"Failed to get market schedule: {e}")
            # Fallback schedule (standard market hours)
            now = datetime.now(self.market_tz)
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            
            return MarketSchedule(
                market_open=market_open,
                market_close=market_close,
                pre_market_start=market_open - timedelta(minutes=self.pre_market_minutes),
                is_trading_day=True
            )
    
    def _get_trading_phase(self, now: datetime, schedule: MarketSchedule) -> str:
        """Determine current trading phase."""
        if now < schedule.pre_market_start:
            return "waiting"
        elif schedule.pre_market_start <= now < schedule.market_open:
            return "pre_market"
        elif schedule.market_open <= now < schedule.market_close:
            return "trading"
        else:
            return "after_hours"
    
    async def _handle_phase_transition(self, new_phase: str, schedule: MarketSchedule):
        """Handle transition between trading phases."""
        if new_phase == "pre_market":
            logger.info("🌅 PRE-MARKET PHASE: Preparing for trading day")
            logger.info(f"Market opens at: {schedule.market_open.strftime('%H:%M %Z')}")
            # Reset after-hours run counter
            self.after_hours_runs_today = 0
            
        elif new_phase == "trading":
            logger.info("📈 TRADING PHASE: Market is open - starting continuous rebalancing")
            logger.info(f"Market closes at: {schedule.market_close.strftime('%H:%M %Z')}")
            
        elif new_phase == "after_hours":
            logger.info("🌙 AFTER-HOURS PHASE: Market closed")
            # Stop continuous rebalancer if running
            if self.rebalancer and self.rebalancer.running:
                logger.info("Stopping continuous rebalancer...")
                self.rebalancer.running = False
                self.rebalancer = None
    
    async def _execute_phase_actions(self, phase: str, schedule: MarketSchedule):
        """Execute actions for current trading phase."""
        if phase == "waiting":
            # Just wait - no action needed
            pass
            
        elif phase == "pre_market":
            # Run once 30 minutes before market open
            logger.info("🔧 PRE-MARKET: Running initial portfolio setup")
            await self._run_single_rebalancing_cycle("pre_market_preparation")
            
        elif phase == "trading":
            # Ensure continuous rebalancer is running
            if not self.rebalancer or not self.rebalancer.running:
                logger.info("🚀 Starting continuous rebalancing for trading hours")
                self.rebalancer = ContinuousRebalancer()
                # Override intervals for trading hours
                self.rebalancer.min_interval_minutes = 10
                self.rebalancer.standard_interval_minutes = self.trading_interval_minutes
                
                # Start continuous operation (non-blocking)
                asyncio.create_task(self.rebalancer.start_continuous_operation())
            
        elif phase == "after_hours":
            # Run once after market close if haven't reached limit
            if self.after_hours_runs_today < self.max_after_hours_runs:
                logger.info("🌙 AFTER-HOURS: Running end-of-day portfolio review")
                await self._run_single_rebalancing_cycle("after_hours_review")
                self.after_hours_runs_today += 1
    
    async def _run_single_rebalancing_cycle(self, cycle_type: str):
        """Run a single rebalancing cycle."""
        try:
            logger.info(f"🔄 Running {cycle_type} rebalancing cycle")
            
            # Import here to avoid circular imports
            from agents.workflow import TradingWorkflow
            from agents.state import create_initial_state
            
            workflow = TradingWorkflow()
            state = create_initial_state()
            config = {"thread_id": f"market_scheduler_{cycle_type}_{int(datetime.now().timestamp())}"}
            
            # Run simplified pipeline
            logger.info("📊 Market Monitor...")
            state = await workflow.market_monitor_agent(state, config)
            
            logger.info("🔍 Universe Filter...")
            state = await workflow.universe_filter_agent(state, config)
            
            logger.info("💭 Sentiment Analysis...")
            state = await workflow.sentiment_analysis_agent(state, config)
            
            logger.info("⚖️ Risk Assessment...")
            state = await workflow.risk_assessment_agent(state, config)
            
            logger.info("🧠 Signal Generation...")
            state = await workflow.signal_generation_agent(state, config)
            
            signals_generated = len(state.get("signals", []))
            
            if signals_generated > 0:
                logger.info("🎯 Strategy Optimization...")
                state = await workflow.strategy_optimization_agent(state, config)
                
                logger.info("💼 Order Management...")
                state = await workflow.order_management_agent(state, config)
            
            logger.info("📈 Portfolio Tracking...")
            state = await workflow.portfolio_tracking_agent(state, config)
            
            portfolio_value = state["portfolio"].get("equity", 0)
            logger.info(f"✅ {cycle_type} cycle complete: {signals_generated} signals, ${portfolio_value:,.2f} portfolio")
            
        except Exception as e:
            logger.error(f"Error in {cycle_type} cycle: {e}")
            import traceback
            traceback.print_exc()
    
    def _calculate_sleep_duration(self, phase: str, schedule: MarketSchedule) -> int:
        """Calculate how long to sleep before next check."""
        now = datetime.now(self.market_tz)
        
        if phase == "waiting":
            # Sleep until pre-market start
            time_until_pre_market = (schedule.pre_market_start - now).total_seconds()
            return max(1, int(time_until_pre_market / 60))  # Convert to minutes, minimum 1
            
        elif phase == "pre_market":
            # Sleep until market open
            time_until_open = (schedule.market_open - now).total_seconds()
            return max(1, int(time_until_open / 60))
            
        elif phase == "trading":
            # Check every few minutes during trading (rebalancer handles its own timing)
            return 5
            
        elif phase == "after_hours":
            # Check every hour after market close
            return 60
            
        else:
            return 30  # Default
    
    async def _wait_until_next_trading_day(self, schedule: MarketSchedule):
        """Wait until the next trading day."""
        if schedule.next_trading_day:
            wait_time = (schedule.next_trading_day - datetime.now(self.market_tz)).total_seconds()
            wait_hours = wait_time / 3600
            
            logger.info(f"⏳ Waiting {wait_hours:.1f} hours until next trading day...")
            
            # Sleep in chunks to allow for graceful shutdown
            while wait_time > 0 and self.running:
                sleep_chunk = min(3600, wait_time)  # Sleep max 1 hour at a time
                await asyncio.sleep(sleep_chunk)
                wait_time -= sleep_chunk
        else:
            # Fallback: wait 12 hours and check again
            logger.info("⏳ No next trading day found, waiting 12 hours...")
            await asyncio.sleep(12 * 3600)
    
    async def _shutdown(self):
        """Perform graceful shutdown."""
        logger.info("🛑 SHUTTING DOWN MARKET-AWARE SCHEDULER")
        
        # Stop continuous rebalancer if running
        if self.rebalancer and self.rebalancer.running:
            logger.info("Stopping continuous rebalancer...")
            self.rebalancer.running = False
        
        self.running = False
        logger.info("👋 Market-aware scheduler shutdown complete")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            "running": self.running,
            "current_mode": self.current_mode,
            "rebalancer_running": bool(self.rebalancer and self.rebalancer.running),
            "after_hours_runs_today": self.after_hours_runs_today,
            "max_after_hours_runs": self.max_after_hours_runs
        }

# Global instance and convenience functions
market_scheduler = MarketAwareScheduler()

async def start_market_aware_trading():
    """Start the market-aware trading system."""
    await market_scheduler.start_market_aware_trading()

def get_scheduler_status():
    """Get current scheduler status."""
    return market_scheduler.get_status()

if __name__ == "__main__":
    print("🚀 Starting Market-Aware Trading System...")
    print("This will automatically start 30 minutes before market open and run continuously during trading hours.")
    print("Press Ctrl+C to stop gracefully.")
    print("=" * 70)
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    try:
        asyncio.run(start_market_aware_trading())
    except KeyboardInterrupt:
        print("\n👋 Market-aware trading system stopped by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()