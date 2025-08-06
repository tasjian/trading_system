#!/usr/bin/env python3
"""
Continuous Rebalancing System
Runs end-to-end rebalancing pipeline continuously with intelligent scheduling,
rate limiting, error handling, and autonomous operation capabilities.
"""

# Apply aggressive universe filter fixes to prevent hanging issues
try:
    from fix_universe_filter_aggressive import patch_universe_filter_aggressive
    patch_universe_filter_aggressive()
    print("✅ Applied AGGRESSIVE universe filter patches")
    print("   - Disabled: social, news, earnings collection")  
    print("   - Enabled: price movement signals only")
except ImportError:
    print("⚠️ Universe filter patches not found - running without fixes")

# Apply complete workflow bypass to eliminate session leaks
try:
    from fix_complete_workflow import patch_complete_workflow
    patch_complete_workflow()
    print("✅ Applied COMPLETE WORKFLOW BYPASS patches")
    print("   - Sentiment analysis: COMPLETELY BYPASSED")
    print("   - HTTP sessions: No more leaks from sentiment collection")
    print("   - Performance: Maximum speed for continuous rebalancing")
except ImportError:
    print("⚠️ Complete workflow patches not found - running without fixes")

# Apply optimized social media collector to prevent session leaks
try:
    from patch_optimized_social_collector import patch_optimized_social_collector
    patch_optimized_social_collector()
    print("✅ Applied OPTIMIZED SOCIAL COLLECTOR patches")
    print("   - Fast 15s timeouts prevent hanging")
    print("   - Proper session cleanup eliminates leaks")
    print("   - Fallback posts when APIs unavailable")
except ImportError:
    print("⚠️ Optimized social collector patch not found - running without fixes")

import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta, time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import traceback

from agents.workflow import TradingWorkflow
from agents.state import create_initial_state
from tools.alpaca_client import alpaca_client
from config.settings import settings

# Configure logging with rotation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/continuous_rebalancer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RebalanceResult:
    """Result of a rebalancing operation."""
    timestamp: datetime
    success: bool
    duration_seconds: float
    signals_generated: int
    orders_executed: int
    portfolio_value: float
    error_message: Optional[str] = None
    pipeline_stage: Optional[str] = None

@dataclass
class SystemHealth:
    """System health metrics."""
    uptime_hours: float
    total_runs: int
    successful_runs: int
    failed_runs: int
    avg_runtime_seconds: float
    last_success: Optional[datetime]
    last_failure: Optional[datetime]
    api_rate_limit_hits: int
    consecutive_failures: int

class ContinuousRebalancer:
    """Autonomous continuous rebalancing system with intelligent scheduling."""
    
    def __init__(self):
        """Initialize the continuous rebalancer."""
        self.workflow = TradingWorkflow()
        self.running = False
        self.start_time = None
        
        # Rate limiting and timing
        self.min_interval_minutes = 15      # Minimum time between runs
        self.standard_interval_minutes = 30  # Standard interval during market hours
        self.after_hours_interval_minutes = 60  # Longer interval after hours
        self.last_run_time = None
        
        # Error handling
        self.max_consecutive_failures = 5
        self.failure_backoff_minutes = 60  # Wait longer after failures
        self.api_cooldown_minutes = 5      # Cooldown after API rate limits
        
        # State tracking
        self.health = SystemHealth(
            uptime_hours=0.0,
            total_runs=0,
            successful_runs=0,
            failed_runs=0,
            avg_runtime_seconds=0.0,
            last_success=None,
            last_failure=None,
            api_rate_limit_hits=0,
            consecutive_failures=0
        )
        
        # Results history
        self.results_history: List[RebalanceResult] = []
        self.max_history_size = 100
        
        # Persistence
        self.state_file = Path("data/continuous_rebalancer_state.json")
        self.state_file.parent.mkdir(exist_ok=True)
        
        # Load previous state if exists
        self._load_state()
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    async def start_continuous_operation(self):
        """Start continuous rebalancing operation."""
        logger.info("🚀 STARTING CONTINUOUS REBALANCING SYSTEM")
        logger.info("=" * 70)
        
        self.running = True
        self.start_time = datetime.now()
        
        logger.info(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Min Interval: {self.min_interval_minutes} minutes")
        logger.info(f"Standard Interval: {self.standard_interval_minutes} minutes")
        logger.info(f"After Hours Interval: {self.after_hours_interval_minutes} minutes")
        logger.info(f"Max Consecutive Failures: {self.max_consecutive_failures}")
        
        # Initial system check
        await self._perform_system_check()
        
        try:
            while self.running:
                await self._continuous_loop_iteration()
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down gracefully...")
        except Exception as e:
            logger.error(f"Fatal error in continuous operation: {e}")
            traceback.print_exc()
        finally:
            await self._shutdown()
    
    async def _continuous_loop_iteration(self):
        """Single iteration of the continuous loop."""
        try:
            # Check if we should run now
            if not await self._should_run_now():
                await asyncio.sleep(60)  # Check again in 1 minute
                return
            
            # Update health metrics
            if self.start_time:
                self.health.uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
            
            # Run rebalancing pipeline
            logger.info(f"\n🔄 STARTING REBALANCING CYCLE #{self.health.total_runs + 1}")
            logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            result = await self._run_rebalancing_pipeline()
            
            # Process results
            self._process_result(result)
            
            # Save state
            self._save_state()
            
            # Calculate next run time
            next_run_delay = self._calculate_next_run_delay(result.success)
            logger.info(f"Next run in {next_run_delay} minutes")
            
            # Sleep until next run
            await asyncio.sleep(next_run_delay * 60)
            
        except Exception as e:
            logger.error(f"Error in continuous loop iteration: {e}")
            traceback.print_exc()
            # Wait before retrying
            await asyncio.sleep(300)  # 5 minutes
    
    async def _should_run_now(self) -> bool:
        """Determine if we should run a rebalancing cycle now."""
        now = datetime.now()
        
        # Check minimum interval
        if self.last_run_time:
            time_since_last = (now - self.last_run_time).total_seconds() / 60
            if time_since_last < self.min_interval_minutes:
                return False
        
        # Check consecutive failures
        if self.health.consecutive_failures >= self.max_consecutive_failures:
            logger.warning(f"Too many consecutive failures ({self.health.consecutive_failures}), pausing operations")
            return False
        
        # Check market hours for optimal timing
        try:
            market_open = alpaca_client.is_market_open()
            market_calendar = alpaca_client.get_market_calendar()
            
            # During market hours: run more frequently
            if market_open:
                return True
            
            # After hours: run less frequently but still monitor
            if self.last_run_time:
                hours_since_last = (now - self.last_run_time).total_seconds() / 3600
                return hours_since_last >= (self.after_hours_interval_minutes / 60)
            
            return True
            
        except Exception as e:
            logger.warning(f"Could not check market status: {e}")
            return True  # Default to running if can't check
    
    async def _run_rebalancing_pipeline(self) -> RebalanceResult:
        """Run the complete rebalancing pipeline with error handling."""
        start_time = datetime.now()
        pipeline_stage = "initialization"
        
        try:
            # Create initial state
            state = create_initial_state()
            config = {"thread_id": f"continuous_rebalancer_{int(start_time.timestamp())}"}
            
            # No hardcoded watchlist - let universe filter discover opportunities dynamically
            state["watchlist"] = []  # Empty watchlist - universe filter will find stocks
            
            initial_portfolio_value = 0.0
            signals_generated = 0
            orders_executed = 0
            
            # Step 1: Market Monitor
            pipeline_stage = "market_monitor"
            logger.info("📊 Market Monitor...")
            state = await self.workflow.market_monitor_agent(state, config)
            initial_portfolio_value = state["portfolio"].get("equity", 0)
            
            # Step 2: Universe Filter (11k+ → ~200 actionable stocks)
            pipeline_stage = "universe_filter"
            logger.info("🔍 Universe Filter (99.7% processing reduction)...")
            state = await self.workflow.universe_filter_agent(state, config)
            
            # Step 3: Sentiment Analysis (only on filtered stocks)
            pipeline_stage = "sentiment_analysis"
            logger.info("💭 Sentiment Analysis (pre-filtered stocks only)...")
            state = await self.workflow.sentiment_analysis_agent(state, config)
            
            # Check for API rate limits
            if self._check_rate_limit_indicators(state):
                self.health.api_rate_limit_hits += 1
                logger.warning("API rate limit detected, implementing cooldown")
                await asyncio.sleep(self.api_cooldown_minutes * 60)
            
            # Step 4: Risk Assessment
            pipeline_stage = "risk_assessment"
            logger.info("⚖️ Risk Assessment...")
            state = await self.workflow.risk_assessment_agent(state, config)
            
            # Check circuit breakers
            circuit_breakers = state.get("circuit_breakers", {})
            if any(circuit_breakers.values()):
                active_breakers = [name for name, active in circuit_breakers.items() if active]
                logger.warning(f"Circuit breakers active: {active_breakers}")
                # Continue but with reduced position sizing
            
            # Step 4.5: RL Decision Layer (Policy Learning & Portfolio Allocation)
            pipeline_stage = "rl_decision_layer"
            logger.info("🤖 RL Decision Layer (Policy Learning & Portfolio Allocation)...")
            state = await self._run_rl_decision_layer(state, config)
            
            # Step 5: Signal Generation (Enhanced by RL)
            pipeline_stage = "signal_generation"
            logger.info("🧠 LLM Signal Generation (RL-Enhanced)...")
            pre_signals = len(state.get("signals", []))
            state = await self.workflow.signal_generation_agent(state, config)
            post_signals = len(state.get("signals", []))
            signals_generated = post_signals - pre_signals
            
            # Step 6: Strategy Optimization
            pipeline_stage = "strategy_optimization"
            logger.info("🎯 Strategy Optimization...")
            state = await self.workflow.strategy_optimization_agent(state, config)
            
            # Step 7: Order Management (if signals exist)
            if signals_generated > 0:
                pipeline_stage = "order_management"
                logger.info(f"💼 Order Management ({signals_generated} signals)...")
                pre_orders = len(state.get("executed_orders", []))
                state = await self.workflow.order_management_agent(state, config)
                post_orders = len(state.get("executed_orders", []))
                orders_executed = post_orders - pre_orders
            else:
                logger.info("No signals generated, skipping order management")
            
            # Step 8: Portfolio Tracking
            pipeline_stage = "portfolio_tracking"
            logger.info("📈 Portfolio Tracking...")
            state = await self.workflow.portfolio_tracking_agent(state, config)
            
            final_portfolio_value = state["portfolio"].get("equity", initial_portfolio_value)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Log results
            logger.info(f"✅ REBALANCING CYCLE COMPLETE")
            logger.info(f"Duration: {duration:.1f}s")
            logger.info(f"Signals Generated: {signals_generated}")
            logger.info(f"Orders Executed: {orders_executed}")
            logger.info(f"Portfolio Value: ${final_portfolio_value:,.2f}")
            
            return RebalanceResult(
                timestamp=start_time,
                success=True,
                duration_seconds=duration,
                signals_generated=signals_generated,
                orders_executed=orders_executed,
                portfolio_value=final_portfolio_value,
                pipeline_stage=pipeline_stage
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"Pipeline failed at {pipeline_stage}: {str(e)}"
            logger.error(error_msg)
            traceback.print_exc()
            
            return RebalanceResult(
                timestamp=start_time,
                success=False,
                duration_seconds=duration,
                signals_generated=signals_generated,
                orders_executed=orders_executed,
                portfolio_value=initial_portfolio_value,
                error_message=error_msg,
                pipeline_stage=pipeline_stage
            )
    
    def _check_rate_limit_indicators(self, state: Dict[str, Any]) -> bool:
        """Check if there are indicators of API rate limiting."""
        # Check for rate limit errors in state
        errors = state.get("errors", [])
        for error in errors:
            if any(keyword in str(error).lower() for keyword in 
                   ['rate limit', 'too many requests', '429', 'quota exceeded']):
                return True
        
        # Check for empty results that might indicate rate limiting
        sentiment_data = state.get("sentiment_data", {})
        if len(sentiment_data) == 0 and len(state.get("watchlist", [])) > 0:
            return True
        
        return False
    
    def _process_result(self, result: RebalanceResult):
        """Process the results of a rebalancing run."""
        self.health.total_runs += 1
        self.last_run_time = result.timestamp
        
        if result.success:
            self.health.successful_runs += 1
            self.health.last_success = result.timestamp
            self.health.consecutive_failures = 0
            logger.info("✅ Rebalancing cycle completed successfully")
        else:
            self.health.failed_runs += 1
            self.health.last_failure = result.timestamp
            self.health.consecutive_failures += 1
            logger.error(f"❌ Rebalancing cycle failed: {result.error_message}")
        
        # Update average runtime
        total_runtime = sum(r.duration_seconds for r in self.results_history) + result.duration_seconds
        total_runs = len(self.results_history) + 1
        self.health.avg_runtime_seconds = total_runtime / total_runs
        
        # Add to history
        self.results_history.append(result)
        if len(self.results_history) > self.max_history_size:
            self.results_history.pop(0)
    
    def _calculate_next_run_delay(self, success: bool) -> int:
        """Calculate minutes to wait before next run."""
        try:
            market_open = alpaca_client.is_market_open()
        except:
            market_open = True  # Default to market hours timing if can't check
        
        if success:
            # Normal intervals based on market hours
            if market_open:
                return self.standard_interval_minutes
            else:
                return self.after_hours_interval_minutes
        else:
            # Exponential backoff for failures
            backoff_multiplier = min(self.health.consecutive_failures, 5)
            return self.failure_backoff_minutes * backoff_multiplier
    
    async def _perform_system_check(self):
        """Perform initial system health check."""
        logger.info("🔧 Performing system check...")
        
        try:
            # Check Alpaca connection
            account = alpaca_client.get_account_info()
            logger.info(f"✅ Alpaca connected: {account['id']}")
            
            # Check market status
            market_open = alpaca_client.is_market_open()
            logger.info(f"📈 Market status: {'Open' if market_open else 'Closed'}")
            
            # Check available cash
            cash = account.get('cash', 0)
            logger.info(f"💰 Available cash: ${cash:,.2f}")
            
            # Test workflow initialization
            workflow_test = TradingWorkflow()
            logger.info("✅ Workflow initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ System check failed: {e}")
            raise
    
    def _save_state(self):
        """Save current state to disk."""
        try:
            state_data = {
                "health": asdict(self.health),
                "results_history": [asdict(r) for r in self.results_history[-20:]],  # Last 20 results
                "last_run_time": self.last_run_time.isoformat() if self.last_run_time else None
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")
    
    def _load_state(self):
        """Load previous state from disk."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)
                
                # Restore health metrics
                health_data = state_data.get("health", {})
                for key, value in health_data.items():
                    if hasattr(self.health, key):
                        if key in ['last_success', 'last_failure'] and value:
                            setattr(self.health, key, datetime.fromisoformat(value))
                        else:
                            setattr(self.health, key, value)
                
                # Restore last run time
                last_run_str = state_data.get("last_run_time")
                if last_run_str:
                    self.last_run_time = datetime.fromisoformat(last_run_str)
                
                logger.info(f"✅ Loaded previous state: {self.health.total_runs} total runs")
                
        except Exception as e:
            logger.warning(f"Could not load previous state: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    async def _shutdown(self):
        """Perform graceful shutdown."""
        logger.info("🛑 SHUTTING DOWN CONTINUOUS REBALANCER")
        
        # Save final state
        self._save_state()
        
        # Print final statistics
        if self.start_time:
            uptime = datetime.now() - self.start_time
            logger.info(f"Total uptime: {uptime}")
        
        logger.info(f"Total runs: {self.health.total_runs}")
        logger.info(f"Successful runs: {self.health.successful_runs}")
        logger.info(f"Failed runs: {self.health.failed_runs}")
        
        if self.health.total_runs > 0:
            success_rate = (self.health.successful_runs / self.health.total_runs) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
        
        logger.info("👋 Continuous rebalancer shutdown complete")
    
    async def _run_rl_decision_layer(self, state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced RL Decision Layer with FinRL Integration
        
        Architecture:
        [LLM Analysis Layer: sentiment_data, market_signals] 
                    ↓
        [FinRL Agent: Advanced RL Decision Layer] ← This method
                    ↓
        [Enhanced state with sophisticated RL portfolio decisions]
        """
        try:
            # Try FinRL integration first (advanced RL system)
            try:
                from agents.finrl_integration import create_finrl_trading_system, integrate_finrl_with_workflow
                
                logger.info("🚀 Initializing Enhanced FinRL Decision Layer...")
                
                # Extract symbols from sentiment signals
                sentiment_signals = state.get("sentiment_signals", [])
                available_symbols = list(set([s['symbol'] for s in sentiment_signals if s.get('signal') == 'BUY']))[:8]
                
                if not available_symbols:
                    logger.warning("No symbols available for FinRL - using fallback")
                    raise ImportError("No symbols for FinRL")
                
                logger.info(f"📊 FinRL Processing: {len(available_symbols)} symbols")
                
                # Create or reuse FinRL orchestrator
                if not hasattr(self, '_finrl_orchestrator') or self._finrl_orchestrator is None:
                    self._finrl_orchestrator = await create_finrl_trading_system(
                        symbols=available_symbols,
                        initial_balance=state.get("portfolio", {}).get("equity", 100000),
                        enable_short_selling=True,
                        enable_limit_orders=True,
                        paper_trading=True,
                        max_position_size=0.15
                    )
                    logger.info("✅ FinRL Trading System created")
                
                # Integrate with workflow
                state = await integrate_finrl_with_workflow(self._finrl_orchestrator, state)
                
                # Extract FinRL decisions
                finrl_decisions = state.get("finrl_decisions", {})
                
                if finrl_decisions and finrl_decisions.get('orders'):
                    # Convert FinRL decisions to standard RL format
                    rl_decisions = {
                        "strategy": finrl_decisions.get('strategy', 'finrl_optimized'),
                        "risk_level": finrl_decisions.get('risk_level', 'moderate'),
                        "allocations": [],
                        "confidence": finrl_decisions.get('confidence', 0.8),
                        "reasoning": f"FinRL agent with {len(finrl_decisions['orders'])} positions",
                        "advanced_features": {
                            "short_selling": True,
                            "limit_orders": True,
                            "risk_management": True
                        }
                    }
                    
                    # Convert orders to allocations
                    portfolio_value = state.get("portfolio", {}).get("equity", 100000)
                    for order in finrl_decisions['orders']:
                        allocation_value = order['quantity'] * order['price']
                        weight = allocation_value / portfolio_value
                        
                        # Adjust weight for short positions
                        if order['side'] in ['short', 'sell']:
                            weight = -abs(weight)
                        
                        allocation = {
                            "symbol": order['symbol'],
                            "weight": weight,
                            "confidence": order.get('confidence', 0.8),
                            "action": order['side'],
                            "reasoning": f"FinRL {order['side']} decision",
                            "order_type": order.get('order_type', 'market'),
                            "limit_price": order.get('limit_price')
                        }
                        
                        rl_decisions["allocations"].append(allocation)
                    
                    state["rl_decisions"] = rl_decisions
                    state["rl_enhanced"] = True
                    state["finrl_integrated"] = True
                    
                    logger.info(f"✅ FinRL Agent generated {len(rl_decisions['allocations'])} advanced allocations")
                    logger.info(f"📈 FinRL Strategy: {rl_decisions['strategy']}")
                    logger.info(f"⚖️ FinRL Risk Level: {rl_decisions['risk_level']}")
                    
                    # Log top allocations with enhanced info
                    for allocation in rl_decisions["allocations"][:3]:
                        weight_str = f"{allocation['weight']:.1%}"
                        action_str = allocation['action']
                        order_type = allocation.get('order_type', 'market')
                        logger.info(f"   🎯 {allocation['symbol']}: {action_str} {weight_str} ({order_type})")
                    
                    return state
                
                else:
                    logger.warning("FinRL generated no decisions - falling back to basic RL")
                    raise Exception("No FinRL decisions generated")
                    
            except (ImportError, Exception) as finrl_error:
                logger.warning(f"FinRL integration failed: {finrl_error}")
                logger.info("Falling back to basic RL orchestrator...")
                
                # Fallback to basic RL orchestrator
                from agents.online_learning_orchestrator import OnlineLearningOrchestrator, TradingEngine, PortfolioManager, RiskManager
                from agents.llm_rl_integration import LLMStateEnricher
                
                logger.info("🤖 Initializing Basic RL Decision Layer...")
                
                # Extract LLM analysis results (Feature Store)
                sentiment_data = state.get("sentiment_data", {})
                sentiment_signals = state.get("sentiment_signals", [])
                market_data = state.get("market_data", {})
                portfolio = state.get("portfolio", {})
                
                logger.info(f"📊 Feature Store Input: {len(sentiment_data)} sentiment analyses, {len(sentiment_signals)} signals")
                
                # Create enhanced state for RL agent
                enricher = LLMStateEnricher()
                enhanced_state = await enricher.enrich_state_from_llm_analysis(
                    sentiment_data=sentiment_data,
                    market_signals=sentiment_signals,
                    portfolio_state=portfolio,
                    market_data=market_data
                )
                
                # Initialize online learning orchestrator with default components
                orchestrator = OnlineLearningOrchestrator(
                    trading_engine=TradingEngine(),
                    portfolio_manager=PortfolioManager(), 
                    risk_manager=RiskManager()
                )
                
                # Get RL portfolio decisions
                logger.info("🧠 Basic RL Agent making portfolio allocation decisions...")
                
                # Run RL decision making
                rl_decisions = await orchestrator.make_portfolio_decisions(
                    enhanced_state=enhanced_state,
                    available_symbols=[s['symbol'] for s in sentiment_signals if s.get('signal') == 'BUY'][:10],
                    portfolio_value=portfolio.get("equity", 50000),
                    risk_tolerance=state.get("risk_tolerance", 0.5)
                )
                
                # Store RL decisions in state for signal generation
                state["rl_decisions"] = rl_decisions
                state["rl_enhanced"] = True
                state["finrl_integrated"] = False
                
                # Log RL insights
                if rl_decisions:
                    logger.info(f"✅ Basic RL Agent generated {len(rl_decisions.get('allocations', []))} allocation decisions")
                    logger.info(f"📈 RL Portfolio Strategy: {rl_decisions.get('strategy', 'adaptive')}")
                    logger.info(f"⚖️ RL Risk Assessment: {rl_decisions.get('risk_level', 'moderate')}")
                    
                    # Log top allocations
                    for allocation in rl_decisions.get('allocations', [])[:3]:
                        logger.info(f"   🎯 {allocation.get('symbol')}: {allocation.get('weight', 0):.1%} allocation")
                
                return state
            
        except Exception as e:
            logger.error(f"RL Decision Layer error: {e}")
            logger.info("Falling back to LLM-only analysis...")
            state["rl_enhanced"] = False
            state["finrl_integrated"] = False
            return state
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get current system status report."""
        now = datetime.now()
        
        return {
            "running": self.running,
            "uptime_hours": self.health.uptime_hours,
            "health": asdict(self.health),
            "last_run_time": self.last_run_time.isoformat() if self.last_run_time else None,
            "next_run_estimate": (self.last_run_time + timedelta(minutes=self.standard_interval_minutes)).isoformat() 
                               if self.last_run_time else None,
            "recent_results": [asdict(r) for r in self.results_history[-5:]],
            "current_time": now.isoformat()
        }

# Global instance and convenience functions
continuous_rebalancer = ContinuousRebalancer()

async def start_continuous_rebalancing():
    """Start the continuous rebalancing system."""
    await continuous_rebalancer.start_continuous_operation()

def get_rebalancer_status():
    """Get current rebalancer status."""
    return continuous_rebalancer.get_status_report()

if __name__ == "__main__":
    print("🚀 Starting Continuous Rebalancing System...")
    print("This will run the end-to-end pipeline continuously with intelligent scheduling.")
    print("Press Ctrl+C to stop gracefully.")
    print("=" * 70)
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    try:
        asyncio.run(start_continuous_rebalancing())
    except KeyboardInterrupt:
        print("\n👋 Continuous rebalancing stopped by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()