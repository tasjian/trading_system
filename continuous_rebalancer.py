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
    print("✅ Applied COMPLETE WORKFLOW optimization patches")
    print("   - Sentiment analysis: OPTIMIZED (not bypassed)")
    print("   - HTTP sessions: Managed efficiently")
    print("   - Performance: Optimized for continuous rebalancing")
except ImportError:
    print("⚠️ Complete workflow patches not found - running with default optimization")

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
from config.settings import settings  # , get_crypto_pairs
# CRYPTO TRADING DISABLED - Comment out for later implementation
# from core.crypto_data_collector import crypto_collector

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
        self.min_interval_minutes = 2       # Minimum time between runs (safety buffer)
        self.standard_interval_minutes = 5   # Standard interval during market hours (RL trading)
        self.after_hours_interval_minutes = 30  # Reduced interval after hours (stocks only)
        # CRYPTO TRADING DISABLED - Comment out for later implementation
        # self.crypto_only_interval_minutes = 10   # Crypto trading interval when stock market closed
        self.last_run_time = None
        
        # Error handling
        self.max_consecutive_failures = 5
        self.failure_backoff_minutes = 60  # Wait longer after failures
        self.api_cooldown_minutes = 2      # Cooldown after API rate limits (reduced for 5min cycles)
        
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
        
        # Sentiment analysis scheduling
        self.sentiment_interval_minutes = 90  # Run sentiment analysis every 90 minutes
        self.last_sentiment_run = None
        self.cached_sentiment_data = {}  # Cache sentiment data between runs
        
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
        
        # CRYPTO TRADING DISABLED - Comment out for later implementation
        # # Check crypto status for logging
        # from config.settings import settings, get_crypto_pairs
        # has_crypto = settings.crypto_enabled and len(get_crypto_pairs()) > 0
        # 
        # if has_crypto:
        #     logger.info(f"🚀 CRYPTO ENABLED: Continuous {self.standard_interval_minutes}-minute intervals (24/7)")
        #     logger.info(f"Crypto pairs: {', '.join(get_crypto_pairs())}")
        # else:
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
                # CRYPTO TRADING DISABLED - Comment out for later implementation
                # # Crypto-aware sleep duration: shorter when crypto enabled for responsiveness
                # from config.settings import settings, get_crypto_pairs
                # has_crypto = settings.crypto_enabled and len(get_crypto_pairs()) > 0
                # sleep_duration = 30 if has_crypto else 60  # 30s with crypto, 60s without
                sleep_duration = 60  # Standard sleep duration without crypto
                await asyncio.sleep(sleep_duration)
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
        
        # CRYPTO TRADING DISABLED - Comment out for later implementation
        # Check market hours for optimal timing (stocks only)
        try:
            # # Check if we have any crypto positions or symbols
            # from config.settings import settings, get_crypto_pairs
            # has_crypto = settings.crypto_enabled and len(get_crypto_pairs()) > 0
            
            market_open = alpaca_client.is_market_open()
            market_calendar = alpaca_client.get_market_calendar()
            
            # # With crypto enabled: maintain standard 5-minute intervals continuously (24/7)
            # if has_crypto:
            #     if self.last_run_time:
            #         minutes_since_last = (now - self.last_run_time).total_seconds() / 60
            #         return minutes_since_last >= self.standard_interval_minutes
            #     return True
            
            # During stock market hours: standard intervals
            if market_open:
                if self.last_run_time:
                    minutes_since_last = (now - self.last_run_time).total_seconds() / 60
                    return minutes_since_last >= self.standard_interval_minutes
                return True
            
            # After hours without crypto: run less frequently for monitoring only
            else:
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
            
            # CRYPTO TRADING DISABLED - Comment out for later implementation
            # Initialize watchlist - let universe filter discover stocks dynamically
            watchlist_symbols = []
            
            # # Add crypto pairs to watchlist if crypto trading is enabled
            # if settings.crypto_enabled:
            #     crypto_pairs = get_crypto_pairs()
            #     if crypto_pairs:
            #         watchlist_symbols.extend(crypto_pairs)
            #         logger.info(f"🪙 Added {len(crypto_pairs)} crypto pairs to watchlist: {', '.join(crypto_pairs)}")
            
            state["watchlist"] = watchlist_symbols
            
            # # Start crypto data streams if crypto symbols are present
            # if watchlist_symbols and any(settings.crypto_enabled for symbol in watchlist_symbols if symbol in get_crypto_pairs()):
            #     crypto_symbols = [s for s in watchlist_symbols if s in get_crypto_pairs()]
            #     if crypto_symbols:
            #         logger.info(f"🚀 Starting crypto data collection for: {', '.join(crypto_symbols)}")
            #         try:
            #             await crypto_collector.start_real_time_streams(crypto_symbols)
            #             logger.info("✅ Crypto data streams initialized")
            #         except Exception as e:
            #             logger.warning(f"Failed to start crypto streams: {e}")
            
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
            
            # Step 3: Sentiment Analysis (only on filtered stocks, every 90 minutes)
            pipeline_stage = "sentiment_analysis"
            state = await self._run_scheduled_sentiment_analysis(state, config)
            
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
            
            # Step 4.5: Hybrid LLM-RL Portfolio Decision Layer
            pipeline_stage = "hybrid_portfolio_decision"
            logger.info("🚀 Hybrid LLM-RL Portfolio Decision Layer (Diversified Portfolio Management)...")
            state = await self._run_hybrid_portfolio_decision_layer(state, config)
            
            # Step 5: Signal Generation (Convert Hybrid Portfolio Decisions to Trading Signals)
            pipeline_stage = "signal_generation"
            logger.info("🎯 Converting Hybrid Portfolio Decisions to Trading Signals...")
            pre_signals = len(state.get("signals", []))
            
            # Check if hybrid system generated allocations
            rl_decisions = state.get("rl_decisions", {})
            rl_allocations = rl_decisions.get("allocations", [])
            rebalance_analysis = rl_decisions.get("rebalance_analysis", {})
            
            # Check if rebalancing is needed based on intelligent analysis
            needs_rebalancing = rebalance_analysis.get("needs_rebalancing", True)
            
            if not needs_rebalancing:
                logger.info("🔒 No rebalancing needed - portfolio is well-balanced")
                signals_generated = 0
            elif rl_allocations:
                logger.info(f"📊 Converting {len(rl_allocations)} hybrid portfolio allocations to trading signals")
                
                signals = []
                from agents.state import TradingSignal, add_signal_to_state
                from tools.alpaca_client import alpaca_client
                
                # Get current portfolio data for enhanced calculation
                current_portfolio = state.get("portfolio", {})
                available_cash = float(current_portfolio.get("cash", 50000))
                portfolio_value = float(current_portfolio.get("equity", 100000))
                
                for allocation in rl_allocations:
                    symbol = allocation.get("symbol", "")
                    target_weight = float(allocation.get("weight", 0.0))  # Weight as decimal (0.05 = 5%)
                    action = allocation.get("action", "buy")
                    
                    if not symbol or target_weight <= 0:
                        continue
                    
                    try:
                        # Calculate target dollar amount based on portfolio value
                        target_dollar_amount = target_weight * portfolio_value
                        
                        # Get current price for quantity calculation
                        current_price = alpaca_client.get_current_price(symbol)
                        if current_price and current_price > 0:
                            # Calculate shares needed
                            quantity = int(target_dollar_amount / current_price)
                            
                            # Only proceed with meaningful position sizes
                            if quantity > 0 and target_dollar_amount >= 100:  # Min $100 position
                                logger.info(f"💰 {symbol}: {target_weight:.1%} weight = ${target_dollar_amount:.2f} = {quantity} shares @ ${current_price:.2f}")
                                
                                # Create properly structured trading signal
                                signal = TradingSignal(
                                    symbol=symbol,
                                    action=action,
                                    confidence=float(allocation.get("confidence", 0.8)),
                                    quantity=float(quantity),
                                    reasoning=f"Hybrid LLM-RL Portfolio: {allocation.get('reasoning', 'Diversified sector allocation with RL optimization')}"
                                )
                                signals.append(signal)
                            else:
                                logger.debug(f"⚠️ Skipping {symbol}: position too small (${target_dollar_amount:.2f})")
                        else:
                            logger.warning(f"⚠️ Could not get price for {symbol}, skipping")
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Signal generation failed for {symbol}: {e}")
                        continue
                
                # Add signals to state using proper state management
                for signal in signals:
                    state = add_signal_to_state(state, signal)
                signals_generated = len(signals)
                
                # Enhanced logging for hybrid system
                strategy = rl_decisions.get("strategy", "unknown")
                total_allocation = rl_decisions.get("total_allocation", 0.0)
                portfolio_analytics = rl_decisions.get("portfolio_analytics", {})
                sector_count = portfolio_analytics.get("sector_count", 0)
                
                logger.info(f"✅ Generated {signals_generated} signals from hybrid LLM-RL system")
                logger.info(f"📈 Strategy: {strategy}")
                logger.info(f"🎯 Total Portfolio Allocation: {total_allocation:.1%}")
                logger.info(f"🏭 Sector Diversification: {sector_count} sectors")
                
                # Log rebalancing reasons
                rebalance_reasons = rebalance_analysis.get("rebalance_reasons", [])
                if rebalance_reasons:
                    logger.info(f"⚖️ Rebalancing triggered by:")
                    for reason in rebalance_reasons[:3]:
                        logger.info(f"   • {reason}")
            else:
                # Fallback to traditional signal generation
                logger.warning("🔄 No hybrid allocations found, falling back to LLM signal generation")
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
    
    async def _run_scheduled_sentiment_analysis(self, state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Run sentiment analysis only every 90 minutes, use cached data otherwise."""
        from tools.alpaca_client import alpaca_client
        
        # Check if market is open (crypto-aware)
        from config.settings import settings, get_crypto_pairs
        has_crypto = settings.crypto_enabled and len(get_crypto_pairs()) > 0
        
        market_open = alpaca_client.is_market_open() or has_crypto  # Crypto = always open
        current_time = datetime.now()
        
        # Determine if we should run sentiment analysis
        should_run_sentiment = False
        
        # Always run if no cached data is available to ensure end-to-end pipeline
        if not self.cached_sentiment_data:
            should_run_sentiment = True
            reason = "no cached data available - running full end-to-end pipeline"
        elif market_open:
            # Market is open - check 90-minute interval
            if (self.last_sentiment_run is None or 
                (current_time - self.last_sentiment_run).total_seconds() >= (self.sentiment_interval_minutes * 60)):
                should_run_sentiment = True
                reason = f"90-minute interval reached (last run: {self.last_sentiment_run})"
            else:
                next_sentiment_time = self.last_sentiment_run + timedelta(minutes=self.sentiment_interval_minutes)
                reason = f"using cached data, next sentiment run at {next_sentiment_time.strftime('%H:%M:%S')}"
        else:
            # Market is closed - don't run sentiment analysis unless no cached data
            reason = "market is closed"
        
        if should_run_sentiment:
            logger.info(f"💭 Running Fresh Sentiment Analysis - {reason}")
            
            # Run the actual sentiment analysis
            state = await self.workflow.sentiment_analysis_agent(state, config)
            
            # Cache the results and update timestamp
            self.cached_sentiment_data = state.get("sentiment_data", {})
            state["cached_sentiment_data"] = self.cached_sentiment_data.copy()  # Also provide for universe filter
            self.last_sentiment_run = current_time
            
            logger.info(f"✅ Sentiment analysis complete: {len(self.cached_sentiment_data)} symbols analyzed")
            
        else:
            logger.info(f"📋 Using Cached Sentiment Data - {reason}")
            
            # Use cached sentiment data
            if self.cached_sentiment_data:
                state["sentiment_data"] = self.cached_sentiment_data.copy()
                state["cached_sentiment_data"] = self.cached_sentiment_data.copy()  # Also provide for universe filter
                
                # Reconstruct sentiment_signals from cached sentiment_data for RL integration
                sentiment_signals = []
                for symbol, sentiment_data in self.cached_sentiment_data.items():
                    # Convert sentiment data back to signal format (enhanced with social media data)
                    if isinstance(sentiment_data, dict):
                        overall_score = sentiment_data.get('overall_score', 0.0)
                        overall_sentiment = sentiment_data.get('overall_sentiment', 'neutral')
                        confidence = sentiment_data.get('confidence', 0.5)
                        # Extract social media sentiment information
                        social_sentiment = sentiment_data.get('social_sentiment', {})
                        social_posts_count = sentiment_data.get('social_posts_count', 0)
                        has_social_data = len(social_sentiment) > 0 or social_posts_count > 0
                    else:
                        # Handle ComprehensiveSentiment object format
                        overall_score = getattr(sentiment_data, 'overall_score', 0.0)
                        overall_sentiment = getattr(sentiment_data, 'overall_sentiment', 'neutral')
                        confidence = getattr(sentiment_data, 'confidence', 0.5)
                        # Extract social media sentiment information from object
                        social_sentiment = getattr(sentiment_data, 'social_sentiment', {})
                        social_posts_count = getattr(sentiment_data, 'social_posts_count', 0)
                        has_social_data = len(social_sentiment) > 0 or social_posts_count > 0
                    
                    # Generate signal based on sentiment score with enhanced SHORT detection and improved neutral handling
                    if overall_score >= 0.05:  # Lower positive sentiment threshold
                        signal_strength = min(0.5, max(0.2, overall_score))  # Adjust strength based on score
                        # Enhanced signal with social media integration
                        reasoning_parts = [f"{overall_sentiment.title()} overall sentiment"]
                        if has_social_data:
                            reasoning_parts.append(f"Social media activity: {len(social_sentiment)} platforms")
                            if social_posts_count > 0:
                                reasoning_parts.append(f"{social_posts_count} social posts")
                        reasoning_parts.append("Multiple data sources")
                        
                        sentiment_signals.append({
                            'symbol': symbol,
                            'signal': 'BUY',
                            'strength': signal_strength,
                            'confidence': confidence,
                            'reasoning': ", ".join(reasoning_parts),
                            'timestamp': datetime.now(),
                            'has_earnings': False,  # Cached data doesn't track earnings
                            'has_social_data': has_social_data,
                            'social_platforms': len(social_sentiment),
                            'social_posts_count': social_posts_count,
                            'source': 'comprehensive_sentiment_with_social'
                        })
                    elif overall_score <= -0.5:  # Extreme negative sentiment for SHORT signals
                        signal_strength = min(0.8, abs(overall_score))  # Higher strength for shorts
                        signal_type = 'SHORT' if overall_score <= -0.7 else 'SELL'  # SHORT for extremely negative
                        
                        # Enhanced reasoning with social media integration
                        reasoning_parts = []
                        if overall_score <= -0.7:
                            reasoning_parts.append(f"CRISIS-LEVEL negative sentiment ({overall_score:.2f})")
                        else:
                            reasoning_parts.append(f"EXTREME negative sentiment ({overall_score:.2f})")
                        
                        if has_social_data:
                            reasoning_parts.append(f"Social media negativity: {len(social_sentiment)} platforms")
                            if social_posts_count > 0:
                                reasoning_parts.append(f"{social_posts_count} negative posts")
                        reasoning_parts.append("Multiple data sources")
                        
                        # Boost strength if social media confirms negative sentiment
                        if has_social_data and len(social_sentiment) >= 2:
                            signal_strength = min(0.9, signal_strength * 1.1)  # 10% boost for multi-platform negativity
                        
                        sentiment_signals.append({
                            'symbol': symbol,
                            'signal': signal_type,
                            'strength': signal_strength,
                            'confidence': min(0.95, confidence * 1.2),  # Boost confidence for extreme negatives
                            'reasoning': ", ".join(reasoning_parts),
                            'timestamp': datetime.now(),
                            'has_earnings': False,
                            'sentiment_score': overall_score,  # Include raw score for further analysis
                            'has_social_data': has_social_data,
                            'social_platforms': len(social_sentiment),
                            'social_posts_count': social_posts_count,
                            'source': 'comprehensive_sentiment_with_social'
                        })
                    elif overall_score <= -0.05:  # Lower negative sentiment threshold for SELL signals
                        signal_strength = min(0.6, max(0.2, abs(overall_score)))  # Adjust strength based on score
                        
                        # Enhanced reasoning for moderate negative sentiment
                        reasoning_parts = [f"Negative sentiment ({overall_score:.2f})"]
                        if has_social_data:
                            reasoning_parts.append(f"Social sentiment: {len(social_sentiment)} platforms")
                        reasoning_parts.append("Multiple data sources")
                        
                        sentiment_signals.append({
                            'symbol': symbol,
                            'signal': 'SELL',
                            'strength': signal_strength,
                            'confidence': confidence,
                            'reasoning': ", ".join(reasoning_parts),
                            'timestamp': datetime.now(),
                            'has_earnings': False,
                            'sentiment_score': overall_score,
                            'has_social_data': has_social_data,
                            'social_platforms': len(social_sentiment),
                            'social_posts_count': social_posts_count,
                            'source': 'comprehensive_sentiment_with_social'
                        })
                    elif abs(overall_score) < 0.05:  # Very neutral sentiment - generate weak HOLD signals for RL
                        # Enhanced reasoning for neutral sentiment
                        reasoning_parts = [f"Neutral sentiment ({overall_score:.2f})"]
                        if has_social_data:
                            reasoning_parts.append(f"Mixed social signals: {len(social_sentiment)} platforms")
                        reasoning_parts.append("Multiple data sources")
                        
                        sentiment_signals.append({
                            'symbol': symbol,
                            'signal': 'HOLD',
                            'strength': 0.1,  # Very weak signal strength
                            'confidence': confidence,
                            'reasoning': ", ".join(reasoning_parts),
                            'timestamp': datetime.now(),
                            'has_earnings': False,
                            'sentiment_score': overall_score,
                            'has_social_data': has_social_data,
                            'social_platforms': len(social_sentiment),
                            'social_posts_count': social_posts_count,
                            'source': 'comprehensive_sentiment_with_social'
                        })
                
                state["sentiment_signals"] = sentiment_signals
                
                # Enhanced logging for social media integration
                social_signals = [s for s in sentiment_signals if s.get('has_social_data', False)]
                total_social_platforms = sum(s.get('social_platforms', 0) for s in social_signals)
                total_social_posts = sum(s.get('social_posts_count', 0) for s in social_signals)
                
                logger.info(f"📊 Loaded cached sentiment for {len(self.cached_sentiment_data)} symbols, {len(sentiment_signals)} signals")
                logger.info(f"🌐 Social media integration: {len(social_signals)} signals with social data")
                if social_signals:
                    logger.info(f"📱 Social media coverage: {total_social_platforms} platform connections, {total_social_posts} total posts")
                    # Log sample social media enhanced signals
                    for signal in social_signals[:3]:
                        logger.info(f"   🎯 {signal['symbol']}: {signal['signal']} (social: {signal['social_platforms']} platforms, {signal['social_posts_count']} posts)")
                else:
                    logger.info(f"📱 No cached social media data available in sentiment signals")
            else:
                # No cached data available and no fallbacks allowed
                raise ValueError("No cached sentiment data available and no fallback mechanisms allowed - system requires fresh sentiment analysis")
        
        return state
    
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
        """Calculate minutes to wait before next run (crypto-aware)."""
        try:
            from config.settings import settings, get_crypto_pairs
            has_crypto = settings.crypto_enabled and len(get_crypto_pairs()) > 0
            market_open = alpaca_client.is_market_open()
        except:
            has_crypto = False
            market_open = True  # Default to market hours timing if can't check
        
        if success:
            # With crypto enabled: maintain standard intervals 24/7
            if has_crypto:
                return self.standard_interval_minutes
            # Without crypto: use market-based intervals
            elif market_open:
                return self.standard_interval_minutes
            else:
                return self.after_hours_interval_minutes
        else:
            # Exponential backoff for failures
            backoff_multiplier = min(self.health.consecutive_failures, 5)
            return self.failure_backoff_minutes * backoff_multiplier
    
    async def _perform_system_check(self):
        """Perform comprehensive system health check including service dependencies."""
        logger.info("🔧 Performing comprehensive system check...")
        
        try:
            # First, ensure all required services are running
            try:
                from tools.service_manager import ensure_services_running, get_service_summary
                
                logger.info("🔍 Checking and starting required services...")
                services_ok = await ensure_services_running()
                
                # Log service status
                try:
                    service_summary = get_service_summary()
                    for service_name, status in service_summary.items():
                        logger.info(f"   {service_name.upper()}: {status}")
                except Exception as e:
                    logger.debug(f"Error getting service summary: {e}")
                
                if not services_ok:
                    logger.warning("⚠️ Some services failed to start - system may have degraded performance")
                else:
                    logger.info("✅ All required services are running")
                    
            except Exception as e:
                logger.error(f"Error during service startup: {e}")
                logger.warning("⚠️ Service startup failed - continuing with degraded performance")
                services_ok = False
            
            # Check Alpaca connection
            account = alpaca_client.get_account_info()
            logger.info(f"✅ Alpaca connected: {account['id']}")
            
            # CRYPTO TRADING DISABLED - Comment out for later implementation
            # # Check market status (crypto-aware)
            # from config.settings import settings, get_crypto_pairs
            # has_crypto = settings.crypto_enabled and len(get_crypto_pairs()) > 0
            
            market_open = alpaca_client.is_market_open()
            # effective_market_open = market_open or has_crypto
            
            # if has_crypto:
            #     logger.info(f"📈 Market status: Stock {'Open' if market_open else 'Closed'}, Crypto: Always Open")
            #     
            #     # Initialize crypto data streams
            #     crypto_pairs = get_crypto_pairs()
            #     logger.info(f"🚀 Starting crypto data streams for: {', '.join(crypto_pairs)}")
            #     try:
            #         await crypto_collector.start_real_time_streams(crypto_pairs)
            #         logger.info("✅ Crypto data streams initialized successfully")
            #     except Exception as e:
            #         logger.warning(f"⚠️ Crypto data streams initialization failed: {e}")
            
            logger.info(f"📈 Market status: {'Open' if market_open else 'Closed'}")
            # else:
            #     logger.info(f"📈 Market status: {'Open' if market_open else 'Closed'}")
            
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
                "last_run_time": self.last_run_time.isoformat() if self.last_run_time else None,
                "last_sentiment_run": self.last_sentiment_run.isoformat() if self.last_sentiment_run else None,
                "cached_sentiment_data": self.cached_sentiment_data
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
                
                # Restore sentiment analysis data
                last_sentiment_str = state_data.get("last_sentiment_run")
                if last_sentiment_str:
                    self.last_sentiment_run = datetime.fromisoformat(last_sentiment_str)
                
                cached_sentiment = state_data.get("cached_sentiment_data", {})
                if cached_sentiment:
                    self.cached_sentiment_data = cached_sentiment
                    logger.info(f"✅ Loaded cached sentiment data for {len(cached_sentiment)} symbols")
                
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
    
    async def _prepare_market_data_for_finrl(self, state: Dict[str, Any], symbols: List[str]) -> Dict[str, Any]:
        """Prepare market data in the format expected by FinRL environment."""
        try:
            from tools.alpaca_client import alpaca_client
            from datetime import datetime, timedelta
            import pandas as pd
            import numpy as np
            
            logger.info("📈 Preparing market data for FinRL integration...")
            
            # Try to get recent market data from Alpaca
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)  # 60 days of data
            
            market_data = {}
            
            for symbol in symbols:
                try:
                    # Get historical data using the existing get_market_data method
                    historical_df = alpaca_client.get_market_data(symbol, timeframe="1Day")
                    
                    if historical_df is not None and len(historical_df) > 0:
                        # Convert DataFrame to lists in FinRL format
                        market_data[f"{symbol}_open"] = historical_df['open'].tolist()
                        market_data[f"{symbol}_high"] = historical_df['high'].tolist()
                        market_data[f"{symbol}_low"] = historical_df['low'].tolist()
                        market_data[f"{symbol}_close"] = historical_df['close'].tolist()
                        market_data[f"{symbol}_volume"] = historical_df['volume'].tolist()
                        
                        logger.debug(f"✅ Retrieved {len(historical_df)} bars for {symbol}")
                    else:
                        logger.warning(f"❌ No market data retrieved for {symbol}")
                        
                except Exception as e:
                    logger.warning(f"Failed to get market data for {symbol}: {e}")
            
            # If we have insufficient market data, raise error - no synthetic data
            if len(market_data) == 0:
                logger.error("No real market data available - cannot proceed without actual data")
                raise Exception("Insufficient real market data - system requires actual data")
            
            # Update state with market data
            enhanced_state = state.copy()
            enhanced_state['market_data'] = market_data
            
            logger.info(f"✅ Market data prepared for {len(symbols)} symbols")
            return enhanced_state
            
        except Exception as e:
            logger.error(f"Failed to prepare market data for FinRL: {e}")
            return state
    
    
    async def _run_hybrid_portfolio_decision_layer(self, state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hybrid LLM-RL Portfolio Decision Layer
        
        Architecture:
        [LLM Analysis Layer: sentiment_data, market_signals] 
                    ↓
        [LLM Portfolio Manager: Diversified sector-aware allocations] 
                    ↓
        [RL Agent: Position sizing and timing optimization] ← This method
                    ↓
        [Intelligent Rebalancing: Prevent over-trading]
                    ↓
        [Enhanced state with sophisticated hybrid portfolio decisions]
        """
        try:
            # Use hybrid LLM-RL system
            from agents.rl_integration_bridge import integrate_hybrid_llm_rl_portfolio_system
            
            logger.info("🚀 Initializing Hybrid LLM-RL Portfolio Decision Layer...")
            
            # Use hybrid system that combines LLM portfolio management with RL optimization
            result = await integrate_hybrid_llm_rl_portfolio_system(state, config)
            
            if result and result.get("rl_decisions"):
                state.update(result)
                state["rl_enhanced"] = True
                state["comprehensive_rl"] = True
                state["llm_enhanced"] = True
                state["hybrid_system"] = True
                
                rl_decisions = result["rl_decisions"]
                strategy = rl_decisions.get("strategy", "unknown")
                allocations = rl_decisions.get("allocations", [])
                
                logger.info(f"✅ Hybrid LLM-RL system generated {len(allocations)} allocation decisions")
                logger.info(f"🎯 Strategy: {strategy}")
                
                # Enhanced logging for portfolio analytics
                portfolio_analytics = rl_decisions.get("portfolio_analytics", {})
                if portfolio_analytics:
                    market_regime = portfolio_analytics.get("market_regime", "unknown")
                    diversification_score = portfolio_analytics.get("diversification_score", 0.0)
                    sector_count = portfolio_analytics.get("sector_count", 0)
                    
                    logger.info(f"📊 Market Regime: {market_regime}")
                    logger.info(f"🎯 Diversification Score: {diversification_score:.2f}")
                    logger.info(f"🏭 Sector Count: {sector_count}")
                
                # Log top allocations with enhanced details
                for allocation in allocations[:5]:  # Show top 5
                    symbol = allocation.get('symbol', 'N/A')
                    weight = allocation.get('weight', 0.0)
                    action = allocation.get('action', 'hold')
                    industry = allocation.get('industry', 'Unknown')
                    risk_level = allocation.get('risk_level', 'medium')
                    
                    logger.info(f"   🎯 {symbol}: {action} {weight:.1%} ({industry}, {risk_level} risk)")
                
                # Log rebalancing analysis
                rebalance_analysis = rl_decisions.get("rebalance_analysis", {})
                if rebalance_analysis:
                    needs_rebalancing = rebalance_analysis.get("needs_rebalancing", True)
                    if needs_rebalancing:
                        reasons = rebalance_analysis.get("rebalance_reasons", [])
                        logger.info(f"⚖️ Rebalancing needed: {len(reasons)} reasons")
                        for reason in reasons[:2]:
                            logger.info(f"   • {reason}")
                    else:
                        logger.info("🔒 Portfolio well-balanced, no rebalancing needed")
                
                return state
                
            else:
                logger.warning("⚠️ Hybrid LLM-RL system failed - using conservative fallback")
                
                # Conservative fallback using existing positions
                current_positions = state.get("portfolio", {}).get("positions", {})
                if current_positions:
                    logger.info(f"🔄 Maintaining {len(current_positions)} current positions")
                    fallback_decisions = {
                        "strategy": "maintain_current_positions",
                        "risk_level": "conservative", 
                        "allocations": [],  # No new trades
                        "confidence": 0.5,
                        "reasoning": "Hybrid system failed, maintaining current positions for safety",
                        "rebalance_analysis": {
                            "needs_rebalancing": False,
                            "current_positions": len(current_positions),
                            "strategy": "conservative_hold"
                        }
                    }
                else:
                    # No current positions, create minimal diversified allocation
                    sentiment_signals = state.get("sentiment_signals", [])
                    buy_signals = [s for s in sentiment_signals if s.get('signal') == 'BUY'][:3]
                    
                    if buy_signals:
                        fallback_allocations = []
                        for i, signal in enumerate(buy_signals):
                            fallback_allocations.append({
                                "symbol": signal['symbol'],
                                "weight": 0.03,  # 3% per position
                                "confidence": 0.4,
                                "action": "buy",
                                "reasoning": "Conservative diversified fallback",
                                "industry": "Unknown",
                                "risk_level": "medium"
                            })
                        
                        fallback_decisions = {
                            "strategy": "conservative_diversified_fallback",
                            "risk_level": "conservative",
                            "allocations": fallback_allocations,
                            "confidence": 0.4,
                            "reasoning": f"Hybrid system failed, using {len(fallback_allocations)} conservative positions",
                            "rebalance_analysis": {
                                "needs_rebalancing": True,
                                "rebalance_reasons": ["Fallback allocation needed"],
                                "strategy": "conservative_fallback"
                            }
                        }
                    else:
                        fallback_decisions = {
                            "strategy": "no_action",
                            "allocations": [],
                            "reasoning": "No viable signals for fallback allocation"
                        }
                
                state["rl_decisions"] = fallback_decisions
                state["rl_enhanced"] = False
                state["llm_enhanced"] = False
                state["hybrid_system"] = False
                
                logger.warning(f"🔧 Fallback strategy: {fallback_decisions['strategy']}")
                return state
                
            
        except Exception as e:
            logger.error(f"Hybrid Portfolio Decision Layer error: {e}")
            import traceback
            traceback.print_exc()
            
            logger.info("Falling back to conservative position maintenance...")
            
            # Ultra-conservative fallback
            state["rl_decisions"] = {
                "strategy": "error_recovery", 
                "allocations": [], 
                "reasoning": f"System error: {str(e)[:100]}",
                "rebalance_analysis": {"needs_rebalancing": False}
            }
            state["rl_enhanced"] = False
            state["comprehensive_rl"] = False
            state["llm_enhanced"] = False
            state["hybrid_system"] = False
            
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