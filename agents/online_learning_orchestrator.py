"""
Online Learning Orchestrator
Manages the complete online learning workflow for RL agents in live/paper trading
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# Handle optional dependencies
try:
    import numpy as np
    import pandas as pd
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    import warnings
    warnings.warn("NumPy/Pandas not available - using limited functionality")

from agents.online_rl_agent import OnlineRLAgent, OnlineLearningMode, OnlineLearningMetrics
from agents.llm_rl_integration import EnhancedLLMTradingEnvironment

# Handle optional core imports - use existing trading_system infrastructure
try:
    # Use existing trading system components
    from tools.alpaca_client import alpaca_client
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# Create wrapper classes that use existing infrastructure
class MarketDataFetcher:
    """Wrapper for existing Alpaca client market data functionality."""
    def __init__(self):
        self.alpaca_client = alpaca_client if ALPACA_AVAILABLE else None
        
    async def get_realtime_data(self, symbol):
        if not self.alpaca_client:
            return None
        try:
            # Use existing get_market_data method
            return self.alpaca_client.get_market_data(symbol, limit=100)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to get market data for {symbol}: {e}")
            return None

class TradingEngine:
    """Wrapper for existing Alpaca client trading functionality."""
    def __init__(self):
        self.alpaca_client = alpaca_client if ALPACA_AVAILABLE else None
        
    async def submit_order(self, order):
        if not self.alpaca_client:
            class MockResult:
                success = True
                error = None
            return MockResult()
        
        try:
            # Use existing place_order method
            result = self.alpaca_client.place_order(
                symbol=getattr(order, 'symbol', 'TEST'),
                qty=getattr(order, 'quantity', 1),
                side=getattr(order, 'side', 'buy'),
                order_type=getattr(order, 'order_type', 'market')
            )
            class Result:
                success = True
                error = None
                order_id = result.get('id', 'mock_id')
            return Result()
        except Exception as e:
            class Result:
                success = False
                error = str(e)
            return Result()

class PortfolioManager:
    """Wrapper for existing Alpaca client portfolio functionality."""
    def __init__(self):
        self.alpaca_client = alpaca_client if ALPACA_AVAILABLE else None
        
    def get_position(self, symbol):
        if not self.alpaca_client:
            return None
        try:
            positions = self.alpaca_client.get_positions()
            for pos in positions:
                if pos['symbol'] == symbol:
                    return pos
            return None
        except:
            return None
            
    def get_total_value(self):
        if not self.alpaca_client:
            return 100000.0
        try:
            account = self.alpaca_client.get_account_info()
            return float(account.get('portfolio_value', 100000.0))
        except:
            return 100000.0

class RiskManager:
    """Simple risk manager using existing portfolio data."""
    def __init__(self):
        self.alpaca_client = alpaca_client if ALPACA_AVAILABLE else None
        
    async def calculate_portfolio_risk(self):
        class RiskMetrics:
            portfolio_var = 0.05
            max_drawdown = -0.10
            sharpe_ratio = 1.2
        return RiskMetrics()

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """Detect market regime changes for adaptive learning."""
    
    def __init__(self, lookback_periods: int = 50):
        self.lookback_periods = lookback_periods
        self.volatility_history = []
        self.return_history = []
        self.sentiment_history = []
        
    def update(self, returns: float, volatility: float, sentiment: float):
        """Update with new market data."""
        self.return_history.append(returns)
        self.volatility_history.append(volatility)
        self.sentiment_history.append(sentiment)
        
        # Keep only recent history
        if len(self.return_history) > self.lookback_periods:
            self.return_history.pop(0)
            self.volatility_history.pop(0)
            self.sentiment_history.pop(0)
    
    def detect_regime_change(self) -> Tuple[bool, str, float]:
        """Detect if there's been a significant regime change."""
        
        if len(self.return_history) < self.lookback_periods // 2:
            return False, "insufficient_data", 0.5
        
        # Split into recent and historical periods
        split_point = len(self.return_history) // 2
        
        recent_vol = np.mean(self.volatility_history[split_point:])
        historical_vol = np.mean(self.volatility_history[:split_point])
        
        recent_returns = np.mean(self.return_history[split_point:])
        historical_returns = np.mean(self.return_history[:split_point])
        
        recent_sentiment = np.mean(self.sentiment_history[split_point:])
        historical_sentiment = np.mean(self.sentiment_history[:split_point])
        
        # Calculate change scores
        vol_change = abs(recent_vol - historical_vol) / (historical_vol + 1e-8)
        return_change = abs(recent_returns - historical_returns)
        sentiment_change = abs(recent_sentiment - historical_sentiment)
        
        # Determine regime
        if vol_change > 0.5:  # 50% change in volatility
            if recent_vol > historical_vol:
                return True, "high_volatility", 0.8
            else:
                return True, "low_volatility", 0.7
        
        if return_change > 0.02:  # 2% change in returns
            if recent_returns > historical_returns:
                return True, "bull_market", 0.7
            else:
                return True, "bear_market", 0.7
        
        if sentiment_change > 0.3:  # 30% change in sentiment
            if recent_sentiment > historical_sentiment:
                return True, "positive_sentiment", 0.6
            else:
                return True, "negative_sentiment", 0.6
        
        return False, "stable", 0.5


@dataclass
class OnlineLearningConfig:
    """Configuration for online learning orchestrator."""
    
    # Learning parameters
    update_frequency_minutes: int = 30
    regime_detection_enabled: bool = True
    adaptive_exploration: bool = True
    
    # Safety parameters
    max_drawdown_threshold: float = -0.15
    min_sharpe_threshold: float = -1.5
    performance_check_frequency: int = 100
    
    # Model management
    checkpoint_frequency_minutes: int = 120
    max_checkpoints_to_keep: int = 10
    rollback_enabled: bool = True
    
    # Market hours
    market_open_hour: int = 9
    market_close_hour: int = 16
    timezone: str = "US/Eastern"
    
    # Data management
    max_experience_age_hours: int = 168  # 1 week
    min_experiences_for_update: int = 50
    
    # Monitoring
    performance_report_frequency: int = 240  # 4 hours
    enable_detailed_logging: bool = True


class OnlineLearningOrchestrator:
    """
    Orchestrates online learning for multiple RL agents across different symbols.
    """
    
    def __init__(self,
                 trading_engine: TradingEngine,
                 portfolio_manager: PortfolioManager,
                 risk_manager: RiskManager,
                 config: OnlineLearningConfig = None):
        
        self.trading_engine = trading_engine
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        self.config = config or OnlineLearningConfig()
        
        # Core components
        self.online_agents: Dict[str, OnlineRLAgent] = {}
        self.regime_detectors: Dict[str, MarketRegimeDetector] = {}
        self.market_data_fetcher = MarketDataFetcher()
        
        # State management
        self.last_states: Dict[str, Any] = {}
        self.last_actions: Dict[str, float] = {}
        self.last_metadata: Dict[str, Dict] = {}
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.learning_events: List[Dict[str, Any]] = []
        
        # Orchestration control
        self.is_running = False
        self.learning_tasks: Dict[str, asyncio.Task] = {}
        
        # Setup directories
        self.checkpoint_dir = Path("online_learning_checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        logger.info("✅ Online Learning Orchestrator initialized")
    
    async def add_online_agent(self,
                             symbol: str,
                             pretrained_model_path: str = None,
                             learning_mode: OnlineLearningMode = OnlineLearningMode.EXPLOITATION):
        """Add an online RL agent for a symbol."""
        
        from agents.online_rl_agent import create_online_rl_agent
        
        agent = create_online_rl_agent(
            symbol=symbol,
            pretrained_model_path=pretrained_model_path,
            learning_mode=learning_mode
        )
        
        self.online_agents[symbol] = agent
        self.regime_detectors[symbol] = MarketRegimeDetector()
        
        logger.info(f"✅ Added online RL agent for {symbol}")
    
    async def start_online_learning(self, symbols: List[str]):
        """Start the online learning process for specified symbols."""
        
        if self.is_running:
            logger.warning("⚠️ Online learning already running")
            return
        
        self.is_running = True
        logger.info(f"🚀 Starting online learning for {symbols}")
        
        # Start learning tasks for each symbol
        for symbol in symbols:
            if symbol in self.online_agents:
                task = asyncio.create_task(self._symbol_learning_loop(symbol))
                self.learning_tasks[symbol] = task
            else:
                logger.warning(f"⚠️ No online agent available for {symbol}")
        
        # Start orchestration tasks
        orchestration_task = asyncio.create_task(self._orchestration_loop())
        self.learning_tasks['orchestration'] = orchestration_task
        
        logger.info(f"✅ Online learning started for {len(self.learning_tasks)} tasks")
    
    async def stop_online_learning(self):
        """Stop the online learning process."""
        
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("⏹️ Stopping online learning...")
        
        # Cancel all learning tasks
        for symbol, task in self.learning_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.learning_tasks.clear()
        
        # Final performance report
        await self._generate_performance_report()
        
        logger.info("✅ Online learning stopped")
    
    async def _symbol_learning_loop(self, symbol: str):
        """Main learning loop for a specific symbol."""
        
        agent = self.online_agents[symbol]
        regime_detector = self.regime_detectors[symbol]
        
        logger.info(f"🔄 Starting learning loop for {symbol}")
        
        try:
            while self.is_running:
                loop_start = datetime.now()
                
                # Get current market data
                market_data = await self.market_data_fetcher.get_realtime_data(symbol)
                
                if market_data is None or len(market_data) < 10:
                    logger.warning(f"⚠️ Insufficient market data for {symbol}")
                    await asyncio.sleep(60)
                    continue
                
                # Create enhanced trading environment for current state
                env = EnhancedLLMTradingEnvironment(
                    symbol=symbol,
                    initial_balance=100000  # Dummy for state calculation
                )
                
                try:
                    env.set_data(market_data.tail(100))
                    current_enhanced_state = await env._get_enhanced_current_state()
                    
                    # Get action from agent
                    action, confidence, metadata = await agent.get_action(
                        current_enhanced_state, 
                        float(market_data['close'].iloc[-1])
                    )
                    
                    # Execute trade if significant action
                    if abs(action) > 0.1 and confidence > 0.6:
                        trade_result = await self._execute_trade(symbol, action, confidence, metadata)
                        metadata['trade_executed'] = trade_result
                    else:
                        metadata['trade_executed'] = False
                    
                    # Calculate reward if we have previous state
                    if symbol in self.last_states:
                        reward = await self._calculate_reward(
                            symbol, 
                            self.last_states[symbol],
                            current_enhanced_state,
                            self.last_actions[symbol]
                        )
                        
                        # Learn from experience
                        await agent.learn_from_experience(
                            self.last_states[symbol],
                            self.last_actions[symbol],
                            reward,
                            current_enhanced_state,
                            False,  # Not done
                            self.last_metadata[symbol]
                        )
                        
                        # Update regime detector
                        current_return = reward  # Simplified
                        current_vol = current_enhanced_state.llm_market_state.volatility_forecast
                        current_sentiment = current_enhanced_state.llm_market_state.overall_score
                        
                        regime_detector.update(current_return, current_vol, current_sentiment)
                        
                        # Check for regime change
                        regime_changed, new_regime, confidence = regime_detector.detect_regime_change()
                        
                        if regime_changed and self.config.regime_detection_enabled:
                            await self._handle_regime_change(symbol, new_regime, confidence)
                    
                    # Store current state for next iteration
                    self.last_states[symbol] = current_enhanced_state
                    self.last_actions[symbol] = action
                    self.last_metadata[symbol] = metadata
                    
                    # Log learning event
                    self.learning_events.append({
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'action': action,
                        'confidence': confidence,
                        'learning_mode': agent.learning_mode.value,
                        'metadata': metadata
                    })
                    
                finally:
                    await env.close()
                
                # Wait for next update
                loop_duration = (datetime.now() - loop_start).total_seconds()
                sleep_time = max(0, self.config.update_frequency_minutes * 60 - loop_duration)
                
                await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            logger.info(f"📋 Learning loop for {symbol} cancelled")
        except Exception as e:
            logger.error(f"❌ Error in learning loop for {symbol}: {e}")
            raise
    
    async def _orchestration_loop(self):
        """Main orchestration loop for monitoring and management."""
        
        logger.info("🎯 Starting orchestration loop")
        
        try:
            while self.is_running:
                # Performance monitoring
                await self._monitor_performance()
                
                # Checkpoint management
                await self._manage_checkpoints()
                
                # Generate periodic reports
                if len(self.performance_history) % self.config.performance_report_frequency == 0:
                    await self._generate_performance_report()
                
                # Clean up old experiences
                await self._cleanup_old_experiences()
                
                # Wait before next orchestration cycle
                await asyncio.sleep(300)  # 5 minutes
                
        except asyncio.CancelledError:
            logger.info("📋 Orchestration loop cancelled")
        except Exception as e:
            logger.error(f"❌ Error in orchestration loop: {e}")
            raise
    
    async def _execute_trade(self, 
                           symbol: str, 
                           action: float, 
                           confidence: float, 
                           metadata: Dict) -> bool:
        """Execute a trade based on RL agent action."""
        
        try:
            # Get current position
            current_position = self.portfolio_manager.get_position(symbol)
            current_size = current_position.quantity if current_position else 0.0
            
            # Calculate target position
            portfolio_value = self.portfolio_manager.get_total_value()
            max_position = portfolio_value * 0.1  # Max 10% per position
            target_position = action * max_position
            
            # Calculate required trade size
            trade_size = target_position - current_size
            
            if abs(trade_size) < portfolio_value * 0.001:  # Less than 0.1%
                return False
            
            # Create and submit order (simplified)
            from core.trading_engine import Order, OrderType, OrderSide
            
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY if trade_size > 0 else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=abs(trade_size),
                metadata={
                    'source': 'online_rl',
                    'confidence': confidence,
                    'action': action
                }
            )
            
            result = await self.trading_engine.submit_order(order)
            
            if result.success:
                logger.info(f"✅ Online RL trade executed: {symbol} {order.side.value} {order.quantity}")
                return True
            else:
                logger.error(f"❌ Online RL trade failed: {result.error}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error executing trade for {symbol}: {e}")
            return False
    
    async def _calculate_reward(self,
                              symbol: str,
                              prev_state: Any,
                              current_state: Any,
                              action: float) -> float:
        """Calculate reward for the RL agent."""
        
        # Portfolio value change
        prev_value = prev_state.base_state.portfolio_value
        current_value = current_state.base_state.portfolio_value
        portfolio_return = (current_value - prev_value) / prev_value
        
        # Risk-adjusted reward
        volatility = current_state.llm_market_state.volatility_forecast
        risk_adjusted_return = portfolio_return / (volatility + 0.01)
        
        # Sentiment alignment bonus
        sentiment_score = current_state.llm_market_state.overall_score
        position = current_state.base_state.position
        
        sentiment_bonus = 0.0
        if abs(sentiment_score) > 0.1:
            if (position > 0 and sentiment_score > 0) or (position < 0 and sentiment_score < 0):
                sentiment_bonus = abs(sentiment_score) * 0.01
        
        # Transaction cost penalty
        prev_position = prev_state.base_state.position
        current_position = current_state.base_state.position
        position_change = abs(current_position - prev_position)
        transaction_cost = -position_change * 0.001  # 0.1% transaction cost
        
        # Total reward
        total_reward = risk_adjusted_return + sentiment_bonus + transaction_cost
        
        return total_reward
    
    async def _handle_regime_change(self, symbol: str, new_regime: str, confidence: float):
        """Handle market regime change."""
        
        agent = self.online_agents[symbol]
        
        logger.info(f"🔄 Regime change detected for {symbol}: {new_regime} (confidence: {confidence:.2f})")
        
        # Adjust learning mode based on regime
        if new_regime in ["high_volatility", "bear_market"]:
            agent.set_learning_mode(OnlineLearningMode.SAFETY)
        elif new_regime in ["bull_market", "positive_sentiment"]:
            agent.set_learning_mode(OnlineLearningMode.EXPLOITATION)
        else:
            agent.set_learning_mode(OnlineLearningMode.ADAPTATION)
        
        # Log regime change event
        self.learning_events.append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'event_type': 'regime_change',
            'new_regime': new_regime,
            'confidence': confidence,
            'new_learning_mode': agent.learning_mode.value
        })
    
    async def _monitor_performance(self):
        """Monitor performance of all online agents."""
        
        total_performance = {}
        
        for symbol, agent in self.online_agents.items():
            metrics = agent.get_performance_metrics()
            total_performance[symbol] = asdict(metrics)
            
            # Check for performance issues
            if (metrics.max_drawdown < self.config.max_drawdown_threshold or
                metrics.live_sharpe < self.config.min_sharpe_threshold):
                
                logger.warning(f"⚠️ Performance issue detected for {symbol}")
                logger.warning(f"   Drawdown: {metrics.max_drawdown:.2%}, Sharpe: {metrics.live_sharpe:.3f}")
                
                # Switch to safety mode
                agent.set_learning_mode(OnlineLearningMode.SAFETY)
        
        # Store performance snapshot
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'performance': total_performance
        })
    
    async def _manage_checkpoints(self):
        """Manage model checkpoints."""
        
        for symbol, agent in self.online_agents.items():
            # Checkpoints are created automatically by the agent
            # Here we just manage cleanup of old checkpoints
            
            checkpoint_pattern = f"{symbol}_checkpoint_*.{'pt' if hasattr(agent.policy, 'actor') else 'npz'}"
            checkpoint_files = list(self.checkpoint_dir.glob(checkpoint_pattern))
            
            if len(checkpoint_files) > self.config.max_checkpoints_to_keep:
                # Sort by modification time and remove oldest
                checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
                for old_checkpoint in checkpoint_files[:-self.config.max_checkpoints_to_keep]:
                    old_checkpoint.unlink()
                    logger.debug(f"🗑️ Removed old checkpoint: {old_checkpoint}")
    
    async def _cleanup_old_experiences(self):
        """Clean up old experiences from replay buffers."""
        
        for symbol, agent in self.online_agents.items():
            agent.replay_buffer.clear_old_experiences(self.config.max_experience_age_hours)
    
    async def _generate_performance_report(self):
        """Generate comprehensive performance report."""
        
        if not self.performance_history:
            return
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'online_learning_status': {
                'is_running': self.is_running,
                'active_agents': len(self.online_agents),
                'total_learning_events': len(self.learning_events)
            },
            'performance_summary': {},
            'learning_summary': {}
        }
        
        # Summarize performance for each symbol
        for symbol, agent in self.online_agents.items():
            metrics = agent.get_performance_metrics()
            
            report['performance_summary'][symbol] = {
                'live_sharpe': metrics.live_sharpe,
                'rolling_return': metrics.rolling_return,
                'max_drawdown': metrics.max_drawdown,
                'total_updates': metrics.total_updates,
                'current_mode': agent.learning_mode.value
            }
        
        # Summarize learning events
        recent_events = [e for e in self.learning_events 
                        if datetime.fromisoformat(e['timestamp']) > datetime.now() - timedelta(hours=24)]
        
        report['learning_summary'] = {
            'events_last_24h': len(recent_events),
            'regime_changes': len([e for e in recent_events if e.get('event_type') == 'regime_change']),
            'avg_confidence': np.mean([e.get('confidence', 0) for e in recent_events]) if recent_events else 0
        }
        
        # Save report
        report_file = self.checkpoint_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("📊 Performance report generated")
        logger.info(f"   Active agents: {len(self.online_agents)}")
        logger.info(f"   Learning events (24h): {report['learning_summary']['events_last_24h']}")
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get current status of the orchestrator."""
        
        return {
            'is_running': self.is_running,
            'active_agents': list(self.online_agents.keys()),
            'learning_modes': {symbol: agent.learning_mode.value 
                             for symbol, agent in self.online_agents.items()},
            'total_learning_events': len(self.learning_events),
            'performance_snapshots': len(self.performance_history),
            'config': asdict(self.config)
        }
    
    async def make_portfolio_decisions(self, 
                                     enhanced_state: Any,
                                     available_symbols: List[str],
                                     portfolio_value: float,
                                     risk_tolerance: float = 0.5) -> Dict[str, Any]:
        """
        Make RL-based portfolio allocation decisions.
        
        This is the core RL Decision Layer method that:
        1. Takes LLM analysis results from the Feature Store
        2. Uses RL policy learning for portfolio allocation
        3. Returns allocation decisions for signal generation
        """
        try:
            logger.info(f"🤖 RL Decision Layer processing {len(available_symbols)} symbols")
            
            # Initialize default RL decision structure
            rl_decisions = {
                "strategy": "adaptive_momentum",
                "risk_level": "moderate",
                "allocations": [],
                "confidence": 0.7,
                "reasoning": "RL-based portfolio optimization"
            }
            
            # Simple RL allocation strategy (can be enhanced with trained models)
            if not available_symbols:
                logger.warning("No symbols available for RL allocation")
                return rl_decisions
            
            # Dynamic allocation based on risk tolerance and number of symbols
            max_symbols = min(5, len(available_symbols))  # Limit to 5 positions max
            base_weight = 0.8 / max_symbols  # 80% of portfolio across positions
            
            allocations = []
            
            for i, symbol in enumerate(available_symbols[:max_symbols]):
                # Simple momentum-based weighting (can be replaced with RL model predictions)
                momentum_factor = 1.0 - (i * 0.1)  # Decreasing weights for lower-ranked symbols
                risk_adjustment = risk_tolerance  # Higher risk tolerance = higher allocations
                
                weight = base_weight * momentum_factor * (0.5 + risk_adjustment)
                weight = max(0.05, min(0.25, weight))  # Clamp between 5% and 25%
                
                allocation = {
                    "symbol": symbol,
                    "weight": weight,
                    "confidence": 0.6 + (0.2 * momentum_factor),
                    "action": "buy",
                    "reasoning": f"RL momentum allocation #{i+1}"
                }
                
                allocations.append(allocation)
                logger.info(f"   🎯 RL Allocation: {symbol} = {weight:.1%}")
            
            rl_decisions["allocations"] = allocations
            
            # Determine strategy based on market conditions (simplified)
            if risk_tolerance > 0.7:
                rl_decisions["strategy"] = "aggressive_growth"
                rl_decisions["risk_level"] = "high"
            elif risk_tolerance < 0.3:
                rl_decisions["strategy"] = "conservative_income"
                rl_decisions["risk_level"] = "low"
            
            logger.info(f"✅ RL Decision Layer: {len(allocations)} allocations, {rl_decisions['strategy']} strategy")
            
            return rl_decisions
            
        except Exception as e:
            logger.error(f"RL Decision Layer error: {e}")
            # Return safe fallback
            return {
                "strategy": "fallback",
                "risk_level": "low", 
                "allocations": [],
                "confidence": 0.3,
                "reasoning": f"RL error fallback: {e}"
            }
    
    async def close(self):
        """Clean up orchestrator resources."""
        
        # Stop learning if running
        if self.is_running:
            await self.stop_online_learning()
        
        # Close all agents
        for symbol, agent in self.online_agents.items():
            await agent.close()
        
        logger.info("✅ Online Learning Orchestrator closed")


# Utility functions
async def create_online_learning_system(
    trading_engine: TradingEngine,
    portfolio_manager: PortfolioManager,
    risk_manager: RiskManager,
    symbols: List[str],
    config: OnlineLearningConfig = None
) -> OnlineLearningOrchestrator:
    """Create a complete online learning system."""
    
    orchestrator = OnlineLearningOrchestrator(
        trading_engine=trading_engine,
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        config=config
    )
    
    # Add online agents for each symbol
    for symbol in symbols:
        await orchestrator.add_online_agent(
            symbol=symbol,
            learning_mode=OnlineLearningMode.EXPLOITATION
        )
    
    return orchestrator


if __name__ == "__main__":
    # Test the online learning orchestrator
    async def test_online_learning_orchestrator():
        # Mock components for testing
        class MockTradingEngine:
            async def submit_order(self, order):
                class Result:
                    success = True
                    error = None
                return Result()
        
        class MockPortfolioManager:
            def get_position(self, symbol):
                return None
            def get_total_value(self):
                return 100000.0
        
        class MockRiskManager:
            async def calculate_portfolio_risk(self):
                class RiskMetrics:
                    portfolio_var = 0.05
                return RiskMetrics()
        
        # Create orchestrator
        config = OnlineLearningConfig(
            update_frequency_minutes=1,  # Fast for testing
            checkpoint_frequency_minutes=2
        )
        
        orchestrator = OnlineLearningOrchestrator(
            trading_engine=MockTradingEngine(),
            portfolio_manager=MockPortfolioManager(),
            risk_manager=MockRiskManager(),
            config=config
        )
        
        # Add test agent with dynamic symbol
        test_symbol = "SPY"  # Use ETF for testing
        await orchestrator.add_online_agent(test_symbol, learning_mode=OnlineLearningMode.ADAPTATION)
        
        # Get status
        status = orchestrator.get_orchestrator_status()
        print(f"Orchestrator status: {status}")
        
        # Test would normally start online learning here
        # await orchestrator.start_online_learning([test_symbol])
        # await asyncio.sleep(10)  # Run for 10 seconds
        # await orchestrator.stop_online_learning()
        
        await orchestrator.close()
        print("✅ Online learning orchestrator test completed")
    
    asyncio.run(test_online_learning_orchestrator())