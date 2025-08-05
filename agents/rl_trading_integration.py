"""
RL Trading System Integration
Integrates RL agents with the existing trading system execution engine
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

from agents.rl_trading_agent import RLTradingAgent, TrainingConfig
from agents.llm_rl_integration import EnhancedLLMTradingEnvironment, LLMStateEnricher, EnhancedTradingState
from agents.rl_backtesting_framework import RLBacktestingFramework, BacktestConfig
# Use existing trading_system infrastructure
from trading_system.core.trading_engine import UnifiedTradingEngine
from tools.alpaca_client import alpaca_client

logger = logging.getLogger(__name__)


class RLStrategy(Enum):
    """RL strategy types."""
    SINGLE_ASSET = "single_asset"
    MULTI_ASSET = "multi_asset"
    REGIME_ADAPTIVE = "regime_adaptive"
    ENSEMBLE = "ensemble"


@dataclass
class RLSignal:
    """RL-generated trading signal."""
    symbol: str
    action: float  # Position size [-1, 1]
    confidence: float
    reasoning: str
    timestamp: datetime
    
    # Enhanced signal data
    sentiment_score: float
    technical_alignment: float
    risk_level: str
    regime: str
    
    # Metadata
    model_version: str
    signal_strength: float
    expected_return: float
    expected_volatility: float


@dataclass
class RLTradeDecision:
    """Final trading decision from RL agent."""
    signal: RLSignal
    current_position: float
    target_position: float
    position_change: float
    order_size: float
    order_side: OrderSide
    risk_adjusted: bool
    
    # Decision metadata
    decision_id: str
    execution_priority: str  # "high", "medium", "low"
    time_horizon: str  # "immediate", "short", "medium"


class RLTradingIntegration:
    """
    Integrates RL agents with the existing trading system.
    """
    
    def __init__(self,
                 trading_engine: TradingEngine,
                 portfolio_manager: PortfolioManager,
                 risk_manager: RiskManager,
                 strategy_type: RLStrategy = RLStrategy.SINGLE_ASSET):
        
        self.trading_engine = trading_engine
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        self.strategy_type = strategy_type
        
        # RL components
        self.rl_agents: Dict[str, RLTradingAgent] = {}
        self.state_enricher = LLMStateEnricher()
        self.market_data_fetcher = MarketDataFetcher()
        
        # Signal management
        self.active_signals: Dict[str, RLSignal] = {}
        self.signal_history: List[RLSignal] = []
        self.decision_history: List[RLTradeDecision] = []
        
        # Performance tracking
        self.performance_metrics = {
            'signals_generated': 0,
            'trades_executed': 0,
            'successful_trades': 0,
            'total_pnl': 0.0,
            'signal_accuracy': 0.0
        }
        
        # Configuration
        self.config = {
            'max_position_size': 1.0,
            'signal_threshold': 0.1,
            'confidence_threshold': 0.6,
            'risk_limit_per_trade': 0.02,
            'rebalance_frequency': timedelta(hours=1),
            'enable_risk_override': True
        }
        
        logger.info(f"✅ RL Trading Integration initialized with {strategy_type.value} strategy")
    
    async def add_rl_agent(self, symbol: str, agent: RLTradingAgent, weight: float = 1.0):
        """Add an RL agent for a specific symbol."""
        
        self.rl_agents[symbol] = {
            'agent': agent,
            'weight': weight,
            'last_signal_time': None,
            'performance': {
                'signals': 0,
                'accuracy': 0.0,
                'avg_return': 0.0
            }
        }
        
        logger.info(f"✅ Added RL agent for {symbol} with weight {weight}")
    
    async def generate_trading_signals(self, symbols: List[str] = None) -> List[RLSignal]:
        """Generate trading signals from RL agents."""
        
        if symbols is None:
            symbols = list(self.rl_agents.keys())
        
        signals = []
        
        for symbol in symbols:
            if symbol not in self.rl_agents:
                logger.warning(f"⚠️ No RL agent available for {symbol}")
                continue
            
            try:
                signal = await self._generate_signal_for_symbol(symbol)
                if signal:
                    signals.append(signal)
                    self.active_signals[symbol] = signal
                    self.signal_history.append(signal)
                    
                    self.performance_metrics['signals_generated'] += 1
                    
            except Exception as e:
                logger.error(f"❌ Error generating signal for {symbol}: {e}")
        
        logger.info(f"📈 Generated {len(signals)} trading signals")
        return signals
    
    async def _generate_signal_for_symbol(self, symbol: str) -> Optional[RLSignal]:
        """Generate trading signal for a specific symbol."""
        
        agent_info = self.rl_agents[symbol]
        agent = agent_info['agent']
        
        if not agent.is_trained:
            logger.warning(f"⚠️ Agent for {symbol} is not trained")
            return None
        
        # Get current market data
        market_data = await self.market_data_fetcher.get_realtime_data(symbol)
        if market_data is None or len(market_data) < 10:
            logger.warning(f"⚠️ Insufficient market data for {symbol}")
            return None
        
        # Create trading environment for inference
        env = EnhancedLLMTradingEnvironment(
            symbol=symbol,
            initial_balance=100000,  # Dummy balance for inference
            use_continuous_actions=True
        )
        
        try:
            env.set_data(market_data.tail(100))  # Use recent data
            
            # Get current enhanced state
            obs = await env.reset()
            
            # Get prediction from agent
            action, confidence = await agent.predict(obs, deterministic=True)
            
            # Get enhanced state for additional insights
            enhanced_state = await env._get_enhanced_current_state()
            
            # Calculate signal strength
            signal_strength = abs(action) * confidence
            
            # Only generate signal if above threshold
            if signal_strength < self.config['signal_threshold']:
                logger.debug(f"Signal strength {signal_strength:.3f} below threshold for {symbol}")
                return None
            
            # Create reasoning
            reasoning = self._generate_signal_reasoning(action, enhanced_state, confidence)
            
            # Create RL signal
            signal = RLSignal(
                symbol=symbol,
                action=float(action),
                confidence=confidence,
                reasoning=reasoning,
                timestamp=datetime.now(),
                sentiment_score=enhanced_state.llm_market_state.overall_score,
                technical_alignment=enhanced_state.technical_alignment,
                risk_level=enhanced_state.llm_market_state.risk_level,
                regime=enhanced_state.llm_market_state.regime_prediction.name,
                model_version=f"rl_agent_v1.0",
                signal_strength=signal_strength,
                expected_return=self._estimate_expected_return(action, enhanced_state),
                expected_volatility=enhanced_state.llm_market_state.volatility_forecast
            )
            
            return signal
            
        finally:
            await env.close()
    
    def _generate_signal_reasoning(self, 
                                 action: float, 
                                 enhanced_state: EnhancedTradingState, 
                                 confidence: float) -> str:
        """Generate human-readable reasoning for the signal."""
        
        reasoning_parts = []
        
        # Action interpretation
        if action > 0.5:
            reasoning_parts.append("Strong BUY signal")
        elif action > 0.1:
            reasoning_parts.append("Moderate BUY signal")
        elif action < -0.5:
            reasoning_parts.append("Strong SELL signal")
        elif action < -0.1:
            reasoning_parts.append("Moderate SELL signal")
        else:
            reasoning_parts.append("HOLD signal")
        
        # Confidence level
        if confidence > 0.8:
            reasoning_parts.append("with high confidence")
        elif confidence > 0.6:
            reasoning_parts.append("with moderate confidence")
        else:
            reasoning_parts.append("with low confidence")
        
        # Market sentiment
        sentiment = enhanced_state.llm_market_state.overall_score
        if sentiment > 0.3:
            reasoning_parts.append("driven by positive market sentiment")
        elif sentiment < -0.3:
            reasoning_parts.append("driven by negative market sentiment")
        
        # Technical alignment
        if enhanced_state.technical_alignment > 0.7:
            reasoning_parts.append("supported by strong technical alignment")
        elif enhanced_state.technical_alignment < 0.3:
            reasoning_parts.append("with weak technical support")
        
        # Risk considerations
        if enhanced_state.llm_market_state.risk_level == "high":
            reasoning_parts.append("but high risk environment")
        
        return ". ".join(reasoning_parts) + "."
    
    def _estimate_expected_return(self, action: float, enhanced_state: EnhancedTradingState) -> float:
        """Estimate expected return based on signal and market state."""
        
        # Base expected return from action strength
        base_return = abs(action) * 0.02  # 2% max expected return
        
        # Adjust for sentiment alignment
        sentiment_factor = 1.0 + (enhanced_state.llm_market_state.overall_score * 0.5)
        
        # Adjust for technical alignment
        technical_factor = 1.0 + (enhanced_state.technical_alignment - 0.5)
        
        # Adjust for consensus strength
        consensus_factor = 1.0 + (enhanced_state.consensus_strength - 0.5) * 0.3
        
        expected_return = base_return * sentiment_factor * technical_factor * consensus_factor
        
        # Apply sign based on action direction
        return expected_return if action > 0 else -expected_return
    
    async def execute_trading_decisions(self, signals: List[RLSignal]) -> List[RLTradeDecision]:
        """Convert signals to trading decisions and execute them."""
        
        decisions = []
        
        for signal in signals:
            try:
                decision = await self._create_trading_decision(signal)
                if decision and decision.order_size > 0:
                    
                    # Risk check
                    if await self._risk_check_decision(decision):
                        
                        # Execute the trade
                        success = await self._execute_decision(decision)
                        
                        if success:
                            decisions.append(decision)
                            self.decision_history.append(decision)
                            self.performance_metrics['trades_executed'] += 1
                            
                            logger.info(f"✅ Executed RL trade: {decision.signal.symbol} {decision.order_side.value} {decision.order_size}")
                        else:
                            logger.error(f"❌ Failed to execute trade for {signal.symbol}")
                    else:
                        logger.warning(f"⚠️ Risk check failed for {signal.symbol}")
                
            except Exception as e:
                logger.error(f"❌ Error processing signal for {signal.symbol}: {e}")
        
        logger.info(f"💼 Executed {len(decisions)} trading decisions")
        return decisions
    
    async def _create_trading_decision(self, signal: RLSignal) -> Optional[RLTradeDecision]:
        """Create a trading decision from an RL signal."""
        
        # Get current position
        current_position = self.portfolio_manager.get_position(signal.symbol)
        current_size = current_position.quantity if current_position else 0.0
        
        # Calculate target position based on signal
        max_position = self.config['max_position_size'] * self.portfolio_manager.get_total_value()
        target_position = signal.action * max_position
        
        # Calculate position change
        position_change = target_position - current_size
        
        # Check if change is significant enough
        if abs(position_change) < max_position * 0.01:  # Less than 1% of max position
            return None
        
        # Determine order details
        order_size = abs(position_change)
        order_side = OrderSide.BUY if position_change > 0 else OrderSide.SELL
        
        # Apply risk adjustments
        risk_adjusted_size = await self._apply_risk_adjustments(
            signal.symbol, order_size, signal
        )
        
        # Determine execution priority based on signal strength and confidence
        if signal.signal_strength > 0.7 and signal.confidence > 0.8:
            priority = "high"
            time_horizon = "immediate"
        elif signal.signal_strength > 0.4:
            priority = "medium"
            time_horizon = "short"
        else:
            priority = "low"
            time_horizon = "medium"
        
        decision = RLTradeDecision(
            signal=signal,
            current_position=current_size,
            target_position=target_position,
            position_change=position_change,
            order_size=risk_adjusted_size,
            order_side=order_side,
            risk_adjusted=(risk_adjusted_size != order_size),
            decision_id=f"rl_{signal.symbol}_{int(datetime.now().timestamp())}",
            execution_priority=priority,
            time_horizon=time_horizon
        )
        
        return decision
    
    async def _apply_risk_adjustments(self, 
                                    symbol: str, 
                                    order_size: float, 
                                    signal: RLSignal) -> float:
        """Apply risk management adjustments to order size."""
        
        # Get current risk metrics
        risk_metrics = await self.risk_manager.calculate_portfolio_risk()
        
        # Check portfolio heat (total risk exposure)
        if risk_metrics.portfolio_var > 0.15:  # More than 15% VaR
            order_size *= 0.5  # Reduce by 50%
            logger.info(f"Reduced order size due to high portfolio VaR")
        
        # Check individual position risk
        current_position = self.portfolio_manager.get_position(symbol)
        if current_position:
            position_value = abs(current_position.quantity * current_position.current_price)
            portfolio_value = self.portfolio_manager.get_total_value()
            position_weight = position_value / portfolio_value
            
            if position_weight > 0.1:  # More than 10% of portfolio
                order_size *= 0.7  # Reduce by 30%
                logger.info(f"Reduced order size due to high position concentration")
        
        # Check signal risk level
        if signal.risk_level == "high":
            order_size *= 0.6  # Reduce by 40% for high-risk signals
        elif signal.risk_level == "medium":
            order_size *= 0.8  # Reduce by 20% for medium-risk signals
        
        # Check expected volatility
        if signal.expected_volatility > 0.4:  # High expected volatility
            order_size *= 0.7
        
        # Minimum order size check
        min_order_value = 1000  # $1000 minimum
        current_price = await self.market_data_fetcher.get_current_price(symbol)
        min_order_size = min_order_value / current_price if current_price else 0
        
        if order_size < min_order_size:
            return 0.0  # Cancel order if too small
        
        return order_size
    
    async def _risk_check_decision(self, decision: RLTradeDecision) -> bool:
        """Perform final risk check on trading decision."""
        
        if not self.config['enable_risk_override']:
            return True
        
        # Check if trade would exceed risk limits
        portfolio_value = self.portfolio_manager.get_total_value()
        trade_value = decision.order_size * await self.market_data_fetcher.get_current_price(decision.signal.symbol)
        trade_risk = trade_value / portfolio_value
        
        if trade_risk > self.config['risk_limit_per_trade']:
            logger.warning(f"Trade risk {trade_risk:.3f} exceeds limit {self.config['risk_limit_per_trade']}")
            return False
        
        # Check drawdown limits
        risk_metrics = await self.risk_manager.calculate_portfolio_risk()
        if risk_metrics.current_drawdown < -0.10:  # More than 10% drawdown
            logger.warning(f"Portfolio drawdown {risk_metrics.current_drawdown:.2%} exceeds limit")
            return False
        
        # Check correlation with existing positions
        correlations = await self._check_position_correlations(decision.signal.symbol)
        if correlations > 0.8:  # High correlation with existing positions
            logger.warning(f"High correlation {correlations:.2f} with existing positions")
            return False
        
        return True
    
    async def _check_position_correlations(self, symbol: str) -> float:
        """Check correlation with existing positions."""
        
        positions = self.portfolio_manager.get_all_positions()
        if not positions:
            return 0.0
        
        # Simplified correlation check - would need historical correlation data
        # For now, return a dummy value
        return 0.3
    
    async def _execute_decision(self, decision: RLTradeDecision) -> bool:
        """Execute the trading decision."""
        
        try:
            # Create order
            order = Order(
                symbol=decision.signal.symbol,
                side=decision.order_side,
                order_type=OrderType.MARKET,
                quantity=decision.order_size,
                metadata={
                    'source': 'rl_agent',
                    'signal_id': id(decision.signal),
                    'confidence': decision.signal.confidence,
                    'reasoning': decision.signal.reasoning
                }
            )
            
            # Submit order through trading engine
            result = await self.trading_engine.submit_order(order)
            
            if result.success:
                # Update performance tracking
                self.performance_metrics['successful_trades'] += 1
                
                logger.info(f"✅ RL order executed: {order.symbol} {order.side.value} {order.quantity}")
                return True
            else:
                logger.error(f"❌ Order execution failed: {result.error}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error executing decision: {e}")
            return False
    
    async def run_trading_loop(self, symbols: List[str], interval: timedelta = timedelta(minutes=30)):
        """Run the main RL trading loop."""
        
        logger.info(f"🔄 Starting RL trading loop for {symbols}")
        logger.info(f"⏰ Rebalance interval: {interval}")
        
        while True:
            try:
                loop_start = datetime.now()
                
                # Generate signals
                signals = await self.generate_trading_signals(symbols)
                
                # Execute decisions
                if signals:
                    decisions = await self.execute_trading_decisions(signals)
                    
                    # Log performance
                    await self._log_performance_metrics()
                
                # Calculate sleep time
                loop_duration = (datetime.now() - loop_start).total_seconds()
                sleep_time = max(0, interval.total_seconds() - loop_duration)
                
                logger.info(f"💤 Loop completed in {loop_duration:.1f}s, sleeping for {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("⏹️ Trading loop interrupted by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in trading loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _log_performance_metrics(self):
        """Log current performance metrics."""
        
        total_signals = self.performance_metrics['signals_generated']
        successful_trades = self.performance_metrics['successful_trades']
        total_trades = self.performance_metrics['trades_executed']
        
        if total_trades > 0:
            success_rate = successful_trades / total_trades
        else:
            success_rate = 0.0
        
        logger.info(f"📊 Performance: {total_signals} signals, {total_trades} trades, {success_rate:.2%} success rate")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        
        total_signals = len(self.signal_history)
        total_decisions = len(self.decision_history)
        
        # Calculate signal accuracy (simplified)
        recent_signals = self.signal_history[-50:] if len(self.signal_history) > 50 else self.signal_history
        accuracy = np.mean([1 if s.expected_return > 0 else 0 for s in recent_signals]) if recent_signals else 0.0
        
        # Calculate average confidence
        avg_confidence = np.mean([s.confidence for s in recent_signals]) if recent_signals else 0.0
        
        return {
            'total_signals_generated': total_signals,
            'total_trading_decisions': total_decisions,
            'trades_executed': self.performance_metrics['trades_executed'],
            'successful_trades': self.performance_metrics['successful_trades'],
            'signal_accuracy': accuracy,
            'average_confidence': avg_confidence,
            'active_agents': len(self.rl_agents),
            'last_signal_time': self.signal_history[-1].timestamp if self.signal_history else None
        }
    
    async def close(self):
        """Clean up resources."""
        
        # Close all RL agents
        for symbol, agent_info in self.rl_agents.items():
            await agent_info['agent'].close()
        
        # Close state enricher
        await self.state_enricher.close()
        
        logger.info("✅ RL Trading Integration closed")


# Utility functions
async def create_rl_trading_system(
    trading_engine: TradingEngine,
    portfolio_manager: PortfolioManager,
    risk_manager: RiskManager,
    symbols: List[str],
    strategy_type: RLStrategy = RLStrategy.SINGLE_ASSET
) -> RLTradingIntegration:
    """Create a complete RL trading system."""
    
    integration = RLTradingIntegration(
        trading_engine=trading_engine,
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        strategy_type=strategy_type
    )
    
    # Create and add RL agents for each symbol
    for symbol in symbols:
        config = TrainingConfig(
            total_timesteps=50000,
            learning_rate=3e-4,
            sentiment_weight=0.4
        )
        
        agent = RLTradingAgent(symbol=symbol, config=config)
        await integration.add_rl_agent(symbol, agent)
    
    return integration


if __name__ == "__main__":
    # Test the RL integration
    async def test_rl_integration():
        # This would normally use real trading system components
        # For testing, we'll create mock components
        
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
            
            def get_all_positions(self):
                return []
        
        class MockRiskManager:
            async def calculate_portfolio_risk(self):
                class RiskMetrics:
                    portfolio_var = 0.05
                    current_drawdown = -0.02
                return RiskMetrics()
        
        # Create integration
        integration = RLTradingIntegration(
            trading_engine=MockTradingEngine(),
            portfolio_manager=MockPortfolioManager(),
            risk_manager=MockRiskManager()
        )
        
        # Add a test agent with dynamic symbol
        test_symbol = "SPY"  # Use ETF for testing
        config = TrainingConfig(total_timesteps=1000)
        agent = RLTradingAgent(test_symbol, config=config)
        await integration.add_rl_agent(test_symbol, agent)
        
        # Generate test signals
        signals = await integration.generate_trading_signals([test_symbol])
        print(f"Generated {len(signals)} signals")
        
        # Get performance summary
        summary = integration.get_performance_summary()
        print(f"Performance summary: {summary}")
        
        await integration.close()
    
    asyncio.run(test_rl_integration())