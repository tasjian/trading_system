#!/usr/bin/env python3
"""
Online RL Integration Bridge
Connects the new online RL system to the existing trading workflow.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from agents.online_rl_system import (
    OnlineRLTradingSystem, 
    create_online_rl_system, 
    OnlineLearningConfig,
    MarketRegime
)
from agents.unified_reward_calculator import (
    UnifiedRewardCalculator, TradeMetrics, TradeType, RewardComponents
)
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Enhanced trading signal with RL confidence."""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    quantity: float
    confidence: float
    reasoning: str
    rl_score: float
    regime: str
    uncertainty: float

class OnlineRLAgent:
    """Integration layer between online RL system and trading workflow."""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.system = None
        self.initialized = False
        
        # Initialize unified reward calculator
        self.reward_calculator = UnifiedRewardCalculator(
            lambda_c=1.2,    # Higher cost penalty weight
            lambda_r=1.5,    # Higher risk penalty weight
            lambda_e=0.8,    # Moderate execution penalty weight
            lambda_s=0.6,    # Moderate sentiment alignment weight
            lambda_reg=0.4,  # Lower regime alignment weight
            lambda_div=0.3   # Lower diversification bonus weight
        )
        
        # Performance tracking
        self.signal_history = []
        self.execution_history = []
        
        # State management
        self.last_market_state = None
        self.last_action = None
        self.last_portfolio_value = None
        self.last_trade_metrics = None
        
        # Configuration
        self.config = OnlineLearningConfig(
            # Conservative settings for live trading
            batch_update_frequency=timedelta(hours=6),      # 4 updates per day
            stable_update_frequency=timedelta(days=7),      # Weekly stable updates
            learner_performance_threshold=0.02,             # 2% outperformance needed
            performance_evaluation_window=200,              # 200 trades for evaluation
            
            # Safety constraints
            safety_constraints=None  # Will use defaults (conservative)
        )
        
        logger.info(f"🤖 Online RL Agent initialized for {len(symbols)} symbols")
    
    async def initialize_system(self):
        """Initialize the online RL system."""
        
        if self.initialized:
            return
            
        try:
            self.system = create_online_rl_system(
                symbols=self.symbols,
                **self.config.__dict__
            )
            
            # Try to load previous state
            state_file = "data/online_rl_state.json"
            try:
                self.system.load_system_state(state_file)
                logger.info("✅ Loaded previous RL system state")
            except Exception as e:
                logger.info(f"No previous state found, starting fresh: {e}")
            
            self.initialized = True
            logger.info("🚀 Online RL system initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RL system: {e}")
            raise
    
    def convert_market_data_to_state(self, 
                                   market_data: Dict[str, Any], 
                                   portfolio_data: Dict[str, Any]) -> np.ndarray:
        """Convert market and portfolio data to RL state vector."""
        
        state_features = []
        
        for symbol in self.symbols:
            symbol_data = market_data.get(symbol, {})
            
            # Price features (normalized)
            current_price = symbol_data.get('price', 0.0)
            price_change = symbol_data.get('price_change_pct', 0.0)
            
            # Volume features
            volume = symbol_data.get('volume', 0.0)
            avg_volume = symbol_data.get('avg_volume', volume)
            volume_ratio = volume / max(avg_volume, 1.0)
            
            # Technical indicators
            rsi = symbol_data.get('rsi', 50.0) / 100.0  # Normalize to 0-1
            macd = symbol_data.get('macd', 0.0)
            bb_position = symbol_data.get('bb_position', 0.5)  # Bollinger Band position
            
            # Portfolio features
            current_position = portfolio_data.get(symbol, {}).get('quantity', 0.0)
            position_value = portfolio_data.get(symbol, {}).get('market_value', 0.0)
            
            # Sentiment features
            sentiment_score = symbol_data.get('sentiment_score', 0.0)
            news_count = symbol_data.get('news_count', 0.0)
            
            # Combine features for this symbol
            symbol_features = [
                price_change,           # 1. Price momentum
                volume_ratio,           # 2. Volume activity
                rsi,                   # 3. RSI momentum
                macd,                  # 4. MACD trend
                bb_position,           # 5. Bollinger position
                current_position,      # 6. Current position
                position_value / 10000, # 7. Position value (scaled)
                sentiment_score,       # 8. Sentiment
                news_count / 10,       # 9. News activity (scaled)
                np.log1p(current_price) if current_price > 0 else 0.0,  # 10. Log price
                np.tanh(price_change * 10),  # 11. Bounded price change
                min(1.0, volume_ratio)       # 12. Bounded volume ratio
            ]
            
            state_features.extend(symbol_features)
        
        return np.array(state_features, dtype=np.float32)
    
    def extract_market_metadata(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract market-level metadata for regime detection and uncertainty."""
        
        # Calculate market-wide volatility
        price_changes = []
        volumes = []
        
        for symbol in self.symbols:
            symbol_data = market_data.get(symbol, {})
            price_change = symbol_data.get('price_change_pct', 0.0)
            volume = symbol_data.get('volume', 0.0)
            
            if abs(price_change) < 0.5:  # Filter outliers
                price_changes.append(price_change)
            if volume > 0:
                volumes.append(volume)
        
        market_volatility = np.std(price_changes) if price_changes else 0.02
        market_trend = np.mean(price_changes) if price_changes else 0.0
        avg_volume = np.mean(volumes) if volumes else 0.0
        
        # Data completeness check
        complete_symbols = sum(1 for s in self.symbols if s in market_data and market_data[s].get('price', 0) > 0)
        data_completeness = complete_symbols / max(1, len(self.symbols))  # Prevent division by zero
        
        return {
            'volatility': market_volatility,
            'trend': market_trend,
            'avg_volume': avg_volume,
            'data_completeness': data_completeness
        }
    
    async def generate_trading_signals(self, 
                                     market_data: Dict[str, Any],
                                     portfolio_data: Dict[str, Any],
                                     portfolio_value: float) -> List[TradingSignal]:
        """Generate trading signals using the online RL system."""
        
        if not self.initialized:
            await self.initialize_system()
        
        try:
            # Convert to RL state
            current_state = self.convert_market_data_to_state(market_data, portfolio_data)
            market_metadata = self.extract_market_metadata(market_data)
            market_metadata['portfolio_value'] = portfolio_value
            
            # Calculate reward using unified reward calculator
            previous_reward = 0.0
            if self.last_portfolio_value is not None and self.last_trade_metrics is not None:
                # Use comprehensive reward calculation
                reward_components = self.reward_calculator.calculate_reward(self.last_trade_metrics)
                previous_reward = reward_components.total_reward
                
                # Log detailed reward breakdown for debugging
                if abs(previous_reward) > 0.001:  # Only log significant rewards
                    breakdown = self.reward_calculator.get_reward_breakdown(self.last_trade_metrics)
                    logger.debug(f"RL Reward breakdown: PnL={breakdown['pnl_change']:.4f}, "
                               f"Cost={breakdown['cost_penalty']:.4f}, "
                               f"Risk={breakdown['risk_penalty']:.4f}, "
                               f"Total={previous_reward:.4f}")
            elif self.last_portfolio_value is not None and self.last_portfolio_value > 0:
                # Fallback to simple calculation if trade metrics unavailable
                previous_reward = (portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
            elif self.last_portfolio_value is not None:
                # Handle case where last portfolio value was 0 or negative
                previous_reward = 0.01 if portfolio_value > 0 else 0.0
            
            # Process market step with RL system
            action, action_info = await self.system.process_market_step(
                market_state=current_state,
                market_data=market_metadata,
                previous_action=self.last_action,
                previous_reward=previous_reward,
                deterministic=False  # Allow exploration in live trading
            )
            
            # Convert RL actions to trading signals
            signals = self._convert_actions_to_signals(
                action, 
                action_info, 
                market_data, 
                portfolio_data
            )
            
            # Create trade metrics for next reward calculation
            self.last_trade_metrics = self._create_trade_metrics(
                portfolio_value, action_info, market_data, portfolio_data, signals
            )
            
            # Update state tracking
            self.last_market_state = current_state
            self.last_action = action
            self.last_portfolio_value = portfolio_value
            
            # Log RL system performance
            stats = self.system.get_training_stats()
            logger.info(f"🤖 RL Agent: {action_info['agent']} policy, "
                       f"regime={action_info['regime']}, "
                       f"uncertainty={action_info['uncertainty']:.3f}, "
                       f"updates={stats['total_updates']}")
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ RL signal generation failed: {e}")
            return []
    
    def _convert_actions_to_signals(self, 
                                  actions: np.ndarray,
                                  action_info: Dict[str, Any],
                                  market_data: Dict[str, Any],
                                  portfolio_data: Dict[str, Any]) -> List[TradingSignal]:
        """Convert RL actions to trading signals."""
        
        signals = []
        action_threshold = 0.1  # Minimum action magnitude to generate signal
        
        for i, symbol in enumerate(self.symbols):
            if i >= len(actions):
                continue
                
            action_value = actions[i]
            
            # Skip if action is too small
            if abs(action_value) < action_threshold:
                continue
            
            # Get current position
            current_position = portfolio_data.get(symbol, {}).get('quantity', 0.0)
            current_price = market_data.get(symbol, {}).get('price', 0.0)
            
            if current_price <= 0:
                continue
            
            # Determine action type and quantity
            if action_value > 0:  # Buy signal
                action_type = 'buy'
                # Calculate position size based on action magnitude and portfolio constraints
                max_position_pct = settings.max_position_size  # From config
                current_portfolio_value = self.last_portfolio_value or 100000
                max_position_value = current_portfolio_value * max_position_pct
                
                # Scale action to position size
                desired_position_value = max_position_value * min(1.0, action_value * 2)  # Scale up action
                desired_quantity = desired_position_value / current_price
                
                # Adjust for current position
                quantity = max(0, desired_quantity - current_position)
                
            else:  # Sell signal
                action_type = 'sell'
                # Sell portion based on action magnitude
                sell_fraction = min(1.0, abs(action_value) * 2)  # Scale up action
                quantity = current_position * sell_fraction
                
            # Skip if quantity is too small
            if quantity < 0.01:  # Minimum meaningful quantity
                continue
            
            # Create signal
            confidence = min(0.95, abs(action_value) + action_info.get('uncertainty', 0.0))
            
            reasoning = (f"RL {action_info['agent']} policy in {action_info['regime']} regime "
                        f"(action={action_value:.3f}, uncertainty={action_info.get('uncertainty', 0.0):.3f})")
            
            signal = TradingSignal(
                symbol=symbol,
                action=action_type,
                quantity=quantity,
                confidence=confidence,
                reasoning=reasoning,
                rl_score=action_value,
                regime=action_info.get('regime', 'unknown'),
                uncertainty=action_info.get('uncertainty', 0.0)
            )
            
            signals.append(signal)
            
        return signals
    
    def record_execution_result(self, 
                              signal: TradingSignal, 
                              execution_result: Dict[str, Any]):
        """Record the result of a signal execution for learning."""
        
        execution_record = {
            'timestamp': datetime.now(),
            'signal': signal,
            'execution_result': execution_result,
            'executed': execution_result.get('success', False)
        }
        
        self.execution_history.append(execution_record)
        
        # Keep only recent history
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
    
    async def save_state(self, filepath: str = "data/online_rl_state.json"):
        """Save the RL system state."""
        
        if self.system and self.initialized:
            self.system.save_system_state(filepath)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from the RL system."""
        
        if not self.system or not self.initialized:
            return {}
        
        stats = self.system.get_training_stats()
        
        # Add execution tracking metrics
        recent_executions = [r for r in self.execution_history 
                           if r['timestamp'] > datetime.now() - timedelta(days=1)]
        
        execution_rate = len([r for r in recent_executions if r['executed']]) / max(1, len(recent_executions))
        
        return {
            **stats,
            'signals_generated_today': len([s for s in self.signal_history 
                                          if s.get('timestamp', datetime.min) > datetime.now().replace(hour=0, minute=0, second=0)]),
            'execution_rate_24h': execution_rate,
            'total_executions': len(self.execution_history)
        }
    
    def _create_trade_metrics(self, 
                             portfolio_value: float, 
                             action_info: Dict[str, Any], 
                             market_data: Dict[str, Any], 
                             portfolio_data: Dict[str, Any],
                             signals: List) -> TradeMetrics:
        """Create comprehensive trade metrics for reward calculation."""
        
        # Determine trade type from action info
        trade_type = TradeType.HOLD  # Default
        position_size = 0.0
        
        if signals:
            # Get dominant signal type
            buy_signals = [s for s in signals if s.action == 'buy']
            sell_signals = [s for s in signals if s.action == 'sell']
            short_signals = [s for s in signals if s.action == 'short']
            
            if buy_signals:
                trade_type = TradeType.MARKET_BUY  # Assume market orders for now
                position_size = sum(s.quantity for s in buy_signals)
            elif sell_signals:
                trade_type = TradeType.MARKET_SELL
                position_size = sum(s.quantity for s in sell_signals)
            elif short_signals:
                trade_type = TradeType.SHORT_SELL
                position_size = sum(s.quantity for s in short_signals)
        
        # Calculate portfolio metrics
        portfolio_value_t = self.last_portfolio_value if self.last_portfolio_value else portfolio_value
        portfolio_value_t1 = portfolio_value
        
        # Estimate trading costs (simplified)
        estimated_slippage = abs(position_size) * 0.001  # 0.1% slippage estimate
        estimated_commission = abs(position_size) * 0.005  # $0.005 per share
        
        # Calculate risk metrics
        total_positions = sum(abs(pos.get('market_value', 0)) for pos in portfolio_data.get('positions', []))
        position_concentration = abs(position_size * 100) / max(portfolio_value, 1) if portfolio_value > 0 else 0.0  # Rough estimate
        
        # Get market sentiment (if available)
        sentiment_score = 0.0
        regime_alignment = 0.0
        volatility = 0.02  # Default volatility estimate
        
        if market_data:
            # Try to extract sentiment from market data
            for symbol_data in market_data.values():
                if isinstance(symbol_data, dict):
                    sentiment_score += symbol_data.get('sentiment_score', 0.0)
                    volatility = max(volatility, symbol_data.get('volatility', 0.02))
            
            if len(market_data) > 0:
                sentiment_score /= len(market_data)  # Average sentiment
        
        # Regime alignment from action info
        if 'regime' in action_info:
            regime_info = action_info['regime']
            if regime_info == 'trending_up' and position_size > 0:
                regime_alignment = 0.3
            elif regime_info == 'trending_down' and position_size < 0:
                regime_alignment = 0.3
            elif regime_info == 'sideways':
                regime_alignment = 0.1 if abs(position_size) < 50 else -0.1
        
        # Calculate drawdown (simplified)
        drawdown = max(0.0, (self.last_portfolio_value - portfolio_value) / max(self.last_portfolio_value, 1)) if self.last_portfolio_value else 0.0
        
        return TradeMetrics(
            portfolio_value_t=portfolio_value_t,
            portfolio_value_t1=portfolio_value_t1,
            trade_type=trade_type,
            position_size=position_size,
            current_price=100.0,  # Placeholder - would need actual price data
            
            # Cost estimates
            slippage=estimated_slippage,
            commission=estimated_commission,
            borrow_fee=abs(position_size) * 0.0001 if trade_type == TradeType.SHORT_SELL else 0.0,
            
            # Risk metrics
            leverage=1.0,  # Assume no leverage for now
            volatility=volatility,
            position_concentration=position_concentration,
            drawdown=drawdown,
            
            # Market alignment
            sentiment_score=sentiment_score,
            regime_alignment=regime_alignment,
            technical_momentum=action_info.get('uncertainty', 0.0)  # Use uncertainty as momentum proxy
        )

# Global instance for integration with existing workflow
_global_rl_agent: Optional[OnlineRLAgent] = None

def initialize_rl_agent(symbols: List[str]) -> OnlineRLAgent:
    """Initialize global RL agent instance."""
    global _global_rl_agent
    
    if _global_rl_agent is None:
        _global_rl_agent = OnlineRLAgent(symbols)
    
    return _global_rl_agent

def get_rl_agent() -> Optional[OnlineRLAgent]:
    """Get the global RL agent instance."""
    return _global_rl_agent

async def generate_rl_enhanced_signals(market_data: Dict[str, Any],
                                     portfolio_data: Dict[str, Any],
                                     portfolio_value: float,
                                     symbols: List[str]) -> List[Dict[str, Any]]:
    """Generate RL-enhanced trading signals for integration with existing workflow."""
    
    # Initialize or get RL agent
    rl_agent = initialize_rl_agent(symbols)
    
    try:
        # Generate signals
        rl_signals = await rl_agent.generate_trading_signals(
            market_data, 
            portfolio_data, 
            portfolio_value
        )
        
        # Convert to format expected by existing workflow
        workflow_signals = []
        for signal in rl_signals:
            workflow_signal = {
                'symbol': signal.symbol,
                'action': signal.action,
                'quantity': signal.quantity,
                'confidence': signal.confidence,
                'reasoning': f"[RL Enhanced] {signal.reasoning}",
                'source': 'online_rl',
                'rl_score': signal.rl_score,
                'regime': signal.regime,
                'uncertainty': signal.uncertainty,
                'timestamp': datetime.now()
            }
            workflow_signals.append(workflow_signal)
        
        return workflow_signals
        
    except Exception as e:
        logger.error(f"❌ RL enhanced signal generation failed: {e}")
        return []

# Async cleanup function
async def cleanup_rl_agent():
    """Cleanup RL agent and save state."""
    global _global_rl_agent
    
    if _global_rl_agent:
        try:
            await _global_rl_agent.save_state()
            logger.info("✅ RL agent state saved successfully")
        except Exception as e:
            logger.error(f"❌ Failed to save RL agent state: {e}")

if __name__ == "__main__":
    # Test the integration
    import asyncio
    
    async def test_integration():
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
        # Create dummy market data
        market_data = {}
        for symbol in symbols:
            market_data[symbol] = {
                'price': np.random.uniform(100, 200),
                'price_change_pct': np.random.uniform(-0.05, 0.05),
                'volume': np.random.uniform(1000000, 10000000),
                'avg_volume': np.random.uniform(2000000, 5000000),
                'rsi': np.random.uniform(30, 70),
                'macd': np.random.uniform(-2, 2),
                'bb_position': np.random.uniform(0, 1),
                'sentiment_score': np.random.uniform(-1, 1),
                'news_count': np.random.randint(0, 10)
            }
        
        portfolio_data = {symbol: {'quantity': np.random.uniform(0, 100), 'market_value': np.random.uniform(0, 20000)} 
                         for symbol in symbols}
        
        portfolio_value = 100000
        
        print("🧪 Testing RL integration...")
        
        # Generate signals
        signals = await generate_rl_enhanced_signals(
            market_data, 
            portfolio_data, 
            portfolio_value, 
            symbols
        )
        
        print(f"✅ Generated {len(signals)} RL signals:")
        for signal in signals:
            print(f"  {signal['symbol']}: {signal['action']} {signal['quantity']:.2f} "
                  f"(confidence: {signal['confidence']:.2f}, rl_score: {signal['rl_score']:.3f})")
        
        # Test performance metrics
        if _global_rl_agent:
            metrics = _global_rl_agent.get_performance_metrics()
            print(f"📊 RL Performance: {metrics}")
        
        print("🧹 Cleaning up...")
        await cleanup_rl_agent()
        
        print("✅ Integration test completed!")
    
    # Run test
    asyncio.run(test_integration())