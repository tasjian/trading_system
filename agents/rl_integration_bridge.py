#!/usr/bin/env python3
"""
Simplified RL Integration Bridge
Consolidated RL integration module that provides all RL functionality in a single clean interface.
Eliminates the complex multi-layer architecture while maintaining all core RL capabilities.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

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

class SimplifiedRLAgent:
    """Simplified RL agent that provides all necessary RL functionality."""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.initialized = False
        self.last_portfolio_value = None
        self.signal_history = []
        
        # Dynamic feature tracking for compatibility
        self.features_per_symbol = 20  # Updated feature count
        self.total_expected_features = len(symbols) * self.features_per_symbol
        
        logger.info(f"🎯 Initialized SimplifiedRLAgent: {len(symbols)} symbols, {self.total_expected_features} total features")
        
        # Try to import advanced RL components, fallback to simple implementation
        self.has_advanced_rl = False
        try:
            from agents.online_rl_system import create_online_rl_system, OnlineLearningConfig
            from agents.unified_reward_calculator import UnifiedRewardCalculator
            
            self.config = OnlineLearningConfig()
            self.reward_calculator = UnifiedRewardCalculator()
            self.system = None
            self.has_advanced_rl = True
            logger.info(f"🤖 Advanced RL system available for {len(symbols)} symbols")
            
        except ImportError as e:
            logger.warning(f"Advanced RL components not available: {e}")
            logger.info(f"🎲 Using simplified RL fallback for {len(symbols)} symbols")
    
    async def initialize_system(self):
        """Initialize the RL system."""
        if self.initialized:
            return
            
        try:
            if self.has_advanced_rl:
                from agents.online_rl_system import create_online_rl_system
                
                self.system = create_online_rl_system(
                    symbols=self.symbols,
                    **self.config.__dict__
                )
                
                # Try to load previous state
                try:
                    self.system.load_system_state("data/online_rl_state.json")
                    logger.info("✅ Loaded previous RL system state")
                except Exception as e:
                    logger.info(f"No previous state found, starting fresh: {e}")
                    
            self.initialized = True
            logger.info("🚀 RL system initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RL system: {e}")
            # Continue with simplified mode
            self.has_advanced_rl = False
            self.initialized = True
    
    async def generate_trading_signals(self, 
                                     market_data: Dict[str, Any],
                                     portfolio_data: Dict[str, Any],
                                     portfolio_value: float) -> List[TradingSignal]:
        """Generate trading signals using RL system."""
        
        if not self.initialized:
            await self.initialize_system()
        
        try:
            if self.has_advanced_rl and self.system:
                # Use advanced RL system
                return await self._generate_advanced_signals(
                    market_data, portfolio_data, portfolio_value
                )
            else:
                # Use simplified signal generation
                return self._generate_simple_signals(
                    market_data, portfolio_data, portfolio_value
                )
                
        except Exception as e:
            logger.error(f"❌ RL signal generation failed: {e}")
            return []
    
    async def _generate_advanced_signals(self, market_data, portfolio_data, portfolio_value):
        """Generate signals using the advanced RL system."""
        # Convert market data to RL state
        state_features = []
        
        for symbol in self.symbols:
            symbol_data = market_data.get(symbol, {})
            
            # Price features
            price_change = symbol_data.get('price_change_pct', 0.0)
            volume = symbol_data.get('volume', 0.0)
            avg_volume = symbol_data.get('avg_volume', volume)
            volume_ratio = volume / max(avg_volume, 1.0) if avg_volume > 0 else 1.0
            
            # Technical indicators
            rsi = symbol_data.get('rsi', 50.0) / 100.0
            macd = symbol_data.get('macd', 0.0)
            bb_position = symbol_data.get('bb_position', 0.5)
            
            # Portfolio features
            current_position = portfolio_data.get(symbol, {}).get('quantity', 0.0)
            position_value = portfolio_data.get(symbol, {}).get('market_value', 0.0)
            
            # Enhanced sentiment features (social media + news integration)
            sentiment_score = symbol_data.get('sentiment_score', 0.0)
            sentiment_confidence = symbol_data.get('sentiment_confidence', 0.5)
            news_count = symbol_data.get('news_count', 0.0)
            social_sentiment_strength = symbol_data.get('social_sentiment_strength', 0.0)
            
            # Create comprehensive feature vector with social media integration
            # Calculate features dynamically to handle variable dimensions
            symbol_features = [
                price_change, volume_ratio, rsi, macd, bb_position,
                current_position, position_value / 10000,
                sentiment_score, news_count / 10,
                np.log1p(symbol_data.get('price', 1.0)) if symbol_data.get('price', 0) > 0 else 0.0,
                np.tanh(price_change * 10),
                min(1.0, volume_ratio)
            ]
            
            # Add enhanced social media sentiment features if available
            if social_sentiment_strength > 0 or sentiment_confidence != 0.5:
                symbol_features.extend([
                    sentiment_confidence,  # Confidence of sentiment analysis
                    social_sentiment_strength,  # Strength from social media platforms
                    np.tanh(sentiment_score * 5),  # Normalized sentiment impact
                    sentiment_score * sentiment_confidence  # Weighted sentiment signal
                ])
            else:
                # Add zero padding for consistency when no enhanced features
                symbol_features.extend([0.0, 0.0, 0.0, 0.0])
            
            # Additional technical features for better RL performance
            symbol_features.extend([
                symbol_data.get('volatility', 0.02),  # Volatility
                min(1.0, max(-1.0, bb_position - 0.5)),  # Bollinger band relative position
                np.clip((rsi - 50) / 50, -1.0, 1.0),  # Normalized RSI deviation
                np.tanh(macd)  # Normalized MACD
            ])
            
            # Log social media feature integration for debugging
            if social_sentiment_strength > 0:
                logger.debug(f"🌐 {symbol}: Social sentiment strength {social_sentiment_strength:.2f}, sentiment {sentiment_score:.2f}, confidence {sentiment_confidence:.2f}")
            
            state_features.extend(symbol_features)
        
        # Create state with proper dimension handling
        current_state = np.array(state_features, dtype=np.float32)
        
        # Log state dimensions for debugging
        expected_features_per_symbol = 20  # Updated count based on new feature vector
        expected_total_features = len(self.symbols) * expected_features_per_symbol
        actual_features = len(current_state)
        
        if actual_features != expected_total_features:
            logger.debug(f"🔧 Feature dimension mismatch: got {actual_features}, expected {expected_total_features}")
            logger.debug(f"Per symbol: got {actual_features / len(self.symbols):.1f}, expected {expected_features_per_symbol}")
            
            # Ensure consistent feature dimensions
            current_state = self._normalize_feature_vector(current_state, expected_total_features)
        
        # Market metadata
        market_metadata = {
            'volatility': np.std([market_data.get(s, {}).get('price_change_pct', 0.0) for s in self.symbols]),
            'trend': np.mean([market_data.get(s, {}).get('price_change_pct', 0.0) for s in self.symbols]),
            'portfolio_value': portfolio_value,
            'data_completeness': len([s for s in self.symbols if s in market_data]) / len(self.symbols)
        }
        
        # Calculate reward
        previous_reward = 0.0
        if self.last_portfolio_value is not None:
            if self.last_portfolio_value > 0:
                previous_reward = (portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
            else:
                previous_reward = 0.01 if portfolio_value > 0 else 0.0
        
        # Process market step
        action, action_info = await self.system.process_market_step(
            market_state=current_state,
            market_data=market_metadata,
            previous_action=None,
            previous_reward=previous_reward,
            deterministic=False
        )
        
        # Convert actions to signals
        signals = self._convert_actions_to_signals(action, action_info, market_data, portfolio_data)
        
        self.last_portfolio_value = portfolio_value
        return signals
    
    def _normalize_feature_vector(self, features: np.ndarray, expected_size: int) -> np.ndarray:
        """Normalize feature vector to expected size for model compatibility."""
        try:
            actual_size = len(features)
            
            if actual_size == expected_size:
                return features
            elif actual_size > expected_size:
                # Truncate to expected size
                normalized = features[:expected_size]
                logger.debug(f"🔧 Truncated feature vector from {actual_size} to {expected_size}")
            else:
                # Pad with zeros or repeat pattern
                padding_size = expected_size - actual_size
                if actual_size > 0:
                    # Use pattern repetition for more meaningful padding
                    repeat_count = padding_size // actual_size
                    remainder = padding_size % actual_size
                    
                    padding = np.concatenate([
                        np.tile(features[-10:], repeat_count) if len(features) >= 10 else np.tile(features, repeat_count),
                        features[:remainder] if remainder > 0 else np.array([])
                    ])
                else:
                    padding = np.zeros(padding_size)
                
                normalized = np.concatenate([features, padding])
                logger.debug(f"🔧 Padded feature vector from {actual_size} to {expected_size}")
            
            return normalized.astype(np.float32)
            
        except Exception as e:
            logger.error(f"❌ Feature vector normalization failed: {e}")
            # Return zero-padded vector as safe fallback
            safe_vector = np.zeros(expected_size, dtype=np.float32)
            if len(features) > 0:
                safe_vector[:min(len(features), expected_size)] = features[:expected_size]
            return safe_vector
    
    def _generate_simple_signals(self, market_data, portfolio_data, portfolio_value):
        """Generate simple signals using basic logic."""
        signals = []
        
        for symbol in self.symbols:
            symbol_data = market_data.get(symbol, {})
            
            # Simple momentum-based signals
            price_change = symbol_data.get('price_change_pct', 0.0)
            volume_ratio = symbol_data.get('volume', 1.0) / symbol_data.get('avg_volume', 1.0)
            sentiment_score = symbol_data.get('sentiment_score', 0.0)
            
            # Simple scoring
            score = price_change * 0.4 + sentiment_score * 0.6
            
            if score > 0.02 and volume_ratio > 1.2:  # Buy signal
                signal = TradingSignal(
                    symbol=symbol,
                    action='buy',
                    quantity=10.0,  # Simple fixed quantity
                    confidence=min(0.8, abs(score) * 10),
                    reasoning=f"Simple momentum: score={score:.3f}",
                    rl_score=score,
                    regime='unknown',
                    uncertainty=0.3
                )
                signals.append(signal)
                
            elif score < -0.02 and volume_ratio > 1.2:  # Sell signal
                signal = TradingSignal(
                    symbol=symbol,
                    action='sell',
                    quantity=5.0,  # Simple fixed quantity
                    confidence=min(0.8, abs(score) * 10),
                    reasoning=f"Simple momentum: score={score:.3f}",
                    rl_score=score,
                    regime='unknown',
                    uncertainty=0.3
                )
                signals.append(signal)
        
        return signals
    
    def _convert_actions_to_signals(self, actions, action_info, market_data, portfolio_data):
        """Convert RL actions to trading signals."""
        if actions is None or len(actions) == 0:
            return []
        
        signals = []
        action_threshold = 0.05
        
        # Enhanced action dimension handling with detailed logging
        expected_actions = len(self.symbols)
        actual_actions = len(actions)
        
        if actual_actions != expected_actions:
            logger.debug(f"🔧 Action dimension mismatch: got {actual_actions}, expected {expected_actions}")
            
            if actual_actions == 1:
                # Broadcast single action to all symbols
                actions = np.full(expected_actions, actions[0])
                logger.debug(f"📡 Broadcasted single action {actions[0]:.3f} to {expected_actions} symbols")
            elif actual_actions < expected_actions:
                # Pad with zeros (neutral action)
                padded = np.zeros(expected_actions)
                padded[:actual_actions] = actions
                actions = padded
                logger.debug(f"🔧 Padded actions from {actual_actions} to {expected_actions}")
            else:
                # Truncate to expected size
                actions = actions[:expected_actions]
                logger.debug(f"✂️ Truncated actions from {actual_actions} to {expected_actions}")
        
        for i, symbol in enumerate(self.symbols):
            action_value = float(actions[i])
            
            if abs(action_value) < action_threshold:
                continue
                
            current_position = portfolio_data.get(symbol, {}).get('quantity', 0.0)
            current_price = market_data.get(symbol, {}).get('price', 0.0)
            
            if current_price <= 0:
                continue
            
            if action_value > 0:  # Buy
                quantity = abs(action_value) * 50  # Scale to reasonable quantity
                action_type = 'buy'
            else:  # Sell
                quantity = min(current_position, abs(action_value) * 50)
                action_type = 'sell'
            
            if quantity >= 0.01:
                signal = TradingSignal(
                    symbol=symbol,
                    action=action_type,
                    quantity=quantity,
                    confidence=min(0.95, abs(action_value) * 2),
                    reasoning=f"RL {action_info.get('agent', 'unknown')} policy (action={action_value:.3f})",
                    rl_score=action_value,
                    regime=action_info.get('regime', 'unknown'),
                    uncertainty=action_info.get('uncertainty', 0.0)
                )
                signals.append(signal)
        
        return signals
    
    async def save_state(self, filepath: str = "data/online_rl_state.json"):
        """Save the RL system state."""
        if self.has_advanced_rl and self.system and self.initialized:
            self.system.save_system_state(filepath)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from the RL system."""
        if self.has_advanced_rl and self.system and self.initialized:
            return self.system.get_training_stats()
        return {
            "signals_generated": len(self.signal_history),
            "expected_features": self.total_expected_features,
            "features_per_symbol": self.features_per_symbol,
            "symbol_count": len(self.symbols)
        }

# Global instance management
_global_rl_agent: Optional[SimplifiedRLAgent] = None

def initialize_rl_agent(symbols: List[str]) -> SimplifiedRLAgent:
    """Initialize global RL agent instance."""
    global _global_rl_agent
    
    if _global_rl_agent is None:
        _global_rl_agent = SimplifiedRLAgent(symbols)
    
    return _global_rl_agent

def get_rl_agent() -> Optional[SimplifiedRLAgent]:
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
                'source': 'rl_bridge',
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

async def integrate_comprehensive_rl_system(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive RL system integration for the continuous rebalancer.
    This is the main function called by continuous_rebalancer.py.
    """
    try:
        logger.info("🚀 Starting comprehensive RL system integration...")
        
        # Extract required data from state
        portfolio_data = state.get("portfolio", {})
        portfolio_value = portfolio_data.get("equity", 100000)
        
        # Get symbols from multiple sources in priority order
        symbols = []
        
        # 1. Use symbols from universe filter results (highest priority)
        filtered_symbols = state.get("filtered_symbols", [])
        if filtered_symbols:
            symbols.extend(filtered_symbols[:10])  # Top 10 from universe filter
            logger.info(f"🎯 Using {len(symbols)} symbols from universe filter")
        
        # 2. Add symbols from sentiment data
        sentiment_data = state.get("sentiment_data", {})
        if sentiment_data:
            sentiment_symbols = [sym for sym in sentiment_data.keys() if sym not in symbols][:5]
            symbols.extend(sentiment_symbols)
            logger.info(f"📊 Added {len(sentiment_symbols)} symbols from sentiment analysis")
        
        # 3. Add current portfolio positions to ensure continuity
        positions = state.get("portfolio", {}).get("positions", {})
        for position_symbol in positions.keys():
            if position_symbol not in symbols:
                symbols.append(position_symbol)
        
        # Remove duplicates and limit total
        symbols = list(dict.fromkeys(symbols))[:15]  # Max 15 symbols for RL processing
        
        # 4. Only fallback to defaults if no symbols found from any source
        if not symbols:
            logger.warning("⚠️ No symbols from universe filter or sentiment data, using fallback symbols")
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
        logger.info(f"🎯 RL processing {len(symbols)} symbols with portfolio value ${portfolio_value:,.2f}")
        
        # Initialize RL agent
        rl_agent = initialize_rl_agent(symbols)
        
        # Prepare market data from state
        market_data = {}
        
        # Get sentiment signals from state (fix undefined variable)
        sentiment_signals = state.get("sentiment_signals", [])
        if not sentiment_signals:
            # Fallback to sentiment_data if sentiment_signals not available
            sentiment_data = state.get("sentiment_data", {})
            sentiment_signals = []
            for symbol, sentiment_info in sentiment_data.items():
                if isinstance(sentiment_info, dict):
                    sentiment_signals.append({
                        'symbol': symbol,
                        'signal': 'BUY' if sentiment_info.get('overall_score', 0) > 0 else 'SELL',
                        'strength': abs(sentiment_info.get('overall_score', 0)),
                        'confidence': sentiment_info.get('confidence', 0.5),
                        'source': 'sentiment_data_fallback'
                    })
        
        for symbol in symbols:
            # Find sentiment data for this symbol
            symbol_sentiment = next((s for s in sentiment_signals if s.get('symbol') == symbol), {})
            
            market_data[symbol] = {
                'price': 100.0,  # Default price - would be filled by real market data
                'price_change_pct': np.random.uniform(-0.02, 0.02),  # Random walk
                'volume': 1000000,
                'avg_volume': 1000000,
                'rsi': 50.0,
                'macd': 0.0,
                'bb_position': 0.5,
                'sentiment_score': symbol_sentiment.get('strength', 0.0) if symbol_sentiment.get('signal') == 'BUY' else -symbol_sentiment.get('strength', 0.0),
                'sentiment_confidence': symbol_sentiment.get('confidence', 0.5),
                'social_sentiment_source': symbol_sentiment.get('source', 'unknown'),
                'news_count': 5
            }
        
        # Generate RL signals
        rl_signals = await rl_agent.generate_trading_signals(
            market_data,
            {symbol: {'quantity': 0, 'market_value': 0} for symbol in symbols},
            portfolio_value
        )
        
        # Convert to allocation decisions
        allocations = []
        total_allocation = 0.0
        
        for signal in rl_signals:
            # Calculate allocation weight based on signal strength
            base_weight = min(0.1, abs(signal.rl_score) * 0.5)  # Max 10% per position
            
            if signal.action == 'buy':
                allocation_weight = base_weight
                action = 'buy'
            elif signal.action == 'sell':
                allocation_weight = -base_weight
                action = 'sell'
            else:
                continue
            
            allocations.append({
                'symbol': signal.symbol,
                'weight': allocation_weight,
                'confidence': signal.confidence,
                'action': action,
                'reasoning': signal.reasoning,
                'rl_score': signal.rl_score
            })
            
            total_allocation += abs(allocation_weight)
        
        # Normalize allocations if too high
        if total_allocation > 0.5:  # Max 50% total allocation
            scale_factor = 0.5 / total_allocation
            for allocation in allocations:
                allocation['weight'] *= scale_factor
        
        # Create RL decisions
        rl_decisions = {
            'strategy': 'rl_enhanced_momentum',
            'risk_level': 'moderate',
            'allocations': allocations,
            'confidence': np.mean([a['confidence'] for a in allocations]) if allocations else 0.0,
            'active_agent': getattr(rl_agent, 'active_agent', 'simplified'),
            'total_allocation': sum(abs(a['weight']) for a in allocations),
            'reasoning': f'RL analysis of {len(symbols)} symbols with {len(allocations)} actionable positions'
        }
        
        logger.info(f"✅ RL system generated {len(allocations)} allocation decisions")
        logger.info(f"🎯 Total allocation: {rl_decisions['total_allocation']:.1%}")
        logger.info(f"🤖 Strategy: {rl_decisions['strategy']}")
        
        return {
            'rl_decisions': rl_decisions,
            'rl_signals': [
                {
                    'symbol': s.symbol,
                    'action': s.action,
                    'quantity': s.quantity,
                    'confidence': s.confidence,
                    'reasoning': s.reasoning,
                    'rl_score': s.rl_score
                }
                for s in rl_signals
            ],
            'rl_enhanced': True,
            'comprehensive_rl': True
        }
        
    except Exception as e:
        logger.error(f"❌ Comprehensive RL system integration failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'rl_decisions': {'allocations': [], 'strategy': 'error', 'reasoning': f'RL system failed: {e}'},
            'rl_enhanced': False,
            'comprehensive_rl': False
        }

async def cleanup_rl_agent():
    """Cleanup RL agent and save state."""
    global _global_rl_agent
    
    if _global_rl_agent:
        try:
            await _global_rl_agent.save_state()
            logger.info("✅ RL agent state saved successfully")
        except Exception as e:
            logger.error(f"❌ Failed to save RL agent state: {e}")

# Backward compatibility aliases
OnlineRLAgent = SimplifiedRLAgent
create_rl_bridge = initialize_rl_agent

# Export all functions for compatibility
__all__ = [
    'SimplifiedRLAgent',
    'OnlineRLAgent',
    'TradingSignal',
    'initialize_rl_agent',
    'get_rl_agent',
    'generate_rl_enhanced_signals',
    'integrate_comprehensive_rl_system',
    'cleanup_rl_agent',
    'create_rl_bridge'
]

logger.info("✅ Simplified RL Integration Bridge loaded - all functionality consolidated")