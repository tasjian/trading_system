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
        # Set as current agent for backtesting validation
        _set_current_rl_agent(_global_rl_agent)
    
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

async def integrate_hybrid_llm_rl_portfolio_system(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hybrid LLM-RL Portfolio Integration System
    
    Architecture:
    1. LLM Portfolio Manager creates diversified sector-aware allocations
    2. RL Agent fine-tunes position sizing and timing
    3. Intelligent rebalancing prevents over-trading
    
    This is the main function called by continuous_rebalancer.py.
    """
    try:
        logger.info("🚀 Starting Hybrid LLM-RL Portfolio System Integration...")
        
        # Extract required data from state
        portfolio_data = state.get("portfolio", {})
        portfolio_value = portfolio_data.get("equity", 100000)
        cash_available = portfolio_data.get("cash", 50000)
        
        # Get symbols from multiple sources in priority order
        symbols = []
        
        # 1. Use symbols from universe filter results (highest priority)
        filtered_symbols = state.get("filtered_symbols", [])
        if filtered_symbols:
            symbols.extend(filtered_symbols[:20])  # More symbols for diversification
            logger.info(f"🎯 Using {len(symbols)} symbols from universe filter")
        
        # 2. Add symbols from sentiment data
        sentiment_data = state.get("sentiment_data", {})
        if sentiment_data:
            sentiment_symbols = [sym for sym in sentiment_data.keys() if sym not in symbols][:10]
            symbols.extend(sentiment_symbols)
            logger.info(f"📊 Added {len(sentiment_symbols)} symbols from sentiment analysis")
        
        # 3. Add current portfolio positions to ensure continuity
        positions = state.get("portfolio", {}).get("positions", {})
        current_positions = list(positions.keys())
        for position_symbol in current_positions:
            if position_symbol not in symbols:
                symbols.append(position_symbol)
        
        # Remove duplicates and limit total for diversification
        symbols = list(dict.fromkeys(symbols))[:25]  # Max 25 symbols for proper diversification
        
        # 4. Only fallback to defaults if no symbols found from any source
        if not symbols:
            logger.warning("⚠️ No symbols from universe filter or sentiment data, using diversified fallback")
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'IWM', 'XLF', 'XLK', 'SPY', 'QQQ']
        
        logger.info(f"🎯 Processing {len(symbols)} symbols for diversified portfolio (${portfolio_value:,.2f})")
        
        # STEP 1: Generate LLM Portfolio Recommendations
        logger.info("🧠 Step 1: LLM Portfolio Manager - Creating Diversified Allocations")
        
        # Import LLM portfolio manager
        try:
            from agents.llm_portfolio_management import construct_llm_portfolio
            
            # Determine risk profile based on market conditions and portfolio state
            circuit_breakers = state.get("circuit_breakers", {})
            risk_profile = "conservative" if any(circuit_breakers.values()) else "moderate"
            
            # Use LLM portfolio manager to create diversified allocation
            portfolio_recommendation = await construct_llm_portfolio(
                candidate_symbols=symbols,
                portfolio_value=portfolio_value,
                risk_profile=risk_profile,
                max_positions=min(20, len(symbols)),  # Proper diversification limit
                sentiment_data=sentiment_data  # Pass sentiment data for enhanced analysis
            )
            
            logger.info(f"🎯 LLM Portfolio Manager Results:")
            logger.info(f"   Market Regime: {portfolio_recommendation.market_regime.value}")
            logger.info(f"   Diversification Score: {portfolio_recommendation.diversification_score:.2f}")
            logger.info(f"   Expected Return: {portfolio_recommendation.expected_return:.2%}")
            logger.info(f"   Risk Score: {portfolio_recommendation.total_risk_score:.2f}")
            logger.info(f"   Portfolio Positions: {len(portfolio_recommendation.allocations)}")
            
            llm_allocations = portfolio_recommendation.allocations
            
        except Exception as e:
            logger.error(f"❌ LLM Portfolio Manager failed: {e}")
            # Fallback to basic diversified allocation
            llm_allocations = []
            from dataclasses import dataclass
            @dataclass
            class StockAllocation:
                symbol: str
                target_weight: float
                confidence: float
                recommended_action: str
                reasoning: str
                sentiment_score: float
                industry: str
                risk_level: str
            
            # Create basic diversified allocation
            base_weight = 0.05  # 5% per position
            for i, symbol in enumerate(symbols[:10]):  # Max 10 positions for fallback
                llm_allocations.append(StockAllocation(
                    symbol=symbol,
                    target_weight=base_weight,
                    confidence=0.6,
                    recommended_action="buy",
                    reasoning="Basic diversification fallback",
                    sentiment_score=0.1,
                    industry="Unknown",
                    risk_level="medium"
                ))
            
            logger.info(f"🔄 Using basic diversification fallback: {len(llm_allocations)} positions")
        
        # STEP 2: RL Enhancement and Position Optimization
        logger.info("🤖 Step 2: RL Agent - Optimizing Position Sizing and Timing")
        
        # Initialize RL agent with LLM-selected symbols
        llm_symbols = [alloc.symbol for alloc in llm_allocations]
        rl_agent = initialize_rl_agent(llm_symbols)
        
        # Prepare enhanced market data for RL agent
        market_data = {}
        sentiment_signals = state.get("sentiment_signals", [])
        
        # Get real market data where possible
        try:
            from tools.alpaca_client import alpaca_client
            real_market_data = True
        except Exception:
            real_market_data = False
            logger.warning("Using synthetic market data for RL processing")
        
        for allocation in llm_allocations:
            symbol = allocation.symbol
            
            # Find sentiment data for this symbol
            symbol_sentiment = next((s for s in sentiment_signals if s.get('symbol') == symbol), {})
            
            # Get real price if available
            current_price = 100.0  # Default
            if real_market_data:
                try:
                    real_price = alpaca_client.get_current_price(symbol)
                    if real_price and real_price > 0:
                        current_price = real_price
                except Exception:
                    pass
            
            market_data[symbol] = {
                'price': current_price,
                'price_change_pct': symbol_sentiment.get('strength', 0.01) if symbol_sentiment.get('signal') == 'BUY' else -symbol_sentiment.get('strength', 0.01),
                'volume': 1000000,
                'avg_volume': 1000000,
                'rsi': 50.0 + (allocation.sentiment_score * 20),  # Adjust RSI based on sentiment
                'macd': allocation.sentiment_score * 0.1,
                'bb_position': 0.5,
                'sentiment_score': allocation.sentiment_score,
                'sentiment_confidence': allocation.confidence,
                'social_sentiment_source': symbol_sentiment.get('source', 'llm_enhanced'),
                'news_count': 5,
                'llm_target_weight': allocation.target_weight,
                'llm_confidence': allocation.confidence,
                'industry': allocation.industry,
                'risk_level': allocation.risk_level
            }
        
        # Generate RL-enhanced signals for position optimization
        current_portfolio_data = {}
        for symbol in llm_symbols:
            position_info = positions.get(symbol, {})
            current_portfolio_data[symbol] = {
                'quantity': position_info.get('qty', 0),
                'market_value': position_info.get('market_value', 0)
            }
        
        rl_signals = await rl_agent.generate_trading_signals(
            market_data,
            current_portfolio_data,
            portfolio_value
        )
        
        # STEP 3: Combine LLM Allocations with RL Optimizations
        logger.info("🔄 Step 3: Hybrid Integration - Combining LLM Strategy with RL Optimization")
        
        final_allocations = []
        total_target_allocation = 0.0
        
        for allocation in llm_allocations:
            symbol = allocation.symbol
            
            # Find corresponding RL signal for this symbol
            rl_signal = next((s for s in rl_signals if s.symbol == symbol), None)
            
            # Base weight from LLM (diversification-focused)
            base_weight = allocation.target_weight
            
            # RL adjustment factor (timing and market dynamics)
            rl_adjustment = 1.0
            rl_confidence_boost = 0.0
            
            if rl_signal:
                # Adjust based on RL signal strength and direction
                if rl_signal.action == 'buy' and allocation.recommended_action == 'buy':
                    # Both systems agree - increase confidence
                    rl_adjustment = min(1.5, 1.0 + abs(rl_signal.rl_score))
                    rl_confidence_boost = 0.1
                elif rl_signal.action == 'sell' and allocation.recommended_action == 'buy':
                    # Systems disagree - reduce allocation
                    rl_adjustment = max(0.3, 1.0 - abs(rl_signal.rl_score))
                    rl_confidence_boost = -0.2
            
            # Apply position size limits
            final_weight = min(0.08, base_weight * rl_adjustment)  # Max 8% per position
            
            # Only include positions with meaningful allocations
            if final_weight >= 0.01:  # At least 1% allocation
                final_allocations.append({
                    'symbol': symbol,
                    'weight': final_weight,
                    'percentage': final_weight * 100,  # For compatibility
                    'confidence': min(0.95, allocation.confidence + rl_confidence_boost),
                    'action': allocation.recommended_action,
                    'reasoning': f"LLM: {allocation.reasoning} | RL: {rl_signal.reasoning if rl_signal else 'No RL signal'}",
                    'industry': allocation.industry,
                    'risk_level': allocation.risk_level,
                    'llm_weight': base_weight,
                    'rl_adjustment': rl_adjustment,
                    'rl_score': rl_signal.rl_score if rl_signal else 0.0,
                    'sentiment_score': allocation.sentiment_score
                })
                
                total_target_allocation += final_weight
        
        # STEP 4: Intelligent Rebalancing Logic with Sector Constraints
        logger.info("⚖️ Step 4: Intelligent Rebalancing - Preventing Over-Trading")
        
        # Calculate rebalancing thresholds
        rebalancing_threshold = 0.05  # 5% deviation triggers rebalancing
        min_trade_size = portfolio_value * 0.005  # Minimum $500 trade size
        max_single_position = 0.08  # Max 8% per position
        max_sector_allocation = 0.25  # Max 25% per sector
        
        # Apply sector concentration limits
        sector_allocations = {}
        for allocation in final_allocations:
            sector = allocation.get('industry', 'Unknown')
            if sector not in sector_allocations:
                sector_allocations[sector] = 0.0
            sector_allocations[sector] += allocation['weight']
        
        # Check and enforce sector limits
        sector_violations = []
        for sector, total_weight in sector_allocations.items():
            if total_weight > max_sector_allocation:
                sector_violations.append(f"{sector}: {total_weight:.1%} > {max_sector_allocation:.1%}")
        
        if sector_violations:
            logger.warning(f"🚨 Sector concentration violations detected: {', '.join(sector_violations)}")
            
            # Scale down over-allocated sectors
            for allocation in final_allocations:
                sector = allocation.get('industry', 'Unknown')
                if sector_allocations[sector] > max_sector_allocation:
                    scale_factor = max_sector_allocation / sector_allocations[sector]
                    allocation['weight'] *= scale_factor
                    logger.info(f"🔧 Scaled down {allocation['symbol']} from {allocation['weight']/scale_factor:.1%} to {allocation['weight']:.1%}")
        
        # Apply individual position size limits
        for allocation in final_allocations:
            if allocation['weight'] > max_single_position:
                logger.warning(f"🚨 Position size limit: {allocation['symbol']} {allocation['weight']:.1%} > {max_single_position:.1%}")
                allocation['weight'] = max_single_position
        
        # Check if significant rebalancing is needed
        needs_rebalancing = False
        rebalance_reasons = []
        
        for allocation in final_allocations:
            symbol = allocation['symbol']
            target_value = allocation['weight'] * portfolio_value
            current_position = positions.get(symbol, {})
            current_value = current_position.get('market_value', 0)
            
            deviation = abs(target_value - current_value) / max(portfolio_value, 1)
            
            if deviation > rebalancing_threshold and abs(target_value - current_value) > min_trade_size:
                needs_rebalancing = True
                rebalance_reasons.append(f"{symbol}: {deviation:.1%} deviation (${target_value - current_value:,.0f})")
        
        # Check for new opportunities (symbols not in current portfolio)
        current_symbols = set(positions.keys())
        target_symbols = set(alloc['symbol'] for alloc in final_allocations)
        new_opportunities = target_symbols - current_symbols
        
        if new_opportunities and len(new_opportunities) >= 2:  # Only if multiple new opportunities
            needs_rebalancing = True
            rebalance_reasons.append(f"New opportunities: {', '.join(list(new_opportunities)[:3])}")
        
        # Check for overweight positions that need trimming
        overweight_positions = []
        for symbol, position_info in positions.items():
            current_value = position_info.get('market_value', 0)
            current_weight = current_value / max(portfolio_value, 1)
            if current_weight > max_single_position:
                overweight_positions.append(f"{symbol}: {current_weight:.1%} > {max_single_position:.1%}")
        
        if overweight_positions:
            needs_rebalancing = True
            rebalance_reasons.extend(overweight_positions[:2])  # Add top 2 overweight positions
        
        # Only proceed with rebalancing if significant changes are needed
        if not needs_rebalancing and len(current_positions) > 5:  # Allow rebalancing if portfolio too small
            logger.info("🔒 No significant rebalancing needed - maintaining current positions")
            return {
                'rl_decisions': {
                    'strategy': 'maintain_positions',
                    'risk_level': risk_profile,
                    'allocations': [],  # No new trades
                    'confidence': 0.7,
                    'reasoning': 'Portfolio is well-balanced, no significant rebalancing required',
                    'rebalance_analysis': {
                        'needs_rebalancing': False,
                        'current_positions': len(current_positions),
                        'target_positions': len(final_allocations),
                        'max_deviation': max([abs(alloc['weight'] * portfolio_value - positions.get(alloc['symbol'], {}).get('market_value', 0)) / portfolio_value for alloc in final_allocations], default=0),
                        'sector_compliance': len(sector_violations) == 0,
                        'position_size_compliance': all(alloc['weight'] <= max_single_position for alloc in final_allocations)
                    }
                },
                'rl_enhanced': True,
                'comprehensive_rl': True,
                'llm_enhanced': True,
                'hybrid_system': True
            }
        
        # STEP 5: Generate Final Trading Decisions
        logger.info(f"📈 Step 5: Generating Trading Decisions - {len(rebalance_reasons)} rebalancing reasons")
        for reason in rebalance_reasons[:3]:  # Log top 3 reasons
            logger.info(f"   🎯 {reason}")
        
        # Normalize final allocations to ensure they don't exceed safe limits
        if total_target_allocation > 0.7:  # Max 70% of portfolio in stocks
            scale_factor = 0.7 / total_target_allocation
            for allocation in final_allocations:
                allocation['weight'] *= scale_factor
                allocation['percentage'] *= scale_factor
        
        # Create final RL decisions structure
        rl_decisions = {
            'strategy': 'hybrid_llm_rl_diversified',
            'risk_level': risk_profile,
            'allocations': final_allocations,
            'confidence': np.mean([alloc['confidence'] for alloc in final_allocations]) if final_allocations else 0.0,
            'active_agent': 'hybrid_llm_rl',
            'total_allocation': sum(alloc['weight'] for alloc in final_allocations),
            'reasoning': f'Hybrid LLM-RL portfolio: {len(final_allocations)} diversified positions across sectors',
            'rebalance_analysis': {
                'needs_rebalancing': True,
                'rebalance_reasons': rebalance_reasons,
                'threshold_exceeded': True,
                'new_positions': len(new_opportunities),
                'total_deviation': sum([abs(alloc['weight'] * portfolio_value - positions.get(alloc['symbol'], {}).get('market_value', 0)) for alloc in final_allocations])
            },
            'portfolio_analytics': {
                'market_regime': getattr(portfolio_recommendation, 'market_regime', 'unknown'),
                'diversification_score': getattr(portfolio_recommendation, 'diversification_score', 0.5),
                'expected_return': getattr(portfolio_recommendation, 'expected_return', 0.1),
                'sector_count': len(set(alloc.get('industry', 'Unknown') for alloc in final_allocations)),
                'risk_distribution': {
                    'low': len([a for a in final_allocations if a.get('risk_level') == 'low']),
                    'medium': len([a for a in final_allocations if a.get('risk_level') == 'medium']),
                    'high': len([a for a in final_allocations if a.get('risk_level') == 'high'])
                }
            }
        }
        
        logger.info(f"✅ Hybrid LLM-RL System Results:")
        logger.info(f"   🎯 Strategy: {rl_decisions['strategy']}")
        logger.info(f"   📊 Positions: {len(final_allocations)} diversified allocations")
        logger.info(f"   📈 Total Allocation: {rl_decisions['total_allocation']:.1%}")
        logger.info(f"   🎲 Confidence: {rl_decisions['confidence']:.1%}")
        logger.info(f"   🏭 Sectors: {rl_decisions['portfolio_analytics']['sector_count']}")
        
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
            'llm_portfolio_recommendation': getattr(portfolio_recommendation, '__dict__', {}),
            'rl_enhanced': True,
            'comprehensive_rl': True,
            'llm_enhanced': True,
            'hybrid_system': True
        }
        
    except Exception as e:
        logger.error(f"❌ Hybrid LLM-RL system integration failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'rl_decisions': {'allocations': [], 'strategy': 'error', 'reasoning': f'Hybrid system failed: {e}'},
            'rl_enhanced': False,
            'comprehensive_rl': False,
            'llm_enhanced': False,
            'hybrid_system': False
        }

# Backward compatibility - redirect the old function name to the new hybrid system
async def integrate_comprehensive_rl_system(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Backward compatibility wrapper for the hybrid LLM-RL system."""
    return await integrate_hybrid_llm_rl_portfolio_system(state, config)

async def cleanup_rl_agent():
    """Cleanup RL agent and save state."""
    global _global_rl_agent
    
    if _global_rl_agent:
        try:
            await _global_rl_agent.save_state()
            logger.info("✅ RL agent state saved successfully")
        except Exception as e:
            logger.error(f"❌ Failed to save RL agent state: {e}")

# Global RL agent instance for backtesting validation
_current_rl_agent = None

def get_current_rl_agent():
    """Get the current RL agent instance for backtesting validation."""
    global _current_rl_agent
    return _current_rl_agent

def _set_current_rl_agent(agent):
    """Set the current RL agent instance (internal use)."""
    global _current_rl_agent
    _current_rl_agent = agent

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
    'get_current_rl_agent',
    'generate_rl_enhanced_signals',
    'integrate_comprehensive_rl_system',
    'cleanup_rl_agent',
    'create_rl_bridge'
]

logger.info("✅ Simplified RL Integration Bridge loaded - all functionality consolidated")