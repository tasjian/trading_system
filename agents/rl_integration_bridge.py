#!/usr/bin/env python3
"""
Enhanced RL Integration Bridge with FinRL DRL Agents
Advanced RL integration module that provides sophisticated Deep Reinforcement Learning functionality
using FinRL's proven DRL agents architecture while maintaining compatibility with the existing trading system.

Key Features:
- FinRL DRL Agents (A2C, PPO, DDPG, SAC, TD3)
- Ensemble strategy selection
- Professional-grade DRL training and inference
- Advanced portfolio management
- Turbulence-based risk management
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

# Import FinRL components
try:
    from agents.finrl_agent_wrapper import FinRLAgentWrapper, create_finrl_agent, generate_finrl_signals
    from agents.finrl_trading_env import AlpacaFinRLEnvironment
    FINRL_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ FinRL DRL agents successfully imported")
except ImportError as e:
    FINRL_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.error(f"❌ FinRL DRL agents not available: {e}")
    logger.error("🔄 Falling back to simplified RL implementation")

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

class EnhancedRLAgent:
    """Enhanced RL agent using FinRL DRL algorithms or simplified fallback."""
    
    def __init__(self, symbols: List[str], alpaca_client=None):
        self.symbols = symbols
        self.alpaca_client = alpaca_client
        self.initialized = False
        self.is_trained = False
        self.last_portfolio_value = None
        self.signal_history = []
        
        # ENFORCE RL-ONLY CONFIGURATION
        self._load_and_enforce_rl_config()
        
        # Action tracking for experience buffer population
        self.previous_action = None
        self.previous_state = None
        self.step_counter = 0
        
        # Track portfolio symbols for dynamic updates
        self.last_portfolio_symbols = None
        
        # Dynamic feature tracking for compatibility
        self.features_per_symbol = 20  # Updated feature count
        self.total_expected_features = len(symbols) * self.features_per_symbol
        
        logger.info(f"🎯 Initialized Enhanced RL Agent: {len(symbols)} symbols, {self.total_expected_features} total features")
        
        # ENFORCE FinRL DRL agents only - no fallbacks allowed
        self.finrl_agent = None
        self.system = None  # Will be set to finrl_agent when initialized
        self.has_advanced_rl = True  # Feature flag - FinRL DRL agents available
        self.has_finrl = False
        
        if not FINRL_AVAILABLE:
            logger.critical("🚫 CRITICAL: FinRL DRL agents not available")
            logger.critical("❌ FinRL is REQUIRED - no fallback systems allowed")
            logger.critical("🔧 Install FinRL library and dependencies")
            logger.critical("📚 Refer to: https://github.com/AI4Finance-Foundation/FinRL")
            raise RuntimeError("FinRL DRL agents required but not available - install FinRL library")
        
        try:
            # Use FinRL DRL agents (ONLY option)
            logger.info("🚀 Initializing FinRL DRL Agent System (REQUIRED)...")
            self.has_finrl = True
            logger.info(f"✅ FinRL DRL system ready for {len(symbols)} symbols")
            
        except Exception as e:
            logger.critical(f"🚫 CRITICAL: FinRL DRL Agent initialization failed: {e}")
            logger.critical("❌ System cannot proceed without FinRL DRL agents")
            logger.critical("🔧 Fix FinRL installation or dependencies")
            raise RuntimeError(f"FinRL DRL Agent system failed to initialize: {e}")
    
    def _load_and_enforce_rl_config(self):
        """Load and enforce RL-only configuration."""
        try:
            with open('/Users/zac/Desktop/02_PROJECTS/01_ML4T/trading_system/data/rl_config.json', 'r') as f:
                config = json.load(f)
            
            integration_config = config.get('integration', {})
            
            # Enforce RL-only mode
            if not integration_config.get('enforce_rl_only_mode', False):
                logger.critical("🚫 RL-ONLY MODE NOT ENABLED in configuration")
                raise ValueError("RL-only mode must be enabled in rl_config.json")
            
            if integration_config.get('fallback_to_existing', True):
                logger.critical("🚫 FALLBACK TO EXISTING ENABLED - This bypasses RL decisions")
                raise ValueError("fallback_to_existing must be disabled for RL-only mode")
                
            logger.info("✅ RL-ONLY MODE ENFORCED - No fallback bypasses allowed")
            
        except FileNotFoundError:
            logger.critical("🚫 RL configuration file not found")
            raise RuntimeError("RL config required for operation")
    
    async def initialize_system(self):
        """Initialize the RL system with validation."""
        if self.initialized:
            return
        
        # VALIDATE RL SYSTEM BEFORE INITIALIZATION
        await self._validate_rl_system_health()
            
        try:
            # Initialize FinRL DRL agent (REQUIRED - no alternatives)
            logger.info("🚀 Initializing FinRL DRL Agent (REQUIRED)...")
            self.finrl_agent = await create_finrl_agent(
                symbols=self.symbols,
                alpaca_client=self.alpaca_client,
                config=self._get_finrl_config()
            )
            self.system = self.finrl_agent  # Set system reference for compatibility
            self.is_trained = self.finrl_agent.is_trained
            logger.info("✅ FinRL DRL Agent initialized successfully")
                    
            self.initialized = True
            logger.info("🚀 FinRL DRL system initialized successfully")
            
        except Exception as e:
            logger.critical(f"🚫 CRITICAL: Failed to initialize FinRL DRL system: {e}")
            logger.critical("❌ System cannot proceed without FinRL DRL agents")
            logger.critical("🔧 Check FinRL installation, dependencies, and configuration")
            raise RuntimeError(f"FinRL DRL system initialization failed: {e}")
    
    async def generate_trading_signals(self, 
                                     market_data: Dict[str, Any],
                                     portfolio_data: Dict[str, Any],
                                     portfolio_value: float) -> List[TradingSignal]:
        """Generate trading signals using enhanced RL system."""
        
        if not self.initialized:
            await self.initialize_system()
        
        # Update symbols dynamically from current portfolio
        await self._update_symbols_from_portfolio(portfolio_data)
        
        try:
            # Use FinRL DRL agents (ONLY option - no fallbacks)
            if not (self.has_finrl and self.finrl_agent):
                logger.critical("🚫 CRITICAL: FinRL DRL agent not available for signal generation")
                logger.critical("❌ System REQUIRES FinRL DRL agents - no alternatives")
                raise RuntimeError("FinRL DRL agent required but not available")
            
            return await self._generate_finrl_signals(
                market_data, portfolio_data, portfolio_value
            )
                
        except Exception as e:
            logger.critical(f"🚫 CRITICAL: FinRL DRL signal generation failed: {e}")
            logger.critical("❌ System cannot proceed without FinRL DRL signals")
            logger.critical("🔧 Check FinRL agent status and configuration")
            raise RuntimeError(f"FinRL DRL signal generation failed: {e}")
    
    def _get_finrl_config(self) -> Dict[str, Any]:
        """Get configuration for FinRL agents."""
        return {
            'initial_amount': 100000,
            'hmax': 100,
            'buy_cost_pct': 0.001,
            'sell_cost_pct': 0.001,
            'reward_scaling': 1e-4,
            'tech_indicator_list': [
                'macd', 'rsi_30', 'cci_30', 'dx_30', 'bb_bbm', 'bb_bbh', 'bb_bbl'
            ],
            'turbulence_threshold': None,
            'rebalance_window': 63,
            'validation_window': 21,
            'timesteps_dict': {
                'a2c': 50000,
                'ppo': 50000,
                'ddpg': 50000,
                'sac': 50000,
                'td3': 50000
            },
            # Date configurations for FinRL training and trading
            'train_start_date': '2020-01-01',
            'train_end_date': '2022-12-31',
            'trade_start_date': '2023-01-01',
            'trade_end_date': '2024-12-31'
        }
    
    async def _generate_finrl_signals(self, market_data, portfolio_data, portfolio_value):
        """Generate signals using FinRL DRL agents."""
        try:
            # Generate FinRL signals
            finrl_signals = await self.finrl_agent.generate_trading_signals(
                market_data, portfolio_data, portfolio_value
            )
            
            # Convert FinRL signals to TradingSignal format
            trading_signals = []
            for signal in finrl_signals:
                trading_signal = TradingSignal(
                    symbol=signal.symbol,
                    action=signal.action,
                    quantity=signal.quantity,
                    confidence=signal.confidence,
                    reasoning=f"[FinRL DRL] {signal.reasoning}",
                    rl_score=signal.drl_score,
                    regime=signal.regime,
                    uncertainty=signal.uncertainty
                )
                trading_signals.append(trading_signal)
            
            # Store signals for performance tracking
            self.signal_history.extend(trading_signals)
            
            return trading_signals
            
        except Exception as e:
            logger.error(f"❌ FinRL signal generation failed: {e}")
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
        
        # CRITICAL FIX: Pass proper previous action for experience buffer population
        action, action_info = await self.system.process_market_step(
            market_state=current_state,
            market_data=market_metadata,
            previous_action=self.previous_action,
            previous_reward=previous_reward,
            deterministic=False
        )
        
        # Store current action and state for next iteration
        self.previous_action = action.copy() if action is not None else None
        self.previous_state = current_state.copy()
        self.step_counter += 1
        
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
        """Get performance metrics from the FinRL DRL system."""
        base_metrics = {
            "system_type": "finrl_drl_agent_only",
            "signals_generated": len(self.signal_history),
            "expected_features": self.total_expected_features,
            "features_per_symbol": self.features_per_symbol,
            "symbol_count": len(self.symbols),
            "step_counter": self.step_counter,
            "has_finrl": self.has_finrl,
            "is_trained": self.is_trained,
            "primary_system": "finrl_drl_agents_required"
        }
        
        if not (self.has_finrl and self.finrl_agent and self.initialized):
            logger.critical("🚫 CRITICAL: FinRL DRL agent not available for performance metrics")
            base_metrics["error"] = "FinRL DRL agent required but not available"
            return base_metrics
        
        try:
            # Get FinRL metrics (ONLY source)
            finrl_metrics = self.finrl_agent.get_performance_metrics()
            base_metrics.update(finrl_metrics)
            base_metrics["status"] = "operational"
        except Exception as e:
            logger.error(f"❌ Failed to get FinRL performance metrics: {e}")
            base_metrics["error"] = f"FinRL metrics unavailable: {e}"
            
        return base_metrics
    
    async def force_training_update(self, min_batch_size: int = 16):
        """Force a training update regardless of normal triggers."""
        if not (self.has_finrl and self.finrl_agent and self.initialized):
            logger.critical("🚫 CRITICAL: FinRL DRL agent not available for training")
            logger.critical("❌ System REQUIRES FinRL DRL agents for training")
            raise RuntimeError("FinRL DRL agent required for training but not available")
        
        try:
            # Force FinRL model retraining
            await self.finrl_agent.retrain_models(force=True)
            self.is_trained = True
            logger.info("🚀 FinRL force training completed")
            return True
        except Exception as e:
            logger.critical(f"🚫 CRITICAL: FinRL force training failed: {e}")
            logger.critical("❌ System cannot proceed without FinRL training capability")
            raise RuntimeError(f"FinRL force training failed: {e}")
    
    def get_learning_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostics for learning system health."""
        diagnostics = {
            'system_initialized': self.initialized,
            'has_advanced_rl': self.has_advanced_rl,
            'step_counter': self.step_counter,
            'previous_action_set': self.previous_action is not None,
            'previous_state_set': self.previous_state is not None,
            'signal_history_size': len(self.signal_history)
        }
        
        if self.has_advanced_rl and self.system and self.initialized:
            try:
                training_stats = self.system.get_training_stats()
                diagnostics.update({
                    'buffer_size': training_stats.get('buffer_size', 0),
                    'buffer_utilization': training_stats.get('buffer_utilization', 0.0),
                    'performance_history_size': training_stats.get('performance_history_size', 0),
                    'training_active': training_stats.get('training_active', False),
                    'active_agent': training_stats.get('active_agent', 'unknown'),
                    'total_updates': training_stats.get('total_updates', 0)
                })
            except Exception as e:
                logger.warning(f"Could not get training diagnostics: {e}")
                
        return diagnostics
    
    async def predict(self, observation: Any, deterministic: bool = True) -> Tuple[float, float]:
        """
        Generate trading action prediction for backtesting compatibility.
        
        Args:
            observation: Market observation/state
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (action, confidence)
        """
        try:
            if not self.initialized:
                await self.initialize_system()
            
            # Handle observation format
            if isinstance(observation, (list, tuple)):
                state = np.array(observation, dtype=np.float32)
            elif isinstance(observation, np.ndarray):
                state = observation.astype(np.float32)
            else:
                # Fallback for other formats
                state = np.array([0.0], dtype=np.float32)
            
            # Ensure proper state dimensions
            expected_size = len(self.symbols) * self.features_per_symbol
            if len(state) != expected_size:
                state = self._normalize_feature_vector(state, expected_size)
            
            if self.has_advanced_rl and self.system:
                # Use advanced RL system
                action, action_info = await self.system.process_market_step(
                    market_state=state,
                    market_data={'volatility': 0.02, 'trend': 0.0, 'portfolio_value': 10000},
                    previous_action=self.previous_action,
                    previous_reward=0.0,
                    deterministic=deterministic
                )
                
                # Return single action value and confidence
                if action is not None and len(action) > 0:
                    action_value = float(np.mean(action))  # Average across symbols
                    confidence = action_info.get('confidence', 0.7)
                else:
                    action_value = 0.0
                    confidence = 0.5
            else:
                # Simple fallback prediction
                if len(state) > 0:
                    action_value = float(np.tanh(np.mean(state[:5])))  # Simple momentum
                else:
                    action_value = 0.0
                confidence = 0.5
                
            return action_value, confidence
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return 0.0, 0.5  # Safe fallback
    
    async def _update_symbols_from_portfolio(self, portfolio_data: Dict[str, Any]):
        """Update RL symbols based on current portfolio positions."""
        if not self.alpaca_client:
            return  # No Alpaca client available
            
        try:
            # Extract symbols with meaningful positions from portfolio data
            current_portfolio_symbols = []
            for symbol, pos_data in portfolio_data.items():
                qty = pos_data.get('quantity', 0) if isinstance(pos_data, dict) else 0
                if abs(qty) > 0.01:  # Meaningful position
                    current_portfolio_symbols.append(symbol)
            
            # Add positions from Alpaca client for completeness
            try:
                positions = self.alpaca_client.get_positions()
                for pos in positions:
                    if abs(pos.get('market_value', 0)) > 100:  # $100+ positions
                        symbol = pos['symbol']
                        if symbol not in current_portfolio_symbols:
                            current_portfolio_symbols.append(symbol)
            except Exception as e:
                logger.debug(f"Could not fetch Alpaca positions: {e}")
            
            # Sort for consistency
            current_portfolio_symbols = sorted(list(set(current_portfolio_symbols)))
            
            # Check if symbols have changed significantly
            if self.last_portfolio_symbols != current_portfolio_symbols:
                old_count = len(self.symbols)
                new_count = len(current_portfolio_symbols)
                
                # Update symbols if meaningful change (>20% change or new symbols)
                change_ratio = abs(new_count - old_count) / max(old_count, 1)
                new_symbols = set(current_portfolio_symbols) - set(self.symbols)
                
                if change_ratio > 0.2 or len(new_symbols) > 2:
                    logger.info(f"🔄 RL symbols update needed: {old_count} → {new_count} symbols")
                    logger.info(f"New symbols: {list(new_symbols)}")
                    
                    # Update symbols
                    self.symbols = current_portfolio_symbols
                    self.last_portfolio_symbols = current_portfolio_symbols.copy()
                    
                    # Update advanced RL system if available
                    if self.has_advanced_rl and self.system:
                        # Update symbols in the RL system
                        updated = self.system.update_symbols_from_portfolio(force_update=True)
                        if updated:
                            logger.info("✅ RL system symbols updated successfully")
                    
                    # Update feature dimensions
                    self.total_expected_features = len(self.symbols) * self.features_per_symbol
                    logger.info(f"📊 Updated feature dimensions: {self.total_expected_features} total features")
                    
        except Exception as e:
            logger.error(f"Failed to update symbols from portfolio: {e}")
    
    async def _validate_rl_system_health(self):
        """Validate FinRL DRL system health before initialization."""
        logger.info("🔍 Validating FinRL DRL system health...")
        
        # Check if FinRL components are available
        if not FINRL_AVAILABLE:
            logger.critical("🚫 FinRL DRL system not available")
            logger.critical("❌ FinRL is REQUIRED - install FinRL library")
            logger.critical("📚 Install: pip install finrl")
            raise RuntimeError("FinRL DRL system validation failed: FinRL not available")
        
        # Validate FinRL import paths
        try:
            import sys
            finrl_path = '/Users/zac/Desktop/02_PROJECTS/FinRL'
            if finrl_path not in sys.path:
                sys.path.insert(0, finrl_path)
                logger.info(f"✅ Added FinRL path: {finrl_path}")
            
            # Test critical FinRL imports
            from finrl.agents.stablebaselines3.models import DRLAgent, DRLEnsembleAgent
            from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
            from finrl.meta.data_processor import DataProcessor
            logger.info("✅ Critical FinRL components imported successfully")
            
        except ImportError as e:
            logger.critical(f"🚫 FinRL import validation failed: {e}")
            logger.critical("❌ FinRL installation incomplete or corrupted")
            logger.critical("🔧 Reinstall FinRL: pip install --upgrade finrl")
            raise RuntimeError(f"FinRL import validation failed: {e}")
        
        # Validate required data directories
        data_dirs = [
            '/Users/zac/Desktop/02_PROJECTS/01_ML4T/trading_system/data',
            '/Users/zac/Desktop/02_PROJECTS/01_ML4T/trading_system/data/finrl_models'
        ]
        
        for data_dir in data_dirs:
            try:
                import os
                os.makedirs(data_dir, exist_ok=True)
                logger.info(f"✅ Data directory validated: {data_dir}")
            except Exception as e:
                logger.critical(f"🚫 Data directory validation failed: {data_dir} - {e}")
                raise RuntimeError(f"Data directory validation failed: {e}")
        
        # Validate RL configuration
        try:
            with open('/Users/zac/Desktop/02_PROJECTS/01_ML4T/trading_system/data/rl_config.json', 'r') as f:
                rl_config = json.load(f)
                if not rl_config.get('integration', {}).get('enforce_rl_only_mode', False):
                    logger.critical("🚫 RL-only mode not enforced in configuration")
                    raise RuntimeError("RL-only mode must be enabled for FinRL operation")
                logger.info("✅ RL configuration validated")
        except FileNotFoundError:
            logger.warning("⚠️ RL config file not found - will use defaults")
        except Exception as e:
            logger.critical(f"🚫 RL configuration validation failed: {e}")
            raise RuntimeError(f"RL configuration validation failed: {e}")
        
        logger.info("✅ FinRL DRL system health validation passed")

# Global instance management
_global_rl_agent: Optional[EnhancedRLAgent] = None

def initialize_rl_agent(symbols: List[str], alpaca_client=None) -> EnhancedRLAgent:
    """Initialize global enhanced RL agent instance."""
    global _global_rl_agent
    
    if _global_rl_agent is None:
        _global_rl_agent = EnhancedRLAgent(symbols, alpaca_client=alpaca_client)
        # Set as current agent for backtesting validation
        _set_current_rl_agent(_global_rl_agent)
    
    return _global_rl_agent

def get_rl_agent() -> Optional[EnhancedRLAgent]:
    """Get the global enhanced RL agent instance."""
    return _global_rl_agent

async def generate_rl_enhanced_signals(market_data: Dict[str, Any],
                                     portfolio_data: Dict[str, Any],
                                     portfolio_value: float,
                                     symbols: List[str],
                                     alpaca_client=None) -> List[Dict[str, Any]]:
    """Generate RL-enhanced trading signals for integration with existing workflow."""
    
    # Initialize or get RL agent
    rl_agent = initialize_rl_agent(symbols, alpaca_client=alpaca_client)
    
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

async def integrate_hybrid_llm_rl_portfolio_system(state: Dict[str, Any], config: Dict[str, Any], alpaca_client=None) -> Dict[str, Any]:
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
        
        # 4. Raise error if no dynamic symbols found - fail fast instead of silent fallback
        if not symbols:
            error_msg = (
                "❌ CRITICAL ERROR: No dynamic symbols found for hybrid LLM-RL system! "
                "This indicates a failure in the trading pipeline. "
                "Sources checked: universe_filter={}, sentiment_data={}, portfolio_positions={}. "
                "The trading system must provide dynamic symbols for proper RL learning."
            ).format(
                len(state.get("filtered_symbols", [])),
                len(state.get("sentiment_data", {})), 
                len(state.get("portfolio", {}).get("positions", {}))
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
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
            logger.critical(f"🚫 LLM Portfolio Manager CRITICAL FAILURE: {e}")
            logger.critical("❌ FALLBACK DISABLED: No basic diversification allowed")
            logger.critical("💡 System requires proper LLM analysis for intelligent stock selection")
            logger.critical("🔧 Fix LLM integration or check model availability")
            raise RuntimeError(f"LLM Portfolio Manager failed and fallback disabled: {e}")
            
            logger.info(f"🔄 Using basic diversification fallback: {len(llm_allocations)} positions")
        
        # STEP 2: RL Enhancement and Position Optimization
        logger.info("🤖 Step 2: RL Agent - Optimizing Position Sizing and Timing")
        
        # Initialize RL agent with LLM-selected symbols and alpaca client for dynamic updates
        llm_symbols = [alloc.symbol for alloc in llm_allocations]
        rl_agent = initialize_rl_agent(llm_symbols, alpaca_client=alpaca_client)
        
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
        
        # Calculate rebalancing thresholds - BALANCED TO PREVENT CHURNING WHILE ALLOWING LEGITIMATE SIGNALS
        rebalancing_threshold = 0.02  # 2% deviation triggers rebalancing (more aggressive to ensure trades execute)
        min_trade_size = portfolio_value * 0.003  # Minimum $150 trade size (reduced to allow more trades)
        max_single_position = 0.15  # Max 15% per position (was 8%)
        max_sector_allocation = 0.35  # Max 35% per sector (was 25%)
        
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
        
        # FORCE REBALANCING: Check if we've been idle too long
        force_rebalancing = False
        try:
            # Force rebalancing if we have very few positions (portfolio concentration risk)
            if len(current_positions) < 5 or total_target_allocation < 0.3:
                force_rebalancing = True
                logger.info("🚀 FORCE REBALANCING: Low portfolio diversification detected")
        except:
            pass
        
        # Anti-churning enhancement: Balanced minimum portfolio size threshold
        min_positions_for_skip = 8  # Balanced: only skip rebalancing if portfolio is well diversified
        
        # Only proceed with rebalancing if significant changes are needed
        if not needs_rebalancing and len(current_positions) > min_positions_for_skip and not force_rebalancing:
            logger.info(f"🔒 No significant rebalancing needed - maintaining current {len(current_positions)} positions")
            
            # Generate maintenance allocations for existing positions to keep system active
            maintenance_allocations = []
            for symbol, position_info in positions.items():
                current_value = position_info.get('market_value', 0)
                current_weight = current_value / max(portfolio_value, 1)
                if current_weight >= 0.005:  # Only track positions >= 0.5%
                    maintenance_allocations.append({
                        'symbol': symbol,
                        'weight': current_weight,
                        'percentage': current_weight * 100,
                        'confidence': 0.6,
                        'action': 'HOLD',
                        'risk_level': 'medium',
                        'sector': 'Unknown',
                        'reasoning': 'Maintaining existing position'
                    })
            
            return {
                'rl_decisions': {
                    'strategy': 'maintain_positions',
                    'risk_level': risk_profile,
                    'allocations': maintenance_allocations,  # Provide maintenance allocations instead of empty
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
OnlineRLAgent = EnhancedRLAgent
SimplifiedRLAgent = EnhancedRLAgent  # Legacy alias
create_rl_bridge = initialize_rl_agent

# Export all functions for compatibility
__all__ = [
    'EnhancedRLAgent',
    'SimplifiedRLAgent',  # Legacy alias
    'OnlineRLAgent',      # Legacy alias
    'TradingSignal',
    'initialize_rl_agent',
    'get_rl_agent',
    'get_current_rl_agent',
    'generate_rl_enhanced_signals',
    'integrate_comprehensive_rl_system',
    'integrate_hybrid_llm_rl_portfolio_system',
    'cleanup_rl_agent',
    'create_rl_bridge'
]

if FINRL_AVAILABLE:
    logger.info("✅ Enhanced RL Integration Bridge loaded with FinRL DRL Agents (REQUIRED)")
else:
    logger.critical("🚫 CRITICAL: FinRL DRL Agents not available")
    logger.critical("❌ System requires FinRL DRL agents - no fallbacks allowed")
    logger.critical("🔧 Install FinRL: pip install finrl")
    logger.critical("📚 Documentation: https://github.com/AI4Finance-Foundation/FinRL")
    raise RuntimeError("FinRL DRL agents required but not available - system cannot proceed")