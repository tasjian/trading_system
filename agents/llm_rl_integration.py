#!/usr/bin/env python3
"""
LLM-RL Integration for Backtesting
Enhanced trading environment that combines LLM reasoning with RL decision making.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class MarketState:
    """Market state representation for RL environment."""
    current_price: float
    price_history: List[float]
    volume: float
    timestamp: datetime
    technical_indicators: Dict[str, float]
    sentiment_score: float = 0.0
    news_signals: List[str] = None
    
    def __post_init__(self):
        if self.news_signals is None:
            self.news_signals = []

class LLMStateEnricher:
    """Enriches market state with LLM-generated insights."""
    
    def __init__(self):
        self.reasoning_history = []
        logger.info("✅ LLM State Enricher initialized")
    
    async def enrich_state(self, market_state: MarketState, symbol: str) -> Dict[str, Any]:
        """
        Enrich market state with LLM reasoning.
        
        Args:
            market_state: Current market state
            symbol: Trading symbol
            
        Returns:
            Enriched state dictionary with LLM insights
        """
        try:
            # Simulate LLM analysis (in practice, this would call actual LLM)
            reasoning = self._generate_market_reasoning(market_state, symbol)
            
            enriched_state = {
                'price': market_state.current_price,
                'volume': market_state.volume,
                'sentiment': market_state.sentiment_score,
                'technical_signals': market_state.technical_indicators,
                'llm_reasoning': reasoning,
                'market_regime': self._detect_market_regime(market_state),
                'risk_assessment': self._assess_market_risk(market_state),
                'trade_opportunity': self._evaluate_trade_opportunity(market_state)
            }
            
            return enriched_state
            
        except Exception as e:
            logger.error(f"❌ State enrichment failed: {e}")
            return {
                'price': market_state.current_price,
                'volume': market_state.volume,
                'sentiment': 0.0,
                'llm_reasoning': "Analysis unavailable",
                'market_regime': 'unknown',
                'risk_assessment': 'medium',
                'trade_opportunity': 0.0
            }
    
    def _generate_market_reasoning(self, state: MarketState, symbol: str) -> str:
        """Generate LLM-style market reasoning."""
        price_trend = "bullish" if len(state.price_history) > 1 and state.current_price > state.price_history[-2] else "bearish"
        
        reasoning = f"""
        Market Analysis for {symbol}:
        - Current Price: ${state.current_price:.2f}
        - Price Trend: {price_trend}
        - Volume: {state.volume:,.0f}
        - Sentiment: {state.sentiment_score:.2f}
        - Technical Signals: {len(state.technical_indicators)} indicators active
        
        The market shows {price_trend} momentum with moderate volume activity.
        Sentiment indicators suggest {'positive' if state.sentiment_score > 0 else 'neutral' if state.sentiment_score == 0 else 'negative'} market conditions.
        """
        
        return reasoning.strip()
    
    def _detect_market_regime(self, state: MarketState) -> str:
        """Detect current market regime."""
        if len(state.price_history) < 10:
            return 'insufficient_data'
        
        recent_volatility = np.std(state.price_history[-10:]) / np.mean(state.price_history[-10:])
        
        if recent_volatility > 0.05:
            return 'high_volatility'
        elif recent_volatility < 0.02:
            return 'low_volatility'
        else:
            return 'normal'
    
    def _assess_market_risk(self, state: MarketState) -> str:
        """Assess current market risk level."""
        risk_factors = 0
        
        # Check volatility
        if len(state.price_history) > 5:
            volatility = np.std(state.price_history[-5:])
            if volatility > state.current_price * 0.03:  # 3% volatility
                risk_factors += 1
        
        # Check sentiment
        if state.sentiment_score < -0.5:
            risk_factors += 1
        
        # Check volume
        if state.volume == 0:
            risk_factors += 1
        
        if risk_factors >= 2:
            return 'high'
        elif risk_factors == 1:
            return 'medium'
        else:
            return 'low'
    
    def _evaluate_trade_opportunity(self, state: MarketState) -> float:
        """Evaluate trade opportunity score (0-1)."""
        opportunity_score = 0.5  # Neutral baseline
        
        # Adjust based on sentiment
        opportunity_score += state.sentiment_score * 0.2
        
        # Adjust based on technical indicators
        if state.technical_indicators:
            avg_signal = np.mean(list(state.technical_indicators.values()))
            opportunity_score += avg_signal * 0.3
        
        return max(0.0, min(1.0, opportunity_score))

class EnhancedLLMTradingEnvironment:
    """
    Enhanced trading environment that combines market simulation with LLM insights.
    Compatible with the backtesting framework.
    """
    
    def __init__(self, 
                 symbol: str,
                 initial_balance: float = 100000,
                 transaction_cost: float = 0.001,
                 max_position_size: float = 1.0):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        
        # Environment state
        self.current_step = 0
        self.position = 0.0  # Current position (-1 to 1)
        self.portfolio_value = initial_balance
        self.price_history = []
        self.trade_history = []
        
        # LLM integration
        self.state_enricher = LLMStateEnricher()
        
        # Market data
        self.market_data = None
        self.data_index = 0
        
        logger.info(f"✅ Enhanced LLM Trading Environment initialized for {symbol}")
    
    def set_data(self, market_data: pd.DataFrame):
        """Set market data for backtesting."""
        self.market_data = market_data
        self.data_index = 0
        logger.info(f"📊 Market data set: {len(market_data)} data points")
    
    async def reset(self) -> Dict[str, Any]:
        """Reset environment to initial state."""
        self.current_step = 0
        self.position = 0.0
        self.current_balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.price_history = []
        self.trade_history = []
        self.data_index = 0
        
        # Get initial observation
        obs = await self._get_observation()
        
        logger.info("🔄 Environment reset")
        return obs
    
    async def step(self, action: List[float]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: List containing action value (typically [-1, 1])
            
        Returns:
            (observation, reward, done, info)
        """
        try:
            # Extract action
            action_value = action[0] if isinstance(action, list) else action
            action_value = np.clip(action_value, -1.0, 1.0)
            
            # Get current market data
            if self.market_data is None or self.data_index >= len(self.market_data):
                # End of data
                obs = await self._get_observation()
                return obs, 0.0, True, {'reason': 'end_of_data'}
            
            current_data = self.market_data.iloc[self.data_index]
            current_price = current_data['close']
            volume = current_data.get('volume', 1000000)
            
            # Execute trade
            trade_executed = False
            if abs(action_value) > 0.1:  # Minimum action threshold
                old_position = self.position
                position_change = action_value * self.max_position_size
                self.position = np.clip(self.position + position_change, -1.0, 1.0)
                
                # Calculate transaction costs
                trade_size = abs(self.position - old_position)
                transaction_cost = trade_size * current_price * self.transaction_cost
                self.current_balance -= transaction_cost
                
                if trade_size > 0.01:  # Significant trade
                    trade_executed = True
                    self.trade_history.append({
                        'step': self.current_step,
                        'action': action_value,
                        'position_change': self.position - old_position,
                        'price': current_price,
                        'cost': transaction_cost
                    })
            
            # Update portfolio value
            position_value = self.position * current_price * self.initial_balance
            self.portfolio_value = self.current_balance + position_value
            
            # Calculate reward
            reward = self._calculate_reward(current_price, action_value)
            
            # Update state
            self.price_history.append(current_price)
            self.current_step += 1
            self.data_index += 1
            
            # Check if done
            done = (self.data_index >= len(self.market_data) or 
                   self.portfolio_value <= self.initial_balance * 0.5)  # 50% drawdown limit
            
            # Get next observation
            obs = await self._get_observation()
            
            # Info dictionary
            info = {
                'portfolio_value': self.portfolio_value,
                'position': self.position,
                'current_price': current_price,
                'trade_executed': trade_executed,
                'step': self.current_step
            }
            
            return obs, reward, done, info
            
        except Exception as e:
            logger.error(f"❌ Environment step failed: {e}")
            obs = await self._get_observation()
            return obs, -1.0, True, {'error': str(e)}
    
    async def _get_observation(self) -> Dict[str, Any]:
        """Get current market observation with LLM enrichment."""
        try:
            if self.market_data is None or self.data_index >= len(self.market_data):
                # Return default observation
                return {
                    'price': 100.0,
                    'position': self.position,
                    'portfolio_value': self.portfolio_value,
                    'step': self.current_step
                }
            
            current_data = self.market_data.iloc[self.data_index]
            current_price = current_data['close']
            
            # Create market state
            market_state = MarketState(
                current_price=current_price,
                price_history=self.price_history[-20:],  # Last 20 prices
                volume=current_data.get('volume', 1000000),
                timestamp=current_data.name,
                technical_indicators={
                    'rsi': self._calculate_rsi() if len(self.price_history) > 14 else 50.0,
                    'moving_avg_ratio': self._calculate_ma_ratio() if len(self.price_history) > 10 else 1.0
                },
                sentiment_score=0.0  # Placeholder
            )
            
            # Enrich with LLM insights
            enriched_state = await self.state_enricher.enrich_state(market_state, self.symbol)
            
            # Add environment-specific data
            enriched_state.update({
                'position': self.position,
                'portfolio_value': self.portfolio_value,
                'step': self.current_step,
                'balance': self.current_balance
            })
            
            return enriched_state
            
        except Exception as e:
            logger.error(f"❌ Observation generation failed: {e}")
            return {
                'price': 100.0,
                'position': self.position,
                'portfolio_value': self.portfolio_value,
                'step': self.current_step,
                'error': str(e)
            }
    
    def _calculate_reward(self, current_price: float, action: float) -> float:
        """Calculate reward for the current step."""
        try:
            if len(self.price_history) < 2:
                return 0.0
            
            # Price change reward
            price_change = (current_price - self.price_history[-1]) / self.price_history[-1]
            position_reward = self.position * price_change
            
            # Portfolio performance reward
            portfolio_change = (self.portfolio_value - self.initial_balance) / self.initial_balance
            
            # Combine rewards
            reward = position_reward * 10 + portfolio_change * 5
            
            # Penalize excessive trading
            if abs(action) > 0.8:
                reward -= 0.1
            
            return reward
            
        except Exception as e:
            logger.debug(f"Reward calculation error: {e}")
            return 0.0
    
    def _calculate_rsi(self, period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(self.price_history) < period + 1:
            return 50.0
        
        prices = np.array(self.price_history[-(period+1):])
        deltas = np.diff(prices)
        
        gains = deltas[deltas > 0].sum()
        losses = -deltas[deltas < 0].sum()
        
        if losses == 0:
            return 100.0
        
        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_ma_ratio(self, period: int = 10) -> float:
        """Calculate moving average ratio."""
        if len(self.price_history) < period:
            return 1.0
        
        recent_prices = self.price_history[-period:]
        ma = np.mean(recent_prices)
        current_price = self.price_history[-1]
        
        return current_price / ma if ma > 0 else 1.0
    
    async def close(self):
        """Cleanup environment resources."""
        logger.info("🧹 Cleaning up trading environment")
    
    @property
    def action_space(self):
        """Mock action space for compatibility."""
        class MockActionSpace:
            def sample(self):
                return [np.random.uniform(-1, 1)]
        
        return MockActionSpace()

# Compatibility alias
LLMTradingEnvironment = EnhancedLLMTradingEnvironment

__all__ = [
    'EnhancedLLMTradingEnvironment',
    'LLMTradingEnvironment', 
    'LLMStateEnricher',
    'MarketState'
]