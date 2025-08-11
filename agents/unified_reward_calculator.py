#!/usr/bin/env python3
"""
Unified Reward Calculator for RL Trading Systems
Consolidates all reward calculation logic into a single, comprehensive system
that supports all trade types: market orders, limit orders, short selling, etc.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
from enum import Enum

logger = logging.getLogger(__name__)

class TradeType(Enum):
    """Supported trade types for reward calculation."""
    MARKET_BUY = "market_buy"
    MARKET_SELL = "market_sell"
    LIMIT_BUY = "limit_buy"
    LIMIT_SELL = "limit_sell"
    SHORT_SELL = "short_sell"
    SHORT_COVER = "short_cover"
    HOLD = "hold"

@dataclass
class RewardComponents:
    """Individual components of the reward calculation."""
    pnl_change: float = 0.0
    cost_penalty: float = 0.0
    risk_penalty: float = 0.0
    execution_penalty: float = 0.0
    sentiment_alignment: float = 0.0
    regime_bonus: float = 0.0
    diversification_bonus: float = 0.0
    total_reward: float = 0.0

@dataclass
class TradeMetrics:
    """Comprehensive trade execution metrics."""
    portfolio_value_t: float
    portfolio_value_t1: float
    trade_type: TradeType
    position_size: float = 0.0
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    current_price: float = 0.0
    
    # Cost components
    slippage: float = 0.0
    commission: float = 0.0
    borrow_fee: float = 0.0  # For short positions
    bid_ask_spread: float = 0.0
    
    # Execution quality metrics
    missed_opportunity_penalty: float = 0.0
    adverse_selection_penalty: float = 0.0
    price_improvement: float = 0.0  # Positive if got better price than expected
    
    # Risk metrics
    leverage: float = 1.0
    volatility: float = 0.0
    position_concentration: float = 0.0  # Portfolio % in this position
    drawdown: float = 0.0
    
    # Market alignment
    sentiment_score: float = 0.0  # -1 to 1
    regime_alignment: float = 0.0  # How well position aligns with detected regime
    technical_momentum: float = 0.0  # Technical analysis signal strength

class UnifiedRewardCalculator:
    """
    Unified reward calculator supporting all trade types and market conditions.
    Consolidates reward logic from all RL environments into a single system.
    """
    
    def __init__(self, 
                 lambda_c: float = 1.0,  # Cost penalty weight
                 lambda_r: float = 1.0,  # Risk penalty weight  
                 lambda_e: float = 1.0,  # Execution penalty weight
                 lambda_s: float = 0.5,  # Sentiment alignment weight
                 lambda_reg: float = 0.3,  # Regime alignment weight
                 lambda_div: float = 0.2,  # Diversification bonus weight
                 risk_free_rate: float = 0.02,  # For Sharpe-like calculations
                 max_drawdown_threshold: float = 0.05,  # 5% drawdown threshold
                 normalization_factor: float = 1000.0):  # Portfolio normalization
        """
        Initialize unified reward calculator with configurable weights.
        
        Args:
            lambda_c: Weight for transaction cost penalties
            lambda_r: Weight for risk-based penalties
            lambda_e: Weight for execution quality penalties
            lambda_s: Weight for sentiment alignment bonuses
            lambda_reg: Weight for market regime alignment bonuses
            lambda_div: Weight for diversification bonuses
            risk_free_rate: Risk-free rate for risk-adjusted returns
            max_drawdown_threshold: Maximum acceptable drawdown
            normalization_factor: Factor for normalizing rewards
        """
        self.lambda_c = lambda_c
        self.lambda_r = lambda_r
        self.lambda_e = lambda_e
        self.lambda_s = lambda_s
        self.lambda_reg = lambda_reg
        self.lambda_div = lambda_div
        self.risk_free_rate = risk_free_rate
        self.max_drawdown_threshold = max_drawdown_threshold
        self.normalization_factor = normalization_factor
        
        # Track performance for risk adjustments
        self.returns_history: List[float] = []
        self.volatility_window = 20
        
    def calculate_reward(self, metrics: TradeMetrics) -> RewardComponents:
        """
        Calculate comprehensive reward using unified approach.
        
        Args:
            metrics: Trade execution and market metrics
            
        Returns:
            RewardComponents: Detailed breakdown of reward calculation
        """
        
        # 1. Core PnL Change (Risk-Adjusted)
        pnl_change = self._calculate_pnl_reward(metrics)
        
        # 2. Transaction Cost Penalties
        cost_penalty = self._calculate_cost_penalty(metrics)
        
        # 3. Risk-Based Penalties
        risk_penalty = self._calculate_risk_penalty(metrics)
        
        # 4. Execution Quality Penalties/Bonuses
        execution_penalty = self._calculate_execution_penalty(metrics)
        
        # 5. Sentiment Alignment Bonus
        sentiment_alignment = self._calculate_sentiment_alignment(metrics)
        
        # 6. Market Regime Alignment Bonus
        regime_bonus = self._calculate_regime_bonus(metrics)
        
        # 7. Diversification Bonus
        diversification_bonus = self._calculate_diversification_bonus(metrics)
        
        # 8. Calculate Total Reward
        total_reward = (
            pnl_change
            - self.lambda_c * cost_penalty
            - self.lambda_r * risk_penalty  
            - self.lambda_e * execution_penalty
            + self.lambda_s * sentiment_alignment
            + self.lambda_reg * regime_bonus
            + self.lambda_div * diversification_bonus
        )
        
        # Normalize reward
        total_reward = total_reward / self.normalization_factor
        
        return RewardComponents(
            pnl_change=pnl_change,
            cost_penalty=cost_penalty,
            risk_penalty=risk_penalty,
            execution_penalty=execution_penalty,
            sentiment_alignment=sentiment_alignment,
            regime_bonus=regime_bonus,
            diversification_bonus=diversification_bonus,
            total_reward=total_reward
        )
    
    def _calculate_pnl_reward(self, metrics: TradeMetrics) -> float:
        """Calculate risk-adjusted PnL reward."""
        
        # Base PnL change
        pnl_change = metrics.portfolio_value_t1 - metrics.portfolio_value_t
        
        # Track returns for volatility calculation
        if metrics.portfolio_value_t > 0:
            portfolio_return = pnl_change / metrics.portfolio_value_t
            self.returns_history.append(portfolio_return)
            
            # Keep only recent returns
            if len(self.returns_history) > self.volatility_window:
                self.returns_history.pop(0)
        
        # Risk adjustment: penalize high volatility
        if len(self.returns_history) > 5:  # Need minimum history
            volatility = np.std(self.returns_history)
            if volatility > 0:
                # Sharpe-like adjustment
                mean_return = np.mean(self.returns_history)
                risk_adjusted_return = (mean_return - self.risk_free_rate/252) / volatility
                
                # Apply risk adjustment to current PnL
                pnl_change = pnl_change * min(2.0, max(0.5, 1.0 + risk_adjusted_return))
        
        return pnl_change
    
    def _calculate_cost_penalty(self, metrics: TradeMetrics) -> float:
        """Calculate transaction cost penalties."""
        
        cost_penalty = 0.0
        
        # Basic trading costs
        cost_penalty += metrics.slippage
        cost_penalty += metrics.commission
        cost_penalty += metrics.bid_ask_spread * abs(metrics.position_size) * 0.5
        
        # Short selling specific costs
        if metrics.trade_type in [TradeType.SHORT_SELL, TradeType.SHORT_COVER]:
            cost_penalty += metrics.borrow_fee
        
        # Scale by position size
        cost_penalty *= abs(metrics.position_size)
        
        return cost_penalty
    
    def _calculate_risk_penalty(self, metrics: TradeMetrics) -> float:
        """Calculate risk-based penalties."""
        
        risk_penalty = 0.0
        
        # Leverage penalty
        if metrics.leverage > 1.0:
            risk_penalty += (metrics.leverage - 1.0) ** 2 * 0.01
        
        # Concentration penalty
        if metrics.position_concentration > 0.1:  # More than 10% of portfolio
            concentration_excess = metrics.position_concentration - 0.1
            risk_penalty += concentration_excess ** 2 * 0.05
        
        # Drawdown penalty
        if metrics.drawdown > self.max_drawdown_threshold:
            drawdown_excess = metrics.drawdown - self.max_drawdown_threshold
            risk_penalty += drawdown_excess ** 2 * 0.1
        
        # Volatility penalty
        risk_penalty += metrics.volatility ** 2 * 0.02
        
        return risk_penalty
    
    def _calculate_execution_penalty(self, metrics: TradeMetrics) -> float:
        """Calculate execution quality penalties and bonuses."""
        
        execution_penalty = 0.0
        
        # Limit order specific penalties/bonuses
        if metrics.trade_type in [TradeType.LIMIT_BUY, TradeType.LIMIT_SELL]:
            execution_penalty += metrics.missed_opportunity_penalty
            execution_penalty += metrics.adverse_selection_penalty
            execution_penalty -= metrics.price_improvement  # Bonus for good execution
        
        # Market impact penalty for large orders
        if abs(metrics.position_size) > 1000:  # Large order threshold
            market_impact = (abs(metrics.position_size) - 1000) / 10000
            execution_penalty += market_impact * 0.01
        
        return execution_penalty
    
    def _calculate_sentiment_alignment(self, metrics: TradeMetrics) -> float:
        """Calculate sentiment alignment bonus."""
        
        if abs(metrics.sentiment_score) < 0.1:  # Neutral sentiment
            return 0.0
        
        # Determine position direction
        position_direction = 0.0
        if metrics.trade_type in [TradeType.MARKET_BUY, TradeType.LIMIT_BUY]:
            position_direction = 1.0
        elif metrics.trade_type in [TradeType.MARKET_SELL, TradeType.LIMIT_SELL, TradeType.SHORT_SELL]:
            position_direction = -1.0
        
        # Reward alignment between sentiment and position
        alignment = position_direction * metrics.sentiment_score
        
        # Scale by position size
        return alignment * abs(metrics.position_size) * 0.001
    
    def _calculate_regime_bonus(self, metrics: TradeMetrics) -> float:
        """Calculate market regime alignment bonus."""
        
        if abs(metrics.regime_alignment) < 0.1:
            return 0.0
        
        # Bonus for aligning trades with detected market regime
        regime_bonus = metrics.regime_alignment * abs(metrics.position_size) * 0.0005
        
        return regime_bonus
    
    def _calculate_diversification_bonus(self, metrics: TradeMetrics) -> float:
        """Calculate diversification bonus."""
        
        # Reward smaller positions (encourages diversification)
        if metrics.position_concentration < 0.05:  # Less than 5% concentration
            return 0.001 * abs(metrics.position_size)
        
        return 0.0
    
    def get_reward_breakdown(self, metrics: TradeMetrics) -> Dict[str, float]:
        """Get detailed breakdown of reward components for analysis."""
        
        components = self.calculate_reward(metrics)
        
        return {
            'pnl_change': components.pnl_change,
            'cost_penalty': -self.lambda_c * components.cost_penalty,
            'risk_penalty': -self.lambda_r * components.risk_penalty,
            'execution_penalty': -self.lambda_e * components.execution_penalty,
            'sentiment_alignment': self.lambda_s * components.sentiment_alignment,
            'regime_bonus': self.lambda_reg * components.regime_bonus,
            'diversification_bonus': self.lambda_div * components.diversification_bonus,
            'total_reward': components.total_reward
        }
    
    def adjust_weights(self, **kwargs):
        """Dynamically adjust reward component weights."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated {key} to {value}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize calculator
    calculator = UnifiedRewardCalculator()
    
    # Example trade metrics
    metrics = TradeMetrics(
        portfolio_value_t=100000,
        portfolio_value_t1=100050,
        trade_type=TradeType.LIMIT_BUY,
        position_size=100,
        entry_price=50.0,
        current_price=50.5,
        slippage=5,
        commission=1,
        missed_opportunity_penalty=10,
        adverse_selection_penalty=3,
        volatility=0.002,
        sentiment_score=0.3,
        regime_alignment=0.2
    )
    
    # Calculate reward
    reward_components = calculator.calculate_reward(metrics)
    breakdown = calculator.get_reward_breakdown(metrics)
    
    print("=== Unified Reward Calculation ===")
    print(f"Total Reward: {reward_components.total_reward:.6f}")
    print("\nBreakdown:")
    for component, value in breakdown.items():
        print(f"  {component}: {value:.6f}")