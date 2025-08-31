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
    """Individual components of the mathematically-principled reward calculation."""
    # Core mathematical components (following R_t = α·(r_t/σ) + β·L_t + γ·S_t - δ·C_t)
    profit_component: float = 0.0      # α·(r_t/σ) - Risk-adjusted realized returns
    loss_cutting_component: float = 0.0   # β·L_t - Reward for cutting losses early
    signal_alignment_component: float = 0.0  # γ·S_t - Alignment with predictive signals
    transaction_cost_component: float = 0.0  # δ·C_t - Cost penalty for overtrading
    
    # Extended components for comprehensive trading
    cash_management_component: float = 0.0   # Emergency cash management bonus
    regime_alignment_component: float = 0.0  # Market regime awareness
    risk_management_component: float = 0.0   # Portfolio risk controls
    
    # Detailed breakdown (for analysis)
    realized_return_raw: float = 0.0
    unrealized_return_raw: float = 0.0
    volatility_normalization: float = 0.0
    total_reward: float = 0.0

@dataclass
class TradeMetrics:
    """Comprehensive trade execution metrics for mathematical reward calculation."""
    # Core mathematical inputs (following the formal model)
    realized_return: float = 0.0       # r_t: realized return at time t (normalized)
    unrealized_return: float = 0.0     # u_t: unrealized return at time t
    position_change: float = 0.0       # Change in position size (for cost calculation)
    signal_confidence: float = 0.0     # s_t: signal confidence (0-1 scale)
    trade_direction: float = 0.0       # +1 = buy/long, -1 = sell/short, 0 = hold
    signal_direction: float = 0.0      # +1 = bullish signal, -1 = bearish signal
    volatility: float = 0.0            # σ: rolling volatility for normalization
    transaction_cost: float = 0.0      # c: total transaction cost fraction
    
    # Extended metrics for comprehensive trading
    portfolio_value_t: float = 0.0
    portfolio_value_t1: float = 0.0
    trade_type: TradeType = TradeType.HOLD
    position_size: float = 0.0
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    current_price: float = 0.0
    
    # Cost breakdown
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
    
    # Cash management (NEW)
    current_cash: float = 0.0  # Current cash balance
    buying_power: float = 0.0  # Available buying power
    unrealized_pnl_pct: float = 0.0  # Position P&L percentage
    is_underperformer: bool = False  # True if position is underperforming
    cash_urgency: float = 0.0  # 0-1 scale of how urgently cash is needed

class UnifiedRewardCalculator:
    """
    Unified reward calculator supporting all trade types and market conditions.
    Consolidates reward logic from all RL environments into a single system.
    """
    
    def __init__(self, 
                 # Core mathematical weights (R_t = α·(r_t/σ) + β·L_t + γ·S_t - δ·C_t)
                 alpha: float = 1.0,     # Profit component weight (r_t/σ)
                 beta: float = 0.5,      # Loss-cutting discipline weight (L_t)
                 gamma: float = 0.3,     # Signal alignment weight (S_t)
                 delta: float = 0.2,     # Transaction cost penalty weight (C_t)
                 
                 # Extended trading system weights
                 lambda_cash: float = 0.4,    # Cash management weight (integrated with loss-cutting)
                 lambda_regime: float = 0.1,  # Market regime alignment weight
                 lambda_risk: float = 0.2,    # Risk management weight
                 
                 # Mathematical parameters
                 L_max: float = 0.05,         # Maximum allowed loss threshold (5%)
                 volatility_window: int = 20, # Rolling volatility window
                 min_volatility: float = 0.001, # Minimum volatility for normalization
                 risk_free_rate: float = 0.02):  # Risk-free rate for calculations
        """
        Initialize mathematically-principled reward calculator.
        
        Implements: R_t = α·(r_t/σ) + β·L_t + γ·S_t - δ·C_t + extensions
        
        Args:
            alpha: Profit component weight (r_t/σ) - rewards profitable trades scaled by risk
            beta: Loss-cutting weight (L_t) - rewards cutting losses before they get too large  
            gamma: Signal alignment weight (S_t) - rewards following predictive signals
            delta: Transaction cost weight (C_t) - penalizes unnecessary overtrading
            lambda_cash: Cash management weight - integrated with loss-cutting for liquidity
            lambda_regime: Market regime alignment weight
            lambda_risk: Portfolio risk management weight
            L_max: Maximum loss threshold before "too late" (default 5%)
            volatility_window: Rolling window for volatility calculation
            min_volatility: Minimum volatility for numerical stability
            risk_free_rate: Risk-free rate for risk calculations
        """
        # Core mathematical weights
        self.alpha = alpha
        self.beta = beta  
        self.gamma = gamma
        self.delta = delta
        
        # Extended system weights
        self.lambda_cash = lambda_cash
        self.lambda_regime = lambda_regime
        self.lambda_risk = lambda_risk
        
        # Mathematical parameters
        self.L_max = L_max
        self.volatility_window = volatility_window
        self.min_volatility = min_volatility
        self.risk_free_rate = risk_free_rate
        
        # Track performance for volatility calculation
        self.returns_history: List[float] = []
        
    def calculate_reward(self, metrics: TradeMetrics) -> RewardComponents:
        """
        Calculate mathematically-principled reward using:
        R_t = α·(r_t/σ) + β·L_t + γ·S_t - δ·C_t + extensions
        
        Args:
            metrics: Trade execution and market metrics with mathematical inputs
            
        Returns:
            RewardComponents: Detailed breakdown of reward calculation
        """
        
        # === CORE MATHEMATICAL COMPONENTS ===
        
        # 1. Profit Component: α·(r_t/σ) - Risk-adjusted realized returns
        profit_component = self._calculate_profit_component(metrics)
        
        # 2. Loss-Cutting Component: β·L_t - Reward for cutting losses early  
        loss_cutting_component = self._calculate_loss_cutting_component(metrics)
        
        # 3. Signal Alignment Component: γ·S_t - Alignment with predictive signals
        signal_alignment_component = self._calculate_signal_alignment_component(metrics)
        
        # 4. Transaction Cost Component: δ·C_t - Cost penalty for overtrading
        transaction_cost_component = self._calculate_transaction_cost_component(metrics)
        
        # === EXTENDED COMPONENTS FOR COMPREHENSIVE TRADING ===
        
        # 5. Cash Management (integrated with loss-cutting for liquidity needs)
        cash_management_component = self._calculate_cash_management_component(metrics)
        
        # 6. Market Regime Alignment (market condition awareness)
        regime_alignment_component = self._calculate_regime_alignment_component(metrics)
        
        # 7. Risk Management (portfolio-level risk controls)
        risk_management_component = self._calculate_risk_management_component(metrics)
        
        # === TOTAL REWARD CALCULATION ===
        
        total_reward = (
            self.alpha * profit_component +
            self.beta * loss_cutting_component + 
            self.gamma * signal_alignment_component -
            self.delta * transaction_cost_component +
            self.lambda_cash * cash_management_component +
            self.lambda_regime * regime_alignment_component +
            self.lambda_risk * risk_management_component
        )
        
        # Final validation and clipping with detailed debugging
        if np.isnan(total_reward) or np.isinf(total_reward):
            logger.warning(f"NaN/Inf total reward detected: {total_reward}")
            logger.warning(f"Component breakdown:")
            logger.warning(f"  alpha * profit_component: {self.alpha} * {profit_component} = {self.alpha * profit_component}")
            logger.warning(f"  beta * loss_cutting: {self.beta} * {loss_cutting_component} = {self.beta * loss_cutting_component}")
            logger.warning(f"  gamma * signal_alignment: {self.gamma} * {signal_alignment_component} = {self.gamma * signal_alignment_component}")
            logger.warning(f"  delta * transaction_cost: {self.delta} * {transaction_cost_component} = {self.delta * transaction_cost_component}")
            logger.warning(f"  lambda_cash * cash_mgmt: {self.lambda_cash} * {cash_management_component} = {self.lambda_cash * cash_management_component}")
            logger.warning(f"  lambda_regime * regime: {self.lambda_regime} * {regime_alignment_component} = {self.lambda_regime * regime_alignment_component}")
            logger.warning(f"  lambda_risk * risk_mgmt: {self.lambda_risk} * {risk_management_component} = {self.lambda_risk * risk_management_component}")
            logger.warning(f"Input metrics: realized_return={metrics.realized_return}, volatility={metrics.volatility}")
            total_reward = np.clip(metrics.realized_return, -1.0, 1.0)
        else:
            # Clip to reasonable bounds to prevent extreme rewards
            total_reward = np.clip(total_reward, -10.0, 10.0)
        
        return RewardComponents(
            # Core mathematical components
            profit_component=profit_component,
            loss_cutting_component=loss_cutting_component,
            signal_alignment_component=signal_alignment_component, 
            transaction_cost_component=transaction_cost_component,
            
            # Extended components
            cash_management_component=cash_management_component,
            regime_alignment_component=regime_alignment_component,
            risk_management_component=risk_management_component,
            
            # Raw values for analysis
            realized_return_raw=metrics.realized_return,
            unrealized_return_raw=metrics.unrealized_return, 
            volatility_normalization=max(metrics.volatility, self.min_volatility),
            total_reward=total_reward
        )
    
    def _calculate_profit_component(self, metrics: TradeMetrics) -> float:
        """
        Calculate profit component: α·(r_t/σ)
        Risk-adjusted realized returns following the mathematical formulation.
        """
        
        # Get realized return (r_t) from metrics
        realized_return = metrics.realized_return
        
        # Get volatility (σ) for normalization
        volatility = max(metrics.volatility, self.min_volatility)  # Avoid division by zero
        
        # Calculate risk-adjusted profit: r_t / σ with validation
        if volatility > 0 and not np.isnan(realized_return) and not np.isinf(realized_return):
            risk_adjusted_profit = realized_return / volatility
            # Validate result and clip extreme values
            if np.isnan(risk_adjusted_profit) or np.isinf(risk_adjusted_profit):
                logger.warning(f"NaN/Inf in profit calculation: {realized_return}/{volatility} = {risk_adjusted_profit}")
                risk_adjusted_profit = 0.0
            else:
                # Clip to prevent extreme rewards from very low volatility
                risk_adjusted_profit = np.clip(risk_adjusted_profit, -50.0, 50.0)
        else:
            risk_adjusted_profit = 0.0
            
        # Track returns for adaptive volatility if needed
        if realized_return != 0:
            self.returns_history.append(realized_return)
            # Keep only recent returns for volatility updates
            if len(self.returns_history) > self.volatility_window:
                self.returns_history.pop(0)
        
        return risk_adjusted_profit
    
    def _calculate_loss_cutting_component(self, metrics: TradeMetrics) -> float:
        """
        Calculate loss-cutting component: β·L_t  
        Rewards agent for cutting losses early (before they get too large).
        
        L_t = min(1, |u_t|/σ) if agent exits at a loss before threshold L_max, 0 otherwise
        """
        
        # Check if this is a loss-cutting trade
        unrealized_return = metrics.unrealized_return
        realized_return = metrics.realized_return
        volatility = max(metrics.volatility, self.min_volatility)
        
        # Reward if agent exits at a loss before losses get too large
        # Logic: If you sold at a loss, but the unrealized loss was worse, reward the discipline
        is_loss_cutting = (
            realized_return < 0 and  # Realized a loss (took the hit)
            abs(unrealized_return) > abs(realized_return) and  # Unrealized loss was worse than realized
            abs(unrealized_return) < self.L_max and  # Position wasn't completely hopeless
            metrics.trade_type in [TradeType.MARKET_SELL, TradeType.LIMIT_SELL, TradeType.SHORT_COVER]
        )
        
        if is_loss_cutting:
            # Reward scales with how much unrealized loss was avoided, normalized by volatility
            if volatility > 0 and not np.isnan(unrealized_return) and not np.isinf(unrealized_return):
                loss_cutting_reward = min(1.0, abs(unrealized_return) / volatility)
                if np.isnan(loss_cutting_reward) or np.isinf(loss_cutting_reward):
                    logger.warning(f"NaN/Inf in loss cutting: {abs(unrealized_return)}/{volatility}")
                    loss_cutting_reward = 0.0
            else:
                loss_cutting_reward = 0.0
            return loss_cutting_reward
        else:
            return 0.0
    
    def _calculate_signal_alignment_component(self, metrics: TradeMetrics) -> float:
        """
        Calculate signal alignment component: γ·S_t
        Rewards trades that align with predictive signals, penalizes contradictory trades.
        
        S_t = +s_t if trade direction matches signal, -s_t if contradictory
        """
        
        trade_direction = metrics.trade_direction  # +1 buy, -1 sell, 0 hold
        signal_direction = metrics.signal_direction  # +1 bullish, -1 bearish
        signal_confidence = metrics.signal_confidence  # 0-1 scale
        
        if trade_direction == 0:  # No trade, no alignment bonus/penalty
            return 0.0
        
        # Check if trade direction matches signal direction
        if (trade_direction > 0 and signal_direction > 0) or (trade_direction < 0 and signal_direction < 0):
            # Trade aligns with signal - positive reward
            return signal_confidence
        elif (trade_direction > 0 and signal_direction < 0) or (trade_direction < 0 and signal_direction > 0):
            # Trade contradicts signal - negative reward (penalty)
            return -signal_confidence
        else:
            return 0.0
    
    def _calculate_transaction_cost_component(self, metrics: TradeMetrics) -> float:
        """
        Calculate transaction cost component: δ·C_t
        Penalizes unnecessary overtrading with transaction costs.
        
        C_t = c · |position_change|
        """
        
        # Get total transaction cost
        transaction_cost = metrics.transaction_cost
        position_change = abs(metrics.position_change)
        
        # Cost penalty scales with position change magnitude
        cost_penalty = transaction_cost * position_change
        
        return cost_penalty
    
    def _calculate_cash_management_component(self, metrics: TradeMetrics) -> float:
        """
        Calculate cash management component - extends the loss-cutting discipline 
        to prioritize selling underperformers when liquidity is needed.
        
        Integrates with β·L_t by providing additional urgency-based rewards.
        """
        
        # Get cash info from metrics (backward compatibility)
        current_cash = getattr(metrics, 'current_cash', 0)
        buying_power = getattr(metrics, 'buying_power', 0)
        portfolio_value = max(metrics.portfolio_value_t, metrics.portfolio_value_t1)
        
        # No bonus if cash info not available
        if buying_power == 0 and current_cash == 0:
            return 0.0
        
        # Calculate cash urgency (0-1 scale)
        if buying_power > 0:
            target_cash_buffer = max(10000, portfolio_value * 0.1)  # 10% of portfolio or $10k minimum
            cash_urgency = max(0.0, min(1.0, 1.0 - (buying_power / target_cash_buffer)))
        else:
            cash_urgency = 1.0  # Maximum urgency when buying power is zero
        
        # Integration with loss-cutting: boost rewards for selling underperformers when cash urgent
        is_underperformer_sell = (
            metrics.trade_type in [TradeType.MARKET_SELL, TradeType.LIMIT_SELL] and
            (getattr(metrics, 'is_underperformer', False) or 
             getattr(metrics, 'unrealized_pnl_pct', 0) < -0.02)  # Losing > 2%
        )
        
        # Penalty for buying when cash urgently needed
        is_urgent_buy = (
            metrics.trade_type in [TradeType.MARKET_BUY, TradeType.LIMIT_BUY] and
            cash_urgency > 0.5
        )
        
        cash_reward = 0.0
        
        if is_underperformer_sell and cash_urgency > 0.3:
            # Reward selling underperformers when cash needed
            underperformance_factor = abs(getattr(metrics, 'unrealized_pnl_pct', 0.02))
            cash_reward = cash_urgency * underperformance_factor * abs(metrics.position_size) * 0.01
            
            # Extra boost for extreme underperformers
            if getattr(metrics, 'unrealized_pnl_pct', 0) < -0.05:  # Losing > 5%
                cash_reward *= 2.0
                
        elif is_urgent_buy:
            # Penalize buying when cash urgently needed
            cash_reward = -cash_urgency * abs(metrics.position_size) * 0.005
        
        return cash_reward
    
    def _calculate_regime_alignment_component(self, metrics: TradeMetrics) -> float:
        """
        Calculate market regime alignment component.
        Rewards trades that align with detected market conditions.
        """
        
        regime_alignment = getattr(metrics, 'regime_alignment', 0.0)  # -1 to 1
        position_size = abs(metrics.position_size)
        
        # Small bonus for regime-aligned trades
        regime_bonus = regime_alignment * position_size * 0.0005
        
        return regime_bonus
    
    def _calculate_risk_management_component(self, metrics: TradeMetrics) -> float:
        """
        Calculate risk management component.
        Penalizes excessive concentration and leverage.
        """
        
        risk_penalty = 0.0
        
        # Concentration risk penalty
        concentration = getattr(metrics, 'position_concentration', 0.0)
        if concentration > 0.15:  # > 15% concentration
            risk_penalty += (concentration - 0.15) * 0.5
        
        # Leverage penalty
        leverage = getattr(metrics, 'leverage', 1.0)
        if leverage > 3.0:  # > 3x leverage
            risk_penalty += (leverage - 3.0) * 0.1
        
        # Drawdown penalty
        drawdown = getattr(metrics, 'drawdown', 0.0)
        if drawdown > 0.1:  # > 10% drawdown
            risk_penalty += drawdown * 0.3
        
        return -risk_penalty  # Return negative as this is a penalty
    
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
    
    def _calculate_cash_management_bonus(self, metrics: TradeMetrics) -> float:
        """
        Calculate cash management bonus/penalty to prioritize selling underperformers when cash is low.
        This integrates cash management into the RL reward system.
        """
        
        # No bonus if cash info not available
        if metrics.buying_power == 0 and metrics.current_cash == 0:
            return 0.0
        
        # Calculate cash urgency (0-1 scale)
        # Higher when buying power is low or cash is negative
        if metrics.buying_power > 0:
            # Normal case: base urgency on buying power level
            target_cash_buffer = max(10000, metrics.portfolio_value_t * 0.1)  # 10% of portfolio or $10k minimum
            cash_ratio = metrics.buying_power / target_cash_buffer
            cash_urgency = max(0.0, min(1.0, 1.0 - cash_ratio))  # Higher urgency when cash is low
        else:
            # Emergency case: negative buying power or very low cash
            cash_urgency = 1.0  # Maximum urgency
        
        # Update metrics for use by other components
        metrics.cash_urgency = cash_urgency
        
        # Determine if this is a sell trade for an underperformer
        is_beneficial_sell = (
            metrics.trade_type in [TradeType.MARKET_SELL, TradeType.LIMIT_SELL] and
            (metrics.is_underperformer or metrics.unrealized_pnl_pct < -0.02)  # Losing > 2%
        )
        
        # Determine if this is a buy trade when cash is urgently needed
        is_problematic_buy = (
            metrics.trade_type in [TradeType.MARKET_BUY, TradeType.LIMIT_BUY] and
            cash_urgency > 0.5  # High cash urgency
        )
        
        # Calculate bonus/penalty based on cash management needs
        cash_management_reward = 0.0
        
        if is_beneficial_sell and cash_urgency > 0.3:
            # REWARD selling underperformers when cash is needed
            # Higher reward for worse-performing positions and higher cash urgency
            underperformance_factor = max(0.1, abs(metrics.unrealized_pnl_pct))  # How much it's losing
            cash_management_reward = (
                cash_urgency * underperformance_factor * abs(metrics.position_size) * 0.01
            )
            
            # Extra bonus for extreme underperformers
            if metrics.unrealized_pnl_pct < -0.05:  # Losing > 5%
                cash_management_reward *= 2.0
                
        elif is_problematic_buy:
            # PENALIZE buying when cash is urgently needed
            cash_management_reward = -cash_urgency * abs(metrics.position_size) * 0.005
        
        # Special bonus for short selling extreme underperformers when cash is needed
        if (metrics.trade_type == TradeType.SHORT_SELL and 
            metrics.unrealized_pnl_pct < -0.06 and  # Very poor performance
            cash_urgency > 0.5):
            # Double reward for shorting extreme underperformers
            cash_management_reward = cash_urgency * abs(metrics.unrealized_pnl_pct) * abs(metrics.position_size) * 0.02
        
        return cash_management_reward
    
    def get_reward_breakdown(self, metrics: TradeMetrics) -> Dict[str, float]:
        """Get detailed breakdown of mathematically-principled reward components."""
        
        components = self.calculate_reward(metrics)
        
        return {
            # Core mathematical components (weighted)
            'profit_component': self.alpha * components.profit_component,
            'loss_cutting_component': self.beta * components.loss_cutting_component,
            'signal_alignment_component': self.gamma * components.signal_alignment_component,
            'transaction_cost_component': -self.delta * components.transaction_cost_component,  # Note: negative since it's a penalty
            
            # Extended components (weighted)  
            'cash_management_component': self.lambda_cash * components.cash_management_component,
            'regime_alignment_component': self.lambda_regime * components.regime_alignment_component,
            'risk_management_component': self.lambda_risk * components.risk_management_component,
            
            # Raw components (unweighted for analysis)
            'profit_raw': components.profit_component,
            'loss_cutting_raw': components.loss_cutting_component,
            'signal_alignment_raw': components.signal_alignment_component,
            'transaction_cost_raw': components.transaction_cost_component,
            
            # Analysis metrics
            'realized_return_raw': components.realized_return_raw,
            'unrealized_return_raw': components.unrealized_return_raw,
            'volatility_normalization': components.volatility_normalization,
            
            # Total
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