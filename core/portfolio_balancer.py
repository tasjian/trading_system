#!/usr/bin/env python3
"""
Intelligent Portfolio Balancer

Implements comprehensive portfolio rebalancing with proper order type selection,
risk management integration, and position-aware decision making.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio

from tools.alpaca_client import alpaca_client
from config.settings import settings
from core.oco_order_manager import oco_manager, OCOType

logger = logging.getLogger(__name__)

class PositionAction(Enum):
    """Portfolio position actions."""
    HOLD = "hold"
    BUY = "buy"
    SELL = "sell"
    SHORT = "sell_short"
    COVER = "buy_to_cover"
    REDUCE = "reduce"
    CLOSE = "close"

class OrderUrgency(Enum):
    """Order execution urgency levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PositionAnalysis:
    """Analysis of current position vs target."""
    symbol: str
    current_quantity: float
    current_value: float
    current_weight: float
    target_weight: float
    target_value: float
    target_quantity: float
    deviation: float
    action_needed: PositionAction
    urgency: OrderUrgency
    reasoning: str
    risk_score: float = 0.0
    sector: str = "Unknown"
    
@dataclass
class RebalanceDecision:
    """Portfolio rebalancing decision."""
    symbol: str
    action: PositionAction
    quantity: float
    order_type: str  # "market", "limit", "stop_limit", "oco_bracket", "oco_breakout"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    urgency: OrderUrgency = OrderUrgency.MEDIUM
    reasoning: str = ""
    confidence: float = 0.5
    risk_adjustments: Dict[str, float] = field(default_factory=dict)
    
    # OCO-specific parameters
    use_oco: bool = False
    take_profit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    oco_type: Optional[str] = None  # "bracket", "breakout"

class IntelligentPortfolioBalancer:
    """
    Intelligent portfolio balancer that properly handles buy/sell/short/limit orders
    based on current positions, target allocations, and risk management.
    """
    
    def __init__(self):
        """Initialize portfolio balancer."""
        self.current_positions = {}
        self.target_allocation = {}
        self.risk_limits = {
            'max_position_size': settings.max_position_size,
            'max_daily_trades': settings.max_daily_trades,
            'max_sector_allocation': settings.max_sector_allocation,
            'stop_loss_percent': settings.stop_loss_percent
        }
        
        # Rebalancing thresholds
        self.rebalance_threshold = settings.min_rebalance_threshold
        self.significant_deviation_threshold = 0.10  # 10%
        self.critical_deviation_threshold = 0.20     # 20%
        
        # Order execution preferences
        self.prefer_limit_orders = True
        self.limit_order_buffer = 0.002  # 0.2% buffer for limit orders
        self.market_order_threshold = 0.15  # Use market orders for >15% deviations
        
    async def analyze_portfolio_balance(self, 
                                      target_allocation: Dict[str, float],
                                      current_signals: List[Dict] = None) -> List[PositionAnalysis]:
        """
        Analyze current portfolio vs target allocation and determine needed actions.
        
        Args:
            target_allocation: Dict of {symbol: target_weight}
            current_signals: Optional list of current trading signals
            
        Returns:
            List of position analyses with recommended actions
        """
        logger.info("🔍 Analyzing portfolio balance...")
        
        # Get current portfolio state
        portfolio_data = await self._get_current_portfolio()
        if not portfolio_data:
            logger.error("Failed to get current portfolio data")
            return []
        
        current_positions = portfolio_data['positions']
        portfolio_value = portfolio_data['total_value']
        cash_available = portfolio_data['cash']
        
        analyses = []
        
        # Analyze each target position
        for symbol, target_weight in target_allocation.items():
            current_pos = current_positions.get(symbol, {
                'quantity': 0.0, 'market_value': 0.0, 'side': 'long'
            })
            
            analysis = await self._analyze_single_position(
                symbol, target_weight, current_pos, portfolio_value, current_signals
            )
            analyses.append(analysis)
        
        # Analyze positions to be closed (not in target allocation)
        for symbol, pos_data in current_positions.items():
            if symbol not in target_allocation and abs(pos_data['quantity']) > 0:
                analysis = await self._analyze_position_to_close(symbol, pos_data, portfolio_value)
                analyses.append(analysis)
        
        # Sort by urgency and deviation size
        analyses.sort(key=lambda x: (x.urgency.value, abs(x.deviation)), reverse=True)
        
        logger.info(f"📊 Portfolio analysis complete: {len(analyses)} positions analyzed")
        return analyses
    
    async def generate_rebalancing_orders(self, 
                                        position_analyses: List[PositionAnalysis],
                                        max_orders: int = 10) -> List[RebalanceDecision]:
        """
        Generate specific rebalancing orders based on position analyses.
        
        Args:
            position_analyses: List of position analyses
            max_orders: Maximum number of orders to generate
            
        Returns:
            List of rebalancing decisions with specific order instructions
        """
        logger.info(f"🎯 Generating rebalancing orders (max: {max_orders})...")
        
        orders = []
        risk_budget_used = 0.0
        
        for analysis in position_analyses[:max_orders]:
            # Skip if no action needed
            if analysis.action_needed == PositionAction.HOLD:
                continue
            
            # Generate order decision
            order_decision = await self._create_order_decision(analysis, risk_budget_used)
            if order_decision:
                orders.append(order_decision)
                risk_budget_used += order_decision.risk_adjustments.get('risk_weight', 0.0)
        
        logger.info(f"✅ Generated {len(orders)} rebalancing orders")
        return orders
    
    async def _analyze_single_position(self, 
                                     symbol: str, 
                                     target_weight: float,
                                     current_pos: Dict,
                                     portfolio_value: float,
                                     signals: List[Dict] = None) -> PositionAnalysis:
        """Analyze a single position for rebalancing needs."""
        
        # Current position metrics
        current_quantity = float(current_pos.get('quantity', 0))
        current_value = float(current_pos.get('market_value', 0))
        current_weight = current_value / portfolio_value if portfolio_value > 0 else 0.0
        
        # Target metrics
        target_value = target_weight * portfolio_value
        
        # Get current price for quantity calculations
        current_price = await self._get_current_price(symbol)
        if not current_price or current_price <= 0:
            logger.warning(f"Could not get valid price for {symbol}")
            current_price = current_value / max(abs(current_quantity), 1)
        
        target_quantity = target_value / current_price if current_price > 0 else 0
        
        # Calculate deviation
        weight_deviation = target_weight - current_weight
        quantity_deviation = target_quantity - current_quantity
        
        # Determine action needed
        action_needed, urgency = self._determine_position_action(
            weight_deviation, current_quantity, target_quantity, symbol, signals
        )
        
        # Get sector information
        sector = await self._get_symbol_sector(symbol)
        
        # Calculate risk score
        risk_score = self._calculate_position_risk(symbol, target_weight, weight_deviation)
        
        # Generate reasoning
        reasoning = self._generate_position_reasoning(
            symbol, current_weight, target_weight, weight_deviation, action_needed
        )
        
        return PositionAnalysis(
            symbol=symbol,
            current_quantity=current_quantity,
            current_value=current_value,
            current_weight=current_weight,
            target_weight=target_weight,
            target_value=target_value,
            target_quantity=target_quantity,
            deviation=weight_deviation,
            action_needed=action_needed,
            urgency=urgency,
            reasoning=reasoning,
            risk_score=risk_score,
            sector=sector
        )
    
    def _determine_position_action(self, 
                                 weight_deviation: float,
                                 current_quantity: float,
                                 target_quantity: float,
                                 symbol: str,
                                 signals: List[Dict] = None) -> Tuple[PositionAction, OrderUrgency]:
        """Determine what action is needed for a position."""
        
        # Check if deviation is significant enough
        abs_deviation = abs(weight_deviation)
        
        if abs_deviation < self.rebalance_threshold:
            return PositionAction.HOLD, OrderUrgency.LOW
        
        # Determine urgency based on deviation size
        if abs_deviation >= self.critical_deviation_threshold:
            urgency = OrderUrgency.CRITICAL
        elif abs_deviation >= self.significant_deviation_threshold:
            urgency = OrderUrgency.HIGH
        else:
            urgency = OrderUrgency.MEDIUM
        
        # Determine action based on current vs target quantities
        if current_quantity == 0:  # No current position
            if target_quantity > 0:
                return PositionAction.BUY, urgency
            elif target_quantity < 0:
                return PositionAction.SHORT, urgency
            else:
                return PositionAction.HOLD, OrderUrgency.LOW
        
        elif current_quantity > 0:  # Current long position
            if target_quantity > current_quantity:
                return PositionAction.BUY, urgency  # Add to position
            elif target_quantity < current_quantity:
                if target_quantity <= 0:
                    return PositionAction.CLOSE, urgency  # Close long position
                else:
                    return PositionAction.REDUCE, urgency  # Reduce position
            else:
                return PositionAction.HOLD, OrderUrgency.LOW
        
        elif current_quantity < 0:  # Current short position
            if target_quantity > current_quantity:
                if target_quantity >= 0:
                    return PositionAction.COVER, urgency  # Cover short position
                else:
                    return PositionAction.REDUCE, urgency  # Reduce short position
            elif target_quantity < current_quantity:
                return PositionAction.SHORT, urgency  # Add to short position
            else:
                return PositionAction.HOLD, OrderUrgency.LOW
        
        return PositionAction.HOLD, OrderUrgency.LOW
    
    async def _create_order_decision(self, 
                                   analysis: PositionAnalysis,
                                   current_risk_budget: float) -> Optional[RebalanceDecision]:
        """Create a specific order decision based on position analysis."""
        
        # Calculate order quantity
        quantity_needed = abs(analysis.target_quantity - analysis.current_quantity)
        
        # Apply risk adjustments
        risk_adjusted_quantity = await self._apply_risk_adjustments(
            analysis.symbol, quantity_needed, analysis.risk_score, current_risk_budget
        )
        
        if risk_adjusted_quantity <= 0:
            logger.debug(f"Risk adjustment resulted in zero quantity for {analysis.symbol}")
            return None
        
        # Determine order type and prices
        order_type, limit_price, stop_price = await self._determine_order_execution_method(
            analysis.symbol, analysis.action_needed, analysis.urgency
        )
        
        # Map position action to order side
        order_action = self._map_position_action_to_order_side(
            analysis.action_needed, analysis.current_quantity
        )
        
        # Calculate confidence based on analysis
        confidence = self._calculate_order_confidence(analysis)
        
        # Determine if OCO should be used
        use_oco, oco_type, take_profit_price, stop_loss_price = await self._evaluate_oco_usage(
            analysis, order_action, confidence
        )
        
        # Update order type if using OCO
        if use_oco:
            order_type = f"oco_{oco_type}"
        
        # Generate enhanced reasoning
        reasoning = f"{analysis.reasoning} | Rebalance: {analysis.deviation:.1%} deviation"
        if use_oco:
            reasoning += f" | OCO {oco_type}: TP=${take_profit_price:.2f}, SL=${stop_loss_price:.2f}"
        
        return RebalanceDecision(
            symbol=analysis.symbol,
            action=order_action,
            quantity=risk_adjusted_quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            urgency=analysis.urgency,
            reasoning=reasoning,
            confidence=confidence,
            risk_adjustments={
                'risk_weight': analysis.risk_score,
                'original_quantity': quantity_needed,
                'risk_adjusted_quantity': risk_adjusted_quantity
            },
            use_oco=use_oco,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            oco_type=oco_type
        )
    
    def _map_position_action_to_order_side(self, 
                                         action: PositionAction, 
                                         current_quantity: float) -> PositionAction:
        """Map position action to specific order side."""
        
        if action == PositionAction.BUY:
            return PositionAction.BUY
        elif action == PositionAction.SELL or action == PositionAction.REDUCE:
            if current_quantity > 0:
                return PositionAction.SELL  # Sell long position
            else:
                return PositionAction.COVER  # Cover short position
        elif action == PositionAction.SHORT:
            return PositionAction.SHORT
        elif action == PositionAction.COVER:
            return PositionAction.COVER
        elif action == PositionAction.CLOSE:
            if current_quantity > 0:
                return PositionAction.SELL  # Close long position
            else:
                return PositionAction.COVER  # Close short position
        else:
            return PositionAction.HOLD
    
    async def _determine_order_execution_method(self, 
                                              symbol: str, 
                                              action: PositionAction,
                                              urgency: OrderUrgency) -> Tuple[str, Optional[float], Optional[float]]:
        """Determine order type and execution prices."""
        
        current_price = await self._get_current_price(symbol)
        if not current_price:
            return "market", None, None
        
        # Use market orders for critical urgency or large deviations
        if urgency == OrderUrgency.CRITICAL:
            return "market", None, None
        
        # Use limit orders with appropriate pricing
        if self.prefer_limit_orders and urgency in [OrderUrgency.LOW, OrderUrgency.MEDIUM]:
            if action in [PositionAction.BUY, PositionAction.COVER]:
                # Buy orders: limit slightly above current price
                limit_price = current_price * (1 + self.limit_order_buffer)
                return "limit", limit_price, None
            elif action in [PositionAction.SELL, PositionAction.SHORT]:
                # Sell orders: limit slightly below current price
                limit_price = current_price * (1 - self.limit_order_buffer)
                return "limit", limit_price, None
        
        # Default to market orders
        return "market", None, None
    
    async def _apply_risk_adjustments(self, 
                                    symbol: str, 
                                    quantity: float,
                                    risk_score: float,
                                    current_risk_budget: float) -> float:
        """Apply risk management adjustments to order quantity."""
        
        # Start with requested quantity
        adjusted_quantity = quantity
        
        # Apply position size limits
        portfolio_value = (await self._get_current_portfolio())['total_value']
        current_price = await self._get_current_price(symbol)
        
        if portfolio_value > 0 and current_price > 0:
            position_value = adjusted_quantity * current_price
            position_weight = position_value / portfolio_value
            
            # Enforce maximum position size
            if position_weight > self.risk_limits['max_position_size']:
                max_quantity = (self.risk_limits['max_position_size'] * portfolio_value) / current_price
                adjusted_quantity = min(adjusted_quantity, max_quantity)
                logger.info(f"Position size limited for {symbol}: {quantity} -> {adjusted_quantity}")
        
        # Apply risk score adjustments
        if risk_score > 0.7:  # High risk
            adjusted_quantity *= 0.7  # Reduce position size by 30%
        elif risk_score > 0.5:  # Medium risk
            adjusted_quantity *= 0.85  # Reduce position size by 15%
        
        # Ensure minimum trade size
        min_trade_value = 100  # $100 minimum trade
        if current_price > 0 and adjusted_quantity * current_price < min_trade_value:
            adjusted_quantity = min_trade_value / current_price
        
        return max(0, adjusted_quantity)
    
    async def _evaluate_oco_usage(self, analysis: PositionAnalysis, order_action: PositionAction, 
                                confidence: float) -> Tuple[bool, Optional[str], Optional[float], Optional[float]]:
        """Evaluate whether to use OCO orders for this rebalancing decision."""
        
        # Check if OCO is enabled globally
        if not settings.oco_enabled:
            return False, None, None, None
        
        # Don't use OCO for HOLD actions
        if order_action == PositionAction.HOLD:
            return False, None, None, None
        
        # Use OCO for significant positions (relaxed thresholds for more usage)
        use_oco = (
            settings.oco_enabled and
            confidence >= 0.5 and  # Medium confidence trades (lowered from 0.7)
            abs(analysis.deviation) >= 0.05 and  # 5% deviation (lowered from 10%)
            analysis.target_weight >= 0.02  # 2% position size (lowered from 5%)
        )
        
        if not use_oco:
            return False, None, None, None
        
        # Get current price for OCO calculations
        try:
            current_price = await self._get_current_price(analysis.symbol)
            if not current_price:
                return False, None, None, None
        except:
            return False, None, None, None
        
        # Determine OCO type and calculate prices
        if order_action in [PositionAction.BUY, PositionAction.SELL]:
            # Bracket OCO for position entry/exit
            oco_type = "bracket"
            
            if order_action == PositionAction.BUY:
                take_profit_price = current_price * (1 + settings.oco_default_take_profit_percent)
                stop_loss_price = current_price * (1 - settings.oco_default_stop_loss_percent)
            else:  # SELL
                take_profit_price = current_price * (1 - settings.oco_default_take_profit_percent)
                stop_loss_price = current_price * (1 + settings.oco_default_stop_loss_percent)
            
            return True, oco_type, take_profit_price, stop_loss_price
        
        elif analysis.current_quantity == 0 and settings.oco_breakout_enabled:
            # Consider breakout OCO for new positions with high volatility
            oco_type = "breakout"
            
            # Calculate breakout levels based on recent volatility
            try:
                upper_breakout = current_price * (1 + settings.oco_default_take_profit_percent)
                lower_breakout = current_price * (1 - settings.oco_default_take_profit_percent)
                return True, oco_type, upper_breakout, lower_breakout
            except:
                return False, None, None, None
        
        return False, None, None, None
    
    def _calculate_position_risk(self, symbol: str, target_weight: float, deviation: float) -> float:
        """Calculate risk score for a position."""
        
        # Base risk from position size
        size_risk = min(target_weight / self.risk_limits['max_position_size'], 1.0)
        
        # Risk from deviation size
        deviation_risk = min(abs(deviation) / self.critical_deviation_threshold, 1.0)
        
        # Combine risks
        combined_risk = (size_risk * 0.6) + (deviation_risk * 0.4)
        
        return min(combined_risk, 1.0)
    
    def _calculate_order_confidence(self, analysis: PositionAnalysis) -> float:
        """Calculate confidence level for an order."""
        
        # Base confidence from deviation size
        deviation_confidence = min(abs(analysis.deviation) / self.significant_deviation_threshold, 1.0)
        
        # Adjust for urgency
        urgency_multiplier = {
            OrderUrgency.CRITICAL: 1.0,
            OrderUrgency.HIGH: 0.9,
            OrderUrgency.MEDIUM: 0.8,
            OrderUrgency.LOW: 0.6
        }
        
        confidence = deviation_confidence * urgency_multiplier.get(analysis.urgency, 0.5)
        return min(max(confidence, 0.1), 0.95)  # Clamp between 0.1 and 0.95
    
    def _generate_position_reasoning(self, 
                                   symbol: str, 
                                   current_weight: float,
                                   target_weight: float,
                                   deviation: float,
                                   action: PositionAction) -> str:
        """Generate human-readable reasoning for position action."""
        
        deviation_pct = deviation * 100
        current_pct = current_weight * 100
        target_pct = target_weight * 100
        
        if action == PositionAction.HOLD:
            return f"Position balanced: {current_pct:.1f}% (target: {target_pct:.1f}%)"
        elif action == PositionAction.BUY:
            return f"Underweight by {abs(deviation_pct):.1f}%: {current_pct:.1f}% -> {target_pct:.1f}%"
        elif action in [PositionAction.SELL, PositionAction.REDUCE]:
            return f"Overweight by {abs(deviation_pct):.1f}%: {current_pct:.1f}% -> {target_pct:.1f}%"
        elif action == PositionAction.SHORT:
            return f"Short position needed: target {target_pct:.1f}%"
        elif action == PositionAction.COVER:
            return f"Cover short position: current {current_pct:.1f}% -> target {target_pct:.1f}%"
        elif action == PositionAction.CLOSE:
            return f"Close position: {current_pct:.1f}% -> 0% (not in target allocation)"
        else:
            return f"Portfolio rebalancing required"
    
    async def _analyze_position_to_close(self, 
                                       symbol: str, 
                                       pos_data: Dict,
                                       portfolio_value: float) -> PositionAnalysis:
        """Analyze a position that should be closed (not in target allocation)."""
        
        current_quantity = float(pos_data.get('quantity', 0))
        current_value = float(pos_data.get('market_value', 0))
        current_weight = current_value / portfolio_value if portfolio_value > 0 else 0.0
        
        # Determine urgency based on position size
        if abs(current_weight) >= self.critical_deviation_threshold:
            urgency = OrderUrgency.CRITICAL
        elif abs(current_weight) >= self.significant_deviation_threshold:
            urgency = OrderUrgency.HIGH
        else:
            urgency = OrderUrgency.MEDIUM
        
        action_needed = PositionAction.CLOSE
        sector = await self._get_symbol_sector(symbol)
        risk_score = min(abs(current_weight) / self.risk_limits['max_position_size'], 1.0)
        
        reasoning = f"Close position: {current_weight*100:.1f}% (not in target allocation)"
        
        return PositionAnalysis(
            symbol=symbol,
            current_quantity=current_quantity,
            current_value=current_value,
            current_weight=current_weight,
            target_weight=0.0,
            target_value=0.0,
            target_quantity=0.0,
            deviation=-current_weight,  # Negative because we want to go to 0
            action_needed=action_needed,
            urgency=urgency,
            reasoning=reasoning,
            risk_score=risk_score,
            sector=sector
        )
    
    async def _get_current_portfolio(self) -> Dict:
        """Get current portfolio data from Alpaca."""
        try:
            account_info = alpaca_client.get_account_info()
            positions = alpaca_client.get_positions()
            
            portfolio_data = {
                'total_value': float(account_info.get('portfolio_value', 0)),
                'cash': float(account_info.get('cash', 0)),
                'positions': {}
            }
            
            for pos in positions:
                symbol = pos['symbol']
                portfolio_data['positions'][symbol] = {
                    'quantity': float(pos.get('qty', 0)),
                    'market_value': float(pos.get('market_value', 0)),
                    'side': pos.get('side', 'long'),
                    'avg_cost': float(pos.get('avg_entry_price', 0))
                }
            
            return portfolio_data
            
        except Exception as e:
            logger.error(f"Failed to get current portfolio: {e}")
            return {'total_value': 0, 'cash': 0, 'positions': {}}
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            # Use Alpaca client to get current price
            price = alpaca_client.get_current_price(symbol)
            return float(price) if price else None
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            return None
    
    async def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector information for a symbol."""
        # This would typically integrate with a data provider
        # For now, return "Unknown" - can be enhanced later
        return "Unknown"
    
    async def _get_daily_trades_count(self) -> int:
        """Get number of trades executed today."""
        try:
            from datetime import date
            today = date.today().isoformat()
            
            orders = alpaca_client.get_orders(status="filled", limit=100)
            daily_trades = 0
            
            for order in orders:
                filled_at = order.get('filled_at')
                if filled_at:
                    # Convert timestamp to string if needed
                    filled_at_str = str(filled_at)
                    if filled_at_str.startswith(today):
                        daily_trades += 1
            
            return daily_trades
        except Exception as e:
            logger.error(f"Failed to get daily trades count: {e}")
            return 0  # Return 0 to allow trading if count fails

# Global instance
portfolio_balancer = IntelligentPortfolioBalancer()

# Convenience functions
async def analyze_portfolio_balance(target_allocation: Dict[str, float], 
                                  current_signals: List[Dict] = None) -> List[PositionAnalysis]:
    """Analyze portfolio balance against target allocation."""
    return await portfolio_balancer.analyze_portfolio_balance(target_allocation, current_signals)

async def generate_rebalancing_orders(position_analyses: List[PositionAnalysis], 
                                    max_orders: int = 10) -> List[RebalanceDecision]:
    """Generate rebalancing orders from position analyses."""
    return await portfolio_balancer.generate_rebalancing_orders(position_analyses, max_orders)

__all__ = [
    'PositionAction', 'OrderUrgency', 'PositionAnalysis', 'RebalanceDecision',
    'IntelligentPortfolioBalancer', 'portfolio_balancer',
    'analyze_portfolio_balance', 'generate_rebalancing_orders'
]