#!/usr/bin/env python3
"""
Enhanced Position Management System

Intelligent position management with dynamic hold/add/reduce/close decisions
based on risk metrics, performance tracking, and market conditions.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio

from core.portfolio_balancer import PositionAction, OrderUrgency
from core.signal_processor import SignalType, MarketContext, MarketRegime
from tools.alpaca_client import alpaca_client
from config.settings import settings

logger = logging.getLogger(__name__)

class PositionStatus(Enum):
    """Position status classifications."""
    HEALTHY = "healthy"
    UNDERPERFORMING = "underperforming"
    AT_RISK = "at_risk"
    CRITICAL = "critical"
    PROFITABLE = "profitable"
    TAKE_PROFIT = "take_profit"

class RiskLevel(Enum):
    """Risk level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PositionMetrics:
    """Comprehensive position performance metrics."""
    symbol: str
    current_quantity: float
    current_value: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    
    # Risk metrics
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    position_risk_score: float = 0.0
    volatility: float = 0.0
    beta: float = 1.0
    
    # Performance tracking
    max_profit: float = 0.0
    max_drawdown: float = 0.0
    days_held: int = 0
    volume_avg_ratio: float = 0.0
    
    # Technical indicators
    rsi: float = 50.0
    moving_avg_20: float = 0.0
    moving_avg_50: float = 0.0
    support_level: float = 0.0
    resistance_level: float = 0.0
    
    # Status
    status: PositionStatus = PositionStatus.HEALTHY
    risk_level: RiskLevel = RiskLevel.MEDIUM

@dataclass
class PositionDecision:
    """Position management decision."""
    symbol: str
    action: PositionAction
    quantity: float
    reasoning: str
    confidence: float
    urgency: OrderUrgency
    risk_level: RiskLevel
    
    # Stop loss and take profit adjustments
    new_stop_loss: Optional[float] = None
    new_take_profit: Optional[float] = None
    
    # Order execution details
    order_type: str = "market"
    limit_price: Optional[float] = None
    
    # Risk management
    max_loss_limit: float = 0.0
    position_size_adjustment: float = 1.0
    
    timestamp: datetime = field(default_factory=datetime.now)

class IntelligentPositionManager:
    """
    Intelligent position manager that makes dynamic decisions about
    holding, adding, reducing, or closing positions based on comprehensive analysis.
    """
    
    def __init__(self):
        """Initialize position manager."""
        
        # Risk management parameters
        self.stop_loss_percent = settings.stop_loss_percent
        self.max_position_risk = settings.max_portfolio_risk
        self.max_drawdown_limit = 0.15  # 15% max drawdown per position
        
        # Performance thresholds
        self.take_profit_threshold = 0.20  # 20% profit target
        self.underperformance_threshold = -0.05  # -5% underperformance flag
        self.critical_loss_threshold = -0.12  # -12% critical loss threshold
        
        # Position sizing parameters
        self.max_add_frequency = 3  # Max times to add to a position
        self.add_on_dip_threshold = -0.03  # Add when position drops 3%
        self.reduce_on_profit_threshold = 0.15  # Reduce when position gains 15%
        
        # Technical analysis thresholds
        self.oversold_rsi = 30
        self.overbought_rsi = 70
        self.trend_confirmation_days = 5
    
    async def analyze_positions(self, 
                              current_positions: Dict[str, Dict],
                              market_context: Optional[MarketContext] = None) -> List[PositionMetrics]:
        """
        Analyze all current positions and generate comprehensive metrics.
        
        Args:
            current_positions: Current portfolio positions
            market_context: Current market conditions
            
        Returns:
            List of position metrics with analysis
        """
        logger.info(f"📊 Analyzing {len(current_positions)} positions...")
        
        if not market_context:
            market_context = await self._get_market_context()
        
        position_metrics = []
        
        for symbol, position_data in current_positions.items():
            if abs(position_data.get('quantity', 0)) == 0:
                continue  # Skip empty positions
            
            metrics = await self._analyze_single_position(symbol, position_data, market_context)
            if metrics:
                position_metrics.append(metrics)
        
        # Sort by risk level and PnL
        position_metrics.sort(key=lambda x: (x.risk_level.value, x.unrealized_pnl_percent))
        
        logger.info(f"✅ Position analysis complete: {len(position_metrics)} positions evaluated")
        return position_metrics
    
    async def generate_position_decisions(self, 
                                        position_metrics: List[PositionMetrics],
                                        market_context: Optional[MarketContext] = None) -> List[PositionDecision]:
        """
        Generate position management decisions based on metrics analysis.
        
        Args:
            position_metrics: List of position metrics
            market_context: Current market conditions
            
        Returns:
            List of position management decisions
        """
        logger.info(f"🎯 Generating position decisions for {len(position_metrics)} positions...")
        
        if not market_context:
            market_context = await self._get_market_context()
        
        decisions = []
        portfolio_risk_budget = await self._calculate_portfolio_risk_budget()
        
        for metrics in position_metrics:
            decision = await self._generate_single_position_decision(
                metrics, market_context, portfolio_risk_budget
            )
            
            if decision and decision.action != PositionAction.HOLD:
                decisions.append(decision)
        
        # Prioritize decisions by urgency and risk
        decisions.sort(key=lambda x: (x.urgency.value, x.risk_level.value), reverse=True)
        
        logger.info(f"✅ Generated {len(decisions)} position decisions")
        return decisions
    
    async def _analyze_single_position(self, 
                                     symbol: str, 
                                     position_data: Dict,
                                     market_context: MarketContext) -> Optional[PositionMetrics]:
        """Analyze a single position comprehensively."""
        
        try:
            # Basic position data
            quantity = float(position_data.get('quantity', 0))
            if quantity == 0:
                return None
            
            market_value = float(position_data.get('market_value', 0))
            avg_entry_price = float(position_data.get('avg_entry_price', 0))
            
            # Get current market data
            current_price = await self._get_current_price(symbol)
            if not current_price:
                logger.warning(f"Could not get current price for {symbol}")
                return None
            
            # Calculate PnL
            unrealized_pnl = market_value - (quantity * avg_entry_price)
            unrealized_pnl_percent = unrealized_pnl / abs(quantity * avg_entry_price) if avg_entry_price > 0 else 0.0
            
            # Get technical indicators
            technical_data = await self._get_technical_indicators(symbol)
            
            # Calculate risk metrics
            volatility = technical_data.get('volatility', 0.02)
            position_risk_score = self._calculate_position_risk_score(
                symbol, unrealized_pnl_percent, volatility, quantity, market_value
            )
            
            # Determine position status
            status = self._determine_position_status(unrealized_pnl_percent, position_risk_score, technical_data)
            risk_level = self._determine_risk_level(position_risk_score, unrealized_pnl_percent, status)
            
            # Calculate stop loss and take profit levels
            stop_loss_price, take_profit_price = self._calculate_stop_take_levels(
                symbol, current_price, avg_entry_price, quantity, volatility
            )
            
            # Get performance tracking data
            performance_data = await self._get_position_performance_data(symbol, position_data)
            
            return PositionMetrics(
                symbol=symbol,
                current_quantity=quantity,
                current_value=market_value,
                avg_entry_price=avg_entry_price,
                current_price=current_price,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_percent=unrealized_pnl_percent,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                position_risk_score=position_risk_score,
                volatility=volatility,
                beta=technical_data.get('beta', 1.0),
                max_profit=performance_data.get('max_profit', 0.0),
                max_drawdown=performance_data.get('max_drawdown', 0.0),
                days_held=performance_data.get('days_held', 0),
                volume_avg_ratio=technical_data.get('volume_ratio', 1.0),
                rsi=technical_data.get('rsi', 50.0),
                moving_avg_20=technical_data.get('ma_20', current_price),
                moving_avg_50=technical_data.get('ma_50', current_price),
                support_level=technical_data.get('support', current_price * 0.95),
                resistance_level=technical_data.get('resistance', current_price * 1.05),
                status=status,
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze position {symbol}: {e}")
            return None
    
    async def _generate_single_position_decision(self, 
                                               metrics: PositionMetrics,
                                               market_context: MarketContext,
                                               portfolio_risk_budget: float) -> Optional[PositionDecision]:
        """Generate decision for a single position."""
        
        symbol = metrics.symbol
        current_quantity = metrics.current_quantity
        unrealized_pnl_pct = metrics.unrealized_pnl_percent
        
        # Critical loss - immediate action required
        if unrealized_pnl_pct <= self.critical_loss_threshold:
            return await self._create_emergency_close_decision(metrics, "Critical loss threshold reached")
        
        # Stop loss triggered
        if metrics.stop_loss_price and metrics.current_price <= metrics.stop_loss_price:
            return await self._create_stop_loss_decision(metrics, "Stop loss triggered")
        
        # Take profit opportunity
        if metrics.take_profit_price and metrics.current_price >= metrics.take_profit_price:
            return await self._create_take_profit_decision(metrics, "Take profit target reached")
        
        # Risk-based decisions
        if metrics.risk_level == RiskLevel.CRITICAL:
            return await self._create_risk_reduction_decision(metrics, market_context)
        
        # Performance-based decisions
        if metrics.status == PositionStatus.UNDERPERFORMING:
            return await self._create_underperformance_decision(metrics, market_context)
        
        # Opportunity-based decisions (add to winners, trim losers)
        if metrics.status == PositionStatus.PROFITABLE:
            return await self._create_profit_management_decision(metrics, market_context)
        
        # Technical analysis decisions
        technical_decision = await self._create_technical_analysis_decision(metrics, market_context)
        if technical_decision:
            return technical_decision
        
        # Market regime adjustments
        regime_decision = await self._create_regime_based_decision(metrics, market_context)
        if regime_decision:
            return regime_decision
        
        # Default: hold position
        return None
    
    async def _create_emergency_close_decision(self, metrics: PositionMetrics, reason: str) -> PositionDecision:
        """Create emergency position close decision."""
        
        return PositionDecision(
            symbol=metrics.symbol,
            action=PositionAction.CLOSE,
            quantity=abs(metrics.current_quantity),
            reasoning=f"EMERGENCY: {reason} ({metrics.unrealized_pnl_percent:.1%} loss)",
            confidence=0.95,
            urgency=OrderUrgency.CRITICAL,
            risk_level=RiskLevel.CRITICAL,
            order_type="market",  # Immediate execution
            max_loss_limit=abs(metrics.unrealized_pnl)
        )
    
    async def _create_stop_loss_decision(self, metrics: PositionMetrics, reason: str) -> PositionDecision:
        """Create stop loss decision."""
        
        return PositionDecision(
            symbol=metrics.symbol,
            action=PositionAction.CLOSE if metrics.current_quantity > 0 else PositionAction.COVER,
            quantity=abs(metrics.current_quantity),
            reasoning=f"STOP LOSS: {reason} (price: ${metrics.current_price:.2f}, stop: ${metrics.stop_loss_price:.2f})",
            confidence=0.90,
            urgency=OrderUrgency.HIGH,
            risk_level=RiskLevel.HIGH,
            order_type="market",
            max_loss_limit=abs(metrics.current_quantity * (metrics.current_price - metrics.stop_loss_price))
        )
    
    async def _create_take_profit_decision(self, metrics: PositionMetrics, reason: str) -> PositionDecision:
        """Create take profit decision."""
        
        # Partial profit taking - close 50% of position
        profit_quantity = abs(metrics.current_quantity) * 0.5
        
        return PositionDecision(
            symbol=metrics.symbol,
            action=PositionAction.REDUCE,
            quantity=profit_quantity,
            reasoning=f"TAKE PROFIT: {reason} ({metrics.unrealized_pnl_percent:.1%} gain)",
            confidence=0.85,
            urgency=OrderUrgency.MEDIUM,
            risk_level=RiskLevel.LOW,
            order_type="limit",
            limit_price=metrics.current_price * 0.998,  # Slight discount for execution
            new_stop_loss=metrics.avg_entry_price * 1.05,  # Raise stop loss to breakeven+
            position_size_adjustment=0.5
        )
    
    async def _create_risk_reduction_decision(self, metrics: PositionMetrics, market_context: MarketContext) -> PositionDecision:
        """Create risk reduction decision."""
        
        # Reduce position size by 30-50% based on risk score
        reduction_factor = min(0.5, metrics.position_risk_score * 0.7)
        reduce_quantity = abs(metrics.current_quantity) * reduction_factor
        
        return PositionDecision(
            symbol=metrics.symbol,
            action=PositionAction.REDUCE,
            quantity=reduce_quantity,
            reasoning=f"RISK REDUCTION: High risk score ({metrics.position_risk_score:.2f})",
            confidence=0.80,
            urgency=OrderUrgency.HIGH,
            risk_level=metrics.risk_level,
            order_type="limit",
            limit_price=metrics.current_price * 0.999,
            position_size_adjustment=1 - reduction_factor
        )
    
    async def _create_underperformance_decision(self, metrics: PositionMetrics, market_context: MarketContext) -> PositionDecision:
        """Create decision for underperforming positions."""
        
        # If fundamentally sound but temporarily down, consider adding
        if (metrics.unrealized_pnl_percent > -0.08 and  # Not too deep
            metrics.rsi < self.oversold_rsi and  # Oversold
            market_context.regime != MarketRegime.BEAR_MARKET):  # Not in bear market
            
            # Add to position (dollar cost averaging)
            add_quantity = abs(metrics.current_quantity) * 0.2  # 20% more
            
            return PositionDecision(
                symbol=metrics.symbol,
                action=PositionAction.BUY if metrics.current_quantity > 0 else PositionAction.SHORT,
                quantity=add_quantity,
                reasoning=f"DCA OPPORTUNITY: Oversold RSI ({metrics.rsi:.1f}), moderate loss ({metrics.unrealized_pnl_percent:.1%})",
                confidence=0.65,
                urgency=OrderUrgency.MEDIUM,
                risk_level=RiskLevel.MEDIUM,
                order_type="limit",
                limit_price=metrics.current_price * 1.001,  # Slight premium
                position_size_adjustment=1.2
            )
        
        else:
            # Cut losses
            cut_quantity = abs(metrics.current_quantity) * 0.4  # Cut 40%
            
            return PositionDecision(
                symbol=metrics.symbol,
                action=PositionAction.REDUCE,
                quantity=cut_quantity,
                reasoning=f"CUT LOSSES: Persistent underperformance ({metrics.unrealized_pnl_percent:.1%})",
                confidence=0.75,
                urgency=OrderUrgency.MEDIUM,
                risk_level=RiskLevel.HIGH,
                order_type="market",
                position_size_adjustment=0.6
            )
    
    async def _create_profit_management_decision(self, metrics: PositionMetrics, market_context: MarketContext) -> Optional[PositionDecision]:
        """Create decision for profitable positions."""
        
        # If very profitable and showing signs of reversal
        if (metrics.unrealized_pnl_percent > 0.15 and  # 15%+ profit
            metrics.rsi > self.overbought_rsi):  # Overbought
            
            # Trim position
            trim_quantity = abs(metrics.current_quantity) * 0.3
            
            return PositionDecision(
                symbol=metrics.symbol,
                action=PositionAction.REDUCE,
                quantity=trim_quantity,
                reasoning=f"PROFIT TRIM: Strong gains ({metrics.unrealized_pnl_percent:.1%}), overbought RSI ({metrics.rsi:.1f})",
                confidence=0.70,
                urgency=OrderUrgency.LOW,
                risk_level=RiskLevel.LOW,
                order_type="limit",
                limit_price=metrics.current_price * 0.998,
                new_stop_loss=metrics.current_price * 0.90,  # Trailing stop
                position_size_adjustment=0.7
            )
        
        # If strongly trending up, let it run but tighten stops
        elif (metrics.unrealized_pnl_percent > 0.10 and
              metrics.current_price > metrics.moving_avg_20 > metrics.moving_avg_50):
            
            # Update stop loss only (no position change)
            new_stop = max(metrics.stop_loss_price or 0, metrics.current_price * 0.95)
            
            return PositionDecision(
                symbol=metrics.symbol,
                action=PositionAction.HOLD,
                quantity=0,
                reasoning=f"TRAIL STOP: Strong uptrend, tightening stop loss",
                confidence=0.60,
                urgency=OrderUrgency.LOW,
                risk_level=RiskLevel.LOW,
                new_stop_loss=new_stop
            )
        
        return None
    
    async def _create_technical_analysis_decision(self, metrics: PositionMetrics, market_context: MarketContext) -> Optional[PositionDecision]:
        """Create decision based on technical analysis."""
        
        # Support/resistance levels
        if metrics.current_price <= metrics.support_level * 1.02:  # Near support
            if metrics.current_quantity > 0:  # Long position
                # Add near support
                add_quantity = abs(metrics.current_quantity) * 0.15
                
                return PositionDecision(
                    symbol=metrics.symbol,
                    action=PositionAction.BUY,
                    quantity=add_quantity,
                    reasoning=f"TECHNICAL: Near support level (${metrics.support_level:.2f})",
                    confidence=0.60,
                    urgency=OrderUrgency.MEDIUM,
                    risk_level=RiskLevel.MEDIUM,
                    order_type="limit",
                    limit_price=metrics.support_level * 1.005
                )
        
        elif metrics.current_price >= metrics.resistance_level * 0.98:  # Near resistance
            if metrics.current_quantity > 0:  # Long position
                # Trim near resistance
                trim_quantity = abs(metrics.current_quantity) * 0.25
                
                return PositionDecision(
                    symbol=metrics.symbol,
                    action=PositionAction.REDUCE,
                    quantity=trim_quantity,
                    reasoning=f"TECHNICAL: Near resistance level (${metrics.resistance_level:.2f})",
                    confidence=0.65,
                    urgency=OrderUrgency.LOW,
                    risk_level=RiskLevel.LOW,
                    order_type="limit",
                    limit_price=metrics.resistance_level * 0.995
                )
        
        return None
    
    async def _create_regime_based_decision(self, metrics: PositionMetrics, market_context: MarketContext) -> Optional[PositionDecision]:
        """Create decision based on market regime."""
        
        if market_context.regime == MarketRegime.CRISIS:
            # Reduce all positions in crisis
            if abs(metrics.current_quantity) > 0:
                reduce_quantity = abs(metrics.current_quantity) * 0.4
                
                return PositionDecision(
                    symbol=metrics.symbol,
                    action=PositionAction.REDUCE,
                    quantity=reduce_quantity,
                    reasoning=f"REGIME: Crisis mode - reducing exposure",
                    confidence=0.85,
                    urgency=OrderUrgency.HIGH,
                    risk_level=RiskLevel.HIGH,
                    order_type="market"
                )
        
        elif market_context.regime == MarketRegime.HIGH_VOLATILITY:
            # Tighten stops in high volatility
            if metrics.stop_loss_price:
                tighter_stop = metrics.current_price * 0.97  # 3% stop
                if tighter_stop > metrics.stop_loss_price:
                    return PositionDecision(
                        symbol=metrics.symbol,
                        action=PositionAction.HOLD,
                        quantity=0,
                        reasoning=f"REGIME: High volatility - tightening stops",
                        confidence=0.70,
                        urgency=OrderUrgency.MEDIUM,
                        risk_level=RiskLevel.MEDIUM,
                        new_stop_loss=tighter_stop
                    )
        
        return None
    
    def _calculate_position_risk_score(self, 
                                     symbol: str, 
                                     pnl_percent: float,
                                     volatility: float,
                                     quantity: float,
                                     market_value: float) -> float:
        """Calculate comprehensive risk score for a position."""
        
        # PnL risk component
        pnl_risk = max(0, -pnl_percent * 2)  # Higher risk for losses
        
        # Volatility risk component
        vol_risk = min(volatility * 10, 1.0)  # Scale volatility to 0-1
        
        # Position size risk component
        try:
            account_info = alpaca_client.get_account_info()
            portfolio_value = float(account_info.get('portfolio_value', 100000))
            position_weight = abs(market_value) / portfolio_value
            size_risk = position_weight / settings.max_position_size
        except:
            size_risk = 0.5  # Default moderate risk
        
        # Combine risk components
        total_risk = (pnl_risk * 0.4) + (vol_risk * 0.3) + (size_risk * 0.3)
        
        return min(max(total_risk, 0.0), 1.0)
    
    def _determine_position_status(self, 
                                 pnl_percent: float, 
                                 risk_score: float,
                                 technical_data: Dict) -> PositionStatus:
        """Determine position status based on performance and risk."""
        
        if pnl_percent >= self.take_profit_threshold:
            return PositionStatus.TAKE_PROFIT
        elif pnl_percent >= 0.05:
            return PositionStatus.PROFITABLE
        elif pnl_percent <= self.critical_loss_threshold:
            return PositionStatus.CRITICAL
        elif pnl_percent <= self.underperformance_threshold:
            return PositionStatus.UNDERPERFORMING
        elif risk_score > 0.7:
            return PositionStatus.AT_RISK
        else:
            return PositionStatus.HEALTHY
    
    def _determine_risk_level(self, 
                            risk_score: float, 
                            pnl_percent: float,
                            status: PositionStatus) -> RiskLevel:
        """Determine risk level for a position."""
        
        if status in [PositionStatus.CRITICAL] or risk_score > 0.8:
            return RiskLevel.CRITICAL
        elif status in [PositionStatus.AT_RISK] or risk_score > 0.6:
            return RiskLevel.HIGH
        elif status in [PositionStatus.UNDERPERFORMING] or risk_score > 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _calculate_stop_take_levels(self, 
                                  symbol: str, 
                                  current_price: float,
                                  entry_price: float,
                                  quantity: float,
                                  volatility: float) -> Tuple[Optional[float], Optional[float]]:
        """Calculate dynamic stop loss and take profit levels."""
        
        # Dynamic stop loss based on volatility
        vol_multiplier = max(1.0, volatility * 20)  # Scale volatility
        stop_distance = self.stop_loss_percent * vol_multiplier
        
        if quantity > 0:  # Long position
            stop_loss = current_price * (1 - stop_distance)
            take_profit = current_price * (1 + self.take_profit_threshold)
        else:  # Short position
            stop_loss = current_price * (1 + stop_distance)
            take_profit = current_price * (1 - self.take_profit_threshold)
        
        return stop_loss, take_profit
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            price = alpaca_client.get_current_price(symbol)
            return float(price) if price else None
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            return None
    
    async def _get_technical_indicators(self, symbol: str) -> Dict:
        """Get technical indicators for a symbol."""
        try:
            # Get recent market data
            df = alpaca_client.get_market_data(symbol, timeframe="1Day", limit=50)
            if df.empty:
                return {}
            
            # Calculate basic indicators
            indicators = {}
            
            # Moving averages
            indicators['ma_20'] = df['close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['close'].iloc[-1]
            indicators['ma_50'] = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else df['close'].iloc[-1]
            
            # RSI calculation (simplified)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs.iloc[-1])) if len(df) >= 14 else 50.0
            
            # Volatility (20-day)
            indicators['volatility'] = df['close'].pct_change().rolling(20).std().iloc[-1] if len(df) >= 20 else 0.02
            
            # Support and resistance (simplified)
            recent_data = df.tail(20) if len(df) >= 20 else df
            indicators['support'] = recent_data['low'].min()
            indicators['resistance'] = recent_data['high'].max()
            
            # Volume ratio
            avg_volume = df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['volume'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            indicators['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Beta (simplified - correlation with SPY)
            indicators['beta'] = 1.0  # Placeholder
            
            return indicators
            
        except Exception as e:
            logger.error(f"Failed to get technical indicators for {symbol}: {e}")
            return {}
    
    async def _get_position_performance_data(self, symbol: str, position_data: Dict) -> Dict:
        """Get position performance tracking data."""
        
        # This would typically track position history
        # For now, return placeholder data
        return {
            'max_profit': 0.0,
            'max_drawdown': 0.0,
            'days_held': 1,
            'add_count': 0,
            'trim_count': 0
        }
    
    async def _calculate_portfolio_risk_budget(self) -> float:
        """Calculate available portfolio risk budget."""
        try:
            account_info = alpaca_client.get_account_info()
            portfolio_value = float(account_info.get('portfolio_value', 100000))
            day_pnl = float(account_info.get('daychange', 0))
            
            # Risk budget based on current P&L and limits
            daily_loss_pct = day_pnl / portfolio_value if portfolio_value > 0 else 0
            remaining_risk_budget = settings.max_daily_loss + daily_loss_pct
            
            return max(0, remaining_risk_budget)
            
        except Exception as e:
            logger.error(f"Failed to calculate portfolio risk budget: {e}")
            return 0.02  # Default 2% risk budget
    
    async def _get_market_context(self) -> MarketContext:
        """Get current market context."""
        # Import here to avoid circular imports
        from core.signal_processor import MarketContext, MarketRegime
        
        # Simplified market context
        return MarketContext(
            volatility=0.02,
            trend_direction=0.0,
            regime=MarketRegime.SIDEWAYS,
            liquidity_score=0.8,
            risk_off_sentiment=0.3
        )

# Global instance
position_manager = IntelligentPositionManager()

# Convenience functions
async def analyze_positions(current_positions: Dict[str, Dict],
                          market_context: Optional[MarketContext] = None) -> List[PositionMetrics]:
    """Analyze current positions."""
    return await position_manager.analyze_positions(current_positions, market_context)

async def generate_position_decisions(position_metrics: List[PositionMetrics],
                                    market_context: Optional[MarketContext] = None) -> List[PositionDecision]:
    """Generate position management decisions."""
    return await position_manager.generate_position_decisions(position_metrics, market_context)

__all__ = [
    'PositionStatus', 'RiskLevel', 'PositionMetrics', 'PositionDecision',
    'IntelligentPositionManager', 'position_manager',
    'analyze_positions', 'generate_position_decisions'
]