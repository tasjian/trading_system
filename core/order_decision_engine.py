#!/usr/bin/env python3
"""
Intelligent Order Decision Engine

MLOps framework for intelligent order type selection and portfolio balancing.
Addresses the core issue of systems only placing buy orders by implementing
sophisticated decision logic for order type selection, position balancing,
and portfolio optimization.

Features:
- Portfolio-aware order type selection (buy/sell/short/limit)
- Position rebalancing logic
- Risk-based order sizing and timing
- Market regime awareness
- Anti-overtrading mechanisms
- Dynamic position management
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

from tools.alpaca_client import alpaca_client
from core.trading_engine import trading_engine, TradingOrder, StrategyType
from order_types.advanced_orders import (
    AdvancedOrderRequest, AdvancedOrderManager, 
    place_stop_limit_order, place_trailing_stop_order, place_oco_order
)
from monitoring.portfolio_monitoring import portfolio_monitor
from config.settings import settings

# Enhanced Short-Selling Modules
from core.enhanced_short_signal_engine import enhanced_short_signal_engine, EnhancedShortSignal
from core.borrow_cost_monitor import borrow_cost_monitor, ShortabilityAnalysis
from core.short_risk_manager import short_risk_manager, SqueezeRiskMetrics
from core.enhanced_position_sizer import enhanced_position_sizer, EnhancedPositionSize

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"

class OrderDecisionType(Enum):
    BUY_NEW = "buy_new"           # New long position
    SELL_CLOSE = "sell_close"     # Close long position
    SELL_SHORT = "sell_short"     # Open short position
    ENHANCED_SHORT = "enhanced_short"  # Enhanced short with comprehensive analysis
    BUY_COVER = "buy_cover"       # Close short position
    REBALANCE_UP = "rebalance_up" # Increase position size
    REBALANCE_DOWN = "rebalance_down" # Decrease position size
    HOLD = "hold"                 # No action
    HEDGE = "hedge"               # Hedging position

@dataclass
class PositionContext:
    """Context about existing position."""
    symbol: str
    current_quantity: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    avg_cost: float
    current_price: float
    days_held: int
    position_weight: float  # Percentage of portfolio
    
@dataclass
class OrderDecision:
    """Order decision with rationale."""
    decision_type: OrderDecisionType
    order_side: str  # buy, sell, sell_short
    order_type: str  # market, limit, stop_limit
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    confidence: float = 0.5
    reasoning: str = ""
    priority: int = 5  # 1-10, higher = more urgent
    estimated_impact: float = 0.0  # Portfolio impact percentage
    
    # Enhanced short-selling fields
    enhanced_short_signal: Optional[EnhancedShortSignal] = None
    shortability_analysis: Optional[ShortabilityAnalysis] = None
    squeeze_risk_metrics: Optional[SqueezeRiskMetrics] = None
    enhanced_position_size: Optional[EnhancedPositionSize] = None
    borrow_cost_validated: bool = False
    ssr_compliant: bool = True

class OrderDecisionEngine:
    """Intelligent order decision engine for portfolio balancing."""
    
    def __init__(self):
        """Initialize order decision engine."""
        self.advanced_order_manager = AdvancedOrderManager()
        
        # Decision parameters
        self.rebalance_threshold = settings.min_rebalance_threshold
        self.max_position_weight = settings.max_position_size
        self.target_portfolio_utilization = 0.85  # 85% target utilization
        self.anti_overtrading_window = timedelta(hours=1)  # Minimum time between trades per symbol
        
        # Position sizing parameters
        self.base_position_size = 0.05  # 5% base position size
        self.max_single_trade_size = 0.10  # 10% max single trade
        self.volatility_scaling = True
        
        # Risk management parameters
        self.default_stop_loss_pct = 0.08  # 8% stop loss
        self.default_take_profit_pct = 0.15  # 15% take profit
        self.trailing_stop_pct = 0.05  # 5% trailing stop
        
        # Trading frequency limits
        self.max_daily_trades = 20
        self.daily_trade_count = 0
        self.last_trade_date = None
        
        # Order tracking for anti-overtrading
        self.recent_orders: Dict[str, datetime] = {}  # symbol -> last order time
    
    def generate_order_decisions(self, signals: List[Dict]) -> List[OrderDecision]:
        """Generate order decisions from signals (synchronous wrapper)."""
        try:
            decisions = []
            
            for signal in signals:
                symbol = signal.get('symbol', '')
                signal_type = signal.get('signal_type', 'hold')
                strength = signal.get('strength', 0.5)
                confidence = signal.get('confidence', 0.5)
                reasoning = signal.get('reasoning', 'Generated from signal')
                
                if not symbol:
                    continue
                
                # Create basic order decision based on signal
                if signal_type == 'short' and settings.enable_short_selling:
                    decision = OrderDecision(
                        decision_type=OrderDecisionType.SELL_SHORT,
                        order_side="sell_short",
                        order_type="limit",
                        quantity=100,  # Base quantity, will be adjusted by position sizing
                        confidence=confidence,
                        reasoning=f"Short signal: {reasoning}",
                        priority=8 if strength > 0.7 else 5
                    )
                    decisions.append(decision)
                    
                elif signal_type == 'buy':
                    decision = OrderDecision(
                        decision_type=OrderDecisionType.BUY_NEW,
                        order_side="buy",
                        order_type="limit", 
                        quantity=100,  # Base quantity
                        confidence=confidence,
                        reasoning=f"Buy signal: {reasoning}",
                        priority=7 if strength > 0.7 else 5
                    )
                    decisions.append(decision)
                    
                elif signal_type == 'sell':
                    decision = OrderDecision(
                        decision_type=OrderDecisionType.SELL_CLOSE,
                        order_side="sell",
                        order_type="limit",
                        quantity=100,  # Will be adjusted based on position
                        confidence=confidence,
                        reasoning=f"Sell signal: {reasoning}",
                        priority=6
                    )
                    decisions.append(decision)
            
            logger.info(f"Generated {len(decisions)} order decisions from {len(signals)} signals")
            return decisions
            
        except Exception as e:
            logger.error(f"Error generating order decisions: {e}")
            return []
        
    async def analyze_and_decide(self, symbol: str, signal_data: Dict, 
                               current_portfolio: Dict) -> List[OrderDecision]:
        """
        Analyze signal and portfolio context to make intelligent order decisions.
        
        Args:
            symbol: Symbol to analyze
            signal_data: Signal information (strength, type, reasoning)
            current_portfolio: Current portfolio state
            
        Returns:
            List of order decisions
        """
        try:
            logger.debug(f"Analyzing order decision for {symbol}")
            
            # Reset daily trade count if new day
            self._reset_daily_counters()
            
            # Check if we can trade this symbol (anti-overtrading)
            if not self._can_trade_symbol(symbol):
                logger.debug(f"Anti-overtrading protection active for {symbol}")
                return [OrderDecision(
                    decision_type=OrderDecisionType.HOLD,
                    order_side="hold",
                    order_type="none",
                    quantity=0,
                    reasoning="Anti-overtrading protection active"
                )]
            
            # Check daily trade limits
            if self.daily_trade_count >= self.max_daily_trades:
                logger.info(f"Daily trade limit reached ({self.max_daily_trades})")
                return []
            
            # Get position context
            position_context = await self._get_position_context(symbol, current_portfolio)
            
            # Get market regime
            market_regime = await self._assess_market_regime(symbol)
            
            # Get portfolio balance state
            portfolio_state = await self._analyze_portfolio_balance(current_portfolio)
            
            # Extract signal information
            signal_strength = signal_data.get('strength', 0.5)
            signal_type = signal_data.get('signal', 'HOLD').upper()
            signal_confidence = signal_data.get('confidence', 0.5)
            
            # Make order decision based on context
            decisions = await self._make_order_decision(
                symbol=symbol,
                signal_type=signal_type,
                signal_strength=signal_strength,
                signal_confidence=signal_confidence,
                position_context=position_context,
                market_regime=market_regime,
                portfolio_state=portfolio_state,
                signal_data=signal_data
            )
            
            # Validate and refine decisions
            validated_decisions = self._validate_decisions(decisions, current_portfolio)
            
            # Add risk management orders
            risk_managed_decisions = await self._add_risk_management(validated_decisions, symbol)
            
            return risk_managed_decisions
            
        except Exception as e:
            logger.error(f"Error in order decision analysis for {symbol}: {e}")
            return []
    
    async def analyze_enhanced_short_opportunity(self, symbol: str, 
                                               current_portfolio: Dict) -> Optional[OrderDecision]:
        """
        Comprehensive enhanced short opportunity analysis using all short-selling modules.
        
        Args:
            symbol: Symbol to analyze for short opportunities
            current_portfolio: Current portfolio state
            
        Returns:
            Enhanced short order decision if opportunity found, None otherwise
        """
        try:
            logger.info(f"Analyzing enhanced short opportunity for {symbol}")
            
            # Check basic eligibility
            if not self._can_trade_symbol(symbol):
                logger.debug(f"Symbol {symbol} not eligible for trading (anti-overtrading)")
                return None
            
            # Step 1: Generate enhanced short signals
            short_signals = await enhanced_short_signal_engine.generate_enhanced_short_signals([symbol], max_signals=1)
            
            if not short_signals:
                logger.debug(f"No enhanced short signals generated for {symbol}")
                return None
            
            enhanced_signal = short_signals[0]
            
            # Only proceed with SHORT or WEAK_SHORT signals
            if enhanced_signal.signal_type not in ['SHORT', 'WEAK_SHORT']:
                logger.debug(f"Enhanced signal type {enhanced_signal.signal_type} not suitable for shorting")
                return None
            
            # Step 2: Analyze shortability (borrow costs, availability, SSR compliance)
            shortability_analysis = await borrow_cost_monitor.get_borrow_cost_analysis(symbol)
            
            if not shortability_analysis:
                logger.warning(f"Could not obtain shortability analysis for {symbol}")
                return None
            
            # Check if shortability score is acceptable
            if shortability_analysis.shortability_score < 40:  # Below 40/100
                logger.info(f"Low shortability score for {symbol}: {shortability_analysis.shortability_score:.1f}/100")
                return None
            
            # Step 3: Assess squeeze risk
            squeeze_risk = await short_risk_manager.squeeze_analyzer.assess_squeeze_risk(symbol)
            
            # High squeeze risk should be avoided
            if squeeze_risk.squeeze_risk_score > 75:  # Above 75/100
                logger.info(f"High squeeze risk for {symbol}: {squeeze_risk.squeeze_risk_score:.1f}/100")
                return None
            
            # Step 4: Calculate enhanced position size
            enhanced_position_size = await enhanced_position_sizer.calculate_enhanced_position_size(
                symbol, enhanced_signal, current_portfolio
            )
            
            # Check if position size is viable
            if enhanced_position_size.recommended_size < enhanced_position_size.min_size:
                logger.debug(f"Recommended position size too small for {symbol}")
                return None
            
            # Step 5: Validate the trade comprehensively
            trade_valid, validation_message, validation_details = await short_risk_manager.validate_short_trade(
                symbol, 
                int(enhanced_position_size.recommended_size * current_portfolio.get('equity', 100000) / (enhanced_signal.technical_indicators.get('current_price', 100) or 100)),
                current_portfolio
            )
            
            if not trade_valid:
                logger.info(f"Short trade validation failed for {symbol}: {validation_message}")
                return None
            
            # Step 6: Validate borrow costs and SSR compliance
            current_price = alpaca_client.get_current_price(symbol) or enhanced_signal.technical_indicators.get('current_price', 100)
            quantity = int(enhanced_position_size.recommended_size * current_portfolio.get('equity', 100000) / current_price)
            
            borrow_valid, borrow_message, borrow_details = await borrow_cost_monitor.validate_short_trade(
                symbol, quantity, current_price
            )
            
            if not borrow_valid:
                logger.info(f"Borrow cost validation failed for {symbol}: {borrow_message}")
                return None
            
            # Step 7: Create enhanced order decision
            # Determine order type based on signal urgency and market conditions
            if enhanced_signal.signal_strength > 0.8 and squeeze_risk.squeeze_risk_score < 30:
                order_type = "market"  # High conviction, low squeeze risk
                limit_price = None
                priority = 8
            elif enhanced_signal.signal_strength > 0.6:
                order_type = "limit"
                limit_price = current_price * 1.002  # Slight premium for execution
                priority = 7
            else:
                order_type = "limit"
                limit_price = current_price * 1.005  # Conservative limit
                priority = 6
            
            # Calculate stop loss and take profit from enhanced position size
            stop_loss_price = enhanced_position_size.stop_loss_level
            take_profit_price = enhanced_position_size.take_profit_level
            
            # Create comprehensive reasoning
            reasoning_parts = [
                f"Enhanced short analysis: {enhanced_signal.reasoning}",
                f"Shortability score: {shortability_analysis.shortability_score:.1f}/100",
                f"Squeeze risk: {squeeze_risk.squeeze_risk_score:.1f}/100",
                f"Position sizing: {enhanced_position_size.primary_method} method",
                f"Borrow cost: {shortability_analysis.borrow_cost.borrow_fee_rate:.1%} annually" if shortability_analysis.borrow_cost else "",
                f"Confidence: {enhanced_position_size.confidence_score:.1%}"
            ]
            reasoning = " | ".join([part for part in reasoning_parts if part])
            
            # Create the enhanced order decision
            decision = OrderDecision(
                decision_type=OrderDecisionType.ENHANCED_SHORT,
                order_side="sell_short",
                order_type=order_type,
                quantity=float(quantity),
                limit_price=limit_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                confidence=enhanced_position_size.confidence_score,
                reasoning=reasoning,
                priority=priority,
                estimated_impact=enhanced_position_size.recommended_size,
                
                # Enhanced short-selling fields
                enhanced_short_signal=enhanced_signal,
                shortability_analysis=shortability_analysis,
                squeeze_risk_metrics=squeeze_risk,
                enhanced_position_size=enhanced_position_size,
                borrow_cost_validated=borrow_valid,
                ssr_compliant=shortability_analysis.ssr_status.ssr_active if shortability_analysis.ssr_status else True
            )
            
            logger.info(f"✅ Enhanced short opportunity identified for {symbol}: {quantity} shares @ {order_type}")
            logger.info(f"   Signal strength: {enhanced_signal.signal_strength:.1%}, Shortability: {shortability_analysis.shortability_score:.0f}/100")
            logger.info(f"   Position size: {enhanced_position_size.recommended_size:.1%}, Risk level: {shortability_analysis.risk_level}")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in enhanced short analysis for {symbol}: {e}")
            return None
    
    async def _get_position_context(self, symbol: str, current_portfolio: Dict) -> Optional[PositionContext]:
        """Get context about existing position."""
        try:
            positions = current_portfolio.get('positions', {})
            portfolio_value = current_portfolio.get('equity', 1)
            
            if symbol in positions:
                pos = positions[symbol]
                return PositionContext(
                    symbol=symbol,
                    current_quantity=float(pos.get('qty', 0)),
                    market_value=float(pos.get('market_value', 0)),
                    unrealized_pnl=float(pos.get('unrealized_pl', 0)),
                    unrealized_pnl_pct=float(pos.get('unrealized_plpc', 0)),
                    avg_cost=float(pos.get('avg_entry_price', 0)),
                    current_price=float(pos.get('current_price', 0)),
                    days_held=0,  # Would need order history
                    position_weight=abs(float(pos.get('market_value', 0))) / portfolio_value if portfolio_value > 0 else 0
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting position context for {symbol}: {e}")
            return None
    
    async def _assess_market_regime(self, symbol: str) -> MarketRegime:
        """Assess current market regime for the symbol."""
        try:
            # Get recent price data
            market_data = alpaca_client.get_market_data(symbol, limit=20)
            
            if market_data is None or len(market_data) < 10:
                return MarketRegime.SIDEWAYS
            
            # Calculate volatility and trend
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std()
            trend = (market_data['close'].iloc[-1] - market_data['close'].iloc[0]) / market_data['close'].iloc[0]
            
            # Classify regime
            if volatility > 0.03:  # High volatility
                return MarketRegime.VOLATILE
            elif volatility < 0.01:  # Low volatility
                return MarketRegime.LOW_VOLATILITY
            elif trend > 0.05:  # Strong uptrend
                return MarketRegime.BULLISH
            elif trend < -0.05:  # Strong downtrend
                return MarketRegime.BEARISH
            else:
                return MarketRegime.SIDEWAYS
                
        except Exception as e:
            logger.error(f"Error assessing market regime for {symbol}: {e}")
            return MarketRegime.SIDEWAYS
    
    async def _analyze_portfolio_balance(self, current_portfolio: Dict) -> Dict[str, Any]:
        """Analyze current portfolio balance and identify needs."""
        try:
            portfolio_value = current_portfolio.get('equity', 1)
            cash = current_portfolio.get('cash', 0)
            positions = current_portfolio.get('positions', {})
            
            # Calculate utilization
            invested_value = sum(abs(float(pos.get('market_value', 0))) for pos in positions.values())
            utilization = invested_value / portfolio_value if portfolio_value > 0 else 0
            
            # Calculate long/short exposure
            long_exposure = sum(float(pos.get('market_value', 0)) for pos in positions.values() 
                              if float(pos.get('qty', 0)) > 0)
            short_exposure = sum(abs(float(pos.get('market_value', 0))) for pos in positions.values() 
                               if float(pos.get('qty', 0)) < 0)
            
            net_exposure = (long_exposure - short_exposure) / portfolio_value if portfolio_value > 0 else 0
            gross_exposure = (long_exposure + short_exposure) / portfolio_value if portfolio_value > 0 else 0
            
            # Calculate concentration
            position_weights = [abs(float(pos.get('market_value', 0))) / portfolio_value 
                              for pos in positions.values() if portfolio_value > 0]
            max_concentration = max(position_weights) if position_weights else 0
            
            # Identify needs
            needs_buying = utilization < self.target_portfolio_utilization
            needs_selling = utilization > 0.95
            needs_rebalancing = max_concentration > self.max_position_weight
            is_overexposed_long = net_exposure > 0.80
            is_overexposed_short = net_exposure < -0.20
            
            return {
                'utilization': utilization,
                'net_exposure': net_exposure,
                'gross_exposure': gross_exposure,
                'max_concentration': max_concentration,
                'position_count': len(positions),
                'cash_ratio': cash / portfolio_value if portfolio_value > 0 else 1,
                'needs_buying': needs_buying,
                'needs_selling': needs_selling,
                'needs_rebalancing': needs_rebalancing,
                'is_overexposed_long': is_overexposed_long,
                'is_overexposed_short': is_overexposed_short,
                'long_exposure': long_exposure,
                'short_exposure': short_exposure
            }
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio balance: {e}")
            return {'utilization': 0.5, 'needs_buying': True, 'needs_selling': False}
    
    async def _make_order_decision(self, symbol: str, signal_type: str, signal_strength: float,
                                 signal_confidence: float, position_context: Optional[PositionContext],
                                 market_regime: MarketRegime, portfolio_state: Dict,
                                 signal_data: Dict) -> List[OrderDecision]:
        """Make intelligent order decision based on all context."""
        try:
            decisions = []
            
            # Get current price for calculations
            current_price = alpaca_client.get_current_price(symbol)
            if not current_price or current_price <= 0:
                return decisions
            
            # Determine base decision type
            if position_context is None:
                # No existing position
                decisions.extend(await self._decide_new_position(
                    symbol, signal_type, signal_strength, signal_confidence,
                    market_regime, portfolio_state, current_price, signal_data
                ))
            else:
                # Existing position
                decisions.extend(await self._decide_existing_position(
                    symbol, signal_type, signal_strength, signal_confidence,
                    position_context, market_regime, portfolio_state, current_price, signal_data
                ))
            
            # Add portfolio rebalancing decisions
            rebalance_decisions = await self._decide_portfolio_rebalancing(
                symbol, position_context, portfolio_state, current_price
            )
            decisions.extend(rebalance_decisions)
            
            return decisions
            
        except Exception as e:
            logger.error(f"Error making order decision for {symbol}: {e}")
            return []
    
    async def _decide_new_position(self, symbol: str, signal_type: str, signal_strength: float,
                                 signal_confidence: float, market_regime: MarketRegime,
                                 portfolio_state: Dict, current_price: float,
                                 signal_data: Dict) -> List[OrderDecision]:
        """Decide on new position orders."""
        decisions = []
        
        try:
            # Check if portfolio needs new positions
            if not portfolio_state.get('needs_buying', True):
                return decisions
            
            # Calculate position size
            position_size = self._calculate_position_size(
                signal_strength, signal_confidence, market_regime, portfolio_state
            )
            
            if position_size <= 0:
                return decisions
            
            # Determine order type and execution strategy
            if signal_type in ['BUY', 'STRONG_BUY']:
                # Long position decision
                if portfolio_state.get('is_overexposed_long', False):
                    # Already too long, skip or reduce size
                    position_size *= 0.5
                    if position_size < 0.01:
                        return decisions
                
                order_type, limit_price, reasoning = self._determine_order_execution_type(
                    "buy", current_price, signal_strength, market_regime
                )
                
                decision = OrderDecision(
                    decision_type=OrderDecisionType.BUY_NEW,
                    order_side="buy",
                    order_type=order_type,
                    quantity=position_size * portfolio_state.get('portfolio_value', 100000) / current_price,
                    limit_price=limit_price,
                    confidence=signal_confidence,
                    reasoning=f"New long position: {reasoning}. Signal: {signal_type} ({signal_strength:.2f})",
                    priority=7,
                    estimated_impact=position_size
                )
                decisions.append(decision)
                
            elif signal_type in ['SELL', 'STRONG_SELL']:
                # Short position decision (if allowed)
                if not portfolio_state.get('allows_shorting', True):
                    return decisions
                    
                if portfolio_state.get('is_overexposed_short', False):
                    return decisions
                
                # Consider shorting only with high confidence in bearish market
                if signal_confidence > 0.7 and market_regime in [MarketRegime.BEARISH, MarketRegime.VOLATILE]:
                    order_type, limit_price, reasoning = self._determine_order_execution_type(
                        "sell_short", current_price, signal_strength, market_regime
                    )
                    
                    # Smaller position size for shorts
                    short_position_size = position_size * 0.7
                    
                    decision = OrderDecision(
                        decision_type=OrderDecisionType.SELL_SHORT,
                        order_side="sell_short",
                        order_type=order_type,
                        quantity=short_position_size * portfolio_state.get('portfolio_value', 100000) / current_price,
                        limit_price=limit_price,
                        confidence=signal_confidence,
                        reasoning=f"New short position: {reasoning}. Signal: {signal_type} ({signal_strength:.2f})",
                        priority=6,
                        estimated_impact=short_position_size
                    )
                    decisions.append(decision)
            
            return decisions
            
        except Exception as e:
            logger.error(f"Error deciding new position for {symbol}: {e}")
            return []
    
    async def _decide_existing_position(self, symbol: str, signal_type: str, signal_strength: float,
                                      signal_confidence: float, position_context: PositionContext,
                                      market_regime: MarketRegime, portfolio_state: Dict,
                                      current_price: float, signal_data: Dict) -> List[OrderDecision]:
        """Decide on existing position management."""
        decisions = []
        
        try:
            is_long = position_context.current_quantity > 0
            is_short = position_context.current_quantity < 0
            
            # Check if position is too large (concentration risk)
            if position_context.position_weight > self.max_position_weight:
                # Force position size reduction
                reduction_amount = (position_context.position_weight - self.max_position_weight) * portfolio_state.get('portfolio_value', 100000) / current_price
                
                decision = OrderDecision(
                    decision_type=OrderDecisionType.REBALANCE_DOWN,
                    order_side="sell" if is_long else "buy",
                    order_type="market",
                    quantity=abs(reduction_amount),
                    confidence=0.9,
                    reasoning=f"Risk management: Position too large ({position_context.position_weight:.1%} > {self.max_position_weight:.1%})",
                    priority=9,
                    estimated_impact=-position_context.position_weight + self.max_position_weight
                )
                decisions.append(decision)
                return decisions
            
            # Profit taking decisions
            if position_context.unrealized_pnl_pct > 0.15:  # 15% profit
                # Consider partial profit taking
                profit_take_amount = abs(position_context.current_quantity) * 0.3  # Take 30% profit
                
                decision = OrderDecision(
                    decision_type=OrderDecisionType.SELL_CLOSE if is_long else OrderDecisionType.BUY_COVER,
                    order_side="sell" if is_long else "buy",
                    order_type="limit",
                    quantity=profit_take_amount,
                    limit_price=current_price * (1.01 if is_long else 0.99),  # Slight premium
                    confidence=0.8,
                    reasoning=f"Profit taking: {position_context.unrealized_pnl_pct:.1%} gain",
                    priority=6,
                    estimated_impact=-profit_take_amount * current_price / portfolio_state.get('portfolio_value', 100000)
                )
                decisions.append(decision)
            
            # Stop loss decisions
            elif position_context.unrealized_pnl_pct < -0.08:  # 8% loss
                # Close losing position
                decision = OrderDecision(
                    decision_type=OrderDecisionType.SELL_CLOSE if is_long else OrderDecisionType.BUY_COVER,
                    order_side="sell" if is_long else "buy",
                    order_type="market",
                    quantity=abs(position_context.current_quantity),
                    confidence=0.9,
                    reasoning=f"Stop loss: {position_context.unrealized_pnl_pct:.1%} loss",
                    priority=8,
                    estimated_impact=-position_context.position_weight
                )
                decisions.append(decision)
            
            # Signal-based decisions for existing positions
            elif signal_type in ['STRONG_SELL', 'SELL'] and is_long:
                # Consider closing or reducing long position
                if signal_confidence > 0.7:
                    close_amount = abs(position_context.current_quantity) * (0.5 + signal_strength * 0.5)
                    
                    decision = OrderDecision(
                        decision_type=OrderDecisionType.SELL_CLOSE,
                        order_side="sell",
                        order_type="limit",
                        quantity=close_amount,
                        limit_price=current_price * 0.995,  # Slight discount for quick execution
                        confidence=signal_confidence,
                        reasoning=f"Signal-based exit: {signal_type} ({signal_strength:.2f})",
                        priority=7,
                        estimated_impact=-close_amount * current_price / portfolio_state.get('portfolio_value', 100000)
                    )
                    decisions.append(decision)
            
            elif signal_type in ['STRONG_BUY', 'BUY'] and is_short:
                # Consider covering short position
                if signal_confidence > 0.7:
                    cover_amount = abs(position_context.current_quantity) * (0.5 + signal_strength * 0.5)
                    
                    decision = OrderDecision(
                        decision_type=OrderDecisionType.BUY_COVER,
                        order_side="buy",
                        order_type="limit",
                        quantity=cover_amount,
                        limit_price=current_price * 1.005,  # Slight premium for quick execution
                        confidence=signal_confidence,
                        reasoning=f"Signal-based cover: {signal_type} ({signal_strength:.2f})",
                        priority=7,
                        estimated_impact=-cover_amount * current_price / portfolio_state.get('portfolio_value', 100000)
                    )
                    decisions.append(decision)
            
            # Position enhancement decisions
            elif signal_type == 'STRONG_BUY' and is_long and signal_confidence > 0.8:
                # Consider adding to winning long position
                if position_context.unrealized_pnl_pct > 0.05 and position_context.position_weight < self.max_position_weight * 0.8:
                    add_amount = min(
                        self.base_position_size * 0.5 * portfolio_state.get('portfolio_value', 100000) / current_price,
                        (self.max_position_weight * 0.8 - position_context.position_weight) * portfolio_state.get('portfolio_value', 100000) / current_price
                    )
                    
                    if add_amount > 0:
                        decision = OrderDecision(
                            decision_type=OrderDecisionType.REBALANCE_UP,
                            order_side="buy",
                            order_type="limit",
                            quantity=add_amount,
                            limit_price=current_price * 0.99,  # Buy on slight dip
                            confidence=signal_confidence,
                            reasoning=f"Add to winning position: {signal_type} ({signal_strength:.2f}), current gain: {position_context.unrealized_pnl_pct:.1%}",
                            priority=5,
                            estimated_impact=add_amount * current_price / portfolio_state.get('portfolio_value', 100000)
                        )
                        decisions.append(decision)
            
            return decisions
            
        except Exception as e:
            logger.error(f"Error deciding existing position for {symbol}: {e}")
            return []
    
    async def _decide_portfolio_rebalancing(self, symbol: str, position_context: Optional[PositionContext],
                                          portfolio_state: Dict, current_price: float) -> List[OrderDecision]:
        """Decide on portfolio-level rebalancing needs."""
        decisions = []
        
        try:
            # Check if portfolio needs overall rebalancing
            if not portfolio_state.get('needs_rebalancing', False):
                return decisions
            
            # If portfolio is too concentrated in one position, suggest diversification
            if portfolio_state.get('max_concentration', 0) > self.max_position_weight:
                # This symbol might be the concentrated one
                if position_context and position_context.position_weight > self.max_position_weight:
                    # Already handled in existing position logic
                    pass
            
            # If portfolio needs more exposure and this is a good candidate
            if portfolio_state.get('utilization', 1) < self.target_portfolio_utilization:
                if position_context is None:  # No position in this symbol
                    # Consider as diversification candidate
                    diversification_size = min(
                        self.base_position_size,
                        (self.target_portfolio_utilization - portfolio_state.get('utilization', 0)) / 3
                    )
                    
                    if diversification_size > 0.01:  # Minimum 1% position
                        decision = OrderDecision(
                            decision_type=OrderDecisionType.BUY_NEW,
                            order_side="buy",
                            order_type="limit",
                            quantity=diversification_size * portfolio_state.get('portfolio_value', 100000) / current_price,
                            limit_price=current_price * 0.995,
                            confidence=0.6,
                            reasoning=f"Portfolio diversification: increase utilization from {portfolio_state.get('utilization', 0):.1%}",
                            priority=4,
                            estimated_impact=diversification_size
                        )
                        decisions.append(decision)
            
            return decisions
            
        except Exception as e:
            logger.error(f"Error deciding portfolio rebalancing for {symbol}: {e}")
            return []
    
    def _calculate_position_size(self, signal_strength: float, signal_confidence: float,
                               market_regime: MarketRegime, portfolio_state: Dict) -> float:
        """Calculate appropriate position size based on signal and context."""
        try:
            # Base position size
            base_size = self.base_position_size
            
            # Adjust for signal strength and confidence
            signal_multiplier = signal_strength * signal_confidence
            adjusted_size = base_size * (0.5 + signal_multiplier)
            
            # Adjust for market regime
            regime_multipliers = {
                MarketRegime.BULLISH: 1.2,
                MarketRegime.BEARISH: 0.8,
                MarketRegime.SIDEWAYS: 1.0,
                MarketRegime.VOLATILE: 0.7,
                MarketRegime.LOW_VOLATILITY: 1.1
            }
            adjusted_size *= regime_multipliers.get(market_regime, 1.0)
            
            # Adjust for portfolio utilization
            utilization = portfolio_state.get('utilization', 0.5)
            if utilization > 0.8:
                adjusted_size *= 0.5  # Reduce size when highly utilized
            elif utilization < 0.3:
                adjusted_size *= 1.3  # Increase size when under-utilized
            
            # Cap at maximum single trade size
            adjusted_size = min(adjusted_size, self.max_single_trade_size)
            
            return max(0.01, adjusted_size)  # Minimum 1% position
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self.base_position_size
    
    def _determine_order_execution_type(self, side: str, current_price: float,
                                      signal_strength: float, market_regime: MarketRegime) -> Tuple[str, Optional[float], str]:
        """Determine optimal order type and price for execution."""
        try:
            # For high conviction trades in stable markets, use limit orders
            if signal_strength > 0.7 and market_regime != MarketRegime.VOLATILE:
                if side == "buy":
                    limit_price = current_price * 0.995  # Buy 0.5% below market
                    return "limit", limit_price, "High conviction limit order for better execution"
                else:  # sell or sell_short
                    limit_price = current_price * 1.005  # Sell 0.5% above market
                    return "limit", limit_price, "High conviction limit order for better execution"
            
            # For volatile markets or urgent trades, use market orders
            elif market_regime == MarketRegime.VOLATILE or signal_strength > 0.9:
                return "market", None, "Market order for immediate execution in volatile conditions"
            
            # For moderate conviction, use stop-limit orders for risk management
            elif signal_strength > 0.5:
                if side == "buy":
                    stop_price = current_price * 1.01
                    limit_price = current_price * 1.02
                    return "stop_limit", limit_price, f"Stop-limit order with stop at ${stop_price:.2f}"
                else:
                    stop_price = current_price * 0.99
                    limit_price = current_price * 0.98
                    return "stop_limit", limit_price, f"Stop-limit order with stop at ${stop_price:.2f}"
            
            # Default to limit orders for conservative execution
            else:
                if side == "buy":
                    limit_price = current_price * 0.99
                    return "limit", limit_price, "Conservative limit order"
                else:
                    limit_price = current_price * 1.01
                    return "limit", limit_price, "Conservative limit order"
                    
        except Exception as e:
            logger.error(f"Error determining order execution type: {e}")
            return "market", None, "Fallback market order"
    
    def _validate_decisions(self, decisions: List[OrderDecision], 
                          current_portfolio: Dict) -> List[OrderDecision]:
        """Validate and filter order decisions."""
        validated_decisions = []
        
        try:
            portfolio_value = current_portfolio.get('equity', 100000)
            cash = current_portfolio.get('cash', 50000)
            buying_power = current_portfolio.get('buying_power', cash)
            
            total_buy_impact = 0
            total_sell_impact = 0
            
            # Sort decisions by priority (higher priority first)
            decisions.sort(key=lambda d: d.priority, reverse=True)
            
            for decision in decisions:
                # Skip zero quantity decisions
                if decision.quantity <= 0:
                    continue
                
                # Estimate order value
                if hasattr(decision, 'limit_price') and decision.limit_price:
                    estimated_price = decision.limit_price
                else:
                    # Use current market price estimate
                    estimated_price = 100  # Fallback price
                
                order_value = decision.quantity * estimated_price
                
                # Check buying power for buy orders
                if decision.order_side in ["buy", "buy_cover"]:
                    if total_buy_impact + order_value > buying_power * 0.95:
                        logger.debug(f"Skipping buy decision: insufficient buying power")
                        continue
                    total_buy_impact += order_value
                
                # Check position sizes don't exceed limits
                if abs(decision.estimated_impact) > self.max_single_trade_size:
                    logger.debug(f"Reducing decision size: {decision.estimated_impact:.2%} > {self.max_single_trade_size:.2%}")
                    scale_factor = self.max_single_trade_size / abs(decision.estimated_impact)
                    decision.quantity *= scale_factor
                    decision.estimated_impact *= scale_factor
                
                validated_decisions.append(decision)
            
            return validated_decisions
            
        except Exception as e:
            logger.error(f"Error validating decisions: {e}")
            return decisions
    
    async def _add_risk_management(self, decisions: List[OrderDecision], symbol: str) -> List[OrderDecision]:
        """Add risk management orders (stop-loss, take-profit) to decisions."""
        try:
            risk_managed_decisions = []
            
            for decision in decisions:
                # Add the main decision
                risk_managed_decisions.append(decision)
                
                # Add risk management for new position decisions
                if decision.decision_type in [OrderDecisionType.BUY_NEW, OrderDecisionType.SELL_SHORT]:
                    if decision.order_side == "buy":
                        # Add stop-loss and take-profit for long positions
                        if not decision.stop_loss_price:
                            current_price = alpaca_client.get_current_price(symbol)
                            if current_price:
                                decision.stop_loss_price = current_price * (1 - self.default_stop_loss_pct)
                                decision.take_profit_price = current_price * (1 + self.default_take_profit_pct)
                    
                    elif decision.order_side == "sell_short":
                        # Add stop-loss and take-profit for short positions
                        if not decision.stop_loss_price:
                            current_price = alpaca_client.get_current_price(symbol)
                            if current_price:
                                decision.stop_loss_price = current_price * (1 + self.default_stop_loss_pct)
                                decision.take_profit_price = current_price * (1 - self.default_take_profit_pct)
            
            return risk_managed_decisions
            
        except Exception as e:
            logger.error(f"Error adding risk management for {symbol}: {e}")
            return decisions
    
    def _can_trade_symbol(self, symbol: str) -> bool:
        """Check if we can trade this symbol (anti-overtrading)."""
        if symbol not in self.recent_orders:
            return True
        
        last_order_time = self.recent_orders[symbol]
        return datetime.now() - last_order_time > self.anti_overtrading_window
    
    def _reset_daily_counters(self):
        """Reset daily trade counters if new day."""
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trade_count = 0
            self.last_trade_date = today
    
    def record_order_execution(self, symbol: str):
        """Record order execution for tracking."""
        self.recent_orders[symbol] = datetime.now()
        self.daily_trade_count += 1
    
    async def execute_decisions(self, decisions: List[OrderDecision], symbol: str) -> List[Dict]:
        """Execute the order decisions."""
        executed_orders = []
        
        try:
            for decision in decisions:
                if decision.decision_type == OrderDecisionType.HOLD:
                    continue
                
                # Create order request
                order_request = AdvancedOrderRequest(
                    symbol=symbol,
                    quantity=decision.quantity,
                    side=decision.order_side,
                    order_type=decision.order_type,
                    limit_price=decision.limit_price,
                    stop_price=decision.stop_price,
                    stop_loss_price=decision.stop_loss_price,
                    take_profit_price=decision.take_profit_price,
                    reasoning=decision.reasoning,
                    confidence=decision.confidence
                )
                
                # Execute order
                result = await self.advanced_order_manager.place_advanced_order(order_request)
                
                if result.success:
                    self.record_order_execution(symbol)
                    executed_orders.append({
                        'order_id': result.order_id,
                        'symbol': symbol,
                        'side': decision.order_side,
                        'quantity': decision.quantity,
                        'type': decision.order_type,
                        'decision_type': decision.decision_type.value,
                        'reasoning': decision.reasoning,
                        'timestamp': datetime.now()
                    })
                    logger.info(f"Executed {decision.decision_type.value} order for {symbol}: {decision.reasoning}")
                else:
                    logger.error(f"Failed to execute order for {symbol}: {result.error_message}")
            
            return executed_orders
            
        except Exception as e:
            logger.error(f"Error executing decisions for {symbol}: {e}")
            return executed_orders

# Global order decision engine instance
order_decision_engine = OrderDecisionEngine()

# Convenience functions
async def analyze_and_decide_orders(symbol: str, signal_data: Dict, 
                                  current_portfolio: Dict) -> List[OrderDecision]:
    """Analyze signal and decide on orders."""
    return await order_decision_engine.analyze_and_decide(symbol, signal_data, current_portfolio)

async def execute_order_decisions(decisions: List[OrderDecision], symbol: str) -> List[Dict]:
    """Execute order decisions."""
    return await order_decision_engine.execute_decisions(decisions, symbol)

async def analyze_enhanced_short_opportunity(symbol: str, current_portfolio: Dict) -> Optional[OrderDecision]:
    """Analyze enhanced short opportunity for a symbol."""
    return await order_decision_engine.analyze_enhanced_short_opportunity(symbol, current_portfolio)

def get_decision_engine_status() -> Dict[str, Any]:
    """Get order decision engine status."""
    return {
        'daily_trade_count': order_decision_engine.daily_trade_count,
        'max_daily_trades': order_decision_engine.max_daily_trades,
        'recent_orders_count': len(order_decision_engine.recent_orders),
        'anti_overtrading_window_hours': order_decision_engine.anti_overtrading_window.total_seconds() / 3600,
        'target_portfolio_utilization': order_decision_engine.target_portfolio_utilization,
        'max_position_weight': order_decision_engine.max_position_weight
    }