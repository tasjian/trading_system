#!/usr/bin/env python3
"""
Enhanced Signal Processing System

Converts trading signals into specific order types (buy/sell/short/limit) based on
market conditions, risk management, and portfolio context.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio

from core.portfolio_balancer import PositionAction, OrderUrgency, portfolio_balancer
from tools.alpaca_client import alpaca_client
from config.settings import settings

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Enhanced signal types."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    HOLD = "hold"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    SHORT = "short"
    COVER = "cover"

class MarketRegime(Enum):
    """Market regime classifications."""
    BULL_MARKET = "bull"
    BEAR_MARKET = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"
    CRISIS = "crisis"

@dataclass
class ProcessedSignal:
    """Enhanced processed trading signal."""
    symbol: str
    signal_type: SignalType
    action: PositionAction
    strength: float
    confidence: float
    order_type: str  # "market", "limit", "stop_limit"
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    urgency: OrderUrgency = OrderUrgency.MEDIUM
    reasoning: str = ""
    source_signals: List[str] = field(default_factory=list)
    risk_adjustments: Dict[str, float] = field(default_factory=dict)
    market_context: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class MarketContext:
    """Market context for signal processing."""
    volatility: float
    trend_direction: float  # -1 to 1
    regime: MarketRegime
    liquidity_score: float
    risk_off_sentiment: float
    sector_rotation: Dict[str, float] = field(default_factory=dict)

class EnhancedSignalProcessor:
    """
    Enhanced signal processor that converts raw signals into actionable orders
    with proper consideration for portfolio context and risk management.
    """
    
    def __init__(self):
        """Initialize signal processor."""
        self.signal_aggregation_window = timedelta(minutes=15)
        self.min_signal_confidence = 0.3
        self.min_signal_strength = 0.1
        
        # Signal strength thresholds
        self.strength_thresholds = {
            'strong': 0.7,
            'medium': 0.4,
            'weak': 0.1
        }
        
        # Market regime adjustments
        self.regime_adjustments = {
            MarketRegime.BULL_MARKET: {'buy_boost': 1.1, 'sell_penalty': 0.9},
            MarketRegime.BEAR_MARKET: {'buy_penalty': 0.9, 'sell_boost': 1.1},
            MarketRegime.HIGH_VOLATILITY: {'market_order_threshold': 0.05},
            MarketRegime.CRISIS: {'risk_reduction': 0.5, 'prefer_market_orders': True}
        }
    
    async def process_signals(self, 
                            raw_signals: List[Dict],
                            current_positions: Dict[str, Dict],
                            target_allocation: Dict[str, float],
                            market_context: Optional[MarketContext] = None) -> List[ProcessedSignal]:
        """
        Process raw signals into actionable orders with portfolio context.
        
        Args:
            raw_signals: List of raw trading signals
            current_positions: Current portfolio positions
            target_allocation: Target portfolio allocation
            market_context: Current market conditions
            
        Returns:
            List of processed signals ready for execution
        """
        logger.info(f"📊 Processing {len(raw_signals)} raw signals...")
        
        if not market_context:
            market_context = await self._get_market_context()
        
        # Group signals by symbol
        symbol_signals = self._group_signals_by_symbol(raw_signals)
        
        processed_signals = []
        
        for symbol, signals in symbol_signals.items():
            # Skip if insufficient signal quality
            if not self._validate_signal_quality(signals):
                logger.debug(f"Skipping {symbol}: insufficient signal quality")
                continue
            
            # Aggregate multiple signals for the same symbol
            aggregated_signal = self._aggregate_symbol_signals(symbol, signals, market_context)
            if not aggregated_signal:
                continue
            
            # Get position context
            current_position = current_positions.get(symbol, {'quantity': 0, 'market_value': 0})
            target_weight = target_allocation.get(symbol, 0.0)
            
            # Process signal with portfolio context
            processed_signal = await self._process_single_signal(
                aggregated_signal, current_position, target_weight, market_context
            )
            
            if processed_signal:
                processed_signals.append(processed_signal)
        
        # Apply portfolio-level risk management
        risk_adjusted_signals = await self._apply_portfolio_risk_management(
            processed_signals, current_positions, market_context
        )
        
        logger.info(f"✅ Processed {len(risk_adjusted_signals)} actionable signals")
        return risk_adjusted_signals
    
    def _group_signals_by_symbol(self, raw_signals: List[Dict]) -> Dict[str, List[Dict]]:
        """Group raw signals by symbol."""
        symbol_groups = {}
        
        for signal in raw_signals:
            symbol = signal.get('symbol', '').upper()
            if not symbol:
                continue
            
            if symbol not in symbol_groups:
                symbol_groups[symbol] = []
            symbol_groups[symbol].append(signal)
        
        return symbol_groups
    
    def _validate_signal_quality(self, signals: List[Dict]) -> bool:
        """Validate signal quality for processing."""
        if not signals:
            return False
        
        # Check for minimum confidence and strength
        valid_signals = [
            s for s in signals 
            if s.get('confidence', 0) >= self.min_signal_confidence
            and s.get('strength', 0) >= self.min_signal_strength
        ]
        
        return len(valid_signals) > 0
    
    def _aggregate_symbol_signals(self, 
                                symbol: str, 
                                signals: List[Dict], 
                                market_context: MarketContext) -> Optional[Dict]:
        """Aggregate multiple signals for the same symbol."""
        
        if not signals:
            return None
        
        # Filter valid signals
        valid_signals = [
            s for s in signals 
            if s.get('confidence', 0) >= self.min_signal_confidence
        ]
        
        if not valid_signals:
            return None
        
        # Aggregate signal components
        total_weight = sum(s.get('confidence', 0.5) for s in valid_signals)
        
        if total_weight == 0:
            return None
        
        # Weighted aggregation
        aggregated_strength = sum(
            s.get('strength', 0) * s.get('confidence', 0.5) for s in valid_signals
        ) / total_weight
        
        aggregated_confidence = sum(s.get('confidence', 0.5) for s in valid_signals) / len(valid_signals)
        
        # Determine dominant signal type
        signal_votes = {}
        for signal in valid_signals:
            signal_type = signal.get('signal', '').lower()
            confidence = signal.get('confidence', 0.5)
            
            if signal_type not in signal_votes:
                signal_votes[signal_type] = 0
            signal_votes[signal_type] += confidence
        
        dominant_signal = max(signal_votes.keys(), key=lambda k: signal_votes[k]) if signal_votes else 'hold'
        
        # Collect source information
        sources = [s.get('source', 'unknown') for s in valid_signals]
        reasoning_parts = [s.get('reasoning', '') for s in valid_signals if s.get('reasoning')]
        
        return {
            'symbol': symbol,
            'signal': dominant_signal,
            'strength': aggregated_strength,
            'confidence': aggregated_confidence,
            'sources': sources,
            'reasoning': ' | '.join(reasoning_parts[:3]),  # Limit reasoning length
            'raw_signals_count': len(valid_signals)
        }
    
    async def _process_single_signal(self, 
                                   aggregated_signal: Dict,
                                   current_position: Dict,
                                   target_weight: float,
                                   market_context: MarketContext) -> Optional[ProcessedSignal]:
        """Process a single aggregated signal with portfolio context."""
        
        symbol = aggregated_signal['symbol']
        signal_str = aggregated_signal['signal'].lower()
        strength = aggregated_signal['strength']
        confidence = aggregated_signal['confidence']
        
        # Convert signal string to enum
        signal_type = self._convert_signal_string_to_enum(signal_str, strength)
        
        # Determine position action based on signal and current position
        current_quantity = float(current_position.get('quantity', 0))
        position_action = self._determine_position_action(
            signal_type, current_quantity, target_weight, market_context
        )
        
        if position_action == PositionAction.HOLD:
            return None
        
        # Calculate order quantity
        quantity = await self._calculate_order_quantity(
            symbol, position_action, current_position, target_weight, strength, confidence
        )
        
        if quantity <= 0:
            return None
        
        # Determine order type and pricing
        order_type, limit_price, stop_price = await self._determine_order_execution(
            symbol, position_action, market_context, confidence
        )
        
        # Determine urgency
        urgency = self._determine_order_urgency(strength, confidence, market_context)
        
        # Apply market regime adjustments
        adjusted_confidence, adjusted_strength = self._apply_regime_adjustments(
            signal_type, confidence, strength, market_context
        )
        
        # Generate enhanced reasoning
        reasoning = self._generate_signal_reasoning(
            aggregated_signal, position_action, current_position, market_context
        )
        
        return ProcessedSignal(
            symbol=symbol,
            signal_type=signal_type,
            action=position_action,
            strength=adjusted_strength,
            confidence=adjusted_confidence,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            urgency=urgency,
            reasoning=reasoning,
            source_signals=aggregated_signal.get('sources', []),
            market_context={
                'volatility': market_context.volatility,
                'trend': market_context.trend_direction,
                'regime': market_context.regime.value
            }
        )
    
    def _convert_signal_string_to_enum(self, signal_str: str, strength: float) -> SignalType:
        """Convert signal string to SignalType enum with strength consideration."""
        
        signal_lower = signal_str.lower()
        
        # Map signal strings to types with strength modifiers
        if 'strong_buy' in signal_lower or (signal_lower == 'buy' and strength > self.strength_thresholds['strong']):
            return SignalType.STRONG_BUY
        elif 'buy' in signal_lower:
            if strength > self.strength_thresholds['medium']:
                return SignalType.BUY
            else:
                return SignalType.WEAK_BUY
        elif 'strong_sell' in signal_lower or 'short' in signal_lower:
            return SignalType.SHORT
        elif 'sell' in signal_lower:
            if strength > self.strength_thresholds['medium']:
                return SignalType.SELL
            else:
                return SignalType.WEAK_SELL
        elif 'cover' in signal_lower:
            return SignalType.COVER
        else:
            return SignalType.HOLD
    
    def _determine_position_action(self, 
                                 signal_type: SignalType,
                                 current_quantity: float,
                                 target_weight: float,
                                 market_context: MarketContext) -> PositionAction:
        """Determine position action based on signal and context."""
        
        # Handle different signal types
        if signal_type in [SignalType.STRONG_BUY, SignalType.BUY, SignalType.WEAK_BUY]:
            if current_quantity <= 0:
                return PositionAction.BUY  # New long position or cover short
            else:
                # Consider adding to position based on target weight
                return PositionAction.BUY if target_weight > 0 else PositionAction.HOLD
        
        elif signal_type in [SignalType.SELL, SignalType.WEAK_SELL]:
            if current_quantity > 0:
                return PositionAction.SELL  # Reduce or close long position
            else:
                return PositionAction.HOLD  # Don't sell if no position
        
        elif signal_type == SignalType.SHORT:
            if current_quantity >= 0:
                return PositionAction.SHORT  # New short position
            else:
                return PositionAction.SHORT  # Add to short position
        
        elif signal_type == SignalType.COVER:
            if current_quantity < 0:
                return PositionAction.COVER  # Cover short position
            else:
                return PositionAction.HOLD
        
        else:  # HOLD or unknown
            return PositionAction.HOLD
    
    async def _calculate_order_quantity(self, 
                                      symbol: str,
                                      action: PositionAction,
                                      current_position: Dict,
                                      target_weight: float,
                                      strength: float,
                                      confidence: float) -> float:
        """Calculate order quantity based on position action and risk parameters."""
        
        # Get portfolio value for calculations
        try:
            account_info = alpaca_client.get_account_info()
            portfolio_value = float(account_info.get('portfolio_value', 100000))
        except:
            portfolio_value = 100000  # Fallback
        
        # Get current price
        try:
            current_price = alpaca_client.get_current_price(symbol)
            if not current_price or current_price <= 0:
                return 0
            current_price = float(current_price)
        except:
            return 0
        
        current_quantity = float(current_position.get('quantity', 0))
        current_value = float(current_position.get('market_value', 0))
        
        # Calculate base position size
        if action in [PositionAction.BUY, PositionAction.SHORT]:
            # New position or addition
            target_value = target_weight * portfolio_value
            
            # Adjust target based on signal strength and confidence
            signal_multiplier = (strength * confidence) ** 0.5  # Geometric mean for balance
            adjusted_target_value = target_value * signal_multiplier
            
            # Calculate quantity needed
            if action == PositionAction.BUY:
                if current_quantity >= 0:
                    # Add to long position or new long
                    quantity_needed = (adjusted_target_value - current_value) / current_price
                else:
                    # Cover short and go long
                    cover_quantity = abs(current_quantity)
                    new_long_quantity = adjusted_target_value / current_price
                    quantity_needed = cover_quantity + new_long_quantity
            else:  # SHORT
                quantity_needed = adjusted_target_value / current_price
            
        elif action in [PositionAction.SELL, PositionAction.COVER]:
            # Reducing or closing positions
            if action == PositionAction.SELL and current_quantity > 0:
                # Sell portion based on signal strength
                sell_fraction = min(strength * confidence, 1.0)
                quantity_needed = current_quantity * sell_fraction
            elif action == PositionAction.COVER and current_quantity < 0:
                # Cover portion of short position
                cover_fraction = min(strength * confidence, 1.0)
                quantity_needed = abs(current_quantity) * cover_fraction
            else:
                quantity_needed = 0
        
        else:
            quantity_needed = 0
        
        # Apply risk limits
        max_position_value = portfolio_value * settings.max_position_size
        max_quantity = max_position_value / current_price
        
        # Ensure minimum viable trade size
        min_trade_value = 100  # $100 minimum
        min_quantity = min_trade_value / current_price
        
        final_quantity = max(min_quantity, min(quantity_needed, max_quantity))
        
        return max(0, final_quantity)
    
    async def _determine_order_execution(self, 
                                       symbol: str,
                                       action: PositionAction,
                                       market_context: MarketContext,
                                       confidence: float) -> Tuple[str, Optional[float], Optional[float]]:
        """Determine order type and execution parameters."""
        
        # Get current price for limit order calculations
        try:
            current_price = alpaca_client.get_current_price(symbol)
            if not current_price:
                return "market", None, None
            current_price = float(current_price)
        except:
            return "market", None, None
        
        # Use market orders in high volatility or crisis conditions
        if market_context.regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.CRISIS]:
            return "market", None, None
        
        # Use market orders for low confidence signals (quick execution)
        if confidence < 0.5:
            return "market", None, None
        
        # Use limit orders for normal conditions with intelligent pricing
        buffer_pct = 0.002  # 0.2% buffer
        
        if action in [PositionAction.BUY, PositionAction.COVER]:
            # Buying: set limit slightly above current price
            limit_price = current_price * (1 + buffer_pct)
            return "limit", limit_price, None
        
        elif action in [PositionAction.SELL, PositionAction.SHORT]:
            # Selling: set limit slightly below current price
            limit_price = current_price * (1 - buffer_pct)
            return "limit", limit_price, None
        
        return "market", None, None
    
    def _determine_order_urgency(self, 
                               strength: float,
                               confidence: float,
                               market_context: MarketContext) -> OrderUrgency:
        """Determine order execution urgency."""
        
        # Base urgency from signal quality
        signal_score = strength * confidence
        
        if market_context.regime == MarketRegime.CRISIS:
            return OrderUrgency.CRITICAL
        
        if signal_score > 0.8:
            return OrderUrgency.HIGH
        elif signal_score > 0.5:
            return OrderUrgency.MEDIUM
        else:
            return OrderUrgency.LOW
    
    def _apply_regime_adjustments(self, 
                                signal_type: SignalType,
                                confidence: float,
                                strength: float,
                                market_context: MarketContext) -> Tuple[float, float]:
        """Apply market regime adjustments to signal strength and confidence."""
        
        regime = market_context.regime
        adjustments = self.regime_adjustments.get(regime, {})
        
        adjusted_confidence = confidence
        adjusted_strength = strength
        
        # Apply regime-specific adjustments
        if signal_type in [SignalType.STRONG_BUY, SignalType.BUY, SignalType.WEAK_BUY]:
            boost = adjustments.get('buy_boost', 1.0)
            penalty = adjustments.get('buy_penalty', 1.0)
            adjusted_strength *= boost * penalty
        elif signal_type in [SignalType.SELL, SignalType.WEAK_SELL, SignalType.SHORT]:
            boost = adjustments.get('sell_boost', 1.0)
            penalty = adjustments.get('sell_penalty', 1.0)
            adjusted_strength *= boost * penalty
        
        # Apply risk reduction in crisis
        if 'risk_reduction' in adjustments:
            adjusted_strength *= adjustments['risk_reduction']
            adjusted_confidence *= adjustments['risk_reduction']
        
        # Clamp values
        adjusted_confidence = max(0.1, min(0.95, adjusted_confidence))
        adjusted_strength = max(0.1, min(1.0, adjusted_strength))
        
        return adjusted_confidence, adjusted_strength
    
    def _generate_signal_reasoning(self, 
                                 aggregated_signal: Dict,
                                 action: PositionAction,
                                 current_position: Dict,
                                 market_context: MarketContext) -> str:
        """Generate comprehensive reasoning for the signal."""
        
        base_reasoning = aggregated_signal.get('reasoning', '')
        sources_count = len(aggregated_signal.get('sources', []))
        confidence = aggregated_signal.get('confidence', 0.5)
        strength = aggregated_signal.get('strength', 0.5)
        
        reasoning_parts = []
        
        # Signal quality
        reasoning_parts.append(f"Signal: {strength:.1%} strength, {confidence:.1%} confidence")
        
        # Sources
        if sources_count > 1:
            reasoning_parts.append(f"{sources_count} data sources")
        
        # Position context
        current_qty = current_position.get('quantity', 0)
        if current_qty != 0:
            position_type = "long" if current_qty > 0 else "short"
            reasoning_parts.append(f"Current: {abs(current_qty):.0f} shares {position_type}")
        
        # Market context
        regime_desc = market_context.regime.value.replace('_', ' ').title()
        reasoning_parts.append(f"Market: {regime_desc}")
        
        # Action
        action_desc = action.value.replace('_', ' ').title()
        reasoning_parts.append(f"Action: {action_desc}")
        
        # Combine with original reasoning
        final_reasoning = f"{base_reasoning} | {' | '.join(reasoning_parts)}"
        
        return final_reasoning[:200]  # Limit length
    
    async def _apply_portfolio_risk_management(self, 
                                             signals: List[ProcessedSignal],
                                             current_positions: Dict[str, Dict],
                                             market_context: MarketContext) -> List[ProcessedSignal]:
        """Apply portfolio-level risk management to processed signals."""
        
        if not signals:
            return signals
        
        # Calculate portfolio metrics
        total_buy_value = sum(
            s.quantity * (s.limit_price or await self._get_signal_price(s.symbol))
            for s in signals if s.action in [PositionAction.BUY, PositionAction.COVER]
        )
        
        total_sell_value = sum(
            s.quantity * (s.limit_price or await self._get_signal_price(s.symbol))
            for s in signals if s.action in [PositionAction.SELL, PositionAction.SHORT]
        )
        
        # Get current portfolio value
        try:
            account_info = alpaca_client.get_account_info()
            portfolio_value = float(account_info.get('portfolio_value', 100000))
            cash_available = float(account_info.get('cash', 0))
        except:
            portfolio_value = 100000
            cash_available = 50000
        
        # Apply cash limit constraint
        if total_buy_value > cash_available * 0.95:  # 95% of available cash
            scale_factor = (cash_available * 0.95) / total_buy_value
            for signal in signals:
                if signal.action in [PositionAction.BUY, PositionAction.COVER]:
                    signal.quantity *= scale_factor
                    signal.risk_adjustments['cash_limit_scaling'] = scale_factor
        
        # Apply diversification limits
        sector_exposure = {}
        for signal in signals:
            sector = signal.market_context.get('sector', 'Unknown')
            if sector not in sector_exposure:
                sector_exposure[sector] = 0
            
            signal_value = signal.quantity * (signal.limit_price or await self._get_signal_price(signal.symbol))
            sector_exposure[sector] += signal_value
        
        # Scale down over-concentrated sectors
        max_sector_value = portfolio_value * settings.max_sector_allocation
        for sector, exposure in sector_exposure.items():
            if exposure > max_sector_value:
                scale_factor = max_sector_value / exposure
                for signal in signals:
                    signal_sector = signal.market_context.get('sector', 'Unknown')
                    if signal_sector == sector:
                        signal.quantity *= scale_factor
                        signal.risk_adjustments['sector_limit_scaling'] = scale_factor
        
        # Filter out signals that became too small
        min_trade_value = 100
        filtered_signals = []
        for signal in signals:
            signal_price = signal.limit_price or await self._get_signal_price(signal.symbol)
            if signal_price and signal.quantity * signal_price >= min_trade_value:
                filtered_signals.append(signal)
        
        return filtered_signals
    
    async def _get_signal_price(self, symbol: str) -> float:
        """Get current price for a symbol (used in risk calculations)."""
        try:
            price = alpaca_client.get_current_price(symbol)
            return float(price) if price else 0
        except:
            return 0
    
    async def _get_market_context(self) -> MarketContext:
        """Get current market context (simplified version)."""
        
        # This would typically integrate with market data providers
        # For now, return a default context
        return MarketContext(
            volatility=0.02,  # 2% daily volatility
            trend_direction=0.0,  # Neutral
            regime=MarketRegime.SIDEWAYS,
            liquidity_score=0.8,
            risk_off_sentiment=0.3
        )

# Global instance
signal_processor = EnhancedSignalProcessor()

# Convenience functions
async def process_signals(raw_signals: List[Dict],
                         current_positions: Dict[str, Dict],
                         target_allocation: Dict[str, float],
                         market_context: Optional[MarketContext] = None) -> List[ProcessedSignal]:
    """Process raw signals into actionable orders."""
    return await signal_processor.process_signals(
        raw_signals, current_positions, target_allocation, market_context
    )

__all__ = [
    'SignalType', 'MarketRegime', 'ProcessedSignal', 'MarketContext',
    'EnhancedSignalProcessor', 'signal_processor', 'process_signals'
]