#!/usr/bin/env python3
"""
Automated Risk Management System

MLOps framework for automated risk management including stop-loss execution,
take-profit automation, position sizing controls, and dynamic risk adjustments.

Features:
- Automated stop-loss and take-profit execution
- Dynamic position sizing based on volatility and risk
- Portfolio-level risk monitoring and intervention
- Automated hedging strategies
- Risk-adjusted order flow management
- Real-time risk metric calculations
- Automated circuit breakers and trading halts
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict

from tools.alpaca_client import alpaca_client
from order_types.advanced_orders import AdvancedOrderManager, AdvancedOrderRequest, place_oco_order
from monitoring.portfolio_monitoring import portfolio_monitor, PortfolioSnapshot
from tools.risk_controls import risk_monitor
from config.settings import settings

logger = logging.getLogger(__name__)

class RiskAction(Enum):
    NONE = "none"
    REDUCE_POSITION = "reduce_position"
    CLOSE_POSITION = "close_position"
    ADD_HEDGE = "add_hedge"
    TIGHTEN_STOPS = "tighten_stops"
    INCREASE_CASH = "increase_cash"
    HALT_TRADING = "halt_trading"

class RiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskManagementRule:
    """Risk management rule definition."""
    name: str
    condition: str  # Python expression to evaluate
    action: RiskAction
    priority: int  # 1-10, higher = more urgent
    cooldown_minutes: int = 60  # Minimum time between actions
    threshold_value: float = 0.0
    enabled: bool = True

@dataclass
class AutomatedRiskEvent:
    """Automated risk management event."""
    timestamp: datetime
    rule_name: str
    symbol: Optional[str]
    action_taken: RiskAction
    trigger_value: float
    threshold_value: float
    description: str
    order_ids: List[str] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None

class AutomatedRiskManager:
    """Automated risk management system with real-time monitoring and intervention."""
    
    def __init__(self):
        """Initialize automated risk manager."""
        self.advanced_order_manager = AdvancedOrderManager()
        
        # Risk management state
        self.is_active = True
        self.last_risk_check = None
        self.risk_events: List[AutomatedRiskEvent] = []
        self.active_stop_orders: Dict[str, List[str]] = defaultdict(list)  # symbol -> order_ids
        self.position_risk_scores: Dict[str, float] = {}
        
        # Risk thresholds and parameters
        self.risk_params = {
            # Position-level risk
            'max_position_loss': 0.15,      # 15% max loss per position
            'max_daily_position_loss': 0.10, # 10% max daily loss per position
            'trailing_stop_distance': 0.05,  # 5% trailing stop
            'profit_protection_threshold': 0.20, # Protect profits above 20%
            
            # Portfolio-level risk
            'max_portfolio_loss': 0.08,     # 8% max portfolio loss
            'max_daily_portfolio_loss': 0.05, # 5% max daily portfolio loss
            'max_drawdown': 0.12,           # 12% max drawdown
            'concentration_limit': 0.25,    # 25% max single position
            'correlation_limit': 0.80,      # 80% max correlation
            
            # Volatility-based risk
            'volatility_multiplier': 2.0,   # Scale stop losses by volatility
            'max_portfolio_volatility': 0.25, # 25% max portfolio volatility
            'volatility_lookback_days': 20,  # Days for volatility calculation
            
            # Market risk
            'vix_spike_threshold': 35,       # VIX level for defensive action
            'market_crash_threshold': -0.05, # 5% market drop in day
            'sector_concentration_limit': 0.40, # 40% max in single sector
        }
        
        # Risk management rules
        self.risk_rules = self._initialize_risk_rules()
        
        # Rule execution tracking
        self.rule_last_executed: Dict[str, datetime] = {}
        
        # Performance tracking
        self.risk_adjusted_returns = []
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        
    def _initialize_risk_rules(self) -> List[RiskManagementRule]:
        """Initialize predefined risk management rules."""
        return [
            # Position-level rules
            RiskManagementRule(
                name="stop_loss_trigger",
                condition="position_loss > max_position_loss",
                action=RiskAction.CLOSE_POSITION,
                priority=9,
                cooldown_minutes=5,
                threshold_value=self.risk_params['max_position_loss']
            ),
            RiskManagementRule(
                name="daily_position_loss_limit",
                condition="daily_position_loss > max_daily_position_loss",
                action=RiskAction.CLOSE_POSITION,
                priority=8,
                cooldown_minutes=30,
                threshold_value=self.risk_params['max_daily_position_loss']
            ),
            RiskManagementRule(
                name="trailing_stop_update",
                condition="position_profit > profit_protection_threshold and not_has_trailing_stop",
                action=RiskAction.TIGHTEN_STOPS,
                priority=6,
                cooldown_minutes=15,
                threshold_value=self.risk_params['profit_protection_threshold']
            ),
            RiskManagementRule(
                name="concentration_risk",
                condition="position_weight > concentration_limit",
                action=RiskAction.REDUCE_POSITION,
                priority=7,
                cooldown_minutes=60,
                threshold_value=self.risk_params['concentration_limit']
            ),
            
            # Portfolio-level rules
            RiskManagementRule(
                name="portfolio_stop_loss",
                condition="portfolio_loss > max_portfolio_loss",
                action=RiskAction.HALT_TRADING,
                priority=10,
                cooldown_minutes=120,
                threshold_value=self.risk_params['max_portfolio_loss']
            ),
            RiskManagementRule(
                name="daily_portfolio_loss",
                condition="daily_portfolio_loss > max_daily_portfolio_loss",
                action=RiskAction.INCREASE_CASH,
                priority=8,
                cooldown_minutes=30,
                threshold_value=self.risk_params['max_daily_portfolio_loss']
            ),
            RiskManagementRule(
                name="drawdown_protection",
                condition="drawdown > max_drawdown",
                action=RiskAction.REDUCE_POSITION,
                priority=9,
                cooldown_minutes=60,
                threshold_value=self.risk_params['max_drawdown']
            ),
            
            # Market risk rules
            RiskManagementRule(
                name="volatility_spike",
                condition="portfolio_volatility > max_portfolio_volatility",
                action=RiskAction.ADD_HEDGE,
                priority=7,
                cooldown_minutes=120,
                threshold_value=self.risk_params['max_portfolio_volatility']
            ),
            RiskManagementRule(
                name="market_crash_protection",
                condition="market_drop > market_crash_threshold",
                action=RiskAction.ADD_HEDGE,
                priority=8,
                cooldown_minutes=60,
                threshold_value=self.risk_params['market_crash_threshold']
            )
        ]
    
    async def monitor_and_manage_risk(self, portfolio_snapshot: PortfolioSnapshot) -> List[AutomatedRiskEvent]:
        """Main risk monitoring and management function."""
        try:
            if not self.is_active:
                return []
            
            logger.debug("Running automated risk management checks...")
            
            events = []
            
            # Update risk scores
            await self._update_position_risk_scores(portfolio_snapshot)
            
            # Check position-level risks
            position_events = await self._check_position_risks(portfolio_snapshot)
            events.extend(position_events)
            
            # Check portfolio-level risks
            portfolio_events = await self._check_portfolio_risks(portfolio_snapshot)
            events.extend(portfolio_events)
            
            # Check market risks
            market_events = await self._check_market_risks(portfolio_snapshot)
            events.extend(market_events)
            
            # Execute risk management actions
            execution_events = await self._execute_risk_actions(events)
            events.extend(execution_events)
            
            # Store events
            self.risk_events.extend(events)
            self._cleanup_old_events()
            
            # Update performance metrics
            await self._update_performance_metrics(portfolio_snapshot)
            
            self.last_risk_check = datetime.now()
            
            if events:
                logger.info(f"Automated risk management: {len(events)} events processed")
            
            return events
            
        except Exception as e:
            logger.error(f"Error in automated risk management: {e}")
            return []
    
    async def _update_position_risk_scores(self, portfolio_snapshot: PortfolioSnapshot):
        """Update risk scores for all positions."""
        try:
            for position in portfolio_snapshot.positions:
                symbol = position.symbol
                
                # Calculate basic risk score
                position_weight = abs(position.market_value) / portfolio_snapshot.total_value if portfolio_snapshot.total_value > 0 else 0
                unrealized_loss = min(0, position.unrealized_pnl_pct)
                
                # Get volatility for the position
                volatility = await self._calculate_position_volatility(symbol)
                
                # Risk score components
                concentration_risk = position_weight / self.risk_params['concentration_limit']
                loss_risk = abs(unrealized_loss) / self.risk_params['max_position_loss']
                volatility_risk = volatility / 0.30  # 30% volatility as baseline
                
                # Combined risk score (0-10 scale)
                risk_score = min(10, (concentration_risk + loss_risk + volatility_risk) * 3.33)
                
                self.position_risk_scores[symbol] = risk_score
                
        except Exception as e:
            logger.error(f"Error updating position risk scores: {e}")
    
    async def _check_position_risks(self, portfolio_snapshot: PortfolioSnapshot) -> List[AutomatedRiskEvent]:
        """Check position-level risks and generate events."""
        events = []
        
        try:
            for position in portfolio_snapshot.positions:
                symbol = position.symbol
                position_weight = abs(position.market_value) / portfolio_snapshot.total_value if portfolio_snapshot.total_value > 0 else 0
                
                # Check stop loss trigger
                loss_pct = abs(min(0, position.unrealized_pnl_pct))
                if loss_pct > self.risk_params['max_position_loss']:
                    if self._can_execute_rule("stop_loss_trigger"):
                        event = AutomatedRiskEvent(
                            timestamp=datetime.now(),
                            rule_name="stop_loss_trigger",
                            symbol=symbol,
                            action_taken=RiskAction.CLOSE_POSITION,
                            trigger_value=loss_pct,
                            threshold_value=self.risk_params['max_position_loss'],
                            description=f"Position loss {loss_pct:.1%} exceeds {self.risk_params['max_position_loss']:.1%} limit"
                        )
                        events.append(event)
                
                # Check concentration risk
                if position_weight > self.risk_params['concentration_limit']:
                    if self._can_execute_rule("concentration_risk"):
                        event = AutomatedRiskEvent(
                            timestamp=datetime.now(),
                            rule_name="concentration_risk",
                            symbol=symbol,
                            action_taken=RiskAction.REDUCE_POSITION,
                            trigger_value=position_weight,
                            threshold_value=self.risk_params['concentration_limit'],
                            description=f"Position weight {position_weight:.1%} exceeds {self.risk_params['concentration_limit']:.1%} limit"
                        )
                        events.append(event)
                
                # Check profit protection (trailing stops)
                if position.unrealized_pnl_pct > self.risk_params['profit_protection_threshold']:
                    if not await self._has_trailing_stop(symbol):
                        if self._can_execute_rule("trailing_stop_update"):
                            event = AutomatedRiskEvent(
                                timestamp=datetime.now(),
                                rule_name="trailing_stop_update",
                                symbol=symbol,
                                action_taken=RiskAction.TIGHTEN_STOPS,
                                trigger_value=position.unrealized_pnl_pct,
                                threshold_value=self.risk_params['profit_protection_threshold'],
                                description=f"Profit {position.unrealized_pnl_pct:.1%} above threshold, adding trailing stop"
                            )
                            events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Error checking position risks: {e}")
            return []
    
    async def _check_portfolio_risks(self, portfolio_snapshot: PortfolioSnapshot) -> List[AutomatedRiskEvent]:
        """Check portfolio-level risks and generate events."""
        events = []
        
        try:
            # Calculate portfolio metrics
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in portfolio_snapshot.positions)
            portfolio_loss_pct = total_unrealized_pnl / portfolio_snapshot.total_value if portfolio_snapshot.total_value > 0 else 0
            
            # Check portfolio stop loss
            if portfolio_loss_pct < -self.risk_params['max_portfolio_loss']:
                if self._can_execute_rule("portfolio_stop_loss"):
                    event = AutomatedRiskEvent(
                        timestamp=datetime.now(),
                        rule_name="portfolio_stop_loss",
                        symbol=None,
                        action_taken=RiskAction.HALT_TRADING,
                        trigger_value=abs(portfolio_loss_pct),
                        threshold_value=self.risk_params['max_portfolio_loss'],
                        description=f"Portfolio loss {portfolio_loss_pct:.1%} exceeds {self.risk_params['max_portfolio_loss']:.1%} limit"
                    )
                    events.append(event)
            
            # Check drawdown
            current_drawdown = await self._calculate_current_drawdown(portfolio_snapshot)
            if current_drawdown > self.risk_params['max_drawdown']:
                if self._can_execute_rule("drawdown_protection"):
                    event = AutomatedRiskEvent(
                        timestamp=datetime.now(),
                        rule_name="drawdown_protection",
                        symbol=None,
                        action_taken=RiskAction.REDUCE_POSITION,
                        trigger_value=current_drawdown,
                        threshold_value=self.risk_params['max_drawdown'],
                        description=f"Drawdown {current_drawdown:.1%} exceeds {self.risk_params['max_drawdown']:.1%} limit"
                    )
                    events.append(event)
            
            # Check portfolio volatility
            portfolio_volatility = await self._calculate_portfolio_volatility(portfolio_snapshot)
            if portfolio_volatility > self.risk_params['max_portfolio_volatility']:
                if self._can_execute_rule("volatility_spike"):
                    event = AutomatedRiskEvent(
                        timestamp=datetime.now(),
                        rule_name="volatility_spike",
                        symbol=None,
                        action_taken=RiskAction.ADD_HEDGE,
                        trigger_value=portfolio_volatility,
                        threshold_value=self.risk_params['max_portfolio_volatility'],
                        description=f"Portfolio volatility {portfolio_volatility:.1%} exceeds {self.risk_params['max_portfolio_volatility']:.1%} limit"
                    )
                    events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Error checking portfolio risks: {e}")
            return []
    
    async def _check_market_risks(self, portfolio_snapshot: PortfolioSnapshot) -> List[AutomatedRiskEvent]:
        """Check market-level risks and generate events."""
        events = []
        
        try:
            # Check for market crash (simplified - would use SPY or market index)
            market_drop = await self._calculate_market_drop()
            if market_drop < self.risk_params['market_crash_threshold']:
                if self._can_execute_rule("market_crash_protection"):
                    event = AutomatedRiskEvent(
                        timestamp=datetime.now(),
                        rule_name="market_crash_protection",
                        symbol=None,
                        action_taken=RiskAction.ADD_HEDGE,
                        trigger_value=abs(market_drop),
                        threshold_value=abs(self.risk_params['market_crash_threshold']),
                        description=f"Market drop {market_drop:.1%} exceeds {self.risk_params['market_crash_threshold']:.1%} threshold"
                    )
                    events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Error checking market risks: {e}")
            return []
    
    async def _execute_risk_actions(self, events: List[AutomatedRiskEvent]) -> List[AutomatedRiskEvent]:
        """Execute risk management actions based on events."""
        execution_events = []
        
        try:
            # Sort events by priority
            events.sort(key=lambda e: self._get_rule_priority(e.rule_name), reverse=True)
            
            for event in events:
                if event.action_taken == RiskAction.NONE:
                    continue
                
                execution_event = await self._execute_single_risk_action(event)
                if execution_event:
                    execution_events.append(execution_event)
                
                # Update rule execution tracking
                self.rule_last_executed[event.rule_name] = datetime.now()
            
            return execution_events
            
        except Exception as e:
            logger.error(f"Error executing risk actions: {e}")
            return []
    
    async def _execute_single_risk_action(self, event: AutomatedRiskEvent) -> Optional[AutomatedRiskEvent]:
        """Execute a single risk management action."""
        try:
            logger.info(f"Executing risk action: {event.action_taken.value} for {event.rule_name}")
            
            if event.action_taken == RiskAction.CLOSE_POSITION:
                return await self._close_position(event)
            
            elif event.action_taken == RiskAction.REDUCE_POSITION:
                return await self._reduce_position(event)
            
            elif event.action_taken == RiskAction.TIGHTEN_STOPS:
                return await self._tighten_stops(event)
            
            elif event.action_taken == RiskAction.ADD_HEDGE:
                return await self._add_hedge(event)
            
            elif event.action_taken == RiskAction.INCREASE_CASH:
                return await self._increase_cash(event)
            
            elif event.action_taken == RiskAction.HALT_TRADING:
                return await self._halt_trading(event)
            
            return None
            
        except Exception as e:
            logger.error(f"Error executing risk action {event.action_taken}: {e}")
            return AutomatedRiskEvent(
                timestamp=datetime.now(),
                rule_name=event.rule_name,
                symbol=event.symbol,
                action_taken=event.action_taken,
                trigger_value=event.trigger_value,
                threshold_value=event.threshold_value,
                description=f"Failed to execute {event.action_taken.value}: {str(e)}",
                success=False,
                error_message=str(e)
            )
    
    async def _close_position(self, event: AutomatedRiskEvent) -> Optional[AutomatedRiskEvent]:
        """Close a position due to risk management."""
        try:
            if not event.symbol:
                return None
            
            # Get current position
            positions = alpaca_client.get_positions()
            position = next((p for p in positions if p['symbol'] == event.symbol), None)
            
            if not position:
                return None
            
            quantity = abs(float(position['qty']))
            side = "sell" if float(position['qty']) > 0 else "buy"
            
            # Create market order for immediate execution
            order_request = AdvancedOrderRequest(
                symbol=event.symbol,
                quantity=quantity,
                side=side,
                order_type="market",
                reasoning=f"Risk management: {event.rule_name} - {event.description}",
                agent_source="automated_risk_manager"
            )
            
            result = await self.advanced_order_manager.place_advanced_order(order_request)
            
            return AutomatedRiskEvent(
                timestamp=datetime.now(),
                rule_name=f"{event.rule_name}_execution",
                symbol=event.symbol,
                action_taken=RiskAction.CLOSE_POSITION,
                trigger_value=event.trigger_value,
                threshold_value=event.threshold_value,
                description=f"Closed position: {side} {quantity} shares",
                order_ids=[result.order_id] if result.success else [],
                success=result.success,
                error_message=result.error_message if not result.success else None
            )
            
        except Exception as e:
            logger.error(f"Error closing position for {event.symbol}: {e}")
            return None
    
    async def _reduce_position(self, event: AutomatedRiskEvent) -> Optional[AutomatedRiskEvent]:
        """Reduce a position due to risk management."""
        try:
            if not event.symbol:
                # Portfolio-wide position reduction
                return await self._reduce_largest_positions(event)
            
            # Get current position
            positions = alpaca_client.get_positions()
            position = next((p for p in positions if p['symbol'] == event.symbol), None)
            
            if not position:
                return None
            
            current_qty = float(position['qty'])
            reduction_pct = 0.3  # Reduce by 30%
            reduction_qty = abs(current_qty) * reduction_pct
            side = "sell" if current_qty > 0 else "buy"
            
            # Create limit order for better execution
            current_price = float(position['current_price'])
            limit_price = current_price * (0.995 if side == "sell" else 1.005)
            
            order_request = AdvancedOrderRequest(
                symbol=event.symbol,
                quantity=reduction_qty,
                side=side,
                order_type="limit",
                limit_price=limit_price,
                reasoning=f"Risk management: {event.rule_name} - Position reduction",
                agent_source="automated_risk_manager"
            )
            
            result = await self.advanced_order_manager.place_advanced_order(order_request)
            
            return AutomatedRiskEvent(
                timestamp=datetime.now(),
                rule_name=f"{event.rule_name}_execution",
                symbol=event.symbol,
                action_taken=RiskAction.REDUCE_POSITION,
                trigger_value=event.trigger_value,
                threshold_value=event.threshold_value,
                description=f"Reduced position: {side} {reduction_qty} shares ({reduction_pct:.0%})",
                order_ids=[result.order_id] if result.success else [],
                success=result.success,
                error_message=result.error_message if not result.success else None
            )
            
        except Exception as e:
            logger.error(f"Error reducing position for {event.symbol}: {e}")
            return None
    
    async def _tighten_stops(self, event: AutomatedRiskEvent) -> Optional[AutomatedRiskEvent]:
        """Add or tighten stop-loss orders."""
        try:
            if not event.symbol:
                return None
            
            # Get current position
            positions = alpaca_client.get_positions()
            position = next((p for p in positions if p['symbol'] == event.symbol), None)
            
            if not position:
                return None
            
            current_qty = float(position['qty'])
            current_price = float(position['current_price'])
            
            if current_qty > 0:  # Long position
                # Set trailing stop at profit protection level
                stop_price = current_price * (1 - self.risk_params['trailing_stop_distance'])
                
                # Use trailing stop order
                result = await place_trailing_stop_order(
                    symbol=event.symbol,
                    quantity=abs(current_qty),
                    side="sell",
                    trail_percent=self.risk_params['trailing_stop_distance'] * 100,
                    reasoning=f"Profit protection trailing stop: {event.description}"
                )
                
            else:  # Short position
                # Set stop for short position
                stop_price = current_price * (1 + self.risk_params['trailing_stop_distance'])
                
                result = await place_trailing_stop_order(
                    symbol=event.symbol,
                    quantity=abs(current_qty),
                    side="buy",
                    trail_percent=self.risk_params['trailing_stop_distance'] * 100,
                    reasoning=f"Short position protection stop: {event.description}"
                )
            
            if result.success:
                self.active_stop_orders[event.symbol].append(result.order_id)
            
            return AutomatedRiskEvent(
                timestamp=datetime.now(),
                rule_name=f"{event.rule_name}_execution",
                symbol=event.symbol,
                action_taken=RiskAction.TIGHTEN_STOPS,
                trigger_value=event.trigger_value,
                threshold_value=event.threshold_value,
                description=f"Added trailing stop at {self.risk_params['trailing_stop_distance']:.1%}",
                order_ids=[result.order_id] if result.success else [],
                success=result.success,
                error_message=result.error_message if not result.success else None
            )
            
        except Exception as e:
            logger.error(f"Error tightening stops for {event.symbol}: {e}")
            return None
    
    async def _add_hedge(self, event: AutomatedRiskEvent) -> Optional[AutomatedRiskEvent]:
        """Add hedging positions to reduce portfolio risk."""
        try:
            # Simplified hedging: reduce overall exposure
            # In production, would use SPY shorts, VIX calls, etc.
            
            account_info = alpaca_client.get_account_info()
            portfolio_value = float(account_info['portfolio_value'])
            
            # Calculate hedge size (5% of portfolio)
            hedge_size = portfolio_value * 0.05
            
            # Use SPY as hedge (simplified)
            hedge_symbol = "SPY"
            spy_price = alpaca_client.get_current_price(hedge_symbol)
            
            if spy_price:
                hedge_quantity = hedge_size / spy_price
                
                order_request = AdvancedOrderRequest(
                    symbol=hedge_symbol,
                    quantity=hedge_quantity,
                    side="sell_short",
                    order_type="market",
                    reasoning=f"Portfolio hedge: {event.description}",
                    agent_source="automated_risk_manager"
                )
                
                result = await self.advanced_order_manager.place_advanced_order(order_request)
                
                return AutomatedRiskEvent(
                    timestamp=datetime.now(),
                    rule_name=f"{event.rule_name}_execution",
                    symbol=hedge_symbol,
                    action_taken=RiskAction.ADD_HEDGE,
                    trigger_value=event.trigger_value,
                    threshold_value=event.threshold_value,
                    description=f"Added hedge: short {hedge_quantity:.0f} {hedge_symbol}",
                    order_ids=[result.order_id] if result.success else [],
                    success=result.success,
                    error_message=result.error_message if not result.success else None
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error adding hedge: {e}")
            return None
    
    async def _increase_cash(self, event: AutomatedRiskEvent) -> Optional[AutomatedRiskEvent]:
        """Increase cash position by reducing overall exposure."""
        try:
            # Reduce positions to increase cash
            positions = alpaca_client.get_positions()
            
            if not positions:
                return None
            
            # Sort by size and reduce largest positions
            positions.sort(key=lambda p: abs(float(p['market_value'])), reverse=True)
            
            orders_placed = []
            target_reduction = 0.20  # Reduce overall exposure by 20%
            
            for position in positions[:3]:  # Reduce top 3 positions
                symbol = position['symbol']
                current_qty = float(position['qty'])
                reduction_qty = abs(current_qty) * (target_reduction / 3)
                side = "sell" if current_qty > 0 else "buy"
                
                order_request = AdvancedOrderRequest(
                    symbol=symbol,
                    quantity=reduction_qty,
                    side=side,
                    order_type="market",
                    reasoning=f"Increase cash: {event.description}",
                    agent_source="automated_risk_manager"
                )
                
                result = await self.advanced_order_manager.place_advanced_order(order_request)
                if result.success:
                    orders_placed.append(result.order_id)
            
            return AutomatedRiskEvent(
                timestamp=datetime.now(),
                rule_name=f"{event.rule_name}_execution",
                symbol=None,
                action_taken=RiskAction.INCREASE_CASH,
                trigger_value=event.trigger_value,
                threshold_value=event.threshold_value,
                description=f"Increased cash by reducing {len(orders_placed)} positions",
                order_ids=orders_placed,
                success=len(orders_placed) > 0
            )
            
        except Exception as e:
            logger.error(f"Error increasing cash: {e}")
            return None
    
    async def _halt_trading(self, event: AutomatedRiskEvent) -> Optional[AutomatedRiskEvent]:
        """Halt trading due to critical risk levels."""
        try:
            # Set trading halt flag
            self.is_active = False
            
            # Cancel all pending orders
            pending_orders = alpaca_client.get_orders(status="open")
            cancelled_orders = []
            
            for order in pending_orders:
                try:
                    if alpaca_client.cancel_order(order['id']):
                        cancelled_orders.append(order['id'])
                except Exception:
                    pass
            
            logger.critical(f"TRADING HALTED: {event.description}")
            
            return AutomatedRiskEvent(
                timestamp=datetime.now(),
                rule_name=f"{event.rule_name}_execution",
                symbol=None,
                action_taken=RiskAction.HALT_TRADING,
                trigger_value=event.trigger_value,
                threshold_value=event.threshold_value,
                description=f"Trading halted, cancelled {len(cancelled_orders)} orders",
                order_ids=cancelled_orders,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error halting trading: {e}")
            return None
    
    async def _reduce_largest_positions(self, event: AutomatedRiskEvent) -> Optional[AutomatedRiskEvent]:
        """Reduce largest positions for portfolio rebalancing."""
        try:
            positions = alpaca_client.get_positions()
            
            if not positions:
                return None
            
            # Sort by absolute market value
            positions.sort(key=lambda p: abs(float(p['market_value'])), reverse=True)
            
            orders_placed = []
            reduction_pct = 0.25  # Reduce by 25%
            
            # Reduce top 2 positions
            for position in positions[:2]:
                symbol = position['symbol']
                current_qty = float(position['qty'])
                reduction_qty = abs(current_qty) * reduction_pct
                side = "sell" if current_qty > 0 else "buy"
                
                order_request = AdvancedOrderRequest(
                    symbol=symbol,
                    quantity=reduction_qty,
                    side=side,
                    order_type="limit",
                    limit_price=float(position['current_price']) * (0.995 if side == "sell" else 1.005),
                    reasoning=f"Portfolio rebalancing: {event.description}",
                    agent_source="automated_risk_manager"
                )
                
                result = await self.advanced_order_manager.place_advanced_order(order_request)
                if result.success:
                    orders_placed.append(result.order_id)
            
            return AutomatedRiskEvent(
                timestamp=datetime.now(),
                rule_name=f"{event.rule_name}_execution",
                symbol=None,
                action_taken=RiskAction.REDUCE_POSITION,
                trigger_value=event.trigger_value,
                threshold_value=event.threshold_value,
                description=f"Reduced largest positions: {len(orders_placed)} orders",
                order_ids=orders_placed,
                success=len(orders_placed) > 0
            )
            
        except Exception as e:
            logger.error(f"Error reducing largest positions: {e}")
            return None
    
    async def _calculate_position_volatility(self, symbol: str, days: int = 20) -> float:
        """Calculate position volatility."""
        try:
            market_data = alpaca_client.get_market_data(symbol, limit=days)
            
            if market_data is None or len(market_data) < 10:
                return 0.20  # Default volatility estimate
            
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            return min(1.0, max(0.05, volatility))  # Cap between 5% and 100%
            
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return 0.20
    
    async def _calculate_current_drawdown(self, portfolio_snapshot: PortfolioSnapshot) -> float:
        """Calculate current drawdown from peak."""
        try:
            # Simple implementation - would need historical portfolio values
            if len(portfolio_monitor.snapshots) < 2:
                return 0.0
            
            # Find peak value in recent history
            recent_values = [s.total_value for s in list(portfolio_monitor.snapshots)[-30:]]
            peak_value = max(recent_values)
            current_value = portfolio_snapshot.total_value
            
            if peak_value <= 0:
                return 0.0
            
            drawdown = (peak_value - current_value) / peak_value
            return max(0.0, drawdown)
            
        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
            return 0.0
    
    async def _calculate_portfolio_volatility(self, portfolio_snapshot: PortfolioSnapshot) -> float:
        """Calculate portfolio volatility."""
        try:
            if len(portfolio_monitor.snapshots) < 10:
                return 0.15  # Default estimate
            
            # Calculate portfolio returns from recent snapshots
            recent_values = [s.total_value for s in list(portfolio_monitor.snapshots)[-30:]]
            returns = [(recent_values[i] - recent_values[i-1]) / recent_values[i-1] 
                      for i in range(1, len(recent_values)) if recent_values[i-1] != 0]
            
            if not returns:
                return 0.15
            
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            return min(1.0, max(0.05, volatility))
            
        except Exception as e:
            logger.error(f"Error calculating portfolio volatility: {e}")
            return 0.15
    
    async def _calculate_market_drop(self) -> float:
        """Calculate market drop (simplified using SPY)."""
        try:
            spy_data = alpaca_client.get_market_data("SPY", limit=2)
            
            if spy_data is None or len(spy_data) < 2:
                return 0.0
            
            today_close = spy_data.iloc[-1]['close']
            yesterday_close = spy_data.iloc[-2]['close']
            
            return (today_close - yesterday_close) / yesterday_close
            
        except Exception as e:
            logger.error(f"Error calculating market drop: {e}")
            return 0.0
    
    async def _has_trailing_stop(self, symbol: str) -> bool:
        """Check if symbol has active trailing stop orders."""
        return len(self.active_stop_orders.get(symbol, [])) > 0
    
    def _can_execute_rule(self, rule_name: str) -> bool:
        """Check if rule can be executed (cooldown period)."""
        if rule_name not in self.rule_last_executed:
            return True
        
        rule = next((r for r in self.risk_rules if r.name == rule_name), None)
        if not rule or not rule.enabled:
            return False
        
        last_execution = self.rule_last_executed[rule_name]
        cooldown_period = timedelta(minutes=rule.cooldown_minutes)
        
        return datetime.now() - last_execution > cooldown_period
    
    def _get_rule_priority(self, rule_name: str) -> int:
        """Get rule priority."""
        rule = next((r for r in self.risk_rules if r.name == rule_name), None)
        return rule.priority if rule else 5
    
    def _cleanup_old_events(self):
        """Clean up old risk events."""
        cutoff_time = datetime.now() - timedelta(days=7)
        self.risk_events = [
            event for event in self.risk_events 
            if event.timestamp > cutoff_time
        ]
    
    async def _update_performance_metrics(self, portfolio_snapshot: PortfolioSnapshot):
        """Update risk-adjusted performance metrics."""
        try:
            if len(portfolio_monitor.snapshots) < 2:
                return
            
            # Calculate recent returns
            recent_values = [s.total_value for s in list(portfolio_monitor.snapshots)[-30:]]
            if len(recent_values) < 2:
                return
            
            returns = [(recent_values[i] - recent_values[i-1]) / recent_values[i-1] 
                      for i in range(1, len(recent_values)) if recent_values[i-1] != 0]
            
            if returns:
                avg_return = np.mean(returns)
                volatility = np.std(returns)
                
                # Calculate Sharpe ratio (assuming 0% risk-free rate)
                self.sharpe_ratio = avg_return / volatility if volatility > 0 else 0
                
                # Update max drawdown
                self.max_drawdown = await self._calculate_current_drawdown(portfolio_snapshot)
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Get comprehensive risk management status."""
        try:
            recent_events = [e for e in self.risk_events if e.timestamp > datetime.now() - timedelta(hours=1)]
            
            return {
                'is_active': self.is_active,
                'last_risk_check': self.last_risk_check.isoformat() if self.last_risk_check else None,
                'total_events_24h': len([e for e in self.risk_events if e.timestamp > datetime.now() - timedelta(hours=24)]),
                'recent_events': len(recent_events),
                'active_rules': len([r for r in self.risk_rules if r.enabled]),
                'position_risk_scores': dict(list(self.position_risk_scores.items())[:10]),  # Top 10
                'performance_metrics': {
                    'sharpe_ratio': self.sharpe_ratio,
                    'max_drawdown': self.max_drawdown
                },
                'risk_parameters': {
                    'max_position_loss': self.risk_params['max_position_loss'],
                    'max_portfolio_loss': self.risk_params['max_portfolio_loss'],
                    'concentration_limit': self.risk_params['concentration_limit'],
                    'max_drawdown': self.risk_params['max_drawdown']
                },
                'recent_actions': [
                    {
                        'timestamp': e.timestamp.isoformat(),
                        'rule': e.rule_name,
                        'action': e.action_taken.value,
                        'symbol': e.symbol,
                        'description': e.description,
                        'success': e.success
                    }
                    for e in recent_events[-5:]  # Last 5 events
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting risk status: {e}")
            return {'error': str(e)}
    
    def enable_trading(self):
        """Re-enable trading after halt."""
        self.is_active = True
        logger.info("Trading re-enabled by risk manager")
    
    def update_risk_parameters(self, new_params: Dict[str, float]):
        """Update risk parameters dynamically."""
        for key, value in new_params.items():
            if key in self.risk_params:
                old_value = self.risk_params[key]
                self.risk_params[key] = value
                logger.info(f"Updated risk parameter {key}: {old_value} -> {value}")

# Global automated risk manager instance
automated_risk_manager = AutomatedRiskManager()

# Convenience functions
async def monitor_portfolio_risk(portfolio_snapshot: PortfolioSnapshot) -> List[AutomatedRiskEvent]:
    """Monitor portfolio risk and execute automated management."""
    return await automated_risk_manager.monitor_and_manage_risk(portfolio_snapshot)

def get_risk_manager_status() -> Dict[str, Any]:
    """Get risk manager status."""
    return automated_risk_manager.get_risk_status()

def enable_risk_management():
    """Enable risk management."""
    automated_risk_manager.enable_trading()

def update_risk_parameters(params: Dict[str, float]):
    """Update risk parameters."""
    automated_risk_manager.update_risk_parameters(params)