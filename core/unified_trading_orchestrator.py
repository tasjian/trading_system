#!/usr/bin/env python3
"""
Unified Trading Orchestrator

Integrates all trading system components for comprehensive portfolio management
with proper buy/sell/short/limit order execution and intelligent rebalancing.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

# Import all our new components
from core.portfolio_balancer import (
    portfolio_balancer, PositionAction, OrderUrgency, 
    analyze_portfolio_balance, generate_rebalancing_orders
)
from core.signal_processor import (
    signal_processor, SignalType, MarketContext, MarketRegime,
    process_signals
)
from core.position_manager import (
    position_manager, PositionStatus, RiskLevel,
    analyze_positions, generate_position_decisions
)
from monitoring.portfolio_monitor import (
    portfolio_monitor, AlertLevel, MetricType,
    get_portfolio_health, check_portfolio_balance, get_monitoring_dashboard
)
from order_types.advanced_orders import (
    advanced_order_manager, AdvancedOrderRequest, OrderExecutionResult
)
from core.trading_engine import trading_engine, TradingOrder
from tools.alpaca_client import alpaca_client
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class TradingDecision:
    """Unified trading decision combining all analysis."""
    symbol: str
    action: PositionAction
    quantity: float
    order_type: str
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    reasoning: str = ""
    confidence: float = 0.5
    urgency: OrderUrgency = OrderUrgency.MEDIUM
    risk_level: RiskLevel = RiskLevel.MEDIUM
    source: str = "unified_orchestrator"

@dataclass
class TradingResult:
    """Result of trading operation."""
    success: bool
    decisions_made: int
    orders_placed: int
    orders_filled: int
    total_value_traded: float
    errors: List[str]
    warnings: List[str]
    execution_time: float
    summary: str

class UnifiedTradingOrchestrator:
    """
    Unified trading orchestrator that coordinates all trading system components
    to ensure proper portfolio balancing with appropriate order types.
    """
    
    def __init__(self):
        """Initialize the unified trading orchestrator."""
        self.last_rebalance = None
        self.rebalance_cooldown = timedelta(minutes=30)
        self.max_concurrent_orders = 10
        
        # Integration settings
        self.enable_portfolio_balancing = True
        self.enable_position_management = True
        self.enable_signal_processing = True
        self.enable_risk_monitoring = True
        
        # Execution preferences
        self.prefer_limit_orders = True
        self.emergency_market_orders = True
        self.partial_fill_management = True
        
    async def execute_comprehensive_trading_cycle(self, 
                                                 target_allocation: Dict[str, float],
                                                 raw_signals: List[Dict] = None) -> TradingResult:
        """
        Execute a comprehensive trading cycle that properly balances the portfolio
        using buy/sell/short/limit orders as appropriate.
        
        Args:
            target_allocation: Target portfolio allocation {symbol: weight}
            raw_signals: Optional raw trading signals from sentiment analysis
            
        Returns:
            TradingResult with execution summary
        """
        start_time = datetime.now()
        logger.info("🚀 Starting comprehensive trading cycle...")
        
        errors = []
        warnings = []
        decisions_made = 0
        orders_placed = 0
        orders_filled = 0
        total_value_traded = 0.0
        
        try:
            # Step 1: Get current portfolio state
            current_positions = await self._get_current_portfolio_state()
            if not current_positions:
                errors.append("Failed to get current portfolio state")
                return self._create_error_result(errors, start_time)
            
            # Step 2: Portfolio balance analysis
            logger.info("📊 Analyzing portfolio balance...")
            balance_analyses = await analyze_portfolio_balance(target_allocation, raw_signals)
            
            # Step 3: Position management analysis
            logger.info("🎯 Analyzing position management...")
            position_metrics = await analyze_positions(current_positions['positions'])
            position_decisions = await generate_position_decisions(position_metrics)
            
            # Step 4: Signal processing (if raw signals provided)
            processed_signals = []
            if raw_signals:
                logger.info("🔄 Processing trading signals...")
                processed_signals = await process_signals(
                    raw_signals, current_positions['positions'], target_allocation
                )
            
            # Step 5: Generate unified trading decisions
            logger.info("🎲 Generating unified trading decisions...")
            unified_decisions = await self._generate_unified_decisions(
                balance_analyses, position_decisions, processed_signals, target_allocation
            )
            decisions_made = len(unified_decisions)
            
            if not unified_decisions:
                logger.info("✅ No trading decisions needed - portfolio is balanced")
                return TradingResult(
                    success=True, decisions_made=0, orders_placed=0, orders_filled=0,
                    total_value_traded=0.0, errors=[], warnings=[],
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    summary="Portfolio is balanced - no trades needed"
                )
            
            # Step 6: Risk validation and filtering
            logger.info("⚖️ Validating decisions against risk limits...")
            validated_decisions = await self._validate_decisions_against_risk(unified_decisions)
            
            if len(validated_decisions) < len(unified_decisions):
                filtered_count = len(unified_decisions) - len(validated_decisions)
                warnings.append(f"Filtered {filtered_count} decisions due to risk limits")
            
            # Step 7: Execute orders
            logger.info(f"💼 Executing {len(validated_decisions)} trading decisions...")
            execution_results = await self._execute_trading_decisions(validated_decisions)
            
            # Step 8: Process execution results
            orders_placed, orders_filled, total_value_traded = self._process_execution_results(execution_results)
            
            # Step 9: Monitor and alert
            if self.enable_risk_monitoring:
                await self._check_and_process_alerts(target_allocation, current_positions['positions'])
            
            # Step 10: Update tracking
            self.last_rebalance = datetime.now()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            success = len(errors) == 0
            
            summary = self._create_execution_summary(
                decisions_made, orders_placed, orders_filled, total_value_traded, execution_time
            )
            
            logger.info(f"✅ Trading cycle complete: {summary}")
            
            return TradingResult(
                success=success,
                decisions_made=decisions_made,
                orders_placed=orders_placed,
                orders_filled=orders_filled,
                total_value_traded=total_value_traded,
                errors=errors,
                warnings=warnings,
                execution_time=execution_time,
                summary=summary
            )
            
        except Exception as e:
            error_msg = f"Trading cycle failed: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            
            return TradingResult(
                success=False,
                decisions_made=decisions_made,
                orders_placed=orders_placed,
                orders_filled=orders_filled,
                total_value_traded=total_value_traded,
                errors=errors,
                warnings=warnings,
                execution_time=(datetime.now() - start_time).total_seconds(),
                summary=f"Failed: {error_msg}"
            )
    
    async def _get_current_portfolio_state(self) -> Optional[Dict]:
        """Get comprehensive current portfolio state."""
        try:
            account_info = alpaca_client.get_account_info()
            positions_data = alpaca_client.get_positions()
            
            positions = {}
            for pos in positions_data:
                symbol = pos['symbol']
                positions[symbol] = {
                    'quantity': float(pos.get('qty', 0)),
                    'market_value': float(pos.get('market_value', 0)),
                    'avg_entry_price': float(pos.get('avg_entry_price', 0)),
                    'unrealized_pnl': float(pos.get('unrealized_pl', 0)),
                    'side': pos.get('side', 'long')
                }
            
            return {
                'account_info': account_info,
                'positions': positions,
                'total_value': float(account_info.get('portfolio_value', 0)),
                'cash': float(account_info.get('cash', 0))
            }
            
        except Exception as e:
            logger.error(f"Failed to get portfolio state: {e}")
            return None
    
    async def _generate_unified_decisions(self, 
                                        balance_analyses: List,
                                        position_decisions: List,
                                        processed_signals: List,
                                        target_allocation: Dict[str, float]) -> List[TradingDecision]:
        """Generate unified trading decisions from all analysis sources."""
        
        # Consolidate decisions by symbol
        symbol_decisions = {}
        
        # 1. Process portfolio balance decisions (highest priority)
        rebalance_orders = await generate_rebalancing_orders(balance_analyses, self.max_concurrent_orders)
        for order in rebalance_orders:
            symbol = order.symbol
            
            decision = TradingDecision(
                symbol=symbol,
                action=order.action,
                quantity=order.quantity,
                order_type=order.order_type,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                reasoning=f"REBALANCE: {order.reasoning}",
                confidence=order.confidence,
                urgency=order.urgency,
                risk_level=order.risk_level,
                source="portfolio_rebalancer"
            )
            
            symbol_decisions[symbol] = decision
        
        # 2. Process position management decisions (medium priority)
        for pos_decision in position_decisions:
            symbol = pos_decision.symbol
            
            # If we already have a rebalancing decision, merge or prioritize
            if symbol in symbol_decisions:
                existing = symbol_decisions[symbol]
                
                # Prioritize emergency actions
                if pos_decision.urgency == OrderUrgency.CRITICAL:
                    symbol_decisions[symbol] = TradingDecision(
                        symbol=symbol,
                        action=pos_decision.action,
                        quantity=pos_decision.quantity,
                        order_type=pos_decision.order_type,
                        limit_price=pos_decision.limit_price,
                        reasoning=f"EMERGENCY: {pos_decision.reasoning}",
                        confidence=pos_decision.confidence,
                        urgency=pos_decision.urgency,
                        risk_level=pos_decision.risk_level,
                        source="position_manager_emergency"
                    )
                # Otherwise, adjust the existing decision
                else:
                    existing.reasoning += f" | POSITION: {pos_decision.reasoning}"
                    existing.confidence = (existing.confidence + pos_decision.confidence) / 2
            else:
                # New position management decision
                decision = TradingDecision(
                    symbol=symbol,
                    action=pos_decision.action,
                    quantity=pos_decision.quantity,
                    order_type=pos_decision.order_type,
                    limit_price=pos_decision.limit_price,
                    reasoning=f"POSITION: {pos_decision.reasoning}",
                    confidence=pos_decision.confidence,
                    urgency=pos_decision.urgency,
                    risk_level=pos_decision.risk_level,
                    source="position_manager"
                )
                
                symbol_decisions[symbol] = decision
        
        # 3. Process signal-based decisions (lowest priority - only for new positions)
        for signal in processed_signals:
            symbol = signal.symbol
            
            # Only add signal-based decisions if no existing decision for this symbol
            # or if it's a complementary action
            if symbol not in symbol_decisions:
                decision = TradingDecision(
                    symbol=symbol,
                    action=signal.action,
                    quantity=signal.quantity,
                    order_type=signal.order_type,
                    limit_price=signal.limit_price,
                    stop_price=signal.stop_price,
                    reasoning=f"SIGNAL: {signal.reasoning}",
                    confidence=signal.confidence,
                    urgency=signal.urgency,
                    risk_level=signal.risk_adjustments.get('risk_level', RiskLevel.MEDIUM),
                    source="signal_processor"
                )
                
                symbol_decisions[symbol] = decision
        
        # Convert to list and sort by priority
        unified_decisions = list(symbol_decisions.values())
        
        # Sort by urgency, then risk level, then confidence
        unified_decisions.sort(
            key=lambda d: (d.urgency.value, d.risk_level.value, -d.confidence), 
            reverse=True
        )
        
        logger.info(f"🎯 Generated {len(unified_decisions)} unified trading decisions")
        return unified_decisions
    
    async def _validate_decisions_against_risk(self, decisions: List[TradingDecision]) -> List[TradingDecision]:
        """Validate trading decisions against risk limits."""
        
        validated_decisions = []
        
        # Get current portfolio metrics for risk validation
        try:
            account_info = alpaca_client.get_account_info()
            portfolio_value = float(account_info.get('portfolio_value', 100000))
            cash_available = float(account_info.get('cash', 0))
            buying_power = float(account_info.get('buying_power', 0))
        except:
            portfolio_value = 100000
            cash_available = 50000
            buying_power = 50000
        
        total_buy_value = 0.0
        daily_trades = await self._get_daily_trades_count()
        
        for decision in decisions:
            # Skip if already hit daily trade limit
            if daily_trades >= settings.max_daily_trades:
                logger.warning(f"Daily trade limit reached, skipping {decision.symbol}")
                continue
            
            # Get current price for value calculations
            try:
                current_price = alpaca_client.get_current_price(decision.symbol)
                if not current_price or current_price <= 0:
                    logger.warning(f"Invalid price for {decision.symbol}, skipping")
                    continue
                current_price = float(current_price)
            except:
                logger.warning(f"Could not get price for {decision.symbol}, skipping")
                continue
            
            order_value = decision.quantity * current_price
            
            # Position size validation
            position_weight = order_value / portfolio_value
            if position_weight > settings.max_position_size:
                # Scale down the order
                max_value = portfolio_value * settings.max_position_size
                decision.quantity = max_value / current_price
                order_value = max_value
                logger.info(f"Scaled down {decision.symbol} order to respect position size limit")
            
            # Cash/buying power validation for buy orders
            if decision.action in [PositionAction.BUY, PositionAction.COVER]:
                if total_buy_value + order_value > buying_power * 0.95:
                    # Scale down to fit available buying power
                    available_value = (buying_power * 0.95) - total_buy_value
                    if available_value > 100:  # Minimum $100 order
                        decision.quantity = available_value / current_price
                        order_value = available_value
                        logger.info(f"Scaled down {decision.symbol} order to fit buying power")
                    else:
                        logger.warning(f"Insufficient buying power for {decision.symbol}, skipping")
                        continue
                
                total_buy_value += order_value
            
            # Minimum order size validation
            if order_value < 100:  # $100 minimum
                logger.debug(f"Order value too small for {decision.symbol} (${order_value:.2f}), skipping")
                continue
            
            # Emergency orders bypass some restrictions
            if decision.urgency == OrderUrgency.CRITICAL:
                validated_decisions.append(decision)
                daily_trades += 1
                continue
            
            # Risk level validation
            if decision.risk_level == RiskLevel.CRITICAL and len(validated_decisions) >= 5:
                logger.warning(f"Too many high-risk orders, skipping {decision.symbol}")
                continue
            
            validated_decisions.append(decision)
            daily_trades += 1
        
        logger.info(f"✅ Validated {len(validated_decisions)}/{len(decisions)} trading decisions")
        return validated_decisions
    
    async def _execute_trading_decisions(self, decisions: List[TradingDecision]) -> List[OrderExecutionResult]:
        """Execute validated trading decisions."""
        
        execution_results = []
        
        for decision in decisions:
            try:
                # Create advanced order request
                order_request = AdvancedOrderRequest(
                    symbol=decision.symbol,
                    quantity=decision.quantity,
                    side=self._map_action_to_order_side(decision.action),
                    order_type=decision.order_type,
                    limit_price=decision.limit_price,
                    stop_price=decision.stop_price,
                    reasoning=decision.reasoning,
                    confidence=decision.confidence,
                    agent_source="unified_orchestrator"
                )
                
                # Execute the order
                logger.info(f"📤 Executing {decision.action.value} order for {decision.symbol}: {decision.quantity} shares")
                result = await advanced_order_manager.place_advanced_order(order_request)
                
                execution_results.append(result)
                
                if result.success:
                    logger.info(f"✅ Order placed successfully: {result.order_id}")
                else:
                    logger.error(f"❌ Order failed: {result.error_message}")
                
                # Small delay between orders to avoid overwhelming the system
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to execute decision for {decision.symbol}: {e}")
                execution_results.append(OrderExecutionResult(
                    success=False,
                    error_message=str(e)
                ))
        
        return execution_results
    
    def _map_action_to_order_side(self, action: PositionAction) -> str:
        """Map position action to order side."""
        
        mapping = {
            PositionAction.BUY: "buy",
            PositionAction.SELL: "sell",
            PositionAction.SHORT: "sell_short",
            PositionAction.COVER: "buy",  # Buy to cover short
            PositionAction.REDUCE: "sell",  # Reduce position by selling
            PositionAction.CLOSE: "sell"   # Close position by selling
        }
        
        return mapping.get(action, "buy")
    
    def _process_execution_results(self, results: List[OrderExecutionResult]) -> Tuple[int, int, float]:
        """Process execution results and return statistics."""
        
        orders_placed = len([r for r in results if r.success])
        orders_filled = len([r for r in results if r.success and r.filled_qty > 0])
        
        total_value_traded = sum(
            r.filled_qty * r.avg_fill_price 
            for r in results 
            if r.success and r.filled_qty > 0 and r.avg_fill_price > 0
        )
        
        return orders_placed, orders_filled, total_value_traded
    
    async def _check_and_process_alerts(self, target_allocation: Dict[str, float], current_positions: Dict):
        """Check for alerts and process them."""
        
        try:
            # Check portfolio balance alerts
            balance_alerts = await check_portfolio_balance(target_allocation, current_positions)
            
            # Get portfolio health and check risk alerts
            health_metrics = await get_portfolio_health()
            risk_alerts = await portfolio_monitor.check_risk_alerts(health_metrics)
            performance_alerts = await portfolio_monitor.check_performance_alerts(health_metrics)
            
            # Process all alerts
            all_alerts = balance_alerts + risk_alerts + performance_alerts
            if all_alerts:
                alert_summary = await portfolio_monitor.process_alerts(all_alerts)
                logger.info(f"🚨 Alert summary: {alert_summary['summary']}")
        
        except Exception as e:
            logger.error(f"Failed to process alerts: {e}")
    
    async def _get_daily_trades_count(self) -> int:
        """Get number of trades executed today."""
        try:
            from datetime import date
            today = date.today().isoformat()
            
            orders = alpaca_client.get_orders(status="filled", limit=100)
            daily_trades = len([o for o in orders if o.get('filled_at', '').startswith(today)])
            
            return daily_trades
        except:
            return 0
    
    def _create_execution_summary(self, decisions: int, placed: int, filled: int, 
                                value: float, time: float) -> str:
        """Create execution summary string."""
        
        return (f"{decisions} decisions → {placed} orders placed → {filled} filled "
                f"(${value:,.0f} traded in {time:.1f}s)")
    
    def _create_error_result(self, errors: List[str], start_time: datetime) -> TradingResult:
        """Create error result."""
        
        return TradingResult(
            success=False,
            decisions_made=0,
            orders_placed=0,
            orders_filled=0,
            total_value_traded=0.0,
            errors=errors,
            warnings=[],
            execution_time=(datetime.now() - start_time).total_seconds(),
            summary=f"Failed: {errors[0] if errors else 'Unknown error'}"
        )
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        try:
            # Get portfolio health
            health_metrics = await get_portfolio_health()
            
            # Get monitoring dashboard
            dashboard = await get_monitoring_dashboard()
            
            # Get component status
            component_status = {
                'portfolio_balancer': self.enable_portfolio_balancing,
                'position_manager': self.enable_position_management,
                'signal_processor': self.enable_signal_processing,
                'risk_monitoring': self.enable_risk_monitoring,
                'advanced_orders': True,
                'trading_engine': True
            }
            
            # Get last rebalance info
            last_rebalance_info = {
                'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None,
                'cooldown_remaining': None
            }
            
            if self.last_rebalance:
                time_since = datetime.now() - self.last_rebalance
                cooldown_remaining = self.rebalance_cooldown - time_since
                if cooldown_remaining.total_seconds() > 0:
                    last_rebalance_info['cooldown_remaining'] = cooldown_remaining.total_seconds()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'system_healthy': True,
                'components': component_status,
                'rebalancing': last_rebalance_info,
                'health_metrics': health_metrics,
                'dashboard': dashboard,
                'settings': {
                    'max_concurrent_orders': self.max_concurrent_orders,
                    'prefer_limit_orders': self.prefer_limit_orders,
                    'emergency_market_orders': self.emergency_market_orders
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'system_healthy': False,
                'error': str(e)
            }

# Global instance
unified_orchestrator = UnifiedTradingOrchestrator()

# Convenience functions
async def execute_comprehensive_trading_cycle(target_allocation: Dict[str, float],
                                            raw_signals: List[Dict] = None) -> TradingResult:
    """Execute comprehensive trading cycle."""
    return await unified_orchestrator.execute_comprehensive_trading_cycle(target_allocation, raw_signals)

async def get_system_status() -> Dict[str, Any]:
    """Get system status."""
    return await unified_orchestrator.get_system_status()

__all__ = [
    'TradingDecision', 'TradingResult', 'UnifiedTradingOrchestrator',
    'unified_orchestrator', 'execute_comprehensive_trading_cycle', 'get_system_status'
]