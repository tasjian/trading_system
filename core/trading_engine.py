"""
Unified Trading Engine

Consolidates all trading functionality into a single, coherent system:
- Order execution and management
- Portfolio rebalancing
- Risk management
- Strategy execution (including pairs trading)
- Performance tracking

This replaces multiple separate trading components with one unified engine.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

from tools.alpaca_client import alpaca_client
from core.market_intelligence import market_intelligence, SignalType, AnalysisResult
from config.settings import settings

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class StrategyType(Enum):
    SINGLE_STOCK = "single_stock"
    PAIRS_TRADING = "pairs_trading"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"

@dataclass
class TradingOrder:
    """Unified trading order representation."""
    symbol: str
    side: str  # "buy", "sell", "sell_short"
    quantity: float
    order_type: str = "market"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    strategy: StrategyType = StrategyType.SINGLE_STOCK
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING

@dataclass
class Position:
    """Portfolio position representation."""
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float

@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    total_value: float
    cash: float
    equity: float
    day_pnl: float
    total_pnl: float
    positions: List[Position]
    allocation: Dict[str, float]  # symbol -> percentage
    risk_metrics: Dict[str, float]

class UnifiedTradingEngine:
    """
    Unified trading engine that handles all trading operations,
    strategies, and portfolio management in one cohesive system.
    """
    
    def __init__(self):
        self.positions = {}
        self.pending_orders = []
        self.order_history = []
        self.performance_history = []
        
        # Risk parameters from settings
        self.max_position_size = settings.max_position_size
        self.max_portfolio_risk = getattr(settings, 'max_portfolio_risk', 0.02)
        self.stop_loss_pct = settings.stop_loss_percent
        
        # Strategy parameters
        self.rebalance_threshold = settings.min_rebalance_threshold
        self.pairs_z_threshold = 2.0  # Z-score threshold for pairs trading
    
    async def validate_cash_balance(self) -> bool:
        """Validate cash balance and halt system if insufficient funds - margin account aware."""
        try:
            account_info = alpaca_client.get_account_info()
            cash_balance = float(account_info.get('cash', 0))
            buying_power = float(account_info.get('buying_power', 0))
            equity = float(account_info.get('equity', 0))
            day_trading_buying_power = float(account_info.get('daytrade_buying_power', 0))
            regt_buying_power = float(account_info.get('regt_buying_power', 0))
            pattern_day_trader = account_info.get('pattern_day_trader', False)
            
            # Check for day trading power issues (skip for paper accounts)
            is_paper_account = 'paper' in account_info.get('id', '').lower() or buying_power > equity * 2.5
            
            if pattern_day_trader and not is_paper_account and day_trading_buying_power <= 0:
                logger.warning(f"⚠️ DAY TRADING POWER EXHAUSTED: ${day_trading_buying_power:.2f}")
                logger.warning("   Reason: Likely unsettled funds or recent day trades")
                logger.warning("   System will operate in limited mode (sells only)")
                # Don't halt system, just log the limitation
                return True
            elif is_paper_account:
                logger.debug(f"📝 Paper trading account detected - using buying power: ${buying_power:,.2f}")
                # Paper accounts don't have real day trading power restrictions
            
            # For margin accounts, use buying power instead of cash balance
            # Pattern Day Trader accounts can have negative cash but positive buying power
            available_funds = buying_power
            
            if available_funds < 50:  # Minimum $50 buying power required (reduced from $100)
                error_msg = (
                    f"❌ CRITICAL SYSTEM HALT: Insufficient buying power\n"
                    f"Cash Balance: ${cash_balance:,.2f}\n"
                    f"Buying Power: ${buying_power:,.2f}\n"
                    f"Day Trading Power: ${day_trading_buying_power:,.2f}\n"
                    f"RegT Buying Power: ${regt_buying_power:,.2f}\n"
                    f"Equity: ${equity:,.2f}\n"
                    f"TRADING SUSPENDED - INSUFFICIENT FUNDS"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Warn if negative cash but continue if buying power is sufficient
            if cash_balance < 0:
                logger.warning(
                    f"⚠️  Margin account trading - Negative cash: ${cash_balance:,.2f}, "
                    f"but buying power available: ${buying_power:,.2f}"
                )
            
            if available_funds < 1000:
                logger.warning(f"Low buying power warning: ${available_funds:,.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Cash balance validation failed: {e}")
            raise RuntimeError(f"Cash balance validation failed: {e}")
    
    async def execute_analysis_signal(self, analysis: AnalysisResult) -> Optional[TradingOrder]:
        """Execute trading based on analysis signal."""
        try:
            # CRITICAL: Validate cash balance first - FAIL-FAST architecture
            await self.validate_cash_balance()
            
            logger.info(f"🎯 Processing signal for {analysis.symbol}: {analysis.signal.value}")
            
            # Skip if confidence too low
            if analysis.confidence.value in ['very_low', 'low']:
                logger.info(f"Skipping {analysis.symbol} due to low confidence: {analysis.confidence.value}")
                return None
            
            # Skip hold signals
            if analysis.signal == SignalType.HOLD:
                return None
            
            # Calculate position size based on confidence and risk
            position_size = await self._calculate_position_size(analysis)
            if position_size <= 0:
                logger.info(f"Position size too small for {analysis.symbol}")
                return None
            
            # Determine order side based on signal type AND current position
            side = await self._determine_order_side(analysis.signal, analysis.symbol)
            if not side:
                logger.info(f"No valid order side for {analysis.symbol} with signal {analysis.signal}")
                return None
            
            # Create order
            order = TradingOrder(
                symbol=analysis.symbol,
                side=side,
                quantity=position_size,
                strategy=StrategyType.SINGLE_STOCK,
                reasoning=f"Signal: {analysis.signal.value}, Score: {analysis.score:.2f}, Confidence: {analysis.confidence.value}"
            )
            
            # Execute order
            return await self._execute_order(order)
            
        except Exception as e:
            logger.error(f"Error executing signal for {analysis.symbol}: {e}")
            return None
    
    async def execute_pairs_trade(self, symbol1: str, symbol2: str, z_score: float, 
                                 hedge_ratio: float) -> List[TradingOrder]:
        """Execute pairs trading strategy."""
        try:
            # CRITICAL: Validate cash balance first - FAIL-FAST architecture
            await self.validate_cash_balance()
            
            logger.info(f"🔄 Executing pairs trade: {symbol1}/{symbol2}, Z-score: {z_score:.2f}")
            
            orders = []
            
            # Determine trade direction based on z-score
            if abs(z_score) < self.pairs_z_threshold:
                logger.info(f"Z-score {z_score:.2f} below threshold {self.pairs_z_threshold}")
                return orders
            
            # Calculate position sizes
            account_info = alpaca_client.get_account_info()
            portfolio_value = account_info['portfolio_value']
            position_value = portfolio_value * 0.20  # 20% position size for pairs
            
            # Get current prices
            price1_data = await market_intelligence.get_market_data(symbol1)
            price2_data = await market_intelligence.get_market_data(symbol2)
            
            if price1_data.price <= 0 or price2_data.price <= 0:
                logger.error(f"Invalid prices for pair {symbol1}/{symbol2}")
                return orders
            
            # Calculate quantities
            qty1 = position_value / price1_data.price / 2
            qty2 = qty1 * hedge_ratio
            
            reasoning = f"Pairs trade: Z-score {z_score:.2f}, Hedge ratio {hedge_ratio:.2f}"
            
            if z_score > self.pairs_z_threshold:
                # Spread too high: sell symbol1, buy symbol2
                order1 = TradingOrder(
                    symbol=symbol1,
                    side="sell_short",
                    quantity=qty1,
                    strategy=StrategyType.PAIRS_TRADING,
                    reasoning=f"{reasoning} - Short overpriced asset"
                )
                
                order2 = TradingOrder(
                    symbol=symbol2,
                    side="buy",
                    quantity=qty2,
                    strategy=StrategyType.PAIRS_TRADING,
                    reasoning=f"{reasoning} - Buy underpriced asset"
                )
                
            else:
                # Spread too low: buy symbol1, sell symbol2
                order1 = TradingOrder(
                    symbol=symbol1,
                    side="buy",
                    quantity=qty1,
                    strategy=StrategyType.PAIRS_TRADING,
                    reasoning=f"{reasoning} - Buy underpriced asset"
                )
                
                order2 = TradingOrder(
                    symbol=symbol2,
                    side="sell_short",
                    quantity=qty2,
                    strategy=StrategyType.PAIRS_TRADING,
                    reasoning=f"{reasoning} - Short overpriced asset"
                )
            
            # Execute both orders
            executed_order1 = await self._execute_order(order1)
            executed_order2 = await self._execute_order(order2)
            
            if executed_order1:
                orders.append(executed_order1)
            if executed_order2:
                orders.append(executed_order2)
            
            return orders
            
        except Exception as e:
            logger.error(f"Error executing pairs trade {symbol1}/{symbol2}: {e}")
            return []
    
    async def rebalance_portfolio(self, target_allocation: Dict[str, float], 
                                 force: bool = False) -> List[TradingOrder]:
        """Rebalance portfolio with FAIL-FAST cash validation."""
        try:
            # CRITICAL: Validate cash balance first
            if not await self.validate_cash_balance():
                logger.error("Portfolio rebalancing halted due to cash balance issues")
                return []
            
            logger.info(f"🔄 Rebalancing portfolio with {len(target_allocation)} positions")
            
            # Get current portfolio
            current_positions = alpaca_client.get_positions()
            account_info = alpaca_client.get_account_info()
            portfolio_value = account_info['portfolio_value']
            
            # Calculate current allocation
            current_allocation = {}
            for pos in current_positions:
                current_allocation[pos['symbol']] = abs(pos['market_value']) / portfolio_value
            
            # Calculate required trades
            orders = []
            
            for symbol, target_pct in target_allocation.items():
                current_pct = current_allocation.get(symbol, 0.0)
                difference = target_pct - current_pct
                
                # Check if rebalancing is needed
                if not force and abs(difference) < self.rebalance_threshold:
                    continue
                
                # Get current price
                price_data = await market_intelligence.get_market_data(symbol)
                if price_data.price <= 0:
                    logger.warning(f"Invalid price for {symbol}, skipping")
                    continue
                
                # Calculate target value and quantity
                target_value = portfolio_value * target_pct
                current_value = portfolio_value * current_pct
                trade_value = target_value - current_value
                
                if abs(trade_value) < 100:  # Skip trades under $100
                    continue
                
                quantity = abs(trade_value) / price_data.price
                
                # Determine side based on trade value and current position
                current_pos = next((pos for pos in current_positions if pos['symbol'] == symbol), None)
                
                if trade_value > 0:
                    side = "buy"
                else:
                    # Need to reduce position
                    if current_pos and float(current_pos['qty']) > 0:
                        # Have long position, sell it
                        side = "sell"
                        # Limit quantity to actual position size to avoid overselling
                        max_sellable = float(current_pos['qty'])
                        quantity = min(quantity, max_sellable)
                    elif current_pos and float(current_pos['qty']) < 0:
                        # Have short position, buy to cover
                        side = "buy"
                        max_coverable = abs(float(current_pos['qty']))
                        quantity = min(quantity, max_coverable)
                    else:
                        # No position but target is lower, skip (can't sell what we don't have)
                        logger.info(f"Skipping {symbol}: no position to reduce")
                        continue
                
                order = TradingOrder(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    strategy=StrategyType.PORTFOLIO_REBALANCE,
                    reasoning=f"Rebalance: {current_pct:.1%} -> {target_pct:.1%} (${trade_value:,.0f})"
                )
                
                executed_order = await self._execute_order(order)
                if executed_order:
                    orders.append(executed_order)
            
            # Handle positions not in target (close them)
            for symbol in current_allocation:
                if symbol not in target_allocation:
                    try:
                        close_result = alpaca_client.close_position(symbol)
                        logger.info(f"Closed position in {symbol}: {close_result}")
                    except Exception as e:
                        logger.error(f"Failed to close position in {symbol}: {e}")
            
            logger.info(f"Rebalancing complete: {len(orders)} orders executed")
            return orders
            
        except Exception as e:
            logger.error(f"Portfolio rebalancing error: {e}")
            return []
    
    async def _execute_order(self, order: TradingOrder) -> Optional[TradingOrder]:
        """Execute a trading order with comprehensive validation and risk checks."""
        try:
            # Pre-trade position validation
            if not await self._validate_position_for_order(order.symbol, order.side, order.quantity):
                logger.warning(f"Position validation failed for {order.symbol}")
                order.status = OrderStatus.REJECTED
                return order
            
            # Pre-trade risk checks
            if not await self._validate_order(order):
                logger.warning(f"Order validation failed for {order.symbol}")
                order.status = OrderStatus.REJECTED
                return order
            
            # Execute via Alpaca
            result = alpaca_client.place_order(
                symbol=order.symbol,
                qty=order.quantity,
                side=order.side,
                order_type=order.order_type,
                limit_price=order.limit_price,
                stop_price=order.stop_price
            )
            
            # Update order with result
            order.order_id = result['id']
            order.status = OrderStatus.FILLED  # Will be updated by order monitoring
            
            # Add to tracking
            self.pending_orders.append(order)
            self.order_history.append(order)
            
            logger.info(f"✅ Order executed: {order.side} {order.quantity} {order.symbol}")
            return order
            
        except Exception as e:
            logger.error(f"Order execution failed for {order.symbol}: {e}")
            order.status = OrderStatus.REJECTED
            return order
    
    async def _validate_order(self, order: TradingOrder) -> bool:
        """FAIL-FAST order validation - strict cash balance enforcement."""
        try:
            # Get account info
            account_info = alpaca_client.get_account_info()
            
            # CRITICAL: Check buying power - margin account aware validation
            cash_balance = float(account_info.get('cash', 0))
            buying_power = float(account_info.get('buying_power', 0))
            equity = float(account_info.get('equity', 0))
            
            if buying_power < 50:  # Minimum $50 buying power required (reduced from $100)
                error_msg = (
                    f"❌ CRITICAL SYSTEM HALT: Insufficient buying power\n"
                    f"Cash Balance: ${cash_balance:,.2f}\n"
                    f"Buying Power: ${buying_power:,.2f}\n"
                    f"Equity: ${equity:,.2f}\n"
                    f"Order: {order.side} {order.quantity} {order.symbol}\n"
                    f"TRADING SUSPENDED - INSUFFICIENT FUNDS"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Log margin account status if negative cash
            if cash_balance < 0:
                logger.debug(
                    f"Margin trading: Negative cash ${cash_balance:,.2f}, "
                    f"but buying power ${buying_power:,.2f} available for {order.symbol}"
                )
            
            # Check if trading is blocked
            if account_info.get('trading_blocked', False):
                logger.error("Trading is blocked")
                return False
            
            # Get current price for value calculations
            price_data = await market_intelligence.get_market_data(order.symbol)
            if price_data.price <= 0:
                logger.error(f"Invalid price for {order.symbol}")
                return False
            
            order_value = order.quantity * price_data.price
            portfolio_value = account_info['portfolio_value']
            
            # Check position size limit
            if portfolio_value > 0:
                position_pct = order_value / portfolio_value
                if position_pct > self.max_position_size:
                    logger.error(f"Position size {position_pct:.2%} exceeds limit {self.max_position_size:.2%}")
                    return False
            
            # STRICT CASH BALANCE VALIDATION for buy orders (no margin allowed)
            if order.side == "buy":
                # Use only cash balance - no margin/buying power for fail-fast architecture
                if order_value > cash_balance:
                    error_msg = (
                        f"❌ INSUFFICIENT CASH: Order exceeds cash balance\n"
                        f"Order Value: ${order_value:,.2f}\n"
                        f"Cash Balance: ${cash_balance:,.2f}\n"
                        f"Shortage: ${order_value - cash_balance:,.2f}\n"
                        f"Order: {order.side} {order.quantity} {order.symbol}\n"
                        f"FAIL-FAST ARCHITECTURE: No margin trading allowed"
                    )
                    logger.error(error_msg)
                    return False
                
                # Additional safety check: ensure sufficient cash remains after trade
                min_cash_reserve = 1000.0  # Minimum $1000 cash reserve
                if (cash_balance - order_value) < min_cash_reserve:
                    logger.error(
                        f"Order would leave insufficient cash reserve: "
                        f"${cash_balance - order_value:,.2f} < ${min_cash_reserve:,.2f}"
                    )
                    return False
            
            return True
            
        except RuntimeError:
            # Re-raise critical system halts
            raise
        except Exception as e:
            logger.error(f"Order validation error: {e}")
            return False
    
    async def _calculate_position_size(self, analysis: AnalysisResult) -> float:
        """FAIL-FAST position sizing with strict cash balance validation."""
        try:
            account_info = alpaca_client.get_account_info()
            portfolio_value = account_info['portfolio_value']
            cash_balance = float(account_info.get('cash', 0))
            
            # Get buying power for margin account aware position sizing
            buying_power = float(account_info.get('buying_power', 0))
            equity = float(account_info.get('equity', 0))
            
            # Use buying power for position sizing in margin accounts
            if buying_power < 50:  # Minimum $50 buying power required (reduced from $100)
                logger.info(f"Insufficient buying power for position sizing: ${buying_power}")
                return 0.0
            
            if portfolio_value <= 0:
                logger.info(f"Invalid portfolio value for position sizing: ${portfolio_value}")
                return 0.0
            
            # CONSERVATIVE position sizing using buying power with enhanced safety margin
            # Keep larger safety buffer for margin requirements to reduce leverage
            safety_buffer = max(2000.0, buying_power * 0.25)  # 25% buffer or $2000, whichever is larger
            available_funds = buying_power - safety_buffer
            
            if available_funds <= 0:
                logger.info(f"No available funds after safety buffer: buying_power=${buying_power}, buffer=${safety_buffer}")
                return 0.0
            
            # Log margin status for visibility
            if cash_balance < 0:
                logger.debug(f"Position sizing with margin: cash=${cash_balance:,.2f}, buying_power=${buying_power:,.2f}")
            
            # Base position size - REDUCED leverage: maximum 5% of portfolio OR 30% of available funds (whichever is smaller)
            base_size = min(portfolio_value * 0.05, available_funds * 0.30)
            
            # Adjust for confidence - REDUCED multipliers for less aggressive leverage
            confidence_multiplier = {
                'very_high': 1.2,  # Reduced from 1.5
                'high': 1.1,       # Reduced from 1.2
                'medium': 1.0,
                'low': 0.8,        # Increased from 0.7
                'very_low': 0.4    # Increased from 0.3
            }.get(analysis.confidence.value, 1.0)
            
            # Adjust for signal strength - REDUCED multipliers for less aggressive leverage
            signal_multiplier = {
                'strong_buy': 1.15,  # Reduced from 1.3
                'buy': 1.0,
                'hold': 0.0,
                'sell': 1.0,
                'strong_sell': 1.15  # Reduced from 1.3
            }.get(analysis.signal.value, 1.0)
            
            # Adjust for risk
            risk_multiplier = max(0.5, 1.0 - analysis.risk_score)
            
            # Calculate final size with safety caps
            position_value = base_size * confidence_multiplier * signal_multiplier * risk_multiplier
            
            # Additional safety: Never exceed available cash
            position_value = min(position_value, available_cash)
            
            # Convert to quantity
            if analysis.price > 0:
                quantity = position_value / analysis.price
                # Final validation: ensure position value doesn't exceed cash
                final_position_value = quantity * analysis.price
                if final_position_value > available_cash:
                    logger.warning(f"Position value ${final_position_value:,.2f} exceeds available cash ${available_cash:,.2f}, reducing...")
                    quantity = available_cash / analysis.price
                
                return max(1.0, quantity) if quantity >= 1.0 else 0.0
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Position size calculation error: {e}")
            return 0.0
    
    async def _determine_order_side(self, signal: SignalType, symbol: str) -> Optional[str]:
        """Determine order side based on signal and current position with validation."""
        try:
            current_positions = alpaca_client.get_positions()
            current_position = next((pos for pos in current_positions if pos['symbol'] == symbol), None)
            
            if signal in [SignalType.BUY, SignalType.STRONG_BUY]:
                return "buy"
            elif signal == SignalType.SHORT:
                return "sell_short"
            elif signal in [SignalType.SELL, SignalType.STRONG_SELL]:
                return self._handle_sell_signal(current_position)
            else:  # HOLD signal
                return None
                
        except Exception as e:
            logger.error(f"Error determining order side for {symbol}: {e}")
            return None
    
    def _handle_sell_signal(self, current_position: Optional[Dict]) -> str:
        """Handle sell signal logic based on current position."""
        if current_position:
            current_qty = float(current_position['qty'])
            if current_qty > 0:
                # Have long position, sell it
                return "sell"
            elif current_qty < 0:
                # Already short, buy to cover
                return "buy"
        
        # No position to sell, consider shorting (will be validated later)
        return "sell_short"
    
    async def _validate_position_for_order(self, symbol: str, side: str, quantity: float) -> bool:
        """Validate that the intended order is possible given current positions."""
        try:
            current_positions = alpaca_client.get_positions()
            current_pos = next((pos for pos in current_positions if pos['symbol'] == symbol), None)
            
            if side == "sell":
                if not current_pos or float(current_pos['qty']) < quantity:
                    logger.warning(f"Cannot sell {quantity} {symbol}: insufficient long position")
                    return False
            elif side == "buy" and current_pos:
                # Check if trying to buy to cover more than short position
                current_qty = float(current_pos['qty'])
                if current_qty < 0 and quantity > abs(current_qty):
                    logger.warning(f"Cannot buy {quantity} {symbol}: exceeds short position of {abs(current_qty)}")
                    # Allow but adjust quantity
                    return True
            
            return True
            
        except Exception as e:
            logger.error(f"Position validation error for {symbol}: {e}")
            return False
    
    async def check_risk_management_triggers(self) -> List[TradingOrder]:
        """Check for stop-loss and risk management triggers that require immediate sells."""
        try:
            risk_orders = []
            current_positions = alpaca_client.get_positions()
            
            for pos_data in current_positions:
                symbol = pos_data['symbol']
                quantity = float(pos_data['qty'])
                avg_cost = float(pos_data['cost_basis']) / abs(quantity) if quantity != 0 else 0
                current_price = float(pos_data['current_price'])
                unrealized_pnl_pct = float(pos_data['unrealized_plpc'])
                market_value = float(pos_data['market_value'])
                
                # Skip if no position
                if quantity == 0:
                    continue
                
                # Check stop-loss triggers (both long and short positions)
                if quantity > 0:  # Long position
                    # Stop-loss: price dropped below threshold
                    if unrealized_pnl_pct <= -self.stop_loss_pct:
                        order = TradingOrder(
                            symbol=symbol,
                            side="sell",
                            quantity=abs(quantity),
                            order_type="market",
                            strategy=StrategyType.SINGLE_STOCK,
                            reasoning=f"Stop-loss triggered: {unrealized_pnl_pct:.1%} loss (limit: {self.stop_loss_pct:.1%})"
                        )
                        risk_orders.append(order)
                        logger.warning(f"🚨 Stop-loss trigger for {symbol}: {unrealized_pnl_pct:.1%} loss")
                        
                elif quantity < 0:  # Short position
                    # Stop-loss: price rose above threshold (loss on short)
                    if unrealized_pnl_pct <= -self.stop_loss_pct:
                        order = TradingOrder(
                            symbol=symbol,
                            side="buy",  # Buy to cover short
                            quantity=abs(quantity),
                            order_type="market",
                            strategy=StrategyType.SINGLE_STOCK,
                            reasoning=f"Short stop-loss triggered: {unrealized_pnl_pct:.1%} loss (limit: {self.stop_loss_pct:.1%})"
                        )
                        risk_orders.append(order)
                        logger.warning(f"🚨 Short stop-loss trigger for {symbol}: {unrealized_pnl_pct:.1%} loss")
                
                # Check position size limits
                account_info = alpaca_client.get_account_info()
                portfolio_value = account_info['portfolio_value']
                position_pct = abs(market_value) / portfolio_value if portfolio_value > 0 else 0
                
                if position_pct > self.max_position_size * 1.5:  # 50% over limit triggers partial sell
                    # Calculate how much to sell to get back to limit
                    target_value = portfolio_value * self.max_position_size
                    excess_value = abs(market_value) - target_value
                    excess_quantity = excess_value / current_price if current_price > 0 else 0
                    
                    if excess_quantity >= 1:  # Only if at least 1 share
                        side = "sell" if quantity > 0 else "buy"
                        order = TradingOrder(
                            symbol=symbol,
                            side=side,
                            quantity=excess_quantity,
                            order_type="limit",
                            limit_price=current_price * 0.99,  # Slightly below market for quick fill
                            strategy=StrategyType.SINGLE_STOCK,
                            reasoning=f"Position size limit: {position_pct:.1%} > {self.max_position_size:.1%} limit"
                        )
                        risk_orders.append(order)
                        logger.warning(f"⚖️ Position size trigger for {symbol}: {position_pct:.1%} > limit")
            
            return risk_orders
            
        except Exception as e:
            logger.error(f"Risk management check error: {e}")
            return []
    
    async def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Get comprehensive portfolio metrics."""
        try:
            # Get account and positions
            account_info = alpaca_client.get_account_info()
            position_data = alpaca_client.get_positions()
            
            # Convert to Position objects
            positions = []
            allocation = {}
            
            portfolio_value = account_info['portfolio_value']
            
            for pos_data in position_data:
                position = Position(
                    symbol=pos_data['symbol'],
                    quantity=float(pos_data['qty']),
                    avg_cost=float(pos_data['cost_basis']) / abs(float(pos_data['qty'])),
                    current_price=float(pos_data['current_price']),
                    market_value=float(pos_data['market_value']),
                    unrealized_pnl=float(pos_data['unrealized_pl']),
                    unrealized_pnl_pct=float(pos_data['unrealized_plpc'])
                )
                positions.append(position)
                
                # Calculate allocation
                if portfolio_value > 0:
                    allocation[position.symbol] = abs(position.market_value) / portfolio_value
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics(positions)
            
            return PortfolioMetrics(
                total_value=portfolio_value,
                cash=float(account_info['cash']),
                equity=float(account_info['equity']),
                day_pnl=sum(pos.unrealized_pnl for pos in positions),
                total_pnl=sum(pos.unrealized_pnl for pos in positions),  # Simplified
                positions=positions,
                allocation=allocation,
                risk_metrics=risk_metrics
            )
            
        except Exception as e:
            logger.error(f"Portfolio metrics calculation error: {e}")
            return PortfolioMetrics(
                total_value=0.0, cash=0.0, equity=0.0, day_pnl=0.0, total_pnl=0.0,
                positions=[], allocation={}, risk_metrics={}
            )
    
    async def _calculate_risk_metrics(self, positions: List[Position]) -> Dict[str, float]:
        """Calculate portfolio risk metrics."""
        try:
            if not positions:
                return {'volatility': 0.0, 'var_1day': 0.0, 'concentration_risk': 0.0}
            
            # Get historical data for volatility calculation
            symbols = [pos.symbol for pos in positions]
            
            # Simple risk metrics for now
            total_value = sum(abs(pos.market_value) for pos in positions)
            
            # Concentration risk (max position percentage)
            if total_value > 0:
                concentrations = [abs(pos.market_value) / total_value for pos in positions]
                max_concentration = max(concentrations) if concentrations else 0.0
            else:
                max_concentration = 0.0
            
            return {
                'volatility': 0.15,  # Placeholder - would calculate from historical data
                'var_1day': 0.02,    # Placeholder - 2% daily VaR
                'concentration_risk': max_concentration,
                'num_positions': len(positions)
            }
            
        except Exception as e:
            logger.error(f"Risk metrics calculation error: {e}")
            return {'volatility': 0.0, 'var_1day': 0.0, 'concentration_risk': 0.0}
    
    async def monitor_orders(self):
        """Monitor and update order status."""
        try:
            if not self.pending_orders:
                return
            
            # Get current order status from Alpaca
            alpaca_orders = alpaca_client.get_orders(status="all", limit=50)
            
            # Update pending orders
            for order in self.pending_orders[:]:  # Copy list to modify during iteration
                if not order.order_id:
                    continue
                
                # Find matching Alpaca order
                alpaca_order = next((o for o in alpaca_orders if o['id'] == order.order_id), None)
                
                if alpaca_order:
                    if alpaca_order['status'] == 'filled':
                        order.status = OrderStatus.FILLED
                        self.pending_orders.remove(order)
                        logger.info(f"Order filled: {order.symbol} {order.side} {order.quantity}")
                    elif alpaca_order['status'] in ['cancelled', 'expired']:
                        order.status = OrderStatus.CANCELLED
                        self.pending_orders.remove(order)
                        logger.info(f"Order cancelled: {order.symbol}")
            
        except Exception as e:
            logger.error(f"Order monitoring error: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get trading performance summary."""
        try:
            filled_orders = [o for o in self.order_history if o.status == OrderStatus.FILLED]
            
            if not filled_orders:
                return {'total_orders': 0, 'success_rate': 0.0, 'strategies': {}}
            
            # Calculate basic metrics
            total_orders = len(filled_orders)
            
            # Group by strategy
            strategy_counts = {}
            for order in filled_orders:
                strategy = order.strategy.value
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            return {
                'total_orders': total_orders,
                'pending_orders': len(self.pending_orders),
                'success_rate': total_orders / len(self.order_history) if self.order_history else 0.0,
                'strategies': strategy_counts,
                'recent_orders': [
                    {
                        'symbol': o.symbol,
                        'side': o.side,
                        'quantity': o.quantity,
                        'strategy': o.strategy.value,
                        'timestamp': o.timestamp.isoformat()
                    }
                    for o in filled_orders[-10:]  # Last 10 orders
                ]
            }
            
        except Exception as e:
            logger.error(f"Performance summary error: {e}")
            return {'total_orders': 0, 'success_rate': 0.0, 'strategies': {}}

# Global instance
trading_engine = UnifiedTradingEngine()

# Convenience functions
async def execute_signal(analysis: AnalysisResult) -> Optional[TradingOrder]:
    """Execute a trading signal."""
    return await trading_engine.execute_analysis_signal(analysis)

async def execute_pairs_trade(symbol1: str, symbol2: str, z_score: float, hedge_ratio: float) -> List[TradingOrder]:
    """Execute a pairs trading strategy."""
    return await trading_engine.execute_pairs_trade(symbol1, symbol2, z_score, hedge_ratio)

async def rebalance_portfolio(target_allocation: Dict[str, float], force: bool = False) -> List[TradingOrder]:
    """Rebalance portfolio to target allocation."""
    return await trading_engine.rebalance_portfolio(target_allocation, force)

async def get_portfolio_metrics() -> PortfolioMetrics:
    """Get current portfolio metrics."""
    return await trading_engine.get_portfolio_metrics()

__all__ = [
    'UnifiedTradingEngine',
    'TradingOrder',
    'Position',
    'PortfolioMetrics',
    'OrderStatus',
    'StrategyType',
    'trading_engine',
    'execute_signal',
    'execute_pairs_trade',
    'rebalance_portfolio',
    'get_portfolio_metrics'
]