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
    
    async def execute_analysis_signal(self, analysis: AnalysisResult) -> Optional[TradingOrder]:
        """Execute trading based on analysis signal."""
        try:
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
            
            # Determine order side
            if analysis.signal in [SignalType.BUY, SignalType.STRONG_BUY]:
                side = "buy"
            else:
                side = "sell"
            
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
        """Rebalance portfolio to target allocation."""
        try:
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
                side = "buy" if trade_value > 0 else "sell"
                
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
        """Execute a trading order with risk checks."""
        try:
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
        """Validate order against risk parameters."""
        try:
            # Get account info
            account_info = alpaca_client.get_account_info()
            
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
            
            # Check buying power for buy orders
            if order.side == "buy":
                buying_power = account_info['buying_power']
                if order_value > buying_power:
                    logger.error(f"Insufficient buying power: ${order_value:,.2f} > ${buying_power:,.2f}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Order validation error: {e}")
            return False
    
    async def _calculate_position_size(self, analysis: AnalysisResult) -> float:
        """Calculate position size based on analysis and risk parameters."""
        try:
            account_info = alpaca_client.get_account_info()
            portfolio_value = account_info['portfolio_value']
            buying_power = account_info['buying_power']
            
            if portfolio_value <= 0:
                return 0.0
            
            # Base position size - use smaller of 15% portfolio or 80% buying power
            base_size = min(portfolio_value * 0.15, buying_power * 0.8)
            
            # Adjust for confidence
            confidence_multiplier = {
                'very_high': 1.5,
                'high': 1.2,
                'medium': 1.0,
                'low': 0.7,
                'very_low': 0.3
            }.get(analysis.confidence.value, 1.0)
            
            # Adjust for signal strength
            signal_multiplier = {
                'strong_buy': 1.3,
                'buy': 1.0,
                'hold': 0.0,
                'sell': 1.0,
                'strong_sell': 1.3
            }.get(analysis.signal.value, 1.0)
            
            # Adjust for risk
            risk_multiplier = max(0.5, 1.0 - analysis.risk_score)
            
            # Calculate final size
            position_value = base_size * confidence_multiplier * signal_multiplier * risk_multiplier
            
            # Convert to quantity
            if analysis.price > 0:
                quantity = position_value / analysis.price
                return max(1.0, quantity)  # Minimum 1 share
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Position size calculation error: {e}")
            return 0.0
    
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