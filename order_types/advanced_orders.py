#!/usr/bin/env python3
"""
Advanced Order Types Management System

Implements stop limits, trailing stops, OCO orders, and other advanced order types
for sophisticated risk management and execution strategies.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
from decimal import Decimal
import asyncio

logger = logging.getLogger(__name__)

OrderType = Literal["market", "limit", "stop", "stop_limit", "trailing_stop", "oco"]
OrderSide = Literal["buy", "sell", "sell_short"]
TimeInForce = Literal["gtc", "day", "ioc", "fok"]

@dataclass
class AdvancedOrderRequest:
    """Advanced order request with comprehensive parameters."""
    symbol: str
    quantity: float
    side: OrderSide
    order_type: OrderType
    
    # Price parameters
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_amount: Optional[float] = None
    trail_percent: Optional[float] = None
    
    # Execution parameters
    time_in_force: TimeInForce = "gtc"
    extended_hours: bool = False
    
    # Risk management
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    max_position_size: Optional[float] = None
    
    # OCO (One-Cancels-Other) parameters
    oco_group_id: Optional[str] = None
    oco_legs: Optional[List['AdvancedOrderRequest']] = None
    
    # Metadata
    reasoning: str = ""
    confidence: float = 0.5
    agent_source: str = "advanced_orders"
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class OrderExecutionResult:
    """Result of order execution with detailed information."""
    success: bool
    order_id: Optional[str] = None
    error_message: Optional[str] = None
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    status: str = "pending"
    execution_time: float = 0.0
    related_orders: List[str] = None
    
    def __post_init__(self):
        if self.related_orders is None:
            self.related_orders = []

class AdvancedOrderManager:
    """Manages advanced order types with sophisticated risk management."""
    
    def __init__(self):
        """Initialize advanced order manager."""
        self.active_orders: Dict[str, AdvancedOrderRequest] = {}
        self.order_history: List[Dict] = []
        self.oco_groups: Dict[str, List[str]] = {}
        self.stop_loss_orders: Dict[str, str] = {}  # symbol -> order_id
        self.take_profit_orders: Dict[str, str] = {}  # symbol -> order_id
        
    async def place_advanced_order(self, order_request: AdvancedOrderRequest) -> OrderExecutionResult:
        """Place an advanced order with comprehensive validation and risk management."""
        try:
            logger.info(f"Processing advanced order: {order_request.side} {order_request.quantity} "
                       f"{order_request.symbol} @ {order_request.order_type}")
            
            # Validate order request
            validation_result = self._validate_order_request(order_request)
            if not validation_result.success:
                return OrderExecutionResult(
                    success=False,
                    error_message=f"Order validation failed: {validation_result.error_message}"
                )
            
            # Get current market data for price validation
            market_data = await self._get_market_data(order_request.symbol)
            if not market_data:
                return OrderExecutionResult(
                    success=False,
                    error_message=f"Could not get market data for {order_request.symbol}"
                )
            
            current_price = market_data.get("current_price", 0)
            
            # Process different order types
            if order_request.order_type == "stop_limit":
                return await self._place_stop_limit_order(order_request, current_price)
            elif order_request.order_type == "trailing_stop":
                return await self._place_trailing_stop_order(order_request, current_price)
            elif order_request.order_type in ["oco", "oco_bracket", "oco_breakout"]:
                return await self._place_oco_order(order_request, current_price)
            elif order_request.order_type in ["market", "limit", "stop"]:
                return await self._place_standard_order(order_request, current_price)
            else:
                return OrderExecutionResult(
                    success=False,
                    error_message=f"Unsupported order type: {order_request.order_type}"
                )
                
        except Exception as e:
            logger.error(f"Advanced order placement failed: {e}")
            return OrderExecutionResult(
                success=False,
                error_message=str(e)
            )
    
    async def _place_stop_limit_order(self, order_request: AdvancedOrderRequest, 
                                    current_price: float) -> OrderExecutionResult:
        """Place a stop-limit order with automatic price validation."""
        
        # Validate stop-limit parameters
        if order_request.stop_price is None or order_request.limit_price is None:
            return OrderExecutionResult(
                success=False,
                error_message="Stop-limit orders require both stop_price and limit_price"
            )
        
        # Validate price logic for stop-limit orders
        if order_request.side == "buy":
            # Buy stop-limit: stop_price should be above current, limit_price >= stop_price
            if order_request.stop_price <= current_price:
                return OrderExecutionResult(
                    success=False,
                    error_message=f"Buy stop price ({order_request.stop_price}) must be above current price ({current_price})"
                )
            if order_request.limit_price < order_request.stop_price:
                logger.warning(f"Buy stop-limit: limit price ({order_request.limit_price}) below stop price ({order_request.stop_price})")
        
        elif order_request.side in ["sell", "sell_short"]:
            # Sell stop-limit: stop_price should be below current, limit_price <= stop_price
            if order_request.stop_price >= current_price:
                return OrderExecutionResult(
                    success=False,
                    error_message=f"Sell stop price ({order_request.stop_price}) must be below current price ({current_price})"
                )
            if order_request.limit_price > order_request.stop_price:
                logger.warning(f"Sell stop-limit: limit price ({order_request.limit_price}) above stop price ({order_request.stop_price})")
        
        # Place the order using Alpaca client
        from tools.alpaca_client import alpaca_client
        
        start_time = datetime.now()
        try:
            order_result = alpaca_client.place_order(
                symbol=order_request.symbol,
                qty=order_request.quantity,
                side=order_request.side,
                order_type="stop_limit",
                limit_price=order_request.limit_price,
                stop_price=order_request.stop_price,
                time_in_force=order_request.time_in_force
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Store order for tracking
            self.active_orders[order_result["id"]] = order_request
            
            # Set up automatic stop-loss/take-profit if specified
            related_orders = []
            if order_request.stop_loss_price or order_request.take_profit_price:
                related_orders = await self._setup_risk_management_orders(
                    order_request, order_result["id"]
                )
            
            logger.info(f"Stop-limit order placed: {order_result['id']} for {order_request.symbol}")
            
            return OrderExecutionResult(
                success=True,
                order_id=order_result["id"],
                status=order_result["status"],
                execution_time=execution_time,
                related_orders=related_orders
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Stop-limit order failed: {e}")
            return OrderExecutionResult(
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    async def _place_trailing_stop_order(self, order_request: AdvancedOrderRequest, 
                                       current_price: float) -> OrderExecutionResult:
        """Place a trailing stop order (simulated using periodic price monitoring)."""
        
        # Validate trailing stop parameters
        if not (order_request.trail_amount or order_request.trail_percent):
            return OrderExecutionResult(
                success=False,
                error_message="Trailing stop orders require either trail_amount or trail_percent"
            )
        
        # Calculate initial stop price
        if order_request.trail_percent:
            trail_distance = current_price * (order_request.trail_percent / 100)
        else:
            trail_distance = order_request.trail_amount
        
        if order_request.side in ["sell", "sell_short"]:
            initial_stop_price = current_price - trail_distance
        else:  # buy trailing stop (less common)
            initial_stop_price = current_price + trail_distance
        
        # Place initial stop order
        from tools.alpaca_client import alpaca_client
        
        try:
            order_result = alpaca_client.place_order(
                symbol=order_request.symbol,
                qty=order_request.quantity,
                side=order_request.side,
                order_type="stop",
                stop_price=initial_stop_price,
                time_in_force=order_request.time_in_force
            )
            
            # Store trailing stop information for monitoring
            self.active_orders[order_result["id"]] = order_request
            
            # Start trailing stop monitoring task
            asyncio.create_task(self._monitor_trailing_stop(
                order_result["id"], order_request, initial_stop_price, trail_distance
            ))
            
            logger.info(f"Trailing stop order initiated: {order_result['id']} for {order_request.symbol}")
            
            return OrderExecutionResult(
                success=True,
                order_id=order_result["id"],
                status=order_result["status"]
            )
            
        except Exception as e:
            logger.error(f"Trailing stop order failed: {e}")
            return OrderExecutionResult(
                success=False,
                error_message=str(e)
            )
    
    async def _place_oco_order(self, order_request: AdvancedOrderRequest, 
                             current_price: float) -> OrderExecutionResult:
        """Place One-Cancels-Other order (take profit + stop loss)."""
        
        if not order_request.oco_legs or len(order_request.oco_legs) != 2:
            return OrderExecutionResult(
                success=False,
                error_message="OCO orders require exactly 2 order legs"
            )
        
        oco_group_id = order_request.oco_group_id or f"oco_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        placed_orders = []
        
        try:
            # Place both legs of the OCO order
            for i, leg in enumerate(order_request.oco_legs):
                leg.oco_group_id = oco_group_id  # Ensure same group ID
                
                result = await self.place_advanced_order(leg)
                if result.success:
                    placed_orders.append(result.order_id)
                else:
                    # If any leg fails, cancel previously placed orders
                    for order_id in placed_orders:
                        await self._cancel_order(order_id)
                    
                    return OrderExecutionResult(
                        success=False,
                        error_message=f"OCO leg {i+1} failed: {result.error_message}"
                    )
            
            # Store OCO group for monitoring
            self.oco_groups[oco_group_id] = placed_orders
            
            # Monitor OCO group for cancellation logic
            asyncio.create_task(self._monitor_oco_group(oco_group_id, placed_orders))
            
            logger.info(f"OCO order placed: group {oco_group_id} with orders {placed_orders}")
            
            return OrderExecutionResult(
                success=True,
                order_id=oco_group_id,  # Return group ID as primary identifier
                related_orders=placed_orders,
                status="active"
            )
            
        except Exception as e:
            # Clean up any placed orders
            for order_id in placed_orders:
                await self._cancel_order(order_id)
            
            logger.error(f"OCO order failed: {e}")
            return OrderExecutionResult(
                success=False,
                error_message=str(e)
            )
    
    async def _place_standard_order(self, order_request: AdvancedOrderRequest, 
                                  current_price: float) -> OrderExecutionResult:
        """Place standard market, limit, or stop orders."""
        
        from tools.alpaca_client import alpaca_client
        
        start_time = datetime.now()
        try:
            order_result = alpaca_client.place_order(
                symbol=order_request.symbol,
                qty=order_request.quantity,
                side=order_request.side,
                order_type=order_request.order_type,
                limit_price=order_request.limit_price,
                stop_price=order_request.stop_price,
                time_in_force=order_request.time_in_force
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Store order for tracking
            self.active_orders[order_result["id"]] = order_request
            
            # Set up risk management orders if specified
            related_orders = []
            if order_request.stop_loss_price or order_request.take_profit_price:
                related_orders = await self._setup_risk_management_orders(
                    order_request, order_result["id"]
                )
            
            logger.info(f"{order_request.order_type.title()} order placed: {order_result['id']} for {order_request.symbol}")
            
            return OrderExecutionResult(
                success=True,
                order_id=order_result["id"],
                status=order_result["status"],
                execution_time=execution_time,
                related_orders=related_orders
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"{order_request.order_type.title()} order failed: {e}")
            return OrderExecutionResult(
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    async def _setup_risk_management_orders(self, original_order: AdvancedOrderRequest, 
                                          parent_order_id: str) -> List[str]:
        """Set up automatic stop-loss and take-profit orders."""
        risk_orders = []
        
        try:
            # Wait for parent order to fill before placing risk management orders
            await asyncio.sleep(2)  # Brief delay to allow fill
            
            # Check if parent order was filled
            from tools.alpaca_client import alpaca_client
            orders = alpaca_client.get_orders(status="all", limit=50)
            parent_order = next((o for o in orders if o["id"] == parent_order_id), None)
            
            if not parent_order or float(parent_order.get("filled_qty", 0)) == 0:
                logger.warning(f"Parent order {parent_order_id} not filled, skipping risk management orders")
                return risk_orders
            
            filled_qty = float(parent_order["filled_qty"])
            
            # Place stop-loss order
            if original_order.stop_loss_price:
                stop_loss_side = "sell" if original_order.side == "buy" else "buy"
                
                stop_loss_request = AdvancedOrderRequest(
                    symbol=original_order.symbol,
                    quantity=filled_qty,
                    side=stop_loss_side,
                    order_type="stop",
                    stop_price=original_order.stop_loss_price,
                    time_in_force=original_order.time_in_force,
                    reasoning=f"Stop-loss for order {parent_order_id}",
                    agent_source="risk_management"
                )
                
                stop_result = await self._place_standard_order(stop_loss_request, original_order.stop_loss_price)
                if stop_result.success:
                    risk_orders.append(stop_result.order_id)
                    self.stop_loss_orders[original_order.symbol] = stop_result.order_id
                    logger.info(f"Stop-loss order placed: {stop_result.order_id}")
            
            # Place take-profit order
            if original_order.take_profit_price:
                take_profit_side = "sell" if original_order.side == "buy" else "buy"
                
                take_profit_request = AdvancedOrderRequest(
                    symbol=original_order.symbol,
                    quantity=filled_qty,
                    side=take_profit_side,
                    order_type="limit",
                    limit_price=original_order.take_profit_price,
                    time_in_force=original_order.time_in_force,
                    reasoning=f"Take-profit for order {parent_order_id}",
                    agent_source="risk_management"
                )
                
                profit_result = await self._place_standard_order(take_profit_request, original_order.take_profit_price)
                if profit_result.success:
                    risk_orders.append(profit_result.order_id)
                    self.take_profit_orders[original_order.symbol] = profit_result.order_id
                    logger.info(f"Take-profit order placed: {profit_result.order_id}")
            
            return risk_orders
            
        except Exception as e:
            logger.error(f"Risk management order setup failed: {e}")
            return risk_orders
    
    def _validate_order_request(self, order_request: AdvancedOrderRequest) -> OrderExecutionResult:
        """Validate order request parameters."""
        
        # Basic parameter validation
        if not order_request.symbol or len(order_request.symbol.strip()) == 0:
            return OrderExecutionResult(success=False, error_message="Symbol is required")
        
        if order_request.quantity <= 0:
            return OrderExecutionResult(success=False, error_message="Quantity must be positive")
        
        if order_request.order_type not in ["market", "limit", "stop", "stop_limit", "trailing_stop", "oco", "oco_bracket", "oco_breakout"]:
            return OrderExecutionResult(success=False, error_message=f"Invalid order type: {order_request.order_type}")
        
        if order_request.side not in ["buy", "sell", "sell_short"]:
            return OrderExecutionResult(success=False, error_message=f"Invalid order side: {order_request.side}")
        
        # Order-type specific validation
        if order_request.order_type == "limit" and order_request.limit_price is None:
            return OrderExecutionResult(success=False, error_message="Limit orders require limit_price")
        
        if order_request.order_type == "stop" and order_request.stop_price is None:
            return OrderExecutionResult(success=False, error_message="Stop orders require stop_price")
        
        if order_request.order_type == "stop_limit":
            if order_request.limit_price is None or order_request.stop_price is None:
                return OrderExecutionResult(success=False, error_message="Stop-limit orders require both limit_price and stop_price")
        
        # Risk management validation
        if order_request.stop_loss_price and order_request.take_profit_price:
            if order_request.side == "buy":
                if order_request.stop_loss_price >= order_request.take_profit_price:
                    return OrderExecutionResult(success=False, error_message="For buy orders: stop_loss_price must be less than take_profit_price")
            else:
                if order_request.stop_loss_price <= order_request.take_profit_price:
                    return OrderExecutionResult(success=False, error_message="For sell orders: stop_loss_price must be greater than take_profit_price")
        
        return OrderExecutionResult(success=True)
    
    async def _get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get current market data for a symbol."""
        try:
            from tools.alpaca_client import alpaca_client
            
            # Get recent market data
            df = alpaca_client.get_market_data(symbol, timeframe="1Day", limit=1)
            if df.empty:
                return None
            
            return {
                "current_price": float(df.iloc[-1]["close"]),
                "high": float(df.iloc[-1]["high"]),
                "low": float(df.iloc[-1]["low"]),
                "volume": float(df.iloc[-1]["volume"])
            }
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return None
    
    async def _cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            from tools.alpaca_client import alpaca_client
            return alpaca_client.cancel_order(order_id)
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def _monitor_trailing_stop(self, order_id: str, order_request: AdvancedOrderRequest,
                                   current_stop_price: float, trail_distance: float):
        """Monitor and update trailing stop orders."""
        try:
            symbol = order_request.symbol
            
            while order_id in self.active_orders:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Get current market price
                market_data = await self._get_market_data(symbol)
                if not market_data:
                    continue
                
                current_price = market_data["current_price"]
                
                # Calculate new stop price
                if order_request.side in ["sell", "sell_short"]:
                    new_stop_price = current_price - trail_distance
                    if new_stop_price > current_stop_price:
                        # Update stop price (trail up)
                        await self._update_trailing_stop(order_id, new_stop_price)
                        current_stop_price = new_stop_price
                        logger.info(f"Trailing stop updated for {symbol}: new stop price ${new_stop_price:.2f}")
                
        except Exception as e:
            logger.error(f"Trailing stop monitoring failed for {order_id}: {e}")
    
    async def _update_trailing_stop(self, order_id: str, new_stop_price: float):
        """Update trailing stop order with new stop price."""
        try:
            from tools.alpaca_client import alpaca_client
            
            # Cancel current order
            if alpaca_client.cancel_order(order_id):
                # Place new order with updated stop price
                order_request = self.active_orders[order_id]
                
                new_order = alpaca_client.place_order(
                    symbol=order_request.symbol,
                    qty=order_request.quantity,
                    side=order_request.side,
                    order_type="stop",
                    stop_price=new_stop_price,
                    time_in_force=order_request.time_in_force
                )
                
                # Update tracking
                del self.active_orders[order_id]
                self.active_orders[new_order["id"]] = order_request
                
                return new_order["id"]
        
        except Exception as e:
            logger.error(f"Failed to update trailing stop {order_id}: {e}")
            return None
    
    async def _monitor_oco_group(self, group_id: str, order_ids: List[str]):
        """Monitor OCO group and cancel remaining orders when one fills."""
        try:
            from tools.alpaca_client import alpaca_client
            
            while group_id in self.oco_groups:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Check status of all orders in group
                orders = alpaca_client.get_orders(status="all", limit=100)
                
                filled_order = None
                for order in orders:
                    if order["id"] in order_ids and float(order.get("filled_qty", 0)) > 0:
                        filled_order = order["id"]
                        break
                
                if filled_order:
                    # Cancel all other orders in the group
                    for order_id in order_ids:
                        if order_id != filled_order:
                            await self._cancel_order(order_id)
                            if order_id in self.active_orders:
                                del self.active_orders[order_id]
                    
                    # Remove from OCO tracking
                    del self.oco_groups[group_id]
                    logger.info(f"OCO group {group_id} completed: order {filled_order} filled, others cancelled")
                    break
        
        except Exception as e:
            logger.error(f"OCO monitoring failed for group {group_id}: {e}")
    
    def get_active_orders_summary(self) -> Dict:
        """Get summary of all active advanced orders."""
        return {
            "total_active_orders": len(self.active_orders),
            "oco_groups": len(self.oco_groups),
            "stop_loss_orders": len(self.stop_loss_orders),
            "take_profit_orders": len(self.take_profit_orders),
            "order_types": [order.order_type for order in self.active_orders.values()]
        }

# Global advanced order manager instance
advanced_order_manager = AdvancedOrderManager()

# Convenience functions for easy integration
async def place_stop_limit_order(symbol: str, quantity: float, side: OrderSide,
                                stop_price: float, limit_price: float,
                                stop_loss: Optional[float] = None,
                                take_profit: Optional[float] = None,
                                reasoning: str = "") -> OrderExecutionResult:
    """Place a stop-limit order with optional risk management."""
    
    order_request = AdvancedOrderRequest(
        symbol=symbol,
        quantity=quantity,
        side=side,
        order_type="stop_limit",
        stop_price=stop_price,
        limit_price=limit_price,
        stop_loss_price=stop_loss,
        take_profit_price=take_profit,
        reasoning=reasoning,
        agent_source="convenience_function"
    )
    
    return await advanced_order_manager.place_advanced_order(order_request)

async def place_trailing_stop_order(symbol: str, quantity: float, side: OrderSide,
                                   trail_percent: float, reasoning: str = "") -> OrderExecutionResult:
    """Place a trailing stop order."""
    
    order_request = AdvancedOrderRequest(
        symbol=symbol,
        quantity=quantity,
        side=side,
        order_type="trailing_stop",
        trail_percent=trail_percent,
        reasoning=reasoning,
        agent_source="convenience_function"
    )
    
    return await advanced_order_manager.place_advanced_order(order_request)

async def place_oco_order(symbol: str, quantity: float, side: OrderSide,
                         take_profit_price: float, stop_loss_price: float,
                         reasoning: str = "") -> OrderExecutionResult:
    """Place a One-Cancels-Other order (take profit + stop loss)."""
    
    # Create the two legs of the OCO order
    opposite_side = "sell" if side == "buy" else "buy"
    
    take_profit_leg = AdvancedOrderRequest(
        symbol=symbol,
        quantity=quantity,
        side=opposite_side,
        order_type="limit",
        limit_price=take_profit_price,
        reasoning=f"Take profit leg: {reasoning}"
    )
    
    stop_loss_leg = AdvancedOrderRequest(
        symbol=symbol,
        quantity=quantity,
        side=opposite_side,
        order_type="stop",
        stop_price=stop_loss_price,
        reasoning=f"Stop loss leg: {reasoning}"
    )
    
    oco_request = AdvancedOrderRequest(
        symbol=symbol,
        quantity=quantity,
        side=side,
        order_type="oco",
        oco_legs=[take_profit_leg, stop_loss_leg],
        reasoning=reasoning,
        agent_source="convenience_function"
    )
    
    return await advanced_order_manager.place_advanced_order(oco_request)