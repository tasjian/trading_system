#!/usr/bin/env python3
"""
OCO (One-Cancels-Other) Order Management System

Provides comprehensive OCO trading functionality including:
- Automatic take-profit and stop-loss placement for existing positions
- Breakout trading strategies for new positions  
- Position monitoring and risk management
- Integration with existing trading engine and portfolio balancer
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from tools.alpaca_client import alpaca_client
from config.settings import settings

logger = logging.getLogger(__name__)

class OCOType(Enum):
    """Types of OCO orders."""
    BRACKET = "bracket"  # Take profit + stop loss for existing position
    BREAKOUT = "breakout"  # Buy breakout + sell breakout for new position
    PROTECT = "protect"  # Protect existing position with OCO
    SCALP = "scalp"  # Short-term scalping OCO

class OCOStatus(Enum):
    """OCO order status."""
    PENDING = "pending"
    ACTIVE = "active"
    PARTIALLY_FILLED = "partially_filled"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

@dataclass
class OCOConfiguration:
    """Configuration for OCO order parameters."""
    # Risk management
    default_take_profit_percent: float = settings.oco_default_take_profit_percent
    default_stop_loss_percent: float = settings.oco_default_stop_loss_percent
    max_oco_position_size: float = settings.oco_max_position_size
    
    # Breakout parameters
    breakout_buffer_percent: float = settings.oco_breakout_buffer_percent
    breakout_lookback_period: int = settings.oco_breakout_lookback_period
    breakout_volatility_multiplier: float = settings.oco_breakout_volatility_multiplier
    
    # Monitoring
    position_check_interval: int = settings.oco_position_check_interval
    oco_status_check_interval: int = settings.oco_status_check_interval
    auto_oco_enabled: bool = settings.oco_auto_placement_enabled
    
    # Risk limits
    max_daily_oco_orders: int = settings.oco_max_daily_orders
    min_oco_profit_ratio: float = settings.oco_min_profit_ratio
    max_oco_holding_period_hours: int = settings.oco_max_holding_period_hours

@dataclass
class OCOOrder:
    """Represents an active OCO order."""
    oco_id: str
    symbol: str
    oco_type: OCOType
    status: OCOStatus
    created_at: datetime
    
    # Order details
    parent_order_id: Optional[str] = None
    take_profit_order_id: Optional[str] = None
    stop_loss_order_id: Optional[str] = None
    buy_order_id: Optional[str] = None  # For breakout OCO
    sell_order_id: Optional[str] = None  # For breakout OCO
    
    # Price levels
    entry_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    upper_breakout_price: Optional[float] = None  # For breakout OCO
    lower_breakout_price: Optional[float] = None  # For breakout OCO
    
    # Position details
    quantity: float = 0.0
    side: str = ""
    filled_quantity: float = 0.0
    
    # Metadata
    reasoning: str = ""
    confidence: float = 0.5
    last_updated: datetime = field(default_factory=datetime.now)
    expiry_time: Optional[datetime] = None

class OCOOrderManager:
    """Comprehensive OCO order management system."""
    
    def __init__(self, config: Optional[OCOConfiguration] = None):
        """Initialize OCO order manager."""
        self.config = config or OCOConfiguration()
        self.active_oco_orders: Dict[str, OCOOrder] = {}
        self.oco_history: List[OCOOrder] = []
        self.position_oco_mapping: Dict[str, str] = {}  # symbol -> oco_id
        self.daily_oco_count = 0
        self.last_daily_reset = datetime.now().date()
        
        # Monitoring tasks
        self._monitoring_task = None
        self._position_monitoring_task = None
        
        logger.info("OCO Order Manager initialized")
    
    async def start_monitoring(self):
        """Start OCO monitoring tasks."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitor_oco_orders())
            logger.info("OCO order monitoring started")
        
        if self._position_monitoring_task is None and self.config.auto_oco_enabled:
            self._position_monitoring_task = asyncio.create_task(self._monitor_positions_for_auto_oco())
            logger.info("Auto OCO position monitoring started")
    
    async def stop_monitoring(self):
        """Stop OCO monitoring tasks."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            self._monitoring_task = None
        
        if self._position_monitoring_task:
            self._position_monitoring_task.cancel()
            self._position_monitoring_task = None
        
        logger.info("OCO monitoring stopped")
    
    async def place_bracket_oco(self, symbol: str, quantity: float, side: str,
                              take_profit_percent: Optional[float] = None,
                              stop_loss_percent: Optional[float] = None,
                              reasoning: str = "") -> Optional[OCOOrder]:
        """
        Place a bracket OCO order (take profit + stop loss) for a position.
        
        Args:
            symbol: Stock symbol
            quantity: Quantity to trade
            side: "buy" or "sell"
            take_profit_percent: Take profit percentage (default from config)
            stop_loss_percent: Stop loss percentage (default from config)
            reasoning: Reasoning for the OCO order
            
        Returns:
            OCOOrder object if successful, None otherwise
        """
        try:
            # Check daily limits
            if not self._check_daily_limits():
                logger.warning(f"Daily OCO limit reached ({self.config.max_daily_oco_orders})")
                return None
            
            # Get current price
            current_price = alpaca_client.get_current_price(symbol)
            if not current_price:
                logger.error(f"Could not get current price for {symbol}")
                return None
            
            # Calculate take profit and stop loss prices
            tp_pct = take_profit_percent or self.config.default_take_profit_percent
            sl_pct = stop_loss_percent or self.config.default_stop_loss_percent
            
            if side.lower() == "buy":
                take_profit_price = current_price * (1 + tp_pct)
                stop_loss_price = current_price * (1 - sl_pct)
            else:  # sell
                take_profit_price = current_price * (1 - tp_pct)
                stop_loss_price = current_price * (1 + sl_pct)
            
            # Validate profit:loss ratio
            if not self._validate_risk_reward_ratio(current_price, take_profit_price, stop_loss_price, side):
                logger.warning(f"Risk:reward ratio too low for {symbol}")
                return None
            
            # Place OCO order with Alpaca
            order_result = alpaca_client.place_oco_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price
            )
            
            # Create OCO tracking object
            oco_id = f"oco_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}"
            oco_order = OCOOrder(
                oco_id=oco_id,
                symbol=symbol,
                oco_type=OCOType.BRACKET,
                status=OCOStatus.ACTIVE,
                created_at=datetime.now(),
                parent_order_id=order_result["id"],
                entry_price=current_price,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                quantity=quantity,
                side=side,
                reasoning=reasoning or f"Bracket OCO for {symbol}: TP={tp_pct:.1%}, SL={sl_pct:.1%}",
                confidence=0.9,
                expiry_time=datetime.now() + timedelta(hours=self.config.max_oco_holding_period_hours)
            )
            
            # Store OCO order
            self.active_oco_orders[oco_id] = oco_order
            self.position_oco_mapping[symbol] = oco_id
            self.daily_oco_count += 1
            
            logger.info(f"Bracket OCO placed: {oco_id} for {symbol} (TP: ${take_profit_price:.2f}, SL: ${stop_loss_price:.2f})")
            return oco_order
            
        except Exception as e:
            logger.error(f"Failed to place bracket OCO for {symbol}: {e}")
            return None
    
    async def place_breakout_oco(self, symbol: str, quantity: float,
                               upper_breakout_percent: Optional[float] = None,
                               lower_breakout_percent: Optional[float] = None,
                               reasoning: str = "") -> Optional[OCOOrder]:
        """
        Place a breakout OCO order that triggers on price breakouts.
        
        Args:
            symbol: Stock symbol
            quantity: Quantity for each breakout direction
            upper_breakout_percent: Percent above current price for buy breakout
            lower_breakout_percent: Percent below current price for sell breakout
            reasoning: Reasoning for the breakout strategy
            
        Returns:
            OCOOrder object if successful, None otherwise
        """
        try:
            # Check daily limits
            if not self._check_daily_limits():
                logger.warning(f"Daily OCO limit reached ({self.config.max_daily_oco_orders})")
                return None
            
            # Get current price and calculate breakout levels
            current_price = alpaca_client.get_current_price(symbol)
            if not current_price:
                logger.error(f"Could not get current price for {symbol}")
                return None
            
            # Calculate breakout levels (use volatility-based or percentage-based)
            if upper_breakout_percent and lower_breakout_percent:
                upper_breakout = current_price * (1 + upper_breakout_percent)
                lower_breakout = current_price * (1 - lower_breakout_percent)
            else:
                # Use ATR-based breakout levels
                upper_breakout, lower_breakout = await self._calculate_atr_breakout_levels(symbol, current_price)
            
            # Validate breakout levels
            if upper_breakout <= current_price or lower_breakout >= current_price:
                logger.error(f"Invalid breakout levels for {symbol}: upper={upper_breakout}, lower={lower_breakout}, current={current_price}")
                return None
            
            # Place breakout OCO order
            order_result = alpaca_client.place_breakout_oco_order(
                symbol=symbol,
                qty=quantity,
                upper_breakout_price=upper_breakout,
                lower_breakout_price=lower_breakout,
                limit_buffer_percent=self.config.breakout_buffer_percent
            )
            
            # Create OCO tracking object
            oco_id = f"breakout_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}"
            oco_order = OCOOrder(
                oco_id=oco_id,
                symbol=symbol,
                oco_type=OCOType.BREAKOUT,
                status=OCOStatus.ACTIVE,
                created_at=datetime.now(),
                buy_order_id=order_result["buy_order"]["id"],
                sell_order_id=order_result["sell_order"]["id"],
                entry_price=current_price,
                upper_breakout_price=upper_breakout,
                lower_breakout_price=lower_breakout,
                quantity=quantity,
                side="breakout",
                reasoning=reasoning or f"Breakout OCO for {symbol}: upper=${upper_breakout:.2f}, lower=${lower_breakout:.2f}",
                confidence=0.7,
                expiry_time=datetime.now() + timedelta(hours=self.config.max_oco_holding_period_hours)
            )
            
            # Store OCO order
            self.active_oco_orders[oco_id] = oco_order
            self.daily_oco_count += 1
            
            logger.info(f"Breakout OCO placed: {oco_id} for {symbol} (Upper: ${upper_breakout:.2f}, Lower: ${lower_breakout:.2f})")
            return oco_order
            
        except Exception as e:
            logger.error(f"Failed to place breakout OCO for {symbol}: {e}")
            return None
    
    async def auto_place_position_oco(self, symbol: str, current_position: Dict) -> Optional[OCOOrder]:
        """
        Automatically place OCO orders for existing positions that don't have them.
        
        Args:
            symbol: Stock symbol
            current_position: Position data from portfolio
            
        Returns:
            OCOOrder object if placed, None otherwise
        """
        try:
            # Check if position already has OCO
            if symbol in self.position_oco_mapping:
                existing_oco_id = self.position_oco_mapping[symbol]
                if existing_oco_id in self.active_oco_orders:
                    logger.debug(f"Position {symbol} already has active OCO: {existing_oco_id}")
                    return None
            
            # Get position details
            position_qty = float(current_position.get('quantity', 0))
            position_side = current_position.get('side', 'long')
            
            if abs(position_qty) == 0:
                return None  # No position to protect
            
            # Calculate position size as percentage of portfolio
            account_info = alpaca_client.get_account_info()
            portfolio_value = account_info.get('portfolio_value', 1)
            position_value = abs(float(current_position.get('market_value', 0)))
            position_weight = position_value / portfolio_value if portfolio_value > 0 else 0
            
            # Only create OCO for significant positions
            if position_weight < self.config.max_oco_position_size * 0.5:  # At least half the max OCO size
                logger.debug(f"Position {symbol} too small for auto OCO: {position_weight:.2%}")
                return None
            
            # Determine OCO side (opposite of position to close)
            oco_side = "sell" if position_side == "long" else "buy"
            
            # Place protective OCO
            oco_order = await self.place_bracket_oco(
                symbol=symbol,
                quantity=abs(position_qty),
                side=oco_side,
                reasoning=f"Auto protective OCO for existing {position_side} position"
            )
            
            if oco_order:
                logger.info(f"Auto OCO placed for position {symbol}: {oco_order.oco_id}")
            
            return oco_order
            
        except Exception as e:
            logger.error(f"Failed to auto place OCO for {symbol}: {e}")
            return None
    
    async def cancel_oco_order(self, oco_id: str) -> bool:
        """Cancel an active OCO order."""
        try:
            if oco_id not in self.active_oco_orders:
                logger.warning(f"OCO order {oco_id} not found")
                return False
            
            oco_order = self.active_oco_orders[oco_id]
            
            # Cancel all related orders
            cancelled_count = 0
            
            if oco_order.parent_order_id:
                if alpaca_client.cancel_order(oco_order.parent_order_id):
                    cancelled_count += 1
            
            if oco_order.buy_order_id:
                if alpaca_client.cancel_order(oco_order.buy_order_id):
                    cancelled_count += 1
            
            if oco_order.sell_order_id:
                if alpaca_client.cancel_order(oco_order.sell_order_id):
                    cancelled_count += 1
            
            # Update OCO status
            oco_order.status = OCOStatus.CANCELLED
            oco_order.last_updated = datetime.now()
            
            # Move to history
            self.oco_history.append(oco_order)
            del self.active_oco_orders[oco_id]
            
            # Remove from position mapping
            if oco_order.symbol in self.position_oco_mapping:
                del self.position_oco_mapping[oco_order.symbol]
            
            logger.info(f"OCO order cancelled: {oco_id} ({cancelled_count} orders cancelled)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel OCO order {oco_id}: {e}")
            return False
    
    async def get_oco_status(self, oco_id: str) -> Optional[Dict]:
        """Get detailed status of an OCO order."""
        try:
            if oco_id not in self.active_oco_orders:
                # Check history
                for historical_oco in self.oco_history:
                    if historical_oco.oco_id == oco_id:
                        return self._oco_to_dict(historical_oco)
                return None
            
            oco_order = self.active_oco_orders[oco_id]
            
            # Update with latest order status
            await self._update_oco_status(oco_order)
            
            return self._oco_to_dict(oco_order)
            
        except Exception as e:
            logger.error(f"Failed to get OCO status for {oco_id}: {e}")
            return None
    
    def get_active_oco_summary(self) -> Dict:
        """Get summary of all active OCO orders."""
        try:
            summary = {
                "total_active": len(self.active_oco_orders),
                "daily_count": self.daily_oco_count,
                "daily_limit": self.config.max_daily_oco_orders,
                "by_type": {},
                "by_symbol": {},
                "by_status": {}
            }
            
            for oco_order in self.active_oco_orders.values():
                # Count by type
                oco_type = oco_order.oco_type.value
                summary["by_type"][oco_type] = summary["by_type"].get(oco_type, 0) + 1
                
                # Count by symbol
                symbol = oco_order.symbol
                summary["by_symbol"][symbol] = summary["by_symbol"].get(symbol, 0) + 1
                
                # Count by status
                status = oco_order.status.value
                summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get OCO summary: {e}")
            return {}
    
    # Private methods
    
    def _check_daily_limits(self) -> bool:
        """Check if daily OCO limits allow new orders."""
        # Reset daily count if new day
        current_date = datetime.now().date()
        if current_date > self.last_daily_reset:
            self.daily_oco_count = 0
            self.last_daily_reset = current_date
        
        return self.daily_oco_count < self.config.max_daily_oco_orders
    
    def _validate_risk_reward_ratio(self, entry_price: float, take_profit: float, 
                                  stop_loss: float, side: str) -> bool:
        """Validate that risk:reward ratio meets minimum requirements."""
        try:
            if side.lower() == "buy":
                profit = take_profit - entry_price
                loss = entry_price - stop_loss
            else:  # sell
                profit = entry_price - take_profit
                loss = stop_loss - entry_price
            
            if loss <= 0:
                return False  # Invalid stop loss
            
            ratio = profit / loss
            return ratio >= self.config.min_oco_profit_ratio
            
        except Exception:
            return False
    
    async def _calculate_atr_breakout_levels(self, symbol: str, current_price: float) -> Tuple[float, float]:
        """Calculate breakout levels based on Average True Range (ATR)."""
        try:
            # Get recent market data for ATR calculation
            df = alpaca_client.get_market_data(symbol, timeframe="1Day", limit=self.config.breakout_lookback_period + 1)
            
            if df.empty or len(df) < 10:
                # Fallback to percentage-based breakout
                logger.warning(f"Insufficient data for ATR calculation for {symbol}, using percentage breakout")
                return (current_price * 1.02, current_price * 0.98)  # 2% breakout levels
            
            # Calculate True Range
            df['high_low'] = df['high'] - df['low']
            df['high_close'] = np.abs(df['high'] - df['close'].shift(1))
            df['low_close'] = np.abs(df['low'] - df['close'].shift(1))
            df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            
            # Calculate ATR
            atr = df['true_range'].rolling(window=14).mean().iloc[-1]
            
            # Calculate breakout levels
            multiplier = self.config.breakout_volatility_multiplier
            upper_breakout = current_price + (atr * multiplier)
            lower_breakout = current_price - (atr * multiplier)
            
            return (upper_breakout, lower_breakout)
            
        except Exception as e:
            logger.error(f"Failed to calculate ATR breakout levels for {symbol}: {e}")
            # Fallback to percentage-based breakout
            return (current_price * 1.02, current_price * 0.98)
    
    async def _monitor_oco_orders(self):
        """Monitor active OCO orders for status updates."""
        while True:
            try:
                await asyncio.sleep(self.config.oco_status_check_interval)
                
                # Check each active OCO order
                for oco_id, oco_order in list(self.active_oco_orders.items()):
                    await self._update_oco_status(oco_order)
                    
                    # Check for expiry
                    if oco_order.expiry_time and datetime.now() > oco_order.expiry_time:
                        logger.info(f"OCO order expired, cancelling: {oco_id}")
                        await self.cancel_oco_order(oco_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in OCO monitoring: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _monitor_positions_for_auto_oco(self):
        """Monitor positions and automatically place OCO orders."""
        while True:
            try:
                await asyncio.sleep(self.config.position_check_interval)
                
                # Get current positions
                positions = alpaca_client.get_positions()
                
                for position in positions:
                    symbol = position['symbol']
                    
                    # Skip if position already has OCO or is too small
                    if symbol in self.position_oco_mapping:
                        continue
                    
                    # Attempt to place auto OCO
                    await self.auto_place_position_oco(symbol, position)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in position monitoring: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def _update_oco_status(self, oco_order: OCOOrder):
        """Update OCO order status based on latest order information."""
        try:
            if oco_order.oco_type == OCOType.BRACKET:
                # Check bracket order status
                if oco_order.parent_order_id:
                    bracket_orders = alpaca_client.get_bracket_orders(oco_order.parent_order_id)
                    
                    filled_orders = [o for o in bracket_orders if float(o.get('filled_qty', 0)) > 0]
                    if filled_orders:
                        oco_order.status = OCOStatus.COMPLETED
                        oco_order.filled_quantity = sum(float(o.get('filled_qty', 0)) for o in filled_orders)
                        
                        # Move to history
                        self.oco_history.append(oco_order)
                        del self.active_oco_orders[oco_order.oco_id]
                        
                        if oco_order.symbol in self.position_oco_mapping:
                            del self.position_oco_mapping[oco_order.symbol]
                        
                        logger.info(f"OCO bracket order completed: {oco_order.oco_id}")
            
            elif oco_order.oco_type == OCOType.BREAKOUT:
                # Check breakout orders
                orders = alpaca_client.get_orders(status="all", limit=50)
                
                buy_filled = any(o["id"] == oco_order.buy_order_id and float(o.get("filled_qty", 0)) > 0 for o in orders)
                sell_filled = any(o["id"] == oco_order.sell_order_id and float(o.get("filled_qty", 0)) > 0 for o in orders)
                
                if buy_filled or sell_filled:
                    oco_order.status = OCOStatus.COMPLETED
                    
                    # Cancel the other order
                    if buy_filled and oco_order.sell_order_id:
                        alpaca_client.cancel_order(oco_order.sell_order_id)
                    elif sell_filled and oco_order.buy_order_id:
                        alpaca_client.cancel_order(oco_order.buy_order_id)
                    
                    # Move to history
                    self.oco_history.append(oco_order)
                    del self.active_oco_orders[oco_order.oco_id]
                    
                    logger.info(f"OCO breakout order completed: {oco_order.oco_id}")
            
            oco_order.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to update OCO status for {oco_order.oco_id}: {e}")
    
    def _oco_to_dict(self, oco_order: OCOOrder) -> Dict:
        """Convert OCO order to dictionary representation."""
        return {
            "oco_id": oco_order.oco_id,
            "symbol": oco_order.symbol,
            "oco_type": oco_order.oco_type.value,
            "status": oco_order.status.value,
            "created_at": oco_order.created_at.isoformat(),
            "parent_order_id": oco_order.parent_order_id,
            "take_profit_order_id": oco_order.take_profit_order_id,
            "stop_loss_order_id": oco_order.stop_loss_order_id,
            "buy_order_id": oco_order.buy_order_id,
            "sell_order_id": oco_order.sell_order_id,
            "entry_price": oco_order.entry_price,
            "take_profit_price": oco_order.take_profit_price,
            "stop_loss_price": oco_order.stop_loss_price,
            "upper_breakout_price": oco_order.upper_breakout_price,
            "lower_breakout_price": oco_order.lower_breakout_price,
            "quantity": oco_order.quantity,
            "side": oco_order.side,
            "filled_quantity": oco_order.filled_quantity,
            "reasoning": oco_order.reasoning,
            "confidence": oco_order.confidence,
            "last_updated": oco_order.last_updated.isoformat(),
            "expiry_time": oco_order.expiry_time.isoformat() if oco_order.expiry_time else None
        }

# Global OCO manager instance
oco_manager = OCOOrderManager()

# Convenience functions for easy integration
async def place_bracket_oco_order(symbol: str, quantity: float, side: str,
                                take_profit_percent: Optional[float] = None,
                                stop_loss_percent: Optional[float] = None,
                                reasoning: str = "") -> Optional[OCOOrder]:
    """Place a bracket OCO order."""
    return await oco_manager.place_bracket_oco(
        symbol, quantity, side, take_profit_percent, stop_loss_percent, reasoning
    )

async def place_breakout_oco_order(symbol: str, quantity: float,
                                 upper_breakout_percent: Optional[float] = None,
                                 lower_breakout_percent: Optional[float] = None,
                                 reasoning: str = "") -> Optional[OCOOrder]:
    """Place a breakout OCO order."""
    return await oco_manager.place_breakout_oco(
        symbol, quantity, upper_breakout_percent, lower_breakout_percent, reasoning
    )

async def get_oco_summary() -> Dict:
    """Get summary of all active OCO orders."""
    return oco_manager.get_active_oco_summary()

async def start_oco_monitoring():
    """Start OCO monitoring services."""
    await oco_manager.start_monitoring()

async def stop_oco_monitoring():
    """Stop OCO monitoring services."""
    await oco_manager.stop_monitoring()

__all__ = [
    'OCOType', 'OCOStatus', 'OCOConfiguration', 'OCOOrder', 'OCOOrderManager',
    'oco_manager', 'place_bracket_oco_order', 'place_breakout_oco_order',
    'get_oco_summary', 'start_oco_monitoring', 'stop_oco_monitoring'
]