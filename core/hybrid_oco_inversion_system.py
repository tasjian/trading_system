#!/usr/bin/env python3
"""
Hybrid OCO-Inversion System

Combines OCO (One-Cancels-Other) orders with loss-based position inversion
for a layered risk management approach:

Layer 1: Normal trading with wide OCO stops
Layer 2: Loss inversion for recovery attempts  
Layer 3: Emergency exit if recovery fails
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from tools.alpaca_client import alpaca_client
from core.oco_order_manager import oco_manager
from core.position_tracker import position_tracker

logger = logging.getLogger(__name__)

class HybridPositionState(Enum):
    """States for hybrid position management."""
    NORMAL = "normal"                    # Standard position with OCO
    INVERSION_TRIGGERED = "inversion"    # Position inverted, monitoring recovery
    EMERGENCY_EXIT = "emergency"         # Final stop triggered
    RECOVERY_SUCCESS = "success"         # Successfully recovered
    
class HybridExitReason(Enum):
    """Reasons for position exit."""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    INVERSION_RECOVERY = "inversion_recovery"
    EMERGENCY_STOP = "emergency_stop"
    TIME_LIMIT = "time_limit"

@dataclass
class HybridPosition:
    """Tracks a hybrid position with both OCO and inversion components."""
    symbol: str
    original_side: str  # 'long' or 'short'
    original_quantity: float
    original_entry_price: float
    original_entry_time: datetime
    
    # Current state
    state: HybridPositionState
    current_unrealized_pl: float = 0.0
    
    # OCO configuration
    oco_take_profit_price: Optional[float] = None
    oco_stop_loss_price: Optional[float] = None
    oco_order_id: Optional[str] = None
    
    # Inversion tracking
    inversion_triggered_at: Optional[datetime] = None
    inversion_side: Optional[str] = None
    inversion_quantity: Optional[float] = None
    inversion_entry_price: Optional[float] = None
    combined_pl: float = 0.0
    
    # Risk limits
    inversion_trigger_loss: float = -5.0      # Trigger inversion at -$5
    emergency_exit_loss: float = -20.0        # Emergency exit at -$20 total
    recovery_target_profit: float = 2.0       # Close at +$2 combined profit
    max_position_time_minutes: int = 60       # Max time before forced exit

class HybridOCOInversionSystem:
    """
    Hybrid system combining OCO orders with loss-based position inversion.
    
    Strategy Flow:
    1. Enter position with wide OCO stops (-$25)
    2. If -$5 loss: trigger inversion (opposite direction, 75% size)
    3. Monitor combined P&L for recovery (+$2) or emergency exit (-$20)
    4. Time-based exit after 60 minutes if no resolution
    """
    
    def __init__(self):
        """Initialize hybrid system."""
        self.active_positions: Dict[str, HybridPosition] = {}
        self.enabled = True
        self.dry_run = True  # Start in dry run mode
        
        # Configuration
        self.inversion_size_ratio = 0.75  # Invert with 75% of original size
        self.max_concurrent_hybrids = 5   # Limit concurrent hybrid positions
        
        logger.info("🔄 Hybrid OCO-Inversion System initialized")
        logger.info(f"   Inversion trigger: -$5 loss")
        logger.info(f"   Emergency exit: -$20 total loss") 
        logger.info(f"   Recovery target: +$2 profit")
        logger.info(f"   Inversion size: {self.inversion_size_ratio:.0%} of original")
        logger.info(f"   Max position time: 60 minutes")
    
    async def create_hybrid_position(self, 
                                   symbol: str,
                                   side: str,
                                   quantity: float,
                                   entry_price: float) -> bool:
        """Create a new hybrid position with OCO orders."""
        
        if not self.enabled:
            return False
            
        if len(self.active_positions) >= self.max_concurrent_hybrids:
            logger.warning(f"Max concurrent hybrid positions reached ({self.max_concurrent_hybrids})")
            return False
            
        try:
            # Calculate OCO prices
            if side == 'long':
                take_profit = entry_price * 1.10  # 10% profit target
                stop_loss = entry_price * 0.85    # 15% stop loss (wide)
            else:  # short
                take_profit = entry_price * 0.90  # 10% profit target  
                stop_loss = entry_price * 1.15    # 15% stop loss (wide)
            
            # Create hybrid position tracking
            hybrid_pos = HybridPosition(
                symbol=symbol,
                original_side=side,
                original_quantity=quantity,
                original_entry_price=entry_price,
                original_entry_time=datetime.now(),
                state=HybridPositionState.NORMAL,
                oco_take_profit_price=take_profit,
                oco_stop_loss_price=stop_loss
            )
            
            # Place OCO orders (if not in dry run)
            if not self.dry_run:
                oco_order_id = await self._place_oco_orders(hybrid_pos)
                hybrid_pos.oco_order_id = oco_order_id
            
            self.active_positions[symbol] = hybrid_pos
            
            logger.info(f"✅ Created hybrid position: {symbol} {side} {quantity} @ ${entry_price:.2f}")
            logger.info(f"   Take Profit: ${take_profit:.2f}, Stop Loss: ${stop_loss:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create hybrid position for {symbol}: {e}")
            return False
    
    async def monitor_hybrid_positions(self) -> int:
        """Monitor all active hybrid positions and execute state transitions."""
        
        if not self.active_positions:
            return 0
            
        actions_taken = 0
        
        for symbol, position in list(self.active_positions.items()):
            try:
                action_taken = await self._monitor_single_position(position)
                if action_taken:
                    actions_taken += 1
                    
            except Exception as e:
                logger.error(f"Error monitoring hybrid position {symbol}: {e}")
        
        return actions_taken
    
    async def _monitor_single_position(self, position: HybridPosition) -> bool:
        """Monitor a single hybrid position and execute state transitions."""
        
        # Get current market data
        current_price = await self._get_current_price(position.symbol)
        if not current_price:
            return False
            
        # Update unrealized P&L
        position.current_unrealized_pl = self._calculate_unrealized_pl(position, current_price)
        
        # Check for state transitions
        if position.state == HybridPositionState.NORMAL:
            return await self._handle_normal_state(position, current_price)
            
        elif position.state == HybridPositionState.INVERSION_TRIGGERED:
            return await self._handle_inversion_state(position, current_price)
            
        elif position.state in [HybridPositionState.EMERGENCY_EXIT, HybridPositionState.RECOVERY_SUCCESS]:
            # Position should be cleaned up
            await self._cleanup_position(position.symbol)
            return True
            
        return False
    
    async def _handle_normal_state(self, position: HybridPosition, current_price: float) -> bool:
        """Handle position in normal state - check for inversion trigger."""
        
        # Check time limit
        if self._is_time_expired(position):
            await self._force_exit_position(position, HybridExitReason.TIME_LIMIT)
            return True
            
        # Check for inversion trigger
        if position.current_unrealized_pl <= position.inversion_trigger_loss:
            logger.info(f"🔄 Triggering inversion for {position.symbol}: ${position.current_unrealized_pl:.2f} loss")
            return await self._trigger_inversion(position, current_price)
            
        return False
    
    async def _handle_inversion_state(self, position: HybridPosition, current_price: float) -> bool:
        """Handle position in inversion state - monitor for recovery or emergency exit."""
        
        # Calculate combined P&L (original loss + inversion profit/loss)
        position.combined_pl = self._calculate_combined_pl(position, current_price)
        
        # Check for recovery success
        if position.combined_pl >= position.recovery_target_profit:
            logger.info(f"✅ Recovery success for {position.symbol}: ${position.combined_pl:.2f} combined profit")
            await self._complete_recovery(position, HybridExitReason.INVERSION_RECOVERY)
            return True
            
        # Check for emergency exit
        if position.combined_pl <= position.emergency_exit_loss:
            logger.warning(f"🚨 Emergency exit for {position.symbol}: ${position.combined_pl:.2f} combined loss")
            await self._emergency_exit(position, HybridExitReason.EMERGENCY_STOP)
            return True
            
        # Check time limit
        if self._is_time_expired(position):
            await self._force_exit_position(position, HybridExitReason.TIME_LIMIT)
            return True
            
        return False
    
    async def _trigger_inversion(self, position: HybridPosition, current_price: float) -> bool:
        """Trigger position inversion."""
        
        try:
            # Cancel existing OCO orders
            if position.oco_order_id and not self.dry_run:
                await self._cancel_oco_orders(position.oco_order_id)
            
            # Calculate inversion parameters
            inversion_side = 'short' if position.original_side == 'long' else 'long'
            inversion_quantity = position.original_quantity * self.inversion_size_ratio
            
            if self.dry_run:
                logger.info(f"🧪 DRY RUN: Would trigger inversion for {position.symbol}")
                logger.info(f"   Original: {position.original_side} {position.original_quantity}")
                logger.info(f"   Inversion: {inversion_side} {inversion_quantity}")
            else:
                # Execute inversion orders
                # 1. Close original position
                close_success = await self._close_position(
                    position.symbol, position.original_quantity, position.original_side
                )
                if not close_success:
                    logger.error(f"Failed to close original position for {position.symbol}")
                    return False
                
                # 2. Open inverse position  
                open_success = await self._open_position(
                    position.symbol, inversion_quantity, inversion_side
                )
                if not open_success:
                    logger.error(f"Failed to open inverse position for {position.symbol}")
                    return False
            
            # Update position state
            position.state = HybridPositionState.INVERSION_TRIGGERED
            position.inversion_triggered_at = datetime.now()
            position.inversion_side = inversion_side
            position.inversion_quantity = inversion_quantity
            position.inversion_entry_price = current_price
            
            logger.info(f"🔄 Inversion triggered for {position.symbol}")
            logger.info(f"   Loss at trigger: ${position.current_unrealized_pl:.2f}")
            logger.info(f"   Inverted to: {inversion_side} {inversion_quantity} @ ${current_price:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to trigger inversion for {position.symbol}: {e}")
            return False
    
    async def _complete_recovery(self, position: HybridPosition, reason: HybridExitReason):
        """Complete successful recovery and close positions."""
        
        try:
            if not self.dry_run:
                # Close the inverted position (keep the profit)
                await self._close_position(
                    position.symbol, position.inversion_quantity, position.inversion_side
                )
            
            position.state = HybridPositionState.RECOVERY_SUCCESS
            
            logger.info(f"✅ Recovery completed for {position.symbol}")
            logger.info(f"   Combined P&L: ${position.combined_pl:.2f}")
            logger.info(f"   Recovery time: {self._get_position_duration(position):.1f} minutes")
            
            # Record successful recovery
            await self._record_hybrid_result(position, reason, success=True)
            
        except Exception as e:
            logger.error(f"Error completing recovery for {position.symbol}: {e}")
    
    async def _emergency_exit(self, position: HybridPosition, reason: HybridExitReason):
        """Execute emergency exit of all position components."""
        
        try:
            if not self.dry_run:
                # Close all positions
                if position.inversion_quantity:
                    await self._close_position(
                        position.symbol, position.inversion_quantity, position.inversion_side
                    )
            
            position.state = HybridPositionState.EMERGENCY_EXIT
            
            logger.warning(f"🚨 Emergency exit completed for {position.symbol}")
            logger.warning(f"   Total loss: ${position.combined_pl:.2f}")
            logger.warning(f"   Position duration: {self._get_position_duration(position):.1f} minutes")
            
            # Record failed recovery
            await self._record_hybrid_result(position, reason, success=False)
            
        except Exception as e:
            logger.error(f"Error during emergency exit for {position.symbol}: {e}")
    
    async def _force_exit_position(self, position: HybridPosition, reason: HybridExitReason):
        """Force exit position due to time limit."""
        
        try:
            if not self.dry_run:
                if position.state == HybridPositionState.INVERSION_TRIGGERED and position.inversion_quantity:
                    await self._close_position(
                        position.symbol, position.inversion_quantity, position.inversion_side
                    )
            
            logger.info(f"⏰ Time-based exit for {position.symbol}")
            logger.info(f"   Final P&L: ${position.combined_pl:.2f}")
            
            await self._record_hybrid_result(position, reason, success=position.combined_pl > 0)
            await self._cleanup_position(position.symbol)
            
        except Exception as e:
            logger.error(f"Error during forced exit for {position.symbol}: {e}")
    
    def _calculate_unrealized_pl(self, position: HybridPosition, current_price: float) -> float:
        """Calculate unrealized P&L for original position."""
        if position.original_side == 'long':
            return (current_price - position.original_entry_price) * position.original_quantity
        else:  # short
            return (position.original_entry_price - current_price) * position.original_quantity
    
    def _calculate_combined_pl(self, position: HybridPosition, current_price: float) -> float:
        """Calculate combined P&L of original loss plus inversion profit/loss."""
        # Original loss (locked in when inverted)
        original_loss = position.inversion_trigger_loss
        
        # Current inversion P&L
        if position.inversion_side and position.inversion_entry_price and position.inversion_quantity:
            if position.inversion_side == 'long':
                inversion_pl = (current_price - position.inversion_entry_price) * position.inversion_quantity
            else:  # short
                inversion_pl = (position.inversion_entry_price - current_price) * position.inversion_quantity
        else:
            inversion_pl = 0
            
        return original_loss + inversion_pl
    
    def _is_time_expired(self, position: HybridPosition) -> bool:
        """Check if position has exceeded maximum time limit."""
        duration_minutes = (datetime.now() - position.original_entry_time).total_seconds() / 60
        return duration_minutes >= position.max_position_time_minutes
    
    def _get_position_duration(self, position: HybridPosition) -> float:
        """Get position duration in minutes."""
        return (datetime.now() - position.original_entry_time).total_seconds() / 60
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol."""
        try:
            return alpaca_client.get_current_price(symbol)
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            return None
    
    async def _place_oco_orders(self, position: HybridPosition) -> Optional[str]:
        """Place OCO orders for position."""
        # Placeholder - integrate with actual OCO system
        return f"oco_{position.symbol}_{int(datetime.now().timestamp())}"
    
    async def _cancel_oco_orders(self, order_id: str):
        """Cancel OCO orders."""
        # Placeholder - integrate with actual OCO system
        logger.info(f"Cancelled OCO orders: {order_id}")
    
    async def _close_position(self, symbol: str, quantity: float, side: str) -> bool:
        """Close a position."""
        try:
            close_side = "sell" if side == "long" else "buy_to_cover"
            return alpaca_client.place_order(
                symbol=symbol,
                qty=quantity,
                side=close_side,
                order_type="market",
                time_in_force="day"
            )
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            return False
    
    async def _open_position(self, symbol: str, quantity: float, side: str) -> bool:
        """Open a position."""
        try:
            order_side = "buy" if side == "long" else "sell_short"
            return alpaca_client.place_order(
                symbol=symbol,
                qty=quantity,
                side=order_side,
                order_type="market",
                time_in_force="day"
            )
        except Exception as e:
            logger.error(f"Failed to open position {symbol}: {e}")
            return False
    
    async def _record_hybrid_result(self, position: HybridPosition, reason: HybridExitReason, success: bool):
        """Record hybrid position result for analysis."""
        result = {
            "timestamp": datetime.now().isoformat(),
            "symbol": position.symbol,
            "original_side": position.original_side,
            "original_quantity": position.original_quantity,
            "inversion_triggered": position.state != HybridPositionState.NORMAL,
            "final_pl": position.combined_pl,
            "exit_reason": reason.value,
            "success": success,
            "duration_minutes": self._get_position_duration(position)
        }
        
        logger.info(f"📊 Hybrid result recorded: {position.symbol} - {'SUCCESS' if success else 'FAILURE'}")
        # TODO: Save to database or file for analysis
    
    async def _cleanup_position(self, symbol: str):
        """Remove position from active tracking."""
        if symbol in self.active_positions:
            del self.active_positions[symbol]
            logger.debug(f"Cleaned up hybrid position: {symbol}")
    
    def get_hybrid_stats(self) -> Dict:
        """Get statistics for hybrid system."""
        active_count = len(self.active_positions)
        
        states = {}
        for pos in self.active_positions.values():
            state = pos.state.value
            states[state] = states.get(state, 0) + 1
        
        return {
            "enabled": self.enabled,
            "dry_run": self.dry_run,
            "active_positions": active_count,
            "max_concurrent": self.max_concurrent_hybrids,
            "position_states": states,
            "inversion_size_ratio": self.inversion_size_ratio
        }

# Global instance
hybrid_oco_inversion_system = HybridOCOInversionSystem()

__all__ = ['HybridOCOInversionSystem', 'HybridPosition', 'HybridPositionState', 'hybrid_oco_inversion_system']