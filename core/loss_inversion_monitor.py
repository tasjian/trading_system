#!/usr/bin/env python3
"""
Loss-Based Position Inversion Monitor

Monitors positions for losses exceeding $5 and triggers position inversions
(long -> short or short -> long) to potentially recover losses.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from tools.alpaca_client import alpaca_client
from core.position_tracker import position_tracker
from core.portfolio_balancer import portfolio_balancer, PositionAction
from config.settings import settings

logger = logging.getLogger(__name__)

class InversionReason(Enum):
    """Reasons for position inversion."""
    LOSS_THRESHOLD = "loss_threshold"
    MANUAL_TRIGGER = "manual_trigger"
    RISK_MANAGEMENT = "risk_management"

@dataclass
class InversionCandidate:
    """A position that is a candidate for inversion."""
    symbol: str
    current_side: str  # 'long' or 'short'
    quantity: float
    unrealized_pl: float
    unrealized_pl_percent: float
    current_price: float
    market_value: float
    cost_basis: float
    reason: InversionReason
    confidence: float = 0.5
    
class LossInversionMonitor:
    """
    Monitors positions for significant losses and triggers inversions when thresholds are exceeded.
    """
    
    def __init__(self, 
                 loss_threshold: float = 5.0,
                 min_position_value: float = 100.0,
                 cooldown_minutes: int = 30,
                 max_inversions_per_day: int = 10):
        """Initialize loss inversion monitor.
        
        Args:
            loss_threshold: Dollar loss amount that triggers inversion (default: $5)
            min_position_value: Minimum position value to consider for inversion
            cooldown_minutes: Minutes to wait before allowing another inversion of same symbol
            max_inversions_per_day: Maximum inversions per day for risk management
        """
        self.loss_threshold = loss_threshold
        self.min_position_value = min_position_value
        self.cooldown_period = timedelta(minutes=cooldown_minutes)
        self.max_inversions_per_day = max_inversions_per_day
        
        # Tracking data
        self.last_inversion_time: Dict[str, datetime] = {}
        self.daily_inversion_count = 0
        self.last_reset_date = datetime.now().date()
        self.inversion_history: List[Dict] = []
        
        # Configuration
        self.enabled = True
        self.dry_run = False  # Set to True for testing without actual trades
        
        logger.info(f"🔄 Loss Inversion Monitor initialized")
        logger.info(f"   Loss threshold: ${loss_threshold}")
        logger.info(f"   Min position value: ${min_position_value}")
        logger.info(f"   Cooldown period: {cooldown_minutes} minutes")
        logger.info(f"   Max inversions per day: {max_inversions_per_day}")
    
    async def scan_for_inversion_candidates(self) -> List[InversionCandidate]:
        """Scan all positions for inversion candidates based on loss threshold."""
        if not self.enabled:
            return []
        
        # Reset daily counter if new day
        self._reset_daily_counter_if_needed()
        
        # Check if we've hit daily limit
        if self.daily_inversion_count >= self.max_inversions_per_day:
            logger.info(f"🚫 Daily inversion limit reached ({self.max_inversions_per_day})")
            return []
        
        try:
            # Get current positions from Alpaca
            positions = alpaca_client.get_positions()
            candidates = []
            
            logger.debug(f"📊 Scanning {len(positions)} positions for inversion candidates...")
            
            for pos in positions:
                candidate = await self._evaluate_position_for_inversion(pos)
                if candidate:
                    candidates.append(candidate)
            
            if candidates:
                logger.info(f"🎯 Found {len(candidates)} inversion candidates")
                for candidate in candidates:
                    logger.info(f"   {candidate.symbol}: {candidate.current_side} position, "
                              f"${candidate.unrealized_pl:.2f} loss ({candidate.unrealized_pl_percent:.1f}%)")
            
            return candidates
            
        except Exception as e:
            logger.error(f"Failed to scan for inversion candidates: {e}")
            return []
    
    async def _evaluate_position_for_inversion(self, position: Dict) -> Optional[InversionCandidate]:
        """Evaluate a single position for inversion eligibility."""
        symbol = position['symbol']
        unrealized_pl = float(position['unrealized_pl'])
        market_value = abs(float(position['market_value']))
        
        # Skip if loss is not significant enough
        if unrealized_pl >= -self.loss_threshold:
            return None
        
        # Skip small positions
        if market_value < self.min_position_value:
            logger.debug(f"⏭️ Skipping {symbol}: position too small (${market_value:.2f})")
            return None
        
        # Check cooldown period
        if self._is_in_cooldown(symbol):
            logger.debug(f"⏭️ Skipping {symbol}: still in cooldown period")
            return None
        
        # Check if position tracking allows exit
        can_exit, reason = position_tracker.can_exit_position(symbol, position['current_price'])
        if not can_exit:
            logger.debug(f"⏭️ Skipping {symbol}: {reason}")
            return None
        
        # Create inversion candidate
        candidate = InversionCandidate(
            symbol=symbol,
            current_side=position['side'],
            quantity=abs(float(position['qty'])),
            unrealized_pl=unrealized_pl,
            unrealized_pl_percent=float(position['unrealized_plpc']) * 100,
            current_price=float(position['current_price']),
            market_value=market_value,
            cost_basis=float(position['cost_basis']),
            reason=InversionReason.LOSS_THRESHOLD,
            confidence=self._calculate_inversion_confidence(position)
        )
        
        return candidate
    
    def _calculate_inversion_confidence(self, position: Dict) -> float:
        """Calculate confidence level for position inversion."""
        # Base confidence on loss magnitude and position characteristics
        unrealized_pl = float(position['unrealized_pl'])
        unrealized_pl_percent = float(position['unrealized_plpc'])
        
        # Higher losses increase confidence in inversion
        loss_magnitude_factor = min(abs(unrealized_pl) / (self.loss_threshold * 3), 1.0)
        
        # Percentage loss factor
        percent_loss_factor = min(abs(unrealized_pl_percent) / 0.1, 1.0)  # 10% max
        
        # Combine factors
        confidence = (loss_magnitude_factor * 0.6) + (percent_loss_factor * 0.4)
        
        # Ensure reasonable bounds
        return max(0.3, min(0.9, confidence))
    
    def _is_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown period after recent inversion."""
        if symbol not in self.last_inversion_time:
            return False
        
        time_since_last = datetime.now() - self.last_inversion_time[symbol]
        return time_since_last < self.cooldown_period
    
    def _reset_daily_counter_if_needed(self):
        """Reset daily inversion counter if it's a new day."""
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.daily_inversion_count = 0
            self.last_reset_date = today
            logger.info(f"📅 Reset daily inversion counter for {today}")
    
    async def execute_position_inversion(self, candidate: InversionCandidate) -> bool:
        """Execute position inversion for a candidate.
        
        Returns:
            True if inversion was successful, False otherwise
        """
        if self.dry_run:
            logger.info(f"🧪 DRY RUN: Would invert {candidate.symbol} position "
                       f"({candidate.current_side} -> {'short' if candidate.current_side == 'long' else 'long'})")
            return True
        
        try:
            logger.info(f"🔄 Executing position inversion for {candidate.symbol}")
            logger.info(f"   Current: {candidate.current_side} {candidate.quantity} shares")
            logger.info(f"   Loss: ${candidate.unrealized_pl:.2f} ({candidate.unrealized_pl_percent:.1f}%)")
            logger.info(f"   Reason: {candidate.reason.value}")
            
            # Step 1: Close current position
            close_success = await self._close_current_position(candidate)
            if not close_success:
                logger.error(f"Failed to close current position for {candidate.symbol}")
                return False
            
            # Step 2: Open inverse position
            inverse_success = await self._open_inverse_position(candidate)
            if not inverse_success:
                logger.error(f"Failed to open inverse position for {candidate.symbol}")
                return False
            
            # Step 3: Record inversion
            self._record_inversion(candidate)
            
            logger.info(f"✅ Successfully inverted {candidate.symbol} position")
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute inversion for {candidate.symbol}: {e}")
            return False
    
    async def _close_current_position(self, candidate: InversionCandidate) -> bool:
        """Close the current position."""
        try:
            # Determine close action based on current side
            if candidate.current_side == 'long':
                action = PositionAction.SELL
            else:  # short
                action = PositionAction.COVER
            
            # Execute close order through alpaca client
            success = alpaca_client.place_order(
                symbol=candidate.symbol,
                qty=candidate.quantity,
                side=action.value,
                order_type="market",
                time_in_force="day"
            )
            
            if success:
                # Record position exit in tracker
                position_tracker.record_position_exit(
                    candidate.symbol, 
                    candidate.current_price, 
                    "loss_inversion"
                )
                logger.info(f"📉 Closed {candidate.current_side} position for {candidate.symbol}")
                return True
            else:
                logger.error(f"Failed to place close order for {candidate.symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position for {candidate.symbol}: {e}")
            return False
    
    async def _open_inverse_position(self, candidate: InversionCandidate) -> bool:
        """Open the inverse position."""
        try:
            # Determine inverse action
            if candidate.current_side == 'long':
                inverse_action = PositionAction.SHORT
                side = "sell_short"
            else:  # short
                inverse_action = PositionAction.BUY
                side = "buy"
            
            # Execute inverse order
            success = alpaca_client.place_order(
                symbol=candidate.symbol,
                qty=candidate.quantity,
                side=side,
                order_type="market",
                time_in_force="day"
            )
            
            if success:
                # Record new position entry in tracker
                new_side = "short" if candidate.current_side == "long" else "long"
                position_tracker.record_position_entry(
                    candidate.symbol,
                    candidate.current_price,
                    candidate.quantity,
                    new_side,
                    candidate.confidence
                )
                logger.info(f"📈 Opened {new_side} position for {candidate.symbol}")
                return True
            else:
                logger.error(f"Failed to place inverse order for {candidate.symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error opening inverse position for {candidate.symbol}: {e}")
            return False
    
    def _record_inversion(self, candidate: InversionCandidate):
        """Record the inversion for tracking and analysis."""
        inversion_record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": candidate.symbol,
            "from_side": candidate.current_side,
            "to_side": "short" if candidate.current_side == "long" else "long",
            "quantity": candidate.quantity,
            "price": candidate.current_price,
            "loss_amount": candidate.unrealized_pl,
            "loss_percent": candidate.unrealized_pl_percent,
            "reason": candidate.reason.value,
            "confidence": candidate.confidence
        }
        
        self.inversion_history.append(inversion_record)
        self.last_inversion_time[candidate.symbol] = datetime.now()
        self.daily_inversion_count += 1
        
        logger.info(f"📝 Recorded inversion #{self.daily_inversion_count} for {candidate.symbol}")
    
    async def monitor_and_execute_inversions(self) -> int:
        """Main monitoring loop - scan for candidates and execute inversions.
        
        Returns:
            Number of inversions executed
        """
        logger.debug("🔍 Monitoring positions for loss-based inversions...")
        
        candidates = await self.scan_for_inversion_candidates()
        if not candidates:
            return 0
        
        inversions_executed = 0
        
        for candidate in candidates:
            # Execute inversion
            success = await self.execute_position_inversion(candidate)
            if success:
                inversions_executed += 1
            
            # Small delay between inversions
            await asyncio.sleep(1)
        
        if inversions_executed > 0:
            logger.info(f"🎯 Executed {inversions_executed} position inversions")
        
        return inversions_executed
    
    def get_inversion_stats(self) -> Dict:
        """Get statistics about inversions."""
        today = datetime.now().date()
        recent_inversions = [
            inv for inv in self.inversion_history 
            if datetime.fromisoformat(inv['timestamp']).date() >= today - timedelta(days=7)
        ]
        
        return {
            "enabled": self.enabled,
            "dry_run": self.dry_run,
            "loss_threshold": self.loss_threshold,
            "daily_count": self.daily_inversion_count,
            "daily_limit": self.max_inversions_per_day,
            "total_inversions_today": len([inv for inv in self.inversion_history 
                                         if datetime.fromisoformat(inv['timestamp']).date() == today]),
            "recent_inversions_7d": len(recent_inversions),
            "average_loss_threshold": sum(abs(inv['loss_amount']) for inv in recent_inversions) / max(len(recent_inversions), 1),
            "symbols_in_cooldown": len(self.last_inversion_time)
        }

# Global instance
loss_inversion_monitor = LossInversionMonitor()

__all__ = ['LossInversionMonitor', 'InversionCandidate', 'InversionReason', 'loss_inversion_monitor']