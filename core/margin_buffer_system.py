#!/usr/bin/env python3
"""
Margin Buffer System - Ensures trading system never stops due to insufficient buying power
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from tools.alpaca_client import alpaca_client

logger = logging.getLogger(__name__)

class MarginBufferSystem:
    """
    Intelligent margin management system that:
    1. Monitors available buying power
    2. Automatically closes profitable positions when power is low
    3. Prevents system shutdown due to margin constraints
    4. Maintains minimum operational buffer
    """
    
    def __init__(self, min_buying_power: float = 2000.0):
        """
        Initialize margin buffer system.
        
        Args:
            min_buying_power: Minimum buying power to maintain
        """
        self.min_buying_power = min_buying_power
        self.last_margin_check = None
        self.positions_closed_today = []
        
    async def check_and_manage_margin(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check margin status and manage positions if necessary.
        
        Returns:
            Updated state with margin status and any actions taken
        """
        try:
            # Get current account status
            account_info = alpaca_client.get_account_info()
            current_buying_power = float(account_info.get('buying_power', 0))
            current_cash = float(account_info.get('cash', 0))
            
            logger.info(f"💰 Margin Check - Buying Power: ${current_buying_power:.2f}, Cash: ${current_cash:.2f}")
            
            # Update state with current margin status
            state["margin_status"] = {
                "buying_power": current_buying_power,
                "cash": current_cash,
                "min_threshold": self.min_buying_power,
                "status": "HEALTHY" if current_buying_power >= self.min_buying_power else "LIMITED"
            }
            
            # If buying power is below threshold, attempt to free up margin
            if current_buying_power < self.min_buying_power:
                logger.warning(f"⚠️ Low buying power detected: ${current_buying_power:.2f} < ${self.min_buying_power:.2f}")
                await self._free_up_margin(state)
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Error in margin check: {e}")
            return state
    
    async def _free_up_margin(self, state: Dict[str, Any]) -> None:
        """
        Free up margin by closing profitable short positions.
        """
        try:
            logger.info("🔧 Attempting to free up margin by closing profitable positions...")
            
            # Get current positions
            positions = alpaca_client.api.list_positions()
            short_positions = [p for p in positions if float(p.qty) < 0]
            
            if not short_positions:
                logger.info("No short positions to close")
                return
            
            # Find profitable shorts to close
            profitable_shorts = []
            for pos in short_positions:
                pnl = float(pos.unrealized_pl)
                if pnl > 5:  # Only close if profit > $5
                    profitable_shorts.append({
                        'symbol': pos.symbol,
                        'qty': abs(float(pos.qty)),
                        'pnl': pnl,
                        'market_value': abs(float(pos.market_value))
                    })
            
            if not profitable_shorts:
                logger.info("No profitable short positions to close")
                return
            
            # Sort by profit descending and close top performers
            profitable_shorts.sort(key=lambda x: x['pnl'], reverse=True)
            
            # Close positions until we have adequate margin
            target_margin_to_free = self.min_buying_power * 2  # Target 2x minimum
            margin_freed = 0
            closed_count = 0
            
            for pos in profitable_shorts:
                if margin_freed >= target_margin_to_free:
                    break
                
                try:
                    symbol = pos['symbol']
                    qty = int(pos['qty'])
                    
                    # Check if we already closed this today
                    if symbol in self.positions_closed_today:
                        continue
                    
                    logger.info(f"Closing profitable short: {symbol} {qty} shares (P/L: ${pos['pnl']:+.2f})")
                    
                    # Submit buy order to close short position
                    order = alpaca_client.api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
                    
                    logger.info(f"✅ Margin-freeing order submitted: {order.id}")
                    
                    # Track the closure
                    self.positions_closed_today.append(symbol)
                    closed_count += 1
                    margin_freed += pos['market_value']
                    
                    # Update state with the action
                    if "margin_actions" not in state:
                        state["margin_actions"] = []
                    
                    state["margin_actions"].append({
                        "action": "close_profitable_short",
                        "symbol": symbol,
                        "quantity": qty,
                        "expected_pnl": pos['pnl'],
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Brief pause between orders
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"❌ Error closing {symbol}: {e}")
                    continue
            
            logger.info(f"🏁 Margin management complete: {closed_count} positions closed, ${margin_freed:.2f} margin freed")
            
        except Exception as e:
            logger.error(f"❌ Error in margin management: {e}")
    
    def should_enter_monitoring_mode(self, buying_power: float) -> bool:
        """
        Determine if system should enter monitoring-only mode.
        """
        return buying_power < (self.min_buying_power * 0.5)  # Half of minimum threshold
    
    def reset_daily_tracking(self):
        """
        Reset daily tracking (call at market open).
        """
        self.positions_closed_today = []
        logger.info("🔄 Reset daily margin management tracking")

# Global instance
margin_buffer_system = MarginBufferSystem()