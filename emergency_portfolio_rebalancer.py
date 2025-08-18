#!/usr/bin/env python3
"""
Emergency Portfolio Rebalancer

Immediately generates SELL orders from existing positions to free up cash
and restore balanced trading capabilities when system has $0 buying power.

This addresses the critical issue where the system only generates BUY orders
but has no cash available to execute them.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from tools.alpaca_client import alpaca_client
from core.portfolio_balancer import portfolio_balancer, PositionAnalysis, RebalanceDecision
from core.order_decision_engine import order_decision_engine, OrderDecision, OrderDecisionType
from agents.state import TradingSignal

logger = logging.getLogger(__name__)

@dataclass
class EmergencyRebalanceResult:
    """Result of emergency rebalancing operation."""
    timestamp: datetime
    positions_analyzed: int
    sell_orders_generated: int
    cash_to_be_freed: float
    portfolio_utilization_before: float
    portfolio_utilization_target: float
    orders_executed: List[Dict]
    success: bool
    error_message: Optional[str] = None

class EmergencyPortfolioRebalancer:
    """Emergency portfolio rebalancer for immediate cash generation."""
    
    def __init__(self):
        """Initialize emergency rebalancer."""
        self.target_cash_ratio = 0.15  # Target 15% cash for trading
        self.min_position_size = 100   # Minimum $100 position size
        self.max_sell_percentage = 0.30  # Max 30% of position per sell
        
    async def emergency_rebalance_for_cash(self, target_cash_amount: float = None) -> EmergencyRebalanceResult:
        """
        Execute emergency rebalancing to generate cash from existing positions.
        
        Args:
            target_cash_amount: Target cash amount to generate (defaults to 15% of portfolio)
            
        Returns:
            EmergencyRebalanceResult with execution details
        """
        start_time = datetime.now()
        logger.info("🚨 EMERGENCY PORTFOLIO REBALANCE: Generating cash from positions")
        
        try:
            # Get current portfolio state
            account_info = alpaca_client.get_account_info()
            current_positions = alpaca_client.get_positions()
            
            portfolio_value = float(account_info['portfolio_value'])
            current_cash = float(account_info['cash'])
            current_utilization = (portfolio_value - current_cash) / portfolio_value if portfolio_value > 0 else 0
            
            logger.info(f"📊 Current Portfolio State:")
            logger.info(f"   Portfolio Value: ${portfolio_value:,.2f}")
            logger.info(f"   Current Cash: ${current_cash:,.2f}")
            logger.info(f"   Utilization: {current_utilization:.1%}")
            logger.info(f"   Positions: {len(current_positions)}")
            
            # Determine target cash amount
            if target_cash_amount is None:
                target_cash_amount = portfolio_value * self.target_cash_ratio
            
            cash_needed = max(0, target_cash_amount - current_cash)
            
            if cash_needed <= 0:
                logger.info("✅ Sufficient cash available, no emergency rebalancing needed")
                return EmergencyRebalanceResult(
                    timestamp=start_time,
                    positions_analyzed=len(current_positions),
                    sell_orders_generated=0,
                    cash_to_be_freed=0,
                    portfolio_utilization_before=current_utilization,
                    portfolio_utilization_target=(portfolio_value - target_cash_amount) / portfolio_value,
                    orders_executed=[],
                    success=True
                )
            
            logger.info(f"💰 Cash needed: ${cash_needed:,.2f}")
            
            # Analyze positions for selling
            sell_candidates = await self._analyze_positions_for_selling(current_positions, cash_needed)
            
            if not sell_candidates:
                logger.warning("⚠️ No suitable positions found for selling")
                return EmergencyRebalanceResult(
                    timestamp=start_time,
                    positions_analyzed=len(current_positions),
                    sell_orders_generated=0,
                    cash_to_be_freed=0,
                    portfolio_utilization_before=current_utilization,
                    portfolio_utilization_target=current_utilization,
                    orders_executed=[],
                    success=False,
                    error_message="No suitable positions available for selling"
                )
            
            # Generate sell orders
            sell_orders = await self._generate_emergency_sell_orders(sell_candidates, cash_needed)
            
            # Execute sell orders
            executed_orders = await self._execute_emergency_orders(sell_orders)
            
            # Calculate results
            total_cash_freed = sum(order.get('estimated_proceeds', 0) for order in executed_orders)
            target_utilization = (portfolio_value - target_cash_amount) / portfolio_value if portfolio_value > 0 else 0
            
            logger.info(f"✅ Emergency rebalancing complete:")
            logger.info(f"   Sell orders generated: {len(sell_orders)}")
            logger.info(f"   Orders executed: {len(executed_orders)}")
            logger.info(f"   Estimated cash freed: ${total_cash_freed:,.2f}")
            
            return EmergencyRebalanceResult(
                timestamp=start_time,
                positions_analyzed=len(current_positions),
                sell_orders_generated=len(sell_orders),
                cash_to_be_freed=total_cash_freed,
                portfolio_utilization_before=current_utilization,
                portfolio_utilization_target=target_utilization,
                orders_executed=executed_orders,
                success=True
            )
            
        except Exception as e:
            error_msg = f"Emergency rebalancing failed: {str(e)}"
            logger.error(error_msg)
            return EmergencyRebalanceResult(
                timestamp=start_time,
                positions_analyzed=0,
                sell_orders_generated=0,
                cash_to_be_freed=0,
                portfolio_utilization_before=0,
                portfolio_utilization_target=0,
                orders_executed=[],
                success=False,
                error_message=error_msg
            )
    
    async def _analyze_positions_for_selling(self, positions: List[Dict], cash_needed: float) -> List[Dict]:
        """Analyze positions to determine best candidates for selling."""
        sell_candidates = []
        
        for pos in positions:
            try:
                symbol = pos['symbol']
                quantity = float(pos['qty'])
                market_value = abs(float(pos['market_value']))
                unrealized_pl = float(pos.get('unrealized_pl', 0))
                unrealized_plpc = float(pos.get('unrealized_plpc', 0))
                
                # Skip small positions
                if market_value < self.min_position_size:
                    continue
                
                # Skip short positions (negative quantity)
                if quantity < 0:
                    continue
                
                # Calculate selling priority
                priority_score = self._calculate_sell_priority(
                    market_value, unrealized_pl, unrealized_plpc
                )
                
                # Calculate optimal sell quantity (partial position)
                sell_percentage = min(self.max_sell_percentage, cash_needed / market_value)
                sell_quantity = quantity * sell_percentage
                estimated_proceeds = market_value * sell_percentage
                
                candidate = {
                    'symbol': symbol,
                    'current_quantity': quantity,
                    'market_value': market_value,
                    'sell_quantity': sell_quantity,
                    'sell_percentage': sell_percentage,
                    'estimated_proceeds': estimated_proceeds,
                    'unrealized_pl': unrealized_pl,
                    'unrealized_plpc': unrealized_plpc,
                    'priority_score': priority_score,
                    'reasoning': self._generate_sell_reasoning(symbol, unrealized_plpc, sell_percentage)
                }
                
                sell_candidates.append(candidate)
                
            except Exception as e:
                logger.warning(f"Failed to analyze position {pos.get('symbol', 'unknown')}: {e}")
                continue
        
        # Sort by priority (higher score = better candidate)
        sell_candidates.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return sell_candidates
    
    def _calculate_sell_priority(self, market_value: float, unrealized_pl: float, unrealized_plpc: float) -> float:
        """Calculate priority score for selling a position."""
        score = 0.0
        
        # Size factor: larger positions get higher priority for partial selling
        size_factor = min(market_value / 5000, 2.0)  # Cap at 2x for $5000+ positions
        score += size_factor * 30
        
        # Profit factor: profitable positions get higher priority
        if unrealized_pl > 0:
            profit_factor = min(unrealized_plpc * 100, 50)  # Cap at 50 points
            score += profit_factor
        else:
            # Loss factor: small losses are okay to sell, large losses get lower priority
            loss_factor = max(-20, unrealized_plpc * 100)  # Cap penalty at -20 points
            score += loss_factor
        
        # Prefer positions with moderate gains (take profits)
        if 0.05 <= unrealized_plpc <= 0.25:  # 5-25% gains
            score += 20  # Bonus for taking reasonable profits
        
        return score
    
    def _generate_sell_reasoning(self, symbol: str, unrealized_plpc: float, sell_percentage: float) -> str:
        """Generate reasoning for sell decision."""
        if unrealized_plpc > 0.15:
            return f"Profit taking: {unrealized_plpc:.1%} gain, selling {sell_percentage:.1%} of position"
        elif unrealized_plpc > 0.05:
            return f"Moderate profit taking: {unrealized_plpc:.1%} gain, partial position sale for cash"
        elif unrealized_plpc > -0.05:
            return f"Portfolio rebalancing: break-even position, partial sale for liquidity"
        else:
            return f"Risk management: {unrealized_plpc:.1%} loss, reducing position size"
    
    async def _generate_emergency_sell_orders(self, candidates: List[Dict], cash_needed: float) -> List[Dict]:
        """Generate specific sell orders from candidates."""
        orders = []
        total_proceeds = 0.0
        
        for candidate in candidates:
            if total_proceeds >= cash_needed:
                break
            
            try:
                symbol = candidate['symbol']
                sell_quantity = candidate['sell_quantity']
                estimated_proceeds = candidate['estimated_proceeds']
                
                # Get current price for order execution
                current_price = alpaca_client.get_current_price(symbol)
                if not current_price or current_price <= 0:
                    logger.warning(f"Could not get current price for {symbol}, skipping")
                    continue
                
                # Determine order type based on urgency and market conditions
                order_type = "limit"
                # Use limit order with small discount for quick execution
                limit_price = current_price * 0.995  # 0.5% below market
                
                order = {
                    'symbol': symbol,
                    'side': 'sell',
                    'quantity': sell_quantity,
                    'order_type': order_type,
                    'limit_price': limit_price,
                    'estimated_proceeds': estimated_proceeds,
                    'reasoning': candidate['reasoning'],
                    'priority': 'emergency_cash_generation',
                    'confidence': 0.9  # High confidence for cash generation
                }
                
                orders.append(order)
                total_proceeds += estimated_proceeds
                
                logger.info(f"📋 Sell Order: {symbol} {sell_quantity:.0f} shares @ ${limit_price:.2f} = ${estimated_proceeds:.2f}")
                
            except Exception as e:
                logger.warning(f"Failed to generate sell order for {candidate.get('symbol', 'unknown')}: {e}")
                continue
        
        logger.info(f"💰 Generated {len(orders)} sell orders for ~${total_proceeds:,.2f} in proceeds")
        return orders
    
    async def _execute_emergency_orders(self, orders: List[Dict]) -> List[Dict]:
        """Execute emergency sell orders."""
        executed_orders = []
        
        for order in orders:
            try:
                # Execute order through Alpaca client
                result = alpaca_client.place_order(
                    symbol=order['symbol'],
                    qty=order['quantity'],
                    side=order['side'],
                    order_type=order['order_type'],
                    limit_price=order.get('limit_price'),
                    time_in_force='gtc'  # Good till cancelled
                )
                
                if result:
                    executed_order = {
                        'order_id': result['id'],
                        'symbol': order['symbol'],
                        'side': order['side'],
                        'quantity': order['quantity'],
                        'order_type': order['order_type'],
                        'limit_price': order.get('limit_price'),
                        'estimated_proceeds': order['estimated_proceeds'],
                        'reasoning': order['reasoning'],
                        'status': result['status'],
                        'timestamp': datetime.now()
                    }
                    
                    executed_orders.append(executed_order)
                    logger.info(f"✅ EXECUTED: Sell {order['quantity']:.0f} {order['symbol']} @ ${order.get('limit_price', 0):.2f}")
                else:
                    logger.error(f"❌ FAILED: Could not execute sell order for {order['symbol']}")
                    
            except Exception as e:
                logger.error(f"❌ EXECUTION ERROR: {order['symbol']} - {str(e)}")
                continue
        
        logger.info(f"🎯 Emergency execution complete: {len(executed_orders)}/{len(orders)} orders executed")
        return executed_orders

# Global instance
emergency_rebalancer = EmergencyPortfolioRebalancer()

async def execute_emergency_rebalance(target_cash_amount: float = None) -> EmergencyRebalanceResult:
    """Execute emergency portfolio rebalancing for cash generation."""
    return await emergency_rebalancer.emergency_rebalance_for_cash(target_cash_amount)

if __name__ == "__main__":
    print("🚨 EMERGENCY PORTFOLIO REBALANCER")
    print("=" * 50)
    print("Generating SELL orders from existing positions to free up cash...")
    
    async def main():
        result = await execute_emergency_rebalance()
        
        print(f"\n📊 RESULTS:")
        print(f"Success: {result.success}")
        print(f"Positions Analyzed: {result.positions_analyzed}")
        print(f"Sell Orders Generated: {result.sell_orders_generated}")
        print(f"Cash to be Freed: ${result.cash_to_be_freed:,.2f}")
        print(f"Orders Executed: {len(result.orders_executed)}")
        
        if result.error_message:
            print(f"Error: {result.error_message}")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Emergency rebalancing stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")