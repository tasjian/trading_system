#!/usr/bin/env python3
"""
Options Trading Agent
====================

Advanced options trading agent that integrates with the existing trading system.
Handles options strategies, risk management, and execution with full logging and audit trails.

Key Features:
- OCC symbol construction and validation
- Multiple options strategies (hedging, income, speculation)
- Integration with existing risk management
- Comprehensive logging and audit trails
- Compatible with RL and sentiment analysis systems
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
from config.settings import settings

logger = logging.getLogger(__name__)

class OptionsStrategy(Enum):
    """Options trading strategies supported by the agent."""
    PROTECTIVE_PUT = "protective_put"
    COVERED_CALL = "covered_call"
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    CASH_SECURED_PUT = "cash_secured_put"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    IRON_CONDOR = "iron_condor"
    STRADDLE = "straddle"

class OptionsAction(Enum):
    """High-level options actions for strategy router integration."""
    HEDGE = "hedge"
    YIELD = "yield" 
    SPECULATE = "speculate"
    VOLATILITY = "volatility"
    SPREAD = "spread"

@dataclass
class OptionsOrder:
    """Unified options order representation."""
    symbol: str
    occ_symbol: str
    strategy: OptionsStrategy
    action: OptionsAction
    contracts: int
    side: str  # "buy", "sell"
    premium: Optional[float] = None
    strike: Optional[float] = None
    expiry: Optional[str] = None
    underlying_price: Optional[float] = None
    rationale: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    order_id: Optional[str] = None
    expected_profit: Optional[float] = None
    max_risk: Optional[float] = None

@dataclass
class OptionsPosition:
    """Options position tracking."""
    occ_symbol: str
    underlying: str
    strategy: OptionsStrategy
    contracts: int
    avg_premium: float
    current_premium: float
    strike: float
    expiry: datetime
    pnl: float
    delta: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None

class OptionsTradingAgent:
    """
    Advanced options trading agent with comprehensive strategy support.
    Integrates seamlessly with existing trading infrastructure.
    """
    
    def __init__(self):
        self.positions = {}
        self.order_history = []
        self.strategy_metrics = {}
        
        # Risk parameters
        self.max_options_allocation = getattr(settings, 'max_options_allocation', 0.20)  # 20% max
        self.max_single_position = getattr(settings, 'max_single_options_position', 0.05)  # 5% max
        self.min_days_to_expiry = getattr(settings, 'min_options_dte', 14)  # 14 days minimum
        self.max_days_to_expiry = getattr(settings, 'max_options_dte', 90)  # 90 days maximum
        
        logger.info("Options Trading Agent initialized with risk parameters")
        logger.info(f"   Max allocation: {self.max_options_allocation:.1%}")
        logger.info(f"   Max single position: {self.max_single_position:.1%}")
        logger.info(f"   Days to expiry range: {self.min_days_to_expiry}-{self.max_days_to_expiry}")

    def build_occ_symbol(self, stock: str, expiry: str, strike: float, option_type: str) -> str:
        """
        Build OCC (Options Clearing Corporation) standardized option symbol.
        Format: SYMBOL + YYMMDD + C/P + STRIKE (8 digits, strike*1000)
        
        Example: AAPL250117P00150000 = AAPL Jan 17 2025 $150 Put
        
        Args:
            stock: Underlying stock symbol (e.g., "AAPL")
            expiry: Expiration date in YYMMDD format (e.g., "250117")
            strike: Strike price as float (e.g., 150.0)
            option_type: "C" for call, "P" for put
            
        Returns:
            OCC formatted option symbol
        """
        try:
            # Validate inputs
            if len(expiry) != 6 or not expiry.isdigit():
                raise ValueError(f"Expiry must be YYMMDD format, got: {expiry}")
            
            if option_type.upper() not in ['C', 'P']:
                raise ValueError(f"Option type must be 'C' or 'P', got: {option_type}")
            
            if strike <= 0 or strike > 10000:
                raise ValueError(f"Strike price must be between 0 and 10000, got: {strike}")
            
            # Format strike as 8-digit integer (multiply by 1000)
            strike_formatted = f"{int(strike * 1000):08d}"
            
            # Build OCC symbol
            occ_symbol = f"{stock.upper()}{expiry}{option_type.upper()}{strike_formatted}"
            
            logger.debug(f"Built OCC symbol: {occ_symbol} for {stock} {strike} {option_type} {expiry}")
            return occ_symbol
            
        except Exception as e:
            logger.error(f"Failed to build OCC symbol for {stock} {expiry} {strike} {option_type}: {e}")
            raise

    async def validate_options_order(self, order: OptionsOrder) -> bool:
        """Validate options order against risk parameters and account limits."""
        try:
            # Get account info for risk checks
            account_info = alpaca_client.get_account_info()
            portfolio_value = float(account_info.get('portfolio_value', 0))
            buying_power = float(account_info.get('buying_power', 0))
            
            # Calculate position value estimate
            estimated_cost = order.contracts * (order.premium or 0) * 100  # Each contract = 100 shares
            
            # Risk checks
            if estimated_cost > portfolio_value * self.max_single_position:
                logger.warning(f"Options order exceeds single position limit: ${estimated_cost:,.2f} > ${portfolio_value * self.max_single_position:,.2f}")
                return False
            
            # Check total options allocation
            current_options_value = sum(pos.contracts * pos.current_premium * 100 
                                      for pos in self.positions.values())
            total_after_trade = current_options_value + estimated_cost
            
            if total_after_trade > portfolio_value * self.max_options_allocation:
                logger.warning(f"Total options allocation would exceed limit: ${total_after_trade:,.2f} > ${portfolio_value * self.max_options_allocation:,.2f}")
                return False
            
            # Validate expiry is within acceptable range
            if order.expiry:
                try:
                    expiry_date = datetime.strptime(order.expiry, '%y%m%d')
                    days_to_expiry = (expiry_date - datetime.now()).days
                    
                    if days_to_expiry < self.min_days_to_expiry:
                        logger.warning(f"Options expiry too close: {days_to_expiry} days < {self.min_days_to_expiry} minimum")
                        return False
                    
                    if days_to_expiry > self.max_days_to_expiry:
                        logger.warning(f"Options expiry too far: {days_to_expiry} days > {self.max_days_to_expiry} maximum")
                        return False
                        
                except ValueError as e:
                    logger.error(f"Invalid expiry date format: {order.expiry}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Options order validation failed: {e}")
            return False

    async def place_options_order(self, order: OptionsOrder) -> Optional[Dict[str, Any]]:
        """
        Place an options order with full validation and error handling.
        
        Args:
            order: OptionsOrder object with all trade details
            
        Returns:
            Dictionary with order result or None if failed
        """
        try:
            # Validate order first
            if not await self.validate_options_order(order):
                logger.error(f"Options order validation failed for {order.occ_symbol}")
                return None
            
            logger.info(f"🎯 Placing options order: {order.strategy.value}")
            logger.info(f"   Symbol: {order.occ_symbol}")
            logger.info(f"   Contracts: {order.contracts}")
            logger.info(f"   Side: {order.side}")
            logger.info(f"   Rationale: {order.rationale}")
            
            # Create Alpaca order request
            # Note: This is a simplified implementation - in production you'd use actual Alpaca options API
            order_request = {
                'symbol': order.occ_symbol,
                'qty': order.contracts,
                'side': order.side,
                'type': 'market',  # Could be 'limit' with premium
                'time_in_force': 'day',
                'order_class': 'simple'
            }
            
            # For now, simulate the order since Alpaca paper doesn't fully support options
            # In production, you'd call: alpaca_client.api.submit_order(**order_request)
            
            # Simulate successful order
            order_result = {
                'id': f"options_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'status': 'filled',
                'filled_qty': order.contracts,
                'filled_avg_price': order.premium or 1.0,  # Simulate premium
                'symbol': order.occ_symbol,
                'side': order.side,
                'strategy': order.strategy.value,
                'timestamp': datetime.now().isoformat()
            }
            
            # Record the order
            order.order_id = order_result['id']
            self.order_history.append(order)
            
            # Update strategy metrics
            if order.strategy not in self.strategy_metrics:
                self.strategy_metrics[order.strategy] = {
                    'total_orders': 0,
                    'total_volume': 0,
                    'success_rate': 0.0
                }
            
            self.strategy_metrics[order.strategy]['total_orders'] += 1
            self.strategy_metrics[order.strategy]['total_volume'] += order.contracts
            
            logger.info(f"✅ Options order placed successfully: {order_result['id']}")
            return order_result
            
        except Exception as e:
            logger.error(f"Failed to place options order: {e}")
            return None

    # Strategy Implementation Methods

    async def hedge_with_put(self, stock: str, expiry: str, strike: float, contracts: int, 
                           rationale: str = "Portfolio hedging") -> Dict[str, Any]:
        """
        Buy protective puts as a hedge against long stock exposure.
        
        Args:
            stock: Underlying stock symbol
            expiry: Expiration date in YYMMDD format
            strike: Strike price
            contracts: Number of contracts
            rationale: Trading rationale
            
        Returns:
            Trade execution result
        """
        try:
            occ_symbol = self.build_occ_symbol(stock, expiry, strike, "P")
            
            # Get current stock price for risk calculation
            market_data = await self._get_market_data(stock)
            current_price = market_data.get('price', 0) if market_data else 0
            
            # Create options order
            order = OptionsOrder(
                symbol=stock,
                occ_symbol=occ_symbol,
                strategy=OptionsStrategy.PROTECTIVE_PUT,
                action=OptionsAction.HEDGE,
                contracts=contracts,
                side="buy",
                strike=strike,
                expiry=expiry,
                underlying_price=current_price,
                rationale=rationale,
                max_risk=contracts * 100 * (current_price - strike) if current_price > strike else 0
            )
            
            result = await self.place_options_order(order)
            
            return {
                "strategy": "protective_put",
                "symbol": occ_symbol,
                "underlying": stock,
                "contracts": contracts,
                "rationale": rationale,
                "order_result": result,
                "risk_profile": {
                    "max_loss": order.max_risk,
                    "breakeven": current_price - 1.0,  # Current price - estimated premium
                    "hedge_level": strike
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to execute protective put strategy: {e}")
            return {"error": str(e), "strategy": "protective_put"}

    async def write_covered_call(self, stock: str, expiry: str, strike: float, contracts: int,
                               rationale: str = "Income generation") -> Dict[str, Any]:
        """
        Sell covered calls to generate income on long stock positions.
        
        Args:
            stock: Underlying stock symbol
            expiry: Expiration date in YYMMDD format  
            strike: Strike price
            contracts: Number of contracts
            rationale: Trading rationale
            
        Returns:
            Trade execution result
        """
        try:
            occ_symbol = self.build_occ_symbol(stock, expiry, strike, "C")
            
            # Get current stock price
            market_data = await self._get_market_data(stock)
            current_price = market_data.get('price', 0) if market_data else 0
            
            # Verify we have sufficient stock position for covered call
            # In production, you'd check actual positions
            
            order = OptionsOrder(
                symbol=stock,
                occ_symbol=occ_symbol,
                strategy=OptionsStrategy.COVERED_CALL,
                action=OptionsAction.YIELD,
                contracts=contracts,
                side="sell",
                strike=strike,
                expiry=expiry,
                underlying_price=current_price,
                rationale=rationale,
                expected_profit=contracts * 100 * 1.0  # Estimated premium income
            )
            
            result = await self.place_options_order(order)
            
            return {
                "strategy": "covered_call",
                "symbol": occ_symbol,
                "underlying": stock,
                "contracts": contracts,
                "rationale": rationale,
                "order_result": result,
                "income_profile": {
                    "premium_income": order.expected_profit,
                    "assignment_risk": strike < current_price,
                    "max_profit": strike - current_price + 1.0  # Estimated premium
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to execute covered call strategy: {e}")
            return {"error": str(e), "strategy": "covered_call"}

    async def speculate_with_call(self, stock: str, expiry: str, strike: float, contracts: int,
                                rationale: str = "Bullish speculation") -> Dict[str, Any]:
        """
        Buy calls for directional speculation on bullish moves.
        
        Args:
            stock: Underlying stock symbol
            expiry: Expiration date in YYMMDD format
            strike: Strike price
            contracts: Number of contracts
            rationale: Trading rationale
            
        Returns:
            Trade execution result
        """
        try:
            occ_symbol = self.build_occ_symbol(stock, expiry, strike, "C")
            
            # Get current stock price
            market_data = await self._get_market_data(stock)
            current_price = market_data.get('price', 0) if market_data else 0
            
            order = OptionsOrder(
                symbol=stock,
                occ_symbol=occ_symbol,
                strategy=OptionsStrategy.LONG_CALL,
                action=OptionsAction.SPECULATE,
                contracts=contracts,
                side="buy",
                strike=strike,
                expiry=expiry,
                underlying_price=current_price,
                rationale=rationale,
                max_risk=contracts * 100 * 1.0  # Estimated premium paid
            )
            
            result = await self.place_options_order(order)
            
            return {
                "strategy": "long_call",
                "symbol": occ_symbol,
                "underlying": stock,
                "contracts": contracts,
                "rationale": rationale,
                "order_result": result,
                "speculation_profile": {
                    "max_loss": order.max_risk,
                    "breakeven": strike + 1.0,  # Strike + estimated premium
                    "leverage_ratio": (contracts * 100) / (order.max_risk or 1)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to execute call speculation strategy: {e}")
            return {"error": str(e), "strategy": "long_call"}

    async def volatility_straddle(self, stock: str, expiry: str, strike: float, contracts: int,
                                rationale: str = "Volatility play") -> Dict[str, Any]:
        """
        Buy straddle (call + put at same strike) to profit from volatility.
        
        Args:
            stock: Underlying stock symbol
            expiry: Expiration date in YYMMDD format
            strike: Strike price (usually ATM)
            contracts: Number of contracts
            rationale: Trading rationale
            
        Returns:
            Trade execution result
        """
        try:
            call_symbol = self.build_occ_symbol(stock, expiry, strike, "C")
            put_symbol = self.build_occ_symbol(stock, expiry, strike, "P")
            
            # Get current stock price
            market_data = await self._get_market_data(stock)
            current_price = market_data.get('price', 0) if market_data else 0
            
            # Execute both legs
            call_order = OptionsOrder(
                symbol=stock,
                occ_symbol=call_symbol,
                strategy=OptionsStrategy.STRADDLE,
                action=OptionsAction.VOLATILITY,
                contracts=contracts,
                side="buy",
                strike=strike,
                expiry=expiry,
                underlying_price=current_price,
                rationale=f"{rationale} - Call leg"
            )
            
            put_order = OptionsOrder(
                symbol=stock,
                occ_symbol=put_symbol,
                strategy=OptionsStrategy.STRADDLE,
                action=OptionsAction.VOLATILITY,
                contracts=contracts,
                side="buy",
                strike=strike,
                expiry=expiry,
                underlying_price=current_price,
                rationale=f"{rationale} - Put leg"
            )
            
            call_result = await self.place_options_order(call_order)
            put_result = await self.place_options_order(put_order)
            
            total_premium = 1.0 + 1.0  # Estimated premium for both legs
            
            return {
                "strategy": "straddle",
                "call_symbol": call_symbol,
                "put_symbol": put_symbol,
                "underlying": stock,
                "contracts": contracts,
                "rationale": rationale,
                "call_result": call_result,
                "put_result": put_result,
                "volatility_profile": {
                    "total_premium": total_premium * contracts * 100,
                    "upper_breakeven": strike + total_premium,
                    "lower_breakeven": strike - total_premium,
                    "profit_zone": f"Stock below {strike - total_premium:.2f} or above {strike + total_premium:.2f}"
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to execute straddle strategy: {e}")
            return {"error": str(e), "strategy": "straddle"}

    async def _get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current market data for a symbol."""
        try:
            # Use existing market data infrastructure
            from tools.alpaca_market_data import fetch_stock_prices
            price_data = fetch_stock_prices([symbol])
            price = price_data.get(symbol) if price_data else None
            return {'price': price} if price else None
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return None

    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get comprehensive strategy performance metrics."""
        return {
            "total_strategies": len(self.strategy_metrics),
            "total_orders": len(self.order_history),
            "active_positions": len(self.positions),
            "strategy_breakdown": self.strategy_metrics,
            "recent_orders": [
                {
                    "timestamp": order.timestamp.isoformat(),
                    "strategy": order.strategy.value,
                    "symbol": order.occ_symbol,
                    "contracts": order.contracts,
                    "rationale": order.rationale
                }
                for order in self.order_history[-10:]  # Last 10 orders
            ]
        }

# Global options agent instance
options_agent = OptionsTradingAgent()

# Example usage for testing
async def example_usage():
    """Example of how to use the options trading agent."""
    
    # Hedge AAPL position with protective put
    hedge_result = await options_agent.hedge_with_put(
        stock="AAPL",
        expiry="250117",  # Jan 17, 2025
        strike=150.0,
        contracts=1,
        rationale="Bearish sentiment detected in RL analysis, protecting long AAPL"
    )
    
    # Generate income with covered call
    income_result = await options_agent.write_covered_call(
        stock="MSFT",
        expiry="241220",  # Dec 20, 2024
        strike=420.0,
        contracts=2,
        rationale="High IV environment, generating income on MSFT holdings"
    )
    
    # Speculate on bullish move
    spec_result = await options_agent.speculate_with_call(
        stock="NVDA", 
        expiry="250221",  # Feb 21, 2025
        strike=500.0,
        contracts=1,
        rationale="Strong earnings sentiment, bullish on AI semiconductor sector"
    )
    
    print("Options trades executed:")
    print(f"Hedge: {hedge_result}")
    print(f"Income: {income_result}")
    print(f"Speculation: {spec_result}")

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())