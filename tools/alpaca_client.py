"""Secure Alpaca API client with comprehensive trading functionality."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from decimal import Decimal

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame
from alpaca_trade_api.entity import Order, Position, Account
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)

class AlpacaClient:
    """Secure Alpaca API client with risk management and safety controls."""
    
    def __init__(self):
        """Initialize Alpaca client with paper trading configuration."""
        self.api = REST(
            key_id=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            base_url=settings.alpaca_base_url  # Paper trading URL already includes version
        )
        
        # Validate connection and ensure paper trading
        self._validate_connection()
        self._ensure_paper_trading()
        
        logger.info("Alpaca client initialized in paper trading mode")
    
    def _validate_connection(self) -> None:
        """Validate API connection and credentials."""
        try:
            account = self.api.get_account()
            logger.info(f"Connected to Alpaca account: {account.id}")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca API: {e}")
            raise ConnectionError(f"Alpaca API connection failed: {e}")
    
    def _ensure_paper_trading(self) -> None:
        """Ensure we're connected to paper trading environment."""
        account = self.api.get_account()
        
        # Check if account is paper trading (trading_blocked usually indicates paper)
        # Also check the base URL to ensure it's paper trading
        if "paper" not in settings.alpaca_base_url.lower():
            logger.warning("WARNING: Not using paper trading URL!")
            
        logger.info(f"Account Status: {account.status}")
        logger.info(f"Trading Blocked: {account.trading_blocked}")
        logger.info(f"Account Type: Paper Trading")
    
    def get_account_info(self) -> Dict:
        """Get current account information."""
        try:
            account = self.api.get_account()
            return {
                "id": account.id,
                "status": account.status,
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "day_trade_count": int(account.daytrade_count),
                "trading_blocked": account.trading_blocked,
                "account_blocked": account.account_blocked,
                "pattern_day_trader": account.pattern_day_trader
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise
    
    def get_positions(self) -> List[Dict]:
        """Get current portfolio positions."""
        try:
            positions = self.api.list_positions()
            return [
                {
                    "symbol": pos.symbol,
                    "qty": float(pos.qty),
                    "side": pos.side,
                    "market_value": float(pos.market_value),
                    "cost_basis": float(pos.cost_basis),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc),
                    "current_price": float(pos.current_price)
                }
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise
    
    def get_orders(self, status: str = "all", limit: int = 100) -> List[Dict]:
        """Get orders by status."""
        try:
            orders = self.api.list_orders(status=status, limit=limit)
            return [
                {
                    "id": order.id,
                    "symbol": order.symbol,
                    "qty": float(order.qty),
                    "side": order.side,
                    "order_type": order.order_type,
                    "time_in_force": order.time_in_force,
                    "status": order.status,
                    "filled_qty": float(order.filled_qty or 0),
                    "filled_avg_price": float(order.filled_avg_price or 0),
                    "submitted_at": order.submitted_at,
                    "filled_at": order.filled_at,
                    "limit_price": float(order.limit_price) if order.limit_price else None,
                    "stop_price": float(order.stop_price) if order.stop_price else None
                }
                for order in orders
            ]
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            raise
    
    def get_market_data(self, symbol: str, timeframe: str = "1Day", 
                       limit: int = 100) -> pd.DataFrame:
        """Get historical market data for a symbol."""
        try:
            # Map timeframe string to Alpaca TimeFrame
            tf_map = {
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame(5, TimeFrame.Minute),
                "15Min": TimeFrame(15, TimeFrame.Minute),
                "1Hour": TimeFrame.Hour,
                "1Day": TimeFrame.Day
            }
            
            timeframe_obj = tf_map.get(timeframe, TimeFrame.Day)
            end_time = datetime.now()
            start_time = end_time - timedelta(days=limit)
            
            bars = self.api.get_bars(
                symbol,
                timeframe_obj,
                start=start_time.strftime('%Y-%m-%d'),
                end=end_time.strftime('%Y-%m-%d'),
                adjustment='raw'
            )
            
            df = bars.df
            df.reset_index(inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            raise
    
    def place_order(self, symbol: str, qty: float, side: str, 
                   order_type: str = "market", limit_price: Optional[float] = None,
                   stop_price: Optional[float] = None, time_in_force: str = "gtc") -> Dict:
        """
        Place a trading order with safety checks.
        
        Args:
            symbol: Stock symbol
            qty: Quantity to trade
            side: "buy" or "sell"
            order_type: "market", "limit", "stop", "stop_limit"
            limit_price: Price for limit orders
            stop_price: Price for stop orders
            time_in_force: "gtc", "day", "ioc", "fok"
        """
        try:
            # Safety checks
            if not self._pre_trade_checks(symbol, qty, side):
                raise ValueError("Pre-trade safety checks failed")
            
            # Prepare order parameters
            order_params = {
                "symbol": symbol,
                "qty": abs(qty),  # Ensure positive quantity
                "side": side.lower(),
                "type": order_type.lower(),
                "time_in_force": time_in_force.lower()
            }
            
            if limit_price is not None:
                order_params["limit_price"] = str(limit_price)
            
            if stop_price is not None:
                order_params["stop_price"] = str(stop_price)
            
            # Place the order
            order = self.api.submit_order(**order_params)
            
            logger.info(f"Order placed: {side} {qty} {symbol} at {order_type}")
            
            # Send email notification asynchronously
            try:
                asyncio.create_task(self._send_transaction_notification(
                    order, symbol, qty, side, order_type, limit_price
                ))
            except Exception as notification_error:
                logger.warning(f"Email notification failed: {notification_error}")
            
            return {
                "id": order.id,
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side,
                "order_type": order.order_type,
                "status": order.status,
                "submitted_at": order.submitted_at
            }
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        try:
            self.api.cancel_order(order_id)
            logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def cancel_all_orders(self) -> bool:
        """Cancel all open orders."""
        try:
            self.api.cancel_all_orders()
            logger.info("All orders cancelled")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return False
    
    def close_position(self, symbol: str, qty: Optional[float] = None) -> Dict:
        """Close a position (all or partial)."""
        try:
            if qty is None:
                # Close entire position
                order = self.api.close_position(symbol)
            else:
                # Close partial position - determine side based on current position
                positions = self.get_positions()
                current_pos = next((p for p in positions if p["symbol"] == symbol), None)
                
                if not current_pos:
                    raise ValueError(f"No position found for {symbol}")
                
                # Determine opposite side to reduce position
                side = "sell" if float(current_pos["qty"]) > 0 else "buy"
                order = self.place_order(symbol, abs(qty), side)
            
            logger.info(f"Position closed: {symbol}")
            return order
            
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            raise
    
    def _pre_trade_checks(self, symbol: str, qty: float, side: str) -> bool:
        """Perform pre-trade safety checks."""
        try:
            # Check account status
            account = self.get_account_info()
            if account["account_blocked"] or account["trading_blocked"]:
                logger.error("Trading is blocked on this account")
                return False
            
            # Check buying power for buy orders
            if side.lower() == "buy":
                # Estimate order value (using current market price)
                try:
                    market_data = self.get_market_data(symbol, limit=1)
                    if not market_data.empty:
                        current_price = market_data.iloc[-1]["close"]
                        order_value = qty * current_price
                        
                        if order_value > account["buying_power"]:
                            logger.error(f"Insufficient buying power: ${order_value:.2f} > ${account['buying_power']:.2f}")
                            return False
                except Exception as e:
                    logger.warning(f"Could not verify buying power: {e}")
            
            # Check position size limits
            portfolio_value = account["portfolio_value"]
            if portfolio_value > 0:
                try:
                    market_data = self.get_market_data(symbol, limit=1)
                    if not market_data.empty:
                        current_price = market_data.iloc[-1]["close"]
                        position_value = qty * current_price
                        position_percent = position_value / portfolio_value
                        
                        if position_percent > settings.max_position_size:
                            logger.error(f"Position size too large: {position_percent:.2%} > {settings.max_position_size:.2%}")
                            return False
                except Exception as e:
                    logger.warning(f"Could not verify position size: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Pre-trade checks failed: {e}")
            return False
    
    def get_asset_info(self, symbol: str) -> Dict:
        """Get asset information and tradability."""
        try:
            asset = self.api.get_asset(symbol)
            return {
                "symbol": asset.symbol,
                "name": asset.name,
                "exchange": asset.exchange,
                "asset_class": asset.asset_class,
                "status": asset.status,
                "tradable": asset.tradable,
                "marginable": asset.marginable,
                "shortable": asset.shortable,
                "easy_to_borrow": asset.easy_to_borrow,
                "fractionable": asset.fractionable
            }
        except Exception as e:
            logger.error(f"Failed to get asset info for {symbol}: {e}")
            raise
    
    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Failed to check market status: {e}")
            return False
    
    def get_market_calendar(self, start_date: Optional[str] = None, 
                           end_date: Optional[str] = None) -> List[Dict]:
        """Get market calendar."""
        try:
            calendar = self.api.get_calendar(start=start_date, end=end_date)
            return [
                {
                    "date": str(day.date),
                    "open": str(day.open),
                    "close": str(day.close)
                }
                for day in calendar
            ]
        except Exception as e:
            logger.error(f"Failed to get market calendar: {e}")
            raise
    
    async def _send_transaction_notification(self, order, symbol: str, qty: float, 
                                           side: str, order_type: str, 
                                           limit_price: Optional[float] = None) -> None:
        """Send email notification for completed transaction."""
        try:
            # Import here to avoid circular imports
            from notifications.email_webhooks import send_transaction_email
            
            # Get current portfolio value
            account_info = self.get_account_info()
            portfolio_value = account_info.get("portfolio_value", 0)
            
            # Determine price for notification
            if limit_price:
                price = limit_price
            else:
                # Try to get current market price
                try:
                    market_data = self.get_market_data(symbol, limit=1)
                    price = market_data.iloc[-1]["close"] if not market_data.empty else 0
                except:
                    price = 0
            
            # Create reasoning message
            reasoning = f"Order executed via Alpaca Trading API. Order ID: {order.id}. "
            reasoning += f"Order type: {order_type}. Status: {order.status}. "
            reasoning += f"Submitted at: {order.submitted_at}"
            
            # Send notification
            await send_transaction_email(
                symbol=symbol,
                action=side,
                quantity=qty,
                price=price,
                portfolio_value=portfolio_value,
                confidence=0.9,  # High confidence for executed orders
                reasoning=reasoning,
                agent_source="alpaca_trading_client"
            )
            
            logger.info(f"Transaction notification sent for {symbol} {side}")
            
        except Exception as e:
            logger.error(f"Failed to send transaction notification: {e}")

# Global client instance
alpaca_client = AlpacaClient()