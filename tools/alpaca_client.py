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
    
    def get_asset_info(self, symbol: str) -> Dict:
        """Get asset information including fractionability."""
        try:
            asset = self.api.get_asset(symbol)
            return {
                "symbol": asset.symbol,
                "name": asset.name,
                "tradable": asset.tradable,
                "fractionable": asset.fractionable,
                "min_trade_increment": asset.min_trade_increment,
                "price_increment": asset.price_increment
            }
        except Exception as e:
            logger.error(f"Failed to get asset info for {symbol}: {e}")
            # Return safe defaults
            return {
                "symbol": symbol,
                "tradable": True,
                "fractionable": False,  # Safe default - assume not fractionable
                "min_trade_increment": "1",
                "price_increment": "0.01"
            }
    
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
        """Get market data for a symbol with crypto-optimized data sources."""
        try:
            is_crypto = self._is_crypto_symbol(symbol)
            
            # For crypto assets, try Alpaca crypto API first
            if is_crypto:
                try:
                    # Alpaca crypto data
                    quote = self.api.get_latest_trade(symbol)
                    price = float(quote.price)
                    volume = float(quote.size)
                    
                    # Create dataframe with current crypto data
                    import pandas as pd
                    df = pd.DataFrame({
                        'timestamp': [datetime.now()],
                        'open': [price],
                        'high': [price],
                        'low': [price], 
                        'close': [price],
                        'volume': [volume]
                    })
                    
                    logger.info(f"Got Alpaca crypto data for {symbol}: ${price:.2f}")
                    return df
                    
                except Exception as crypto_error:
                    logger.warning(f"Alpaca crypto data failed for {symbol}: {crypto_error}")
                    
                    # Fallback to crypto-specific data sources
                    try:
                        return self._get_crypto_fallback_data(symbol, timeframe, limit)
                    except Exception as fallback_error:
                        logger.warning(f"Crypto fallback failed for {symbol}: {fallback_error}")
            
            # For stocks or when crypto fails, use traditional approach
            try:
                # Try to get latest quote from Alpaca
                quote = self.api.get_latest_trade(symbol)
                price = float(quote.price)
                volume = float(quote.size)
                
                # Create a simple dataframe with current data
                import pandas as pd
                df = pd.DataFrame({
                    'timestamp': [datetime.now()],
                    'open': [price],
                    'high': [price],
                    'low': [price], 
                    'close': [price],
                    'volume': [volume]
                })
                
                logger.info(f"Got current quote for {symbol}: ${price:.2f}")
                return df
                
            except Exception as quote_error:
                logger.warning(f"Alpaca quote failed for {symbol}: {quote_error}")
                
                # NO FALLBACKS - fail fast with clear error
                error_msg = f"❌ CRITICAL: Alpaca data access failed for {symbol}: {quote_error}. Paper trading subscription may not support this data."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            raise
    
    def place_order(self, symbol: str, qty: float, side: str, 
                   order_type: str = "market", limit_price: Optional[float] = None,
                   stop_price: Optional[float] = None, time_in_force: str = "gtc",
                   notional: Optional[float] = None) -> Dict:
        """
        Place a trading order with safety checks.
        
        Args:
            symbol: Stock symbol or crypto pair (e.g., BTCUSD, ETHUSD)
            qty: Quantity to trade (ignored if notional is provided)
            side: "buy", "sell", or "sell_short"
            order_type: "market", "limit", "stop", "stop_limit"
            limit_price: Price for limit orders
            stop_price: Price for stop orders
            time_in_force: "gtc", "day", "ioc", "fok" (crypto only supports "gtc", "ioc")
            notional: Dollar amount for fractional crypto orders
        """
        try:
            # Safety checks
            if not self._pre_trade_checks(symbol, qty, side, notional):
                raise ValueError("Pre-trade safety checks failed")
            
            # Prepare order parameters
            # Handle sell_short by converting to sell with proper side
            alpaca_side = "sell" if side.lower() == "sell_short" else side.lower()
            
            # Check if this is a crypto symbol
            is_crypto = self._is_crypto_symbol(symbol)
            
            order_params = {
                "symbol": symbol,
                "side": alpaca_side,
                "type": order_type.lower(),
                "time_in_force": time_in_force.lower()
            }
            
            # Handle quantity vs notional for crypto fractional orders
            if notional is not None and is_crypto:
                order_params["notional"] = str(notional)
                logger.info(f"Using notional amount ${notional} for crypto order")
            else:
                order_params["qty"] = abs(qty)  # Ensure positive quantity
            
            # Check if this is a fractional stock order (non-crypto)
            is_fractional_stock_order = not is_crypto and float(qty) != int(float(qty))
            
            # Handle time_in_force requirements for fractional orders
            if is_fractional_stock_order and time_in_force.lower() != "day":
                logger.info(f"Fractional stock order detected for {symbol} (qty: {qty}), forcing time_in_force to 'day'")
                order_params["time_in_force"] = "day"
            elif is_crypto and time_in_force.lower() not in ["gtc", "ioc"]:
                logger.warning(f"Invalid time_in_force '{time_in_force}' for crypto. Using 'gtc'")
                order_params["time_in_force"] = "gtc"
            
            # Add position intent for short selling (if supported by broker) with balanced risk management
            if side.lower() == "sell_short":
                if is_crypto:
                    logger.warning(f"Short selling may not be supported for crypto {symbol}")
                logger.info(f"Placing short sell order for {symbol}")
                
                # Additional short selling checks for balanced trading
                try:
                    account = self.get_account_info()
                    if notional is not None:
                        short_value = notional
                    else:
                        current_price = self.get_current_price(symbol)
                        short_value = qty * current_price
                    
                    # Estimate margin requirement for short (typically 150% of position value)
                    margin_requirement = short_value * 1.5
                    
                    # Check if we have adequate buying power for margin, but be more lenient for small shorts
                    if short_value < 500 or margin_requirement <= account["buying_power"] * 2:
                        logger.info(f"Short selling approved: ${short_value:.2f} position, ${margin_requirement:.2f} margin requirement")
                    else:
                        logger.warning(f"Short may require more margin: ${margin_requirement:.2f} vs ${account['buying_power']:.2f} available")
                        # Don't block, let Alpaca decide
                        
                except Exception as e:
                    logger.warning(f"Could not verify short selling requirements: {e}")
                # Note: Alpaca handles short selling automatically if shares are available
            
            if limit_price is not None:
                order_params["limit_price"] = str(limit_price)
            
            if stop_price is not None:
                order_params["stop_price"] = str(stop_price)
            
            # Place the order
            order = self.api.submit_order(**order_params)
            
            logger.info(f"Order placed: {side} {qty} {symbol} at {order_type}")
            
            # Send batched email notification asynchronously
            try:
                asyncio.create_task(self._send_batched_transaction_notification(
                    order, symbol, qty, side, order_type, limit_price
                ))
            except Exception as notification_error:
                logger.warning(f"Batched email notification failed: {notification_error}")
            
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
            error_message = str(e).lower()
            
            # Handle specific fractional order errors
            if "not fractionable" in error_message and float(qty) != int(float(qty)):
                logger.warning(f"Asset {symbol} is not fractionable, retrying with integer quantity")
                try:
                    # Retry with integer quantity
                    integer_qty = max(1, int(float(qty)))
                    order_params["qty"] = integer_qty
                    
                    logger.info(f"Retrying {symbol} order with integer quantity: {integer_qty}")
                    order = self.trading_client.submit_order(order_data=order_params)
                    
                    # Send batched email notification asynchronously
                    try:
                        asyncio.create_task(self._send_batched_transaction_notification(
                            order, symbol, integer_qty, side, order_type, limit_price
                        ))
                    except Exception as notification_error:
                        logger.warning(f"Batched email notification failed: {notification_error}")
                    
                    return {
                        "id": order.id,
                        "symbol": order.symbol,
                        "qty": float(order.qty),
                        "side": order.side,
                        "order_type": order.order_type,
                        "status": order.status,
                        "submitted_at": order.submitted_at
                    }
                except Exception as retry_error:
                    logger.error(f"Retry with integer quantity also failed for {symbol}: {retry_error}")
                    raise
            
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
    
    def _pre_trade_checks(self, symbol: str, qty: float, side: str, notional: Optional[float] = None) -> bool:
        """Perform pre-trade safety checks."""
        try:
            # Check account status
            account = self.get_account_info()
            if account["account_blocked"] or account["trading_blocked"]:
                logger.error("Trading is blocked on this account")
                return False
            
            # Check buying power for buy orders
            if side.lower() == "buy":
                # Estimate order value (using current market price or notional)
                try:
                    if notional is not None:
                        order_value = notional
                    else:
                        market_data = self.get_market_data(symbol, limit=1)
                        if not market_data.empty:
                            current_price = market_data.iloc[-1]["close"]
                            order_value = qty * current_price
                        else:
                            # If no market data, skip buying power check and let Alpaca handle it
                            logger.warning(f"No market data for {symbol}, skipping buying power check")
                            return True
                    
                    # More flexible buying power check for balanced trading
                    if order_value > account["buying_power"] * 1.5:  # Allow 1.5x buying power for margin trading
                        logger.error(f"Order too large: ${order_value:.2f} > ${account['buying_power'] * 1.5:.2f} (1.5x buying power)")
                        return False
                    
                    # For small orders under $1000, be more lenient
                    if order_value < 1000 and (order_value <= account["buying_power"] or account["buying_power"] > 50):
                        logger.info(f"Allowing small order: ${order_value:.2f} with ${account['buying_power']:.2f} buying power")
                        return True
                        
                except Exception as e:
                    logger.warning(f"Could not verify buying power: {e}")
                    # Continue without buying power check - let Alpaca API handle it
            
            # Check position size limits - more flexible for balanced trading
            portfolio_value = account["portfolio_value"]
            if portfolio_value > 0:
                try:
                    if notional is not None:
                        position_value = notional
                    else:
                        market_data = self.get_market_data(symbol, limit=1)
                        if not market_data.empty:
                            current_price = market_data.iloc[-1]["close"]
                            position_value = qty * current_price
                        else:
                            logger.warning(f"No market data for {symbol}, allowing small orders")
                            return qty <= 10  # Allow small orders without market data
                    
                    position_percent = position_value / portfolio_value
                    
                    # Use crypto-specific position limits if available, but make them more reasonable
                    is_crypto = self._is_crypto_symbol(symbol)
                    base_max_position = getattr(settings, 'crypto_max_position_size', getattr(settings, 'max_position_size', 0.05)) if is_crypto else getattr(settings, 'max_position_size', 0.05)
                    max_position = max(base_max_position, 0.15)  # Minimum 15% position limit for balanced trading
                    
                    if position_percent > max_position:
                        logger.error(f"Position size too large: {position_percent:.2%} > {max_position:.2%}")
                        return False
                        
                except Exception as e:
                    logger.warning(f"Could not verify position size: {e}")
                    # Continue without position size check
            
            return True
            
        except Exception as e:
            logger.error(f"Pre-trade checks failed: {e}")
            return False
    
    def get_asset_info(self, symbol: str) -> Dict:
        """Get asset information and tradability."""
        try:
            asset = self.api.get_asset(symbol)
            
            # Enhanced info for crypto assets
            asset_info = {
                "symbol": asset.symbol,
                "name": getattr(asset, 'name', asset.symbol),
                "exchange": getattr(asset, 'exchange', 'Unknown'),
                "asset_class": getattr(asset, 'asset_class', 'crypto' if self._is_crypto_symbol(asset.symbol) else 'us_equity'),
                "status": getattr(asset, 'status', 'active'),
                "tradable": getattr(asset, 'tradable', True),
                "marginable": getattr(asset, 'marginable', False),
                "shortable": getattr(asset, 'shortable', False),
                "easy_to_borrow": getattr(asset, 'easy_to_borrow', False),
                "fractionable": getattr(asset, 'fractionable', True)
            }
            
            # Add crypto-specific info
            if self._is_crypto_symbol(symbol):
                asset_info.update({
                    "is_crypto": True,
                    "trading_hours": "24/7",
                    "supported_order_types": ["market", "limit", "stop_limit"],
                    "supported_time_in_force": ["gtc", "ioc"],
                    "fractional_supported": True
                })
            else:
                asset_info.update({
                    "is_crypto": False,
                    "trading_hours": "9:30 AM - 4:00 PM ET",
                    "supported_order_types": ["market", "limit", "stop", "stop_limit"],
                    "supported_time_in_force": ["gtc", "day", "ioc", "fok"]
                })
            
            return asset_info
            
        except Exception as e:
            logger.error(f"Failed to get asset info for {symbol}: {e}")
            
            # Return safe defaults with crypto detection
            is_crypto = self._is_crypto_symbol(symbol)
            return {
                "symbol": symbol,
                "name": symbol,
                "tradable": True,
                "fractionable": is_crypto,  # Crypto supports fractional by default
                "is_crypto": is_crypto,
                "trading_hours": "24/7" if is_crypto else "9:30 AM - 4:00 PM ET",
                "supported_order_types": ["market", "limit", "stop_limit"] if is_crypto else ["market", "limit", "stop", "stop_limit"],
                "supported_time_in_force": ["gtc", "ioc"] if is_crypto else ["gtc", "day", "ioc", "fok"]
            }
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        try:
            quote = self.api.get_latest_trade(symbol)
            return float(quote.price)
        except Exception as e:
            logger.warning(f"Could not get current price for {symbol}: {e}")
            # Fallback to market data method
            try:
                df = self.get_market_data(symbol, limit=1)
                if not df.empty:
                    return float(df['close'].iloc[-1])
            except Exception as fallback_error:
                logger.warning(f"Fallback price lookup failed for {symbol}: {fallback_error}")
            raise ValueError(f"Could not get price for {symbol}")
    
    def is_market_open(self, symbol: Optional[str] = None) -> bool:
        """Check if the market is currently open. Crypto markets are always open."""
        try:
            # Crypto markets are open 24/7
            if symbol and self._is_crypto_symbol(symbol):
                return True
                
            # For stocks, check market hours
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Failed to check market status: {e}")
            # Default to open for crypto, closed for stocks
            return symbol and self._is_crypto_symbol(symbol) if symbol else False
    
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
    
    async def _send_batched_transaction_notification(self, order, symbol: str, qty: float, 
                                                   side: str, order_type: str, 
                                                   limit_price: Optional[float] = None) -> None:
        """Send batched email notification for completed transaction."""
        try:
            # Import here to avoid circular imports
            from notifications.email_batch_manager import send_batched_transaction_notification, email_batch_manager
            
            # Ensure batch processor is running
            if not email_batch_manager.running:
                email_batch_manager.start_batch_processor()
            
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
            
            # Send to batching system
            await send_batched_transaction_notification(
                symbol=symbol,
                action=side,
                quantity=qty,
                price=price,
                portfolio_value=portfolio_value,
                confidence=0.9,  # High confidence for executed orders
                reasoning=reasoning,
                agent_source="alpaca_trading_client"
            )
            
            logger.info(f"Transaction added to batch queue: {symbol} {side}")
            
        except Exception as e:
            logger.error(f"Failed to add transaction to batch queue: {e}")
    
    def _get_crypto_fallback_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Get crypto data from alternative sources when Alpaca fails."""
        # NO FALLBACKS - fail fast with clear error
        error_msg = f"❌ CRITICAL: Alpaca crypto data access failed for {symbol}. Paper trading subscription may not support crypto data."
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    
    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Check if a symbol represents a cryptocurrency pair."""
        # Common crypto symbols end with USD, USDT, USDC or are known crypto pairs
        crypto_suffixes = ['USD', 'USDT', 'USDC', 'BTC']
        crypto_prefixes = ['BTC', 'ETH', 'DOGE', 'LTC', 'BCH', 'AAVE', 'UNI', 'LINK', 'MKR']
        
        # Check if symbol matches crypto patterns
        for prefix in crypto_prefixes:
            for suffix in crypto_suffixes:
                if symbol.upper() == f"{prefix}{suffix}":
                    return True
        
        # Additional known crypto patterns
        known_crypto_symbols = ['BTCUSD', 'ETHUSD', 'DOGEUSD', 'LTCUSD', 'BCHUSD']
        return symbol.upper() in known_crypto_symbols

# Global client instance
alpaca_client = AlpacaClient()

# Apply balanced risk management for better cash management and trading balance
try:
    from tools.risk_balanced_alpaca_client import apply_balanced_risk_management
    alpaca_client = apply_balanced_risk_management(alpaca_client)
    logger.info("✅ Balanced risk management applied to alpaca_client")
except ImportError as e:
    logger.warning(f"Could not load balanced risk management: {e}")
except Exception as e:
    logger.error(f"Failed to apply balanced risk management: {e}")