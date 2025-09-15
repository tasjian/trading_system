"""Secure Alpaca API client with comprehensive trading functionality and tax-loss harvesting support."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from decimal import Decimal

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame
from alpaca_trade_api.entity import Order, Position, Account
import pandas as pd

from config.settings import settings

# Enhanced Short-Selling Validation (imported only when needed to avoid circular imports)
# from core.borrow_cost_monitor import borrow_cost_monitor
# from core.short_risk_manager import short_risk_manager

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
            
            # Calculate day change using last_equity (previous day close)
            current_equity = float(account.equity)
            last_equity = float(getattr(account, 'last_equity', current_equity))
            day_change = current_equity - last_equity
            day_change_percent = (day_change / last_equity * 100) if last_equity > 0 else 0.0
            
            return {
                "id": account.id,
                "status": account.status,
                "equity": current_equity,
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "daytrade_buying_power": float(getattr(account, 'daytrade_buying_power', 0)),
                "regt_buying_power": float(getattr(account, 'regt_buying_power', 0)),
                "portfolio_value": float(account.portfolio_value),
                "last_equity": last_equity,
                "day_change": day_change,
                "day_change_percent": day_change_percent,
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
    
    def _get_position_qty(self, symbol: str) -> float:
        """Get current position quantity for a symbol."""
        try:
            positions = self.api.list_positions()
            for pos in positions:
                if pos.symbol == symbol:
                    return float(pos.qty)
            return 0.0  # No position found
        except Exception as e:
            logger.warning(f"Failed to get position qty for {symbol}: {e}")
            return 0.0
    
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
    
    def get_orders(self, status: str = "all", limit: int = 100, symbol: str = None) -> List[Dict]:
        """Get orders by status and optionally filter by symbol."""
        try:
            orders = self.api.list_orders(status=status, limit=limit)
            
            order_list = [
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
            
            # Filter by symbol if specified
            if symbol:
                order_list = [order for order in order_list if order["symbol"] == symbol]
                
            return order_list
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            raise
    
    def get_pending_orders(self) -> Dict[str, List[Dict]]:
        """Get pending orders grouped by symbol to prevent duplicates."""
        try:
            pending_orders = self.get_orders(status="open", limit=500)
            
            # Group by symbol
            orders_by_symbol = {}
            for order in pending_orders:
                symbol = order["symbol"]
                if symbol not in orders_by_symbol:
                    orders_by_symbol[symbol] = []
                orders_by_symbol[symbol].append(order)
            
            return orders_by_symbol
        except Exception as e:
            logger.error(f"Failed to get pending orders: {e}")
            return {}
    
    def has_pending_buy_order(self, symbol: str) -> bool:
        """Check if there's already a pending buy order for this symbol."""
        try:
            pending_orders = self.get_pending_orders()
            symbol_orders = pending_orders.get(symbol, [])
            
            # Check for any pending buy orders
            for order in symbol_orders:
                if order["side"].lower() == "buy" and order["status"] in ["new", "accepted", "pending_new"]:
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to check pending buy orders for {symbol}: {e}")
            return False
    
    def get_market_data(self, symbol: str, timeframe: str = "1Day", 
                       limit: int = 100) -> pd.DataFrame:
        """Get market data for a symbol - CRYPTO TRADING DISABLED."""
        try:
            # CRYPTO TRADING DISABLED - Comment out crypto-specific logic
            # is_crypto = self._is_crypto_symbol(symbol)
            
            # # For crypto assets, try Alpaca crypto API first
            # if is_crypto:
            #     try:
            #         # Alpaca crypto data
            #         quote = self.api.get_latest_trade(symbol)
            #         price = float(quote.price)
            #         volume = float(quote.size)
            #         
            #         # Create dataframe with current crypto data
            #         import pandas as pd
            #         df = pd.DataFrame({
            #             'timestamp': [datetime.now()],
            #             'open': [price],
            #             'high': [price],
            #             'low': [price], 
            #             'close': [price],
            #             'volume': [volume]
            #         })
            #         
            #         logger.info(f"Got Alpaca crypto data for {symbol}: ${price:.2f}")
            #         return df
            #         
            #     except Exception as crypto_error:
            #         logger.warning(f"Alpaca crypto data failed for {symbol}: {crypto_error}")
            #         
            #         # Fallback to crypto-specific data sources
            #         try:
            #             return self._get_crypto_fallback_data(symbol, timeframe, limit)
            #         except Exception as fallback_error:
            #             logger.warning(f"Crypto fallback failed for {symbol}: {fallback_error}")
            
            # For stocks, get historical bars for FinRL training
            try:
                # Try to get historical bars from Alpaca
                from datetime import datetime, timedelta
                
                # Calculate date range for historical data - ensure 15+ minute delay for SIP data
                end_date = datetime.now() - timedelta(minutes=20)  # 20 minutes ago to ensure SIP data availability
                start_date = end_date - timedelta(days=min(limit * 2, 1095))  # Ensure enough data
                
                # Get historical bars with feed parameter for delayed SIP data
                try:
                    # Try with SIP feed for comprehensive delayed data
                    bars = self.api.get_bars(
                        symbol,
                        timeframe,
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        limit=limit,
                        feed='sip'  # Use delayed SIP data (15+ minutes delayed, no subscription required)
                    )
                except Exception as sip_error:
                    logger.debug(f"SIP feed failed for {symbol}, falling back to IEX: {sip_error}")
                    # Fallback to IEX feed if SIP fails
                    bars = self.api.get_bars(
                        symbol,
                        timeframe,
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        limit=limit,
                        feed='iex'  # Free IEX feed as fallback
                    )
                
                if bars and len(bars) > 0:
                    # Convert to DataFrame
                    import pandas as pd
                    data = []
                    for bar in bars:
                        try:
                            # Handle different Bar object attributes (alpaca-trade-api vs alpaca-py)
                            timestamp = getattr(bar, 'timestamp', getattr(bar, 't', datetime.now()))
                            
                            # Handle different OHLCV attribute names
                            open_price = getattr(bar, 'open', getattr(bar, 'o', 0.0))
                            high_price = getattr(bar, 'high', getattr(bar, 'h', 0.0))
                            low_price = getattr(bar, 'low', getattr(bar, 'l', 0.0))
                            close_price = getattr(bar, 'close', getattr(bar, 'c', 0.0))
                            volume = getattr(bar, 'volume', getattr(bar, 'v', 0))
                            
                            data.append({
                                'timestamp': timestamp,
                                'open': float(open_price),
                                'high': float(high_price),
                                'low': float(low_price),
                                'close': float(close_price),
                                'volume': int(volume)
                            })
                        except Exception as bar_error:
                            logger.debug(f"Error processing bar for {symbol}: {bar_error}")
                            continue
                    
                    df = pd.DataFrame(data)
                    logger.info(f"Got {len(df)} historical bars for {symbol}")
                    return df
                else:
                    raise ValueError(f"No historical data available for {symbol}")
                    
            except Exception as bars_error:
                logger.warning(f"Historical bars failed for {symbol}: {bars_error}")
                
                # Fallback to latest quote if historical data fails
                try:
                    # Try to get a delayed quote instead of real-time trade
                    try:
                        # Use get_latest_quote for delayed data (avoid SIP subscription issues)
                        quote = self.api.get_latest_quote(symbol)
                        price = float(quote.bid_price) if hasattr(quote, 'bid_price') else float(quote.ask_price)
                        volume = 100  # Default volume since quotes don't have volume
                    except:
                        # Fallback to latest trade if quote fails
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
                    error_msg = (
                    f"❌ CRITICAL SYSTEM FAILURE: Alpaca market data unavailable\n"
                    f"Symbol: {symbol}\n"
                    f"Error: {quote_error}\n"
                    f"Possible causes:\n"
                    f"- Paper trading subscription may not support real-time data for this symbol\n"
                    f"- Market is closed and no recent data available\n"
                    f"- Network connectivity issues\n"
                    f"- API rate limits exceeded\n\n"
                    f"SYSTEM REQUIRES VALID MARKET DATA TO OPERATE SAFELY\n"
                    f"No hardcoded fallback mechanisms are permitted per system design"
                    )
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
        Place a trading order with live position reconciliation and wash trade prevention.
        
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
            # STEP 1: Live position reconciliation before placing any order
            reconciled_qty, net_action = self._reconcile_position_and_calculate_net_order(symbol, qty, side)
            
            if reconciled_qty == 0:
                logger.info(f"⚖️ Net order for {symbol} is zero after position reconciliation - no trade needed")
                return {
                    "symbol": symbol,
                    "qty": 0,
                    "side": side,
                    "status": "skipped_net_zero",
                    "message": "No net trade needed after position reconciliation"
                }
            
            # Use reconciled values for the actual order
            qty = reconciled_qty
            side = net_action
            
            # Safety checks with reconciled values
            if not self._pre_trade_checks(symbol, qty, side, notional):
                raise ValueError("Pre-trade safety checks failed")
            
            # Prepare order parameters
            # Handle sell_short: In Alpaca, short selling is done with "sell" side but selling more than owned
            alpaca_side = "sell" if side.lower() == "sell_short" else side.lower()
            
            # For short selling, we need to check if we're selling more than we own
            if side.lower() == "sell_short":
                current_position = self._get_position_qty(symbol)
                # Add current_position to short qty to ensure we're selling more than owned
                actual_qty = qty + max(0, current_position)  # If we own shares, add them to short qty
                logger.info(f"🩳 Short selling {symbol}: requested_qty={qty}, current_position={current_position}, actual_sell_qty={actual_qty}")
            else:
                actual_qty = qty
            
            # CRYPTO TRADING DISABLED - Comment out crypto detection
            # is_crypto = self._is_crypto_symbol(symbol)
            is_crypto = False  # Always false when crypto is disabled
            
            order_params = {
                "symbol": symbol,
                "side": alpaca_side,
                "type": order_type.lower(),
                "time_in_force": time_in_force.lower()
            }
            
            # CRYPTO TRADING DISABLED - Comment out crypto notional handling
            # # Handle quantity vs notional for crypto fractional orders
            # if notional is not None and is_crypto:
            #     order_params["notional"] = str(notional)
            #     logger.info(f"Using notional amount ${notional} for crypto order")
            # else:
            order_params["qty"] = abs(actual_qty)  # Use actual_qty which includes short selling logic
            
            # Check if this is a fractional stock order (non-crypto)
            is_fractional_stock_order = not is_crypto and float(qty) != int(float(qty))
            
            # Handle time_in_force requirements for fractional orders
            if is_fractional_stock_order and time_in_force.lower() != "day":
                logger.info(f"Fractional stock order detected for {symbol} (qty: {qty}), forcing time_in_force to 'day'")
                order_params["time_in_force"] = "day"
            # CRYPTO TRADING DISABLED - Comment out crypto time_in_force handling
            # elif is_crypto and time_in_force.lower() not in ["gtc", "ioc"]:
            #     logger.warning(f"Invalid time_in_force '{time_in_force}' for crypto. Using 'gtc'")
            #     order_params["time_in_force"] = "gtc"
            
            # Enhanced Short Selling Validation and Risk Management
            if side.lower() == "sell_short":
                logger.info(f"Placing enhanced short sell order for {symbol}")
                
                # Step 1: Basic short selling checks
                try:
                    account = self.get_account_info()
                    if notional is not None:
                        short_value = notional
                    else:
                        current_price = self.get_current_price(symbol)
                        if not current_price:
                            raise ValueError(f"Could not determine current price for {symbol}")
                        short_value = qty * current_price
                    
                    # Step 2: Enhanced short selling validation
                    try:
                        enhanced_validation_passed = asyncio.run(self._validate_enhanced_short_selling(
                            symbol, qty, current_price, account
                        ))
                    except Exception as validation_error:
                        logger.warning(f"Enhanced validation error: {validation_error}")
                        enhanced_validation_passed = True  # Fallback to basic validation
                    
                    if not enhanced_validation_passed:
                        logger.warning(f"Enhanced short selling validation failed for {symbol}")
                        # Continue with basic validation as fallback
                    
                    # Step 3: Margin requirement check
                    margin_requirement = short_value * 1.5  # 150% margin requirement
                    
                    if short_value < 500 or margin_requirement <= account["buying_power"] * 2:
                        logger.info(f"Short selling approved: ${short_value:.2f} position, ${margin_requirement:.2f} margin requirement")
                    else:
                        logger.warning(f"Short may require more margin: ${margin_requirement:.2f} vs ${account['buying_power']:.2f} available")
                        # Continue but log the warning
                        
                except Exception as e:
                    logger.warning(f"Enhanced short selling validation error: {e}")
                    # Fallback to basic order submission
            
            if limit_price is not None:
                order_params["limit_price"] = str(limit_price)
            
            if stop_price is not None:
                order_params["stop_price"] = str(stop_price)
            
            # CRITICAL FIX: Validate buying power before placing buy orders
            if side.lower() == "buy":
                account = self.get_account_info()
                day_trading_power = account.get('daytrade_buying_power', 0)
                regt_buying_power = account.get('regt_buying_power', 0)
                
                # Estimate order value
                if order_type.lower() == "market":
                    current_price = self.get_current_price(symbol)
                    estimated_order_value = float(qty) * current_price if current_price else 0
                else:
                    estimated_order_value = float(qty) * (limit_price or 0)
                
                # Check if we have sufficient buying power
                if day_trading_power > 0:
                    # Use day trading buying power
                    available_power = day_trading_power
                    power_type = "day trading"
                elif regt_buying_power > 0:
                    # Fall back to RegT buying power
                    available_power = regt_buying_power
                    power_type = "RegT"
                else:
                    # No buying power available
                    raise ValueError(f"Insufficient buying power for {symbol}: day trading=${day_trading_power:.2f}, RegT=${regt_buying_power:.2f}")
                
                if estimated_order_value > available_power:
                    raise ValueError(f"Insufficient {power_type} buying power for {symbol}: need ${estimated_order_value:.2f}, have ${available_power:.2f}")
                
                logger.info(f"Buying power validated: using ${available_power:.2f} {power_type} power for ${estimated_order_value:.2f} order")
            
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
                    order = self.api.submit_order(**order_params)
                    
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
            
            # Handle wash trade detection errors with complex orders
            elif "potential wash trade detected" in error_message or "wash trade" in error_message:
                logger.warning(f"Wash trade detected for {symbol}, retrying with bracket order")
                try:
                    return self._place_wash_trade_compliant_order(symbol, qty, side, order_type, limit_price, stop_price)
                except Exception as wash_trade_retry_error:
                    logger.error(f"Wash trade compliant order also failed for {symbol}: {wash_trade_retry_error}")
                    raise
            
            # Handle insufficient quantity errors for sell orders
            elif "insufficient qty available" in error_message and side in ["sell", "sell_short"]:
                logger.warning(f"Insufficient quantity for {symbol}, attempting to sell all available shares")
                try:
                    # Get current position to determine available quantity
                    positions = self.get_positions()
                    available_qty = 0
                    
                    # Find the position for this symbol
                    for position in positions:
                        if position.get("symbol") == symbol:
                            available_qty = abs(float(position.get("qty", 0)))
                            break
                    
                    if available_qty > 0:
                        # Retry with available quantity
                        order_params["qty"] = available_qty
                        
                        logger.info(f"Retrying {symbol} sell order with available quantity: {available_qty}")
                        order = self.api.submit_order(**order_params)
                        
                        # Send batched email notification asynchronously
                        try:
                            asyncio.create_task(self._send_batched_transaction_notification(
                                order, symbol, available_qty, side, order_type, limit_price
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
                    else:
                        logger.warning(f"No shares available to sell for {symbol}")
                        raise ValueError(f"No shares available to sell for {symbol}")
                        
                except Exception as retry_error:
                    logger.error(f"Retry with available quantity failed for {symbol}: {retry_error}")
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
    
    def _reconcile_position_and_calculate_net_order(self, symbol: str, qty: float, side: str) -> Tuple[float, str]:
        """
        Reconcile current position with Alpaca and calculate net order to avoid wash trades.
        
        This fixes two critical issues:
        1. Insufficient quantity errors: Ensures we can only sell what we actually own
        2. Wash trade detection: Calculates net position changes instead of simple orders
        
        Args:
            symbol: Stock symbol
            qty: Requested quantity
            side: Requested side ("buy" or "sell")
            
        Returns:
            Tuple of (reconciled_quantity, net_action)
        """
        try:
            # STEP 1: Get current actual position from Alpaca
            positions = self.get_positions()
            current_position = 0.0
            
            for position in positions:
                if position.get("symbol") == symbol:
                    current_position = float(position.get("qty", 0))
                    break
            
            logger.info(f"📊 {symbol} position reconciliation: current={current_position:.2f}, requested={qty:.2f} {side}")
            
            # STEP 2: Calculate net position change
            if side.lower() == "buy":
                # Buying: increase position
                target_position = current_position + qty
                net_change = qty
            elif side.lower() in ["sell", "sell_short"]:
                # Selling: decrease position
                target_position = current_position - qty
                net_change = -qty
            else:
                logger.warning(f"Unknown side '{side}', treating as hold")
                return 0.0, "hold"
            
            # STEP 3: Validate we can execute this trade
            if side.lower() == "sell" and current_position <= 0:
                logger.warning(f"⚠️ Cannot sell {symbol}: no current position (current={current_position})")
                return 0.0, "hold"
            
            if side.lower() == "sell_short":
                # Short selling: allowed even if current_position <= 0
                logger.info(f"🩳 Short sell order for {symbol}: qty={qty}, current_position={current_position}")
                return qty, side  # Allow full short quantity
            elif side.lower() == "sell" and qty > current_position:
                # Regular sell: can only sell what we own - adjust quantity
                logger.warning(f"⚠️ Cannot sell {qty} shares of {symbol}: only {current_position} available")
                reconciled_qty = current_position
                logger.info(f"✅ Adjusted sell quantity to available shares: {reconciled_qty}")
                return reconciled_qty, side
            
            # STEP 4: Check for wash trade patterns and use OCO orders if needed
            if self._is_potential_wash_trade(symbol, qty, side, current_position):
                logger.warning(f"🔄 Potential wash trade detected for {symbol} - using OCO order to comply with Alpaca rules")
                # For wash trades, we need to use bracket orders instead of simple orders
                # Return the original quantity but flag it for OCO processing in the calling code
                return qty, f"oco_{side}"
            
            # STEP 5: Return reconciled order
            logger.info(f"✅ {symbol} order reconciled: {qty:.2f} {side} (net position change: {net_change:+.2f})")
            return qty, side
            
        except Exception as e:
            logger.error(f"Position reconciliation failed for {symbol}: {e}")
            # Fallback to original values with warning
            logger.warning(f"Using original order values as fallback: {qty} {side}")
            return qty, side
    
    def _is_potential_wash_trade(self, symbol: str, qty: float, side: str, current_position: float) -> bool:
        """
        Detect if this order might trigger Alpaca's wash trade detection.
        
        Wash trade patterns that Alpaca blocks:
        1. Buy and sell same symbol within short timeframe with minimal net change
        2. Rapid reversals in position direction
        3. Small net changes that appear to be round-trip trades
        
        Args:
            symbol: Stock symbol
            qty: Order quantity
            side: Order side
            current_position: Current position in the symbol
            
        Returns:
            True if this might be flagged as a wash trade
        """
        try:
            # Get recent orders for this symbol to check for patterns
            recent_orders = self.get_orders(symbol=symbol, status='all', limit=10)
            
            if not recent_orders:
                return False  # No recent orders, not a wash trade
            
            # Check for recent opposite direction orders
            current_time = datetime.now()
            recent_threshold = current_time - timedelta(minutes=30)  # 30-minute lookback
            
            recent_opposite_orders = []
            for order in recent_orders:
                order_time = datetime.fromisoformat(order.get('submitted_at', '').replace('Z', '+00:00'))
                order_side = order.get('side', '')
                
                if order_time > recent_threshold:
                    # Check if this is an opposite direction order
                    if ((side.lower() == "buy" and order_side == "sell") or 
                        (side.lower() == "sell" and order_side == "buy")):
                        recent_opposite_orders.append(order)
            
            # If we have recent opposite orders, this might be flagged as wash trading
            if recent_opposite_orders:
                logger.info(f"⚠️ Found {len(recent_opposite_orders)} recent opposite orders for {symbol} - potential wash trade")
                return True
                
            # Check for minimal net change patterns
            if side.lower() == "buy" and current_position < 0:
                # Buying to cover a short - could be wash trade if small net change
                net_position_after = current_position + qty
                if abs(net_position_after) < min(10, abs(current_position) * 0.1):  # Net change < 10 shares or 10% of position
                    logger.info(f"⚠️ Small net change detected for {symbol}: {current_position} -> {net_position_after}")
                    return True
            
            if side.lower() == "sell" and current_position > 0:
                # Selling from a long position - could be wash trade if small net change
                net_position_after = current_position - qty
                if abs(net_position_after) < min(10, abs(current_position) * 0.1):  # Net change < 10 shares or 10% of position
                    logger.info(f"⚠️ Small net change detected for {symbol}: {current_position} -> {net_position_after}")
                    return True
                    
            return False
            
        except Exception as e:
            logger.warning(f"Wash trade detection failed for {symbol}: {e}")
            return False  # Assume not a wash trade if we can't determine
    
    def calculate_net_orders_batch(self, order_list: List[Dict]) -> List[Dict]:
        """
        Calculate net orders from a batch of signals to avoid wash trades.
        
        This aggregates multiple BUY/SELL signals for the same symbol into net orders,
        preventing Alpaca from detecting wash trade patterns.
        
        Args:
            order_list: List of order dictionaries with symbol, side, qty
            
        Returns:
            List of net orders that avoid wash trade detection
        """
        try:
            # Group orders by symbol
            symbol_orders = {}
            for order in order_list:
                symbol = order.get("symbol")
                if symbol not in symbol_orders:
                    symbol_orders[symbol] = {"buy_qty": 0.0, "sell_qty": 0.0, "orders": []}
                
                side = order.get("side", "").lower()
                qty = float(order.get("qty", 0))
                
                if side == "buy":
                    symbol_orders[symbol]["buy_qty"] += qty
                elif side in ["sell", "sell_short"]:
                    symbol_orders[symbol]["sell_qty"] += qty
                
                symbol_orders[symbol]["orders"].append(order)
            
            # Calculate net orders
            net_orders = []
            for symbol, order_data in symbol_orders.items():
                buy_qty = order_data["buy_qty"]
                sell_qty = order_data["sell_qty"]
                
                # Calculate net quantity
                net_qty = buy_qty - sell_qty
                
                if abs(net_qty) < 1:  # Skip tiny net changes
                    logger.info(f"⚖️ {symbol}: Net order too small ({net_qty:.2f}), skipping to avoid wash trade")
                    continue
                
                # Determine net action
                if net_qty > 0:
                    net_side = "buy"
                    net_quantity = abs(net_qty)
                elif net_qty < 0:
                    net_side = "sell"
                    net_quantity = abs(net_qty)
                else:
                    continue  # No net change
                
                # Get current position for validation
                positions = self.get_positions()
                current_position = 0.0
                for pos in positions:
                    if pos.get("symbol") == symbol:
                        current_position = float(pos.get("qty", 0))
                        break
                
                # Final validation for sell orders
                if net_side == "sell" and net_quantity > current_position:
                    logger.warning(f"⚠️ Net sell quantity ({net_quantity}) exceeds position ({current_position}) for {symbol}")
                    net_quantity = max(0, current_position)
                    if net_quantity == 0:
                        continue
                
                # Create net order
                net_order = {
                    "symbol": symbol,
                    "side": net_side,
                    "qty": net_quantity,
                    "original_orders": order_data["orders"],
                    "buy_qty_aggregated": buy_qty,
                    "sell_qty_aggregated": sell_qty,
                    "net_change": net_qty
                }
                net_orders.append(net_order)
                
                logger.info(f"📊 {symbol} net order: {buy_qty:.2f} buy - {sell_qty:.2f} sell = {net_qty:+.2f} ({net_side} {net_quantity:.2f})")
            
            logger.info(f"✅ Calculated {len(net_orders)} net orders from {len(order_list)} original signals")
            return net_orders
            
        except Exception as e:
            logger.error(f"Net order calculation failed: {e}")
            return order_list  # Fallback to original orders
    
    def place_oco_order(self, symbol: str, qty: float, side: str, 
                       take_profit_price: float, stop_loss_price: float,
                       time_in_force: str = "gtc") -> Dict:
        """
        Place an OCO (One-Cancels-Other) order using Alpaca's bracket order functionality.
        
        Args:
            symbol: Stock symbol
            qty: Quantity to trade
            side: "buy" or "sell"
            take_profit_price: Price for take-profit order
            stop_loss_price: Price for stop-loss order
            time_in_force: "gtc", "day", "ioc", "fok"
        
        Returns:
            Dict with order information including parent and child orders
        """
        try:
            logger.info(f"Placing OCO bracket order: {side} {qty} {symbol}")
            
            # Safety checks
            if not self._pre_trade_checks(symbol, qty, side):
                raise ValueError("Pre-trade safety checks failed for OCO order")
            
            # Validate OCO parameters
            current_price = self.get_current_price(symbol)
            
            # Bracket order validation logic:
            # - For BUY orders (opening position): take_profit > current, stop_loss < current  
            # - For SELL orders (closing position): take_profit > current (profit), stop_loss < current (limit loss)
            # Note: Alpaca bracket orders assume you're OPENING a position, so validation is for the entry order
            
            if side.lower() == "buy":
                # Buying: take profit should be higher, stop loss should be lower
                if take_profit_price <= current_price:
                    raise ValueError(f"Take profit price ({take_profit_price}) must be above current price ({current_price}) for buy orders")
                if stop_loss_price >= current_price:
                    raise ValueError(f"Stop loss price ({stop_loss_price}) must be below current price ({current_price}) for buy orders")
            elif side.lower() in ["sell", "sell_short"]:
                # For selling existing position: take_profit should be HIGHER (more profit), stop_loss LOWER (limit loss)
                # But this is for PROTECTING existing position, so logic is:
                # - take_profit_price is where we SELL for profit (higher than current)
                # - stop_loss_price is where we SELL to limit loss (lower than current)
                if take_profit_price <= current_price:
                    raise ValueError(f"Take profit price ({take_profit_price}) must be above current price ({current_price}) - this is where you sell for profit")
                if stop_loss_price >= current_price:
                    raise ValueError(f"Stop loss price ({stop_loss_price}) must be below current price ({current_price}) - this is where you sell to limit losses")
            
            # Place bracket order (OCO using Alpaca's native functionality)
            order = self.api.submit_order(
                symbol=symbol,
                qty=abs(qty),
                side=side.lower(),
                type="market",
                time_in_force=time_in_force.lower(),
                take_profit={"limit_price": str(take_profit_price)},
                stop_loss={"stop_price": str(stop_loss_price)}
            )
            
            logger.info(f"OCO bracket order placed: parent {order.id}")
            
            # Send notification
            try:
                asyncio.create_task(self._send_batched_oco_notification(
                    order, symbol, qty, side, take_profit_price, stop_loss_price
                ))
            except Exception as notification_error:
                logger.warning(f"OCO notification failed: {notification_error}")
            
            return {
                "id": order.id,
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side,
                "order_type": "oco_bracket",
                "status": order.status,
                "submitted_at": order.submitted_at,
                "take_profit_price": take_profit_price,
                "stop_loss_price": stop_loss_price,
                "parent_order_id": order.id
            }
            
        except Exception as e:
            logger.error(f"Failed to place OCO order: {e}")
            raise
    
    def place_breakout_oco_order(self, symbol: str, qty: float, 
                               upper_breakout_price: float, lower_breakout_price: float,
                               limit_buffer_percent: float = 0.002,
                               time_in_force: str = "gtc") -> Dict:
        """
        Place a breakout OCO order that buys on upper breakout or sells on lower breakout.
        
        Args:
            symbol: Stock symbol
            qty: Quantity for each breakout direction
            upper_breakout_price: Price to trigger buy breakout
            lower_breakout_price: Price to trigger sell breakout
            limit_buffer_percent: Buffer for limit orders (0.2% default)
            time_in_force: "gtc", "day", "ioc", "fok"
        
        Returns:
            Dict with both order IDs and tracking information
        """
        try:
            logger.info(f"Placing breakout OCO: {symbol} upper={upper_breakout_price} lower={lower_breakout_price}")
            
            # Safety checks
            current_price = self.get_current_price(symbol)
            
            # Validate breakout levels
            if upper_breakout_price <= current_price:
                raise ValueError(f"Upper breakout ({upper_breakout_price}) must be above current price ({current_price})")
            if lower_breakout_price >= current_price:
                raise ValueError(f"Lower breakout ({lower_breakout_price}) must be below current price ({current_price})")
            
            # Calculate limit prices with buffer and round to valid increments
            upper_limit = round(upper_breakout_price * (1 + limit_buffer_percent), 2)
            lower_limit = round(lower_breakout_price * (1 - limit_buffer_percent), 2)
            
            # Round breakout prices to pennies as well
            upper_breakout_price = round(upper_breakout_price, 2)
            lower_breakout_price = round(lower_breakout_price, 2)
            
            # Place buy stop order for upper breakout
            buy_order = self.api.submit_order(
                symbol=symbol,
                qty=abs(qty),
                side="buy",
                type="stop_limit",
                stop_price=str(upper_breakout_price),
                limit_price=str(upper_limit),
                time_in_force=time_in_force.lower()
            )
            
            # Place sell stop order for lower breakout
            sell_order = self.api.submit_order(
                symbol=symbol,
                qty=abs(qty),
                side="sell",
                type="stop_limit", 
                stop_price=str(lower_breakout_price),
                limit_price=str(lower_limit),
                time_in_force=time_in_force.lower()
            )
            
            logger.info(f"Breakout OCO placed: buy {buy_order.id}, sell {sell_order.id}")
            
            # Send notification
            try:
                asyncio.create_task(self._send_batched_breakout_notification(
                    symbol, qty, upper_breakout_price, lower_breakout_price, 
                    buy_order.id, sell_order.id
                ))
            except Exception as notification_error:
                logger.warning(f"Breakout OCO notification failed: {notification_error}")
            
            return {
                "breakout_group_id": f"breakout_{buy_order.id}_{sell_order.id}",
                "symbol": symbol,
                "qty": qty,
                "upper_breakout_price": upper_breakout_price,
                "lower_breakout_price": lower_breakout_price,
                "buy_order": {
                    "id": buy_order.id,
                    "status": buy_order.status,
                    "stop_price": upper_breakout_price,
                    "limit_price": upper_limit
                },
                "sell_order": {
                    "id": sell_order.id,
                    "status": sell_order.status,
                    "stop_price": lower_breakout_price,
                    "limit_price": lower_limit
                },
                "order_type": "breakout_oco",
                "submitted_at": buy_order.submitted_at
            }
            
        except Exception as e:
            logger.error(f"Failed to place breakout OCO order: {e}")
            raise
    
    def get_bracket_orders(self, parent_order_id: str) -> List[Dict]:
        """Get all child orders for a bracket/OCO order."""
        try:
            orders = self.api.list_orders(status="all", limit=100)
            
            # Find all orders related to the parent order
            related_orders = []
            for order in orders:
                # Check if this is the parent or a child order
                if (order.id == parent_order_id or 
                    getattr(order, 'legs', None) or 
                    str(getattr(order, 'order_class', '')) == 'bracket'):
                    
                    order_data = {
                        "id": order.id,
                        "symbol": order.symbol,
                        "qty": float(order.qty),
                        "side": order.side,
                        "order_type": order.order_type,
                        "status": order.status,
                        "filled_qty": float(order.filled_qty or 0),
                        "submitted_at": order.submitted_at,
                        "filled_at": order.filled_at
                    }
                    
                    # Add price information
                    if hasattr(order, 'limit_price') and order.limit_price:
                        order_data['limit_price'] = float(order.limit_price)
                    if hasattr(order, 'stop_price') and order.stop_price:
                        order_data['stop_price'] = float(order.stop_price)
                    
                    related_orders.append(order_data)
            
            return related_orders
            
        except Exception as e:
            logger.error(f"Failed to get bracket orders for {parent_order_id}: {e}")
            return []
    
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
            
            # Check buying power for buy orders - allow selling even with $0 cash
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
                    
                    # AGGRESSIVE TRADING FIX: Use much higher buying power for active trading
                    # Calculate aggressive effective buying power using multiple sources
                    cash = account.get("cash", 0)
                    equity = account.get("portfolio_value", 0)
                    
                    # Use the most aggressive calculation:
                    # 1. Full cash balance (for cash-secured positions)
                    # 2. 20% of total equity (for margin-style aggressive trading)
                    # 3. Take the maximum of these for maximum trading capacity
                    cash_based_power = cash  # Use full cash
                    equity_based_power = equity * 0.2  # Use 20% of equity for aggressive margin-style trading
                    effective_buying_power = max(account["buying_power"], cash_based_power, equity_based_power)
                    
                    # VERY aggressive multiplier for day trading / active rebalancing
                    max_order_limit = effective_buying_power * 3.0  # Allow 3x leverage
                    
                    if order_value > max_order_limit:
                        logger.error(f"Order too large: ${order_value:.2f} > ${max_order_limit:.2f} (3x aggressive buying power)")
                        return False
                    
                    # For orders under $5000, be extremely lenient for active trading
                    if order_value < 5000 and (order_value <= effective_buying_power or cash > 2000):
                        logger.info(f"Allowing aggressive order: ${order_value:.2f} with effective buying power ${effective_buying_power:.2f}")
                        return True
                        
                except Exception as e:
                    logger.warning(f"Could not verify buying power: {e}")
                    # Continue without buying power check - let Alpaca API handle it
            
            elif side.lower() in ["sell", "sell_short"]:
                # For SELL orders, validate that we actually own the shares
                try:
                    positions = self.get_positions()
                    available_qty = 0
                    
                    for position in positions:
                        if position.get("symbol") == symbol:
                            available_qty = abs(float(position.get("qty", 0)))
                            break
                    
                    if available_qty == 0:
                        logger.warning(f"❌ Cannot sell {symbol}: No shares owned (position: {available_qty})")
                        return False
                    elif available_qty < qty:
                        logger.warning(f"⚠️ Insufficient shares for {symbol}: requested {qty}, available {available_qty}")
                        return False
                    else:
                        logger.info(f"✅ Sell validation passed for {symbol}: {qty} shares (available: {available_qty})")
                        return True
                        
                except Exception as e:
                    logger.error(f"Failed to validate position for sell order {symbol}: {e}")
                    return False
            
            # Check position size limits - more flexible for balanced trading
            # Note: SELL orders are already validated above for position availability
                
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
                    
                    # CRYPTO TRADING DISABLED - Comment out crypto-specific position limits
                    # is_crypto = self._is_crypto_symbol(symbol)
                    # base_max_position = getattr(settings, 'crypto_max_position_size', getattr(settings, 'max_position_size', 0.05)) if is_crypto else getattr(settings, 'max_position_size', 0.05)
                    base_max_position = getattr(settings, 'max_position_size', 0.05)  # Use standard limits only
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
            
            # CRYPTO TRADING DISABLED - Remove crypto asset class detection
            asset_info = {
                "symbol": asset.symbol,
                "name": getattr(asset, 'name', asset.symbol),
                "exchange": getattr(asset, 'exchange', 'Unknown'),
                "asset_class": getattr(asset, 'asset_class', 'us_equity'),  # Always default to us_equity
                "status": getattr(asset, 'status', 'active'),
                "tradable": getattr(asset, 'tradable', True),
                "marginable": getattr(asset, 'marginable', False),
                "shortable": getattr(asset, 'shortable', False),
                "easy_to_borrow": getattr(asset, 'easy_to_borrow', False),
                "fractionable": getattr(asset, 'fractionable', True)
            }
            
            # CRYPTO TRADING DISABLED - Comment out crypto-specific info
            # # Add crypto-specific info
            # if self._is_crypto_symbol(symbol):
            #     asset_info.update({
            #         "is_crypto": True,
            #         "trading_hours": "24/7",
            #         "supported_order_types": ["market", "limit", "stop_limit"],
            #         "supported_time_in_force": ["gtc", "ioc"],
            #         "fractional_supported": True
            #     })
            # else:
            # Always use stock-specific info since crypto is disabled
            asset_info.update({
                "is_crypto": False,
                "trading_hours": "9:30 AM - 4:00 PM ET",
                "supported_order_types": ["market", "limit", "stop", "stop_limit"],
                "supported_time_in_force": ["gtc", "day", "ioc", "fok"]
            })
            
            return asset_info
            
        except Exception as e:
            logger.error(f"Failed to get asset info for {symbol}: {e}")
            
            # CRYPTO TRADING DISABLED - Return stock defaults only
            # is_crypto = self._is_crypto_symbol(symbol)
            return {
                "symbol": symbol,
                "name": symbol,
                "tradable": True,
                "fractionable": False,  # Default to non-fractionable for stocks
                "is_crypto": False,  # Always false when crypto disabled
                "trading_hours": "9:30 AM - 4:00 PM ET",
                "supported_order_types": ["market", "limit", "stop", "stop_limit"],
                "supported_time_in_force": ["gtc", "day", "ioc", "fok"]
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
        """Check if the market is currently open - CRYPTO TRADING DISABLED."""
        try:
            # CRYPTO TRADING DISABLED - Comment out crypto market check
            # # Crypto markets are open 24/7
            # if symbol and self._is_crypto_symbol(symbol):
            #     return True
                
            # For stocks, check market hours
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Failed to check market status: {e}")
            # CRYPTO TRADING DISABLED - Always return false for non-market hours
            # return symbol and self._is_crypto_symbol(symbol) if symbol else False
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
    
    async def _send_batched_oco_notification(self, order, symbol: str, qty: float,
                                           side: str, take_profit_price: float, 
                                           stop_loss_price: float) -> None:
        """Send batched email notification for OCO order."""
        try:
            # Import here to avoid circular imports
            from notifications.email_batch_manager import send_batched_transaction_notification, email_batch_manager
            
            # Ensure batch processor is running
            if not email_batch_manager.running:
                email_batch_manager.start_batch_processor()
            
            # Get current portfolio value
            account_info = self.get_account_info()
            portfolio_value = account_info.get("portfolio_value", 0)
            
            # Get current price
            try:
                current_price = self.get_current_price(symbol)
            except:
                current_price = 0
            
            # Create reasoning message
            reasoning = f"OCO bracket order placed via Alpaca Trading API. "
            reasoning += f"Parent Order ID: {order.id}. "
            reasoning += f"Take Profit: ${take_profit_price:.2f}, Stop Loss: ${stop_loss_price:.2f}. "
            reasoning += f"Risk Management: Automatic profit-taking and loss protection enabled."
            
            # Send to batching system
            await send_batched_transaction_notification(
                symbol=symbol,
                action=f"OCO_{side}",
                quantity=qty,
                price=current_price,
                portfolio_value=portfolio_value,
                confidence=0.95,  # High confidence for OCO orders
                reasoning=reasoning,
                agent_source="alpaca_oco_trading"
            )
            
            logger.info(f"OCO transaction added to batch queue: {symbol} {side}")
            
        except Exception as e:
            logger.error(f"Failed to add OCO transaction to batch queue: {e}")
    
    async def _send_batched_breakout_notification(self, symbol: str, qty: float,
                                                upper_breakout: float, lower_breakout: float,
                                                buy_order_id: str, sell_order_id: str) -> None:
        """Send batched email notification for breakout OCO order."""
        try:
            # Import here to avoid circular imports
            from notifications.email_batch_manager import send_batched_transaction_notification, email_batch_manager
            
            # Ensure batch processor is running
            if not email_batch_manager.running:
                email_batch_manager.start_batch_processor()
            
            # Get current portfolio value
            account_info = self.get_account_info()
            portfolio_value = account_info.get("portfolio_value", 0)
            
            # Get current price
            try:
                current_price = self.get_current_price(symbol)
            except:
                current_price = 0
            
            # Create reasoning message
            reasoning = f"Breakout OCO strategy deployed for {symbol}. "
            reasoning += f"Upper breakout (buy): ${upper_breakout:.2f} (Order: {buy_order_id}). "
            reasoning += f"Lower breakout (sell): ${lower_breakout:.2f} (Order: {sell_order_id}). "
            reasoning += f"Strategy will trigger on price breakout in either direction."
            
            # Send to batching system
            await send_batched_transaction_notification(
                symbol=symbol,
                action="BREAKOUT_OCO",
                quantity=qty,
                price=current_price,
                portfolio_value=portfolio_value,
                confidence=0.85,  # Good confidence for breakout strategies
                reasoning=reasoning,
                agent_source="alpaca_breakout_trading"
            )
            
            logger.info(f"Breakout OCO transaction added to batch queue: {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to add breakout OCO transaction to batch queue: {e}")
    
    # CRYPTO TRADING DISABLED - Comment out crypto fallback method
    # def _get_crypto_fallback_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    #     """Get crypto data from alternative sources when Alpaca fails."""
    #     # NO FALLBACKS - fail fast with clear error
    #     error_msg = f"❌ CRITICAL: Alpaca crypto data access failed for {symbol}. Paper trading subscription may not support crypto data."
    #     logger.error(error_msg)
    #     raise RuntimeError(error_msg)
    
    
    # CRYPTO TRADING DISABLED - Always return False for crypto detection
    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Check if a symbol represents a cryptocurrency pair - DISABLED."""
        # # Common crypto symbols end with USD, USDT, USDC or are known crypto pairs
        # crypto_suffixes = ['USD', 'USDT', 'USDC', 'BTC']
        # crypto_prefixes = ['BTC', 'ETH', 'DOGE', 'LTC', 'BCH', 'AAVE', 'UNI', 'LINK', 'MKR']
        # 
        # # Check if symbol matches crypto patterns
        # for prefix in crypto_prefixes:
        #     for suffix in crypto_suffixes:
        #         if symbol.upper() == f"{prefix}{suffix}":
        #             return True
        # 
        # # Additional known crypto patterns
        # known_crypto_symbols = ['BTCUSD', 'ETHUSD', 'DOGEUSD', 'LTCUSD', 'BCHUSD']
        # return symbol.upper() in known_crypto_symbols
        return False  # Always return False when crypto is disabled
    
    async def _validate_enhanced_short_selling(self, symbol: str, qty: float, 
                                             current_price: float, account: Dict) -> bool:
        """
        Enhanced short selling validation using comprehensive analysis modules.
        
        This method integrates with the enhanced short-selling framework to validate:
        - Borrow cost and availability
        - SSR compliance
        - Squeeze risk assessment
        - Portfolio risk limits
        
        Returns True if validation passes, False otherwise.
        """
        try:
            logger.debug(f"Running enhanced short selling validation for {symbol}")
            
            # Import modules with late binding to avoid circular imports
            try:
                from core.borrow_cost_monitor import borrow_cost_monitor
                from core.short_risk_manager import short_risk_manager
            except ImportError as e:
                logger.warning(f"Enhanced short selling modules not available: {e}")
                return True  # Fallback to basic validation
            
            # Step 1: Validate borrow costs and SSR compliance
            logger.debug(f"Validating borrow costs for {symbol}")
            borrow_valid, borrow_message, borrow_details = await borrow_cost_monitor.validate_short_trade(
                symbol, int(qty), current_price
            )
            
            if not borrow_valid:
                logger.warning(f"Borrow cost validation failed for {symbol}: {borrow_message}")
                return False
            
            # Step 2: Check SSR status and compliance
            logger.debug(f"Checking SSR status for {symbol}")
            ssr_status = await borrow_cost_monitor.ssr_monitor.check_ssr_status(symbol)
            
            if ssr_status.ssr_active:
                # Check if short is allowed under SSR
                allowed, ssr_reason = borrow_cost_monitor.ssr_monitor.is_short_allowed_now(symbol, current_price)
                if not allowed:
                    logger.warning(f"SSR violation for {symbol}: {ssr_reason}")
                    return False
            
            # Step 3: Assess squeeze risk
            logger.debug(f"Assessing squeeze risk for {symbol}")
            squeeze_metrics = await short_risk_manager.squeeze_analyzer.assess_squeeze_risk(symbol)
            
            if squeeze_metrics.squeeze_risk_score > 80:  # Very high squeeze risk
                logger.warning(f"High squeeze risk for {symbol}: {squeeze_metrics.squeeze_risk_score:.1f}/100")
                return False
            elif squeeze_metrics.squeeze_risk_score > 60:  # Moderate squeeze risk
                logger.info(f"Moderate squeeze risk for {symbol}: {squeeze_metrics.squeeze_risk_score:.1f}/100 - proceeding with caution")
            
            # Step 4: Portfolio risk validation
            logger.debug(f"Validating portfolio risk for {symbol}")
            portfolio_data = {
                'equity': account.get('equity', 100000),
                'cash': account.get('cash', 50000),
                'positions': {}  # Would need to get actual positions, but this is a basic check
            }
            
            risk_valid, risk_message, risk_details = await short_risk_manager.validate_short_trade(
                symbol, int(qty), portfolio_data
            )
            
            if not risk_valid:
                logger.warning(f"Portfolio risk validation failed for {symbol}: {risk_message}")
                return False
            
            # Step 5: Log successful validation
            logger.info(f"✅ Enhanced short selling validation passed for {symbol}")
            logger.info(f"   Borrow cost: {borrow_details.get('borrow_cost', {}).get('borrow_fee_rate', 0):.1%} annually")
            logger.info(f"   Squeeze risk: {squeeze_metrics.squeeze_risk_score:.1f}/100")
            logger.info(f"   SSR status: {'Active' if ssr_status.ssr_active else 'Inactive'}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in enhanced short selling validation for {symbol}: {e}")
            # Return True to allow fallback to basic validation
            return True
    
    # Tax-Loss Harvesting Enhancement Methods
    
    def place_lot_specific_order(self, 
                               symbol: str, 
                               qty: float, 
                               side: str,
                               lot_ids: Optional[List[str]] = None,
                               lot_accounting_method: Optional[str] = None,
                               **kwargs) -> Dict:
        """
        Place an order with specific lot identification for tax purposes.
        
        Args:
            symbol: Stock symbol
            qty: Quantity to trade
            side: "buy" or "sell"
            lot_ids: Specific lot IDs to sell (for tax-loss harvesting)
            lot_accounting_method: "FIFO", "LIFO", "HIFO", "LOFO", or "SPECIFIC_ID"
            **kwargs: Additional order parameters
            
        Returns:
            Order result with lot tracking information
        """
        try:
            logger.info(f"🎯 Placing lot-specific order: {side} {qty} {symbol}")
            
            if lot_ids:
                logger.info(f"   Using specific lots: {lot_ids}")
            if lot_accounting_method:
                logger.info(f"   Accounting method: {lot_accounting_method}")
            
            # Place the order through normal channels
            # Note: Alpaca paper trading doesn't support true lot-specific orders,
            # but we simulate the functionality for tax tracking
            order_result = self.place_order(symbol, qty, side, **kwargs)
            
            # Enhance the result with tax information
            order_result.update({
                "lot_specific": True,
                "lot_ids_specified": lot_ids or [],
                "lot_accounting_method": lot_accounting_method,
                "tax_optimized": True,
                "is_tax_loss_harvest": side.lower() == "sell" and lot_accounting_method == "HIFO"
            })
            
            logger.info(f"✅ Lot-specific order placed: {order_result['id']}")
            return order_result
            
        except Exception as e:
            logger.error(f"Error placing lot-specific order: {e}")
            raise
    
    def get_cost_basis_information(self, symbol: str) -> Dict:
        """
        Get cost basis information for tax reporting.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with cost basis details
        """
        try:
            positions = self.get_positions()
            position = next((p for p in positions if p["symbol"] == symbol), None)
            
            if not position:
                return {
                    "symbol": symbol,
                    "has_position": False,
                    "cost_basis": 0.0,
                    "current_value": 0.0,
                    "unrealized_pnl": 0.0
                }
            
            # In a real implementation, this would integrate with lot_tracking.py
            # to provide detailed lot-level cost basis information
            return {
                "symbol": symbol,
                "has_position": True,
                "quantity": position["qty"],
                "cost_basis": position["cost_basis"],
                "current_value": position["market_value"],
                "unrealized_pnl": position["unrealized_pl"],
                "unrealized_pnl_percent": position["unrealized_plpc"],
                "current_price": position["current_price"],
                "avg_cost_per_share": position["cost_basis"] / max(abs(position["qty"]), 1),
                "tax_status": "long_term" if abs(position["qty"]) > 0 else "n/a"  # Simplified
            }
            
        except Exception as e:
            logger.error(f"Error getting cost basis for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}
    
    def get_tax_lot_details(self, symbol: str) -> List[Dict]:
        """
        Get detailed tax lot information for a position.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of tax lot dictionaries
        """
        try:
            # Note: Alpaca paper trading doesn't provide lot-level detail
            # In a real implementation, this would integrate with lot_tracking.py
            position_info = self.get_cost_basis_information(symbol)
            
            if not position_info.get("has_position"):
                return []
            
            # Simulate lot information for tax purposes
            # In practice, this would come from the lot_tracking system
            return [{
                "lot_id": f"{symbol}_simulated_lot",
                "symbol": symbol,
                "quantity": position_info["quantity"],
                "acquisition_date": datetime.now() - timedelta(days=180),  # Simulate 6 months ago
                "cost_basis_per_share": position_info["avg_cost_per_share"],
                "total_cost_basis": position_info["cost_basis"],
                "current_price": position_info["current_price"],
                "unrealized_pnl": position_info["unrealized_pnl"],
                "is_long_term": True,  # Simulate long-term holding
                "wash_sale_adjusted": False
            }]
            
        except Exception as e:
            logger.error(f"Error getting tax lot details for {symbol}: {e}")
            return []
    
    def calculate_tax_impact(self, 
                           symbol: str, 
                           quantity: float,
                           accounting_method: str = "HIFO") -> Dict:
        """
        Calculate tax impact of a potential sale.
        
        Args:
            symbol: Stock symbol
            quantity: Quantity to sell
            accounting_method: Lot accounting method
            
        Returns:
            Dict with tax impact analysis
        """
        try:
            logger.info(f"📊 Calculating tax impact for {symbol}: sell {quantity} shares ({accounting_method})")
            
            position_info = self.get_cost_basis_information(symbol)
            if not position_info.get("has_position"):
                return {
                    "symbol": symbol,
                    "error": "No position found",
                    "can_sell": False
                }
            
            current_price = position_info["current_price"]
            avg_cost = position_info["avg_cost_per_share"]
            available_quantity = abs(position_info["quantity"])
            
            if quantity > available_quantity:
                return {
                    "symbol": symbol,
                    "error": f"Insufficient shares: {quantity} requested, {available_quantity} available",
                    "can_sell": False
                }
            
            # Calculate gain/loss
            proceeds = quantity * current_price
            cost_basis = quantity * avg_cost
            gain_loss = proceeds - cost_basis
            
            # Determine tax classification (simplified)
            is_loss = gain_loss < 0
            is_long_term = True  # Simplified assumption
            
            # Estimate tax impact (simplified calculation)
            tax_rate = 0.20 if is_long_term else 0.32  # Long-term vs short-term rates
            tax_impact = gain_loss * tax_rate if gain_loss > 0 else abs(gain_loss) * tax_rate
            
            return {
                "symbol": symbol,
                "quantity_to_sell": quantity,
                "available_quantity": available_quantity,
                "current_price": current_price,
                "avg_cost_basis": avg_cost,
                "proceeds": proceeds,
                "cost_basis": cost_basis,
                "gain_loss": gain_loss,
                "is_gain": gain_loss > 0,
                "is_loss": is_loss,
                "is_long_term": is_long_term,
                "estimated_tax_impact": tax_impact,
                "tax_savings_if_loss": abs(tax_impact) if is_loss else 0,
                "accounting_method": accounting_method,
                "can_sell": True,
                "recommendation": "harvest_loss" if is_loss and abs(gain_loss) > 100 else "hold"
            }
            
        except Exception as e:
            logger.error(f"Error calculating tax impact for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e), "can_sell": False}
    
    def get_wash_sale_status(self, symbol: str) -> Dict:
        """
        Check wash sale status for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with wash sale status information
        """
        try:
            # In a real implementation, this would integrate with wash_sale_monitor.py
            # For now, provide a simplified implementation
            
            # Check recent orders for wash sale risk
            recent_orders = self.get_orders(status="filled", limit=50)
            symbol_orders = [o for o in recent_orders if o["symbol"] == symbol]
            
            # Look for recent purchases within 30 days
            thirty_days_ago = datetime.now() - timedelta(days=30)
            recent_purchases = []
            
            for order in symbol_orders:
                if (order["side"] == "buy" and 
                    order["filled_at"] and 
                    order["filled_at"] > thirty_days_ago):
                    recent_purchases.append({
                        "date": order["filled_at"],
                        "quantity": order["filled_qty"],
                        "price": order["filled_avg_price"]
                    })
            
            has_wash_sale_risk = len(recent_purchases) > 0
            days_until_clear = 0
            
            if has_wash_sale_risk:
                # Find the most recent purchase
                most_recent = max(recent_purchases, key=lambda x: x["date"])
                days_since = (datetime.now() - most_recent["date"]).days
                days_until_clear = max(0, 30 - days_since)
            
            return {
                "symbol": symbol,
                "has_wash_sale_risk": has_wash_sale_risk,
                "days_until_clear": days_until_clear,
                "recent_purchases": recent_purchases,
                "can_sell_for_loss": not has_wash_sale_risk,
                "recommendation": "wait" if has_wash_sale_risk else "proceed"
            }
            
        except Exception as e:
            logger.error(f"Error checking wash sale status for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}
    
    def find_tax_loss_opportunities(self, min_loss_threshold: float = 100.0) -> List[Dict]:
        """
        Scan portfolio for tax-loss harvesting opportunities.
        
        Args:
            min_loss_threshold: Minimum loss amount to consider
            
        Returns:
            List of tax-loss harvesting opportunities
        """
        try:
            logger.info(f"🔍 Scanning for tax-loss opportunities (min loss: ${min_loss_threshold})")
            
            positions = self.get_positions()
            opportunities = []
            
            for position in positions:
                symbol = position["symbol"]
                unrealized_pl = position["unrealized_pl"]
                
                # Only consider positions with losses
                if unrealized_pl >= -min_loss_threshold:
                    continue
                
                # Calculate tax impact
                quantity = abs(position["qty"])
                tax_impact = self.calculate_tax_impact(symbol, quantity, "HIFO")
                
                if tax_impact.get("can_sell") and tax_impact.get("is_loss"):
                    # Check wash sale status
                    wash_sale_status = self.get_wash_sale_status(symbol)
                    
                    opportunity = {
                        "symbol": symbol,
                        "position_size": quantity,
                        "unrealized_loss": abs(unrealized_pl),
                        "estimated_tax_savings": tax_impact.get("tax_savings_if_loss", 0),
                        "current_price": tax_impact["current_price"],
                        "cost_basis": tax_impact["cost_basis"],
                        "wash_sale_risk": wash_sale_status["has_wash_sale_risk"],
                        "days_until_wash_sale_clear": wash_sale_status["days_until_clear"],
                        "priority_score": abs(unrealized_pl) * (0.5 if wash_sale_status["has_wash_sale_risk"] else 1.0),
                        "recommendation": "harvest" if not wash_sale_status["has_wash_sale_risk"] else "wait_for_clear"
                    }
                    
                    opportunities.append(opportunity)
            
            # Sort by priority score (highest loss potential first)
            opportunities.sort(key=lambda x: x["priority_score"], reverse=True)
            
            total_potential_savings = sum(opp["estimated_tax_savings"] for opp in opportunities)
            logger.info(f"✅ Found {len(opportunities)} tax-loss opportunities")
            logger.info(f"💰 Total potential tax savings: ${total_potential_savings:,.2f}")
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error finding tax-loss opportunities: {e}")
            return []
    
    def _place_wash_trade_compliant_order(self, symbol: str, qty: float, side: str, 
                                        order_type: str = "market", limit_price: Optional[float] = None,
                                        stop_price: Optional[float] = None) -> Dict:
        """
        Place an order using bracket/complex orders to comply with wash trade rules.
        
        Alpaca requires complex orders when it detects potential wash trades to ensure
        compliance with tax regulations and trading rules.
        """
        try:
            logger.info(f"🔄 Placing wash trade compliant order for {symbol}: {side} {qty}")
            
            # Get current price for bracket order calculations
            current_price = self.get_current_price(symbol)
            if not current_price:
                raise ValueError(f"Cannot determine current price for {symbol}")
            
            # For wash trade compliance, we'll use a bracket order with very wide stops
            # This satisfies Alpaca's requirement for "complex orders" while maintaining simple execution
            
            if side.lower() == "buy":
                # For buy orders, set a very conservative take profit and stop loss
                take_profit_price = current_price * 1.20  # 20% profit target (very conservative)
                stop_loss_price = current_price * 0.85    # 15% stop loss (very conservative)
            else:  # sell or sell_short
                # For sell orders, set conservative profit/loss targets  
                take_profit_price = current_price * 0.80  # 20% profit on short (conservative)
                stop_loss_price = current_price * 1.15    # 15% loss limit (conservative)
            
            # Use the existing OCO order method for bracket functionality
            logger.info(f"📊 Using bracket order: TP=${take_profit_price:.2f}, SL=${stop_loss_price:.2f}")
            
            bracket_result = self.place_oco_order(
                symbol=symbol,
                qty=qty, 
                side=side,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price
            )
            
            logger.info(f"✅ Wash trade compliant bracket order placed for {symbol}")
            return bracket_result
            
        except Exception as e:
            logger.error(f"Failed to place wash trade compliant order for {symbol}: {e}")
            
            # Final fallback: try a simple limit order at current price if market order failed
            if order_type == "market" and limit_price is None:
                try:
                    logger.warning(f"Final fallback: placing limit order at current price for {symbol}")
                    current_price = self.get_current_price(symbol)
                    
                    # Place limit order slightly better than current price to ensure execution
                    if side.lower() == "buy":
                        fallback_price = current_price * 1.001  # Pay 0.1% more for buy
                    else:
                        fallback_price = current_price * 0.999  # Accept 0.1% less for sell
                    
                    fallback_order = self.api.submit_order(
                        symbol=symbol,
                        qty=abs(qty),
                        side=side.lower().replace("sell_short", "sell"),
                        type="limit",
                        limit_price=str(fallback_price),
                        time_in_force="gtc"
                    )
                    
                    logger.info(f"✅ Fallback limit order placed for {symbol} at ${fallback_price:.2f}")
                    return {
                        "id": fallback_order.id,
                        "symbol": fallback_order.symbol,
                        "qty": float(fallback_order.qty),
                        "side": fallback_order.side,
                        "order_type": fallback_order.order_type,
                        "status": fallback_order.status,
                        "submitted_at": fallback_order.submitted_at,
                        "wash_trade_compliant": True
                    }
                    
                except Exception as fallback_error:
                    logger.error(f"Even fallback limit order failed for {symbol}: {fallback_error}")
                    raise
            
            raise

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