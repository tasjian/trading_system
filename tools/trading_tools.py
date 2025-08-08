"""LangChain tools for trading operations."""

import logging
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool
from datetime import datetime
import pandas as pd

from tools.alpaca_client import alpaca_client
from config.settings import settings

logger = logging.getLogger(__name__)

@tool
def get_account_info() -> Dict[str, Any]:
    """Get current account information including cash, equity, and buying power."""
    try:
        return alpaca_client.get_account_info()
    except Exception as e:
        logger.error(f"Error getting account info: {e}")
        return {"error": str(e)}

@tool
def get_current_positions() -> List[Dict[str, Any]]:
    """Get all current portfolio positions."""
    try:
        return alpaca_client.get_positions()
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        return [{"error": str(e)}]

@tool
def get_market_data(symbol: str, timeframe: str = "1Day", limit: int = 50) -> Dict[str, Any]:
    """
    Get historical market data for a symbol.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'SPY')
        timeframe: Time interval ('1Min', '5Min', '15Min', '1Hour', '1Day')
        limit: Number of data points to retrieve
    """
    try:
        df = alpaca_client.get_market_data(symbol, timeframe, limit)
        if df.empty:
            return {"error": f"No data available for {symbol}"}
        
        # Convert to dict for JSON serialization
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "data_points": len(df),
            "latest_price": float(df.iloc[-1]["close"]),
            "latest_volume": float(df.iloc[-1]["volume"]),
            "price_change": float(df.iloc[-1]["close"] - df.iloc[-2]["close"]) if len(df) > 1 else 0,
            "data": df.to_dict("records")
        }
    except Exception as e:
        logger.error(f"Error getting market data for {symbol}: {e}")
        return {"error": str(e)}

@tool
def calculate_technical_indicators(symbol: str, indicators: List[str] = None) -> Dict[str, Any]:
    """
    Calculate technical indicators for a symbol.
    
    Args:
        symbol: Stock symbol
        indicators: List of indicators to calculate ['sma', 'rsi', 'macd', 'bollinger']
    """
    if indicators is None:
        indicators = ['sma', 'rsi']
    
    try:
        df = alpaca_client.get_market_data(symbol, limit=100)
        if df.empty:
            return {"error": f"No data available for {symbol}"}
        
        results = {"symbol": symbol, "indicators": {}}
        
        if "sma" in indicators:
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            results["indicators"]["sma"] = {
                "sma_20": float(df.iloc[-1]['sma_20']) if not pd.isna(df.iloc[-1]['sma_20']) else None,
                "sma_50": float(df.iloc[-1]['sma_50']) if not pd.isna(df.iloc[-1]['sma_50']) else None
            }
        
        if "rsi" in indicators:
            # Simple RSI calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            results["indicators"]["rsi"] = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
        
        if "macd" in indicators:
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            results["indicators"]["macd"] = {
                "macd": float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else None,
                "signal": float(signal.iloc[-1]) if not pd.isna(signal.iloc[-1]) else None,
                "histogram": float(macd.iloc[-1] - signal.iloc[-1]) if not pd.isna(macd.iloc[-1]) and not pd.isna(signal.iloc[-1]) else None
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Error calculating indicators for {symbol}: {e}")
        return {"error": str(e)}

@tool
def place_market_order(symbol: str, quantity: float, side: str) -> Dict[str, Any]:
    """
    Place a market order for immediate execution.
    
    Args:
        symbol: Stock symbol to trade
        quantity: Number of shares to trade (positive number)
        side: 'buy' or 'sell'
    """
    try:
        if side.lower() not in ['buy', 'sell']:
            return {"error": "Side must be 'buy' or 'sell'"}
        
        if quantity <= 0:
            return {"error": "Quantity must be positive"}
        
        # Additional safety check - skip market hours check for crypto
        from config.settings import is_crypto_symbol
        if not is_crypto_symbol(symbol) and not alpaca_client.is_market_open(symbol):
            return {"error": "Market is currently closed"}
        
        order = alpaca_client.place_order(
            symbol=symbol,
            qty=quantity,
            side=side.lower(),
            order_type="market"
        )
        
        return {
            "status": "success",
            "order_id": order["id"],
            "symbol": order["symbol"],
            "quantity": order["qty"],
            "side": order["side"],
            "submitted_at": order["submitted_at"]
        }
        
    except Exception as e:
        logger.error(f"Error placing market order: {e}")
        return {"error": str(e)}

@tool
def place_limit_order(symbol: str, quantity: float, side: str, limit_price: float) -> Dict[str, Any]:
    """
    Place a limit order at a specific price.
    
    Args:
        symbol: Stock symbol to trade
        quantity: Number of shares to trade
        side: 'buy' or 'sell'
        limit_price: Price limit for the order
    """
    try:
        if side.lower() not in ['buy', 'sell']:
            return {"error": "Side must be 'buy' or 'sell'"}
        
        if quantity <= 0:
            return {"error": "Quantity must be positive"}
        
        if limit_price <= 0:
            return {"error": "Limit price must be positive"}
        
        order = alpaca_client.place_order(
            symbol=symbol,
            qty=quantity,
            side=side.lower(),
            order_type="limit",
            limit_price=limit_price
        )
        
        return {
            "status": "success",
            "order_id": order["id"],
            "symbol": order["symbol"],
            "quantity": order["qty"],
            "side": order["side"],
            "limit_price": limit_price,
            "submitted_at": order["submitted_at"]
        }
        
    except Exception as e:
        logger.error(f"Error placing limit order: {e}")
        return {"error": str(e)}

@tool
def cancel_order(order_id: str) -> Dict[str, Any]:
    """Cancel an existing order by ID."""
    try:
        success = alpaca_client.cancel_order(order_id)
        return {
            "status": "success" if success else "failed",
            "order_id": order_id,
            "cancelled": success
        }
    except Exception as e:
        logger.error(f"Error cancelling order {order_id}: {e}")
        return {"error": str(e)}

@tool
def get_open_orders() -> List[Dict[str, Any]]:
    """Get all currently open orders."""
    try:
        return alpaca_client.get_orders(status="open")
    except Exception as e:
        logger.error(f"Error getting open orders: {e}")
        return [{"error": str(e)}]

@tool
def close_position(symbol: str, quantity: Optional[float] = None) -> Dict[str, Any]:
    """
    Close a position entirely or partially.
    
    Args:
        symbol: Stock symbol
        quantity: Amount to close (None for entire position)
    """
    try:
        order = alpaca_client.close_position(symbol, quantity)
        return {
            "status": "success",
            "symbol": symbol,
            "order": order
        }
    except Exception as e:
        logger.error(f"Error closing position {symbol}: {e}")
        return {"error": str(e)}

@tool
def get_asset_info(symbol: str) -> Dict[str, Any]:
    """Get information about a tradeable asset."""
    try:
        return alpaca_client.get_asset_info(symbol)
    except Exception as e:
        logger.error(f"Error getting asset info for {symbol}: {e}")
        return {"error": str(e)}

@tool
def check_market_status(symbol: Optional[str] = None) -> Dict[str, Any]:
    """Check if the market is currently open."""
    try:
        is_open = alpaca_client.is_market_open(symbol)
        calendar = alpaca_client.get_market_calendar()
        
        return {
            "market_open": is_open,
            "today": calendar[0] if calendar else None
        }
    except Exception as e:
        logger.error(f"Error checking market status: {e}")
        return {"error": str(e)}

@tool
def calculate_portfolio_metrics() -> Dict[str, Any]:
    """Calculate portfolio performance metrics."""
    try:
        account = alpaca_client.get_account_info()
        positions = alpaca_client.get_positions()
        
        total_value = account["equity"]
        cash = account["cash"]
        invested = total_value - cash
        
        # Calculate concentration risk
        concentration = {}
        if total_value > 0:
            for pos in positions:
                symbol = pos["symbol"]
                market_value = abs(pos["market_value"])
                concentration[symbol] = market_value / total_value
        
        # Calculate total unrealized P&L
        total_unrealized_pl = sum(pos.get("unrealized_pl", 0) for pos in positions)
        
        return {
            "portfolio_value": total_value,
            "cash": cash,
            "invested": invested,
            "unrealized_pl": total_unrealized_pl,
            "concentration_risk": concentration,
            "number_of_positions": len(positions),
            "largest_position": max(concentration.values()) if concentration else 0,
            "diversification_score": 1 - max(concentration.values()) if concentration else 1
        }
        
    except Exception as e:
        logger.error(f"Error calculating portfolio metrics: {e}")
        return {"error": str(e)}

@tool
def validate_trade_risk(symbol: str, quantity: float, side: str) -> Dict[str, Any]:
    """
    Validate if a trade meets risk management criteria.
    
    Args:
        symbol: Stock symbol
        quantity: Proposed trade quantity
        side: 'buy' or 'sell'
    """
    try:
        # Get current account and position info
        account = alpaca_client.get_account_info()
        positions = alpaca_client.get_positions()
        
        portfolio_value = account["equity"]
        buying_power = account["buying_power"]
        
        # Get current price
        market_data = alpaca_client.get_market_data(symbol, limit=1)
        if market_data.empty:
            return {"error": f"Cannot get price for {symbol}"}
        
        current_price = float(market_data.iloc[-1]["close"])
        trade_value = quantity * current_price
        
        validation_results = {
            "symbol": symbol,
            "quantity": quantity,
            "side": side,
            "trade_value": trade_value,
            "current_price": current_price,
            "checks": {}
        }
        
        # Check 1: Sufficient buying power
        if side.lower() == "buy":
            validation_results["checks"]["buying_power"] = {
                "passed": trade_value <= buying_power,
                "required": trade_value,
                "available": buying_power
            }
        
        # Check 2: Position size limit
        position_size_pct = trade_value / portfolio_value if portfolio_value > 0 else 0
        max_position_size = settings.max_position_size
        
        validation_results["checks"]["position_size"] = {
            "passed": position_size_pct <= max_position_size,
            "percentage": position_size_pct,
            "limit": max_position_size
        }
        
        # Check 3: Portfolio risk
        portfolio_risk = trade_value * settings.max_portfolio_risk
        validation_results["checks"]["portfolio_risk"] = {
            "passed": portfolio_risk <= portfolio_value * 0.1,  # Max 10% portfolio risk
            "risk_amount": portfolio_risk,
            "risk_percentage": settings.max_portfolio_risk
        }
        
        # Overall validation
        all_checks_passed = all(
            check.get("passed", False) 
            for check in validation_results["checks"].values()
        )
        
        validation_results["validation_passed"] = all_checks_passed
        validation_results["recommendation"] = "APPROVE" if all_checks_passed else "REJECT"
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Error validating trade risk: {e}")
        return {"error": str(e)}

# Tool collection for easy import
trading_tools = [
    get_account_info,
    get_current_positions,
    get_market_data,
    calculate_technical_indicators,
    place_market_order,
    place_limit_order,
    cancel_order,
    get_open_orders,
    close_position,
    get_asset_info,
    check_market_status,
    calculate_portfolio_metrics,
    validate_trade_risk
]