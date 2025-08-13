"""
YFinance Utilities
Simple, function-based approach for fetching market data following best practices.
"""

import yfinance as yf
import pandas as pd
import time
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def fetch_stock_prices(tickers: List[str]) -> Dict[str, float]:
    """
    Fetch current stock prices using simple function-based approach.
    
    Args:
        tickers: List of stock symbols to fetch
        
    Returns:
        Dictionary mapping symbols to current prices
    """
    data = {}
    for symbol in tickers:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                last_price = hist["Close"].iloc[-1]
                data[symbol] = round(last_price, 2)
        except Exception as e:
            logger.warning(f"[Stock Error] {symbol}: {e}")
    return data


def fetch_stock_history(symbol: str, period: str = "1mo", interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Fetch historical stock data with proper error handling.
    
    Args:
        symbol: Stock symbol to fetch
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
    Returns:
        DataFrame with historical data or None if failed
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        if not hist.empty:
            return hist
    except Exception as e:
        logger.warning(f"[Stock History Error] {symbol}: {e}")
    return None


def fetch_stock_info(symbol: str) -> Optional[Dict]:
    """
    Fetch stock info/fundamentals with proper error handling.
    
    Args:
        symbol: Stock symbol to fetch
        
    Returns:
        Dictionary with stock info or None if failed
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if info:
            return info
    except Exception as e:
        logger.warning(f"[Stock Info Error] {symbol}: {e}")
    return None


def fetch_multiple_stock_histories(symbols: List[str], period: str = "1mo", 
                                 interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple stocks with individual error handling.
    
    Args:
        symbols: List of stock symbols
        period: Time period for each stock
        interval: Data interval for each stock
        
    Returns:
        Dictionary mapping symbols to DataFrames
    """
    data = {}
    for symbol in symbols:
        hist = fetch_stock_history(symbol, period, interval)
        if hist is not None:
            data[symbol] = hist
        
        # Brief pause to avoid rate limiting
        time.sleep(0.1)
    
    return data


def get_price_change_signals(symbols: List[str], threshold: float = 0.02) -> List[Dict]:
    """
    Get price change signals for symbols with proper error handling.
    
    Args:
        symbols: List of symbols to check
        threshold: Minimum price change threshold (default 2%)
        
    Returns:
        List of signal dictionaries
    """
    signals = []
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            
            if len(hist) >= 2:
                prev_close = hist["Close"].iloc[-2]
                curr_price = hist["Close"].iloc[-1]
                
                # Validate price data
                if pd.isna(prev_close) or pd.isna(curr_price) or prev_close <= 0 or curr_price <= 0:
                    continue
                
                price_change = (curr_price - prev_close) / prev_close
                
                if abs(price_change) >= threshold:
                    strength = min(1.0, abs(price_change) / 0.1)  # Scale to 0-1
                    direction = "up" if price_change > 0 else "down"
                    
                    signals.append({
                        "symbol": symbol,
                        "price_change": price_change,
                        "strength": strength,
                        "direction": direction,
                        "description": f"{direction} {price_change:.1%}"
                    })
                    
        except Exception as e:
            # Only log if it's not a common "delisted" or JSON parsing error
            error_str = str(e).lower()
            if "no price data found" not in error_str and "delisted" not in error_str and "expecting value" not in error_str:
                logger.debug(f"[Price Signal Error] {symbol}: {e}")
            continue
            
        # Brief pause to avoid rate limiting
        time.sleep(0.1)
    
    return signals


def get_technical_indicators(symbol: str, period: str = "3mo") -> Optional[Dict]:
    """
    Calculate basic technical indicators for a symbol.
    
    Args:
        symbol: Stock symbol
        period: Historical period for calculation
        
    Returns:
        Dictionary with technical indicators or None
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        
        if hist.empty or len(hist) < 20:
            return None
        
        # Calculate basic indicators
        close_prices = hist["Close"]
        
        # Moving averages
        ma_20 = close_prices.rolling(window=20).mean().iloc[-1] if len(close_prices) >= 20 else None
        ma_50 = close_prices.rolling(window=50).mean().iloc[-1] if len(close_prices) >= 50 else None
        
        # RSI calculation (simplified)
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1] if len(delta) >= 14 else None
        
        # Current price vs moving averages
        current_price = close_prices.iloc[-1]
        
        indicators = {
            "current_price": current_price,
            "ma_20": ma_20,
            "ma_50": ma_50,
            "rsi": rsi,
            "above_ma_20": ma_20 and current_price > ma_20,
            "above_ma_50": ma_50 and current_price > ma_50,
        }
        
        return indicators
        
    except Exception as e:
        logger.warning(f"[Technical Indicators Error] {symbol}: {e}")
        return None