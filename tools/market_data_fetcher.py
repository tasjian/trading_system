#!/usr/bin/env python3
"""
Market Data Fetcher for Backtesting
Provides historical market data interface for the backtesting framework using Alpaca API.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import yfinance as yf

from tools.alpaca_client import alpaca_client

logger = logging.getLogger(__name__)

class MarketDataFetcher:
    """
    Market data fetcher that provides historical data for backtesting.
    Uses yfinance for historical data - raises errors if data unavailable.
    """
    
    def __init__(self):
        self.alpaca_client = alpaca_client
        logger.info("✅ MarketDataFetcher initialized with yfinance backend")
    
    async def get_historical_data(self, 
                                symbol: str, 
                                start_date: str, 
                                end_date: str, 
                                interval: str = "1h") -> Optional[pd.DataFrame]:
        """
        Get historical market data for backtesting.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval ('1h', '1d', etc.)
            
        Returns:
            DataFrame with columns: ['open', 'high', 'low', 'close', 'volume']
            Index: DatetimeIndex
        """
        try:
            logger.info(f"📊 Fetching historical data for {symbol}: {start_date} to {end_date}")
            
            # Convert interval to yfinance format
            yf_interval = self._convert_interval_to_yfinance(interval)
            
            # Use yfinance for reliable historical data (Alpaca has limited historical access in paper trading)
            try:
                ticker = yf.Ticker(symbol)
                
                # Fetch historical data
                hist_data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=yf_interval,
                    auto_adjust=True,
                    prepost=False
                )
                
                if hist_data.empty:
                    error_msg = f"❌ CRITICAL: No historical data available for {symbol} from yfinance - cannot proceed with backtesting"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Standardize column names (yfinance uses Title Case)
                hist_data.columns = hist_data.columns.str.lower()
                
                # Ensure required columns exist
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in required_columns:
                    if col not in hist_data.columns:
                        logger.error(f"❌ Missing required column '{col}' in {symbol} data")
                        return None
                
                # Clean the data
                hist_data = hist_data.dropna()
                hist_data = hist_data[hist_data['close'] > 0]
                
                logger.info(f"✅ Fetched {len(hist_data)} data points for {symbol}")
                return hist_data
                
            except Exception as yf_error:
                error_msg = f"❌ CRITICAL: yfinance error for {symbol}: {yf_error} - cannot proceed with backtesting"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
        except Exception as e:
            error_msg = f"❌ CRITICAL: Failed to fetch historical data for {symbol}: {e} - system cannot proceed"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _convert_interval_to_yfinance(self, interval: str) -> str:
        """Convert internal interval format to yfinance format."""
        interval_mapping = {
            "1h": "1h",
            "1d": "1d", 
            "1Day": "1d",
            "daily": "1d",
            "hourly": "1h"
        }
        return interval_mapping.get(interval, "1d")
    
    
    def get_available_symbols(self) -> list:
        """Get list of available symbols for backtesting."""
        # Return common symbols that should be available
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'SPY', 'QQQ', 'IWM', 'XLF', 'XLK', 'VTI', 'ARKK'
        ]
    
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol is available for backtesting."""
        try:
            # Quick validation using current data
            data = self.alpaca_client.get_market_data(symbol, limit=1)
            return not data.empty
        except:
            return False

# Global instance for easy access
market_data_fetcher = MarketDataFetcher()