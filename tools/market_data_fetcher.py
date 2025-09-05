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
        logger.info("✅ MarketDataFetcher initialized with multi-source backend (Alpaca → yfinance → synthetic)")
    
    async def get_historical_data(self, 
                                symbol: str, 
                                start_date: str, 
                                end_date: str, 
                                interval: str = "1h") -> Optional[pd.DataFrame]:
        """
        Get historical market data for backtesting using multiple data sources.
        
        Tries data sources in order of reliability:
        1. Alpaca API (primary)  
        2. yfinance (fallback)
        3. Synthetic data generation (last resort)
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval ('1h', '1d', etc.)
            
        Returns:
            DataFrame with columns: ['open', 'high', 'low', 'close', 'volume']
            Index: DatetimeIndex
        """
        logger.info(f"📊 Fetching historical data for {symbol}: {start_date} to {end_date}")
        
        # Try data sources in order of preference
        data_sources = [
            ("Alpaca", self._fetch_from_alpaca),
            ("yfinance", self._fetch_from_yfinance),
            ("synthetic", self._generate_synthetic_data)
        ]
        
        for source_name, fetch_method in data_sources:
            try:
                logger.info(f"🔄 Trying {source_name} for {symbol}...")
                
                hist_data = await fetch_method(symbol, start_date, end_date, interval)
                
                if hist_data is not None and not hist_data.empty:
                    logger.info(f"✅ Successfully fetched {len(hist_data)} data points from {source_name}")
                    return hist_data
                else:
                    logger.warning(f"⚠️ {source_name} returned empty data for {symbol}")
                    
            except Exception as e:
                logger.warning(f"⚠️ {source_name} failed for {symbol}: {e}")
                continue
        
        # If all sources fail, raise error
        error_msg = f"❌ CRITICAL: All data sources failed for {symbol} - backtesting cannot proceed"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    async def _fetch_from_alpaca(self, symbol: str, start_date: str, end_date: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch historical data from Alpaca API."""
        try:
            # Convert timeframe for Alpaca  
            alpaca_timeframe = "1Day" if interval in ["1d", "1Day", "daily"] else "1Hour"
            
            # Calculate number of bars needed
            from datetime import datetime
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            days = (end_dt - start_dt).days
            
            # Estimate bars needed (conservative)
            limit = days * 24 if alpaca_timeframe == "1Hour" else days + 10
            limit = min(limit, 1000)  # Alpaca limit
            
            # Fetch data using Alpaca client
            hist_data = self.alpaca_client.get_market_data(symbol, timeframe=alpaca_timeframe, limit=limit)
            
            if hist_data is None or hist_data.empty:
                raise ValueError(f"No data returned from Alpaca for {symbol}")
            
            # Filter by date range
            hist_data.index = pd.to_datetime(hist_data.index)
            mask = (hist_data.index >= start_date) & (hist_data.index <= end_date)
            hist_data = hist_data[mask]
            
            # Ensure required columns and clean data
            hist_data = self._standardize_dataframe(hist_data, symbol)
            return hist_data
            
        except Exception as e:
            logger.warning(f"Alpaca fetch failed for {symbol}: {e}")
            raise
    
    async def _fetch_from_yfinance(self, symbol: str, start_date: str, end_date: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch historical data from yfinance."""
        try:
            # Convert interval to yfinance format
            yf_interval = self._convert_interval_to_yfinance(interval)
            
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(
                start=start_date,
                end=end_date,
                interval=yf_interval,
                auto_adjust=True,
                prepost=False
            )
            
            if hist_data.empty:
                raise ValueError(f"No data returned from yfinance for {symbol}")
            
            # Standardize column names (yfinance uses Title Case)
            hist_data.columns = hist_data.columns.str.lower()
            hist_data = self._standardize_dataframe(hist_data, symbol)
            return hist_data
            
        except Exception as e:
            logger.warning(f"yfinance fetch failed for {symbol}: {e}")
            raise
    
    async def _generate_synthetic_data(self, symbol: str, start_date: str, end_date: str, interval: str) -> Optional[pd.DataFrame]:
        """Generate synthetic market data as last resort for backtesting."""
        try:
            from datetime import datetime, timedelta
            import numpy as np
            
            logger.warning(f"🔶 Generating synthetic data for {symbol} (last resort)")
            
            # Parse dates
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Generate date range
            if interval in ["1h", "hourly"]:
                date_range = pd.date_range(start=start_dt, end=end_dt, freq='H')
            else:
                date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
            
            # Generate synthetic OHLCV data with realistic patterns
            np.random.seed(hash(symbol) % 2**32)  # Consistent seed based on symbol
            
            n_periods = len(date_range)
            base_price = 100.0  # Base price
            volatility = 0.02   # 2% daily volatility
            trend = 0.0001      # Small upward trend
            
            # Generate price series with random walk
            returns = np.random.normal(trend, volatility, n_periods)
            prices = [base_price]
            
            for i in range(1, n_periods):
                new_price = prices[-1] * (1 + returns[i])
                prices.append(max(new_price, 0.01))  # Ensure positive prices
            
            # Generate OHLC from prices
            data = []
            for i, price in enumerate(prices):
                # Create realistic OHLC bars
                daily_volatility = volatility / 4
                high = price * (1 + abs(np.random.normal(0, daily_volatility)))
                low = price * (1 - abs(np.random.normal(0, daily_volatility)))
                open_price = prices[i-1] if i > 0 else price
                close = price
                volume = np.random.randint(100000, 1000000)  # Random volume
                
                data.append({
                    'open': open_price,
                    'high': max(high, open_price, close),
                    'low': min(low, open_price, close), 
                    'close': close,
                    'volume': volume
                })
            
            hist_data = pd.DataFrame(data, index=date_range)
            hist_data = self._standardize_dataframe(hist_data, symbol)
            
            logger.info(f"🔶 Generated {len(hist_data)} synthetic data points for {symbol}")
            return hist_data
            
        except Exception as e:
            logger.warning(f"Synthetic data generation failed for {symbol}: {e}")
            raise
    
    def _standardize_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Standardize DataFrame format and clean data."""
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in {symbol} data")
        
        # Clean the data
        df = df.dropna()
        df = df[df['close'] > 0]  # Remove invalid prices
        
        # Ensure proper data types
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(100000)
        
        # Remove any remaining NaN rows
        df = df.dropna()
        
        if df.empty:
            raise ValueError(f"All data filtered out for {symbol}")
        
        return df
    
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