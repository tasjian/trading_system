#!/usr/bin/env python3
"""
YFinance Utilities - Production-Ready Market Data Fallback
Based on successful commit c4975e0 with robust error handling and rate limiting.
Serves as fallback when Alpaca paper trading API lacks historical data access.
"""

import logging
import asyncio
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
import yfinance as yf

logger = logging.getLogger(__name__)

class YFinanceMarketData:
    """
    Robust yfinance wrapper with circuit breakers, retry mechanisms, and rate limiting.
    Implementation based on successful commit c4975e0 pattern.
    """
    
    def __init__(self):
        """Initialize YFinance market data client with circuit breaker."""
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        self.batch_size = 20  # Reduced batch size to avoid rate limits
        self.retry_delay = 2.0  # Base delay for retries
        self.batch_delay = 1.0  # Delay between batches
        self.request_delay = 0.1  # Delay between individual requests
        logger.info("🔗 YFinance Market Data client initialized (fallback mode)")
    
    def _check_circuit_breaker(self):
        """Check if circuit breaker should be triggered."""
        if self.consecutive_errors >= self.max_consecutive_errors:
            error_msg = f"❌ CIRCUIT BREAKER TRIGGERED: {self.consecutive_errors} consecutive yfinance errors"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _reset_circuit_breaker(self):
        """Reset circuit breaker on successful operation."""
        if self.consecutive_errors > 0:
            logger.info(f"✅ Circuit breaker reset after {self.consecutive_errors} errors")
            self.consecutive_errors = 0
    
    async def _async_delay(self, seconds: float):
        """Async delay helper."""
        await asyncio.sleep(seconds)
    
    def _sync_delay(self, seconds: float):
        """Sync delay helper."""
        time.sleep(seconds)
    
    async def _fetch_batch_with_retry(self, batch: List[str], max_retries: int = 3) -> Dict[str, any]:
        """
        Fetch a batch of tickers with retry mechanism.
        Based on successful commit c4975e0 pattern with enhanced error handling.
        """
        tickers_obj = None
        
        for retry in range(max_retries):
            try:
                # Add exponential backoff delay before creating Tickers object
                if retry > 0:
                    delay = self.retry_delay * (2 ** (retry - 1))  # Exponential backoff
                    logger.debug(f"⏳ Retry delay: {delay}s")
                    await self._async_delay(delay)
                
                # Create Tickers object with batch
                tickers_obj = yf.Tickers(' '.join(batch))
                logger.debug(f"✅ Batch fetch successful for {len(batch)} symbols (retry {retry + 1})")
                break  # Success, exit retry loop
                
            except Exception as retry_error:
                error_msg = str(retry_error).lower()
                
                # Check for specific yfinance errors that indicate data unavailability
                if any(phrase in error_msg for phrase in [
                    'expecting value', 'json', 'no price data found', 
                    'delisted', 'invalid symbol', '404'
                ]):
                    logger.warning(f"⚠️ YFinance data unavailable (retry {retry + 1}): {retry_error}")
                    if retry == max_retries - 1:
                        # Don't fail entirely - return empty object but log the issue
                        logger.warning(f"❌ YFinance batch unavailable after {max_retries} retries: {retry_error}")
                        return None
                else:
                    logger.debug(f"⚠️ Retry {retry + 1}/{max_retries} for batch: {retry_error}")
                    if retry == max_retries - 1:
                        logger.warning(f"❌ Failed to fetch batch after {max_retries} retries: {retry_error}")
                        raise retry_error
        
        return tickers_obj
    
    def fetch_stock_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        Fetch current stock prices using yfinance with robust error handling.
        Enhanced for better yfinance API stability based on commit c4975e0.
        
        Args:
            tickers: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to current prices
        """
        if not tickers:
            return {}
        
        data = {}
        successful_fetches = 0
        
        try:
            self._check_circuit_breaker()
            
            # Use individual ticker approach for better error isolation
            # This is more reliable than batch processing for yfinance
            for i, symbol in enumerate(tickers):
                try:
                    # Add progressive delay to avoid rate limiting
                    if i > 0:
                        self._sync_delay(self.request_delay * (1 + i // 10))  # Increase delay over time
                    
                    # Create individual ticker object with multiple fallback periods
                    ticker = yf.Ticker(symbol)
                    
                    # Try multiple data retrieval strategies
                    hist = None
                    for period, interval in [('1d', '1m'), ('2d', '1h'), ('5d', '1d')]:
                        try:
                            hist = ticker.history(period=period, interval=interval)
                            if not hist.empty:
                                break
                        except Exception as period_error:
                            logger.debug(f"⚠️ {symbol} period {period} failed: {period_error}")
                            continue
                    
                    if hist is not None and not hist.empty:
                        # Get most recent price
                        current_price = float(hist['Close'].iloc[-1])
                        
                        # Validate price is reasonable (not NaN, not zero, not negative)
                        if pd.notna(current_price) and current_price > 0:
                            data[symbol] = round(current_price, 2)
                            successful_fetches += 1
                            logger.debug(f"📈 {symbol}: ${current_price:.2f}")
                        else:
                            logger.warning(f"⚠️ Invalid price data for {symbol}: {current_price}")
                    else:
                        logger.warning(f"⚠️ No price data available for {symbol}")
                        
                except Exception as symbol_error:
                    error_msg = str(symbol_error).lower()
                    
                    # Handle specific yfinance errors gracefully
                    if any(phrase in error_msg for phrase in [
                        'expecting value', 'json', 'no price data found', 
                        'delisted', 'invalid symbol', '404', 'connection'
                    ]):
                        logger.debug(f"🔍 YFinance data issue for {symbol}: {symbol_error}")
                        # Don't count data unavailability as consecutive errors
                    else:
                        self.consecutive_errors += 1
                        logger.warning(f"❌ Error fetching {symbol}: {symbol_error}")
                        
                        # Check circuit breaker but don't fail entire operation
                        if self.consecutive_errors >= self.max_consecutive_errors:
                            logger.warning(f"⚠️ Circuit breaker threshold reached, stopping further fetches")
                            break
                    
                    continue
            
            # Reset circuit breaker on any successful fetches
            if successful_fetches > 0:
                self._reset_circuit_breaker()
            
            # Log success rate
            success_rate = successful_fetches / len(tickers) * 100 if tickers else 0
            logger.info(f"📊 YFinance fetched prices for {successful_fetches}/{len(tickers)} symbols ({success_rate:.1f}%)")
            
            return data
            
        except Exception as e:
            self.consecutive_errors += 1
            error_msg = f"❌ CRITICAL: YFinance fetch_stock_prices failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def fetch_stock_history(self, symbol: str, period: str = "1mo", 
                           interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Fetch historical stock data with robust error handling.
        
        Args:
            symbol: Stock symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            self._check_circuit_breaker()
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if not hist.empty:
                self._reset_circuit_breaker()
                logger.debug(f"📊 Retrieved {len(hist)} bars for {symbol}")
                return hist
            else:
                logger.warning(f"⚠️ No historical data for {symbol}")
                return None
                
        except Exception as e:
            self.consecutive_errors += 1
            logger.warning(f"❌ Historical data error for {symbol}: {e}")
            self._check_circuit_breaker()
            return None
    
    async def get_price_change_signals_async(self, symbols: List[str], 
                                           threshold: float = 0.02) -> List[Dict]:
        """
        Async version of price change signals with enhanced error handling.
        Based on successful commit c4975e0 pattern with individual ticker processing.
        """
        signals = []
        successful_signals = 0
        
        try:
            self._check_circuit_breaker()
            
            # Process individually for better error isolation
            for i, symbol in enumerate(symbols):
                try:
                    # Add progressive delay to avoid rate limiting
                    if i > 0:
                        await self._async_delay(self.request_delay * (1 + i // 5))
                    
                    # Create individual ticker with multiple fallback strategies
                    ticker = yf.Ticker(symbol)
                    
                    # Try multiple periods for historical data
                    hist = None
                    for period in ['2d', '5d', '1wk']:
                        try:
                            hist = ticker.history(period=period)
                            if len(hist) >= 2:
                                break
                        except Exception as period_error:
                            logger.debug(f"⚠️ {symbol} period {period} failed: {period_error}")
                            continue
                    
                    if hist is not None and len(hist) >= 2:
                        current_price = float(hist['Close'].iloc[-1])
                        prev_close = float(hist['Close'].iloc[-2])
                        
                        # Validate price data
                        if (pd.notna(prev_close) and pd.notna(current_price) and 
                            prev_close > 0 and current_price > 0):
                            
                            price_change = (current_price - prev_close) / prev_close
                            
                            if abs(price_change) >= threshold:
                                strength = min(1.0, abs(price_change) / 0.1)  # Scale to 0-1
                                direction = "up" if price_change > 0 else "down"
                                
                                signals.append({
                                    "symbol": symbol,
                                    "price_change": price_change,
                                    "strength": strength,
                                    "direction": direction,
                                    "description": f"{direction} {price_change:.1%}",
                                    "current_price": current_price,
                                    "previous_close": prev_close,
                                    "data_source": "yfinance_fallback"
                                })
                                
                                successful_signals += 1
                                logger.debug(f"📈 Signal: {symbol} {direction} {price_change:.1%}")
                    else:
                        logger.debug(f"🔍 Insufficient historical data for {symbol}")
                        
                except Exception as symbol_error:
                    error_msg = str(symbol_error).lower()
                    
                    # Handle specific yfinance errors gracefully
                    if any(phrase in error_msg for phrase in [
                        'expecting value', 'json', 'no price data found', 
                        'delisted', 'invalid symbol', '404', 'connection'
                    ]):
                        logger.debug(f"🔍 YFinance data issue for {symbol}: {symbol_error}")
                        # Don't count data unavailability as consecutive errors
                    else:
                        self.consecutive_errors += 1
                        logger.warning(f"❌ Price signal error for {symbol}: {symbol_error}")
                        
                        # Check circuit breaker but don't fail entire operation
                        if self.consecutive_errors >= self.max_consecutive_errors:
                            logger.warning(f"⚠️ Circuit breaker threshold reached, stopping signal generation")
                            break
                    
                    continue
            
            # Reset circuit breaker on any successful signals
            if successful_signals > 0:
                self._reset_circuit_breaker()
            
            # Log success rate
            success_rate = successful_signals / len(symbols) * 100 if symbols else 0
            logger.info(f"🎯 Generated {len(signals)} yfinance price signals from {len(symbols)} symbols ({success_rate:.1f}%)")
            
            return signals
            
        except Exception as e:
            self.consecutive_errors += 1
            error_msg = f"❌ CRITICAL: YFinance price signal generation failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def get_price_change_signals(self, symbols: List[str], 
                                threshold: float = 0.02) -> List[Dict]:
        """
        Synchronous wrapper for price change signals.
        """
        try:
            # Run async function in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.get_price_change_signals_async(symbols, threshold)
                )
            finally:
                loop.close()
        except Exception as e:
            error_msg = f"❌ CRITICAL: Sync price change signals failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def get_technical_indicators(self, symbol: str, period: str = "3mo") -> Optional[Dict]:
        """
        Calculate technical indicators with robust error handling.
        
        Args:
            symbol: Stock symbol
            period: Historical period for calculations
            
        Returns:
            Dictionary with technical indicators or None
        """
        try:
            self._check_circuit_breaker()
            
            hist = self.fetch_stock_history(symbol, period=period, interval='1d')
            
            if hist is None or len(hist) < 20:
                return None
            
            prices = hist['Close']
            volumes = hist['Volume']
            
            # Calculate technical indicators
            indicators = {}
            
            # Moving averages
            if len(prices) >= 20:
                indicators['sma_20'] = float(prices.tail(20).mean())
                indicators['sma_50'] = float(prices.tail(min(50, len(prices))).mean())
            
            # RSI (simple version)
            if len(prices) >= 14:
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).tail(14).mean()
                loss = (-delta.where(delta < 0, 0)).tail(14).mean()
                rs = gain / loss if loss != 0 else 0
                indicators['rsi'] = float(100 - (100 / (1 + rs)))
            
            # Volatility
            if len(prices) >= 20:
                returns = prices.pct_change().dropna()
                indicators['volatility'] = float(returns.tail(20).std() * np.sqrt(252))
            
            # Volume indicators
            if len(volumes) >= 10:
                indicators['avg_volume'] = float(volumes.tail(10).mean())
                indicators['volume_ratio'] = float(volumes.iloc[-1] / indicators['avg_volume'])
            
            # Current values
            indicators['current_price'] = float(prices.iloc[-1])
            if len(prices) >= 2:
                indicators['price_change_1d'] = float(
                    (prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2]
                )
            
            self._reset_circuit_breaker()
            logger.debug(f"📊 Technical indicators calculated for {symbol}")
            return indicators
            
        except Exception as e:
            self.consecutive_errors += 1
            logger.warning(f"❌ Technical indicators error for {symbol}: {e}")
            self._check_circuit_breaker()
            return None
    
    def get_market_status(self) -> Dict[str, bool]:
        """
        Get basic market status (yfinance doesn't provide this directly).
        
        Returns:
            Dictionary with basic market status
        """
        try:
            # Simple market hours check (US Eastern Time)
            now = datetime.now()
            
            # Basic market hours (9:30 AM to 4:00 PM ET, Monday-Friday)
            # This is a simplified version - for production use proper timezone handling
            weekday = now.weekday()  # 0 = Monday, 6 = Sunday
            hour = now.hour
            
            is_trading_day = weekday < 5  # Monday to Friday
            is_trading_hours = 9 <= hour < 16  # Simplified 9 AM to 4 PM
            is_open = is_trading_day and is_trading_hours
            
            return {
                'is_open': is_open,
                'source': 'yfinance_estimate',
                'timezone': 'US/Eastern (estimated)',
                'note': 'Simplified market hours check - use Alpaca for accurate status'
            }
            
        except Exception as e:
            logger.warning(f"❌ Market status error: {e}")
            return {'is_open': True, 'source': 'fallback'}  # Assume open if unknown
    
    def batch_fetch_prices(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Efficiently fetch comprehensive price data for multiple symbols.
        
        Args:
            symbols: List of symbols
            
        Returns:
            Dictionary with comprehensive price data per symbol
        """
        results = {}
        
        try:
            # Get basic price data
            prices = self.fetch_stock_prices(symbols)
            
            # Enhance with additional data for successful price fetches
            for symbol, price in prices.items():
                try:
                    # Get basic quote-like data from recent history
                    hist = self.fetch_stock_history(symbol, period='1d', interval='1m')
                    
                    if hist is not None and not hist.empty:
                        latest_data = hist.iloc[-1]
                        
                        results[symbol] = {
                            'symbol': symbol,
                            'price': price,
                            'timestamp': latest_data.name,  # Index is timestamp
                            'open': float(latest_data['Open']),
                            'high': float(latest_data['High']),
                            'low': float(latest_data['Low']),
                            'volume': float(latest_data['Volume']),
                            'source': 'yfinance_fallback'
                        }
                    else:
                        # Fallback to basic price data
                        results[symbol] = {
                            'symbol': symbol,
                            'price': price,
                            'timestamp': datetime.now(),
                            'source': 'yfinance_basic'
                        }
                        
                except Exception as e:
                    logger.warning(f"❌ Batch enhancement error for {symbol}: {e}")
                    # Still include basic price data
                    results[symbol] = {
                        'symbol': symbol,
                        'price': price,
                        'timestamp': datetime.now(),
                        'source': 'yfinance_basic'
                    }
                    continue
                
                # Rate limiting between symbols
                self._sync_delay(self.request_delay)
        
        except Exception as e:
            logger.error(f"❌ Batch fetch error: {e}")
            return {}
        
        logger.info(f"📊 YFinance batch fetched data for {len(results)}/{len(symbols)} symbols")
        return results


# Global instance for easy import
yfinance_market_data = YFinanceMarketData()

# Convenience functions that match alpaca_market_data API
def fetch_stock_prices(tickers: List[str]) -> Dict[str, float]:
    """Convenience function matching alpaca_market_data API."""
    return yfinance_market_data.fetch_stock_prices(tickers)

def fetch_stock_history(symbol: str, period: str = "1mo", interval: str = "1d") -> Optional[pd.DataFrame]:
    """Convenience function matching alpaca_market_data API."""
    return yfinance_market_data.fetch_stock_history(symbol, period, interval)

def get_price_change_signals(symbols: List[str], threshold: float = 0.02) -> List[Dict]:
    """Convenience function matching alpaca_market_data API."""
    return yfinance_market_data.get_price_change_signals(symbols, threshold)

def get_technical_indicators(symbol: str, period: str = "3mo") -> Optional[Dict]:
    """Convenience function matching alpaca_market_data API."""
    return yfinance_market_data.get_technical_indicators(symbol, period)

def get_market_status() -> Dict[str, bool]:
    """Convenience function for market status."""
    return yfinance_market_data.get_market_status()

def batch_fetch_prices(symbols: List[str]) -> Dict[str, Dict]:
    """Convenience function for batch price fetching."""
    return yfinance_market_data.batch_fetch_prices(symbols)

logger.info("🚀 YFinance Market Data module loaded - Alpaca fallback ready")