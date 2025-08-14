from tools.alpaca_market_data import fetch_stock_prices as alpaca_fetch_stock_prices
from tools.dual_provider_market_data import get_market_data, get_price_signals
from tools.resilient_signal_orchestrator import get_resilient_price_signals_sync
from tools.market_data_proxy import (
    get_current_price, 
    get_market_data as get_cached_market_data, 
    get_bulk_market_data,
    subscribe_symbol,
    market_data_proxy
)
import asyncio
import requests
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# --- CACHE-FIRST STOCK FETCH FUNCTION (Redis Cache → Alpaca → Dual-Provider) ---
def fetch_stock_prices(tickers):
    """
    Fetch stock prices using cache-first strategy to eliminate rate limiting:
    1. Try Redis cache (sub-second latency, no rate limits)
    2. Try Alpaca API (backup for fresh data)
    3. Try Dual-Provider system (final fallback)
    
    Auto-subscribes symbols to WebSocket for future cache hits.
    """
    async def _fetch_cached_prices():
        # First, ensure symbols are subscribed for future updates
        for ticker in tickers:
            try:
                await subscribe_symbol(ticker)
            except Exception as e:
                logger.debug(f"Subscription warning for {ticker}: {e}")
        
        # Try to get data from cache first
        cached_data = await get_bulk_market_data(tickers)
        
        prices = {}
        missing_tickers = []
        
        # Process cached data
        for ticker in tickers:
            ticker_upper = ticker.upper()
            if ticker_upper in cached_data:
                cache_age = time.time() - cached_data[ticker_upper]['timestamp']
                if cache_age < 300:  # Use cache if less than 5 minutes old
                    prices[ticker] = cached_data[ticker_upper]['price']
                    logger.debug(f"📦 Cache hit for {ticker}: ${prices[ticker]:.2f} (age: {cache_age:.1f}s)")
                else:
                    missing_tickers.append(ticker)
                    logger.debug(f"📦 Cache stale for {ticker} (age: {cache_age:.1f}s)")
            else:
                missing_tickers.append(ticker)
                logger.debug(f"📦 Cache miss for {ticker}")
        
        return prices, missing_tickers
    
    try:
        # Primary: Try cache first
        logger.debug(f"📦 Checking cache for {len(tickers)} tickers")
        
        # Run async cache check with proper event loop handling
        try:
            # Check if we're in an async context
            try:
                asyncio.get_running_loop()
                # We're in a running event loop - use thread pool to avoid conflicts
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _fetch_cached_prices())
                    cached_prices, missing_tickers = future.result(timeout=30)
                logger.debug("📦 Used thread pool for cache check in async context")
            except RuntimeError:
                # No running event loop - safe to use asyncio.run
                cached_prices, missing_tickers = asyncio.run(_fetch_cached_prices())
                logger.debug("📦 Used asyncio.run for cache check")
        except Exception as e:
            logger.warning(f"📦 Cache check failed: {e}")
            cached_prices, missing_tickers = {}, tickers
        
        if cached_prices:
            logger.info(f"✅ Cache provided {len(cached_prices)}/{len(tickers)} prices")
            
            # If we got all prices from cache, return immediately
            if len(cached_prices) == len(tickers):
                return cached_prices
        
        # Secondary: Try Alpaca for missing tickers
        if missing_tickers:
            try:
                logger.debug(f"📊 Fetching {len(missing_tickers)} missing tickers from Alpaca")
                alpaca_data = alpaca_fetch_stock_prices(missing_tickers)
                
                if alpaca_data:
                    cached_prices.update(alpaca_data)
                    logger.info(f"✅ Alpaca provided {len(alpaca_data)} additional prices")
                    
                    # If we now have all prices, return
                    if len(cached_prices) == len(tickers):
                        return cached_prices
                    
            except Exception as alpaca_error:
                logger.warning(f"⚠️ Alpaca fetch failed: {alpaca_error}")
        
        # Tertiary: Dual-Provider fallback for any remaining missing tickers
        still_missing = [t for t in tickers if t not in cached_prices]
        if still_missing:
            try:
                logger.info(f"🔄 Dual-Provider fallback for {len(still_missing)} tickers")
                
                async def _get_dual_provider_data():
                    market_data = await get_market_data(still_missing)
                    prices = {}
                    for symbol, data in market_data.items():
                        prices[symbol] = data.get('price', 0.0)
                    return prices
                
                try:
                    dual_provider_data = loop.run_until_complete(_get_dual_provider_data())
                except RuntimeError:
                    # Already in event loop, use asyncio.run
                    dual_provider_data = asyncio.run(_get_dual_provider_data())
                
                if dual_provider_data:
                    cached_prices.update(dual_provider_data)
                    logger.info(f"✅ Dual-Provider provided {len(dual_provider_data)} additional prices")
                    
            except Exception as dual_provider_error:
                logger.warning(f"⚠️ Dual-Provider fallback failed: {dual_provider_error}")
        
        # Return whatever we managed to collect
        if cached_prices:
            logger.info(f"✅ Total prices collected: {len(cached_prices)}/{len(tickers)}")
            return cached_prices
        else:
            raise ValueError("No price data available from any source")
            
    except Exception as e:
        logger.error(f"❌ All price fetch methods failed: {e}")
        raise RuntimeError(f"❌ CRITICAL: Cache, Alpaca, and Dual-Provider all failed. "
                         f"Error: {e}. Check connectivity and API keys.")

# --- CACHE-ENHANCED PRICE CHANGE SIGNALS ---
def get_price_change_signals(symbols, threshold=0.02):
    """
    Get price change signals using cache-enhanced strategy:
    1. Check cache for recent price data to generate signals
    2. Fall back to Alpaca if cache is insufficient
    3. Use Resilient Signal Orchestrator as final fallback
    
    This eliminates rate limiting by leveraging cached real-time data.
    """
    async def _generate_cache_signals():
        # Subscribe to symbols for future updates
        for symbol in symbols:
            try:
                await subscribe_symbol(symbol)
            except Exception as e:
                logger.debug(f"Subscription warning for {symbol}: {e}")
        
        # Get current market data from cache
        cached_data = await get_bulk_market_data(symbols)
        
        signals = []
        cache_symbols = []
        
        for symbol in symbols:
            symbol_upper = symbol.upper()
            if symbol_upper in cached_data:
                data = cached_data[symbol_upper]
                cache_age = time.time() - data['timestamp']
                
                # Use cache if recent (less than 5 minutes old)
                if cache_age < 300:
                    price = data['price']
                    change_percent = data.get('change_percent', 0)
                    
                    # Generate signal if meets threshold
                    if abs(change_percent) >= threshold:
                        signal = {
                            "symbol": symbol,
                            "price_change": change_percent,
                            "strength": min(1.0, abs(change_percent) / 0.1),
                            "direction": "up" if change_percent > 0 else "down",
                            "description": f"Cache signal: {change_percent:.1%}",
                            "current_price": price,
                            "data_source": "market_data_cache",
                            "confidence": 0.95,  # High confidence for recent cache data
                            "timestamp": datetime.now().isoformat()
                        }
                        signals.append(signal)
                        cache_symbols.append(symbol)
        
        return signals, cache_symbols
    
    try:
        # Primary: Try cache-based signals
        logger.debug(f"📦 Generating cache-based signals for {len(symbols)} symbols")
        
        # Run async cache signal generation
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # We're in an async context, skip cache for sync function
                cache_signals, cache_symbols = [], []
                logger.debug("📦 Skipping cache signals - already in async context")
            else:
                cache_signals, cache_symbols = loop.run_until_complete(_generate_cache_signals())
        except RuntimeError:
            # No event loop running, create new one
            cache_signals, cache_symbols = asyncio.run(_generate_cache_signals())
        
        if cache_signals:
            logger.info(f"✅ Cache generated {len(cache_signals)} signals from {len(cache_symbols)} symbols")
            
            # If we have enough signals from cache, return them
            if len(cache_signals) >= 2:
                return cache_signals
        
        # Secondary: Try Alpaca for additional signals
        remaining_symbols = [s for s in symbols if s not in cache_symbols]
        if remaining_symbols:
            try:
                # Import here to avoid circular imports
                from tools.alpaca_market_data import get_price_change_signals as alpaca_signals
                
                logger.debug(f"📊 Alpaca signals for {len(remaining_symbols)} remaining symbols")
                alpaca_data = alpaca_signals(remaining_symbols, threshold)
                
                if alpaca_data:
                    cache_signals.extend(alpaca_data)
                    logger.info(f"✅ Alpaca provided {len(alpaca_data)} additional signals")
                    
                    # If we now have enough signals, return
                    if len(cache_signals) >= 2:
                        return cache_signals
                        
            except Exception as alpaca_error:
                logger.warning(f"⚠️ Alpaca signals failed: {alpaca_error}")
        
        # Tertiary: Resilient Signal Orchestrator fallback
        if len(cache_signals) < 2:
            try:
                logger.info(f"🔄 Resilient Signal Orchestrator fallback for all {len(symbols)} symbols")
                
                # Use the resilient orchestrator for comprehensive signal generation
                resilient_data = get_resilient_price_signals_sync(symbols, threshold, min_signals=2)
                
                if resilient_data:
                    # Merge with any cache signals we already have
                    cache_signals.extend(resilient_data)
                    logger.info(f"✅ Resilient Signal Orchestrator provided {len(resilient_data)} signals")
                    
            except Exception as resilient_error:
                logger.warning(f"⚠️ Resilient Signal Orchestrator failed: {resilient_error}")
        
        # Return whatever signals we managed to collect
        if cache_signals:
            logger.info(f"✅ Total signals generated: {len(cache_signals)}")
            return cache_signals
        else:
            raise ValueError("No signals available from any source")
            
    except Exception as e:
        logger.error(f"❌ All signal generation methods failed: {e}")
        raise RuntimeError(f"❌ CRITICAL: Cache, Alpaca, and Resilient Orchestrator all failed. "
                         f"Error: {e}. This indicates systemic data source failure.")

# --- HYBRID HISTORICAL DATA ---
def fetch_stock_history(symbol, period="1mo", interval="1d"):
    """
    Fetch historical data with Alpaca primary + Alpha Vantage fallback.
    Updated to use Alpha Vantage instead of yfinance to eliminate rate limiting issues.
    Historical data is less critical for rate limiting since it's typically cached longer.
    """
    try:
        # Import here to avoid circular imports
        from tools.alpaca_market_data import fetch_stock_history as alpaca_history
        
        # Primary: Try Alpaca
        logger.debug(f"📊 Attempting Alpaca historical data for {symbol}")
        alpaca_data = alpaca_history(symbol, period, interval)
        
        if alpaca_data is not None and not alpaca_data.empty:
            logger.info(f"✅ Alpaca historical data successful for {symbol}: {len(alpaca_data)} bars")
            return alpaca_data
        else:
            raise ValueError(f"Alpaca historical data empty for {symbol} - switching to Alpha Vantage fallback")
            
    except Exception as alpaca_error:
        logger.warning(f"⚠️ Alpaca historical data failed for {symbol}: {alpaca_error}")
        
        # Fallback: Alpha Vantage (via dual-provider system)
        try:
            logger.info(f"🔄 Falling back to Alpha Vantage for historical data: {symbol}")
            
            # For historical data, we'll return a simple current price as pandas DataFrame
            # This is a limitation but avoids the yfinance rate limiting issues
            async def _get_historical_fallback():
                market_data = await get_market_data([symbol])
                if symbol in market_data:
                    import pandas as pd
                    from datetime import datetime
                    
                    # Create minimal historical data with current price
                    data = market_data[symbol]
                    df = pd.DataFrame({
                        'Open': [data.get('price', 0.0)],
                        'High': [data.get('price', 0.0)],
                        'Low': [data.get('price', 0.0)],
                        'Close': [data.get('price', 0.0)],
                        'Volume': [data.get('volume', 0)]
                    }, index=[datetime.now()])
                    return df
                return None
            
            # Run async function
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            fallback_data = loop.run_until_complete(_get_historical_fallback())
            
            if fallback_data is not None and not fallback_data.empty:
                logger.info(f"✅ Alpha Vantage historical fallback successful for {symbol}: {len(fallback_data)} bars")
                return fallback_data
            else:
                raise ValueError(f"Alpha Vantage historical data also empty for {symbol}")
                
        except Exception as av_error:
            logger.error(f"❌ Alpha Vantage historical fallback also failed for {symbol}: {av_error}")
            # NO SILENT FAILURES - raise clear error
            raise RuntimeError(f"❌ CRITICAL: Both Alpaca and Alpha Vantage historical data failed for {symbol}. "
                             f"Alpaca error: {alpaca_error}. Alpha Vantage error: {av_error}. "
                             f"This is likely due to Alpaca paper trading historical data limitations.")

# --- CRYPTO FETCH FUNCTION (via Binance REST API) ---
def fetch_crypto_prices(pairs):
    url = "https://api.binance.com/api/v3/ticker/price"
    data = {}
    try:
        response = requests.get(url, timeout=5)
        prices = response.json()
        price_dict = {item['symbol']: float(item['price']) for item in prices}
        for pair in pairs:
            if pair in price_dict:
                data[pair] = round(price_dict[pair], 6)  # More precision for crypto
    except Exception as e:
        print(f"[Crypto Error] {e}")
    return data

# --- MAIN LOOP ---
if __name__ == "__main__":
    STOCKS = ["AAPL", "MSFT", "TSLA"]
    CRYPTOS = ["BTCUSDT", "ETHUSDT"]

    while True:
        stock_data = fetch_stock_prices(STOCKS)
        crypto_data = fetch_crypto_prices(CRYPTOS)

        combined_data = {
            "stocks": stock_data,
            "crypto": crypto_data
        }

        print(combined_data)  # Could also push to DB, file, websocket, etc.

        time.sleep(30)  # Fetch every 30 seconds