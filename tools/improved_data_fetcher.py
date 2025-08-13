from tools.alpaca_market_data import fetch_stock_prices as alpaca_fetch_stock_prices
from tools.yfinance_utils import fetch_stock_prices as yfinance_fetch_stock_prices
import requests
import time
import logging

logger = logging.getLogger(__name__)

# --- HYBRID STOCK FETCH FUNCTION (Alpaca with YFinance Fallback) ---
def fetch_stock_prices(tickers):
    """
    Fetch stock prices using Alpaca API with YFinance fallback.
    Based on successful commit c4975e0 pattern for robust data fetching.
    """
    try:
        # Primary: Try Alpaca API
        logger.debug(f"📊 Attempting Alpaca fetch for {len(tickers)} tickers")
        alpaca_data = alpaca_fetch_stock_prices(tickers)
        
        if alpaca_data and len(alpaca_data) > 0:
            logger.info(f"✅ Alpaca successful: {len(alpaca_data)}/{len(tickers)} prices fetched")
            return alpaca_data
        else:
            raise ValueError("Alpaca returned empty data - switching to yfinance fallback")
            
    except Exception as alpaca_error:
        logger.warning(f"⚠️ Alpaca fetch failed: {alpaca_error}")
        
        # Fallback: Try YFinance with robust error handling
        try:
            logger.info(f"🔄 Falling back to YFinance for {len(tickers)} tickers")
            yfinance_data = yfinance_fetch_stock_prices(tickers)
            
            if yfinance_data and len(yfinance_data) > 0:
                logger.info(f"✅ YFinance fallback successful: {len(yfinance_data)}/{len(tickers)} prices fetched")
                return yfinance_data
            else:
                raise ValueError("YFinance also returned empty data")
                
        except Exception as yfinance_error:
            logger.error(f"❌ YFinance fallback also failed: {yfinance_error}")
            # NO SILENT FAILURES - raise clear error
            raise RuntimeError(f"❌ CRITICAL: Both Alpaca and YFinance failed. "
                             f"Alpaca error: {alpaca_error}. YFinance error: {yfinance_error}. "
                             f"Check API keys and network connectivity.")

# --- HYBRID PRICE CHANGE SIGNALS ---
def get_price_change_signals(symbols, threshold=0.02):
    """
    Get price change signals with Alpaca primary + YFinance fallback.
    Critical for systems where Alpaca paper trading lacks historical data.
    """
    try:
        # Import here to avoid circular imports
        from tools.alpaca_market_data import get_price_change_signals as alpaca_signals
        from tools.yfinance_utils import get_price_change_signals as yfinance_signals
        
        # Primary: Try Alpaca
        logger.debug(f"📈 Attempting Alpaca price change signals for {len(symbols)} symbols")
        alpaca_data = alpaca_signals(symbols, threshold)
        
        if alpaca_data and len(alpaca_data) > 0:
            logger.info(f"✅ Alpaca signals successful: {len(alpaca_data)} signals generated")
            return alpaca_data
        else:
            raise ValueError("Alpaca price change signals returned empty - switching to yfinance fallback")
            
    except Exception as alpaca_error:
        logger.warning(f"⚠️ Alpaca price change signals failed: {alpaca_error}")
        
        # Fallback: YFinance with robust error handling
        try:
            logger.info(f"🔄 Falling back to YFinance price change signals for {len(symbols)} symbols")
            yfinance_data = yfinance_signals(symbols, threshold)
            
            if yfinance_data and len(yfinance_data) > 0:
                logger.info(f"✅ YFinance price change fallback successful: {len(yfinance_data)} signals generated")
                return yfinance_data
            else:
                raise ValueError("YFinance price change signals also returned empty")
                
        except Exception as yfinance_error:
            logger.error(f"❌ YFinance price change fallback also failed: {yfinance_error}")
            # NO SILENT FAILURES - raise clear error  
            raise RuntimeError(f"❌ CRITICAL: Both Alpaca and YFinance price change signals failed. "
                             f"Alpaca error: {alpaca_error}. YFinance error: {yfinance_error}. "
                             f"This may indicate Alpaca paper trading limitations with historical data.")

# --- HYBRID HISTORICAL DATA ---
def fetch_stock_history(symbol, period="1mo", interval="1d"):
    """
    Fetch historical data with Alpaca primary + YFinance fallback.
    Critical since Alpaca paper trading often lacks historical data access.
    """
    try:
        # Import here to avoid circular imports
        from tools.alpaca_market_data import fetch_stock_history as alpaca_history
        from tools.yfinance_utils import fetch_stock_history as yfinance_history
        
        # Primary: Try Alpaca
        logger.debug(f"📊 Attempting Alpaca historical data for {symbol}")
        alpaca_data = alpaca_history(symbol, period, interval)
        
        if alpaca_data is not None and not alpaca_data.empty:
            logger.info(f"✅ Alpaca historical data successful for {symbol}: {len(alpaca_data)} bars")
            return alpaca_data
        else:
            raise ValueError(f"Alpaca historical data empty for {symbol} - switching to yfinance fallback")
            
    except Exception as alpaca_error:
        logger.warning(f"⚠️ Alpaca historical data failed for {symbol}: {alpaca_error}")
        
        # Fallback: YFinance
        try:
            logger.info(f"🔄 Falling back to YFinance historical data for {symbol}")
            yfinance_data = yfinance_history(symbol, period, interval)
            
            if yfinance_data is not None and not yfinance_data.empty:
                logger.info(f"✅ YFinance historical fallback successful for {symbol}: {len(yfinance_data)} bars")
                return yfinance_data
            else:
                raise ValueError(f"YFinance historical data also empty for {symbol}")
                
        except Exception as yfinance_error:
            logger.error(f"❌ YFinance historical fallback also failed for {symbol}: {yfinance_error}")
            # NO SILENT FAILURES - raise clear error
            raise RuntimeError(f"❌ CRITICAL: Both Alpaca and YFinance historical data failed for {symbol}. "
                             f"Alpaca error: {alpaca_error}. YFinance error: {yfinance_error}. "
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