from tools.alpaca_market_data import fetch_stock_prices as alpaca_fetch_stock_prices
import requests
import time

# --- STOCK FETCH FUNCTION (via Alpaca API) ---
def fetch_stock_prices(tickers):
    """Fetch stock prices using Alpaca API - YFINANCE REPLACEMENT."""
    try:
        return alpaca_fetch_stock_prices(tickers)
    except Exception as e:
        print(f"[Stock Error] Alpaca fetch failed: {e}")
        return {}

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