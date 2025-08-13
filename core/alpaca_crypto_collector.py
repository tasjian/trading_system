#!/usr/bin/env python3
"""
CRYPTO TRADING DISABLED - All code commented out for later implementation
Uncomment when crypto trading is re-enabled
"""

# CRYPTO TRADING DISABLED - Comment out entire file for later implementation
'''
"""
Alpaca Crypto Data Collector
Real-time crypto data using Alpaca API endpoints instead of Binance WebSocket
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class CryptoTicker:
    """Real-time crypto ticker data."""
    symbol: str
    price: float
    change_24h: float
    volume_24h: float
    bid: float
    ask: float
    timestamp: datetime

class AlpacaCryptoCollector:
    """Real-time crypto data via Alpaca API."""
    
    def __init__(self):
        self.tickers: Dict[str, CryptoTicker] = {}
        self.running = False
        self.update_interval = 30  # Update every 30 seconds
        self.last_update = {}
        
        # Import Alpaca client
        from tools.alpaca_client import alpaca_client
        self.alpaca = alpaca_client
        
        # Supported crypto symbols
        self.supported_symbols = [
            'BTCUSD', 'ETHUSD', 'DOGEUSD', 'LTCUSD', 'BCHUSD',
            'LINKUSD', 'UNIUSD', 'AAVEUSD'
        ]
        
        logger.info(f"✅ Alpaca Crypto Collector initialized with {len(self.supported_symbols)} symbols")
    
    async def start_streams(self, symbols: List[str]):
        """Start collecting crypto data for specified symbols using Alpaca API."""
        
        # Filter to only supported symbols
        valid_symbols = [s for s in symbols if s.upper() in self.supported_symbols]
        if not valid_symbols:
            logger.warning(f"No valid Alpaca crypto symbols in {symbols}")
            return
        
        self.running = True
        logger.info(f"🚀 Starting Alpaca crypto data collection for: {', '.join(valid_symbols)}")
        
        # Start the polling loop
        while self.running:
            try:
                await self._collect_ticker_data(valid_symbols)
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in crypto data collection loop: {e}")
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _collect_ticker_data(self, symbols: List[str]):
        """Collect latest ticker data from Alpaca for all symbols using crypto-specific API."""
        
        # Convert symbols to Alpaca crypto format (BTCUSD -> BTC/USD)
        alpaca_symbols = []
        symbol_mapping = {}  # Maps BTC/USD back to BTCUSD
        
        for symbol in symbols:
            if symbol.endswith('USD') and len(symbol) > 3:
                # Convert BTCUSD to BTC/USD format
                base_currency = symbol[:-3]  # Remove 'USD'
                alpaca_format = f"{base_currency}/USD"
                alpaca_symbols.append(alpaca_format)
                symbol_mapping[alpaca_format] = symbol
            else:
                # Keep original if it doesn't match expected pattern
                alpaca_symbols.append(symbol)
                symbol_mapping[symbol] = symbol
        
        if not alpaca_symbols:
            logger.warning("No valid symbols to collect crypto data for")
            return
        
        try:
            # Try crypto-specific quote API first
            current_prices = {}
            volumes = {}
            
            try:
                crypto_quotes = self.alpaca.api.get_latest_crypto_quotes(alpaca_symbols)
                for alpaca_symbol, quote in crypto_quotes.items():
                    if quote and hasattr(quote, 'ask_price') and hasattr(quote, 'bid_price'):
                        current_prices[alpaca_symbol] = float((quote.ask_price + quote.bid_price) / 2)
                        logger.info(f"✅ Got Alpaca crypto quote for {alpaca_symbol}: ${current_prices[alpaca_symbol]:.2f}")
            except Exception as e:
                logger.debug(f"Crypto quotes failed: {e}")
            
            # Fallback to crypto trades for missing prices
            try:
                crypto_trades = self.alpaca.api.get_latest_crypto_trades(alpaca_symbols)
                for alpaca_symbol, trade in crypto_trades.items():
                    if alpaca_symbol not in current_prices and trade and hasattr(trade, 'price'):
                        current_prices[alpaca_symbol] = float(trade.price)
                        if hasattr(trade, 'size'):
                            volumes[alpaca_symbol] = float(trade.size)
                        logger.info(f"✅ Got Alpaca crypto trade for {alpaca_symbol}: ${current_prices[alpaca_symbol]:.2f}")
            except Exception as e:
                logger.debug(f"Crypto trades failed: {e}")
            
        except Exception as e:
            logger.warning(f"Alpaca crypto API failed: {e}")
            current_prices = {}
            volumes = {}
        
        # Process collected data and create tickers
        for alpaca_symbol, original_symbol in symbol_mapping.items():
            try:
                current_price = current_prices.get(alpaca_symbol)
                volume = volumes.get(alpaca_symbol, 0)
                
                # If still no price, fallback to market data method
                if not current_price:
                    market_data = self.alpaca.get_market_data(original_symbol, timeframe="1Min", limit=1)
                    if not market_data.empty:
                        current_price = float(market_data.iloc[-1]['close'])
                        volume = float(market_data.iloc[-1]['volume'])
                        logger.debug(f"Got market data fallback for {original_symbol}: ${current_price:.2f}")
                
                if not current_price:
                    logger.warning(f"No price data available for {original_symbol}")
                    continue
                
                # Get 24h change if available (using historical data)
                change_24h = await self._get_24h_change(original_symbol, current_price)
                
                # Create ticker
                ticker = CryptoTicker(
                    symbol=original_symbol,
                    price=current_price,
                    change_24h=change_24h,
                    volume_24h=volume,
                    bid=current_price * 0.9995,  # Approximate bid/ask spread
                    ask=current_price * 1.0005,
                    timestamp=datetime.now()
                )
                
                self.tickers[original_symbol] = ticker
                self.last_update[original_symbol] = datetime.now()
                
                logger.debug(f"📊 {original_symbol}: ${ticker.price:.2f} ({ticker.change_24h:+.2f}%)")
                
            except Exception as e:
                logger.warning(f"Failed to get data for {original_symbol}: {e}")
    
    async def _get_24h_change(self, symbol: str, current_price: float) -> float:
        """Calculate 24h price change using historical data."""
        try:
            # Get historical data for 24h change calculation
            market_data = self.alpaca.get_market_data(symbol, timeframe="1Day", limit=2)
            
            if not market_data.empty and len(market_data) >= 2:
                yesterday_close = market_data.iloc[-2]['close']
                change_24h = ((current_price - yesterday_close) / yesterday_close) * 100
                return change_24h
            
        except Exception as e:
            logger.debug(f"Could not calculate 24h change for {symbol}: {e}")
        
        return 0.0  # Default to 0% change if calculation fails
    
    def get_latest_ticker(self, symbol: str) -> Optional[CryptoTicker]:
        """Get the latest ticker data for a symbol."""
        return self.tickers.get(symbol.upper())
    
    def get_all_tickers(self) -> Dict[str, CryptoTicker]:
        """Get all ticker data."""
        return self.tickers.copy()
    
    async def stop_streams(self):
        """Stop the data collection."""
        self.running = False
        logger.info("🛑 Alpaca crypto data collection stopped")
    
    def is_symbol_supported(self, symbol: str) -> bool:
        """Check if a crypto symbol is supported by Alpaca."""
        return symbol.upper() in self.supported_symbols

# Singleton instance
alpaca_crypto_collector = AlpacaCryptoCollector()'''
