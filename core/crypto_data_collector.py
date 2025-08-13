#!/usr/bin/env python3
"""
Crypto Data Collector - DISABLED FOR LATER IMPLEMENTATION
Integrates Binance WebSocket API and CryptoCompare for real-time crypto data,
sentiment analysis, and news collection.

CRYPTO TRADING DISABLED - All code commented out for later implementation
"""

# CRYPTO TRADING DISABLED - Comment out entire file for later implementation
# Uncomment the code below when crypto trading is re-enabled

'''

import asyncio
import json
import logging
import websockets
import aiohttp
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from config.settings import settings
from .alpaca_crypto_collector import alpaca_crypto_collector

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

@dataclass
class CryptoSentiment:
    """Crypto sentiment analysis result."""
    symbol: str
    sentiment_score: float  # -1 to 1
    sentiment_label: str    # bullish, bearish, neutral
    confidence: float       # 0 to 1
    social_mentions: int
    news_count: int
    fear_greed_index: Optional[int]
    timestamp: datetime

@dataclass
class CryptoNews:
    """Crypto news item."""
    title: str
    summary: str
    url: str
    sentiment: str
    published_at: datetime
    source: str
    categories: List[str]

class BinanceWebSocketCollector:
    """Real-time crypto data via Binance WebSocket API."""
    
    def __init__(self):
        self.base_ws_url = "wss://stream.binance.com:9443"
        self.tickers: Dict[str, CryptoTicker] = {}
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.running = False
        
        # Convert Alpaca symbols to Binance format
        self.symbol_mapping = {
            'BTCUSD': 'btcusdt',
            'ETHUSD': 'ethusdt', 
            'BNBUSD': 'bnbusdt',
            'ADAUSD': 'adausdt',
            'SOLUSD': 'solusdt'
        }
    
    async def start_streams(self, symbols: List[str]):
        """Start WebSocket streams for given symbols."""
        self.running = True
        
        # Convert symbols to Binance format
        binance_symbols = []
        for symbol in symbols:
            binance_symbol = self.symbol_mapping.get(symbol)
            if binance_symbol:
                binance_symbols.append(binance_symbol)
            else:
                # Try to convert automatically (BTCUSD -> btcusdt)
                if symbol.endswith('USD'):
                    binance_symbol = symbol[:-3].lower() + 'usdt'
                    binance_symbols.append(binance_symbol)
        
        if not binance_symbols:
            logger.warning("No valid Binance symbols to stream")
            return
        
        # Create WebSocket URL - try single stream first to avoid 451 error
        if len(binance_symbols) == 1:
            # Single stream format for better compatibility
            stream_url = f"{self.base_ws_url}/ws/{binance_symbols[0]}@ticker"
        else:
            # Combined stream format
            streams = [f"{symbol}@ticker" for symbol in binance_symbols]
            stream_url = f"{self.base_ws_url}/stream?streams={'/'.join(streams)}"
        
        logger.info(f"🔄 Starting Binance WebSocket for {len(binance_symbols)} symbols")
        logger.info(f"📡 WebSocket URL: {stream_url}")
        if len(binance_symbols) > 1:
            logger.info(f"📊 Streams: {', '.join(streams)}")
        else:
            logger.info(f"📊 Single stream: {binance_symbols[0]}@ticker")
        
        try:
            async with websockets.connect(stream_url) as websocket:
                self.connections['combined'] = websocket
                
                async for message in websocket:
                    if not self.running:
                        break
                    
                    try:
                        message_data = json.loads(message)
                        # Handle combined stream format: {"stream":"btcusdt@ticker","data":{...}}
                        if 'stream' in message_data and 'data' in message_data:
                            await self._process_ticker_data(message_data['data'])
                        else:
                            # Single stream format (fallback)
                            await self._process_ticker_data(message_data)
                    except Exception as e:
                        logger.error(f"Error processing Binance message: {e}")
                        
        except Exception as e:
            if "451" in str(e):
                logger.warning(f"⚠️ Binance WebSocket unavailable (HTTP 451 - geographical restriction)")
                logger.info("💡 Crypto data collection disabled - trading system will continue with stocks only")
                logger.info("🌍 To enable crypto: use VPN or alternative crypto data source")
            else:
                logger.error(f"Binance WebSocket connection failed: {e}")
            
            # Set running to False to prevent retry loops
            self.running = False
    
    async def _process_ticker_data(self, data: Dict[str, Any]):
        """Process incoming ticker data from Binance."""
        try:
            symbol = data.get('s', '').upper()  # BTCUSDT -> BTCUSDT
            
            # Convert back to Alpaca format
            alpaca_symbol = None
            for alpaca, binance in self.symbol_mapping.items():
                if binance == symbol.lower():
                    alpaca_symbol = alpaca
                    break
            
            if not alpaca_symbol and symbol.endswith('USDT'):
                # Auto-convert: BTCUSDT -> BTCUSD
                alpaca_symbol = symbol[:-4] + 'USD'
            
            if not alpaca_symbol:
                return
            
            ticker = CryptoTicker(
                symbol=alpaca_symbol,
                price=float(data.get('c', 0)),      # Current price
                change_24h=float(data.get('P', 0)), # 24h change %
                volume_24h=float(data.get('v', 0)), # 24h volume
                bid=float(data.get('b', 0)),        # Best bid
                ask=float(data.get('a', 0)),        # Best ask
                timestamp=datetime.now()
            )
            
            self.tickers[alpaca_symbol] = ticker
            logger.debug(f"📊 {alpaca_symbol}: ${ticker.price:.2f} ({ticker.change_24h:+.2f}%)")
            
        except Exception as e:
            logger.error(f"Error parsing Binance ticker data: {e}")
    
    def get_latest_ticker(self, symbol: str) -> Optional[CryptoTicker]:
        """Get latest ticker data for symbol."""
        return self.tickers.get(symbol)
    
    def get_all_tickers(self) -> Dict[str, CryptoTicker]:
        """Get all latest ticker data."""
        return self.tickers.copy()
    
    async def stop_streams(self):
        """Stop all WebSocket streams."""
        self.running = False
        for ws in self.connections.values():
            await ws.close()
        self.connections.clear()

class CryptoCompareCollector:
    """Crypto sentiment and news via CryptoCompare API."""
    
    def __init__(self):
        self.base_url = "https://min-api.cryptocompare.com"
        self.api_key = settings.cryptocompare_api_key
        self.session = None
        
        # Rate limiting
        self.requests_per_second = 10
        self.last_request_time = 0
    
    async def _get_session(self):
        """Get or create aiohttp session."""
        if self.session is None:
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Apikey {self.api_key}'
            
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self.session
    
    async def _rate_limit(self):
        """Implement rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.requests_per_second
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    async def get_social_stats(self, symbol: str) -> Dict[str, Any]:
        """Get social media statistics for crypto symbol."""
        await self._rate_limit()
        session = await self._get_session()
        
        # Convert BTCUSD -> BTC
        crypto_symbol = symbol.replace('USD', '').replace('USDT', '')
        
        url = f"{self.base_url}/data/social/coin/histo/day"
        params = {
            'fsym': crypto_symbol,
            'limit': 7,  # Last 7 days
            'aggregate': 1
        }
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('Response') == 'Success':
                        return data.get('Data', [])
                    else:
                        logger.warning(f"CryptoCompare social API error: {data.get('Message')}")
                else:
                    logger.warning(f"CryptoCompare social API returned status {response.status}")
        except Exception as e:
            logger.error(f"Error fetching social stats for {symbol}: {e}")
        
        return []
    
    async def get_news(self, symbols: List[str], limit: int = 50) -> List[CryptoNews]:
        """Get crypto news for given symbols."""
        await self._rate_limit()
        session = await self._get_session()
        
        # Convert symbols: BTCUSD -> BTC
        crypto_symbols = [s.replace('USD', '').replace('USDT', '') for s in symbols]
        
        url = f"{self.base_url}/data/v2/news/"
        params = {
            'categories': ','.join(crypto_symbols),
            'excludeCategories': 'Sponsored',
            'limit': limit,
            'sortOrder': 'latest'
        }
        
        news_items = []
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('Response') == 'Success':
                        articles = data.get('Data', [])
                        
                        for article in articles:
                            try:
                                news_item = CryptoNews(
                                    title=article.get('title', ''),
                                    summary=article.get('body', '')[:500],  # First 500 chars
                                    url=article.get('url', ''),
                                    sentiment=self._classify_sentiment(article.get('title', '') + ' ' + article.get('body', '')),
                                    published_at=datetime.fromtimestamp(article.get('published_on', 0)),
                                    source=article.get('source_info', {}).get('name', 'Unknown'),
                                    categories=article.get('categories', '').split(',')
                                )
                                news_items.append(news_item)
                            except Exception as e:
                                logger.debug(f"Error parsing news item: {e}")
                                continue
                    else:
                        logger.warning(f"CryptoCompare news API error: {data.get('Message')}")
                else:
                    logger.warning(f"CryptoCompare news API returned status {response.status}")
                    
        except Exception as e:
            logger.error(f"Error fetching crypto news: {e}")
        
        logger.info(f"📰 Collected {len(news_items)} crypto news articles")
        return news_items
    
    def _classify_sentiment(self, text: str) -> str:
        """Simple sentiment classification for news text."""
        text = text.lower()
        
        # Bullish keywords
        bullish_words = ['bull', 'bullish', 'surge', 'rally', 'moon', 'pump', 'breakout', 'adoption', 
                        'institutional', 'upgrade', 'partnership', 'launch', 'positive', 'growth',
                        'rising', 'gain', 'profit', 'uptrend']
        
        # Bearish keywords  
        bearish_words = ['bear', 'bearish', 'crash', 'dump', 'correction', 'decline', 'drop',
                        'regulation', 'ban', 'hack', 'scam', 'negative', 'loss', 'down', 'fall',
                        'sell-off', 'liquidation', 'fear']
        
        bullish_score = sum(1 for word in bullish_words if word in text)
        bearish_score = sum(1 for word in bearish_words if word in text)
        
        if bullish_score > bearish_score:
            return 'bullish'
        elif bearish_score > bullish_score:
            return 'bearish'
        else:
            return 'neutral'
    
    async def analyze_sentiment(self, symbol: str) -> CryptoSentiment:
        """Comprehensive sentiment analysis for crypto symbol."""
        
        # Get social stats
        social_data = await self.get_social_stats(symbol)
        
        # Get recent news
        news_items = await self.get_news([symbol], limit=20)
        
        # Calculate sentiment metrics
        total_mentions = 0
        sentiment_score = 0.0
        
        if social_data:
            for day_data in social_data[-3:]:  # Last 3 days
                mentions = day_data.get('comments', 0) + day_data.get('posts', 0)
                total_mentions += mentions
        
        # News sentiment analysis
        news_sentiment_scores = []
        for news in news_items:
            if news.sentiment == 'bullish':
                news_sentiment_scores.append(0.6)
            elif news.sentiment == 'bearish':
                news_sentiment_scores.append(-0.6)
            else:
                news_sentiment_scores.append(0.0)
        
        if news_sentiment_scores:
            sentiment_score = np.mean(news_sentiment_scores)
        
        # Determine overall sentiment label
        if sentiment_score > 0.2:
            sentiment_label = 'bullish'
        elif sentiment_score < -0.2:
            sentiment_label = 'bearish'
        else:
            sentiment_label = 'neutral'
        
        # Calculate confidence based on data availability
        confidence = min(1.0, (len(news_items) / 20.0) + (min(total_mentions, 1000) / 1000.0)) / 2
        
        return CryptoSentiment(
            symbol=symbol,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            confidence=confidence,
            social_mentions=total_mentions,
            news_count=len(news_items),
            fear_greed_index=None,  # TODO: Integrate Fear & Greed Index
            timestamp=datetime.now()
        )
    
    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None

class CryptoDataCollector:
    """Main crypto data collector using Alpaca API."""
    
    def __init__(self):
        self.alpaca_collector = alpaca_crypto_collector
        self.cryptocompare = CryptoCompareCollector()
        self.websocket_task = None
        
    async def start_real_time_streams(self, symbols: List[str]):
        """Start real-time data collection for crypto symbols."""
        logger.info(f"🚀 Starting crypto data collection for: {', '.join(symbols)}")
        
        # Start Alpaca crypto data streams
        self.websocket_task = asyncio.create_task(
            self.alpaca_collector.start_streams(symbols)
        )
        
        logger.info("✅ Crypto real-time data streams started")
    
    async def get_comprehensive_crypto_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive crypto data including prices, sentiment, and news."""
        
        results = {}
        
        # Process each symbol
        for symbol in symbols:
            symbol_data = {
                'symbol': symbol,
                'timestamp': datetime.now()
            }
            
            try:
                # Get real-time price data from Alpaca
                ticker = self.alpaca_collector.get_latest_ticker(symbol)
                if ticker:
                    symbol_data['price_data'] = asdict(ticker)
                else:
                    logger.warning(f"No real-time price data available for {symbol}")
                
                # Get sentiment analysis
                sentiment = await self.cryptocompare.analyze_sentiment(symbol)
                symbol_data['sentiment'] = asdict(sentiment)
                
                results[symbol] = symbol_data
                
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {e}")
                continue
        
        logger.info(f"📊 Collected comprehensive data for {len(results)} crypto symbols")
        return results
    
    async def get_crypto_news_feed(self, symbols: List[str]) -> List[CryptoNews]:
        """Get latest crypto news feed."""
        return await self.cryptocompare.get_news(symbols)
    
    async def cleanup(self):
        """Clean up resources."""
        if self.websocket_task:
            self.binance.running = False
            await self.binance.stop_streams()
            self.websocket_task.cancel()
            try:
                await self.websocket_task
            except asyncio.CancelledError:
                pass
        
        await self.cryptocompare.close()
        logger.info("🧹 Crypto data collector cleanup completed")

# Global instance
crypto_collector = CryptoDataCollector()'''
