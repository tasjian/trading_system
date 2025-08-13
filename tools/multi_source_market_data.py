#!/usr/bin/env python3
"""
Multi-Source Market Data Client
Comprehensive solution for price signal generation using multiple data providers:
- Alpha Vantage (primary historical data)
- Finnhub (real-time quotes & fundamentals)
- Financial Modeling Prep (backup historical data)
- News API (sentiment-driven signals)

Designed to resolve critical Alpaca paper trading limitations by providing
reliable price change signals from multiple data sources.
"""

import logging
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
import time
import json

from config.settings import settings
from tools.yfinance_utils import yfinance_market_data

logger = logging.getLogger(__name__)

@dataclass
class PriceSignal:
    """Standardized price signal from any data source."""
    symbol: str
    current_price: float
    previous_close: float
    price_change: float
    price_change_percent: float
    strength: float
    direction: str
    data_source: str
    timestamp: datetime
    confidence: float = 1.0
    volume: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for compatibility."""
        return {
            "symbol": self.symbol,
            "price_change": self.price_change_percent,
            "strength": self.strength,
            "direction": self.direction,
            "description": f"{self.direction} {self.price_change_percent:.1%}",
            "current_price": self.current_price,
            "previous_close": self.previous_close,
            "data_source": self.data_source,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class NewsSignal:
    """News-based signal for trading decisions."""
    symbol: str
    headline: str
    sentiment_score: float
    strength: float
    source: str
    published_at: datetime
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for compatibility."""
        return {
            "symbol": self.symbol,
            "signal_type": "news",
            "strength": self.strength,
            "direction": "up" if self.sentiment_score > 0 else "down",
            "description": f"News sentiment: {self.sentiment_score:.2f}",
            "headline": self.headline,
            "source": self.source,
            "confidence": self.confidence,
            "data_source": "news_api"
        }

class MultiSourceMarketData:
    """Multi-source market data client with intelligent fallback strategy."""
    
    def __init__(self):
        """Initialize multi-source market data client."""
        self.alpha_vantage_key = getattr(settings, 'alpha_vantage_api_key', None)
        self.finnhub_key = getattr(settings, 'finnhub_api_key', None)
        self.fmp_key = getattr(settings, 'fmp_api_key', None)
        self.news_api_key = getattr(settings, 'news_api_key', None)
        
        # Rate limiting
        self.alpha_vantage_calls = 0
        self.alpha_vantage_reset_time = datetime.now()
        self.finnhub_calls = 0
        self.finnhub_reset_time = datetime.now()
        
        # Cache for performance
        self._price_cache = {}
        self._cache_duration = timedelta(minutes=5)
        
        logger.info("🔗 Multi-source market data client initialized")
        logger.info(f"Available sources: Alpha Vantage: {'✅' if self.alpha_vantage_key else '❌'}, "
                   f"Finnhub: {'✅' if self.finnhub_key else '❌'}, "
                   f"FMP: {'✅' if self.fmp_key else '❌'}, "
                   f"News API: {'✅' if self.news_api_key else '❌'}")
    
    async def get_price_change_signals(self, symbols: List[str], 
                                     threshold: float = 0.02) -> List[Dict]:
        """
        Get price change signals using multiple data sources with intelligent fallback.
        
        NEW PRIORITY ORDER (addressing Alpaca historical data limitations):
        1. YFinance (primary - free, reliable, no API key needed)
        2. Alpha Vantage (secondary - premium but rate limited)
        3. Finnhub (tertiary - good real-time quotes)
        4. Financial Modeling Prep (quaternary - backup historical)
        5. Enhanced news-based signals (sentiment)
        
        Args:
            symbols: List of symbols to analyze
            threshold: Minimum price change threshold
            
        Returns:
            List of signal dictionaries (minimum 2 guaranteed)
        """
        logger.info(f"🎯 Generating price signals for {len(symbols)} symbols using multi-source strategy")
        logger.info(f"Priority order: YFinance → Alpha Vantage → Finnhub → FMP → News")
        
        signals = []
        failed_symbols = list(symbols)  # Start with all symbols
        successful_sources = []
        
        # Strategy 1: YFinance (NEW PRIMARY - no API key, reliable)
        try:
            yf_signals = await self._get_yfinance_signals(symbols, threshold)
            signals.extend(yf_signals)
            successful_sources.append('yfinance')
            # Remove successfully processed symbols
            successful_symbols = {s["symbol"] for s in yf_signals}
            failed_symbols = [s for s in failed_symbols if s not in successful_symbols]
            logger.info(f"🚀 YFinance (PRIMARY): {len(yf_signals)} signals")
        except Exception as e:
            logger.warning(f"YFinance failed: {e}")
        
        # Strategy 2: Alpha Vantage (secondary - only if still need signals)
        if len(signals) < 2 and failed_symbols:
            try:
                av_signals = await self._get_alpha_vantage_signals(failed_symbols, threshold)
                signals.extend(av_signals)
                successful_sources.append('alpha_vantage')
                # Remove successfully processed symbols  
                successful_symbols = {s["symbol"] for s in av_signals}
                failed_symbols = [s for s in failed_symbols if s not in successful_symbols]
                logger.info(f"📈 Alpha Vantage (SECONDARY): {len(av_signals)} signals")
            except Exception as e:
                logger.warning(f"Alpha Vantage failed: {e}")
        
        # Strategy 3: Finnhub (tertiary - only if still insufficient)
        if len(signals) < 2 and failed_symbols:
            try:
                finnhub_signals = await self._get_finnhub_signals(failed_symbols, threshold)
                signals.extend(finnhub_signals)
                logger.info(f"📊 Finnhub: {len(finnhub_signals)} signals")
                # Remove successfully processed symbols
                successful_symbols = {s["symbol"] for s in finnhub_signals}
                failed_symbols = [s for s in failed_symbols if s not in successful_symbols]
            except Exception as e:
                logger.warning(f"Finnhub failed: {e}")
        
        # Strategy 4: FMP (quaternary - final data source fallback)
        if len(signals) < 2 and failed_symbols:
            try:
                fmp_signals = await self._get_fmp_signals(failed_symbols, threshold)
                signals.extend(fmp_signals)
                logger.info(f"💰 FMP: {len(fmp_signals)} signals")
                successful_symbols = {s["symbol"] for s in fmp_signals}
                failed_symbols = [s for s in failed_symbols if s not in successful_symbols]
            except Exception as e:
                logger.warning(f"FMP failed: {e}")
        
        # Strategy 5: Enhanced news-based signals (always run if still need signals)
        if len(signals) < 2:
            try:
                news_signals = await self._get_news_based_signals(symbols)
                signals.extend(news_signals)
                successful_sources.append('news_api')
                logger.info(f"📰 News API (ENHANCEMENT): {len(news_signals)} signals")
            except Exception as e:
                logger.warning(f"News signals failed: {e}")
        
        # Final validation - with YFinance as primary, this should always succeed
        if len(signals) < 2:
            # Emergency diagnostic
            logger.error(f"❌ CRITICAL: Insufficient signals after all sources tried")
            logger.error(f"Signals found: {len(signals)}")
            logger.error(f"Sources attempted: {successful_sources}")
            logger.error(f"Failed symbols remaining: {len(failed_symbols)}")
            
            # This should not happen with YFinance as primary since it's very reliable
            raise RuntimeError(
                f"❌ CRITICAL: Unable to generate minimum 2 signals ({len(signals)} found). "
                f"Sources attempted: {successful_sources}. "
                f"This indicates a systemic failure - even YFinance primary source failed. "
                f"Check network connectivity and symbol validity."
            )
        
        logger.info(f"✅ Successfully generated {len(signals)} total price signals")
        logger.info(f"Successful sources: {successful_sources}")
        logger.info(f"YFinance PRIMARY integration working: {'yfinance' in successful_sources}")
        return [signal.to_dict() if hasattr(signal, 'to_dict') else signal for signal in signals]
    
    async def _get_yfinance_signals(self, symbols: List[str], 
                                  threshold: float) -> List[Dict]:
        """
        Get price signals using YFinance API (NEW PRIMARY SOURCE).
        
        This addresses the Alpaca paper trading historical data limitation
        by using free, reliable yfinance data without API keys.
        """
        try:
            # Use the working yfinance implementation
            signals_raw = await yfinance_market_data.get_price_change_signals_async(symbols, threshold)
            
            # Convert to standard format if needed
            signals = []
            for signal in signals_raw:
                # YFinance signals are already in the correct format
                if isinstance(signal, dict):
                    # Ensure consistent data_source labeling
                    signal['data_source'] = 'yfinance_primary'
                    signal['confidence'] = 0.95  # High confidence for yfinance data
                    signals.append(signal)
                else:
                    # Handle PriceSignal objects
                    signal_dict = signal.to_dict() if hasattr(signal, 'to_dict') else signal
                    signal_dict['data_source'] = 'yfinance_primary'
                    signal_dict['confidence'] = 0.95
                    signals.append(signal_dict)
            
            logger.info(f"🎯 YFinance primary source generated {len(signals)} signals")
            return signals
            
        except Exception as e:
            logger.warning(f"YFinance primary source failed: {e}")
            # Don't re-raise - let the orchestrator try other sources
            return []
    
    async def _get_alpha_vantage_signals(self, symbols: List[str], 
                                       threshold: float) -> List[PriceSignal]:
        """Get price signals using Alpha Vantage API."""
        signals = []
        
        if not self.alpha_vantage_key:
            raise ValueError("Alpha Vantage API key not available")
        
        # Rate limiting: 5 calls per minute
        await self._check_alpha_vantage_rate_limit()
        
        async with aiohttp.ClientSession() as session:
            for symbol in symbols[:10]:  # Limit to 10 for rate limiting
                try:
                    url = f"https://www.alphavantage.co/query"
                    params = {
                        'function': 'GLOBAL_QUOTE',
                        'symbol': symbol,
                        'apikey': self.alpha_vantage_key
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if 'Global Quote' in data:
                                quote = data['Global Quote']
                                current_price = float(quote['05. price'])
                                prev_close = float(quote['08. previous close'])
                                change_percent = float(quote['10. change percent'][:-1]) / 100
                                
                                if abs(change_percent) >= threshold:
                                    signal = PriceSignal(
                                        symbol=symbol,
                                        current_price=current_price,
                                        previous_close=prev_close,
                                        price_change=current_price - prev_close,
                                        price_change_percent=change_percent,
                                        strength=min(1.0, abs(change_percent) / 0.1),
                                        direction="up" if change_percent > 0 else "down",
                                        data_source="alpha_vantage",
                                        timestamp=datetime.now(),
                                        confidence=0.95,
                                        volume=float(quote['06. volume'])
                                    )
                                    signals.append(signal)
                    
                    self.alpha_vantage_calls += 1
                    await asyncio.sleep(12)  # 5 calls per minute = 12 second intervals
                    
                except Exception as e:
                    logger.warning(f"Alpha Vantage error for {symbol}: {e}")
                    continue
        
        return signals
    
    async def _get_finnhub_signals(self, symbols: List[str], 
                                 threshold: float) -> List[PriceSignal]:
        """Get price signals using Finnhub API."""
        signals = []
        
        if not self.finnhub_key:
            raise ValueError("Finnhub API key not available")
        
        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                try:
                    # Get current quote
                    quote_url = f"https://finnhub.io/api/v1/quote"
                    quote_params = {'symbol': symbol, 'token': self.finnhub_key}
                    
                    async with session.get(quote_url, params=quote_params) as response:
                        if response.status == 200:
                            quote_data = await response.json()
                            
                            current_price = quote_data.get('c', 0)  # Current price
                            prev_close = quote_data.get('pc', 0)    # Previous close
                            
                            if current_price > 0 and prev_close > 0:
                                change_percent = (current_price - prev_close) / prev_close
                                
                                if abs(change_percent) >= threshold:
                                    signal = PriceSignal(
                                        symbol=symbol,
                                        current_price=current_price,
                                        previous_close=prev_close,
                                        price_change=current_price - prev_close,
                                        price_change_percent=change_percent,
                                        strength=min(1.0, abs(change_percent) / 0.1),
                                        direction="up" if change_percent > 0 else "down",
                                        data_source="finnhub",
                                        timestamp=datetime.now(),
                                        confidence=0.90
                                    )
                                    signals.append(signal)
                    
                    await asyncio.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    logger.warning(f"Finnhub error for {symbol}: {e}")
                    continue
        
        return signals
    
    async def _get_fmp_signals(self, symbols: List[str], 
                             threshold: float) -> List[PriceSignal]:
        """Get price signals using Financial Modeling Prep API."""
        signals = []
        
        if not self.fmp_key:
            raise ValueError("FMP API key not available")
        
        async with aiohttp.ClientSession() as session:
            # Batch request for efficiency
            symbol_list = ','.join(symbols[:20])  # Limit to 20 symbols
            url = f"https://financialmodelingprep.com/api/v3/quote/{symbol_list}"
            params = {'apikey': self.fmp_key}
            
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for item in data:
                            try:
                                symbol = item['symbol']
                                current_price = item['price']
                                prev_close = item['previousClose']
                                change_percent = item['changesPercentage'] / 100
                                
                                if abs(change_percent) >= threshold:
                                    signal = PriceSignal(
                                        symbol=symbol,
                                        current_price=current_price,
                                        previous_close=prev_close,
                                        price_change=item['change'],
                                        price_change_percent=change_percent,
                                        strength=min(1.0, abs(change_percent) / 0.1),
                                        direction="up" if change_percent > 0 else "down",
                                        data_source="fmp",
                                        timestamp=datetime.now(),
                                        confidence=0.85,
                                        volume=item.get('volume')
                                    )
                                    signals.append(signal)
                            except (KeyError, TypeError) as e:
                                continue
                                
            except Exception as e:
                logger.warning(f"FMP batch request failed: {e}")
        
        return signals
    
    async def _get_news_based_signals(self, symbols: List[str]) -> List[NewsSignal]:
        """Generate trading signals based on news sentiment analysis."""
        signals = []
        
        if not self.news_api_key:
            logger.warning("News API key not available")
            return signals
        
        async with aiohttp.ClientSession() as session:
            for symbol in symbols[:15]:  # Limit for rate limiting
                try:
                    # Get news for this symbol
                    url = "https://newsapi.org/v2/everything"
                    params = {
                        'q': f"{symbol} stock",
                        'language': 'en',
                        'sortBy': 'publishedAt',
                        'pageSize': 5,
                        'from': (datetime.now() - timedelta(days=1)).isoformat(),
                        'apiKey': self.news_api_key
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            articles = data.get('articles', [])
                            
                            if articles:
                                # Simple sentiment analysis
                                total_sentiment = 0
                                article_count = 0
                                
                                for article in articles:
                                    headline = article.get('title', '')
                                    description = article.get('description', '')
                                    text = f"{headline} {description}".lower()
                                    
                                    # Basic sentiment scoring
                                    positive_words = ['up', 'gain', 'rise', 'surge', 'boost', 'growth', 'profit', 'buy', 'strong']
                                    negative_words = ['down', 'fall', 'drop', 'crash', 'loss', 'decline', 'sell', 'weak']
                                    
                                    sentiment = 0
                                    for word in positive_words:
                                        sentiment += text.count(word) * 0.1
                                    for word in negative_words:
                                        sentiment -= text.count(word) * 0.1
                                    
                                    total_sentiment += sentiment
                                    article_count += 1
                                
                                if article_count > 0:
                                    avg_sentiment = total_sentiment / article_count
                                    
                                    # Create signal if sentiment is strong enough
                                    if abs(avg_sentiment) > 0.1:
                                        signal = NewsSignal(
                                            symbol=symbol,
                                            headline=articles[0]['title'],
                                            sentiment_score=avg_sentiment,
                                            strength=min(1.0, abs(avg_sentiment) * 2),
                                            source="news_api",
                                            published_at=datetime.fromisoformat(
                                                articles[0]['publishedAt'].replace('Z', '+00:00')
                                            ),
                                            confidence=0.75
                                        )
                                        signals.append(signal)
                    
                    await asyncio.sleep(0.2)  # Rate limiting
                    
                except Exception as e:
                    logger.warning(f"News signal error for {symbol}: {e}")
                    continue
        
        return signals
    
    async def _get_momentum_signals(self, symbols: List[str], 
                                  threshold: float) -> List[PriceSignal]:
        """Generate momentum signals from cached/synthesized data."""
        signals = []
        
        # Use simple momentum indicators from available data
        for symbol in symbols[:10]:
            try:
                # Generate momentum signal based on symbol characteristics
                # This is a backup when other data sources fail
                
                # Simple heuristic: certain symbols tend to have momentum
                tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
                volatile_stocks = ['GME', 'AMC', 'MEME', 'SPCE', 'PLTR']
                
                if symbol in tech_stocks or symbol in volatile_stocks:
                    # NO SYNTHETIC SIGNALS - require real momentum data
                    raise RuntimeError(f"❌ CRITICAL: No real momentum data available for {symbol} - synthetic signal generation disabled")
                        
            except Exception as e:
                logger.warning(f"Momentum signal error for {symbol}: {e}")
                continue
        
        return signals
    
    async def _generate_emergency_signals(self, symbols: List[str]) -> List[PriceSignal]:
        """REMOVED: Emergency signals disabled - system must use real market data only."""
        raise RuntimeError("❌ CRITICAL: Emergency signal generation disabled - system requires valid market data from Alpha Vantage, Finnhub, or FMP APIs")
    
    async def _check_alpha_vantage_rate_limit(self):
        """Check and enforce Alpha Vantage rate limits (5 calls per minute)."""
        now = datetime.now()
        
        # Reset counter every minute
        if now - self.alpha_vantage_reset_time > timedelta(minutes=1):
            self.alpha_vantage_calls = 0
            self.alpha_vantage_reset_time = now
        
        # Wait if we've hit the limit
        if self.alpha_vantage_calls >= 5:
            wait_time = 60 - (now - self.alpha_vantage_reset_time).seconds
            if wait_time > 0:
                logger.info(f"Alpha Vantage rate limit reached, waiting {wait_time}s")
                await asyncio.sleep(wait_time)
                self.alpha_vantage_calls = 0
                self.alpha_vantage_reset_time = datetime.now()

# Global instance for easy import
multi_source_data = MultiSourceMarketData()

# Convenience functions matching existing API
async def get_price_change_signals(symbols: List[str], threshold: float = 0.02) -> List[Dict]:
    """Get price change signals using multi-source strategy."""
    return await multi_source_data.get_price_change_signals(symbols, threshold)

def get_price_change_signals_sync(symbols: List[str], threshold: float = 0.02) -> List[Dict]:
    """Synchronous wrapper for existing code compatibility."""
    return asyncio.run(get_price_change_signals(symbols, threshold))

logger.info("🚀 Multi-source market data module loaded - enterprise-grade signal generation ready")