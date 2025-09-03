"""
Unified Market Intelligence System

Consolidates all market data sources and analysis capabilities into a single,
coherent system that provides:
- Multi-source data aggregation with fallbacks
- Technical and fundamental analysis
- LLM-powered sentiment analysis from news and earnings calls
- Risk assessment and signal generation
- Portfolio-level intelligence

This replaces multiple separate systems with one unified approach.
"""

import asyncio
import aiohttp
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import os
import warnings

warnings.filterwarnings('ignore')

from config.settings import settings, is_crypto_symbol
from .llm_sentiment_analyzer import LLMSentimentAnalyzer, SentimentAnalysis
from .earnings_scraper import EarningsCallScraper
from tools.alpaca_market_data import fetch_stock_prices, fetch_stock_history, get_technical_indicators
from utils.connection_pool import connection_pool

logger = logging.getLogger(__name__)

class SignalType(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    SHORT = "short"

class ConfidenceLevel(Enum):
    VERY_HIGH = "very_high"    # 0.8+
    HIGH = "high"              # 0.6-0.8
    MEDIUM = "medium"          # 0.4-0.6
    LOW = "low"                # 0.2-0.4
    VERY_LOW = "very_low"      # <0.2

@dataclass
class MarketData:
    """Unified market data point."""
    symbol: str
    price: float
    volume: Optional[int] = None
    timestamp: datetime = None
    source: str = "unified"
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class AnalysisResult:
    """Comprehensive analysis result."""
    symbol: str
    signal: SignalType
    confidence: ConfidenceLevel
    score: float  # -1 to 1
    price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    expected_return: float = 0.0
    risk_score: float = 0.5
    components: Dict[str, float] = None
    reasoning: List[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.components is None:
            self.components = {}
        if self.reasoning is None:
            self.reasoning = []

class UnifiedMarketIntelligence:
    """
    Unified market intelligence system that consolidates all data sources
    and analysis capabilities into a single, efficient interface.
    """
    
    def __init__(self):
        self.session = None
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # API keys from environment
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        self.fmp_key = os.getenv('FMP_API_KEY')
        self.news_key = os.getenv('NEWS_API_KEY')
        
        # Initialize LLM sentiment analyzer and earnings scraper
        self.sentiment_analyzer = LLMSentimentAnalyzer(
            anthropic_api_key=settings.anthropic_api_key,
            ollama_base_url=settings.ollama_base_url
        )
        self.earnings_scraper = EarningsCallScraper()
        
        # Rate limiting
        self.last_calls = {
            'alpha_vantage': [],
            'finnhub': [],
            'fmp': [],
            'news_api': [],
            'earnings_scraper': []
        }
        
        # Analysis weights
        self.weights = {
            'technical': 0.25,
            'fundamental': 0.30,
            'sentiment': 0.25,
            'market_structure': 0.20
        }
    
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data with intelligent fallback."""
        cache_key = f"price_{symbol}"
        
        # Check cache
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < 60:  # 1 minute cache for prices
                return data
        
        # Try premium sources first, then fallback
        price = await self._get_price_premium(symbol)
        if not price:
            price = await self._get_price_fallback(symbol)
        
        if price:
            market_data = MarketData(symbol=symbol, price=price)
            self.cache[cache_key] = (market_data, datetime.now())
            return market_data
        
        return MarketData(symbol=symbol, price=0.0, source="failed")
    
    async def _get_price_premium(self, symbol: str) -> Optional[float]:
        """Get price from premium APIs."""
        await self._ensure_session()
        
        # Try Finnhub first (fastest)
        if self.finnhub_key and await self._check_rate_limit('finnhub'):
            try:
                url = f"https://finnhub.io/api/v1/quote"
                params = {'symbol': symbol, 'token': self.finnhub_key}
                
                session = await self._ensure_session()
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'c' in data and data['c'] > 0:
                            return float(data['c'])
            except Exception as e:
                logger.debug(f"Finnhub price error for {symbol}: {e}")
        
        # Try FMP as backup
        if self.fmp_key and await self._check_rate_limit('fmp'):
            try:
                url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}"
                params = {'apikey': self.fmp_key}
                
                session = await self._ensure_session()
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and isinstance(data, list) and data[0].get('price'):
                            return float(data[0]['price'])
            except Exception as e:
                logger.debug(f"FMP price error for {symbol}: {e}")
        
        return None
    
    async def _get_price_fallback(self, symbol: str) -> Optional[float]:
        """Fallback to Yahoo Finance with multiple methods."""
        try:
            # Use utility function for price fetching
            prices = fetch_stock_prices([symbol])
            if symbol in prices:
                return prices[symbol]
            
            # If utility function failed, try fallback with historical data
            hist = fetch_stock_history(symbol, period="1d", interval="1m")
            if hist is not None and not hist.empty:
                return float(hist['Close'].iloc[-1])
            
            # Try regular daily data
            hist = fetch_stock_history(symbol, period="2d")
            if hist is not None and not hist.empty:
                return float(hist['Close'].iloc[-1])
            
            # Try info
            info = fetch_stock_info(symbol)
            if info and 'currentPrice' in info and info['currentPrice']:
                return float(info['currentPrice'])
                
        except Exception as e:
            logger.debug(f"Yahoo Finance error for {symbol}: {e}")
        
        return None
    
    async def analyze_stock(self, symbol: str) -> AnalysisResult:
        """Comprehensive stock analysis."""
        try:
            logger.info(f"📊 Analyzing {symbol} {'(crypto)' if is_crypto_symbol(symbol) else '(stock)'}")
            
            # Get current price
            market_data = await self.get_market_data(symbol)
            if market_data.price <= 0:
                return self._create_fallback_result(symbol, "No price data available")
            
            # Run analysis components concurrently - adjust for crypto vs stocks
            if is_crypto_symbol(symbol):
                # Crypto analysis focuses on technical and sentiment (no fundamentals/earnings)
                tasks = [
                    self._analyze_technical(symbol, market_data.price),
                    self._analyze_sentiment(symbol),
                    self._analyze_crypto_momentum(symbol, market_data.price)
                ]
            else:
                # Traditional stock analysis
                tasks = [
                    self._analyze_technical(symbol, market_data.price),
                    self._analyze_fundamental(symbol),
                    self._analyze_sentiment(symbol),
                    self._analyze_market_structure(symbol)
                ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            components = {}
            for i, (name, result) in enumerate(zip(['technical', 'fundamental', 'sentiment', 'market_structure'], results)):
                if isinstance(result, Exception):
                    logger.warning(f"{name} analysis error for {symbol}: {result}")
                    components[name] = 0.0
                else:
                    components[name] = result if result is not None else 0.0
            
            # Calculate overall score
            overall_score = sum(
                components[comp] * self.weights[comp] 
                for comp in components
            )
            
            # Generate signal and confidence
            signal = self._score_to_signal(overall_score)
            confidence = self._calculate_confidence(components, market_data)
            
            # Calculate targets and risk
            target_price, stop_loss = self._calculate_targets(market_data.price, overall_score)
            risk_score = self._calculate_risk(components)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(symbol, components, overall_score)
            
            return AnalysisResult(
                symbol=symbol,
                signal=signal,
                confidence=confidence,
                score=overall_score,
                price=market_data.price,
                target_price=target_price,
                stop_loss=stop_loss,
                expected_return=overall_score * 0.15,  # Max 15% expected return
                risk_score=risk_score,
                components=components,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}")
            return self._create_fallback_result(symbol, str(e))
    
    async def _analyze_technical(self, symbol: str, current_price: float) -> Optional[float]:
        """Technical analysis component."""
        try:
            # Use utility function for technical indicators
            indicators = get_technical_indicators(symbol, period="3mo")
            if indicators:
                score = 0.0
                factors = 0
                
                # Moving averages
                if indicators.get("above_ma_20") is not None:
                    factors += 1
                    if indicators["above_ma_20"]:
                        score += 0.3
                    else:
                        score -= 0.1
                
                if indicators.get("above_ma_50") is not None:
                    factors += 1
                    if indicators["above_ma_50"]:
                        score += 0.4
                    else:
                        score -= 0.1
                
                # RSI
                if indicators.get("rsi") is not None:
                    factors += 1
                    rsi = indicators["rsi"]
                    if rsi < 30:
                        score += 0.3  # Oversold
                    elif rsi > 70:
                        score -= 0.3  # Overbought
                
                return score / factors if factors > 0 else 0.0
            
            # Fallback to historical data if indicators failed
            hist = fetch_stock_history(symbol, period="3mo")
            if hist is None or hist.empty or len(hist) < 20:
                return None
            
            score = 0.0
            factors = 0
            
            # Moving averages
            if len(hist) >= 20:
                ma_20 = hist['Close'].rolling(20).mean().iloc[-1]
                factors += 1
                if current_price > ma_20:
                    score += 0.3
                else:
                    score -= 0.3
            
            if len(hist) >= 50:
                ma_50 = hist['Close'].rolling(50).mean().iloc[-1]
                factors += 1
                if current_price > ma_50:
                    score += 0.2
                else:
                    score -= 0.2
            
            # RSI
            if len(hist) >= 14:
                delta = hist['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1]
                
                factors += 1
                if current_rsi < 30:  # Oversold
                    score += 0.3
                elif current_rsi > 70:  # Overbought
                    score -= 0.3
                elif 40 <= current_rsi <= 60:  # Neutral
                    score += 0.1
            
            # Volume trend
            if len(hist) >= 10:
                avg_volume = hist['Volume'].rolling(10).mean().iloc[-1]
                recent_volume = hist['Volume'].iloc[-1]
                factors += 1
                if recent_volume > avg_volume * 1.5:  # High volume
                    score += 0.2
            
            return score / max(factors, 1) if factors > 0 else None
            
        except Exception as e:
            logger.warning(f"Technical analysis error for {symbol}: {e}")
            return None
    
    async def _analyze_fundamental(self, symbol: str) -> Optional[float]:
        """Fundamental analysis component."""
        try:
            # Use premium API if available
            if self.fmp_key and await self._check_rate_limit('fmp'):
                try:
                    if not self.session:
                        await self._ensure_session()
                    
                    url = f"https://financialmodelingprep.com/api/v3/key-metrics/{symbol}"
                    params = {'apikey': self.fmp_key}
                    
                    session = await self._ensure_session()
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data and isinstance(data, list) and data:
                                return self._score_fundamentals(data[0])
                except Exception as e:
                    logger.debug(f"FMP fundamental error: {e}")
            
            # Fallback to Yahoo Finance using utility function
            info = fetch_stock_info(symbol)
            if info:
                return self._score_fundamentals(info)
            
        except Exception as e:
            logger.warning(f"Fundamental analysis error for {symbol}: {e}")
            return None
    
    def _score_fundamentals(self, data: Dict) -> float:
        """Score fundamental data."""
        score = 0.0
        factors = 0
        
        # P/E Ratio
        pe = data.get('peRatio') or data.get('trailingPE')
        if pe and 0 < pe < 50:
            factors += 1
            if pe < 15:
                score += 0.3
            elif pe < 25:
                score += 0.1
            else:
                score -= 0.1
        
        # Revenue Growth
        growth = data.get('revenueGrowth') or data.get('revenue_growth')
        if growth is not None:
            factors += 1
            score += min(growth * 2, 0.4)  # Cap at 0.4
        
        # Profit Margin
        margin = data.get('profitMargins') or data.get('netProfitMargin')
        if margin is not None:
            factors += 1
            if margin > 0.15:
                score += 0.2
            elif margin > 0:
                score += 0.1
            else:
                score -= 0.3
        
        # Debt to Equity
        debt_equity = data.get('debtToEquity')
        if debt_equity is not None:
            factors += 1
            if debt_equity < 0.5:
                score += 0.1
            elif debt_equity > 2:
                score -= 0.2
        
        return score / max(factors, 1) if factors > 0 else 0.0
    
    async def _analyze_sentiment(self, symbol: str) -> Optional[float]:
        """Comprehensive LLM-powered sentiment analysis."""
        try:
            # Use the sentiment agent for comprehensive analysis
            from agents.sentiment_agent import sentiment_agent
            
            # Check cache first
            cache_key = f"sentiment_analysis_{symbol}"
            if cache_key in self.cache:
                data, timestamp = self.cache[cache_key]
                if (datetime.now() - timestamp).seconds < self.cache_duration:
                    return data
            
            # Get comprehensive sentiment
            comprehensive_sentiment = await sentiment_agent.analyze_comprehensive_sentiment(symbol)
            
            if comprehensive_sentiment:
                # Cache the result
                self.cache[cache_key] = (comprehensive_sentiment.overall_score, datetime.now())
                
                logger.info(f"LLM sentiment for {symbol}: {comprehensive_sentiment.overall_sentiment} "
                          f"({comprehensive_sentiment.overall_score:.3f}, confidence: {comprehensive_sentiment.confidence:.3f})")
                
                return comprehensive_sentiment.overall_score
            
            # FAIL-FAST: No fallback sentiment allowed
            error_msg = (
                f"❌ CRITICAL: LLM sentiment analysis failed for {symbol}\n"
                f"SYSTEM REQUIRES VALID SENTIMENT DATA TO OPERATE\n"
                f"No fallback mechanisms permitted per fail-fast design"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        except Exception as e:
            error_msg = (
                f"❌ CRITICAL: LLM sentiment analysis error for {symbol}: {e}\n"
                f"SYSTEM REQUIRES VALID SENTIMENT DATA TO OPERATE\n"
                f"No fallback mechanisms permitted per fail-fast design"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    # REMOVED: Fallback sentiment method - fail-fast architecture only
    # def _fallback_sentiment(self, symbol: str) -> float:
    #     DISABLED: No fallback sentiment allowed per fail-fast design
    
    async def _get_news_articles(self, symbol: str) -> List[Dict]:
        """Get news articles for sentiment analysis. Used by sentiment agent."""
        try:
            if not self.news_key or not await self._check_rate_limit('news_api'):
                return []
            
            if not self.session:
                await self._ensure_session()
            
            # Get recent news
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'"{symbol}"',
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': 20,  # Get more articles for LLM analysis
                'from': (datetime.now() - timedelta(days=2)).isoformat(),
                'apiKey': self.news_key
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                        if data and isinstance(data, dict) and 'articles' in data:
                            return data.get('articles', [])
                        else:
                            logger.debug(f"Invalid response format for {symbol}: {type(data)} - {data}")
                            return []
                    except Exception as json_error:
                        logger.warning(f"Failed to parse JSON response for {symbol}: {json_error}")
                        return []
                else:
                    logger.debug(f"News API returned status {response.status} for {symbol}")
                    return []
            
            return []
            
        except Exception as e:
            logger.warning(f"Error fetching news articles for {symbol}: {e}")
            return []
    
    async def _analyze_market_structure(self, symbol: str) -> Optional[float]:
        """Market structure and momentum analysis."""
        try:
            hist = fetch_stock_history(symbol, period="6mo")
            if hist is None or hist.empty or len(hist) < 50:
                return None
            
            score = 0.0
            
            # Trend strength
            prices = hist['Close'].values
            if len(prices) >= 50:
                short_trend = np.polyfit(range(20), prices[-20:], 1)[0]
                long_trend = np.polyfit(range(50), prices[-50:], 1)[0]
                
                # Normalize trends
                price_level = prices[-1]
                short_trend_norm = short_trend / price_level * 1000  # Scale for readability
                long_trend_norm = long_trend / price_level * 1000
                
                # Score based on trend alignment
                if short_trend_norm > 0 and long_trend_norm > 0:
                    score += 0.4  # Both trends positive
                elif short_trend_norm > long_trend_norm:
                    score += 0.2  # Accelerating uptrend
                elif short_trend_norm < 0 and long_trend_norm < 0:
                    score -= 0.4  # Both trends negative
                else:
                    score -= 0.2  # Mixed signals
            
            # Volatility assessment
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std()
            
            # Lower volatility = better structure
            if volatility < 0.02:
                score += 0.1
            elif volatility > 0.05:
                score -= 0.1
            
            return score
            
        except Exception as e:
            logger.warning(f"Market structure analysis error for {symbol}: {e}")
            return None
    
    def _score_to_signal(self, score: float) -> SignalType:
        """Convert score to signal."""
        if score > 0.4:
            return SignalType.STRONG_BUY
        elif score > 0.1:
            return SignalType.BUY
        elif score > -0.1:
            return SignalType.HOLD
        elif score > -0.4:
            return SignalType.SELL
        else:
            return SignalType.STRONG_SELL
    
    def _calculate_confidence(self, components: Dict[str, float], market_data: MarketData) -> ConfidenceLevel:
        """Calculate confidence level."""
        # Base confidence on data availability
        available_components = sum(1 for v in components.values() if v is not None)
        data_confidence = available_components / len(components)
        
        # Agreement between components
        valid_scores = [score for score in components.values() if score is not None]
        if len(valid_scores) >= 2:
            agreement = 1 - np.var(valid_scores)
        else:
            agreement = 0.5
        
        # Price data quality
        price_quality = 1.0 if market_data.source != "failed" else 0.0
        
        overall_confidence = (data_confidence * 0.4 + agreement * 0.4 + price_quality * 0.2)
        
        if overall_confidence >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif overall_confidence >= 0.6:
            return ConfidenceLevel.HIGH
        elif overall_confidence >= 0.4:
            return ConfidenceLevel.MEDIUM
        elif overall_confidence >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _calculate_targets(self, current_price: float, score: float) -> Tuple[Optional[float], Optional[float]]:
        """Calculate price targets."""
        try:
            target_price = current_price * (1 + score * 0.20)  # Max 20% move
            
            if score > 0:  # Long position
                stop_loss = current_price * 0.92  # 8% stop loss
            else:  # Short position
                stop_loss = current_price * 1.08  # 8% stop loss
            
            return target_price, stop_loss
        except:
            return None, None
    
    def _calculate_risk(self, components: Dict[str, float]) -> float:
        """Calculate risk score."""
        # Higher absolute scores = higher risk
        valid_scores = [abs(score) for score in components.values() if score is not None]
        if not valid_scores:
            return 0.5
        
        # Average absolute score as risk proxy
        risk = np.mean(valid_scores)
        return min(risk, 1.0)
    
    def _generate_reasoning(self, symbol: str, components: Dict[str, float], score: float) -> List[str]:
        """Generate human-readable reasoning."""
        reasoning = []
        
        # Component analysis
        for component, component_score in components.items():
            if component_score is None:
                continue
            
            if component_score > 0.2:
                reasoning.append(f"{component.title()} analysis is positive ({component_score:.2f})")
            elif component_score < -0.2:
                reasoning.append(f"{component.title()} analysis is negative ({component_score:.2f})")
        
        # Overall conclusion
        if score > 0.3:
            reasoning.append(f"Strong buy signal with overall score of {score:.2f}")
        elif score > 0.1:
            reasoning.append(f"Buy signal with positive momentum ({score:.2f})")
        elif score < -0.3:
            reasoning.append(f"Strong sell signal with overall score of {score:.2f}")
        elif score < -0.1:
            reasoning.append(f"Sell signal with negative indicators ({score:.2f})")
        else:
            reasoning.append(f"Hold recommendation with neutral score ({score:.2f})")
        
        return reasoning
    
    def _create_fallback_result(self, symbol: str, error: str) -> AnalysisResult:
        """Create fallback result for errors."""
        return AnalysisResult(
            symbol=symbol,
            signal=SignalType.HOLD,
            confidence=ConfidenceLevel.VERY_LOW,
            score=0.0,
            price=0.0,
            reasoning=[f"Analysis failed: {error}"]
        )
    
    async def _check_rate_limit(self, api_name: str) -> bool:
        """Check API rate limits."""
        limits = {'alpha_vantage': 5, 'finnhub': 60, 'fmp': 250, 'news_api': 1000}
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old calls
        self.last_calls[api_name] = [
            call_time for call_time in self.last_calls[api_name]
            if call_time > minute_ago
        ]
        
        # Check limit
        if len(self.last_calls[api_name]) >= limits.get(api_name, 5):
            return False
        
        self.last_calls[api_name].append(now)
        return True
    
    async def analyze_portfolio(self, symbols: List[str]) -> List[AnalysisResult]:
        """Analyze multiple stocks for portfolio decisions."""
        logger.info(f"📊 Analyzing portfolio of {len(symbols)} stocks")
        
        # Analyze stocks concurrently
        tasks = [self.analyze_stock(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for result in results:
            if isinstance(result, AnalysisResult):
                valid_results.append(result)
            else:
                logger.error(f"Analysis failed: {result}")
        
        return valid_results
    
    async def _analyze_crypto_momentum(self, symbol: str, current_price: float) -> Optional[float]:
        """Crypto-specific momentum analysis with 24/7 market patterns."""
        try:
            # Get extended historical data for crypto (24/7 market)
            yf_symbol = self._convert_crypto_symbol_for_yf(symbol)
            
            if yf_symbol:
                hist = fetch_stock_history(yf_symbol, period='7d', interval='1h')  # 7 days of hourly data
                
                if hist is not None and len(hist) >= 24:  # At least 24 hours of data
                    prices = hist['Close']
                    volumes = hist['Volume']
                    
                    # Crypto-specific momentum indicators
                    # 1. 24h momentum
                    momentum_24h = (prices.iloc[-1] / prices.iloc[-24] - 1) if len(prices) >= 24 else 0
                    
                    # 2. Volume-weighted momentum
                    recent_volume = volumes.iloc[-6:].mean()  # Last 6 hours
                    avg_volume = volumes.mean()
                    volume_factor = min(2.0, recent_volume / avg_volume) if avg_volume > 0 else 1.0
                    
                    # 3. Volatility-adjusted score (crypto has higher volatility tolerance)
                    volatility = prices.pct_change().std()
                    
                    # Combine factors with crypto-appropriate weights
                    score = 0.0
                    
                    # Strong 24h momentum (higher threshold for crypto)
                    if abs(momentum_24h) > 0.10:  # 10% daily move
                        score += 0.4 if momentum_24h > 0 else -0.4
                    
                    # Volume confirmation
                    if volume_factor > 1.2:
                        score += 0.3
                    
                    # Trend consistency (check if momentum is sustained)
                    if len(prices) >= 48:  # 48 hours
                        momentum_48h = (prices.iloc[-1] / prices.iloc[-48] - 1)
                        if momentum_24h * momentum_48h > 0:  # Same direction
                            score += 0.2
                    
                    logger.debug(f"Crypto momentum analysis for {symbol}: 24h={momentum_24h:.2%}, volume_factor={volume_factor:.2f}, score={score:.2f}")
                    return max(-1.0, min(1.0, score))
            
            return None
            
        except Exception as e:
            logger.warning(f"Crypto momentum analysis error for {symbol}: {e}")
            return None
    
    def _convert_crypto_symbol_for_yf(self, symbol: str) -> Optional[str]:
        """Convert crypto symbol to standardized format."""
        crypto_mapping = {
            'BTCUSD': 'BTC-USD',
            'ETHUSD': 'ETH-USD', 
            'DOGEUSD': 'DOGE-USD',
            'LTCUSD': 'LTC-USD',
            'BCHUSD': 'BCH-USD'
        }
        return crypto_mapping.get(symbol.upper())
    
    async def close(self):
        """Clean up resources."""
        await connection_pool.close_session("market_intelligence")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper cleanup."""
        await self.close()
    
    async def _ensure_session(self):
        """Get session from connection pool."""
        return await connection_pool.get_async_session(
            name="market_intelligence",
            timeout=aiohttp.ClientTimeout(total=20),
            connector_limit=20
        )

# Global instance
market_intelligence = UnifiedMarketIntelligence()

# Convenience functions
async def get_market_data(symbol: str) -> MarketData:
    """Get current market data for a symbol."""
    return await market_intelligence.get_market_data(symbol)

async def analyze_stock(symbol: str) -> AnalysisResult:
    """Analyze a single stock."""
    return await market_intelligence.analyze_stock(symbol)

async def analyze_portfolio(symbols: List[str]) -> List[AnalysisResult]:
    """Analyze multiple stocks."""
    return await market_intelligence.analyze_portfolio(symbols)

# Cleanup function
async def cleanup_market_intelligence():
    """Cleanup market intelligence resources."""
    await market_intelligence.close()

__all__ = [
    'UnifiedMarketIntelligence',
    'MarketData', 
    'AnalysisResult',
    'SignalType',
    'ConfidenceLevel',
    'market_intelligence',
    'get_market_data',
    'analyze_stock', 
    'analyze_portfolio',
    'cleanup_market_intelligence'
]