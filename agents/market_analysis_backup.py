"""Enhanced multi-source market analysis agents with comprehensive data aggregation."""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

warnings.filterwarnings('ignore')

# Core data libraries
import yfinance as yf
import yahoo_fin.stock_info as si

# Multi-source API clients
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.techindicators import TechIndicators
import finnhub
from newsapi import NewsApiClient

from agents.state import TradingSignal, MarketCondition
from config.settings import settings

logger = logging.getLogger(__name__)

class DataSource(Enum):
    """Available data sources for market analysis."""
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    FINNHUB = "finnhub"
    FMP = "financial_modeling_prep"
    NASDAQ = "nasdaq_data_link"
    SEC_EDGAR = "sec_edgar"
    NEWS_API = "news_api"

class DataReliability(Enum):
    """Data reliability scoring for source weighting."""
    HIGH = 1.0
    MEDIUM = 0.7
    LOW = 0.4
    UNAVAILABLE = 0.0

@dataclass
class DataPoint:
    """Individual data point with source attribution and reliability."""
    value: Any
    source: DataSource
    reliability: DataReliability
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class AggregatedData:
    """Aggregated data from multiple sources with confidence scoring."""
    value: Any
    confidence: float  # 0.0 to 1.0
    sources_used: List[DataSource]
    data_points: List[DataPoint]
    aggregation_method: str
    timestamp: datetime

class MultiSourceDataProvider:
    """Comprehensive data provider aggregating multiple financial APIs."""
    
    def __init__(self):
        """Initialize all available data sources."""
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        self.rate_limits = {
            DataSource.ALPHA_VANTAGE: {"calls": 0, "reset_time": time.time()},
            DataSource.FINNHUB: {"calls": 0, "reset_time": time.time()},
            DataSource.FMP: {"calls": 0, "reset_time": time.time()},
            DataSource.NASDAQ: {"calls": 0, "reset_time": time.time()},
            DataSource.NEWS_API: {"calls": 0, "reset_time": time.time()}
        }
        
        # Initialize API clients
        self._init_api_clients()
        
    def _init_api_clients(self):
        """Initialize all API clients with error handling."""
        try:
            # Alpha Vantage
            if settings.alpha_vantage_api_key:
                self.av_ts = TimeSeries(key=settings.alpha_vantage_api_key, output_format='pandas')
                self.av_fd = FundamentalData(key=settings.alpha_vantage_api_key, output_format='pandas')
                self.av_ti = TechIndicators(key=settings.alpha_vantage_api_key, output_format='pandas')
                logger.info("✅ Alpha Vantage client initialized")
            else:
                self.av_ts = self.av_fd = self.av_ti = None
                logger.warning("⚠️ Alpha Vantage API key not configured")
                
            # Finnhub
            if settings.finnhub_api_key:
                self.finnhub_client = finnhub.Client(api_key=settings.finnhub_api_key)
                logger.info("✅ Finnhub client initialized")
            else:
                self.finnhub_client = None
                logger.warning("⚠️ Finnhub API key not configured")
                
            # NewsAPI
            if settings.news_api_key:
                self.news_client = NewsApiClient(api_key=settings.news_api_key)
                logger.info("✅ NewsAPI client initialized")
            else:
                self.news_client = None
                logger.warning("⚠️ NewsAPI key not configured")
                
        except Exception as e:
            logger.error(f"Error initializing API clients: {e}")
    
    def _check_rate_limit(self, source: DataSource) -> bool:
        """Check if we're within rate limits for a data source."""
        current_time = time.time()
        rate_info = self.rate_limits[source]
        
        # Reset counters every hour
        if current_time - rate_info["reset_time"] > 3600:
            rate_info["calls"] = 0
            rate_info["reset_time"] = current_time
        
        # Different limits per source
        limits = {
            DataSource.ALPHA_VANTAGE: 500,  # 500 calls/day
            DataSource.FINNHUB: 1000,      # 1000 calls/day free tier
            DataSource.FMP: 250,           # 250 calls/day free tier
            DataSource.NASDAQ: 50,         # Conservative limit
            DataSource.NEWS_API: 500       # 500 requests/day free tier
        }
        
        if rate_info["calls"] >= limits.get(source, 100):
            logger.warning(f"Rate limit reached for {source.value}")
            return False
            
        rate_info["calls"] += 1
        return True
    
    async def get_stock_data_multi_source(self, symbol: str, period: str = "1y") -> AggregatedData:
        """Get historical stock data from multiple sources."""
        data_points = []
        
        # Yahoo Finance (primary)
        try:
            yf_data = yf.Ticker(symbol).history(period=period)
            if not yf_data.empty:
                data_points.append(DataPoint(
                    value=yf_data,
                    source=DataSource.YAHOO_FINANCE,
                    reliability=DataReliability.HIGH,
                    timestamp=datetime.now(),
                    metadata={"period": period, "source_priority": 1}
                ))
        except Exception as e:
            logger.warning(f"Yahoo Finance error for {symbol}: {e}")
        
        # Alpha Vantage (secondary)
        if self.av_ts and self._check_rate_limit(DataSource.ALPHA_VANTAGE):
            try:
                av_data, _ = self.av_ts.get_daily_adjusted(symbol=symbol, outputsize='full')
                if av_data is not None and not av_data.empty:
                    data_points.append(DataPoint(
                        value=av_data,
                        source=DataSource.ALPHA_VANTAGE,
                        reliability=DataReliability.HIGH,
                        timestamp=datetime.now(),
                        metadata={"period": period, "source_priority": 2}
                    ))
            except Exception as e:
                logger.warning(f"Alpha Vantage error for {symbol}: {e}")
        
        # Finnhub (tertiary)
        if self.finnhub_client and self._check_rate_limit(DataSource.FINNHUB):
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
                
                candles = self.finnhub_client.stock_candles(
                    symbol, 'D', 
                    int(start_date.timestamp()), 
                    int(end_date.timestamp())
                )
                
                if candles['s'] == 'ok':
                    # Convert to DataFrame format
                    fh_data = pd.DataFrame({
                        'open': candles['o'],
                        'high': candles['h'], 
                        'low': candles['l'],
                        'close': candles['c'],
                        'volume': candles['v']
                    }, index=pd.to_datetime(candles['t'], unit='s'))
                    
                    data_points.append(DataPoint(
                        value=fh_data,
                        source=DataSource.FINNHUB,
                        reliability=DataReliability.MEDIUM,
                        timestamp=datetime.now(),
                        metadata={"period": period, "source_priority": 3}
                    ))
            except Exception as e:
                logger.warning(f"Finnhub error for {symbol}: {e}")
        
        return self._aggregate_stock_data(data_points, symbol)
    
    def _aggregate_stock_data(self, data_points: List[DataPoint], symbol: str) -> AggregatedData:
        """Aggregate stock data from multiple sources."""
        if not data_points:
            return AggregatedData(
                value=pd.DataFrame(),
                confidence=0.0,
                sources_used=[],
                data_points=[],
                aggregation_method="no_data",
                timestamp=datetime.now()
            )
        
        # Use highest priority available source as primary
        primary_data = max(data_points, key=lambda x: x.metadata.get('source_priority', 0))
        
        # Calculate confidence based on number of sources and their reliability
        total_reliability = sum(dp.reliability.value for dp in data_points)
        confidence = min(total_reliability / len(data_points), 1.0)
        
        return AggregatedData(
            value=primary_data.value,
            confidence=confidence,
            sources_used=[dp.source for dp in data_points],
            data_points=data_points,
            aggregation_method="weighted_primary",
            timestamp=datetime.now()
        )
    
    async def get_financial_metrics_multi_source(self, symbol: str) -> AggregatedData:
        """Get financial metrics from multiple sources."""
        data_points = []
        
        # Yahoo Finance
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            metrics = {
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "forward_pe": info.get("forwardPE", 0),
                "peg_ratio": info.get("pegRatio", 0),
                "price_to_book": info.get("priceToBook", 0),
                "debt_to_equity": info.get("debtToEquity", 0),
                "roe": info.get("returnOnEquity", 0),
                "roa": info.get("returnOnAssets", 0),
                "profit_margin": info.get("profitMargins", 0),
                "revenue_growth": info.get("revenueGrowth", 0),
                "earnings_growth": info.get("earningsGrowth", 0),
                "beta": info.get("beta", 1.0),
                "dividend_yield": info.get("dividendYield", 0),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown")
            }
            
            data_points.append(DataPoint(
                value=metrics,
                source=DataSource.YAHOO_FINANCE,
                reliability=DataReliability.HIGH,
                timestamp=datetime.now(),
                metadata={"source_priority": 1}
            ))
        except Exception as e:
            logger.warning(f"Yahoo Finance metrics error for {symbol}: {e}")
        
        # Alpha Vantage Fundamentals
        if self.av_fd and self._check_rate_limit(DataSource.ALPHA_VANTAGE):
            try:
                overview = self.av_fd.get_company_overview(symbol=symbol)[0]
                if not overview.empty:
                    av_metrics = {
                        "market_cap": float(overview.get("MarketCapitalization", 0) or 0),
                        "pe_ratio": float(overview.get("PERatio", 0) or 0),
                        "forward_pe": float(overview.get("ForwardPE", 0) or 0),
                        "peg_ratio": float(overview.get("PEGRatio", 0) or 0),
                        "price_to_book": float(overview.get("PriceToBookRatio", 0) or 0),
                        "roe": float(overview.get("ReturnOnEquityTTM", 0) or 0),
                        "roa": float(overview.get("ReturnOnAssetsTTM", 0) or 0),
                        "profit_margin": float(overview.get("ProfitMargin", 0) or 0),
                        "revenue_growth": float(overview.get("QuarterlyRevenueGrowthYOY", 0) or 0),
                        "earnings_growth": float(overview.get("QuarterlyEarningsGrowthYOY", 0) or 0),
                        "beta": float(overview.get("Beta", 1.0) or 1.0),
                        "dividend_yield": float(overview.get("DividendYield", 0) or 0),
                        "sector": overview.get("Sector", "Unknown"),
                        "industry": overview.get("Industry", "Unknown")
                    }
                    
                    data_points.append(DataPoint(
                        value=av_metrics,
                        source=DataSource.ALPHA_VANTAGE,
                        reliability=DataReliability.HIGH,
                        timestamp=datetime.now(),
                        metadata={"source_priority": 2}
                    ))
            except Exception as e:
                logger.warning(f"Alpha Vantage fundamentals error for {symbol}: {e}")
        
        # Finnhub Company Profile
        if self.finnhub_client and self._check_rate_limit(DataSource.FINNHUB):
            try:
                profile = self.finnhub_client.company_profile2(symbol=symbol)
                metrics = self.finnhub_client.company_basic_financials(symbol, 'all')
                
                fh_metrics = {
                    "market_cap": float(profile.get("marketCapitalization", 0) * 1000000),  # Convert from millions
                    "beta": float(metrics.get("metric", {}).get("beta", 1.0) or 1.0),
                    "pe_ratio": float(metrics.get("metric", {}).get("peBasicExclExtraTTM", 0) or 0),
                    "price_to_book": float(metrics.get("metric", {}).get("pbAnnual", 0) or 0),
                    "roe": float(metrics.get("metric", {}).get("roeRfy", 0) or 0),
                    "roa": float(metrics.get("metric", {}).get("roaRfy", 0) or 0),
                    "sector": profile.get("finnhubIndustry", "Unknown"),
                    "country": profile.get("country", "Unknown")
                }
                
                data_points.append(DataPoint(
                    value=fh_metrics,
                    source=DataSource.FINNHUB,
                    reliability=DataReliability.MEDIUM,
                    timestamp=datetime.now(),
                    metadata={"source_priority": 3}
                ))
            except Exception as e:
                logger.warning(f"Finnhub metrics error for {symbol}: {e}")
        
        # Financial Modeling Prep API
        if settings.fmp_api_key and self._check_rate_limit(DataSource.FMP):
            try:
                fmp_url = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{symbol}?apikey={settings.fmp_api_key}"
                response = requests.get(fmp_url, timeout=10)
                
                if response.status_code == 200:
                    fmp_data = response.json()
                    if fmp_data:
                        data = fmp_data[0]
                        fmp_metrics = {
                            "pe_ratio": float(data.get("priceEarningsRatioTTM", 0) or 0),
                            "price_to_book": float(data.get("priceToBookRatioTTM", 0) or 0),
                            "debt_to_equity": float(data.get("debtEquityRatioTTM", 0) or 0),
                            "roe": float(data.get("returnOnEquityTTM", 0) or 0),
                            "roa": float(data.get("returnOnAssetsTTM", 0) or 0),
                            "profit_margin": float(data.get("netProfitMarginTTM", 0) or 0),
                            "current_ratio": float(data.get("currentRatioTTM", 0) or 0),
                            "quick_ratio": float(data.get("quickRatioTTM", 0) or 0)
                        }
                        
                        data_points.append(DataPoint(
                            value=fmp_metrics,
                            source=DataSource.FMP,
                            reliability=DataReliability.HIGH,
                            timestamp=datetime.now(),
                            metadata={"source_priority": 4}
                        ))
            except Exception as e:
                logger.warning(f"FMP metrics error for {symbol}: {e}")
        
        return self._aggregate_financial_metrics(data_points, symbol)
    
    def _aggregate_financial_metrics(self, data_points: List[DataPoint], symbol: str) -> AggregatedData:
        """Aggregate financial metrics from multiple sources using weighted averaging."""
        if not data_points:
            return AggregatedData(
                value={},
                confidence=0.0,
                sources_used=[],
                data_points=[],
                aggregation_method="no_data",
                timestamp=datetime.now()
            )
        
        # Aggregate common metrics using weighted average
        all_metrics = {}
        metric_values = {}
        
        # Collect all metrics across sources
        for dp in data_points:
            for metric, value in dp.value.items():
                if metric not in metric_values:
                    metric_values[metric] = []
                
                # Weight by reliability
                weight = dp.reliability.value
                if isinstance(value, (int, float)) and value > 0:
                    metric_values[metric].append((value, weight))
        
        # Calculate weighted averages
        for metric, values in metric_values.items():
            if values and all(isinstance(v[0], (int, float)) for v in values):
                # Weighted average for numeric values
                weighted_sum = sum(val * weight for val, weight in values)
                total_weight = sum(weight for _, weight in values)
                all_metrics[metric] = weighted_sum / total_weight if total_weight > 0 else 0
            elif values:
                # Take most reliable source for non-numeric values
                all_metrics[metric] = max(values, key=lambda x: x[1])[0]
        
        # Calculate confidence
        total_reliability = sum(dp.reliability.value for dp in data_points)
        confidence = min(total_reliability / len(data_points), 1.0)
        
        return AggregatedData(
            value=all_metrics,
            confidence=confidence,
            sources_used=[dp.source for dp in data_points],
            data_points=data_points,
            aggregation_method="weighted_average",
            timestamp=datetime.now()
        )
    
    async def get_news_sentiment_multi_source(self, symbol: str, company_name: str = None) -> AggregatedData:
        """Get news sentiment from multiple sources."""
        data_points = []
        
        # Yahoo Finance News
        try:
            ticker = yf.Ticker(symbol)
            news_data = ticker.news
            
            if news_data:
                yf_sentiment = self._analyze_news_sentiment(news_data[:10], symbol)
                data_points.append(DataPoint(
                    value=yf_sentiment,
                    source=DataSource.YAHOO_FINANCE,
                    reliability=DataReliability.MEDIUM,
                    timestamp=datetime.now(),
                    metadata={"articles_count": len(news_data[:10])}
                ))
        except Exception as e:
            logger.warning(f"Yahoo Finance news error for {symbol}: {e}")
        
        # Finnhub News
        if self.finnhub_client and self._check_rate_limit(DataSource.FINNHUB):
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)
                
                news = self.finnhub_client.company_news(
                    symbol, 
                    _from=start_date.strftime('%Y-%m-%d'),
                    to=end_date.strftime('%Y-%m-%d')
                )
                
                if news:
                    fh_sentiment = self._analyze_finnhub_news(news[:15], symbol)
                    data_points.append(DataPoint(
                        value=fh_sentiment,
                        source=DataSource.FINNHUB,
                        reliability=DataReliability.HIGH,
                        timestamp=datetime.now(),
                        metadata={"articles_count": len(news[:15])}
                    ))
            except Exception as e:
                logger.warning(f"Finnhub news error for {symbol}: {e}")
        
        # NewsAPI
        if self.news_client and self._check_rate_limit(DataSource.NEWS_API):
            try:
                # Search for company-specific news
                query = company_name if company_name else symbol
                news = self.news_client.get_everything(
                    q=query,
                    language='en',
                    sort_by='relevancy',
                    page_size=20,
                    from_param=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
                )
                
                if news['articles']:
                    na_sentiment = self._analyze_newsapi_sentiment(news['articles'], symbol)
                    data_points.append(DataPoint(
                        value=na_sentiment,
                        source=DataSource.NEWS_API,
                        reliability=DataReliability.HIGH,
                        timestamp=datetime.now(),
                        metadata={"articles_count": len(news['articles'])}
                    ))
            except Exception as e:
                logger.warning(f"NewsAPI error for {symbol}: {e}")
        
        return self._aggregate_news_sentiment(data_points, symbol)
    
    def _analyze_news_sentiment(self, articles: List[Dict], symbol: str) -> Dict[str, Any]:
        """Analyze sentiment from Yahoo Finance news."""
        positive_keywords = [
            'growth', 'profit', 'gain', 'rise', 'increase', 'beat', 'exceeds', 'strong', 
            'positive', 'upgrade', 'buy', 'bullish', 'surge', 'rally', 'breakthrough',
            'success', 'record', 'expansion', 'acquisition', 'innovation'
        ]
        
        negative_keywords = [
            'loss', 'decline', 'fall', 'drop', 'decrease', 'miss', 'disappoints', 'weak',
            'negative', 'downgrade', 'sell', 'bearish', 'crash', 'plunge', 'concern',
            'risk', 'warning', 'lawsuit', 'investigation', 'bankruptcy', 'layoffs'
        ]
        
        sentiment_scores = []
        processed_articles = []
        
        for article in articles:
            title = article.get('title', '').lower()
            summary = article.get('summary', '').lower()
            text = f"{title} {summary}"
            
            positive_count = sum(1 for keyword in positive_keywords if keyword in text)
            negative_count = sum(1 for keyword in negative_keywords if keyword in text)
            
            # Sentiment scoring with neutral bias
            if positive_count > negative_count:
                sentiment = 0.6 + min(0.3, (positive_count - negative_count) * 0.1)
            elif negative_count > positive_count:
                sentiment = 0.4 - min(0.3, (negative_count - positive_count) * 0.1)
            else:
                sentiment = 0.5
            
            sentiment_scores.append(sentiment)
            processed_articles.append({
                'title': article.get('title', ''),
                'source': article.get('publisher', ''),
                'sentiment': sentiment,
                'relevance': 1.0 if symbol.lower() in text else 0.5
            })
        
        # Weighted average by relevance
        if sentiment_scores:
            weights = [art['relevance'] for art in processed_articles]
            weighted_sentiment = np.average(sentiment_scores, weights=weights)
        else:
            weighted_sentiment = 0.5
        
        return {
            'sentiment_score': weighted_sentiment,
            'article_count': len(processed_articles),
            'articles': processed_articles[:5],
            'confidence': min(len(processed_articles) / 10, 1.0)
        }
    
    def _analyze_finnhub_news(self, articles: List[Dict], symbol: str) -> Dict[str, Any]:
        """Analyze sentiment from Finnhub news."""
        sentiment_scores = []
        
        for article in articles:
            headline = article.get('headline', '').lower()
            summary = article.get('summary', '').lower()
            
            # Simple sentiment analysis based on headline sentiment
            positive_words = ['growth', 'profit', 'beat', 'strong', 'positive', 'upgrade']
            negative_words = ['loss', 'decline', 'miss', 'weak', 'negative', 'downgrade']
            
            text = f"{headline} {summary}"
            pos_score = sum(1 for word in positive_words if word in text)
            neg_score = sum(1 for word in negative_words if word in text)
            
            if pos_score > neg_score:
                sentiment = 0.65
            elif neg_score > pos_score:
                sentiment = 0.35
            else:
                sentiment = 0.5
                
            sentiment_scores.append(sentiment)
        
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.5
        
        return {
            'sentiment_score': avg_sentiment,
            'article_count': len(articles),
            'confidence': min(len(articles) / 15, 1.0)
        }
    
    def _analyze_newsapi_sentiment(self, articles: List[Dict], symbol: str) -> Dict[str, Any]:
        """Analyze sentiment from NewsAPI articles."""
        sentiment_scores = []
        
        for article in articles:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            
            text = f"{title} {description}"
            
            # Enhanced sentiment keywords
            positive_indicators = [
                'surge', 'soar', 'rally', 'jump', 'gain', 'rise', 'increase', 'growth',
                'profit', 'revenue', 'beat', 'exceed', 'outperform', 'strong', 'robust',
                'positive', 'optimistic', 'bullish', 'upgrade', 'buy', 'recommendation'
            ]
            
            negative_indicators = [
                'plunge', 'crash', 'fall', 'drop', 'decline', 'decrease', 'loss',
                'deficit', 'miss', 'disappoint', 'underperform', 'weak', 'poor',
                'negative', 'pessimistic', 'bearish', 'downgrade', 'sell', 'warning'
            ]
            
            pos_count = sum(1 for indicator in positive_indicators if indicator in text)
            neg_count = sum(1 for indicator in negative_indicators if indicator in text)
            
            # More nuanced scoring
            if pos_count > neg_count:
                sentiment = 0.6 + min(0.35, pos_count * 0.1)
            elif neg_count > pos_count:
                sentiment = 0.4 - min(0.35, neg_count * 0.1)
            else:
                sentiment = 0.5
            
            sentiment_scores.append(sentiment)
        
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.5
        
        return {
            'sentiment_score': avg_sentiment,
            'article_count': len(articles),
            'confidence': min(len(articles) / 20, 1.0)
        }
    
    def _aggregate_news_sentiment(self, data_points: List[DataPoint], symbol: str) -> AggregatedData:
        """Aggregate news sentiment from multiple sources."""
        if not data_points:
            return AggregatedData(
                value={'sentiment_score': 0.5, 'confidence': 0.0, 'article_count': 0},
                confidence=0.0,
                sources_used=[],
                data_points=[],
                aggregation_method="no_data",
                timestamp=datetime.now()
            )
        
        # Weight sentiment by source reliability and article count
        weighted_sentiments = []
        total_articles = 0
        
        for dp in data_points:
            sentiment = dp.value['sentiment_score']
            article_count = dp.value['article_count']
            reliability = dp.reliability.value
            
            # Weight by both reliability and article count
            weight = reliability * min(article_count / 10, 1.0)
            weighted_sentiments.append((sentiment, weight))
            total_articles += article_count
        
        # Calculate weighted average sentiment
        if weighted_sentiments:
            total_weight = sum(weight for _, weight in weighted_sentiments)
            if total_weight > 0:
                avg_sentiment = sum(sent * weight for sent, weight in weighted_sentiments) / total_weight
            else:
                avg_sentiment = 0.5
        else:
            avg_sentiment = 0.5
        
        # Overall confidence based on sources and article count
        source_confidence = len(data_points) / 3  # Normalize by max expected sources
        article_confidence = min(total_articles / 30, 1.0)  # Normalize by target article count
        overall_confidence = min((source_confidence + article_confidence) / 2, 1.0)
        
        aggregated_value = {
            'sentiment_score': avg_sentiment,
            'confidence': overall_confidence,
            'article_count': total_articles,
            'source_breakdown': {dp.source.value: dp.value for dp in data_points}
        }
        
        return AggregatedData(
            value=aggregated_value,
            confidence=overall_confidence,
            sources_used=[dp.source for dp in data_points],
            data_points=data_points,
            aggregation_method="weighted_sentiment",
            timestamp=datetime.now()
        )
    
    async def get_technical_indicators_multi_source(self, symbol: str, period: str = "1y") -> AggregatedData:
        """Get technical indicators from multiple sources."""
        data_points = []
        
        # Get base stock data
        stock_data = await self.get_stock_data_multi_source(symbol, period)
        if stock_data.confidence == 0:
            return AggregatedData(
                value={},
                confidence=0.0,
                sources_used=[],
                data_points=[],
                aggregation_method="no_data",
                timestamp=datetime.now()
            )
        
        df = stock_data.value
        
        # Calculate indicators from stock data
        try:
            indicators = {}
            
            # Price-based indicators
            indicators['current_price'] = float(df['Close'].iloc[-1])
            indicators['sma_20'] = df['Close'].rolling(20).mean().iloc[-1]
            indicators['sma_50'] = df['Close'].rolling(50).mean().iloc[-1]
            indicators['ema_12'] = df['Close'].ewm(span=12).mean().iloc[-1]
            indicators['ema_26'] = df['Close'].ewm(span=26).mean().iloc[-1]
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
            
            # MACD
            macd_line = indicators['ema_12'] - indicators['ema_26']
            macd_signal = pd.Series([macd_line]).ewm(span=9).mean().iloc[0]
            indicators['macd'] = macd_line
            indicators['macd_signal'] = macd_signal
            indicators['macd_histogram'] = macd_line - macd_signal
            
            # Bollinger Bands
            sma_20 = df['Close'].rolling(20).mean()
            std_20 = df['Close'].rolling(20).std()
            indicators['bb_upper'] = (sma_20 + (std_20 * 2)).iloc[-1]
            indicators['bb_lower'] = (sma_20 - (std_20 * 2)).iloc[-1]
            
            # Volatility
            indicators['volatility'] = df['Close'].pct_change().std() * np.sqrt(252)
            
            # Volume indicators
            if 'Volume' in df.columns:
                indicators['volume_sma'] = df['Volume'].rolling(20).mean().iloc[-1]
                indicators['volume_ratio'] = df['Volume'].iloc[-1] / indicators['volume_sma']
            
            data_points.append(DataPoint(
                value=indicators,
                source=DataSource.YAHOO_FINANCE,
                reliability=DataReliability.HIGH,
                timestamp=datetime.now(),
                metadata={"calculation_method": "pandas_ta"}
            ))
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
        
        # Alpha Vantage technical indicators
        if self.av_ti and self._check_rate_limit(DataSource.ALPHA_VANTAGE):
            try:
                # Get RSI from Alpha Vantage
                rsi_data, _ = self.av_ti.get_rsi(symbol=symbol, interval='daily', time_period=14)
                if rsi_data is not None and not rsi_data.empty:
                    av_indicators = {
                        'rsi_av': float(rsi_data.iloc[-1]['RSI'])
                    }
                    
                    data_points.append(DataPoint(
                        value=av_indicators,
                        source=DataSource.ALPHA_VANTAGE,
                        reliability=DataReliability.HIGH,
                        timestamp=datetime.now(),
                        metadata={"indicator_source": "alpha_vantage_api"}
                    ))
            except Exception as e:
                logger.warning(f"Alpha Vantage technical indicators error: {e}")
        
        return self._aggregate_technical_indicators(data_points, symbol)
    
    async def get_economic_data_nasdaq(self, dataset_code: str = "NASDAQOMX/COMP") -> AggregatedData:
        """Get economic data from Nasdaq Data Link (formerly Quandl)."""
        data_points = []
        
        if settings.nasdaq_api_key and self._check_rate_limit(DataSource.NASDAQ):
            try:
                nasdaq_url = f"https://data.nasdaq.com/api/v3/datasets/{dataset_code}.json?api_key={settings.nasdaq_api_key}&limit=30"
                response = requests.get(nasdaq_url, timeout=10)
                
                if response.status_code == 200:
                    nasdaq_data = response.json()
                    dataset = nasdaq_data.get('dataset', {})
                    data = dataset.get('data', [])
                    
                    if data:
                        # Convert to DataFrame
                        columns = dataset.get('column_names', [])
                        df = pd.DataFrame(data, columns=columns)
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                        
                        economic_metrics = {
                            'latest_value': df.iloc[0, 0] if len(df) > 0 else 0,
                            'change_1d': ((df.iloc[0, 0] - df.iloc[1, 0]) / df.iloc[1, 0]) if len(df) > 1 else 0,
                            'change_1w': ((df.iloc[0, 0] - df.iloc[6, 0]) / df.iloc[6, 0]) if len(df) > 6 else 0,
                            'change_1m': ((df.iloc[0, 0] - df.iloc[21, 0]) / df.iloc[21, 0]) if len(df) > 21 else 0,
                            'volatility': df.iloc[:, 0].pct_change().std() * np.sqrt(252),
                            'dataset_code': dataset_code
                        }
                        
                        data_points.append(DataPoint(
                            value=economic_metrics,
                            source=DataSource.NASDAQ,
                            reliability=DataReliability.HIGH,
                            timestamp=datetime.now(),
                            metadata={"dataset": dataset_code}
                        ))
                        
            except Exception as e:
                logger.warning(f"Nasdaq Data Link error for {dataset_code}: {e}")
        
        return AggregatedData(
            value=data_points[0].value if data_points else {},
            confidence=1.0 if data_points else 0.0,
            sources_used=[dp.source for dp in data_points],
            data_points=data_points,
            aggregation_method="nasdaq_direct",
            timestamp=datetime.now()
        )
    
    async def get_sec_filings_summary(self, symbol: str) -> AggregatedData:
        """Get SEC filing information for fundamental analysis."""
        data_points = []
        
        try:
            # SEC Edgar API (no key required, but rate limited)
            sec_url = f"https://data.sec.gov/submissions/CIK{self._get_cik_for_symbol(symbol)}.json"
            
            headers = {
                'User-Agent': 'Trading System 1.0 (contact@example.com)',  # Required by SEC
                'Accept-Encoding': 'gzip, deflate',
                'Host': 'data.sec.gov'
            }
            
            response = requests.get(sec_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                sec_data = response.json()
                filings = sec_data.get('filings', {}).get('recent', {})
                
                # Extract recent 10-K and 10-Q filings
                forms = filings.get('form', [])
                filing_dates = filings.get('filingDate', [])
                
                recent_10k = None
                recent_10q = None
                
                for i, form in enumerate(forms[:20]):  # Check last 20 filings
                    if form == '10-K' and not recent_10k:
                        recent_10k = filing_dates[i]
                    elif form == '10-Q' and not recent_10q:
                        recent_10q = filing_dates[i]
                
                sec_summary = {
                    'recent_10k_date': recent_10k,
                    'recent_10q_date': recent_10q,
                    'total_filings': len(forms),
                    'filing_frequency': len([f for f in forms[:30] if f in ['10-K', '10-Q']]),
                    'compliance_score': 1.0 if recent_10k and recent_10q else 0.5
                }
                
                data_points.append(DataPoint(
                    value=sec_summary,
                    source=DataSource.SEC_EDGAR,
                    reliability=DataReliability.HIGH,
                    timestamp=datetime.now(),
                    metadata={"filing_analysis": True}
                ))
                
        except Exception as e:
            logger.warning(f"SEC Edgar error for {symbol}: {e}")
        
        return AggregatedData(
            value=data_points[0].value if data_points else {},
            confidence=1.0 if data_points else 0.0,
            sources_used=[dp.source for dp in data_points],
            data_points=data_points,
            aggregation_method="sec_direct",
            timestamp=datetime.now()
        )
    
    def _get_cik_for_symbol(self, symbol: str) -> str:
        """Get CIK (Central Index Key) for a stock symbol from SEC mapping."""
        try:
            # SEC maintains a mapping of tickers to CIKs
            mapping_url = "https://www.sec.gov/files/company_tickers.json"
            headers = {
                'User-Agent': 'Trading System 1.0 (contact@example.com)',
                'Accept-Encoding': 'gzip, deflate',
                'Host': 'www.sec.gov'
            }
            
            response = requests.get(mapping_url, headers=headers, timeout=10)
            if response.status_code == 200:
                ticker_data = response.json()
                
                for entry in ticker_data.values():
                    if entry.get('ticker', '').upper() == symbol.upper():
                        cik = str(entry.get('cik_str', '')).zfill(10)  # Pad with zeros
                        return cik
                        
        except Exception as e:
            logger.warning(f"Error getting CIK for {symbol}: {e}")
        
        # Fallback: return placeholder CIK (this will fail gracefully)
        return "0000000000"
    
    def _aggregate_technical_indicators(self, data_points: List[DataPoint], symbol: str) -> AggregatedData:
        """Aggregate technical indicators from multiple sources."""
        if not data_points:
            return AggregatedData(
                value={},
                confidence=0.0,
                sources_used=[],
                data_points=[],
                aggregation_method="no_data",
                timestamp=datetime.now()
            )
        
        # Merge all indicators
        all_indicators = {}
        for dp in data_points:
            all_indicators.update(dp.value)
        
        # Calculate confidence
        confidence = min(sum(dp.reliability.value for dp in data_points) / len(data_points), 1.0)
        
        return AggregatedData(
            value=all_indicators,
            confidence=confidence,
            sources_used=[dp.source for dp in data_points],
            data_points=data_points,
            aggregation_method="merged_indicators",
            timestamp=datetime.now()
        )

# Global multi-source data provider
multi_source_data = MultiSourceDataProvider()

# Update existing classes to use new data provider
class SignalStrength(Enum):
    VERY_WEAK = 0.2
    WEAK = 0.4
    MODERATE = 0.6
    STRONG = 0.8
    VERY_STRONG = 1.0

class MarketRegime(Enum):
    BULL = "bullish"
    BEAR = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

@dataclass
class AnalysisResult:
    """Enhanced analysis result with data source attribution."""
    symbol: str
    signal_type: str
    action: str
    confidence: float
    strength: SignalStrength
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    reasoning: str = ""
    indicators: Dict[str, float] = None
    data_sources: List[DataSource] = None
    data_confidence: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.indicators is None:
            self.indicators = {}
        if self.data_sources is None:
            self.data_sources = []

class EnhancedTechnicalAnalysisAgent:
    """Enhanced technical analysis agent using multi-source data."""
    
    def __init__(self):
        self.name = "Enhanced Multi-Source Technical Analysis Agent"
        self.lookback_period = 100
        self.min_data_points = 50
    
    async def analyze_symbol(self, symbol: str) -> AnalysisResult:
        """Perform comprehensive technical analysis using multiple data sources."""
        try:
            logger.info(f"Multi-source technical analysis for {symbol}")
            
            # Get technical indicators from multiple sources
            indicators_data = await multi_source_data.get_technical_indicators_multi_source(symbol)
            
            if indicators_data.confidence < 0.3:
                return AnalysisResult(
                    symbol=symbol,
                    signal_type="insufficient_data",
                    action="hold",
                    confidence=0.0,
                    strength=SignalStrength.VERY_WEAK,
                    reasoning="Insufficient data from multiple sources",
                    data_sources=indicators_data.sources_used,
                    data_confidence=indicators_data.confidence
                )
            
            indicators = indicators_data.value
            
            # Enhanced signal analysis with multi-source validation
            signals = []
            
            # RSI Analysis (with cross-validation if available)
            rsi = indicators.get('rsi', 50)
            rsi_av = indicators.get('rsi_av', rsi)  # Alpha Vantage RSI for validation
            
            # Average RSI values if multiple sources available
            if abs(rsi - rsi_av) < 10:  # Values are consistent
                avg_rsi = (rsi + rsi_av) / 2
                rsi_confidence = 1.0
            else:
                avg_rsi = rsi  # Use primary source
                rsi_confidence = 0.7
            
            if avg_rsi < 30:
                signals.append(("rsi_oversold", "buy", 0.7 * rsi_confidence))
            elif avg_rsi > 70:
                signals.append(("rsi_overbought", "sell", 0.7 * rsi_confidence))
            
            # MACD Analysis
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            macd_histogram = indicators.get('macd_histogram', 0)
            
            if macd > macd_signal and macd_histogram > 0:
                signals.append(("macd_bullish", "buy", 0.6))
            elif macd < macd_signal and macd_histogram < 0:
                signals.append(("macd_bearish", "sell", 0.6))
            
            # Moving Average Analysis
            current_price = indicators.get('current_price', 0)
            sma_20 = indicators.get('sma_20', 0)
            sma_50 = indicators.get('sma_50', 0)
            
            if current_price > sma_20 > sma_50:
                signals.append(("ma_bullish", "buy", 0.5))
            elif current_price < sma_20 < sma_50:
                signals.append(("ma_bearish", "sell", 0.5))
            
            # Bollinger Bands Analysis
            bb_upper = indicators.get('bb_upper', 0)
            bb_lower = indicators.get('bb_lower', 0)
            
            if current_price <= bb_lower:
                signals.append(("bb_oversold", "buy", 0.6))
            elif current_price >= bb_upper:
                signals.append(("bb_overbought", "sell", 0.6))
            
            # Aggregate signals
            buy_signals = [s for s in signals if s[1] == "buy"]
            sell_signals = [s for s in signals if s[1] == "sell"]
            
            if buy_signals:
                action = "buy"
                confidence = sum(s[2] for s in buy_signals) / len(buy_signals)
                reasoning = f"Technical buy signals: {', '.join(s[0] for s in buy_signals)}"
            elif sell_signals:
                action = "sell"
                confidence = sum(s[2] for s in sell_signals) / len(sell_signals)
                reasoning = f"Technical sell signals: {', '.join(s[0] for s in sell_signals)}"
            else:
                action = "hold"
                confidence = 0.5
                reasoning = "Mixed or neutral technical signals"
            
            # Adjust confidence by data quality
            final_confidence = confidence * indicators_data.confidence
            
            # Determine signal strength
            if final_confidence >= 0.8:
                strength = SignalStrength.VERY_STRONG
            elif final_confidence >= 0.6:
                strength = SignalStrength.STRONG
            elif final_confidence >= 0.4:
                strength = SignalStrength.MODERATE
            elif final_confidence >= 0.2:
                strength = SignalStrength.WEAK
            else:
                strength = SignalStrength.VERY_WEAK
            
            # Calculate price targets
            volatility = indicators.get('volatility', 0.2)
            price_target = None
            stop_loss = None
            
            if action == "buy":
                price_target = current_price * (1 + volatility * 0.5)
                stop_loss = current_price * (1 - volatility * 0.3)
            elif action == "sell":
                price_target = current_price * (1 - volatility * 0.5)
                stop_loss = current_price * (1 + volatility * 0.3)
            
            return AnalysisResult(
                symbol=symbol,
                signal_type="multi_source_technical",
                action=action,
                confidence=final_confidence,
                strength=strength,
                price_target=price_target,
                stop_loss=stop_loss,
                reasoning=f"{reasoning}. Data from {len(indicators_data.sources_used)} sources with {indicators_data.confidence:.1%} confidence.",
                indicators=indicators,
                data_sources=indicators_data.sources_used,
                data_confidence=indicators_data.confidence
            )
            
        except Exception as e:
            logger.error(f"Enhanced technical analysis error for {symbol}: {e}")
            return AnalysisResult(
                symbol=symbol,
                signal_type="error",
                action="hold",
                confidence=0.0,
                strength=SignalStrength.VERY_WEAK,
                reasoning=f"Analysis error: {e}",
                data_confidence=0.0
            )

class EnhancedFundamentalAnalysisAgent:
    """Enhanced fundamental analysis agent using multi-source data."""
    
    def __init__(self):
        self.name = "Enhanced Multi-Source Fundamental Analysis Agent"
    
    async def analyze_symbol(self, symbol: str) -> AnalysisResult:
        """Perform comprehensive fundamental analysis using multiple data sources."""
        try:
            logger.info(f"Multi-source fundamental analysis for {symbol}")
            
            # Get financial metrics from multiple sources
            metrics_data = await multi_source_data.get_financial_metrics_multi_source(symbol)
            
            if metrics_data.confidence < 0.3:
                return AnalysisResult(
                    symbol=symbol,
                    signal_type="insufficient_fundamental_data",
                    action="hold",
                    confidence=0.0,
                    strength=SignalStrength.VERY_WEAK,
                    reasoning="Insufficient fundamental data from multiple sources",
                    data_sources=metrics_data.sources_used,
                    data_confidence=metrics_data.confidence
                )
            
            metrics = metrics_data.value
            
            # Fundamental analysis signals
            signals = []
            
            # P/E Ratio Analysis
            pe_ratio = metrics.get('pe_ratio', 0)
            if 0 < pe_ratio < 15:
                signals.append(("pe_undervalued", "buy", 0.7))
            elif pe_ratio > 30:
                signals.append(("pe_overvalued", "sell", 0.6))
            
            # PEG Ratio Analysis
            peg_ratio = metrics.get('peg_ratio', 0)
            if 0 < peg_ratio < 1:
                signals.append(("peg_undervalued", "buy", 0.8))
            elif peg_ratio > 2:
                signals.append(("peg_overvalued", "sell", 0.7))
            
            # ROE Analysis
            roe = metrics.get('roe', 0)
            if roe > 0.15:  # 15%
                signals.append(("high_roe", "buy", 0.6))
            elif roe < 0.05:  # 5%
                signals.append(("low_roe", "sell", 0.5))
            
            # Debt-to-Equity Analysis
            debt_to_equity = metrics.get('debt_to_equity', 0)
            if debt_to_equity < 0.3:
                signals.append(("low_debt", "buy", 0.5))
            elif debt_to_equity > 1.0:
                signals.append(("high_debt", "sell", 0.6))
            
            # Growth Analysis
            revenue_growth = metrics.get('revenue_growth', 0)
            earnings_growth = metrics.get('earnings_growth', 0)
            
            if revenue_growth > 0.1 and earnings_growth > 0.1:  # 10% growth
                signals.append(("strong_growth", "buy", 0.8))
            elif revenue_growth < 0 or earnings_growth < -0.1:
                signals.append(("declining_growth", "sell", 0.7))
            
            # Aggregate signals
            buy_signals = [s for s in signals if s[1] == "buy"]
            sell_signals = [s for s in signals if s[1] == "sell"]
            
            if buy_signals:
                action = "buy"
                confidence = sum(s[2] for s in buy_signals) / len(buy_signals)
                reasoning = f"Fundamental buy signals: {', '.join(s[0] for s in buy_signals)}"
            elif sell_signals:
                action = "sell"
                confidence = sum(s[2] for s in sell_signals) / len(sell_signals)
                reasoning = f"Fundamental sell signals: {', '.join(s[0] for s in sell_signals)}"
            else:
                action = "hold"
                confidence = 0.5
                reasoning = "Neutral fundamental metrics"
            
            # Adjust confidence by data quality
            final_confidence = confidence * metrics_data.confidence
            
            # Determine signal strength
            if final_confidence >= 0.8:
                strength = SignalStrength.VERY_STRONG
            elif final_confidence >= 0.6:
                strength = SignalStrength.STRONG
            elif final_confidence >= 0.4:
                strength = SignalStrength.MODERATE
            elif final_confidence >= 0.2:
                strength = SignalStrength.WEAK
            else:
                strength = SignalStrength.VERY_WEAK
            
            return AnalysisResult(
                symbol=symbol,
                signal_type="multi_source_fundamental",
                action=action,
                confidence=final_confidence,
                strength=strength,
                reasoning=f"{reasoning}. Fundamental data from {len(metrics_data.sources_used)} sources with {metrics_data.confidence:.1%} confidence.",
                indicators=metrics,
                data_sources=metrics_data.sources_used,
                data_confidence=metrics_data.confidence
            )
            
        except Exception as e:
            logger.error(f"Enhanced fundamental analysis error for {symbol}: {e}")
            return AnalysisResult(
                symbol=symbol,
                signal_type="error",
                action="hold",
                confidence=0.0,
                strength=SignalStrength.VERY_WEAK,
                reasoning=f"Analysis error: {e}",
                data_confidence=0.0
            )

class EnhancedSentimentAnalysisAgent:
    """Enhanced sentiment analysis agent using multi-source news data."""
    
    def __init__(self):
        self.name = "Enhanced Multi-Source Sentiment Analysis Agent"
    
    async def analyze_symbol(self, symbol: str, company_name: str = None) -> AnalysisResult:
        """Perform comprehensive sentiment analysis using multiple news sources."""
        try:
            logger.info(f"Multi-source sentiment analysis for {symbol}")
            
            # Get news sentiment from multiple sources
            sentiment_data = await multi_source_data.get_news_sentiment_multi_source(symbol, company_name)
            
            if sentiment_data.confidence < 0.2:
                return AnalysisResult(
                    symbol=symbol,
                    signal_type="insufficient_sentiment_data",
                    action="hold",
                    confidence=0.0,
                    strength=SignalStrength.VERY_WEAK,
                    reasoning="Insufficient news data for sentiment analysis",
                    data_sources=sentiment_data.sources_used,
                    data_confidence=sentiment_data.confidence
                )
            
            sentiment_info = sentiment_data.value
            sentiment_score = sentiment_info['sentiment_score']
            
            # Sentiment-based signals
            if sentiment_score >= 0.65:
                action = "buy"
                confidence = (sentiment_score - 0.5) * 2  # Scale 0.5-1.0 to 0-1.0
                reasoning = f"Positive sentiment ({sentiment_score:.2f}) from {sentiment_info['article_count']} articles"
            elif sentiment_score <= 0.35:
                action = "sell"
                confidence = (0.5 - sentiment_score) * 2  # Scale 0.0-0.5 to 1.0-0
                reasoning = f"Negative sentiment ({sentiment_score:.2f}) from {sentiment_info['article_count']} articles"
            else:
                action = "hold"
                confidence = 0.5
                reasoning = f"Neutral sentiment ({sentiment_score:.2f}) from {sentiment_info['article_count']} articles"
            
            # Adjust confidence by data quality and article count
            article_confidence = min(sentiment_info['article_count'] / 20, 1.0)
            final_confidence = confidence * sentiment_data.confidence * article_confidence
            
            # Determine signal strength
            if final_confidence >= 0.8:
                strength = SignalStrength.VERY_STRONG
            elif final_confidence >= 0.6:
                strength = SignalStrength.STRONG
            elif final_confidence >= 0.4:
                strength = SignalStrength.MODERATE
            elif final_confidence >= 0.2:
                strength = SignalStrength.WEAK
            else:
                strength = SignalStrength.VERY_WEAK
            
            return AnalysisResult(
                symbol=symbol,
                signal_type="multi_source_sentiment",
                action=action,
                confidence=final_confidence,
                strength=strength,
                reasoning=f"{reasoning}. News from {len(sentiment_data.sources_used)} sources.",
                indicators={'sentiment_score': sentiment_score, 'article_count': sentiment_info['article_count']},
                data_sources=sentiment_data.sources_used,
                data_confidence=sentiment_data.confidence
            )
            
        except Exception as e:
            logger.error(f"Enhanced sentiment analysis error for {symbol}: {e}")
            return AnalysisResult(
                symbol=symbol,
                signal_type="error",
                action="hold",
                confidence=0.0,
                strength=SignalStrength.VERY_WEAK,
                reasoning=f"Analysis error: {e}",
                data_confidence=0.0
            )

# Factory function for creating enhanced analysis agents
def create_enhanced_market_analysis_factory():
    """Factory function to create enhanced market analysis agents."""
    return {
        'technical': EnhancedTechnicalAnalysisAgent(),
        'fundamental': EnhancedFundamentalAnalysisAgent(),
        'sentiment': EnhancedSentimentAnalysisAgent()
    }

class ComprehensiveMarketAnalysisAgent:
    """Master analysis agent that combines all data sources for final trading decisions."""
    
    def __init__(self):
        self.name = "Comprehensive Multi-Source Market Analysis Agent"
        self.technical_agent = EnhancedTechnicalAnalysisAgent()
        self.fundamental_agent = EnhancedFundamentalAnalysisAgent()
        self.sentiment_agent = EnhancedSentimentAnalysisAgent()
        
        # Import H2O agent if available
        try:
            # H2O prediction agent removed - using alternative ML analysis
            self.h2o_agent = None
            logger.info("✅ H2O.ai prediction agent integrated")
        except ImportError:
            self.h2o_agent = None
            logger.warning("⚠️ H2O.ai prediction agent not available")
    
    async def comprehensive_analysis(self, symbol: str, company_name: str = None) -> AnalysisResult:
        """Perform comprehensive analysis combining all data sources."""
        try:
            logger.info(f"Comprehensive multi-source analysis for {symbol}")
            
            # Run all analysis types concurrently
            tasks = [
                self.technical_agent.analyze_symbol(symbol),
                self.fundamental_agent.analyze_symbol(symbol),
                self.sentiment_agent.analyze_symbol(symbol, company_name)
            ]
            
            # Add H2O prediction if available
            if self.h2o_agent:
                tasks.append(self.h2o_agent.analyze_symbol(symbol))
            
            # Get additional context data
            context_tasks = [
                multi_source_data.get_economic_data_nasdaq(),
                multi_source_data.get_sec_filings_summary(symbol)
            ]
            
            # Execute all analyses
            analysis_results = await asyncio.gather(*tasks, return_exceptions=True)
            context_results = await asyncio.gather(*context_tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_analyses = [r for r in analysis_results if isinstance(r, AnalysisResult)]
            
            if not valid_analyses:
                return AnalysisResult(
                    symbol=symbol,
                    signal_type="comprehensive_analysis_failed",
                    action="hold",
                    confidence=0.0,
                    strength=SignalStrength.VERY_WEAK,
                    reasoning="All analysis methods failed",
                    data_confidence=0.0
                )
            
            # Aggregate analyses with different weights
            weights = {
                'multi_source_technical': 0.3,
                'multi_source_fundamental': 0.3,
                'multi_source_sentiment': 0.2,
                'h2o_ml_prediction': 0.2  # H2O ML prediction weight
            }
            
            # Collect signals by action
            buy_signals = []
            sell_signals = []
            hold_signals = []
            
            all_sources = set()
            total_data_confidence = 0
            
            for analysis in valid_analyses:
                weight = weights.get(analysis.signal_type, 0.2)
                weighted_confidence = analysis.confidence * weight
                
                if analysis.action == "buy":
                    buy_signals.append((analysis, weighted_confidence))
                elif analysis.action == "sell":
                    sell_signals.append((analysis, weighted_confidence))
                else:
                    hold_signals.append((analysis, weighted_confidence))
                
                all_sources.update(analysis.data_sources or [])
                total_data_confidence += analysis.data_confidence
            
            # Calculate aggregate confidence for each action
            buy_confidence = sum(conf for _, conf in buy_signals)
            sell_confidence = sum(conf for _, conf in sell_signals)
            hold_confidence = sum(conf for _, conf in hold_signals)
            
            # Determine final action
            if buy_confidence > sell_confidence and buy_confidence > hold_confidence:
                final_action = "buy"
                final_confidence = buy_confidence
                signal_details = [f"{a.signal_type}({a.confidence:.2f})" for a, _ in buy_signals]
            elif sell_confidence > buy_confidence and sell_confidence > hold_confidence:
                final_action = "sell"
                final_confidence = sell_confidence
                signal_details = [f"{a.signal_type}({a.confidence:.2f})" for a, _ in sell_signals]
            else:
                final_action = "hold"
                final_confidence = max(hold_confidence, 0.3)
                signal_details = ["mixed_signals"]
            
            # Add context from economic and filing data
            context_info = []
            if context_results:
                for context in context_results:
                    if isinstance(context, AggregatedData) and context.confidence > 0:
                        if context.sources_used and DataSource.NASDAQ in context.sources_used:
                            context_info.append("economic_data")
                        if context.sources_used and DataSource.SEC_EDGAR in context.sources_used:
                            context_info.append("sec_filings")
            
            # Calculate overall data confidence
            avg_data_confidence = total_data_confidence / len(valid_analyses) if valid_analyses else 0
            
            # Determine signal strength based on confidence and agreement
            agreement_bonus = 0
            if len(buy_signals) > 1 or len(sell_signals) > 1:
                agreement_bonus = 0.1  # Bonus for multiple methods agreeing
            
            adjusted_confidence = min(final_confidence + agreement_bonus, 1.0)
            
            if adjusted_confidence >= 0.8:
                strength = SignalStrength.VERY_STRONG
            elif adjusted_confidence >= 0.6:
                strength = SignalStrength.STRONG
            elif adjusted_confidence >= 0.4:
                strength = SignalStrength.MODERATE
            elif adjusted_confidence >= 0.2:
                strength = SignalStrength.WEAK
            else:
                strength = SignalStrength.VERY_WEAK
            
            # Aggregate all indicators
            all_indicators = {}
            for analysis in valid_analyses:
                if analysis.indicators:
                    all_indicators.update({
                        f"{analysis.signal_type}_{k}": v 
                        for k, v in analysis.indicators.items()
                    })
            
            # Build comprehensive reasoning
            reasoning_parts = [
                f"Multi-source {final_action} signal",
                f"Sources: {', '.join(signal_details)}",
                f"Data from {len(all_sources)} APIs"
            ]
            
            if context_info:
                reasoning_parts.append(f"Context: {', '.join(context_info)}")
            
            reasoning = ". ".join(reasoning_parts)
            
            return AnalysisResult(
                symbol=symbol,
                signal_type="comprehensive_multi_source",
                action=final_action,
                confidence=adjusted_confidence,
                strength=strength,
                reasoning=reasoning,
                indicators=all_indicators,
                data_sources=list(all_sources),
                data_confidence=avg_data_confidence
            )
            
        except Exception as e:
            logger.error(f"Comprehensive analysis error for {symbol}: {e}")
            return AnalysisResult(
                symbol=symbol,
                signal_type="error",
                action="hold",
                confidence=0.0,
                strength=SignalStrength.VERY_WEAK,
                reasoning=f"Comprehensive analysis error: {e}",
                data_confidence=0.0
            )

# Global enhanced agents
enhanced_market_analysis_factory = create_enhanced_market_analysis_factory()
comprehensive_analyst = ComprehensiveMarketAnalysisAgent()