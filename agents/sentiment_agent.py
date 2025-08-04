"""
Comprehensive Sentiment Agent

Orchestrates sentiment analysis from multiple data sources:
- Financial news APIs (NewsAPI, Alpha Vantage, FMP)
- Social media platforms (Reddit, Twitter, TikTok)
- Earnings call transcripts (Motley Fool)
- Market data sentiment (price momentum, volume)

Uses LLM-based analysis for sophisticated sentiment understanding.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from core.llm_sentiment_analyzer import LLMSentimentAnalyzer, SentimentAnalysis
from core.social_media_collector import SocialMediaCollector, SocialMediaPost
from core.earnings_scraper import EarningsCallScraper, EarningsTranscript
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ComprehensiveSentiment:
    """Comprehensive sentiment analysis result for a stock."""
    symbol: str
    timestamp: datetime
    
    # Overall metrics
    overall_score: float  # -1 to 1
    overall_sentiment: str  # very_negative, negative, neutral, positive, very_positive
    confidence: float  # 0 to 1
    
    # Source breakdown
    news_sentiment: Optional[SentimentAnalysis] = None
    social_sentiment: Dict[str, SentimentAnalysis] = None
    earnings_sentiment: Dict[str, SentimentAnalysis] = None
    market_sentiment: Optional[SentimentAnalysis] = None
    
    # Metadata
    data_sources_count: int = 0
    news_articles_count: int = 0
    social_posts_count: int = 0
    has_recent_earnings: bool = False
    
    # Key insights
    key_themes: List[str] = None
    risk_factors: List[str] = None
    opportunities: List[str] = None
    
    def __post_init__(self):
        if self.social_sentiment is None:
            self.social_sentiment = {}
        if self.earnings_sentiment is None:
            self.earnings_sentiment = {}
        if self.key_themes is None:
            self.key_themes = []
        if self.risk_factors is None:
            self.risk_factors = []
        if self.opportunities is None:
            self.opportunities = []


class SentimentAgent:
    """Main sentiment analysis agent."""
    
    def __init__(self):
        # Initialize components
        self.llm_analyzer = LLMSentimentAnalyzer(
            anthropic_api_key=settings.anthropic_api_key,
            openai_api_key=settings.openai_api_key,
            ollama_base_url=settings.ollama_base_url
        )
        self.social_collector = SocialMediaCollector()
        self.earnings_scraper = EarningsCallScraper()
        
        # Caching
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # Source weights for aggregation
        self.source_weights = {
            'news': 0.35,      # Financial news most important
            'earnings': 0.30,   # Earnings calls very important
            'social': 0.25,     # Social media significant
            'market': 0.10      # Market sentiment as background
        }
    
    async def analyze_comprehensive_sentiment(self, symbol: str) -> ComprehensiveSentiment:
        """Perform comprehensive sentiment analysis for a stock."""
        cache_key = f"comprehensive_sentiment_{symbol}"
        
        # Check cache
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_duration:
                logger.info(f"Using cached sentiment for {symbol}")
                return data
        
        logger.info(f"Starting comprehensive sentiment analysis for {symbol}")
        
        # Collect data from all sources in parallel with timeouts
        news_task = asyncio.wait_for(self._analyze_news_sentiment(symbol), timeout=30.0)
        social_task = asyncio.wait_for(self._analyze_social_sentiment(symbol), timeout=45.0)
        earnings_task = asyncio.wait_for(self._analyze_earnings_sentiment(symbol), timeout=60.0)
        market_task = asyncio.wait_for(self._analyze_market_sentiment(symbol), timeout=15.0)
        
        # Execute all tasks with graceful failure handling
        results = await asyncio.gather(
            news_task, social_task, earnings_task, market_task,
            return_exceptions=True
        )
        
        news_sentiment, social_sentiment, earnings_sentiment, market_sentiment = results
        
        # Handle exceptions with detailed logging
        data_sources_available = []
        
        if isinstance(news_sentiment, Exception):
            if isinstance(news_sentiment, asyncio.TimeoutError):
                logger.warning(f"News sentiment analysis timed out for {symbol} - continuing without news data")
            else:
                logger.warning(f"News sentiment analysis failed for {symbol}: {type(news_sentiment).__name__} - continuing without news data")
            news_sentiment = None
        else:
            data_sources_available.append("news")
            
        if isinstance(social_sentiment, Exception):
            if isinstance(social_sentiment, asyncio.TimeoutError):
                logger.warning(f"Social sentiment analysis timed out for {symbol} - continuing without social data")
            else:
                logger.warning(f"Social sentiment analysis failed for {symbol}: {type(social_sentiment).__name__} - continuing without social data")
            social_sentiment = {}
        else:
            if social_sentiment:
                data_sources_available.append("social")
            
        if isinstance(earnings_sentiment, Exception):
            if isinstance(earnings_sentiment, asyncio.TimeoutError):
                logger.warning(f"Earnings sentiment analysis timed out for {symbol} - continuing without earnings data")
            else:
                logger.warning(f"Earnings sentiment analysis failed for {symbol}: {type(earnings_sentiment).__name__} - continuing without earnings data")
            earnings_sentiment = {}
        else:
            if earnings_sentiment:
                data_sources_available.append("earnings")
                logger.info(f"✅ Successfully integrated Motley Fool earnings data for {symbol}")
            
        if isinstance(market_sentiment, Exception):
            if isinstance(market_sentiment, asyncio.TimeoutError):
                logger.warning(f"Market sentiment analysis timed out for {symbol} - continuing without market data")
            else:
                logger.warning(f"Market sentiment analysis failed for {symbol}: {type(market_sentiment).__name__} - continuing without market data")
            market_sentiment = None
        else:
            data_sources_available.append("market")
        
        logger.info(f"Data sources available for {symbol}: {', '.join(data_sources_available)} ({len(data_sources_available)}/4 sources)")
        
        # Aggregate sentiments
        overall_score, overall_sentiment, confidence = self._aggregate_sentiments(
            news_sentiment, social_sentiment, earnings_sentiment, market_sentiment
        )
        
        # Extract insights
        key_themes, risk_factors, opportunities = self._extract_insights(
            news_sentiment, social_sentiment, earnings_sentiment, market_sentiment
        )
        
        # Count data sources
        data_sources_count = sum([
            1 if news_sentiment else 0,
            len(social_sentiment),
            len(earnings_sentiment),
            1 if market_sentiment else 0
        ])
        
        # Create comprehensive result
        result = ComprehensiveSentiment(
            symbol=symbol,
            timestamp=datetime.now(),
            overall_score=overall_score,
            overall_sentiment=overall_sentiment,
            confidence=confidence,
            news_sentiment=news_sentiment,
            social_sentiment=social_sentiment or {},
            earnings_sentiment=earnings_sentiment or {},
            market_sentiment=market_sentiment,
            data_sources_count=data_sources_count,
            news_articles_count=self._count_news_articles(news_sentiment),
            social_posts_count=self._count_social_posts(social_sentiment),
            has_recent_earnings=bool(earnings_sentiment),
            key_themes=key_themes,
            risk_factors=risk_factors,
            opportunities=opportunities
        )
        
        # Cache result
        self.cache[cache_key] = (result, datetime.now())
        
        logger.info(f"Completed sentiment analysis for {symbol}: {overall_sentiment} ({overall_score:.3f})")
        return result
    
    async def _analyze_news_sentiment(self, symbol: str) -> Optional[SentimentAnalysis]:
        """Analyze sentiment from financial news."""
        try:
            # This would integrate with existing news collection in market_intelligence.py
            # For now, we'll create a placeholder that interfaces with the existing system
            from core.market_intelligence import UnifiedMarketIntelligence
            
            market_intel = UnifiedMarketIntelligence()
            
            # Get news articles (this would be extracted from market_intelligence.py)
            articles = await self._get_news_articles(symbol, market_intel)
            
            if not articles:
                return None
            
            # Combine all articles into one text for analysis
            combined_text = ""
            for article in articles[:10]:  # Limit to 10 articles
                title = article.get('title', '')
                description = article.get('description', '')
                combined_text += f"{title}. {description}\n"
            
            if not combined_text.strip():
                return None
            
            # Analyze with LLM
            return await self.llm_analyzer.analyze_text(combined_text, "financial_news")
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment for {symbol}: {e}")
            return None
    
    async def _analyze_social_sentiment(self, symbol: str) -> Dict[str, SentimentAnalysis]:
        """Analyze sentiment from social media platforms."""
        try:
            # Collect posts from all platforms
            platform_results = await self.social_collector.collect_all_platforms(symbol)
            
            sentiment_results = {}
            
            for platform, posts in platform_results.items():
                if not posts:
                    continue
                
                # Combine posts into text for analysis
                combined_text = ""
                for post in posts[:20]:  # Limit to 20 posts per platform
                    combined_text += f"{post.content}\n"
                
                if combined_text.strip():
                    sentiment = await self.llm_analyzer.analyze_text(combined_text, "social_media")
                    sentiment_results[platform] = sentiment
            
            return sentiment_results
            
        except Exception as e:
            logger.error(f"Error analyzing social sentiment for {symbol}: {e}")
            return {}
    
    async def _analyze_earnings_sentiment(self, symbol: str) -> Dict[str, SentimentAnalysis]:
        """Analyze sentiment from earnings call transcripts."""
        try:
            # Use context manager for proper session cleanup
            async with self.earnings_scraper as scraper:
                # Get latest earnings transcript from Motley Fool
                transcript = await scraper.get_latest_transcript(symbol)
                
                if not transcript:
                    logger.info(f"No recent earnings transcript found for {symbol}")
                    return {}
                
                logger.info(f"Analyzing earnings transcript for {symbol}: {transcript.company_name} {transcript.quarter} {transcript.year}")
                logger.info(f"Transcript content: {len(transcript.full_text):,} characters, Management: {len(transcript.management_section):,}, Q&A: {len(transcript.qa_section):,}")
                
                # Analyze different sections of the transcript using our enhanced LLM
                earnings_sentiment = await self.llm_analyzer.analyze_earnings_transcript(transcript)
                
                # Log results
                for section, sentiment in earnings_sentiment.items():
                    if isinstance(sentiment, SentimentAnalysis):
                        logger.info(f"Earnings {section} sentiment for {symbol}: {sentiment.sentiment} (score: {sentiment.score:.3f}, confidence: {sentiment.confidence:.3f})")
                
                return earnings_sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing earnings sentiment for {symbol}: {e}")
            return {}
    
    async def _analyze_market_sentiment(self, symbol: str) -> Optional[SentimentAnalysis]:
        """Analyze market-based sentiment from price action and volume."""
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo")
            
            if hist.empty or len(hist) < 5:
                return None
            
            # Calculate momentum indicators
            recent_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[-5]) - 1
            volume_trend = hist['Volume'].iloc[-5:].mean() / hist['Volume'].iloc[-20:-5].mean()
            
            # Create sentiment text based on market indicators
            price_direction = "increasing" if recent_return > 0 else "decreasing"
            volume_direction = "higher" if volume_trend > 1.1 else "lower" if volume_trend < 0.9 else "stable"
            
            market_text = f"""
            Stock price has been {price_direction} by {abs(recent_return)*100:.1f}% over the past week.
            Trading volume is {volume_direction} than the recent average.
            Market momentum appears {'positive' if recent_return > 0 else 'negative'}.
            """
            
            return await self.llm_analyzer.analyze_text(market_text, "financial_news")
            
        except Exception as e:
            logger.error(f"Error analyzing market sentiment for {symbol}: {e}")
            return None
    
    async def _get_news_articles(self, symbol: str, market_intel) -> List[Dict]:
        """Get news articles from market intelligence system."""
        try:
            # This integrates with the existing news collection
            if hasattr(market_intel, '_get_news_articles'):
                return await market_intel._get_news_articles(symbol)
            else:
                # Fallback: direct NewsAPI call
                import aiohttp
                
                if not settings.news_api_key:
                    return []
                
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': f'"{symbol}"',
                    'sortBy': 'publishedAt',
                    'language': 'en',
                    'pageSize': 10,
                    'from': (datetime.now() - timedelta(days=1)).isoformat(),
                    'apiKey': settings.news_api_key
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data.get('articles', [])
                        return []
        except Exception as e:
            logger.error(f"Error getting news articles for {symbol}: {e}")
            return []
    
    def _aggregate_sentiments(self, news_sentiment, social_sentiment, earnings_sentiment, market_sentiment) -> Tuple[float, str, float]:
        """Aggregate all sentiment sources into overall sentiment."""
        scores = []
        confidences = []
        weights = []
        
        # Add news sentiment
        if news_sentiment:
            scores.append(news_sentiment.score)
            confidences.append(news_sentiment.confidence)
            weights.append(self.source_weights['news'])
        
        # Add social sentiment (average across platforms)
        if social_sentiment:
            social_scores = [s.score for s in social_sentiment.values()]
            social_confidences = [s.confidence for s in social_sentiment.values()]
            if social_scores:
                avg_social_score = sum(social_scores) / len(social_scores)
                avg_social_confidence = sum(social_confidences) / len(social_confidences)
                scores.append(avg_social_score)
                confidences.append(avg_social_confidence)
                weights.append(self.source_weights['social'])
        
        # Add earnings sentiment (average across sections)
        if earnings_sentiment:
            earnings_scores = [s.score for s in earnings_sentiment.values()]
            earnings_confidences = [s.confidence for s in earnings_sentiment.values()]
            if earnings_scores:
                avg_earnings_score = sum(earnings_scores) / len(earnings_scores)
                avg_earnings_confidence = sum(earnings_confidences) / len(earnings_confidences)
                scores.append(avg_earnings_score)
                confidences.append(avg_earnings_confidence)
                weights.append(self.source_weights['earnings'])
        
        # Add market sentiment
        if market_sentiment:
            scores.append(market_sentiment.score)
            confidences.append(market_sentiment.confidence)
            weights.append(self.source_weights['market'])
        
        if not scores:
            return 0.0, "neutral", 0.0
        
        # Calculate weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            overall_score = sum(scores) / len(scores)
            overall_confidence = sum(confidences) / len(confidences)
        else:
            overall_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
            overall_confidence = sum(c * w for c, w in zip(confidences, weights)) / total_weight
        
        # Adjust confidence based on number of available data sources
        data_source_count = len(scores)
        if data_source_count < 4:  # We have 4 possible sources
            # Reduce confidence for missing data sources
            source_penalty = (4 - data_source_count) * 0.1  # 10% penalty per missing source
            overall_confidence = max(0.1, overall_confidence - source_penalty)
            logger.info(f"Adjusted confidence from missing data sources: -{source_penalty:.1%} penalty for {4-data_source_count} missing sources")
        
        # Convert score to sentiment label
        if overall_score > 0.5:
            overall_sentiment = "very_positive"
        elif overall_score > 0.2:
            overall_sentiment = "positive"
        elif overall_score > -0.2:
            overall_sentiment = "neutral"
        elif overall_score > -0.5:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "very_negative"
        
        return overall_score, overall_sentiment, overall_confidence
    
    def _extract_insights(self, news_sentiment, social_sentiment, earnings_sentiment, market_sentiment) -> Tuple[List[str], List[str], List[str]]:
        """Extract key themes, risks, and opportunities from all sentiment sources."""
        all_themes = set()
        all_risks = set()
        all_opportunities = set()
        
        # Extract from news sentiment
        if news_sentiment:
            all_themes.update(news_sentiment.key_phrases)
            all_risks.update(news_sentiment.risk_factors)
            all_opportunities.update(news_sentiment.opportunities)
        
        # Extract from social sentiment
        for sentiment in social_sentiment.values():
            all_themes.update(sentiment.key_phrases)
            all_risks.update(sentiment.risk_factors)
            all_opportunities.update(sentiment.opportunities)
        
        # Extract from earnings sentiment
        for sentiment in earnings_sentiment.values():
            all_themes.update(sentiment.key_phrases)
            all_risks.update(sentiment.risk_factors)
            all_opportunities.update(sentiment.opportunities)
        
        # Extract from market sentiment
        if market_sentiment:
            all_themes.update(market_sentiment.key_phrases)
            all_risks.update(market_sentiment.risk_factors)
            all_opportunities.update(market_sentiment.opportunities)
        
        return list(all_themes)[:10], list(all_risks)[:5], list(all_opportunities)[:5]
    
    def _count_news_articles(self, news_sentiment) -> int:
        """Count number of news articles analyzed."""
        # This would be enhanced with actual article tracking
        return 10 if news_sentiment else 0
    
    def _count_social_posts(self, social_sentiment) -> int:
        """Count number of social media posts analyzed."""
        # This would be enhanced with actual post tracking
        return sum(20 for platform in social_sentiment.keys())
    
    async def run_sentiment_cycle(self, tickers: List[str], platforms: List[str] = None) -> Dict[str, ComprehensiveSentiment]:
        """Run sentiment analysis cycle for multiple tickers."""
        if platforms is None:
            platforms = ['reddit', 'twitter', 'tiktok']
        
        results = {}
        
        # Analyze each ticker
        for ticker in tickers:
            try:
                sentiment = await self.analyze_comprehensive_sentiment(ticker)
                results[ticker] = sentiment
                
                # Brief pause to avoid overwhelming APIs
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error analyzing sentiment for {ticker}: {e}")
                continue
        
        return results
    
    async def close(self):
        """Close all connections and clean up resources."""
        await self.llm_analyzer.close()
        await self.earnings_scraper.close()


# Global sentiment agent instance
sentiment_agent = SentimentAgent()