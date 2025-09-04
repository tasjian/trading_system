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
from core.social_media_collector_optimized import SocialMediaCollector, SocialMediaPost
from core.earnings_scraper import EarningsCallScraper, EarningsTranscript
from core.sec_edgar_client import SECEdgarClient, SECFiling
# CRYPTO TRADING DISABLED - Comment out crypto sentiment analyzer
# from core.crypto_sentiment_analyzer import crypto_sentiment_analyzer, CryptoSentimentResult
from config.settings import settings, is_crypto_symbol
from tools.alpaca_market_data import fetch_stock_history
from utils.enhanced_api_cache import get_enhanced_cache, CacheType, cached_api

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
    sec_filings_sentiment: Dict[str, SentimentAnalysis] = None
    market_sentiment: Optional[SentimentAnalysis] = None
    
    # Metadata
    data_sources_count: int = 0
    news_articles_count: int = 0
    social_posts_count: int = 0
    sec_filings_count: int = 0
    has_recent_earnings: bool = False
    has_recent_sec_filings: bool = False
    
    # Key insights
    key_themes: List[str] = None
    risk_factors: List[str] = None
    opportunities: List[str] = None
    
    def __post_init__(self):
        if self.social_sentiment is None:
            self.social_sentiment = {}
        if self.earnings_sentiment is None:
            self.earnings_sentiment = {}
        if self.sec_filings_sentiment is None:
            self.sec_filings_sentiment = {}
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
            ollama_base_url=settings.ollama_base_url
        )
        self.social_collector = SocialMediaCollector()
        self.earnings_scraper = EarningsCallScraper()
        self.sec_edgar_client = SECEdgarClient()
        
        # Caching
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # Source weights for aggregation
        self.source_weights = {
            'news': 0.30,      # Financial news important
            'earnings': 0.25,   # Earnings calls important
            'sec_filings': 0.25, # SEC filings very important for fundamentals
            'social': 0.15,     # Social media less weight
            'market': 0.05      # Market sentiment as background
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
        
        # Check if this is a crypto symbol and use specialized analysis
        if is_crypto_symbol(symbol):
            logger.info(f"🚀 Using crypto-specialized sentiment analysis for {symbol}")
            return await self._analyze_crypto_sentiment(symbol)
        
        # Collect data from all sources with staggered execution to prevent API overload
        # Add small delays between task creation to stagger API calls
        news_task = asyncio.wait_for(self._analyze_news_sentiment(symbol), timeout=180.0)
        
        await asyncio.sleep(0.5)  # Small delay to stagger API calls
        social_task = asyncio.wait_for(self._analyze_social_sentiment(symbol), timeout=120.0)  # Reduced from 240s to prevent churning delays
        
        await asyncio.sleep(0.5)  # Small delay to stagger API calls  
        earnings_task = asyncio.wait_for(self._analyze_earnings_sentiment(symbol), timeout=300.0)
        
        await asyncio.sleep(0.5)  # Small delay to stagger API calls
        sec_task = asyncio.wait_for(self._analyze_sec_filings_sentiment(symbol), timeout=120.0)
        
        await asyncio.sleep(0.5)  # Small delay to stagger API calls
        market_task = asyncio.wait_for(self._analyze_market_sentiment(symbol), timeout=150.0)
        
        # Execute all tasks with proper exception handling
        # Use asyncio.gather with return_exceptions=True to prevent one failure from cancelling others
        try:
            results = await asyncio.gather(
                news_task, social_task, earnings_task, sec_task, market_task,
                return_exceptions=True
            )
        except asyncio.CancelledError:
            logger.warning(f"Comprehensive sentiment analysis was cancelled for {symbol}")
            # Clean cancellation - re-raise to properly propagate
            raise
        
        news_sentiment, social_sentiment, earnings_sentiment, sec_sentiment, market_sentiment = results
        
        # Handle exceptions with detailed logging
        data_sources_available = []
        
        if isinstance(news_sentiment, Exception):
            if isinstance(news_sentiment, asyncio.TimeoutError):
                logger.warning(f"News sentiment analysis timed out for {symbol} - continuing without news data")
            elif isinstance(news_sentiment, asyncio.CancelledError):
                logger.warning(f"News sentiment analysis was cancelled for {symbol} - continuing without news data")
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
        
        if isinstance(sec_sentiment, Exception):
            if isinstance(sec_sentiment, asyncio.TimeoutError):
                logger.warning(f"SEC filings sentiment analysis timed out for {symbol} - continuing without SEC data")
            else:
                logger.warning(f"SEC filings sentiment analysis failed for {symbol}: {type(sec_sentiment).__name__} - continuing without SEC data")
            sec_sentiment = {}
        else:
            if sec_sentiment:
                data_sources_available.append("sec_filings")
                logger.info(f"✅ Successfully integrated SEC EDGAR filings data for {symbol}")
            
        if isinstance(market_sentiment, Exception):
            if isinstance(market_sentiment, asyncio.TimeoutError):
                logger.warning(f"Market sentiment analysis timed out for {symbol} - continuing without market data")
            else:
                logger.warning(f"Market sentiment analysis failed for {symbol}: {type(market_sentiment).__name__} - continuing without market data")
            market_sentiment = None
        else:
            data_sources_available.append("market")
        
        logger.info(f"Data sources available for {symbol}: {', '.join(data_sources_available)} ({len(data_sources_available)}/5 sources)")
        
        # Aggregate sentiments
        overall_score, overall_sentiment, confidence = self._aggregate_sentiments(
            news_sentiment, social_sentiment, earnings_sentiment, sec_sentiment, market_sentiment
        )
        
        # Extract insights
        key_themes, risk_factors, opportunities = self._extract_insights(
            news_sentiment, social_sentiment, earnings_sentiment, sec_sentiment, market_sentiment
        )
        
        # Count data sources
        data_sources_count = sum([
            1 if news_sentiment else 0,
            len(social_sentiment),
            len(earnings_sentiment),
            len(sec_sentiment),
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
            sec_filings_sentiment=sec_sentiment or {},
            market_sentiment=market_sentiment,
            data_sources_count=data_sources_count,
            news_articles_count=self._count_news_articles(news_sentiment),
            social_posts_count=self._count_social_posts(social_sentiment),
            sec_filings_count=self._count_sec_filings(sec_sentiment),
            has_recent_earnings=bool(earnings_sentiment),
            has_recent_sec_filings=bool(sec_sentiment),
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
            # Collect posts from all platforms with generous timeout
            platform_results = await asyncio.wait_for(
                self.social_collector.collect_all_platforms(symbol), 
                timeout=45.0  # Increased timeout for social media collection
            )
            
            sentiment_results = {}
            
            for platform, posts in platform_results.items():
                if not posts:
                    continue
                
                # Combine posts into text for analysis (limit text length)
                combined_text = ""
                for post in posts[:10]:  # Reduced from 20 to 10 for faster processing
                    combined_text += f"{post.content}\n"
                    if len(combined_text) > 2000:  # Limit text length
                        break
                
                if combined_text.strip():
                    # Add timeout for LLM analysis (increased for Ollama)
                    try:
                        sentiment = await asyncio.wait_for(
                            self.llm_analyzer.analyze_text(combined_text, "social_media"),
                            timeout=180.0  # Further increased timeout for CPU Ollama processing
                        )
                        if sentiment:
                            sentiment_results[platform] = sentiment
                    except asyncio.TimeoutError:
                        logger.warning(f"LLM sentiment analysis timed out for {platform}")
                        continue
            
            return sentiment_results
            
        except asyncio.TimeoutError:
            logger.warning(f"Social media collection timed out for {symbol}")
            return {}
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
            # Use utility function for market data fetch with timeout
            hist = await asyncio.wait_for(
                asyncio.to_thread(fetch_stock_history, symbol, "1mo"),
                timeout=5.0
            )
            
            if hist is None or hist.empty or len(hist) < 5:
                logger.debug(f"Insufficient market data for {symbol}")
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
            
            # Add timeout for LLM analysis (increased for Ollama)
            return await asyncio.wait_for(
                self.llm_analyzer.analyze_text(market_text, "financial_news"),
                timeout=150.0  # Further increased timeout for CPU Ollama to fully process
            )
            
        except asyncio.TimeoutError:
            logger.warning(f"Market sentiment analysis timed out for {symbol}")
            return None
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
    
    def _aggregate_sentiments(self, news_sentiment, social_sentiment, earnings_sentiment, sec_sentiment, market_sentiment) -> Tuple[float, str, float]:
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
        if social_sentiment and isinstance(social_sentiment, dict) and social_sentiment:
            social_scores = [s.score for s in social_sentiment.values() if isinstance(s, SentimentAnalysis)]
            social_confidences = [s.confidence for s in social_sentiment.values() if isinstance(s, SentimentAnalysis)]
            if social_scores:
                avg_social_score = sum(social_scores) / len(social_scores)
                avg_social_confidence = sum(social_confidences) / len(social_confidences)
                scores.append(avg_social_score)
                confidences.append(avg_social_confidence)
                weights.append(self.source_weights['social'])
        
        # Add earnings sentiment (average across sections)
        if earnings_sentiment and isinstance(earnings_sentiment, dict) and earnings_sentiment:
            earnings_scores = [s.score for s in earnings_sentiment.values() if isinstance(s, SentimentAnalysis)]
            earnings_confidences = [s.confidence for s in earnings_sentiment.values() if isinstance(s, SentimentAnalysis)]
            if earnings_scores:
                avg_earnings_score = sum(earnings_scores) / len(earnings_scores)
                avg_earnings_confidence = sum(earnings_confidences) / len(earnings_confidences)
                scores.append(avg_earnings_score)
                confidences.append(avg_earnings_confidence)
                weights.append(self.source_weights['earnings'])
        
        # Add SEC filings sentiment (average across filings)
        if sec_sentiment and isinstance(sec_sentiment, dict) and sec_sentiment:
            sec_scores = [s.score for s in sec_sentiment.values() if isinstance(s, SentimentAnalysis)]
            sec_confidences = [s.confidence for s in sec_sentiment.values() if isinstance(s, SentimentAnalysis)]
            if sec_scores:
                avg_sec_score = sum(sec_scores) / len(sec_scores)
                avg_sec_confidence = sum(sec_confidences) / len(sec_confidences)
                scores.append(avg_sec_score)
                confidences.append(avg_sec_confidence)
                weights.append(self.source_weights['sec_filings'])
        
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
    
    def _extract_insights(self, news_sentiment, social_sentiment, earnings_sentiment, sec_sentiment, market_sentiment) -> Tuple[List[str], List[str], List[str]]:
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
        if isinstance(social_sentiment, dict):
            for sentiment in social_sentiment.values():
                if isinstance(sentiment, SentimentAnalysis):
                    all_themes.update(sentiment.key_phrases)
                    all_risks.update(sentiment.risk_factors)
                    all_opportunities.update(sentiment.opportunities)
        
        # Extract from earnings sentiment
        if isinstance(earnings_sentiment, dict):
            for sentiment in earnings_sentiment.values():
                if isinstance(sentiment, SentimentAnalysis):
                    all_themes.update(sentiment.key_phrases)
                    all_risks.update(sentiment.risk_factors)
                    all_opportunities.update(sentiment.opportunities)
        
        # Extract from SEC filings sentiment
        if isinstance(sec_sentiment, dict):
            for sentiment in sec_sentiment.values():
                if isinstance(sentiment, SentimentAnalysis):
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
    
    def _count_sec_filings(self, sec_sentiment) -> int:
        """Count number of SEC filings analyzed."""
        return len(sec_sentiment) if sec_sentiment else 0
    
    async def _analyze_sec_filings_sentiment(self, symbol: str) -> Dict[str, SentimentAnalysis]:
        """Analyze sentiment from recent SEC filings."""
        try:
            logger.info(f"Analyzing SEC filings sentiment for {symbol}")
            
            # Initialize SEC EDGAR client if needed
            if not hasattr(self.sec_edgar_client, '_company_tickers') or not self.sec_edgar_client._company_tickers:
                await self.sec_edgar_client.load_company_tickers()
            
            # Get recent filings (10-K, 10-Q, 8-K) - reduced lookback for speed
            recent_filings = await self.sec_edgar_client.search_filings(
                ticker=symbol,
                form_types=['10-K', '10-Q', '8-K'],
                days_back=30  # Reduced from 90 days for faster performance
            )
            
            if not recent_filings:
                logger.info(f"No recent SEC filings found for {symbol}")
                return {}
            
            logger.info(f"Found {len(recent_filings)} recent filings for {symbol}")
            
            sentiment_results = {}
            
            # Analyze up to 2 most recent filings for speed (reduced from 5)
            for filing in recent_filings[:2]:
                try:
                    # Download filing content
                    filing_with_content = await self.sec_edgar_client.get_filing_content(filing)
                    
                    if not filing_with_content or not filing_with_content.content:
                        logger.debug(f"No content available for filing {filing.accession_number}")
                        continue
                    
                    # Prepare content for LLM analysis
                    filing_summary = self._prepare_filing_summary(filing_with_content)
                    
                    if not filing_summary:
                        continue
                    
                    # Analyze sentiment with LLM
                    prompt = self._create_sec_filing_prompt(symbol, filing_with_content, filing_summary)
                    
                    sentiment = await self.llm_analyzer.analyze_text(
                        prompt, 
                        context="analyst_report"  # Using analyst_report context for SEC filings
                    )
                    
                    if sentiment:
                        # Enhance with SEC-specific signals
                        sentiment = self._enhance_sec_sentiment(sentiment, filing_with_content)
                        
                        filing_key = f"{filing.filing_type}_{filing.filing_date}"
                        sentiment_results[filing_key] = sentiment
                        logger.info(f"Analyzed sentiment for {filing.filing_type} filing: {sentiment.sentiment} ({sentiment.score:.3f})")
                    
                except Exception as e:
                    logger.error(f"Error analyzing filing {filing.accession_number}: {e}")
                    continue
            
            logger.info(f"Successfully analyzed {len(sentiment_results)} SEC filings for {symbol}")
            return sentiment_results
            
        except Exception as e:
            logger.error(f"Error in SEC filings sentiment analysis for {symbol}: {e}")
            return {}
    
    def _prepare_filing_summary(self, filing: SECFiling) -> str:
        """Prepare a summary of filing content for LLM analysis."""
        try:
            if not filing.content:
                return ""
            
            # Start with key sections if available
            summary_parts = []
            
            if filing.key_sections:
                for section_name, section_content in filing.key_sections.items():
                    if section_content:
                        summary_parts.append(f"{section_name.replace('_', ' ').title()}: {section_content[:1000]}")
            
            # If no key sections, use first part of content
            if not summary_parts:
                # Clean up HTML and get meaningful text
                clean_content = self._clean_filing_content(filing.content)[:3000]
                summary_parts.append(clean_content)
            
            # Add sentiment signals
            if filing.sentiment_signals:
                signals_text = ", ".join(filing.sentiment_signals)
                summary_parts.append(f"Key signals detected: {signals_text}")
            
            return "\n\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error preparing filing summary: {e}")
            return filing.content[:2000] if filing.content else ""
    
    def _clean_filing_content(self, content: str) -> str:
        """Clean HTML and formatting from filing content."""
        try:
            from bs4 import BeautifulSoup
            
            # Parse HTML and extract text
            soup = BeautifulSoup(content, 'html.parser')
            clean_text = soup.get_text(separator=' ', strip=True)
            
            # Remove excessive whitespace
            import re
            clean_text = re.sub(r'\s+', ' ', clean_text)
            
            return clean_text
            
        except Exception:
            # Fallback: basic text cleaning
            import re
            clean_text = re.sub(r'<[^>]+>', '', content)  # Remove HTML tags
            clean_text = re.sub(r'\s+', ' ', clean_text)  # Normalize whitespace
            return clean_text
    
    def _create_sec_filing_prompt(self, symbol: str, filing: SECFiling, summary: str) -> str:
        """Create prompt for SEC filing sentiment analysis."""
        return f"""
        Analyze the sentiment and key insights from this SEC {filing.filing_type} filing for {symbol} ({filing.company_name}):
        
        Filing Date: {filing.filing_date}
        Form Type: {filing.filing_type}
        
        Content Summary:
        {summary}
        
        Please analyze:
        1. Overall sentiment (very negative, negative, neutral, positive, very positive)
        2. Confidence level (0-1)
        3. Key themes and topics mentioned
        4. Risk factors identified
        5. Growth opportunities mentioned
        6. Any material changes or events
        7. Management tone and outlook
        
        Focus on information that would be relevant for investment sentiment analysis.
        Pay special attention to forward-looking statements, risk disclosures, and material changes.
        """
    
    def _enhance_sec_sentiment(self, sentiment: SentimentAnalysis, filing: SECFiling) -> SentimentAnalysis:
        """Enhance sentiment analysis with SEC-specific signals."""
        try:
            # Adjust sentiment based on SEC-specific signals
            signal_adjustment = 0.0
            additional_risks = []
            additional_opportunities = []
            
            for signal in filing.sentiment_signals:
                if 'financial_stress' in signal or 'departure' in signal:
                    signal_adjustment -= 0.1
                    additional_risks.append(signal.replace('_', ' ').title())
                elif 'positive' in signal or 'record' in signal:
                    signal_adjustment += 0.1
                    additional_opportunities.append(signal.replace('_', ' ').title())
                elif 'ma_' in signal:  # M&A activity
                    additional_opportunities.append("M&A Activity")
            
            # Create enhanced sentiment
            enhanced_sentiment = SentimentAnalysis(
                score=max(-1.0, min(1.0, sentiment.score + signal_adjustment)),
                sentiment=sentiment.sentiment,
                confidence=sentiment.confidence,
                key_phrases=sentiment.key_phrases,
                risk_factors=list(set(sentiment.risk_factors + additional_risks)),
                opportunities=list(set(sentiment.opportunities + additional_opportunities)),
                reasoning=sentiment.reasoning + f" SEC signals adjustment: {signal_adjustment:.2f}"
            )
            
            return enhanced_sentiment
            
        except Exception as e:
            logger.error(f"Error enhancing SEC sentiment: {e}")
            return sentiment
    
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
    
    async def _analyze_crypto_sentiment(self, symbol: str) -> ComprehensiveSentiment:
        """Specialized sentiment analysis for cryptocurrency symbols."""
        
        try:
            # Use crypto sentiment analyzer
            # CRYPTO TRADING DISABLED - Skip crypto sentiment analysis
            # crypto_result = await crypto_sentiment_analyzer.analyze_crypto_sentiment(symbol)
            crypto_result = None
            
            # Convert to standard ComprehensiveSentiment format
            sentiment_result = ComprehensiveSentiment(
                symbol=symbol,
                timestamp=crypto_result.timestamp,
                overall_score=crypto_result.overall_score,
                overall_sentiment=crypto_result.overall_sentiment,
                confidence=crypto_result.confidence,
                
                # Map crypto components to standard format
                news_sentiment=SentimentAnalysis(
                    score=crypto_result.news_sentiment,
                    confidence=crypto_result.confidence,
                    reasoning=f"Analyzed {crypto_result.news_count} crypto news articles"
                ) if crypto_result.news_count > 0 else None,
                
                social_sentiment={
                    'crypto_social': SentimentAnalysis(
                        score=crypto_result.social_sentiment,
                        confidence=crypto_result.confidence,
                        reasoning=f"Social mentions: {crypto_result.social_mentions}"
                    )
                } if crypto_result.social_mentions > 0 else {},
                
                market_sentiment=SentimentAnalysis(
                    score=crypto_result.technical_sentiment,
                    confidence=min(0.8, abs(crypto_result.price_momentum) / 10),
                    reasoning=f"Price momentum: {crypto_result.price_momentum:.2f}%"
                ),
                
                # Crypto-specific metadata
                data_sources_count=3 if crypto_result.news_count > 0 else 2,
                news_articles_count=crypto_result.news_count,
                social_posts_count=crypto_result.social_mentions,
                
                # Map crypto insights to standard format
                key_themes=crypto_result.key_themes,
                risk_factors=crypto_result.risk_factors,
                opportunities=[theme for theme in crypto_result.key_themes 
                             if theme in ['adoption', 'technology', 'partnerships']],
                
                # Additional crypto metrics
                volume_spike=crypto_result.volume_spike,
                news_sources=crypto_result.news_sources
            )
            
            # Cache the result
            cache_key = f"{symbol}_{datetime.now().date()}"
            self.cache[cache_key] = (sentiment_result, datetime.now())
            
            logger.info(f"✅ Crypto sentiment analysis complete for {symbol}: "
                       f"{sentiment_result.overall_sentiment} ({sentiment_result.overall_score:.3f})")
            
            return sentiment_result
            
        except Exception as e:
            logger.error(f"Error in crypto sentiment analysis for {symbol}: {e}")
            
            # Return neutral result on error
            return ComprehensiveSentiment(
                symbol=symbol,
                timestamp=datetime.now(),
                overall_score=0.0,
                overall_sentiment='neutral',
                confidence=0.1,
                data_sources_count=0,
                key_themes=['insufficient_data'],
                risk_factors=['analysis_error']
            )
    
    async def close(self):
        """Close all connections and clean up resources."""
        try:
            await self.llm_analyzer.close()
        except Exception as e:
            logger.debug(f"LLM analyzer close error: {e}")
        
        try:
            await self.social_collector.cleanup()
        except Exception as e:
            logger.debug(f"Social collector cleanup error: {e}")
        
        try:
            if hasattr(self.earnings_scraper, 'close'):
                await self.earnings_scraper.close()
            elif hasattr(self.earnings_scraper, 'cleanup'):
                await self.earnings_scraper.cleanup()
        except Exception as e:
            logger.debug(f"Earnings scraper close error: {e}")
        
        try:
            await self.sec_edgar_client.cleanup()
        except Exception as e:
            logger.debug(f"SEC client cleanup error: {e}")
        
        try:
            # CRYPTO TRADING DISABLED - Skip crypto sentiment analyzer cleanup
            # await crypto_sentiment_analyzer.close()
            pass
        except Exception as e:
            logger.debug(f"Crypto sentiment analyzer close error: {e}")


# Global sentiment agent instance
sentiment_agent = SentimentAgent()