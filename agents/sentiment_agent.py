#!/usr/bin/env python3
"""
Consumer Financial Sentiment Agent

Three-layer architecture for tracking financial sentiment from social media:
1. Ingestion Layer - Reddit, Twitter/X, StockTwits data collection
2. Processing & Reasoning Layer - LLM-based sentiment analysis and trend detection
3. Structured Output & Handoff - JSON messages to other agents

Designed as a modular service with configurable intervals and structured output.
"""

import asyncio
import logging
import json
import time
import re
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from collections import deque, defaultdict

# Optional Redis import
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

# Reddit API with fallback
try:
    import praw
    from prawcore import NotFound, Forbidden
    REDDIT_AVAILABLE = True
except ImportError:
    praw = None
    NotFound = Exception
    Forbidden = Exception  
    REDDIT_AVAILABLE = False

# LLM Integration
from tools.llm_client import llm_client, LLMResponse
from config.settings import settings

logger = logging.getLogger(__name__)

class SentimentScore(Enum):
    """Sentiment classification levels."""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"  
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"

class TrendSignal(Enum):
    """Trend detection signals."""
    SURGE_POSITIVE = "surge_positive"
    SURGE_NEGATIVE = "surge_negative"
    VOLUME_SPIKE = "volume_spike"
    COORDINATION_DETECTED = "coordination_detected"
    NORMAL = "normal"

@dataclass
class SocialPost:
    """Individual social media post data structure."""
    id: str
    platform: str  # 'reddit', 'twitter', 'stocktwits'
    content: str
    author: str
    timestamp: datetime
    upvotes: int = 0
    comments: int = 0
    url: str = ""
    subreddit: str = ""  # For Reddit posts
    
@dataclass
class ExtractedEntity:
    """Extracted financial entity from social content."""
    ticker: str
    confidence: float
    context: str  # Surrounding text
    entity_type: str  # 'stock', 'etf', 'crypto', 'sector'

@dataclass
class SentimentAnalysis:
    """LLM sentiment analysis result."""
    post_id: str
    sentiment: SentimentScore
    confidence: float
    entities: List[ExtractedEntity]
    reasoning: str
    trend_signals: List[TrendSignal]
    timestamp: datetime

@dataclass
class StructuredSentimentOutput:
    """Final structured output for other agents."""
    ticker: str
    overall_sentiment: SentimentScore
    confidence: float
    volume: int  # Number of mentions
    trend_signal: TrendSignal
    key_insights: List[str]
    time_window: str
    data_sources: List[str]
    timestamp: datetime
    raw_posts_sample: List[str] = field(default_factory=list)

class SocialMediaIngestionLayer:
    """Layer 1: Efficient social media content ingestion."""
    
    def __init__(self):
        self.reddit_client = None
        self.twitter_client = None
        self.stocktwits_client = None
        self.content_cache = {}
        self.rate_limits = {
            'reddit': {'calls': 0, 'reset_time': time.time()},
            'twitter': {'calls': 0, 'reset_time': time.time()},
            'stocktwits': {'calls': 0, 'reset_time': time.time()}
        }
        
        # Initialize clients
        self._init_reddit_client()
        # self._init_twitter_client()  # Add when needed
        # self._init_stocktwits_client()  # Add when needed
        
    def _init_reddit_client(self):
        """Initialize Reddit client using PRAW."""
        if not REDDIT_AVAILABLE:
            logger.warning("⚠️ PRAW (Reddit API) not installed, sentiment analysis limited")
            self.reddit_client = None
            return
            
        try:
            # Check for Reddit credentials in settings
            reddit_credentials = {
                'client_id': getattr(settings, 'reddit_client_id', None),
                'client_secret': getattr(settings, 'reddit_client_secret', None), 
                'user_agent': getattr(settings, 'reddit_user_agent', 'TradingBot/1.0')
            }
            
            if reddit_credentials['client_id'] and reddit_credentials['client_secret']:
                self.reddit_client = praw.Reddit(
                    client_id=reddit_credentials['client_id'],
                    client_secret=reddit_credentials['client_secret'],
                    user_agent=reddit_credentials['user_agent']
                )
                logger.info("✅ Reddit client initialized")
            else:
                logger.warning("⚠️ Reddit credentials not configured, using read-only mode")
                # Fallback to read-only Reddit client
                self.reddit_client = praw.Reddit(
                    client_id='dummy',
                    client_secret='dummy', 
                    user_agent='TradingBot/1.0'
                )
                
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")
            self.reddit_client = None
    
    async def fetch_reddit_posts(self, 
                                subreddits: List[str] = None,
                                keywords: List[str] = None,
                                limit: int = 100,
                                time_filter: str = "hour") -> List[SocialPost]:
        """Fetch recent Reddit posts with keyword/ticker filtering."""
        if not self.reddit_client:
            logger.warning("Reddit client not available")
            return []
            
        if not subreddits:
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'SecurityAnalysis']
            
        if not keywords:
            keywords = ['$', 'buy', 'sell', 'calls', 'puts', 'earnings', 'moon', 'diamond hands']
        
        posts = []
        
        try:
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit_client.subreddit(subreddit_name)
                    
                    # Get hot posts from subreddit
                    for submission in subreddit.hot(limit=limit//len(subreddits)):
                        # Apply keyword filtering
                        content = f"{submission.title} {submission.selftext}".lower()
                        
                        if any(keyword.lower() in content for keyword in keywords):
                            post = SocialPost(
                                id=submission.id,
                                platform='reddit',
                                content=f"{submission.title}\n\n{submission.selftext}"[:500],
                                author=str(submission.author) if submission.author else 'deleted',
                                timestamp=datetime.fromtimestamp(submission.created_utc),
                                upvotes=submission.score,
                                comments=submission.num_comments,
                                url=f"https://reddit.com{submission.permalink}",
                                subreddit=subreddit_name
                            )
                            posts.append(post)
                            
                except (NotFound, Forbidden) as e:
                    logger.warning(f"Cannot access subreddit {subreddit_name}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error fetching Reddit posts: {e}")
            
        logger.info(f"Fetched {len(posts)} filtered Reddit posts")
        return posts
    
    async def fetch_social_content(self, 
                                  platforms: List[str] = None,
                                  tickers: List[str] = None) -> List[SocialPost]:
        """Unified method to fetch content from multiple platforms."""
        if not platforms:
            platforms = ['reddit']  # Start with Reddit, expand later
            
        all_posts = []
        
        if 'reddit' in platforms:
            # Create ticker-specific keywords
            keywords = ['$', 'buy', 'sell', 'calls', 'puts', 'earnings']
            if tickers:
                keywords.extend([f"${ticker}" for ticker in tickers])
                keywords.extend(tickers)
            
            reddit_posts = await self.fetch_reddit_posts(keywords=keywords)
            all_posts.extend(reddit_posts)
        
        # Add Twitter/StockTwits here when implemented
        # if 'twitter' in platforms:
        #     twitter_posts = await self.fetch_twitter_posts(tickers=tickers)
        #     all_posts.extend(twitter_posts)
        
        return all_posts

class LLMSentimentProcessor:
    """Layer 2: LLM-based processing and reasoning."""
    
    def __init__(self):
        self.entity_cache = {}  # Cache for ticker recognition
        self.sentiment_history = deque(maxlen=1000)  # Recent sentiment history
        
    async def extract_entities(self, content: str) -> List[ExtractedEntity]:
        """Extract financial entities (tickers, sectors) from content."""
        try:
            # Create LLM prompt for entity extraction
            analysis_data = {
                "content": content,
                "instructions": """
                Extract all financial entities from this social media content.
                
                Look for:
                1. Stock tickers (e.g., AAPL, $TSLA, SPY)
                2. Company names (e.g., Apple, Tesla, Microsoft)
                3. Sectors (e.g., tech, healthcare, energy)
                4. ETFs and funds
                
                For each entity found, provide:
                - ticker symbol (if stock)
                - confidence level (0-1)
                - surrounding context (5-10 words around mention)
                - entity type (stock, etf, crypto, sector, company)
                
                Return as JSON list with format:
                [{"ticker": "AAPL", "confidence": 0.95, "context": "buying AAPL calls", "entity_type": "stock"}]
                """
            }
            
            llm_response = await llm_client.analyze_financial_data(
                agent_name="Entity Extractor",
                analysis_data=analysis_data,
                system_prompt="You are a financial entity extraction specialist. Extract tickers, companies, and financial instruments from social media content.",
                temperature=0.3
            )
            
            # Parse LLM response to extract entities
            entities = self._parse_entity_response(llm_response.content, content)
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            # Fallback to simple regex extraction
            return self._fallback_entity_extraction(content)
    
    def _parse_entity_response(self, llm_content: str, original_content: str) -> List[ExtractedEntity]:
        """Parse LLM response to extract entities."""
        entities = []
        
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\[.*\]', llm_content, re.DOTALL)
            if json_match:
                entity_data = json.loads(json_match.group())
                
                for item in entity_data:
                    if isinstance(item, dict) and 'ticker' in item:
                        entity = ExtractedEntity(
                            ticker=item.get('ticker', '').upper(),
                            confidence=float(item.get('confidence', 0.5)),
                            context=item.get('context', '')[:100],
                            entity_type=item.get('entity_type', 'unknown')
                        )
                        entities.append(entity)
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM entity response: {e}")
            # Fall back to regex extraction
            return self._fallback_entity_extraction(original_content)
            
        return entities
    
    def _fallback_entity_extraction(self, content: str) -> List[ExtractedEntity]:
        """Simple regex-based entity extraction as fallback."""
        entities = []
        
        # Look for ticker patterns
        ticker_patterns = [
            r'\$([A-Z]{1,5})\b',  # $AAPL pattern
            r'\b([A-Z]{2,5})\b',  # AAPL pattern (be careful with false positives)
        ]
        
        for pattern in ticker_patterns:
            matches = re.finditer(pattern, content.upper())
            for match in matches:
                ticker = match.group(1) if pattern.startswith(r'\$') else match.group(1)
                
                # Filter out common false positives
                if ticker not in ['THE', 'AND', 'FOR', 'YOU', 'ARE', 'NOT', 'BUT', 'CAN', 'ALL']:
                    entity = ExtractedEntity(
                        ticker=ticker,
                        confidence=0.7 if pattern.startswith(r'\$') else 0.4,
                        context=content[max(0, match.start()-20):match.end()+20],
                        entity_type='stock'
                    )
                    entities.append(entity)
        
        return entities
    
    async def analyze_sentiment(self, posts: List[SocialPost]) -> List[SentimentAnalysis]:
        """Analyze sentiment for a batch of social posts."""
        analyses = []
        
        for post in posts:
            try:
                analysis = await self._analyze_single_post(post)
                analyses.append(analysis)
            except Exception as e:
                logger.error(f"Failed to analyze post {post.id}: {e}")
                
        return analyses
    
    async def _analyze_single_post(self, post: SocialPost) -> SentimentAnalysis:
        """Analyze sentiment for a single post using LLM."""
        try:
            # Extract entities first
            entities = await self.extract_entities(post.content)
            
            # Create sentiment analysis prompt
            analysis_data = {
                "content": post.content,
                "platform": post.platform,
                "upvotes": post.upvotes,
                "comments": post.comments,
                "entities": [e.ticker for e in entities],
                "instructions": """
                Analyze the financial sentiment of this social media post.
                
                Consider:
                1. Overall sentiment (very_positive, positive, neutral, negative, very_negative)
                2. Confidence level (0-1)
                3. Specific sentiment toward mentioned tickers
                4. Market trend signals (surge_positive, surge_negative, volume_spike, coordination_detected, normal)
                5. Context clues (earnings, news events, technical analysis mentions)
                
                Provide reasoning for your sentiment classification.
                Look for patterns like:
                - Coordinated posting (copy-paste content, bot-like behavior)
                - Volume surges (unusually high engagement)
                - Extreme sentiment (rocket emojis, diamond hands, etc.)
                
                Return sentiment as one of: very_positive, positive, neutral, negative, very_negative
                Return trend_signal as one of: surge_positive, surge_negative, volume_spike, coordination_detected, normal
                """
            }
            
            llm_response = await llm_client.analyze_financial_data(
                agent_name="Sentiment Analyzer",
                analysis_data=analysis_data,
                system_prompt="You are a financial sentiment analysis specialist. Analyze social media posts for trading sentiment, market trends, and coordination patterns.",
                temperature=0.4
            )
            
            # Parse sentiment response
            sentiment, trend_signals = self._parse_sentiment_response(llm_response.content)
            
            analysis = SentimentAnalysis(
                post_id=post.id,
                sentiment=sentiment,
                confidence=llm_response.confidence,
                entities=entities,
                reasoning=llm_response.content[:300],
                trend_signals=trend_signals,
                timestamp=datetime.now()
            )
            
            # Store in history for trend analysis
            self.sentiment_history.append(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for post {post.id}: {e}")
            # Return neutral sentiment as fallback
            return SentimentAnalysis(
                post_id=post.id,
                sentiment=SentimentScore.NEUTRAL,
                confidence=0.0,
                entities=entities if 'entities' in locals() else [],
                reasoning=f"Analysis failed: {e}",
                trend_signals=[TrendSignal.NORMAL],
                timestamp=datetime.now()
            )
    
    def _parse_sentiment_response(self, llm_content: str) -> Tuple[SentimentScore, List[TrendSignal]]:
        """Parse LLM sentiment analysis response."""
        sentiment = SentimentScore.NEUTRAL  # Default
        trend_signals = [TrendSignal.NORMAL]  # Default
        
        content_lower = llm_content.lower()
        
        # Parse sentiment
        if 'very_positive' in content_lower:
            sentiment = SentimentScore.VERY_POSITIVE
        elif 'very_negative' in content_lower:
            sentiment = SentimentScore.VERY_NEGATIVE
        elif 'positive' in content_lower:
            sentiment = SentimentScore.POSITIVE
        elif 'negative' in content_lower:
            sentiment = SentimentScore.NEGATIVE
        else:
            sentiment = SentimentScore.NEUTRAL
        
        # Parse trend signals
        trend_signals = []
        if 'surge_positive' in content_lower:
            trend_signals.append(TrendSignal.SURGE_POSITIVE)
        elif 'surge_negative' in content_lower:
            trend_signals.append(TrendSignal.SURGE_NEGATIVE)
        elif 'volume_spike' in content_lower:
            trend_signals.append(TrendSignal.VOLUME_SPIKE)
        elif 'coordination_detected' in content_lower:
            trend_signals.append(TrendSignal.COORDINATION_DETECTED)
        else:
            trend_signals.append(TrendSignal.NORMAL)
        
        return sentiment, trend_signals

class StructuredOutputHandler:
    """Layer 3: Structured output and handoff to other agents."""
    
    def __init__(self):
        self.redis_client = None
        self.memory_queue = deque(maxlen=1000)  # Fallback in-memory queue
        self._init_redis()
        
    def _init_redis(self):
        """Initialize Redis client for inter-agent communication."""
        try:
            if REDIS_AVAILABLE:
                # Try to connect to Redis
                redis_host = getattr(settings, 'redis_host', 'localhost')
                redis_port = getattr(settings, 'redis_port', 6379)
                redis_db = getattr(settings, 'redis_db', 0)
                
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port, 
                    db=redis_db,
                    decode_responses=True
                )
                
                # Test connection
                self.redis_client.ping()
                logger.info("✅ Redis client initialized")
            else:
                logger.warning("⚠️ Redis not available, using in-memory queue")
                
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using in-memory queue")
            self.redis_client = None
    
    async def aggregate_sentiment_by_ticker(self, 
                                          analyses: List[SentimentAnalysis],
                                          time_window: str = "1h") -> List[StructuredSentimentOutput]:
        """Aggregate individual sentiment analyses by ticker."""
        ticker_data = defaultdict(list)
        
        # Group analyses by ticker
        for analysis in analyses:
            for entity in analysis.entities:
                if entity.confidence > 0.5:  # Only high-confidence entities
                    ticker_data[entity.ticker].append(analysis)
        
        # Create structured output for each ticker
        structured_outputs = []
        
        for ticker, ticker_analyses in ticker_data.items():
            if len(ticker_analyses) < 2:  # Minimum threshold for meaningful analysis
                continue
                
            output = await self._create_ticker_summary(ticker, ticker_analyses, time_window)
            structured_outputs.append(output)
        
        return structured_outputs
    
    async def _create_ticker_summary(self, 
                                   ticker: str,
                                   analyses: List[SentimentAnalysis],
                                   time_window: str) -> StructuredSentimentOutput:
        """Create aggregated sentiment summary for a specific ticker."""
        
        # Calculate overall sentiment
        sentiment_scores = []
        sentiment_map = {
            SentimentScore.VERY_NEGATIVE: -2,
            SentimentScore.NEGATIVE: -1,
            SentimentScore.NEUTRAL: 0,
            SentimentScore.POSITIVE: 1,
            SentimentScore.VERY_POSITIVE: 2
        }
        
        for analysis in analyses:
            score = sentiment_map[analysis.sentiment]
            sentiment_scores.append(score)
        
        # Weighted average sentiment
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        # Convert back to enum
        if avg_sentiment > 1.5:
            overall_sentiment = SentimentScore.VERY_POSITIVE
        elif avg_sentiment > 0.5:
            overall_sentiment = SentimentScore.POSITIVE
        elif avg_sentiment > -0.5:
            overall_sentiment = SentimentScore.NEUTRAL
        elif avg_sentiment > -1.5:
            overall_sentiment = SentimentScore.NEGATIVE
        else:
            overall_sentiment = SentimentScore.VERY_NEGATIVE
        
        # Calculate confidence (based on volume and consistency)
        confidence = min(len(analyses) / 10, 1.0)  # Volume factor
        sentiment_std = np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0
        consistency_factor = max(0, 1 - sentiment_std / 2)  # Consistency factor
        total_confidence = (confidence + consistency_factor) / 2
        
        # Detect trend signals
        trend_signal = self._detect_dominant_trend(analyses)
        
        # Extract key insights
        key_insights = self._extract_key_insights(ticker, analyses)
        
        # Sample of raw posts for context
        raw_posts_sample = [analysis.reasoning[:100] for analysis in analyses[:3]]
        
        return StructuredSentimentOutput(
            ticker=ticker,
            overall_sentiment=overall_sentiment,
            confidence=total_confidence,
            volume=len(analyses),
            trend_signal=trend_signal,
            key_insights=key_insights,
            time_window=time_window,
            data_sources=['reddit'],  # Expand as we add more platforms
            timestamp=datetime.now(),
            raw_posts_sample=raw_posts_sample
        )
    
    def _detect_dominant_trend(self, analyses: List[SentimentAnalysis]) -> TrendSignal:
        """Detect the dominant trend signal from multiple analyses."""
        trend_counts = defaultdict(int)
        
        for analysis in analyses:
            for trend in analysis.trend_signals:
                trend_counts[trend] += 1
        
        # Return most common trend, default to NORMAL
        if not trend_counts:
            return TrendSignal.NORMAL
            
        return max(trend_counts.items(), key=lambda x: x[1])[0]
    
    def _extract_key_insights(self, ticker: str, analyses: List[SentimentAnalysis]) -> List[str]:
        """Extract key insights from sentiment analyses."""
        insights = []
        
        # Volume insight
        if len(analyses) > 20:
            insights.append(f"High social media volume detected for {ticker} ({len(analyses)} mentions)")
        
        # Sentiment consistency
        sentiments = [a.sentiment for a in analyses]
        positive_count = sum(1 for s in sentiments if s in [SentimentScore.POSITIVE, SentimentScore.VERY_POSITIVE])
        negative_count = sum(1 for s in sentiments if s in [SentimentScore.NEGATIVE, SentimentScore.VERY_NEGATIVE])
        
        if positive_count > len(analyses) * 0.7:
            insights.append(f"Strong positive sentiment consensus ({positive_count}/{len(analyses)} posts)")
        elif negative_count > len(analyses) * 0.7:
            insights.append(f"Strong negative sentiment consensus ({negative_count}/{len(analyses)} posts)")
        
        # Trend signals
        trend_counts = defaultdict(int)
        for analysis in analyses:
            for trend in analysis.trend_signals:
                trend_counts[trend] += 1
        
        for trend, count in trend_counts.items():
            if trend != TrendSignal.NORMAL and count > len(analyses) * 0.3:
                insights.append(f"Trend signal detected: {trend.value} ({count} occurrences)")
        
        return insights[:5]  # Limit to top 5 insights
    
    async def emit_structured_output(self, outputs: List[StructuredSentimentOutput]):
        """Emit structured outputs to other agents via Redis or in-memory queue."""
        for output in outputs:
            message = {
                'type': 'sentiment_analysis',
                'ticker': output.ticker,
                'data': {
                    'sentiment': output.overall_sentiment.value,
                    'confidence': output.confidence,
                    'volume': output.volume,
                    'trend_signal': output.trend_signal.value,
                    'key_insights': output.key_insights,
                    'time_window': output.time_window,
                    'timestamp': output.timestamp.isoformat(),
                    'data_sources': output.data_sources
                }
            }
            
            await self._send_message(message)
    
    async def _send_message(self, message: Dict[str, Any]):
        """Send message via Redis or in-memory queue."""
        try:
            if self.redis_client:
                # Send to Redis queue
                channel = f"sentiment_updates_{message['ticker']}"
                self.redis_client.lpush(channel, json.dumps(message))
                self.redis_client.expire(channel, 3600)  # Expire after 1 hour
                logger.debug(f"Sent sentiment update for {message['ticker']} to Redis")
            else:
                # Use in-memory queue as fallback
                self.memory_queue.append(message)
                logger.debug(f"Sent sentiment update for {message['ticker']} to memory queue")
                
        except Exception as e:
            logger.error(f"Failed to send sentiment message: {e}")

class ConsumerSentimentAgent:
    """Main sentiment agent orchestrating all three layers."""
    
    def __init__(self):
        self.ingestion_layer = SocialMediaIngestionLayer()
        self.processing_layer = LLMSentimentProcessor() 
        self.output_layer = StructuredOutputHandler()
        self.is_running = False
        self.polling_interval = 300  # 5 minutes default
        
    async def run_sentiment_cycle(self, 
                                tickers: List[str] = None,
                                platforms: List[str] = None) -> List[StructuredSentimentOutput]:
        """Run a complete sentiment analysis cycle."""
        logger.info("🔍 Starting sentiment analysis cycle...")
        
        try:
            # Layer 1: Ingest social content
            posts = await self.ingestion_layer.fetch_social_content(
                platforms=platforms or ['reddit'],
                tickers=tickers
            )
            
            if not posts:
                logger.warning("No social media posts retrieved")
                return []
            
            logger.info(f"Ingested {len(posts)} social media posts")
            
            # Layer 2: Process with LLM
            analyses = await self.processing_layer.analyze_sentiment(posts)
            
            if not analyses:
                logger.warning("No sentiment analyses produced")
                return []
                
            logger.info(f"Analyzed sentiment for {len(analyses)} posts")
            
            # Layer 3: Aggregate and structure output
            structured_outputs = await self.output_layer.aggregate_sentiment_by_ticker(analyses)
            
            logger.info(f"Generated {len(structured_outputs)} ticker sentiment summaries")
            
            # Emit to other agents
            await self.output_layer.emit_structured_output(structured_outputs)
            
            return structured_outputs
            
        except Exception as e:
            logger.error(f"Sentiment cycle failed: {e}")
            return []
    
    async def start_continuous_monitoring(self, 
                                        tickers: List[str] = None,
                                        interval_seconds: int = 300):
        """Start continuous sentiment monitoring."""
        self.is_running = True
        self.polling_interval = interval_seconds
        
        logger.info(f"🚀 Starting continuous sentiment monitoring (interval: {interval_seconds}s)")
        
        while self.is_running:
            try:
                outputs = await self.run_sentiment_cycle(tickers=tickers)
                
                if outputs:
                    logger.info(f"✅ Sentiment cycle complete: {len(outputs)} ticker updates")
                    for output in outputs:
                        logger.info(f"   {output.ticker}: {output.overall_sentiment.value} "
                                  f"(confidence: {output.confidence:.2f}, volume: {output.volume})")
                else:
                    logger.info("ℹ️ Sentiment cycle complete: no significant updates")
                    
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                
            # Wait for next cycle
            await asyncio.sleep(self.polling_interval)
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.is_running = False
        logger.info("🛑 Stopping sentiment monitoring")
    
    async def get_recent_sentiment(self, ticker: str) -> Optional[StructuredSentimentOutput]:
        """Get most recent sentiment analysis for a specific ticker."""
        try:
            if self.output_layer.redis_client:
                # Try to get from Redis
                channel = f"sentiment_updates_{ticker}"
                recent_message = self.output_layer.redis_client.lrange(channel, 0, 0)
                
                if recent_message:
                    data = json.loads(recent_message[0])
                    return self._message_to_output(data)
            else:
                # Check in-memory queue
                for message in reversed(self.output_layer.memory_queue):
                    if message['ticker'] == ticker:
                        return self._message_to_output(message)
                        
        except Exception as e:
            logger.error(f"Failed to get recent sentiment for {ticker}: {e}")
            
        return None
    
    def _message_to_output(self, message: Dict[str, Any]) -> StructuredSentimentOutput:
        """Convert message back to StructuredSentimentOutput."""
        data = message['data']
        
        return StructuredSentimentOutput(
            ticker=message['ticker'],
            overall_sentiment=SentimentScore(data['sentiment']),
            confidence=data['confidence'],
            volume=data['volume'],
            trend_signal=TrendSignal(data['trend_signal']),
            key_insights=data['key_insights'],
            time_window=data['time_window'],
            data_sources=data['data_sources'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            raw_posts_sample=[]
        )

# Global sentiment agent instance
sentiment_agent = ConsumerSentimentAgent()

# Export main components
__all__ = [
    'ConsumerSentimentAgent',
    'StructuredSentimentOutput',
    'SentimentScore',
    'TrendSignal', 
    'SocialPost',
    'sentiment_agent'
]