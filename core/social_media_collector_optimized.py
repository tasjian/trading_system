"""
Optimized Social Media Data Collector

Focused on core improvements:
- Efficient ticker detection
- Proper session management
- Intelligent caching
- Fast fallback when APIs are unavailable
- No session leaks
"""

import asyncio
import logging
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
import hashlib
from pathlib import Path

import aiohttp
import asyncpraw
import tweepy

from config.settings import settings
from utils.connection_pool import connection_pool

logger = logging.getLogger(__name__)

# Optimized ticker detection patterns
TICKER_PATTERNS = [
    r'\$([A-Z]{1,5})\b',  # $AAPL format
    r'\b([A-Z]{2,5})\s+(?:stock|shares|calls|puts|options|mooning|rocket)\b',
    r'\b(?:bought|sold|holding|long|short)\s+([A-Z]{2,5})\b',
]

@dataclass
class SocialMediaPost:
    """Lightweight social media post structure."""
    platform: str
    post_id: str
    author: str
    content: str
    timestamp: datetime
    score: int = 0
    tickers: List[str] = None
    
    def __post_init__(self):
        if self.tickers is None:
            self.tickers = []

class OptimizedTickerDetector:
    """High-performance ticker detection."""
    
    def __init__(self):
        self.ticker_regex = re.compile('|'.join(TICKER_PATTERNS), re.IGNORECASE)
        # Common false positives to filter out
        self.false_positives = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER',
            'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'USE', 'HIM', 'OLD', 'SEE',
            'NOW', 'WAY', 'WHO', 'ITS', 'DID', 'YES', 'HIS', 'HAS', 'HAD', 'LET',
            'PUT', 'TOO', 'OLD', 'ANY', 'MAY', 'SAY', 'SHE', 'TWO', 'HOW', 'BOY'
        }
    
    def extract_tickers(self, text: str) -> Set[str]:
        """Extract valid ticker symbols from text."""
        tickers = set()
        
        for match in self.ticker_regex.finditer(text):
            for group in match.groups():
                if group and 2 <= len(group) <= 5:
                    ticker = group.upper()
                    # Filter out common false positives
                    if ticker not in self.false_positives:
                        tickers.add(ticker)
        
        return tickers
    
    def is_ticker_relevant(self, text: str, target_ticker: str) -> bool:
        """Quick relevance check for target ticker."""
        text_upper = text.upper()
        target_upper = target_ticker.upper()
        
        # Check for direct mentions
        if f'${target_upper}' in text_upper:
            return True
        if target_upper in text_upper and any(word in text_upper for word in 
                                            ['STOCK', 'CALL', 'PUT', 'BUY', 'SELL']):
            return True
        
        return False

class FastSessionManager:
    """Efficient HTTP session manager with cleanup."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = aiohttp.ClientTimeout(total=timeout, connect=10)
        self._session_name = f"reddit_collector_{id(self)}"
        self._closed = False
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get session from connection pool."""
        return await connection_pool.get_async_session(
            name=self._session_name,
            timeout=self.timeout
        )
    
    async def get(self, url: str, **kwargs):
        """Make GET request with proper connection pool usage."""
        try:
            async with connection_pool.get_temp_session(timeout=self.timeout) as session:
                async with session.get(url, **kwargs) as response:
                    return await response.json() if response.status == 200 else None
        except Exception as e:
            logger.debug(f"Request failed: {e}")
            return None
    
    async def cleanup(self):
        """Clean up session via connection pool."""
        if not self._closed:
            await connection_pool.close_session(self._session_name)
            self._closed = True

class OptimizedRedditCollector:
    """Reddit collector with fallback and proper cleanup."""
    
    def __init__(self):
        self.reddit = None
        self.session_manager = FastSessionManager()
        self.ticker_detector = OptimizedTickerDetector()
        self._initialize_reddit()
    
    def _initialize_reddit(self):
        """Initialize Reddit with error handling."""
        try:
            if settings.reddit_client_id and settings.reddit_client_secret:
                self.reddit = asyncpraw.Reddit(
                    client_id=settings.reddit_client_id,
                    client_secret=settings.reddit_client_secret,
                    user_agent=settings.reddit_user_agent or "TradingBot/1.0",
                    read_only=True
                )
                logger.info("Reddit API initialized successfully")
            else:
                logger.info("Reddit API credentials not configured - using fallback")
        except Exception as e:
            logger.info(f"Reddit API unavailable: {e} - using fallback")
            self.reddit = None
    
    async def collect_posts_fast(self, symbol: str, limit: int = 20) -> List[SocialMediaPost]:
        """Fast Reddit collection with fallbacks."""
        posts = []
        
        if not self.reddit:
            # No fallback posts for trading - return empty list
            logger.warning(f"Reddit API unavailable for {symbol} - no fallback posts for trading")
            return []
        
        try:
            # Try official Reddit API with aggressive timeout
            subreddits = ['wallstreetbets', 'investing', 'stocks']
            
            for subreddit_name in subreddits[:2]:  # Limit to 2 subreddits
                try:
                    # Set timeout for Reddit operations
                    timeout_task = asyncio.create_task(self._collect_subreddit(subreddit_name, symbol, limit // 2))
                    subreddit_posts = await asyncio.wait_for(timeout_task, timeout=10.0)
                    posts.extend(subreddit_posts)
                    
                    if len(posts) >= limit:
                        break
                        
                except asyncio.TimeoutError:
                    logger.debug(f"Reddit timeout for r/{subreddit_name}")
                    continue
                except Exception as e:
                    logger.debug(f"Reddit error for r/{subreddit_name}: {e}")
                    continue
        
        except Exception as e:
            logger.debug(f"Reddit collection failed: {e}")
        
        # No fallback posts for trading - return what we got from real APIs only
        if len(posts) < 3:
            logger.warning(f"Only collected {len(posts)} posts for {symbol} - no fallback posts for trading")
        
        return posts[:limit]
    
    async def _collect_subreddit(self, subreddit_name: str, symbol: str, limit: int) -> List[SocialMediaPost]:
        """Collect from single subreddit with timeout."""
        posts = []
        
        try:
            subreddit = await self.reddit.subreddit(subreddit_name)
            
            # Search for posts with ticker
            async for submission in subreddit.search(f'${symbol}', limit=limit, time_filter='week'):
                text = f"{submission.title} {submission.selftext}"
                
                # Quick relevance check
                if not self.ticker_detector.is_ticker_relevant(text, symbol):
                    continue
                
                post = SocialMediaPost(
                    platform="reddit",
                    post_id=submission.id,
                    author=str(submission.author) if submission.author else "unknown",
                    content=text[:300],  # Limit content
                    timestamp=datetime.fromtimestamp(submission.created_utc),
                    score=submission.score,
                    tickers=list(self.ticker_detector.extract_tickers(text))
                )
                
                posts.append(post)
                
                if len(posts) >= limit:
                    break
        
        except Exception as e:
            logger.debug(f"Subreddit collection error: {e}")
        
        return posts
    
    # REMOVED: _generate_fallback_posts() method for Reddit
    # Synthetic/fallback social media posts are not allowed for trading decisions
    
    async def cleanup(self):
        """Clean up resources."""
        if self.reddit:
            try:
                await self.reddit.close()
            except Exception:
                pass
        await self.session_manager.cleanup()

class OptimizedTwitterCollector:
    """Twitter collector with rate limit handling."""
    
    def __init__(self):
        self.api = None
        self.ticker_detector = OptimizedTickerDetector()
        self._initialize_twitter()
    
    def _initialize_twitter(self):
        """Initialize Twitter API."""
        try:
            if settings.twitter_bearer_token:
                self.api = tweepy.Client(
                    bearer_token=settings.twitter_bearer_token,
                    wait_on_rate_limit=False  # Don't wait for rate limits
                )
                logger.info("Twitter API initialized successfully (no rate limit blocking)")
            else:
                logger.info("Twitter API credentials not configured")
        except Exception as e:
            logger.info(f"Twitter API unavailable: {e}")
    
    async def collect_posts_fast(self, symbol: str, limit: int = 10) -> List[SocialMediaPost]:
        """Fast Twitter collection with rate limit handling."""
        posts = []
        
        if not self.api:
            logger.warning(f"Twitter API unavailable for {symbol} - no fallback posts for trading")
            return []
        
        try:
            # Quick search with timeout
            query = f"${symbol} -is:retweet lang:en"
            
            # Use a much smaller limit to avoid rate limits
            tweets = self.api.search_recent_tweets(
                query=query,
                max_results=min(limit, 10),  # Very conservative limit
                tweet_fields=['created_at', 'public_metrics']
            )
            
            if tweets and tweets.data:
                for tweet in tweets.data:
                    if not self.ticker_detector.is_ticker_relevant(tweet.text, symbol):
                        continue
                    
                    post = SocialMediaPost(
                        platform="twitter",
                        post_id=str(tweet.id),
                        author="twitter_user",
                        content=tweet.text[:200],
                        timestamp=tweet.created_at,
                        score=0,
                        tickers=list(self.ticker_detector.extract_tickers(tweet.text))
                    )
                    posts.append(post)
        
        except Exception as e:
            logger.debug(f"Twitter collection failed: {e}")
            # No fallback posts for trading - return empty list
            return []
        
        # No fallback posts for trading - return what we got from real APIs only
        if len(posts) < 2:
            logger.warning(f"Only collected {len(posts)} posts for {symbol} - no fallback posts for trading")
        
        return posts[:limit]
    
    # REMOVED: _generate_fallback_posts() method for Twitter  
    # Synthetic/fallback social media posts are not allowed for trading decisions

class OptimizedSocialMediaCollector:
    """Main optimized social media collector with no session leaks."""
    
    def __init__(self):
        self.reddit_collector = OptimizedRedditCollector()
        self.twitter_collector = OptimizedTwitterCollector()
    
    async def collect_all_platforms(self, symbol: str, limit_per_platform: int = 10) -> Dict[str, List]:
        """Collect from all platforms with enhanced caching and reduced timeouts."""
        try:
            # Import enhanced cache
            from utils.enhanced_api_cache import get_enhanced_cache, CacheType
            cache = await get_enhanced_cache()
            
            # Define collection function for caching
            async def collect_social_media():
                results = {'reddit': [], 'twitter': []}
                
                # Collect Reddit posts with reduced timeout (was 25s, now 15s)
                try:
                    reddit_task = asyncio.create_task(
                        self.reddit_collector.collect_posts_fast(symbol, limit_per_platform)
                    )
                    reddit_posts = await asyncio.wait_for(reddit_task, timeout=15.0)
                    results['reddit'] = reddit_posts
                    logger.info(f"✅ Collected {len(reddit_posts)} Reddit posts for {symbol}")
                    
                except asyncio.TimeoutError:
                    logger.debug(f"Reddit collection timeout for {symbol}")
                    results['reddit'] = []
                except Exception as e:
                    logger.debug(f"Reddit collection error for {symbol}: {e}")
                    results['reddit'] = []
                
                # Collect Twitter posts with reduced timeout (was 20s, now 10s)
                try:
                    twitter_task = asyncio.create_task(
                        self.twitter_collector.collect_posts_fast(symbol, limit_per_platform)
                    )
                    twitter_posts = await asyncio.wait_for(twitter_task, timeout=10.0)
                    results['twitter'] = twitter_posts
                    logger.info(f"✅ Collected {len(twitter_posts)} Twitter posts for {symbol}")
                    
                except asyncio.TimeoutError:
                    logger.debug(f"Twitter collection timeout for {symbol}")
                    results['twitter'] = []
                except Exception as e:
                    logger.debug(f"Twitter collection error for {symbol}: {e}")
                    results['twitter'] = []
                
                return results
            
            # Use cached API call with buffer and error handling
            cache_key = f"social_media_{symbol}"
            result = await cache.cached_api_call(CacheType.SOCIAL_MEDIA, cache_key, collect_social_media)
            return result if result is not None else {'reddit': [], 'twitter': []}
            
        except Exception as e:
            logger.warning(f"Enhanced social media collection error for {symbol}: {e}")
            return {'reddit': [], 'twitter': []}
    
    async def cleanup(self):
        """Clean up all resources."""
        try:
            await self.reddit_collector.cleanup()
        except Exception as e:
            logger.debug(f"Reddit cleanup error: {e}")

# Global instance for backward compatibility  
optimized_collector = OptimizedSocialMediaCollector()

# Backward compatibility wrapper
class SocialMediaCollector(OptimizedSocialMediaCollector):
    """Backward compatible wrapper."""
    pass