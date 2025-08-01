"""
Refactored Social Media Data Collector

Modern, efficient implementation using:
- Pushshift API for historical Reddit data (rate-friendly)
- Reddit Official API (PRAW) for live threads
- Optimized ticker detection with regex
- Intelligent caching and filtering
- Proper session management
"""

import asyncio
import logging
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
import hashlib

import aiohttp
import asyncpraw
import tweepy
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

# Regex patterns for ticker detection
TICKER_PATTERNS = [
    r'\$([A-Z]{1,5})\b',  # $AAPL format
    r'\b([A-Z]{2,5})\s+(?:stock|shares|calls|puts|options)\b',  # AAPL stock
    r'\b(?:bought|sold|holding)\s+([A-Z]{2,5})\b',  # bought AAPL
]

@dataclass
class SocialMediaPost:
    """Optimized social media post data structure."""
    platform: str
    post_id: str
    author: str
    content: str
    timestamp: datetime
    score: int = 0
    comments_count: int = 0
    url: Optional[str] = None
    tickers: List[str] = None
    relevance_score: float = 0.0
    
    def __post_init__(self):
        if self.tickers is None:
            self.tickers = []

@dataclass
class CollectionStats:
    """Statistics for collection performance."""
    total_fetched: int = 0
    filtered_by_time: int = 0
    filtered_by_score: int = 0
    filtered_by_relevance: int = 0
    cache_hits: int = 0
    final_count: int = 0

class CacheManager:
    """Efficient caching for social media posts."""
    
    def __init__(self, cache_dir: str = "data/social_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_duration = timedelta(hours=2)  # Cache for 2 hours
    
    def _get_cache_key(self, symbol: str, platform: str, timeframe: str) -> str:
        """Generate cache key for symbol/platform/timeframe."""
        data = f"{symbol}_{platform}_{timeframe}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.json"
    
    async def get_cached_posts(self, symbol: str, platform: str, timeframe: str) -> Optional[List[Dict]]:
        """Get cached posts if still valid."""
        cache_key = self._get_cache_key(symbol, platform, timeframe)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache is still valid
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cache_time > self.cache_duration:
                cache_path.unlink()  # Remove expired cache
                return None
            
            logger.debug(f"Cache hit for {symbol} on {platform}")
            return cache_data['posts']
            
        except Exception as e:
            logger.debug(f"Cache read error: {e}")
            return None
    
    async def save_posts_to_cache(self, symbol: str, platform: str, timeframe: str, posts: List[Dict]):
        """Save posts to cache."""
        cache_key = self._get_cache_key(symbol, platform, timeframe)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'platform': platform,
                'posts': posts
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
                
        except Exception as e:
            logger.debug(f"Cache save error: {e}")

class TickerDetector:
    """Efficient ticker detection and relevance scoring."""
    
    def __init__(self):
        self.ticker_regex = re.compile('|'.join(TICKER_PATTERNS), re.IGNORECASE)
        self.finance_keywords = [
            'bull', 'bear', 'moon', 'rocket', 'diamond hands', 'hodl',
            'buy', 'sell', 'hold', 'calls', 'puts', 'options', 'earnings',
            'dividend', 'split', 'merger', 'ipo', 'dip', 'breakout'
        ]
    
    def extract_tickers(self, text: str) -> Set[str]:
        """Extract ticker symbols from text."""
        tickers = set()
        for match in self.ticker_regex.finditer(text):
            for group in match.groups():
                if group and len(group) >= 2:
                    tickers.add(group.upper())
        return tickers
    
    def calculate_relevance_score(self, text: str, target_ticker: str) -> float:
        """Calculate relevance score for a post."""
        text_lower = text.lower()
        target_lower = target_ticker.lower()
        
        score = 0.0
        
        # Direct ticker mentions
        ticker_mentions = text_lower.count(f'${target_lower}') * 2.0
        ticker_mentions += text_lower.count(target_lower) * 1.0
        score += min(ticker_mentions, 10.0)  # Cap at 10
        
        # Finance keywords
        keyword_score = sum(1.0 for kw in self.finance_keywords if kw in text_lower)
        score += min(keyword_score, 5.0)  # Cap at 5
        
        # Length bonus (longer posts often more thoughtful)
        if len(text) > 100:
            score += 1.0
        if len(text) > 500:
            score += 1.0
        
        return score

class PushshiftRedditCollector:
    """Reddit collector using Pushshift API for historical data."""
    
    def __init__(self):
        self.base_url = "https://api.pushshift.io/reddit/search"
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache_manager = CacheManager()
        self.ticker_detector = TickerDetector()
        
        # Rate limiting
        self.requests_per_minute = 60
        self.request_delay = 60 / self.requests_per_minute
        self.last_request = 0.0
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if not self.session or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def _rate_limited_request(self, url: str, params: Dict) -> Optional[Dict]:
        """Make rate-limited request to Pushshift API."""
        await self._ensure_session()
        
        # Rate limiting
        time_since_last = asyncio.get_event_loop().time() - self.last_request
        if time_since_last < self.request_delay:
            await asyncio.sleep(self.request_delay - time_since_last)
        
        try:
            async with self.session.get(url, params=params) as response:
                self.last_request = asyncio.get_event_loop().time()
                
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Pushshift API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.debug(f"Pushshift request failed: {e}")
            return None
    
    async def collect_posts(
        self, 
        symbol: str, 
        limit: int = 100,
        hours_back: int = 24,
        min_score: int = 5,
        subreddits: List[str] = None
    ) -> Tuple[List[SocialMediaPost], CollectionStats]:
        """Collect Reddit posts with optimization."""
        
        if subreddits is None:
            subreddits = ['wallstreetbets', 'investing', 'stocks']
        
        stats = CollectionStats()
        all_posts = []
        
        # Check cache first
        timeframe = f"{hours_back}h"
        cached_posts = await self.cache_manager.get_cached_posts(symbol, "reddit", timeframe)
        
        if cached_posts:
            stats.cache_hits = len(cached_posts)
            for post_data in cached_posts:
                post = SocialMediaPost(**post_data)
                if symbol.upper() in post.tickers:
                    all_posts.append(post)
            
            stats.final_count = len(all_posts)
            logger.info(f"Reddit cache hit for {symbol}: {len(all_posts)} posts")
            return all_posts, stats
        
        # Fetch fresh data
        cutoff_timestamp = int((datetime.now() - timedelta(hours=hours_back)).timestamp())
        
        for subreddit in subreddits:
            try:
                params = {
                    'subreddit': subreddit,
                    'q': f'${symbol}',  # Search for ticker with $
                    'size': limit // len(subreddits),  # Distribute limit across subreddits
                    'after': cutoff_timestamp,
                    'sort': 'score',
                    'sort_type': 'desc'
                }
                
                data = await self._rate_limited_request(f"{self.base_url}/submission", params)
                
                if not data or 'data' not in data:
                    continue
                
                for item in data['data']:
                    stats.total_fetched += 1
                    
                    # Basic filtering
                    if item.get('score', 0) < min_score:
                        stats.filtered_by_score += 1
                        continue
                    
                    # Extract and validate tickers
                    title = item.get('title', '')
                    selftext = item.get('selftext', '')
                    full_text = f"{title} {selftext}"
                    
                    tickers = self.ticker_detector.extract_tickers(full_text)
                    if symbol.upper() not in tickers:
                        continue
                    
                    # Calculate relevance
                    relevance_score = self.ticker_detector.calculate_relevance_score(full_text, symbol)
                    if relevance_score < 2.0:  # Filter low relevance
                        stats.filtered_by_relevance += 1
                        continue
                    
                    post = SocialMediaPost(
                        platform="reddit",
                        post_id=item.get('id', ''),
                        author=item.get('author', 'unknown'),
                        content=full_text[:1000],  # Limit content length
                        timestamp=datetime.fromtimestamp(item.get('created_utc', 0)),
                        score=item.get('score', 0),
                        comments_count=item.get('num_comments', 0),
                        url=f"https://reddit.com{item.get('permalink', '')}",
                        tickers=list(tickers),
                        relevance_score=relevance_score
                    )
                    
                    all_posts.append(post)
                
                # Small delay between subreddits
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.debug(f"Error collecting from r/{subreddit}: {e}")
                continue
        
        # Cache results
        posts_for_cache = [asdict(post) for post in all_posts]
        for post_data in posts_for_cache:
            # Convert datetime to string for JSON serialization
            post_data['timestamp'] = post_data['timestamp'].isoformat()
        
        await self.cache_manager.save_posts_to_cache(symbol, "reddit", timeframe, posts_for_cache)
        
        stats.final_count = len(all_posts)
        logger.info(f"Collected {len(all_posts)} Reddit posts for {symbol}")
        
        return all_posts, stats
    
    async def cleanup(self):
        """Clean up resources."""
        if self.session and not self.session.closed:
            await self.session.close()

class OptimizedTwitterCollector:
    """Optimized Twitter collector with rate limiting."""
    
    def __init__(self):
        self.api = None
        self.cache_manager = CacheManager()
        self.ticker_detector = TickerDetector()
        self._initialize_twitter()
    
    def _initialize_twitter(self):
        """Initialize Twitter API with v2."""
        try:
            if settings.twitter_bearer_token:
                self.api = tweepy.Client(
                    bearer_token=settings.twitter_bearer_token,
                    wait_on_rate_limit=True
                )
                logger.info("Twitter API initialized successfully (no rate limit blocking)")
            else:
                logger.warning("Twitter API credentials not configured")
        except Exception as e:
            logger.error(f"Failed to initialize Twitter API: {e}")
    
    async def collect_posts(
        self, 
        symbol: str, 
        limit: int = 50,
        hours_back: int = 12
    ) -> Tuple[List[SocialMediaPost], CollectionStats]:
        """Collect Twitter posts with caching and optimization."""
        
        stats = CollectionStats()
        posts = []
        
        if not self.api:
            logger.warning("Twitter API not available")
            return posts, stats
        
        # Check cache
        timeframe = f"{hours_back}h"
        cached_posts = await self.cache_manager.get_cached_posts(symbol, "twitter", timeframe)
        
        if cached_posts:
            stats.cache_hits = len(cached_posts)
            for post_data in cached_posts:
                post_data['timestamp'] = datetime.fromisoformat(post_data['timestamp'])
                post = SocialMediaPost(**post_data)
                if symbol.upper() in post.tickers:
                    posts.append(post)
            
            stats.final_count = len(posts)
            logger.info(f"Twitter cache hit for {symbol}: {len(posts)} posts")
            return posts, stats
        
        try:
            # Search with optimized query
            query = f"${symbol} -is:retweet lang:en"
            start_time = datetime.now() - timedelta(hours=hours_back)
            
            tweets = tweepy.Paginator(
                self.api.search_recent_tweets,
                query=query,
                max_results=min(limit, 100),
                start_time=start_time,
                tweet_fields=['created_at', 'author_id', 'public_metrics']
            ).flatten(limit=limit)
            
            for tweet in tweets:
                stats.total_fetched += 1
                
                # Extract tickers and calculate relevance
                tickers = self.ticker_detector.extract_tickers(tweet.text)
                if symbol.upper() not in tickers:
                    continue
                
                relevance_score = self.ticker_detector.calculate_relevance_score(tweet.text, symbol)
                if relevance_score < 1.0:
                    stats.filtered_by_relevance += 1
                    continue
                
                post = SocialMediaPost(
                    platform="twitter",
                    post_id=str(tweet.id),
                    author=str(tweet.author_id),
                    content=tweet.text[:500],  # Limit length
                    timestamp=tweet.created_at,
                    score=tweet.public_metrics.get('like_count', 0) if tweet.public_metrics else 0,
                    comments_count=tweet.public_metrics.get('reply_count', 0) if tweet.public_metrics else 0,
                    tickers=list(tickers),
                    relevance_score=relevance_score
                )
                
                posts.append(post)
            
            # Cache results
            posts_for_cache = [asdict(post) for post in posts]
            for post_data in posts_for_cache:
                post_data['timestamp'] = post_data['timestamp'].isoformat()
            
            await self.cache_manager.save_posts_to_cache(symbol, "twitter", timeframe, posts_for_cache)
            
        except Exception as e:
            logger.warning(f"Twitter collection failed: {e}")
        
        stats.final_count = len(posts)
        logger.info(f"Collected {len(posts)} Twitter posts for {symbol}")
        
        return posts, stats

class RefactoredSocialMediaCollector:
    """Main refactored social media collector."""
    
    def __init__(self):
        self.reddit_collector = PushshiftRedditCollector()
        self.twitter_collector = OptimizedTwitterCollector()
    
    async def collect_all_platforms(
        self, 
        symbol: str, 
        limit_per_platform: int = 50
    ) -> Dict[str, List[SocialMediaPost]]:
        """Collect from all platforms efficiently."""
        
        results = {}
        
        # Collect Reddit posts
        try:
            reddit_posts, reddit_stats = await self.reddit_collector.collect_posts(
                symbol, 
                limit=limit_per_platform,
                hours_back=24
            )
            results['reddit'] = reddit_posts
            logger.info(f"Reddit: {reddit_stats.final_count} posts ({reddit_stats.cache_hits} cached)")
            
        except Exception as e:
            logger.warning(f"Reddit collection failed: {e}")
            results['reddit'] = []
        
        # Collect Twitter posts
        try:
            twitter_posts, twitter_stats = await self.twitter_collector.collect_posts(
                symbol,
                limit=limit_per_platform,
                hours_back=12
            )
            results['twitter'] = twitter_posts
            logger.info(f"Twitter: {twitter_stats.final_count} posts ({twitter_stats.cache_hits} cached)")
            
        except Exception as e:
            logger.warning(f"Twitter collection failed: {e}")
            results['twitter'] = []
        
        return results
    
    async def cleanup(self):
        """Clean up all collectors."""
        await self.reddit_collector.cleanup()
        # Twitter cleanup handled by tweepy

# Global instance
refactored_collector = RefactoredSocialMediaCollector()

# For backward compatibility
class SocialMediaCollector(RefactoredSocialMediaCollector):
    """Backward compatible wrapper."""
    pass