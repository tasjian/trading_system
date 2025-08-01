"""
Enhanced RapidAPI Twitter Collector

Based on the provided RapidAPI example with anti-throttling strategies:
- Exponential backoff on rate limits
- Rate limit-aware fetchers with delays
- Prioritized ticker scanning
- Cache results to avoid re-querying
- Multiple provider redundancy
- Proxy rotation support (future)
"""

import asyncio
import aiohttp
import logging
import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
import hashlib
from pathlib import Path
import sqlite3

from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class TwitterPost:
    """Twitter post optimized for sentiment analysis."""
    post_id: str
    author: str
    content: str
    timestamp: datetime
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    url: Optional[str] = None
    tickers: List[str] = None
    relevance_score: float = 0.0
    sentiment_indicators: List[str] = None
    engagement_score: float = 0.0
    
    def __post_init__(self):
        if self.tickers is None:
            self.tickers = []
        if self.sentiment_indicators is None:
            self.sentiment_indicators = []
        # Calculate engagement score
        self.engagement_score = (self.likes * 1.0) + (self.retweets * 2.0) + (self.replies * 1.5)

class SQLiteCache:
    """SQLite-based cache for Twitter results to avoid re-querying."""
    
    def __init__(self, cache_dir: str = "data/twitter_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "twitter_cache.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS twitter_posts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    query_hash TEXT NOT NULL,
                    post_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_query ON twitter_posts(symbol, query_hash)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at ON twitter_posts(expires_at)
            """)
    
    def _get_query_hash(self, symbol: str, query: str, limit: int) -> str:
        """Generate hash for query parameters."""
        data = f"{symbol}_{query}_{limit}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def get_cached_posts(self, symbol: str, query: str, limit: int, cache_minutes: int = 15) -> Optional[List[Dict]]:
        """Get cached posts if still valid."""
        query_hash = self._get_query_hash(symbol, query, limit)
        expires_after = datetime.now() - timedelta(minutes=cache_minutes)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT post_data FROM twitter_posts 
                    WHERE symbol = ? AND query_hash = ? AND created_at > ?
                    ORDER BY created_at DESC LIMIT 1
                """, (symbol, query_hash, expires_after))
                
                row = cursor.fetchone()
                if row:
                    logger.debug(f"Cache hit for {symbol} query: {query}")
                    return json.loads(row[0])
        
        except Exception as e:
            logger.debug(f"Cache read error: {e}")
        
        return None
    
    def save_posts(self, symbol: str, query: str, limit: int, posts: List[Dict], cache_minutes: int = 15):
        """Save posts to cache."""
        query_hash = self._get_query_hash(symbol, query, limit)
        expires_at = datetime.now() + timedelta(minutes=cache_minutes)
        post_data = json.dumps(posts, default=str)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO twitter_posts (symbol, query_hash, post_data, expires_at)
                    VALUES (?, ?, ?, ?)
                """, (symbol, query_hash, post_data, expires_at))
                
                # Clean up expired entries
                conn.execute("DELETE FROM twitter_posts WHERE expires_at < ?", (datetime.now(),))
        
        except Exception as e:
            logger.debug(f"Cache save error: {e}")

class RateLimitManager:
    """Advanced rate limiting with exponential backoff."""
    
    def __init__(self):
        self.provider_stats = {}
        self.global_delays = {}
    
    def can_make_request(self, provider: str, requests_per_minute: int = 100) -> bool:
        """Check if we can make a request to provider."""
        now = time.time()
        
        if provider not in self.provider_stats:
            self.provider_stats[provider] = {
                "requests": [],
                "failures": 0,
                "last_failure": 0
            }
        
        stats = self.provider_stats[provider]
        
        # Remove requests older than 1 minute
        stats["requests"] = [req_time for req_time in stats["requests"] if now - req_time < 60]
        
        # Check rate limit
        if len(stats["requests"]) >= requests_per_minute:
            return False
        
        # Check if we're in a backoff period
        if stats["failures"] > 0:
            backoff_time = min(300, 2 ** stats["failures"])  # Max 5 minute backoff
            if now - stats["last_failure"] < backoff_time:
                return False
        
        return True
    
    def record_request(self, provider: str, success: bool = True):
        """Record a request and its outcome."""
        now = time.time()
        
        if provider not in self.provider_stats:
            self.provider_stats[provider] = {
                "requests": [],
                "failures": 0,
                "last_failure": 0
            }
        
        stats = self.provider_stats[provider]
        stats["requests"].append(now)
        
        if success:
            stats["failures"] = max(0, stats["failures"] - 1)  # Reduce failure count on success
        else:
            stats["failures"] += 1
            stats["last_failure"] = now
    
    async def wait_if_needed(self, provider: str):
        """Wait if rate limiting requires it."""
        if provider in self.global_delays:
            delay = self.global_delays[provider]
            if delay > 0:
                await asyncio.sleep(delay)
                self.global_delays[provider] = max(0, delay - 1)

class EnhancedRapidAPITwitterCollector:
    """Enhanced Twitter collector using RapidAPI with anti-throttling strategies."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache = SQLiteCache()
        self.rate_limiter = RateLimitManager()
        
        # Multiple RapidAPI providers for redundancy (based on your example)
        self.providers = [
            {
                "name": "twitter-api45",
                "base_url": "https://twitter-api45.p.rapidapi.com",
                "search_endpoint": "/search.php",
                "host": "twitter-api45.p.rapidapi.com",
                "requests_per_minute": 80,
                "priority": 1  # Primary provider
            },
            {
                "name": "twitter135",
                "base_url": "https://twitter135.p.rapidapi.com", 
                "search_endpoint": "/v2/Search/",
                "host": "twitter135.p.rapidapi.com",
                "requests_per_minute": 60,
                "priority": 2
            },
            # Add more providers as backups
        ]
        
        # Sort providers by priority
        self.providers.sort(key=lambda p: p["priority"])
        
        # Financial sentiment keywords for enhanced analysis
        self.sentiment_keywords = {
            'strong_bullish': ['moon', 'rocket', '🚀', 'mooning', 'breakout', 'squeeze'],
            'bullish': ['bull', 'bullish', 'buy', 'long', 'calls', 'pump', 'up', 'rise'],
            'strong_bearish': ['crash', 'dump', 'tank', 'collapse', 'dead'],
            'bearish': ['bear', 'bearish', 'sell', 'short', 'puts', 'drop', 'fall', 'down'],
            'emotional': ['diamond hands', '💎🙌', 'apes', 'hodl', 'yolo', 'fomo'],
            'analytical': ['support', 'resistance', 'chart', 'technical', 'analysis']
        }
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists with optimized settings."""
        if not self.session or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=15, connect=5)
            connector = aiohttp.TCPConnector(
                limit=10,  # Connection pool limit
                limit_per_host=5,
                keepalive_timeout=30
            )
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
    
    def _get_headers(self, provider: Dict) -> Dict[str, str]:
        """Get headers for RapidAPI request (following your example)."""
        return {
            "X-RapidAPI-Key": settings.rapidapi_key or "your_rapidapi_key_here",
            "X-RapidAPI-Host": provider["host"],
            "User-Agent": "TradingBot/2.0 (Sentiment Analysis)"
        }
    
    async def _make_request_with_backoff(self, provider: Dict, query: str, count: int = 10) -> Optional[Dict]:
        """Make request with exponential backoff (following your example structure)."""
        await self._ensure_session()
        
        # Check rate limiting
        if not self.rate_limiter.can_make_request(provider["name"], provider["requests_per_minute"]):
            logger.debug(f"Rate limit hit for {provider['name']}")
            return None
        
        # Wait if needed
        await self.rate_limiter.wait_if_needed(provider["name"])
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Build request (following your example)
                url = provider["base_url"] + provider["search_endpoint"]
                headers = self._get_headers(provider)
                
                # Query parameters (following your example format)
                querystring = {
                    "query": query,
                    "count": str(min(count, 50))  # Conservative limit
                }
                
                logger.debug(f"Making request to {provider['name']}: {query}")
                
                async with self.session.get(url, headers=headers, params=querystring) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.rate_limiter.record_request(provider["name"], success=True)
                        logger.debug(f"✅ {provider['name']}: {response.status}")
                        return data
                    
                    elif response.status == 429:  # Rate limited
                        self.rate_limiter.record_request(provider["name"], success=False)
                        wait_time = 2 ** attempt
                        logger.debug(f"Rate limited by {provider['name']}, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    else:
                        logger.debug(f"❌ {provider['name']}: {response.status}")
                        self.rate_limiter.record_request(provider["name"], success=False)
                        return None
            
            except Exception as e:
                logger.debug(f"Request error for {provider['name']}: {e}")
                self.rate_limiter.record_request(provider["name"], success=False)
                
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(wait_time)
                continue
        
        return None
    
    def _extract_tickers(self, text: str) -> Set[str]:
        """Extract ticker symbols with enhanced patterns."""
        import re
        
        ticker_patterns = [
            r'\$([A-Z]{1,5})\b',  # $AAPL format (primary)
            r'\b([A-Z]{2,5})\s+(?:stock|shares|calls|puts|options|earnings)\b',
            r'\b(?:bought|sold|holding|long|short)\s+([A-Z]{2,5})\b',
            r'([A-Z]{2,5})\s+(?:moon|rocket|pump|dump)\b'
        ]
        
        tickers = set()
        for pattern in ticker_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                for group in match.groups():
                    if group and 2 <= len(group) <= 5:
                        ticker = group.upper()
                        # Enhanced false positive filtering
                        if ticker not in {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 
                                        'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 
                                        'USE', 'HIM', 'OLD', 'SEE', 'NOW', 'WAY', 'WHO', 'ITS'}:
                            tickers.add(ticker)
        
        return tickers
    
    def _calculate_relevance_score(self, text: str, target_ticker: str, engagement: Dict) -> float:
        """Enhanced relevance scoring for trading signals."""
        text_lower = text.lower()
        target_lower = target_ticker.lower()
        score = 0.0
        
        # Ticker mention scoring (highest weight)
        if f'${target_lower}' in text_lower:
            score += 10.0  # Strong indicator
        elif target_lower in text_lower:
            score += 5.0
        
        # Sentiment analysis scoring
        for sentiment_type, keywords in self.sentiment_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                if sentiment_type == 'strong_bullish':
                    score += matches * 3.0
                elif sentiment_type == 'strong_bearish':
                    score += matches * 3.0
                elif sentiment_type == 'emotional':
                    score += matches * 2.5  # Emotional content often drives price
                elif sentiment_type == 'analytical':
                    score += matches * 2.0  # Technical analysis valuable
                else:
                    score += matches * 1.5
        
        # Engagement boost (viral content = market impact)
        likes = engagement.get('likes', 0)
        retweets = engagement.get('retweets', 0)
        replies = engagement.get('replies', 0)
        
        # Logarithmic scaling for engagement
        if likes > 1000:
            score += 5.0
        elif likes > 500:
            score += 3.0
        elif likes > 100:
            score += 2.0
        elif likes > 20:
            score += 1.0
        
        if retweets > 500:
            score += 4.0
        elif retweets > 100:
            score += 2.5
        elif retweets > 20:
            score += 1.5
        elif retweets > 5:
            score += 1.0
        
        # Quality indicators
        if len(text) > 120:  # Longer tweets often more informative
            score += 1.5
        if any(char in text for char in ['#', '@']):  # Social engagement
            score += 1.0
        if any(word in text_lower for word in ['chart', 'analysis', 'target', 'pt']):
            score += 2.0  # Technical analysis
        
        return score
    
    def _extract_sentiment_indicators(self, text: str) -> List[str]:
        """Extract detailed sentiment indicators for analysis."""
        text_lower = text.lower()
        indicators = []
        
        for sentiment_type, keywords in self.sentiment_keywords.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            for kw in found_keywords:
                indicators.append(f"{sentiment_type}:{kw}")
        
        return indicators[:10]  # Limit to top 10 indicators
    
    def _parse_twitter_response(self, data: Dict, provider_name: str) -> List[Dict]:
        """Parse Twitter response with robust error handling."""
        tweets = []
        
        try:
            # Handle different response formats
            if provider_name == "twitter-api45":
                # Following your example structure
                tweet_list = data.get("timeline", data.get("results", []))
            else:
                tweet_list = data.get("data", [])
            
            if not isinstance(tweet_list, list):
                return []
            
            for tweet_data in tweet_list:
                try:
                    normalized = self._normalize_tweet(tweet_data, provider_name)
                    if normalized and normalized.get("text"):
                        tweets.append(normalized)
                except Exception as e:
                    logger.debug(f"Error normalizing tweet: {e}")
                    continue
            
            return tweets
            
        except Exception as e:
            logger.debug(f"Error parsing {provider_name} response: {e}")
            return []
    
    def _normalize_tweet(self, tweet_data: Dict, provider_name: str) -> Optional[Dict]:
        """Normalize tweet data across providers."""
        try:
            # Extract text
            text = tweet_data.get("text", tweet_data.get("full_text", ""))
            if not text:
                return None
            
            # Extract ID
            tweet_id = str(tweet_data.get("id", tweet_data.get("id_str", f"unknown_{int(time.time())}")))
            
            # Parse timestamp
            created_at = tweet_data.get("created_at", "")
            try:
                if "T" in created_at:
                    timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                else:
                    timestamp = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
            except:
                timestamp = datetime.now()
            
            # Extract engagement metrics
            public_metrics = tweet_data.get("public_metrics", {})
            likes = public_metrics.get("like_count", tweet_data.get("favorite_count", 0))
            retweets = public_metrics.get("retweet_count", tweet_data.get("retweet_count", 0))
            replies = public_metrics.get("reply_count", tweet_data.get("reply_count", 0))
            
            # Extract author
            author_data = tweet_data.get("author", tweet_data.get("user", {}))
            if isinstance(author_data, dict):
                author = author_data.get("username", author_data.get("screen_name", "unknown"))
            else:
                author = str(author_data) if author_data else "unknown"
            
            return {
                "id": tweet_id,
                "text": text,
                "timestamp": timestamp,
                "author": author,
                "likes": int(likes) if likes else 0,
                "retweets": int(retweets) if retweets else 0,
                "replies": int(replies) if replies else 0,
                "provider": provider_name
            }
            
        except Exception as e:
            logger.debug(f"Tweet normalization error: {e}")
            return None
    
    async def collect_posts_for_symbol(self, symbol: str, limit: int = 30, prioritize: bool = True) -> List[TwitterPost]:
        """Collect Twitter posts with prioritization and caching."""
        posts = []
        
        # Prioritized queries for better results (following your tip to prioritize tickers)
        if prioritize:
            queries = [
                f"${symbol}",  # Primary ticker format
                f"${symbol} stock",
                f"{symbol} earnings",
                f"${symbol} calls puts"
            ]
        else:
            queries = [f"${symbol}"]
        
        for query in queries:
            if len(posts) >= limit:
                break
            
            # Check cache first (following your caching tip)
            cached_posts = self.cache.get_cached_posts(symbol, query, limit // len(queries))
            if cached_posts:
                logger.debug(f"Using cached results for {symbol} query: {query}")
                posts.extend(self._process_cached_posts(cached_posts, symbol))
                continue
            
            # Try providers in priority order
            query_posts = []
            for provider in self.providers:
                if len(query_posts) >= (limit // len(queries)):
                    break
                
                try:
                    data = await self._make_request_with_backoff(
                        provider, query, min(20, limit // len(queries))
                    )
                    
                    if data:
                        tweet_list = self._parse_twitter_response(data, provider["name"])
                        for tweet in tweet_list:
                            processed_post = self._process_tweet(tweet, symbol)
                            if processed_post:
                                query_posts.append(asdict(processed_post))
                        
                        # Cache successful results
                        if query_posts:
                            self.cache.save_posts(symbol, query, limit // len(queries), query_posts)
                        
                        break  # Success with this provider
                        
                except Exception as e:
                    logger.debug(f"Error with provider {provider['name']}: {e}")
                    continue
            
            posts.extend([TwitterPost(**post) for post in query_posts])
            
            # Respectful delay between queries (following your rate limiting tip)
            await asyncio.sleep(random.uniform(0.5, 1.0))
        
        # Sort by relevance and engagement
        posts.sort(key=lambda p: (p.relevance_score, p.engagement_score), reverse=True)
        return posts[:limit]
    
    def _process_cached_posts(self, cached_posts: List[Dict], symbol: str) -> List[TwitterPost]:
        """Process cached posts into TwitterPost objects."""
        posts = []
        for post_data in cached_posts:
            try:
                # Handle datetime conversion
                if isinstance(post_data.get('timestamp'), str):
                    post_data['timestamp'] = datetime.fromisoformat(post_data['timestamp'])
                posts.append(TwitterPost(**post_data))
            except Exception as e:
                logger.debug(f"Error processing cached post: {e}")
        return posts
    
    def _process_tweet(self, tweet: Dict, target_symbol: str) -> Optional[TwitterPost]:
        """Process raw tweet into TwitterPost with relevance scoring."""
        try:
            # Extract tickers
            tickers = self._extract_tickers(tweet["text"])
            if target_symbol.upper() not in tickers:
                return None
            
            # Calculate engagement and relevance
            engagement = {
                'likes': tweet.get('likes', 0),
                'retweets': tweet.get('retweets', 0),
                'replies': tweet.get('replies', 0)
            }
            
            relevance_score = self._calculate_relevance_score(tweet["text"], target_symbol, engagement)
            
            # Filter low relevance posts
            if relevance_score < 2.0:
                return None
            
            # Extract sentiment indicators
            sentiment_indicators = self._extract_sentiment_indicators(tweet["text"])
            
            return TwitterPost(
                post_id=tweet["id"],
                author=tweet["author"],
                content=tweet["text"][:500],  # Limit content length
                timestamp=tweet["timestamp"],
                likes=engagement['likes'],
                retweets=engagement['retweets'],
                replies=engagement['replies'],
                tickers=list(tickers),
                relevance_score=relevance_score,
                sentiment_indicators=sentiment_indicators
            )
            
        except Exception as e:
            logger.debug(f"Error processing tweet: {e}")
            return None
    
    async def collect_trending_symbols(self, symbols: List[str], limit_per_symbol: int = 20) -> Dict[str, List[TwitterPost]]:
        """Collect posts for trending symbols efficiently."""
        results = {}
        
        # Process in batches to manage rate limits
        batch_size = 3
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            # Collect batch concurrently
            tasks = [
                self.collect_posts_for_symbol(symbol, limit_per_symbol, prioritize=True)
                for symbol in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for j, result in enumerate(batch_results):
                symbol = batch[j]
                if isinstance(result, Exception):
                    logger.warning(f"Failed to collect for {symbol}: {result}")
                    results[symbol] = []
                else:
                    results[symbol] = result
                    logger.info(f"📊 {symbol}: {len(result)} posts (avg relevance: {sum(p.relevance_score for p in result)/len(result) if result else 0:.2f})")
            
            # Batch delay
            if i + batch_size < len(symbols):
                await asyncio.sleep(random.uniform(2.0, 4.0))
        
        return results
    
    async def cleanup(self):
        """Clean up resources."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.debug("Enhanced RapidAPI Twitter session closed")

# Global instance
enhanced_rapidapi_twitter = EnhancedRapidAPITwitterCollector()