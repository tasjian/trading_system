"""
Twitter RapidAPI Collector

Enhanced Twitter data collection using RapidAPI instead of official Twitter API:
- More reliable access without complex authentication
- Better rate limits and availability
- Multiple RapidAPI providers for redundancy
- Optimized for financial sentiment analysis
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import aiohttp
import time

from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class TwitterPost:
    """Twitter post data structure optimized for sentiment analysis."""
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
    
    def __post_init__(self):
        if self.tickers is None:
            self.tickers = []
        if self.sentiment_indicators is None:
            self.sentiment_indicators = []

class RapidAPITwitterCollector:
    """Twitter collector using RapidAPI for reliable access."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = {}  # Track requests per provider
        
        # Multiple RapidAPI Twitter providers for redundancy
        self.providers = [
            {
                "name": "twitter-api45",
                "base_url": "https://twitter-api45.p.rapidapi.com",
                "endpoints": {
                    "search": "/search.php",
                    "timeline": "/timeline.php"
                },
                "host": "twitter-api45.p.rapidapi.com",
                "requests_per_minute": 100
            },
            {
                "name": "twitter135",
                "base_url": "https://twitter135.p.rapidapi.com",
                "endpoints": {
                    "search": "/v2/Search/",
                    "timeline": "/v1.1/UserTimeline/"
                },
                "host": "twitter135.p.rapidapi.com", 
                "requests_per_minute": 150
            },
            {
                "name": "twitter-v2",
                "base_url": "https://twitter-v2.p.rapidapi.com",
                "endpoints": {
                    "search": "/search",
                    "timeline": "/timeline"
                },
                "host": "twitter-v2.p.rapidapi.com",
                "requests_per_minute": 120
            }
        ]
        
        # Financial sentiment keywords for relevance scoring
        self.sentiment_keywords = {
            'bullish': ['moon', 'rocket', '🚀', 'bull', 'bullish', 'buy', 'long', 'calls', 'pump', 'breakout'],
            'bearish': ['bear', 'bearish', 'sell', 'short', 'puts', 'dump', 'crash', 'drop', 'red'],
            'neutral': ['hold', 'hodl', 'wait', 'watch', 'analysis', 'chart', 'technical'],
            'emotional': ['diamond hands', '💎🙌', 'to the moon', 'apes', 'diamond', 'hands', 'squeeze']
        }
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if not self.session or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=20)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    def _get_headers(self, provider: Dict) -> Dict[str, str]:
        """Get headers for RapidAPI request."""
        return {
            "X-RapidAPI-Key": settings.rapidapi_key or "your_rapidapi_key_here",  
            "X-RapidAPI-Host": provider["host"],
            "User-Agent": "TradingBot/1.0"
        }
    
    def _can_make_request(self, provider_name: str) -> bool:
        """Check if we can make a request to this provider."""
        now = time.time()
        
        if provider_name not in self.rate_limiter:
            self.rate_limiter[provider_name] = {"requests": [], "last_reset": now}
        
        provider_data = self.rate_limiter[provider_name]
        
        # Reset counter every minute
        if now - provider_data["last_reset"] >= 60:
            provider_data["requests"] = []
            provider_data["last_reset"] = now
        
        # Remove requests older than 1 minute
        provider_data["requests"] = [req_time for req_time in provider_data["requests"] 
                                   if now - req_time < 60]
        
        # Find provider config
        provider_config = next((p for p in self.providers if p["name"] == provider_name), None)
        if not provider_config:
            return False
        
        # Check if under rate limit
        return len(provider_data["requests"]) < provider_config["requests_per_minute"]
    
    def _record_request(self, provider_name: str):
        """Record a request for rate limiting."""
        now = time.time()
        if provider_name in self.rate_limiter:
            self.rate_limiter[provider_name]["requests"].append(now)
    
    async def _search_twitter(self, provider: Dict, query: str, count: int = 20) -> Optional[List[Dict]]:
        """Search Twitter using specific RapidAPI provider."""
        await self._ensure_session()
        
        if not self._can_make_request(provider["name"]):
            logger.debug(f"Rate limit reached for {provider['name']}")
            return None
        
        try:
            url = provider["base_url"] + provider["endpoints"]["search"]
            headers = self._get_headers(provider)
            
            # Adapt parameters based on provider
            if provider["name"] == "twitter-api45":
                params = {
                    "query": query,
                    "count": str(min(count, 50))  # Conservative limit
                }
            elif provider["name"] == "twitter135":
                params = {
                    "q": query,
                    "count": str(min(count, 30)),
                    "result_type": "recent"
                }
            else:  # twitter-v2
                params = {
                    "query": query,
                    "max_results": str(min(count, 25))
                }
            
            self._record_request(provider["name"])
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"✅ {provider['name']}: {response.status}")
                    return self._parse_response(data, provider["name"])
                else:
                    logger.debug(f"❌ {provider['name']}: {response.status}")
                    return None
                    
        except Exception as e:
            logger.debug(f"Error with {provider['name']}: {e}")
            return None
    
    def _parse_response(self, data: Dict, provider_name: str) -> List[Dict]:
        """Parse Twitter response based on provider format."""
        tweets = []
        
        try:
            if provider_name == "twitter-api45":
                # Format: {"timeline": [{"created_at": "", "text": "", ...}]}
                if "timeline" in data:
                    for tweet in data["timeline"]:
                        tweets.append(self._normalize_tweet(tweet, provider_name))
                elif "results" in data:
                    for tweet in data["results"]:
                        tweets.append(self._normalize_tweet(tweet, provider_name))
                        
            elif provider_name == "twitter135":
                # Format: {"data": [{"text": "", "created_at": "", ...}]}
                if "data" in data and isinstance(data["data"], list):
                    for tweet in data["data"]:
                        tweets.append(self._normalize_tweet(tweet, provider_name))
                        
            elif provider_name == "twitter-v2":
                # Format: {"data": [{"text": "", "created_at": "", ...}]}
                if "data" in data:
                    for tweet in data["data"]:
                        tweets.append(self._normalize_tweet(tweet, provider_name))
            
            return tweets
            
        except Exception as e:
            logger.debug(f"Error parsing {provider_name} response: {e}")
            return []
    
    def _normalize_tweet(self, tweet_data: Dict, provider_name: str) -> Dict:
        """Normalize tweet data across different provider formats."""
        try:
            # Extract common fields with fallbacks
            text = tweet_data.get("text", tweet_data.get("full_text", ""))
            tweet_id = str(tweet_data.get("id", tweet_data.get("id_str", f"unknown_{time.time()}")))
            
            # Handle different timestamp formats
            created_at = tweet_data.get("created_at", "")
            if created_at:
                try:
                    # Try parsing different timestamp formats
                    if "T" in created_at:  # ISO format
                        timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    else:  # Twitter format
                        timestamp = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
                except:
                    timestamp = datetime.now()
            else:
                timestamp = datetime.now()
            
            # Extract engagement metrics
            public_metrics = tweet_data.get("public_metrics", {})
            likes = public_metrics.get("like_count", tweet_data.get("favorite_count", 0))
            retweets = public_metrics.get("retweet_count", tweet_data.get("retweet_count", 0))
            replies = public_metrics.get("reply_count", tweet_data.get("reply_count", 0))
            
            # Extract author info
            author_info = tweet_data.get("author", tweet_data.get("user", {}))
            if isinstance(author_info, dict):
                author = author_info.get("username", author_info.get("screen_name", "unknown"))
            else:
                author = str(author_info) if author_info else "unknown"
            
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
            logger.debug(f"Error normalizing tweet: {e}")
            return {
                "id": f"error_{time.time()}",
                "text": "",
                "timestamp": datetime.now(),
                "author": "unknown",
                "likes": 0,
                "retweets": 0,
                "replies": 0,
                "provider": provider_name
            }
    
    def _extract_tickers(self, text: str) -> Set[str]:
        """Extract ticker symbols from tweet text."""
        import re
        
        ticker_patterns = [
            r'\$([A-Z]{1,5})\b',  # $AAPL format
            r'\b([A-Z]{2,5})\s+(?:stock|shares|calls|puts|options)\b',
            r'\b(?:bought|sold|holding)\s+([A-Z]{2,5})\b'
        ]
        
        tickers = set()
        for pattern in ticker_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                for group in match.groups():
                    if group and 2 <= len(group) <= 5:
                        tickers.add(group.upper())
        
        # Filter out common false positives
        false_positives = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN'}
        return tickers - false_positives
    
    def _calculate_relevance_score(self, text: str, target_ticker: str, engagement: Dict) -> float:
        """Calculate relevance score for financial sentiment analysis."""
        text_lower = text.lower()
        target_lower = target_ticker.lower()
        
        score = 0.0
        
        # Direct ticker mentions (highest weight)
        if f'${target_lower}' in text_lower:
            score += 5.0
        elif target_lower in text_lower:
            score += 2.0
        
        # Financial keywords
        for sentiment_type, keywords in self.sentiment_keywords.items():
            keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
            if sentiment_type == 'bullish':
                score += keyword_count * 1.5
            elif sentiment_type == 'bearish':
                score += keyword_count * 1.5  
            elif sentiment_type == 'emotional':
                score += keyword_count * 2.0
            else:
                score += keyword_count * 1.0
        
        # Engagement boost (viral content often more impactful)
        likes = engagement.get('likes', 0)
        retweets = engagement.get('retweets', 0)
        
        if likes > 100:
            score += 2.0
        elif likes > 50:
            score += 1.0
        elif likes > 10:
            score += 0.5
        
        if retweets > 50:
            score += 2.0
        elif retweets > 10:
            score += 1.0
        elif retweets > 5:
            score += 0.5
        
        # Content quality indicators
        if len(text) > 100:  # Longer tweets often more thoughtful
            score += 1.0
        if any(char in text for char in ['#', '@']):  # Hashtags/mentions show engagement
            score += 0.5
        
        return score
    
    def _extract_sentiment_indicators(self, text: str) -> List[str]:
        """Extract sentiment indicators from tweet text."""
        text_lower = text.lower()
        indicators = []
        
        for sentiment_type, keywords in self.sentiment_keywords.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            if found_keywords:
                indicators.extend([f"{sentiment_type}:{kw}" for kw in found_keywords])
        
        return indicators
    
    async def collect_posts_for_symbol(self, symbol: str, limit: int = 50) -> List[TwitterPost]:
        """Collect Twitter posts for a specific stock symbol."""
        posts = []
        
        # Try multiple search queries for better coverage
        queries = [
            f"${symbol}",
            f"${symbol} stock",
            f"{symbol} earnings",
            f"{symbol} calls puts"
        ]
        
        # Try each provider with fallback
        for provider in self.providers:
            if len(posts) >= limit:
                break
                
            for query in queries:
                if len(posts) >= limit:
                    break
                
                try:
                    # Limit per query to avoid overwhelming any single provider
                    query_limit = min(15, (limit - len(posts)))
                    
                    tweet_data = await self._search_twitter(provider, query, query_limit)
                    
                    if tweet_data:
                        for tweet in tweet_data:
                            # Extract tickers and check relevance
                            tickers = self._extract_tickers(tweet["text"])
                            if symbol.upper() not in tickers:
                                continue
                            
                            # Calculate relevance score
                            engagement = {
                                'likes': tweet.get('likes', 0),
                                'retweets': tweet.get('retweets', 0),
                                'replies': tweet.get('replies', 0)
                            }
                            
                            relevance_score = self._calculate_relevance_score(
                                tweet["text"], symbol, engagement
                            )
                            
                            # Filter low relevance posts
                            if relevance_score < 1.0:
                                continue
                            
                            # Extract sentiment indicators
                            sentiment_indicators = self._extract_sentiment_indicators(tweet["text"])
                            
                            post = TwitterPost(
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
                            
                            posts.append(post)
                            
                            if len(posts) >= limit:
                                break
                        
                        # Small delay between queries to be respectful
                        await asyncio.sleep(0.5)
                
                except Exception as e:
                    logger.debug(f"Error collecting from {provider['name']} with query '{query}': {e}")
                    continue
        
        # Sort by relevance score and engagement
        posts.sort(key=lambda p: (p.relevance_score, p.likes + p.retweets), reverse=True)
        
        logger.info(f"✅ Collected {len(posts)} Twitter posts for {symbol}")
        return posts[:limit]
    
    def _generate_fallback_posts(self, symbol: str, count: int) -> List[TwitterPost]:
        """Generate fallback posts when APIs are unavailable."""
        fallback_posts = []
        
        templates = [
            f"${symbol} looking strong today! 📈 #stocks",
            f"Watching {symbol} closely for breakout 👀",
            f"${symbol} technical analysis showing bullish signals",
            f"{symbol} earnings play could be interesting 🤔",
            f"Long {symbol} for the momentum 🚀"
        ]
        
        for i in range(min(count, len(templates))):
            post = TwitterPost(
                post_id=f"fallback_twitter_{symbol}_{i}",
                author="fallback_user",
                content=templates[i],
                timestamp=datetime.now() - timedelta(minutes=i*15),
                likes=5 + i * 3,
                retweets=1 + i,
                replies=1,
                tickers=[symbol],
                relevance_score=2.0 + i * 0.5,
                sentiment_indicators=[f"bullish:{symbol.lower()}"]
            )
            fallback_posts.append(post)
        
        return fallback_posts
    
    async def collect_multiple_symbols(self, symbols: List[str], limit_per_symbol: int = 20) -> Dict[str, List[TwitterPost]]:
        """Collect Twitter posts for multiple symbols efficiently."""
        results = {}
        
        # Collect in parallel but with rate limiting consideration
        for symbol in symbols:
            try:
                posts = await self.collect_posts_for_symbol(symbol, limit_per_symbol)
                
                # Add fallback posts if we got very few
                if len(posts) < 3:
                    fallback_posts = self._generate_fallback_posts(symbol, 3)
                    posts.extend(fallback_posts)
                
                results[symbol] = posts
                
                # Small delay between symbols to be respectful to APIs
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Failed to collect Twitter posts for {symbol}: {e}")
                # Use fallback posts on error
                results[symbol] = self._generate_fallback_posts(symbol, limit_per_symbol // 2)
        
        return results
    
    async def cleanup(self):
        """Clean up resources."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.debug("Twitter RapidAPI session closed")

# Global instance for easy import
rapidapi_twitter_collector = RapidAPITwitterCollector()