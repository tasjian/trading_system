"""
Social Media Data Collector

Comprehensive social media data collection from multiple platforms:
- Reddit (r/wallstreetbets, r/investing, r/stocks, etc.)
- Twitter/X (financial hashtags and mentions)  
- TikTok (finance-related content)

Integrates with LLM sentiment analyzer for sophisticated sentiment analysis.
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

import asyncpraw
import tweepy
from playwright.async_api import async_playwright
import aiohttp

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SocialMediaPost:
    """Social media post data structure."""
    platform: str
    post_id: str
    author: str
    content: str
    timestamp: datetime
    score: Optional[int] = None  # upvotes, likes, etc.
    comments_count: Optional[int] = None
    url: Optional[str] = None
    hashtags: List[str] = None
    mentions: List[str] = None
    
    def __post_init__(self):
        if self.hashtags is None:
            self.hashtags = []
        if self.mentions is None:
            self.mentions = []


class RedditCollector:
    """Reddit data collection using PRAW."""
    
    def __init__(self):
        self.reddit = None
        self._initialize_reddit()
        
        # Financial subreddits to monitor
        self.subreddits = [
            'wallstreetbets', 'investing', 'stocks', 'SecurityAnalysis',
            'ValueInvesting', 'financialindependence', 'StockMarket',
            'options', 'pennystocks', 'dividends'
        ]
        
    def _initialize_reddit(self):
        """Initialize Reddit API connection."""
        try:
            if settings.reddit_client_id and settings.reddit_client_secret:
                self.reddit = asyncpraw.Reddit(
                    client_id=settings.reddit_client_id,
                    client_secret=settings.reddit_client_secret,
                    user_agent=settings.reddit_user_agent,
                    read_only=True
                )
                logger.info("Reddit API initialized successfully")
            else:
                logger.warning("Reddit API credentials not configured")
        except Exception as e:
            logger.error(f"Failed to initialize Reddit API: {e}")
            self.reddit = None
    
    async def collect_posts(self, symbol: str, limit: int = 50, hours_back: int = 24) -> List[SocialMediaPost]:
        """Collect Reddit posts mentioning a stock symbol."""
        if not self.reddit:
            logger.warning("Reddit API not available")
            return []
        
        posts = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        try:
            # Search across multiple subreddits
            for subreddit_name in self.subreddits:
                try:
                    subreddit = await self.reddit.subreddit(subreddit_name)
                    
                    # Search for symbol mentions
                    search_results = subreddit.search(
                        f"${symbol} OR {symbol}",
                        sort='new',
                        time_filter='day',
                        limit=limit // len(self.subreddits)
                    )
                    
                    async for submission in search_results:
                        post_time = datetime.fromtimestamp(submission.created_utc)
                        
                        if post_time < cutoff_time:
                            continue
                        
                        # Extract content
                        content = f"{submission.title}"
                        if submission.selftext:
                            content += f" {submission.selftext}"
                        
                        # Filter out posts that don't actually mention the symbol
                        if not self._mentions_symbol(content, symbol):
                            continue
                        
                        post = SocialMediaPost(
                            platform="reddit",
                            post_id=submission.id,
                            author=str(submission.author) if submission.author else "deleted",
                            content=content,
                            timestamp=post_time,
                            score=submission.score,
                            comments_count=submission.num_comments,
                            url=f"https://reddit.com{submission.permalink}",
                            hashtags=self._extract_hashtags(content),
                            mentions=self._extract_mentions(content)
                        )
                        
                        posts.append(post)
                        
                except Exception as e:
                    logger.warning(f"Error collecting from r/{subreddit_name}: {e}")
                    continue
            
            # Sort by timestamp descending
            posts.sort(key=lambda x: x.timestamp, reverse=True)
            return posts[:limit]
            
        except Exception as e:
            logger.error(f"Error collecting Reddit posts for {symbol}: {e}")
            return []
    
    def _mentions_symbol(self, text: str, symbol: str) -> bool:
        """Check if text actually mentions the stock symbol."""
        text_upper = text.upper()
        symbol_upper = symbol.upper()
        
        # Look for $SYMBOL or SYMBOL in text
        patterns = [
            f"${symbol_upper}",
            f" {symbol_upper} ",
            f" {symbol_upper}.",
            f" {symbol_upper},",
            f"({symbol_upper})",
            f"#{symbol_upper}"
        ]
        
        return any(pattern in text_upper for pattern in patterns)
    
    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text."""
        return re.findall(r'#(\w+)', text)
    
    def _extract_mentions(self, text: str) -> List[str]:
        """Extract stock mentions from text."""
        return re.findall(r'\$([A-Z]{1,5})', text)
    
    async def close(self):
        """Close Reddit connection."""
        if self.reddit:
            await self.reddit.close()


class TwitterCollector:
    """Twitter/X data collection using Tweepy."""
    
    def __init__(self):
        self.client = None
        self._initialize_twitter()
    
    def _initialize_twitter(self):
        """Initialize Twitter API connection."""
        try:
            if settings.twitter_bearer_token:
                self.client = tweepy.Client(
                    bearer_token=settings.twitter_bearer_token,
                    wait_on_rate_limit=True
                )
                logger.info("Twitter API initialized successfully")
            elif settings.twitter_api_key and settings.twitter_api_secret:
                # Use API v1.1 if bearer token not available
                auth = tweepy.OAuth2BearerHandler(settings.twitter_bearer_token)
                self.client = tweepy.Client(auth=auth, wait_on_rate_limit=True)
                logger.info("Twitter API initialized with OAuth")
            else:
                logger.warning("Twitter API credentials not configured")
        except Exception as e:
            logger.error(f"Failed to initialize Twitter API: {e}")
            self.client = None
    
    async def collect_posts(self, symbol: str, limit: int = 50, hours_back: int = 24) -> List[SocialMediaPost]:
        """Collect tweets mentioning a stock symbol."""
        if not self.client:
            logger.warning("Twitter API not available")
            return []
        
        posts = []
        
        try:
            # Search query for the symbol (updated for Twitter API v2)
            query = f"({symbol} OR #{symbol}) -is:retweet lang:en"
            end_time = datetime.now() - timedelta(hours=hours_back)
            
            tweets = tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                max_results=min(limit, 100),
                end_time=end_time,
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'entities']
            ).flatten(limit=limit)
            
            for tweet in tweets:
                # Extract hashtags and mentions
                hashtags = []
                mentions = []
                
                if hasattr(tweet, 'entities') and tweet.entities:
                    if 'hashtags' in tweet.entities:
                        hashtags = [tag['tag'] for tag in tweet.entities['hashtags']]
                    if 'mentions' in tweet.entities:
                        mentions = [mention['username'] for mention in tweet.entities['mentions']]
                
                post = SocialMediaPost(
                    platform="twitter",
                    post_id=tweet.id,
                    author=tweet.author_id,
                    content=tweet.text,
                    timestamp=tweet.created_at,
                    score=tweet.public_metrics['like_count'] if hasattr(tweet, 'public_metrics') else 0,
                    comments_count=tweet.public_metrics['reply_count'] if hasattr(tweet, 'public_metrics') else 0,
                    url=f"https://twitter.com/user/status/{tweet.id}",
                    hashtags=hashtags,
                    mentions=mentions
                )
                
                posts.append(post)
            
            return sorted(posts, key=lambda x: x.timestamp, reverse=True)
            
        except Exception as e:
            logger.error(f"Error collecting Twitter posts for {symbol}: {e}")
            return []


class TikTokCollector:
    """TikTok data collection using web scraping."""
    
    def __init__(self):
        self.enabled = settings.tiktok_enabled
        self.max_posts = settings.tiktok_max_posts
        self.hashtags = settings.tiktok_hashtags.split(',')
        self.timeout = settings.tiktok_timeout
    
    async def collect_posts(self, symbol: str, limit: int = 20, hours_back: int = 24) -> List[SocialMediaPost]:
        """Collect TikTok posts mentioning a stock symbol."""
        if not self.enabled:
            logger.info("TikTok collection disabled")
            return []
        
        posts = []
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                )
                page = await context.new_page()
                
                # Search for hashtags related to the symbol
                search_queries = [
                    f"#{symbol}stock",
                    f"#{symbol}",
                    f"{symbol}stock",
                    "stockanalysis",
                    "investing"
                ]
                
                for query in search_queries[:3]:  # Limit searches to avoid rate limiting
                    try:
                        search_url = f"https://www.tiktok.com/search?q={query}"
                        await page.goto(search_url, timeout=self.timeout)
                        
                        # Wait for content to load
                        await page.wait_for_timeout(3000)
                        
                        # Extract video information
                        videos = await page.query_selector_all('[data-e2e="search_top-item"]')
                        
                        for video in videos[:limit // len(search_queries)]:
                            try:
                                # Extract video metadata
                                desc_element = await video.query_selector('[data-e2e="search-card-desc"]')
                                author_element = await video.query_selector('[data-e2e="search-card-user-unique-id"]')
                                
                                if desc_element and author_element:
                                    description = await desc_element.inner_text()
                                    author = await author_element.inner_text()
                                    
                                    # Filter for symbol mentions
                                    if symbol.upper() in description.upper():
                                        post = SocialMediaPost(
                                            platform="tiktok",
                                            post_id=f"tiktok_{len(posts)}",  # TikTok doesn't expose IDs easily
                                            author=author.replace('@', ''),
                                            content=description,
                                            timestamp=datetime.now(),  # TikTok timestamps are hard to extract
                                            hashtags=re.findall(r'#(\w+)', description),
                                            mentions=re.findall(r'@(\w+)', description)
                                        )
                                        
                                        posts.append(post)
                                        
                            except Exception as e:
                                logger.debug(f"Error processing TikTok video: {e}")
                                continue
                        
                        # Rate limiting
                        await page.wait_for_timeout(2000)
                        
                    except Exception as e:
                        logger.warning(f"Error searching TikTok for {query}: {e}")
                        continue
                
                await browser.close()
                
        except Exception as e:
            logger.error(f"Error collecting TikTok posts for {symbol}: {e}")
            return []
        
        return posts[:limit]


class SocialMediaCollector:
    """Main social media data collector orchestrator."""
    
    def __init__(self):
        self.reddit_collector = RedditCollector()
        self.twitter_collector = TwitterCollector()
        self.tiktok_collector = TikTokCollector()
        
        # Cache to avoid duplicate collections
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
    
    async def collect_all_platforms(self, symbol: str, limit_per_platform: int = 50) -> Dict[str, List[SocialMediaPost]]:
        """Collect posts from all available platforms."""
        cache_key = f"social_{symbol}"
        
        # Check cache
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_duration:
                return data
        
        results = {}
        
        # Collect from all platforms concurrently
        tasks = [
            self._collect_reddit(symbol, limit_per_platform),
            self._collect_twitter(symbol, limit_per_platform),
            self._collect_tiktok(symbol, min(limit_per_platform, 20))  # TikTok is slower
        ]
        
        platform_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        platform_names = ['reddit', 'twitter', 'tiktok']
        for i, result in enumerate(platform_results):
            platform_name = platform_names[i]
            if isinstance(result, Exception):
                logger.error(f"Error collecting from {platform_name}: {result}")
                results[platform_name] = []
            else:
                results[platform_name] = result or []
                logger.info(f"Collected {len(results[platform_name])} posts from {platform_name}")
        
        # Cache results
        self.cache[cache_key] = (results, datetime.now())
        
        return results
    
    async def _collect_reddit(self, symbol: str, limit: int) -> List[SocialMediaPost]:
        """Collect Reddit posts with error handling."""
        try:
            return await self.reddit_collector.collect_posts(symbol, limit)
        except Exception as e:
            logger.error(f"Reddit collection error: {e}")
            return []
    
    async def _collect_twitter(self, symbol: str, limit: int) -> List[SocialMediaPost]:
        """Collect Twitter posts with error handling."""
        try:
            return await self.twitter_collector.collect_posts(symbol, limit)
        except Exception as e:
            logger.error(f"Twitter collection error: {e}")
            return []
    
    async def _collect_tiktok(self, symbol: str, limit: int) -> List[SocialMediaPost]:
        """Collect TikTok posts with error handling."""
        try:
            return await self.tiktok_collector.collect_posts(symbol, limit)
        except Exception as e:
            logger.error(f"TikTok collection error: {e}")
            return []
    
    def get_all_posts(self, platform_results: Dict[str, List[SocialMediaPost]]) -> List[SocialMediaPost]:
        """Get all posts from all platforms combined."""
        all_posts = []
        for platform, posts in platform_results.items():
            all_posts.extend(posts)
        
        # Sort by timestamp descending
        return sorted(all_posts, key=lambda x: x.timestamp, reverse=True)
    
    def filter_by_engagement(self, posts: List[SocialMediaPost], min_score: int = 1) -> List[SocialMediaPost]:
        """Filter posts by minimum engagement (likes, upvotes, etc.)."""
        return [post for post in posts if post.score and post.score >= min_score]
    
    def get_platform_stats(self, platform_results: Dict[str, List[SocialMediaPost]]) -> Dict[str, Dict]:
        """Get statistics for each platform."""
        stats = {}
        
        for platform, posts in platform_results.items():
            if posts:
                total_posts = len(posts)
                total_engagement = sum(post.score or 0 for post in posts)
                avg_engagement = total_engagement / total_posts if total_posts > 0 else 0
                
                stats[platform] = {
                    'total_posts': total_posts,
                    'total_engagement': total_engagement,
                    'avg_engagement': avg_engagement,
                    'latest_post': max(posts, key=lambda x: x.timestamp).timestamp if posts else None
                }
            else:
                stats[platform] = {
                    'total_posts': 0,
                    'total_engagement': 0,
                    'avg_engagement': 0,
                    'latest_post': None
                }
        
        return stats
    
    async def close(self):
        """Close all connections."""
        await self.reddit_collector.close()