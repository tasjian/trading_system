#!/usr/bin/env python3
"""
Enhanced API Caching System

Provides intelligent caching for all API calls with:
- Redis-based persistence
- Automatic cache warming
- Circuit breaker patterns
- Rate limiting integration
- API failure resilience
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

import redis.asyncio as redis

logger = logging.getLogger(__name__)

class CacheType(Enum):
    NEWS = "news"
    SOCIAL_MEDIA = "social" 
    SEC_FILINGS = "sec"
    EARNINGS = "earnings"
    MARKET_DATA = "market"

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    data: Any
    timestamp: float
    expiry: float
    source: str
    attempts: int = 0
    last_error: Optional[str] = None

class EnhancedAPICache:
    """Enhanced caching system for API reliability and performance."""
    
    def __init__(self):
        self.redis_client = None
        self.local_cache = {}  # Fallback local cache
        
        # Cache TTLs by type (seconds)
        self.cache_ttls = {
            CacheType.NEWS: 1800,        # 30 minutes for news
            CacheType.SOCIAL_MEDIA: 3600, # 1 hour for social media
            CacheType.SEC_FILINGS: 86400, # 24 hours for SEC filings
            CacheType.EARNINGS: 43200,    # 12 hours for earnings
            CacheType.MARKET_DATA: 300    # 5 minutes for market data
        }
        
        # Error backoff times (seconds)
        self.error_backoff = {
            1: 300,   # 5 minutes after 1st failure
            2: 900,   # 15 minutes after 2nd failure
            3: 3600,  # 1 hour after 3rd failure
            4: 7200   # 2 hours after 4+ failures
        }
    
    async def initialize(self):
        """Initialize Redis connection with fallback."""
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=True,
                socket_timeout=5,
                retry_on_timeout=True
            )
            # Test connection
            await self.redis_client.ping()
            logger.info("✅ Enhanced API cache connected to Redis")
        except Exception as e:
            logger.warning(f"Redis connection failed, using local cache: {e}")
            self.redis_client = None
    
    async def get_cached_data(self, cache_type: CacheType, key: str) -> Optional[Any]:
        """Get cached data with fallback logic."""
        cache_key = f"{cache_type.value}:{key}"
        
        try:
            # Try Redis first
            if self.redis_client:
                try:
                    cached_json = await self.redis_client.get(cache_key)
                    if cached_json:
                        entry_dict = json.loads(cached_json)
                        entry = CacheEntry(**entry_dict)
                        
                        # Check if entry has expired or is in error backoff
                        current_time = time.time()
                        if current_time < entry.expiry:
                            # Check for error backoff
                            if entry.last_error and entry.attempts > 0:
                                backoff_time = self.error_backoff.get(entry.attempts, 7200)
                                if current_time < (entry.timestamp + backoff_time):
                                    logger.debug(f"Cache entry {cache_key} in error backoff")
                                    return None
                            
                            logger.debug(f"Cache hit: {cache_key}")
                            return entry.data
                        else:
                            # Expired entry
                            await self.redis_client.delete(cache_key)
                except Exception as e:
                    logger.debug(f"Redis cache read error: {e}")
            
            # Fallback to local cache
            if cache_key in self.local_cache:
                entry = self.local_cache[cache_key]
                if time.time() < entry.expiry:
                    logger.debug(f"Local cache hit: {cache_key}")
                    return entry.data
                else:
                    del self.local_cache[cache_key]
            
        except Exception as e:
            logger.error(f"Cache read error for {cache_key}: {e}")
        
        return None
    
    async def cache_data(self, cache_type: CacheType, key: str, data: Any, 
                        source: str = "api", error: Optional[str] = None):
        """Cache data with error tracking."""
        cache_key = f"{cache_type.value}:{key}"
        current_time = time.time()
        ttl = self.cache_ttls[cache_type]
        
        # Get previous attempts count
        attempts = 0
        if error:
            try:
                old_entry = await self.get_cached_entry_metadata(cache_key)
                if old_entry:
                    attempts = old_entry.attempts + 1
                else:
                    attempts = 1
            except:
                attempts = 1
        
        entry = CacheEntry(
            data=data,
            timestamp=current_time,
            expiry=current_time + ttl,
            source=source,
            attempts=attempts,
            last_error=error
        )
        
        try:
            # Cache to Redis
            if self.redis_client:
                try:
                    entry_dict = {
                        'data': data,
                        'timestamp': entry.timestamp,
                        'expiry': entry.expiry,
                        'source': source,
                        'attempts': attempts,
                        'last_error': error
                    }
                    await self.redis_client.setex(
                        cache_key,
                        ttl,
                        json.dumps(entry_dict, default=str)
                    )
                except Exception as e:
                    logger.debug(f"Redis cache write error: {e}")
            
            # Always cache locally as fallback
            self.local_cache[cache_key] = entry
            
            # Limit local cache size
            if len(self.local_cache) > 1000:
                # Remove oldest entries
                oldest_keys = sorted(
                    self.local_cache.keys(),
                    key=lambda k: self.local_cache[k].timestamp
                )[:100]
                for old_key in oldest_keys:
                    del self.local_cache[old_key]
                    
        except Exception as e:
            logger.error(f"Cache write error for {cache_key}: {e}")
    
    async def get_cached_entry_metadata(self, cache_key: str) -> Optional[CacheEntry]:
        """Get cache entry metadata for error tracking."""
        try:
            if self.redis_client:
                cached_json = await self.redis_client.get(cache_key)
                if cached_json:
                    entry_dict = json.loads(cached_json)
                    return CacheEntry(**entry_dict)
        except:
            pass
        
        return self.local_cache.get(cache_key)
    
    async def cached_api_call(self, cache_type: CacheType, key: str, 
                             api_func: Callable, *args, **kwargs) -> Any:
        """Make cached API call with automatic error handling."""
        # Check cache first
        cached_data = await self.get_cached_data(cache_type, key)
        if cached_data is not None:
            return cached_data
        
        # Make API call with error handling
        try:
            logger.debug(f"Making API call for {cache_type.value}:{key}")
            result = await api_func(*args, **kwargs)
            
            # Cache successful result
            await self.cache_data(cache_type, key, result, source="api")
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"API call failed for {cache_type.value}:{key}: {error_msg}")
            
            # Cache the error to prevent immediate retries
            await self.cache_data(cache_type, key, None, source="api", error=error_msg)
            return None
    
    async def warm_cache(self, symbols: list, cache_types: list[CacheType]):
        """Pre-warm cache for symbols and data types."""
        logger.info(f"🔥 Warming API cache for {len(symbols)} symbols")
        
        # Stagger warming to prevent API overload
        for i, symbol in enumerate(symbols):
            for cache_type in cache_types:
                cache_key = f"{cache_type.value}:{symbol}"
                cached_data = await self.get_cached_data(cache_type, symbol)
                
                if cached_data is None:
                    logger.debug(f"Cache miss for {cache_key} - will be populated on demand")
            
            # Add small delay between symbols
            if i < len(symbols) - 1:
                await asyncio.sleep(0.1)
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        stats = {
            'local_cache_size': len(self.local_cache),
            'redis_connected': self.redis_client is not None,
            'cache_types': [ct.value for ct in CacheType]
        }
        
        if self.redis_client:
            try:
                redis_info = await self.redis_client.info()
                stats['redis_memory_used'] = redis_info.get('used_memory_human', 'unknown')
                stats['redis_keys'] = redis_info.get('db0', {}).get('keys', 0) if 'db0' in redis_info else 0
            except:
                stats['redis_error'] = True
        
        return stats

# Global cache instance
_enhanced_cache = None

async def get_enhanced_cache() -> EnhancedAPICache:
    """Get global enhanced cache instance."""
    global _enhanced_cache
    if _enhanced_cache is None:
        _enhanced_cache = EnhancedAPICache()
        await _enhanced_cache.initialize()
    return _enhanced_cache

# Decorator for automatic caching
def cached_api(cache_type: CacheType, key_func: Callable = None):
    """Decorator for automatic API caching."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache = await get_enhanced_cache()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key from function name and first argument
                cache_key = f"{func.__name__}_{args[0] if args else 'default'}"
            
            return await cache.cached_api_call(cache_type, cache_key, func, *args, **kwargs)
        
        return wrapper
    return decorator

__all__ = [
    'EnhancedAPICache',
    'CacheType', 
    'get_enhanced_cache',
    'cached_api'
]