#!/usr/bin/env python3
"""
Cache Manager

Centralized cache management for the trading system including Redis operations,
cache warming, and coordinated cache invalidation.
"""

import asyncio
import logging
import redis
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
import time

from config.settings import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Centralized cache management system."""
    
    def __init__(self):
        """Initialize cache manager."""
        self.redis_client = None
        self.cache_keys_by_category = {
            'market_data': [],
            'sentiment': [],
            'universe_filter': [],
            'technical_indicators': [],
            'news': [],
            'sec_filings': [],
            'social_media': []
        }
        self._setup_redis()
    
    def _setup_redis(self):
        """Setup Redis connection."""
        try:
            self.redis_client = redis.Redis(
                host=getattr(settings, 'redis_host', 'localhost'),
                port=getattr(settings, 'redis_port', 6379),
                decode_responses=True,
                socket_connect_timeout=10,
                socket_timeout=10,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Cache Manager Redis connection established")
            
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed in Cache Manager: {e}")
            self.redis_client = None
    
    def is_available(self) -> bool:
        """Check if Redis cache is available."""
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.ping()
            return True
        except:
            return False
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        if not self.is_available():
            return {"status": "unavailable", "error": "Redis not connected"}
        
        try:
            info = self.redis_client.info()
            
            # Get key counts by pattern
            key_stats = {}
            for category in self.cache_keys_by_category.keys():
                pattern = f"{category}:*"
                keys = self.redis_client.keys(pattern)
                key_stats[f"{category}_keys"] = len(keys)
            
            # Memory usage
            memory_info = {
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'used_memory_peak': info.get('used_memory_peak', 0),
                'used_memory_peak_human': info.get('used_memory_peak_human', '0B')
            }
            
            # Connection info
            connection_info = {
                'connected_clients': info.get('connected_clients', 0),
                'total_connections_received': info.get('total_connections_received', 0)
            }
            
            return {
                "status": "available",
                "total_keys": info.get('db0', {}).get('keys', 0) if 'db0' in info else 0,
                "key_stats": key_stats,
                "memory": memory_info,
                "connections": connection_info,
                "uptime_seconds": info.get('uptime_in_seconds', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting cache statistics: {e}")
            return {"status": "error", "error": str(e)}
    
    async def flush_all_cache(self) -> bool:
        """Flush all cache data."""
        if not self.is_available():
            logger.warning("Cannot flush cache - Redis not available")
            return False
        
        try:
            stats_before = await self.get_cache_statistics()
            
            logger.info("🧹 Flushing all cache data...")
            self.redis_client.flushall()
            
            stats_after = await self.get_cache_statistics()
            
            logger.info(f"✅ Cache flush completed: "
                       f"{stats_before.get('total_keys', 0)} → {stats_after.get('total_keys', 0)} keys")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to flush cache: {e}")
            return False
    
    async def flush_category_cache(self, category: str) -> bool:
        """Flush cache data for specific category."""
        if not self.is_available():
            logger.warning(f"Cannot flush {category} cache - Redis not available")
            return False
        
        if category not in self.cache_keys_by_category:
            logger.warning(f"Unknown cache category: {category}")
            return False
        
        try:
            pattern = f"{category}:*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                logger.info(f"🧹 Flushing {len(keys)} keys from {category} cache...")
                self.redis_client.delete(*keys)
                logger.info(f"✅ {category} cache flushed successfully")
            else:
                logger.info(f"ℹ️ No keys found for {category} cache")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to flush {category} cache: {e}")
            return False
    
    async def warm_cache_category(self, category: str, symbols: List[str] = None) -> bool:
        """Pre-populate cache for specific category."""
        logger.info(f"🔥 Warming {category} cache...")
        
        try:
            if category == 'market_data':
                await self._warm_market_data_cache(symbols or [])
            elif category == 'sentiment':
                await self._warm_sentiment_cache(symbols or [])
            elif category == 'universe_filter':
                await self._warm_universe_filter_cache()
            elif category == 'technical_indicators':
                await self._warm_technical_indicators_cache(symbols or [])
            else:
                logger.info(f"ℹ️ No cache warming logic implemented for {category}")
                return True
            
            logger.info(f"✅ {category} cache warming completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to warm {category} cache: {e}")
            return False
    
    async def _warm_market_data_cache(self, symbols: List[str]):
        """Warm market data cache."""
        if not symbols:
            return
        
        try:
            from tools.alpaca_market_data import fetch_stock_prices, fetch_stock_history
            
            # Warm current prices
            logger.info(f"Warming market data for {len(symbols)} symbols...")
            fetch_stock_prices(symbols)
            
            # Warm historical data for key symbols (limit to prevent overload)
            key_symbols = symbols[:20]  # Limit to top 20 symbols
            for symbol in key_symbols:
                fetch_stock_history(symbol, "1mo")
                await asyncio.sleep(0.1)  # Rate limiting
                
        except Exception as e:
            logger.warning(f"Error warming market data cache: {e}")
    
    async def _warm_sentiment_cache(self, symbols: List[str]):
        """Warm sentiment analysis cache."""
        if not symbols:
            return
        
        try:
            from agents.sentiment_agent import sentiment_agent
            
            # Warm sentiment for key symbols (limit to prevent overload)
            key_symbols = symbols[:10]  # Limit to top 10 symbols
            
            logger.info(f"Warming sentiment cache for {len(key_symbols)} symbols...")
            
            for symbol in key_symbols:
                try:
                    await sentiment_agent.analyze_comprehensive_sentiment(symbol)
                    await asyncio.sleep(1)  # Rate limiting between sentiment analyses
                except Exception as e:
                    logger.warning(f"Error warming sentiment for {symbol}: {e}")
                    
        except Exception as e:
            logger.warning(f"Error warming sentiment cache: {e}")
    
    async def _warm_universe_filter_cache(self):
        """Warm universe filter cache."""
        try:
            from core.universe_filter import UniverseFilter
            
            logger.info("Warming universe filter cache...")
            universe_filter = UniverseFilter()
            await universe_filter.filter_universe()
            
        except Exception as e:
            logger.warning(f"Error warming universe filter cache: {e}")
    
    async def _warm_technical_indicators_cache(self, symbols: List[str]):
        """Warm technical indicators cache."""
        if not symbols:
            return
        
        try:
            from tools.alpaca_market_data import get_technical_indicators
            
            # Warm technical indicators for key symbols
            key_symbols = symbols[:15]  # Limit to top 15 symbols
            
            logger.info(f"Warming technical indicators for {len(key_symbols)} symbols...")
            
            for symbol in key_symbols:
                try:
                    get_technical_indicators(symbol)
                    await asyncio.sleep(0.2)  # Rate limiting
                except Exception as e:
                    logger.warning(f"Error warming technical indicators for {symbol}: {e}")
                    
        except Exception as e:
            logger.warning(f"Error warming technical indicators cache: {e}")
    
    async def invalidate_symbol_cache(self, symbol: str) -> bool:
        """Invalidate all cache entries for a specific symbol."""
        if not self.is_available():
            return False
        
        try:
            patterns = [
                f"market_data:*{symbol}*",
                f"sentiment:*{symbol}*",
                f"technical_indicators:*{symbol}*",
                f"news:*{symbol}*"
            ]
            
            total_deleted = 0
            for pattern in patterns:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                    total_deleted += len(keys)
            
            if total_deleted > 0:
                logger.info(f"🗑️ Invalidated {total_deleted} cache entries for {symbol}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error invalidating cache for {symbol}: {e}")
            return False
    
    async def get_cache_health(self) -> Dict[str, Any]:
        """Get cache health status."""
        health = {
            "redis_available": self.is_available(),
            "timestamp": datetime.now().isoformat()
        }
        
        if self.is_available():
            try:
                # Test basic operations
                test_key = f"health_check:{int(time.time())}"
                self.redis_client.setex(test_key, 10, "test")
                retrieved = self.redis_client.get(test_key)
                self.redis_client.delete(test_key)
                
                health["basic_operations"] = retrieved == "test"
                health["response_time_ms"] = self._measure_redis_response_time()
                
                stats = await self.get_cache_statistics()
                health["memory_usage_mb"] = stats.get("memory", {}).get("used_memory", 0) / (1024 * 1024)
                health["total_keys"] = stats.get("total_keys", 0)
                
            except Exception as e:
                health["error"] = str(e)
                health["basic_operations"] = False
        
        return health
    
    def _measure_redis_response_time(self) -> float:
        """Measure Redis response time in milliseconds."""
        if not self.is_available():
            return -1
        
        try:
            start_time = time.time()
            self.redis_client.ping()
            end_time = time.time()
            return (end_time - start_time) * 1000  # Convert to milliseconds
            
        except:
            return -1
    
    async def schedule_cache_maintenance(self):
        """Run periodic cache maintenance."""
        while True:
            try:
                if self.is_available():
                    health = await self.get_cache_health()
                    
                    # Log health status
                    if health.get("basic_operations"):
                        logger.debug(f"💚 Cache health good: "
                                   f"{health.get('total_keys', 0)} keys, "
                                   f"{health.get('response_time_ms', 0):.1f}ms response")
                    else:
                        logger.warning("💔 Cache health issues detected")
                    
                    # Clean up expired keys (Redis handles this automatically, but we can help)
                    try:
                        info = self.redis_client.info()
                        expired_keys = info.get('expired_keys', 0)
                        if expired_keys > 0:
                            logger.debug(f"🧽 Redis cleaned up {expired_keys} expired keys")
                    except:
                        pass
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in cache maintenance: {e}")
                await asyncio.sleep(300)


# Global cache manager instance
cache_manager = CacheManager()


async def flush_all_cache() -> bool:
    """Convenience function to flush all cache."""
    return await cache_manager.flush_all_cache()


async def get_cache_health() -> Dict[str, Any]:
    """Convenience function to get cache health."""
    return await cache_manager.get_cache_health()


if __name__ == "__main__":
    # Test cache manager
    async def test_cache_manager():
        cm = CacheManager()
        
        print("📊 Cache Statistics:")
        stats = await cm.get_cache_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n💚 Cache Health:")
        health = await cm.get_cache_health()
        for key, value in health.items():
            print(f"  {key}: {value}")
        
        print(f"\n🧪 Testing cache flush...")
        success = await cm.flush_all_cache()
        print(f"Result: {'✅ Success' if success else '❌ Failed'}")
    
    asyncio.run(test_cache_manager())