#!/usr/bin/env python3
"""
Market Data Proxy Service
Real-time WebSocket-based market data proxy with Redis caching and symbol subscription management.

Architecture:
[Exchange/Broker WebSocket] → [Market Data Proxy] → [Redis Cache] → [RL Agent/Dashboard/Paper Trader]

Features:
- WebSocket connections to Finnhub and Alpha Vantage
- Redis caching for sub-second latency
- Dynamic symbol subscription management
- Automatic reconnection and error handling
- Multi-provider data aggregation
- Rate limit elimination through caching
"""

import asyncio
import aiohttp
import websockets
import redis.asyncio as redis
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import signal
import sys

from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for cache behavior."""
    ttl_seconds: int = 300  # 5 minutes TTL for market data
    historical_ttl_seconds: int = 3600  # 1 hour for historical data
    max_reconnect_attempts: int = 3
    ping_interval: int = 20
    ping_timeout: int = 10
    bulk_operation_timeout: int = 30
    connection_timeout: int = 5

@dataclass
class MarketDataPoint:
    """Standardized market data point."""
    symbol: str
    price: float
    timestamp: float
    volume: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    source: str = "market_data_proxy"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketDataPoint':
        """Create from dictionary (Redis retrieval)."""
        return cls(**data)

@dataclass
class SubscriptionMetrics:
    """Metrics for subscription monitoring."""
    symbol: str
    last_update: float
    update_count: int
    source: str
    latency_ms: float

class CircuitBreaker:
    """Circuit breaker for data source failures."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
    
    def call_failed(self):
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def call_succeeded(self):
        """Record a successful call."""
        self.failure_count = 0
        self.state = "closed"
    
    def can_call(self) -> bool:
        """Check if calls are allowed."""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                return True
            return False
        
        # half-open state
        return True
    
class SymbolSubscriptionManager:
    """Manages dynamic symbol subscriptions across WebSocket connections."""
    
    def __init__(self):
        self.active_subscriptions: Set[str] = set()
        self.subscription_callbacks: Dict[str, List[Callable]] = {}
        self.metrics: Dict[str, SubscriptionMetrics] = {}
        self._lock = asyncio.Lock()
        
        # Auto-subscribe to core symbols from settings
        self.core_symbols = self._get_core_symbols()
        logger.info(f"🎯 Core symbols for auto-subscription: {self.core_symbols}")
    
    def _get_core_symbols(self) -> List[str]:
        """Get core symbols from settings that should always be subscribed."""
        core_symbols = []
        
        # Add focus ETFs
        if settings.focus_etfs:
            core_symbols.extend([s.strip() for s in settings.focus_etfs.split(',') if s.strip()])
        
        # Add any additional core symbols
        core_symbols.extend(['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL'])  # Market leaders
        
        return list(set(core_symbols))  # Remove duplicates
    
    async def subscribe(self, symbol: str, callback: Optional[Callable] = None) -> bool:
        """Subscribe to a symbol's market data."""
        async with self._lock:
            symbol = symbol.upper()
            was_new = symbol not in self.active_subscriptions
            
            self.active_subscriptions.add(symbol)
            
            if callback:
                if symbol not in self.subscription_callbacks:
                    self.subscription_callbacks[symbol] = []
                self.subscription_callbacks[symbol].append(callback)
            
            if was_new:
                logger.info(f"📡 New subscription: {symbol}")
                return True
            return False
    
    async def unsubscribe(self, symbol: str, callback: Optional[Callable] = None) -> bool:
        """Unsubscribe from a symbol's market data."""
        async with self._lock:
            symbol = symbol.upper()
            
            if callback and symbol in self.subscription_callbacks:
                try:
                    self.subscription_callbacks[symbol].remove(callback)
                    if not self.subscription_callbacks[symbol]:
                        del self.subscription_callbacks[symbol]
                except ValueError:
                    pass
            
            # Only remove from active subscriptions if no callbacks remain
            if symbol not in self.subscription_callbacks and symbol not in self.core_symbols:
                self.active_subscriptions.discard(symbol)
                logger.info(f"📡 Unsubscribed: {symbol}")
                return True
            return False
    
    def get_subscriptions(self) -> Set[str]:
        """Get current active subscriptions."""
        return self.active_subscriptions.copy()
    
    async def notify_update(self, data_point: MarketDataPoint):
        """Notify all callbacks for a symbol update."""
        symbol = data_point.symbol.upper()
        
        # Update metrics
        now = time.time()
        self.metrics[symbol] = SubscriptionMetrics(
            symbol=symbol,
            last_update=now,
            update_count=self.metrics.get(symbol, SubscriptionMetrics(symbol, 0, 0, "", 0)).update_count + 1,
            source=data_point.source,
            latency_ms=(now - data_point.timestamp) * 1000
        )
        
        # Notify callbacks
        if symbol in self.subscription_callbacks:
            for callback in self.subscription_callbacks[symbol]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data_point)
                    else:
                        callback(data_point)
                except Exception as e:
                    logger.error(f"Callback error for {symbol}: {e}")

class RedisCache:
    """Redis-based cache for market data with optimized performance."""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.cache_ttl = 300  # 5 minutes TTL for market data
        self.connection_retries = 3
        
    async def connect(self):
        """Connect to Redis."""
        for attempt in range(self.connection_retries):
            try:
                self.redis_client = redis.Redis(
                    host=settings.redis_host,
                    port=settings.redis_port,
                    db=settings.redis_db,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    health_check_interval=30
                )
                
                # Test connection
                await self.redis_client.ping()
                logger.info(f"✅ Redis connected: {settings.redis_host}:{settings.redis_port}")
                return
                
            except Exception as e:
                logger.warning(f"Redis connection attempt {attempt + 1} failed: {e}")
                if attempt < self.connection_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error("❌ Redis connection failed after all retries")
                    raise
    
    async def set_market_data(self, symbol: str, data_point: MarketDataPoint):
        """Store market data in Redis."""
        if not self.redis_client:
            return
        
        try:
            # Store current price data
            key = f"market_data:{symbol.upper()}"
            data = data_point.to_dict()
            
            # Use pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            pipe.hset(key, mapping=data)
            pipe.expire(key, self.cache_ttl)
            
            # Store in time series for historical tracking
            ts_key = f"market_data_ts:{symbol.upper()}"
            pipe.zadd(ts_key, {json.dumps(data): data_point.timestamp})
            pipe.zremrangebyscore(ts_key, 0, time.time() - 3600)  # Keep 1 hour of data
            pipe.expire(ts_key, 3600)
            
            await pipe.execute()
            
        except Exception as e:
            logger.error(f"Redis set error for {symbol}: {e}")
    
    async def get_market_data(self, symbol: str) -> Optional[MarketDataPoint]:
        """Retrieve market data from Redis."""
        if not self.redis_client:
            return None
        
        try:
            key = f"market_data:{symbol.upper()}"
            data = await self.redis_client.hgetall(key)
            
            if data:
                # Convert string values back to appropriate types
                data['price'] = float(data['price'])
                data['timestamp'] = float(data['timestamp'])
                if data.get('volume'):
                    data['volume'] = float(data['volume'])
                if data.get('bid'):
                    data['bid'] = float(data['bid'])
                if data.get('ask'):
                    data['ask'] = float(data['ask'])
                if data.get('change'):
                    data['change'] = float(data['change'])
                if data.get('change_percent'):
                    data['change_percent'] = float(data['change_percent'])
                
                return MarketDataPoint.from_dict(data)
            
        except Exception as e:
            logger.error(f"Redis get error for {symbol}: {e}")
        
        return None
    
    async def get_bulk_market_data(self, symbols: List[str]) -> Dict[str, MarketDataPoint]:
        """Retrieve market data for multiple symbols efficiently."""
        if not self.redis_client or not symbols:
            return {}
        
        try:
            # Use pipeline for bulk operations
            pipe = self.redis_client.pipeline()
            for symbol in symbols:
                key = f"market_data:{symbol.upper()}"
                pipe.hgetall(key)
            
            results = await pipe.execute()
            
            market_data = {}
            for symbol, data in zip(symbols, results):
                if data:
                    try:
                        # Convert string values back to appropriate types
                        data['price'] = float(data['price'])
                        data['timestamp'] = float(data['timestamp'])
                        if data.get('volume'):
                            data['volume'] = float(data['volume'])
                        if data.get('bid'):
                            data['bid'] = float(data['bid'])
                        if data.get('ask'):
                            data['ask'] = float(data['ask'])
                        if data.get('change'):
                            data['change'] = float(data['change'])
                        if data.get('change_percent'):
                            data['change_percent'] = float(data['change_percent'])
                        
                        market_data[symbol.upper()] = MarketDataPoint.from_dict(data)
                    except (ValueError, KeyError) as e:
                        logger.debug(f"Invalid data format for {symbol}: {e}")
                        continue
            
            return market_data
            
        except Exception as e:
            logger.error(f"Redis bulk get error: {e}")
            return {}
    
    async def get_symbols_with_recent_data(self, max_age_seconds: int = 60) -> List[str]:
        """Get symbols that have recent market data."""
        if not self.redis_client:
            return []
        
        try:
            # Scan for market data keys
            symbols = []
            cutoff_time = time.time() - max_age_seconds
            
            async for key in self.redis_client.scan_iter(match="market_data:*"):
                symbol = key.split(":", 1)[1]
                data = await self.redis_client.hget(key, "timestamp")
                if data and float(data) > cutoff_time:
                    symbols.append(symbol)
            
            return symbols
            
        except Exception as e:
            logger.error(f"Redis symbols scan error: {e}")
            return []
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()

class FinnhubWebSocketClient:
    """WebSocket client for Finnhub real-time data."""
    
    def __init__(self, api_key: str, cache: RedisCache, subscription_manager: SymbolSubscriptionManager):
        self.api_key = api_key
        self.cache = cache
        self.subscription_manager = subscription_manager
        self.websocket = None
        self.url = f"wss://ws.finnhub.io?token={api_key}"
        self.reconnect_interval = 5
        self.max_reconnect_interval = 60
        self.running = False
        
    async def connect(self):
        """Connect to Finnhub WebSocket."""
        self.running = True
        reconnect_count = 0
        
        while self.running:
            try:
                logger.info("🔌 Connecting to Finnhub WebSocket...")
                
                async with websockets.connect(
                    self.url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                ) as websocket:
                    self.websocket = websocket
                    reconnect_count = 0
                    logger.info("✅ Finnhub WebSocket connected")
                    
                    # Subscribe to symbols
                    await self._subscribe_to_symbols()
                    
                    # Listen for messages
                    await self._listen()
                    
            except Exception as e:
                reconnect_count += 1
                sleep_time = min(self.reconnect_interval * (2 ** min(reconnect_count, 5)), self.max_reconnect_interval)
                logger.error(f"❌ Finnhub WebSocket error: {e}")
                logger.info(f"🔄 Reconnecting in {sleep_time}s (attempt {reconnect_count})")
                await asyncio.sleep(sleep_time)
    
    async def _subscribe_to_symbols(self):
        """Subscribe to symbols via WebSocket."""
        if not self.websocket:
            return
        
        symbols = self.subscription_manager.get_subscriptions()
        logger.info(f"📡 Subscribing to {len(symbols)} symbols via Finnhub")
        
        for symbol in symbols:
            try:
                subscribe_msg = {"type": "subscribe", "symbol": symbol}
                await self.websocket.send(json.dumps(subscribe_msg))
                logger.debug(f"📡 Subscribed to {symbol}")
                await asyncio.sleep(0.1)  # Small delay to avoid overwhelming
            except Exception as e:
                logger.error(f"Subscribe error for {symbol}: {e}")
    
    async def _listen(self):
        """Listen for WebSocket messages."""
        if not self.websocket:
            return
        
        try:
            async for message in self.websocket:
                await self._process_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("🔌 Finnhub WebSocket connection closed")
        except Exception as e:
            logger.error(f"❌ Finnhub WebSocket listen error: {e}")
    
    async def _process_message(self, message: str):
        """Process incoming WebSocket message."""
        try:
            data = json.loads(message)
            
            if data.get('type') == 'trade':
                for trade in data.get('data', []):
                    # Validate required fields
                    symbol = trade.get('s', '').upper()
                    price = trade.get('p')
                    timestamp = trade.get('t', time.time() * 1000) / 1000
                    volume = trade.get('v')
                    
                    # Data validation
                    if not symbol or not price:
                        logger.debug(f"Invalid trade data: missing symbol or price")
                        continue
                    
                    try:
                        price_float = float(price)
                        if price_float <= 0:
                            logger.debug(f"Invalid price for {symbol}: {price}")
                            continue
                    except (ValueError, TypeError):
                        logger.debug(f"Invalid price format for {symbol}: {price}")
                        continue
                    
                    # Create validated data point
                    data_point = MarketDataPoint(
                        symbol=symbol,
                        price=price_float,
                        timestamp=timestamp,
                        volume=float(volume) if volume and volume > 0 else None,
                        source="finnhub_ws"
                    )
                    
                    # Store in cache
                    await self.cache.set_market_data(symbol, data_point)
                    
                    # Notify subscribers
                    await self.subscription_manager.notify_update(data_point)
                        
        except json.JSONDecodeError as e:
            logger.debug(f"Invalid JSON message: {e}")
        except Exception as e:
            logger.error(f"Message processing error: {e}")
    
    async def add_symbol(self, symbol: str):
        """Add a new symbol subscription."""
        if self.websocket:
            try:
                subscribe_msg = {"type": "subscribe", "symbol": symbol.upper()}
                await self.websocket.send(json.dumps(subscribe_msg))
                logger.info(f"📡 Added Finnhub subscription: {symbol}")
            except Exception as e:
                logger.error(f"Failed to subscribe to {symbol}: {e}")
    
    async def remove_symbol(self, symbol: str):
        """Remove a symbol subscription."""
        if self.websocket:
            try:
                unsubscribe_msg = {"type": "unsubscribe", "symbol": symbol.upper()}
                await self.websocket.send(json.dumps(unsubscribe_msg))
                logger.info(f"📡 Removed Finnhub subscription: {symbol}")
            except Exception as e:
                logger.error(f"Failed to unsubscribe from {symbol}: {e}")
    
    async def close(self):
        """Close WebSocket connection."""
        self.running = False
        if self.websocket:
            await self.websocket.close()

class MarketDataProxy:
    """Main market data proxy service."""
    
    def __init__(self):
        self.cache = RedisCache()
        self.subscription_manager = SymbolSubscriptionManager()
        self.finnhub_client = None
        self.running = False
        
        # Initialize WebSocket clients if API keys are available
        if settings.finnhub_api_key:
            self.finnhub_client = FinnhubWebSocketClient(
                settings.finnhub_api_key,
                self.cache,
                self.subscription_manager
            )
        
        logger.info("🚀 Market Data Proxy initialized")
    
    async def start(self):
        """Start the market data proxy service."""
        logger.info("🚀 Starting Market Data Proxy...")
        self.running = True
        
        # Connect to Redis
        await self.cache.connect()
        
        # Subscribe to core symbols
        for symbol in self.subscription_manager.core_symbols:
            await self.subscription_manager.subscribe(symbol)
        
        # Start WebSocket clients
        tasks = []
        
        if self.finnhub_client:
            tasks.append(asyncio.create_task(self.finnhub_client.connect()))
        
        # Start monitoring task
        tasks.append(asyncio.create_task(self._monitor_subscriptions()))
        
        logger.info("✅ Market Data Proxy started")
        
        # Wait for tasks
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down Market Data Proxy...")
            await self.stop()
    
    async def stop(self):
        """Stop the market data proxy service."""
        self.running = False
        
        if self.finnhub_client:
            await self.finnhub_client.close()
        
        await self.cache.close()
        logger.info("✅ Market Data Proxy stopped")
    
    async def _monitor_subscriptions(self):
        """Monitor subscription health and performance."""
        while self.running:
            try:
                symbols = await self.cache.get_symbols_with_recent_data(max_age_seconds=60)
                active_subs = len(self.subscription_manager.get_subscriptions())
                
                logger.info(f"📊 Active subscriptions: {active_subs}, Recent data: {len(symbols)}")
                
                # Log metrics for subscriptions
                for symbol, metrics in list(self.subscription_manager.metrics.items())[:5]:
                    age = time.time() - metrics.last_update
                    logger.debug(f"📈 {symbol}: {metrics.update_count} updates, "
                               f"last {age:.1f}s ago, {metrics.latency_ms:.1f}ms latency")
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(10)
    
    # Public API methods for external use
    
    async def subscribe_symbol(self, symbol: str) -> bool:
        """Subscribe to a symbol's market data."""
        symbol = symbol.upper()
        was_new = await self.subscription_manager.subscribe(symbol)
        
        if was_new:
            # Add to WebSocket clients
            if self.finnhub_client:
                await self.finnhub_client.add_symbol(symbol)
        
        return was_new
    
    async def unsubscribe_symbol(self, symbol: str) -> bool:
        """Unsubscribe from a symbol's market data."""
        symbol = symbol.upper()
        was_removed = await self.subscription_manager.unsubscribe(symbol)
        
        if was_removed:
            # Remove from WebSocket clients
            if self.finnhub_client:
                await self.finnhub_client.remove_symbol(symbol)
        
        return was_removed
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        data = await self.cache.get_market_data(symbol.upper())
        return data.price if data else None
    
    async def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get full market data for a symbol."""
        data = await self.cache.get_market_data(symbol.upper())
        return data.to_dict() if data else None
    
    async def get_bulk_market_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get market data for multiple symbols."""
        data_points = await self.cache.get_bulk_market_data(symbols)
        return {symbol: data.to_dict() for symbol, data in data_points.items()}
    
    async def get_subscription_metrics(self) -> Dict[str, Any]:
        """Get subscription and performance metrics."""
        recent_symbols = await self.cache.get_symbols_with_recent_data()
        active_subs = self.subscription_manager.get_subscriptions()
        
        return {
            "active_subscriptions": len(active_subs),
            "symbols_with_recent_data": len(recent_symbols),
            "subscription_list": list(active_subs),
            "recent_symbols": recent_symbols,
            "total_updates": sum(m.update_count for m in self.subscription_manager.metrics.values()),
            "avg_latency_ms": sum(m.latency_ms for m in self.subscription_manager.metrics.values()) / 
                            max(len(self.subscription_manager.metrics), 1)
        }

# Global instance
market_data_proxy = MarketDataProxy()

# Convenience functions for external use
async def get_current_price(symbol: str) -> Optional[float]:
    """Get current price for a symbol."""
    return await market_data_proxy.get_current_price(symbol)

async def get_market_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Get market data for a symbol."""
    return await market_data_proxy.get_market_data(symbol)

async def get_bulk_market_data(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Get market data for multiple symbols."""
    return await market_data_proxy.get_bulk_market_data(symbols)

async def subscribe_symbol(symbol: str) -> bool:
    """Subscribe to a symbol."""
    return await market_data_proxy.subscribe_symbol(symbol)

async def unsubscribe_symbol(symbol: str) -> bool:
    """Unsubscribe from a symbol."""
    return await market_data_proxy.unsubscribe_symbol(symbol)

# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

if __name__ == "__main__":
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start the proxy
    try:
        asyncio.run(market_data_proxy.start())
    except KeyboardInterrupt:
        logger.info("Market Data Proxy stopped by user")
    except Exception as e:
        logger.error(f"Market Data Proxy crashed: {e}")
        sys.exit(1)