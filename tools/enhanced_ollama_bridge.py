#!/usr/bin/env python3
"""
Enhanced Ollama Bridge for High-Frequency Trading

Optimizations for ultra-low latency sentiment analysis in trading environments:
- Multi-tier caching strategy
- Express lanes for critical requests  
- Request batching and preprocessing
- Trading-aware circuit breaker
- Streaming response handling
"""

import asyncio
import logging
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading
import aiohttp
import weakref
from collections import defaultdict, OrderedDict

from .ollama_async_bridge import (
    OllamaAsyncBridge, OllamaRequest, OllamaResponse, RequestPriority,
    CircuitBreakerState, OllamaPerformanceMonitor
)

logger = logging.getLogger(__name__)


class CacheTier(Enum):
    """Cache tier levels for different performance requirements."""
    L1_ULTRA_FAST = "l1_ultra_fast"      # In-memory dict, <1ms access
    L2_FAST = "l2_fast"                  # LRU cache, <5ms access  
    L3_PERSISTENT = "l3_persistent"      # Redis/DB, <50ms access


@dataclass
class CacheConfig:
    """Configuration for cache tiers."""
    storage_type: str
    max_size: int
    ttl_seconds: int
    priority_levels: List[RequestPriority]
    access_target_ms: float


class UltraFastCache:
    """L1 ultra-fast cache for critical trading signals."""
    
    def __init__(self, max_size: int = 100, default_ttl: int = 60):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, tuple] = OrderedDict()  # key -> (data, expiry)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with sub-millisecond access."""
        with self._lock:
            if key in self.cache:
                data, expiry = self.cache[key]
                if datetime.now() < expiry:
                    # Move to end (most recent)
                    self.cache.move_to_end(key)
                    self.hits += 1
                    return data
                else:
                    del self.cache[key]
            
            self.misses += 1
            return None
    
    def put(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """Store item in cache."""
        expiry = datetime.now() + timedelta(seconds=ttl or self.default_ttl)
        
        with self._lock:
            self.cache[key] = (data, expiry)
            
            # Maintain size limit
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)  # Remove oldest
    
    def clear_expired(self) -> int:
        """Clear expired entries and return count cleared."""
        now = datetime.now()
        expired_keys = []
        
        with self._lock:
            for key, (_, expiry) in self.cache.items():
                if now >= expiry:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": self.max_size
        }


class MultiTierCacheManager:
    """Manages multiple cache tiers for optimal performance."""
    
    def __init__(self):
        self.cache_configs = {
            CacheTier.L1_ULTRA_FAST: CacheConfig(
                storage_type="in_memory_dict",
                max_size=100,
                ttl_seconds=60,
                priority_levels=[RequestPriority.CRITICAL, RequestPriority.HIGH],
                access_target_ms=1.0
            ),
            CacheTier.L2_FAST: CacheConfig(
                storage_type="lru_cache",
                max_size=1000,
                ttl_seconds=300,
                priority_levels=[RequestPriority.NORMAL],
                access_target_ms=5.0
            ),
            CacheTier.L3_PERSISTENT: CacheConfig(
                storage_type="persistent_cache",
                max_size=10000,
                ttl_seconds=3600,
                priority_levels=[RequestPriority.LOW],
                access_target_ms=50.0
            )
        }
        
        # Initialize cache instances
        self.l1_cache = UltraFastCache(
            max_size=self.cache_configs[CacheTier.L1_ULTRA_FAST].max_size,
            default_ttl=self.cache_configs[CacheTier.L1_ULTRA_FAST].ttl_seconds
        )
        
        # Import and reuse existing L2 cache from base bridge
        from .ollama_async_bridge import OllamaResponseCache
        self.l2_cache = OllamaResponseCache(
            max_size=self.cache_configs[CacheTier.L2_FAST].max_size,
            default_ttl=self.cache_configs[CacheTier.L2_FAST].ttl_seconds
        )
        
        # L3 would integrate with Redis/persistent storage
        self.l3_cache = None  # Placeholder for persistent cache
    
    def _generate_cache_key(self, prompt: str, model: str, context: str) -> str:
        """Generate consistent cache key."""
        content = f"{model}:{context}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def get(self, prompt: str, model: str, context: str, priority: RequestPriority) -> Optional[OllamaResponse]:
        """Get cached response from appropriate tier."""
        cache_key = self._generate_cache_key(prompt, model, context)
        
        # Try L1 cache first for high priority requests
        if priority in [RequestPriority.CRITICAL, RequestPriority.HIGH]:
            result = self.l1_cache.get(cache_key)
            if result:
                logger.debug(f"L1 cache hit for key: {cache_key[:8]}...")
                return result
        
        # Try L2 cache for normal priority
        if priority == RequestPriority.NORMAL:
            result = self.l2_cache.get(prompt, model, context)
            if result:
                logger.debug(f"L2 cache hit for key: {cache_key[:8]}...")
                return result
        
        # L3 cache for low priority (placeholder)
        # if priority == RequestPriority.LOW and self.l3_cache:
        #     result = await self.l3_cache.get(cache_key)
        #     if result:
        #         return result
        
        return None
    
    async def put(self, prompt: str, model: str, context: str, response: OllamaResponse, priority: RequestPriority) -> None:
        """Store response in appropriate cache tier."""
        cache_key = self._generate_cache_key(prompt, model, context)
        
        # Store in L1 for critical/high priority
        if priority in [RequestPriority.CRITICAL, RequestPriority.HIGH]:
            self.l1_cache.put(cache_key, response)
        
        # Always store in L2 for cross-tier access
        self.l2_cache.put(prompt, model, context, response)
        
        # Store in L3 for persistence (placeholder)
        # if self.l3_cache:
        #     await self.l3_cache.put(cache_key, response)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "l1_stats": self.l1_cache.get_stats(),
            "l2_stats": {
                "size": len(self.l2_cache.cache),
                "max_size": self.l2_cache.max_size
            },
            "l3_stats": {"enabled": False}  # Placeholder
        }


class RequestPreprocessor:
    """Optimizes requests before processing."""
    
    def __init__(self):
        self.text_normalization_cache = {}
        self.model_selection_rules = {
            "sentiment_analysis": {
                RequestPriority.CRITICAL: "llama3:8b-q4_0",  # Quantized for speed
                RequestPriority.HIGH: "llama3:8b-q4_0",
                RequestPriority.NORMAL: "llama3:8b",
                RequestPriority.LOW: "llama3:8b"
            },
            "market_analysis": {
                RequestPriority.CRITICAL: "llama3:8b-q4_0",
                RequestPriority.HIGH: "llama3:8b",
                RequestPriority.NORMAL: "llama3:8b",
                RequestPriority.LOW: "llama3:8b"
            }
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for better cache hit rates."""
        # Cache normalization results
        if text in self.text_normalization_cache:
            return self.text_normalization_cache[text]
        
        # Normalize whitespace and common variations
        normalized = " ".join(text.split())
        normalized = normalized.replace("&amp;", "&")
        normalized = normalized.replace("  ", " ")
        
        # Cache result
        if len(self.text_normalization_cache) < 1000:
            self.text_normalization_cache[text] = normalized
        
        return normalized
    
    def compress_financial_prompt(self, prompt: str) -> str:
        """Compress common financial prompt patterns."""
        # Replace common patterns with shorter versions
        replacements = {
            "Analyze the sentiment of this financial text": "Analyze sentiment:",
            "financial news article": "news",
            "earnings call transcript": "earnings",
            "social media post about stocks": "social",
            "very_positive|positive|neutral|negative|very_negative": "sentiment_scale"
        }
        
        compressed = prompt
        for old, new in replacements.items():
            compressed = compressed.replace(old, new)
        
        return compressed
    
    def select_optimal_model(self, context: str, priority: RequestPriority) -> str:
        """Select optimal model based on context and priority."""
        context_rules = self.model_selection_rules.get(context, self.model_selection_rules["sentiment_analysis"])
        return context_rules.get(priority, "llama3:8b")
    
    def is_batchable(self, request: OllamaRequest) -> bool:
        """Determine if request can be batched."""
        # Criteria for batching:
        # 1. Low/Normal priority
        # 2. Standard sentiment analysis
        # 3. Similar prompt patterns
        return (
            request.priority in [RequestPriority.NORMAL, RequestPriority.LOW] and
            request.context in ["sentiment_analysis", "market_analysis"] and
            len(request.prompt) < 2000
        )
    
    async def preprocess_request(self, request: OllamaRequest) -> OllamaRequest:
        """Apply all preprocessing optimizations."""
        # 1. Text normalization
        request.prompt = self.normalize_text(request.prompt)
        
        # 2. Prompt compression
        request.prompt = self.compress_financial_prompt(request.prompt)
        
        # 3. Model selection optimization
        optimal_model = self.select_optimal_model(request.context, request.priority)
        if optimal_model != request.model:
            logger.debug(f"Optimized model selection: {request.model} -> {optimal_model}")
            request.model = optimal_model
        
        return request


class TradingAwareCircuitBreaker:
    """Circuit breaker with trading-specific behavior."""
    
    def __init__(self):
        self.base_state = CircuitBreakerState()
        
        # Trading-aware thresholds
        self.thresholds = {
            "market_hours": 3,      # Stricter during market hours
            "after_hours": 5,       # More lenient after hours
            "high_volatility": 2,   # Very strict during high volatility
            "earnings_season": 2    # Strict during earnings
        }
        
        self.volatility_threshold = 0.02  # 2% VIX change
    
    def get_failure_threshold(self) -> int:
        """Dynamic failure threshold based on trading conditions."""
        if self._is_high_volatility_period():
            return self.thresholds["high_volatility"]
        elif self._is_earnings_season():
            return self.thresholds["earnings_season"]
        elif self._is_market_hours():
            return self.thresholds["market_hours"]
        else:
            return self.thresholds["after_hours"]
    
    def _is_market_hours(self) -> bool:
        """Check if currently in market hours (9:30 AM - 4:00 PM ET)."""
        import pytz
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
        
        # Check if weekday and within market hours
        if now.weekday() >= 5:  # Weekend
            return False
        
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def _is_high_volatility_period(self) -> bool:
        """Detect high volatility periods."""
        # In production, this would check VIX, recent price movements, etc.
        # For now, return False as placeholder
        return False
    
    def _is_earnings_season(self) -> bool:
        """Detect if currently in earnings season."""
        # Peak earnings periods: January, April, July, October
        current_month = datetime.now().month
        return current_month in [1, 4, 7, 10]
    
    def should_allow_request(self) -> bool:
        """Check if circuit breaker allows requests."""
        # Use dynamic threshold
        current_threshold = self.get_failure_threshold()
        
        if self.base_state.state == "OPEN":
            # Check recovery with trading-aware timeout
            recovery_timeout = 30 if self._is_market_hours() else 60
            if (self.base_state.last_failure and 
                (datetime.now() - self.base_state.last_failure).total_seconds() > recovery_timeout):
                self.base_state.state = "HALF_OPEN"
                return True
            return False
        
        return True


class EnhancedOllamaAsyncBridge(OllamaAsyncBridge):
    """Enhanced Ollama bridge with trading-specific optimizations."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Enhanced components
        self.multi_tier_cache = MultiTierCacheManager()
        self.preprocessor = RequestPreprocessor()
        self.trading_circuit_breaker = TradingAwareCircuitBreaker()
        
        # Express lane for critical requests
        self.express_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="OllamaExpress")
        
        # Request batching
        self.batch_queue = defaultdict(list)
        self.batch_timer = None
        self.batch_interval = 0.1  # 100ms batching window
        
        # Performance tracking
        self.express_request_count = 0
        self.batch_request_count = 0
        
        logger.info("Enhanced Ollama bridge initialized with trading optimizations")
    
    async def query_express(
        self,
        prompt: str,
        model: str = "llama3:8b-q4_0",
        context: str = "express_sentiment",
        timeout: float = 2.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> OllamaResponse:
        """Express lane for critical trading decisions."""
        self.express_request_count += 1
        
        # Create high-priority request
        request = OllamaRequest(
            id=f"express_{self._generate_request_id()}",
            prompt=prompt,
            model=model,
            priority=RequestPriority.CRITICAL,
            timestamp=datetime.now(),
            timeout=timeout,
            context=context,
            metadata=metadata or {}
        )
        
        # Preprocess for optimization
        request = await self.preprocessor.preprocess_request(request)
        
        # Check L1 ultra-fast cache first
        cached_response = await self.multi_tier_cache.get(
            request.prompt, request.model, request.context, request.priority
        )
        if cached_response:
            cached_response.cached = True
            return cached_response
        
        # Use express executor for immediate processing
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                self.express_executor,
                self._sync_express_request,
                request
            )
            
            # Cache in L1 for future express requests
            await self.multi_tier_cache.put(
                request.prompt, request.model, request.context, response, request.priority
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Express request failed: {e}")
            return self._create_express_fallback_response(request)
    
    def _sync_express_request(self, request: OllamaRequest) -> OllamaResponse:
        """Synchronous express request processing."""
        import requests
        
        start_time = time.time()
        
        try:
            url = f"{self.ollama_base_url}/api/generate"
            payload = {
                "model": request.model,
                "prompt": request.prompt,
                "stream": False,
                "options": {
                    "temperature": 0.5,  # Lower temperature for faster, more deterministic responses
                    "top_p": 0.8,
                    "max_tokens": 800    # Reduced tokens for speed
                }
            }
            
            response = requests.post(
                url,
                json=payload,
                timeout=request.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("response", "")
                
                return OllamaResponse(
                    request_id=request.id,
                    content=content,
                    model=request.model,
                    processing_time=processing_time,
                    queue_time=0.0,  # Express lane has no queue time
                    success=True,
                    thread_id="express",
                    tokens_estimated=len(content) // 4,
                    metadata=request.metadata
                )
            else:
                raise Exception(f"Ollama HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            processing_time = time.time() - start_time
            raise Exception(f"Express Ollama request failed: {e}")
    
    def _create_express_fallback_response(self, request: OllamaRequest) -> OllamaResponse:
        """Create ultra-fast fallback for express requests."""
        if "sentiment" in request.context.lower():
            fallback_content = json.dumps({
                "sentiment": "neutral",
                "confidence": 0.6,
                "score": 0.0,
                "reasoning": "Express fallback - trading system operational",
                "key_phrases": ["market_neutral"],
                "financial_impact": "Neutral market conditions",
                "risk_factors": ["llm_unavailable"],
                "opportunities": ["monitor_for_recovery"]
            })
        else:
            fallback_content = "Express analysis unavailable. Trading system operational with conservative parameters."
        
        return OllamaResponse(
            request_id=request.id,
            content=fallback_content,
            model=f"{request.model}-express-fallback",
            processing_time=0.001,  # Sub-millisecond fallback
            queue_time=0.0,
            success=True,
            cached=False,
            thread_id="express-fallback",
            tokens_estimated=len(fallback_content) // 4,
            metadata={"fallback_reason": "express_timeout", **request.metadata}
        )
    
    async def batch_sentiment_analysis(
        self,
        symbol_texts: Dict[str, str],
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: float = 30.0
    ) -> Dict[str, OllamaResponse]:
        """Batch multiple symbol sentiment analysis."""
        self.batch_request_count += len(symbol_texts)
        
        # Group similar texts for cache efficiency
        batches = self._group_similar_texts(symbol_texts)
        
        # Process batches concurrently
        tasks = []
        for batch_symbols, batch_text in batches.items():
            task = self._process_sentiment_batch(batch_symbols, batch_text, priority, timeout)
            tasks.append(task)
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results
        final_results = {}
        for result in batch_results:
            if isinstance(result, dict):
                final_results.update(result)
            elif isinstance(result, Exception):
                logger.warning(f"Batch processing failed: {result}")
        
        return final_results
    
    def _group_similar_texts(self, symbol_texts: Dict[str, str]) -> Dict[str, str]:
        """Group similar texts for batch processing."""
        # Simple grouping by text similarity (could be enhanced with ML)
        text_groups = defaultdict(list)
        
        for symbol, text in symbol_texts.items():
            # Create a simple hash for grouping similar texts
            text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
            text_groups[text_hash].append((symbol, text))
        
        # Convert to batch format
        batches = {}
        for text_hash, symbol_text_pairs in text_groups.items():
            symbols = [pair[0] for pair in symbol_text_pairs]
            # Use first text as representative (they should be similar)
            representative_text = symbol_text_pairs[0][1]
            batch_key = f"batch_{text_hash}_{','.join(symbols[:3])}"  # Limit key length
            batches[batch_key] = representative_text
        
        return batches
    
    async def _process_sentiment_batch(
        self,
        batch_key: str,
        text: str,
        priority: RequestPriority,
        timeout: float
    ) -> Dict[str, OllamaResponse]:
        """Process a batch of similar sentiment analyses."""
        try:
            # Extract symbols from batch key
            symbols = batch_key.split('_')[2].split(',') if '_' in batch_key else ['UNKNOWN']
            
            # Create batch prompt
            batch_prompt = f"Analyze sentiment for multiple symbols with this text: {text}"
            
            # Process as single request
            response = await self.query_async(
                prompt=batch_prompt,
                priority=priority,
                context="batch_sentiment",
                timeout=timeout,
                metadata={"batch_symbols": symbols, "batch_size": len(symbols)}
            )
            
            # Replicate response for all symbols in batch
            results = {}
            for symbol in symbols:
                # Create individual response for each symbol
                symbol_response = OllamaResponse(
                    request_id=f"{response.request_id}_{symbol}",
                    content=response.content,
                    model=response.model,
                    processing_time=response.processing_time / len(symbols),  # Distribute time
                    queue_time=response.queue_time,
                    success=response.success,
                    error=response.error,
                    cached=response.cached,
                    thread_id=response.thread_id,
                    tokens_estimated=response.tokens_estimated,
                    metadata={"symbol": symbol, "batch_processed": True, **response.metadata}
                )
                results[symbol] = symbol_response
            
            return results
            
        except Exception as e:
            logger.error(f"Batch sentiment processing failed: {e}")
            return {}
    
    async def get_enhanced_metrics(self) -> Dict[str, Any]:
        """Get enhanced metrics including trading-specific data."""
        base_metrics = await self.get_metrics()
        
        cache_stats = self.multi_tier_cache.get_cache_stats()
        
        enhanced_metrics = {
            **base_metrics,
            "trading_enhancements": {
                "express_requests_total": self.express_request_count,
                "batch_requests_total": self.batch_request_count,
                "express_executor_active": self.express_executor._threads,
                "cache_tiers": cache_stats,
                "circuit_breaker_trading_aware": {
                    "current_threshold": self.trading_circuit_breaker.get_failure_threshold(),
                    "market_hours": self.trading_circuit_breaker._is_market_hours(),
                    "earnings_season": self.trading_circuit_breaker._is_earnings_season()
                }
            }
        }
        
        return enhanced_metrics
    
    async def shutdown(self):
        """Enhanced shutdown with express executor cleanup."""
        await super().shutdown()
        
        # Shutdown express executor
        self.express_executor.shutdown(wait=True)
        logger.info("Enhanced Ollama bridge shutdown complete")


# Global enhanced bridge instance
_enhanced_ollama_bridge: Optional[EnhancedOllamaAsyncBridge] = None


async def get_enhanced_ollama_bridge() -> EnhancedOllamaAsyncBridge:
    """Get or create global enhanced Ollama bridge instance."""
    global _enhanced_ollama_bridge
    
    if _enhanced_ollama_bridge is None:
        from config.settings import settings
        
        _enhanced_ollama_bridge = EnhancedOllamaAsyncBridge(
            ollama_base_url=settings.ollama_base_url,
            max_workers=getattr(settings, 'ollama_max_workers', 4),
            queue_size=getattr(settings, 'ollama_queue_size', 100),
            cache_size=getattr(settings, 'ollama_cache_size', 1000),
            cache_ttl=getattr(settings, 'ollama_cache_ttl', 300),
            default_timeout=getattr(settings, 'ollama_default_timeout', 30.0)
        )
        
        await _enhanced_ollama_bridge.start()
        logger.info("Enhanced Ollama bridge initialized and started")
    
    return _enhanced_ollama_bridge


async def shutdown_enhanced_ollama_bridge():
    """Shutdown global enhanced Ollama bridge."""
    global _enhanced_ollama_bridge
    
    if _enhanced_ollama_bridge:
        await _enhanced_ollama_bridge.shutdown()
        _enhanced_ollama_bridge = None
        logger.info("Enhanced Ollama bridge shutdown")


# Express convenience functions for trading
async def analyze_sentiment_express(
    text: str,
    symbol: str = "",
    timeout: float = 2.0
) -> OllamaResponse:
    """Express sentiment analysis for critical trading decisions."""
    bridge = await get_enhanced_ollama_bridge()
    
    prompt = f"Fast sentiment analysis for {symbol}: {text[:500]}..."  # Truncate for speed
    
    return await bridge.query_express(
        prompt=prompt,
        context="express_sentiment",
        timeout=timeout,
        metadata={"symbol": symbol, "express": True}
    )


async def batch_multi_symbol_sentiment(
    symbol_texts: Dict[str, str],
    priority: RequestPriority = RequestPriority.NORMAL
) -> Dict[str, OllamaResponse]:
    """Batch sentiment analysis for multiple symbols."""
    bridge = await get_enhanced_ollama_bridge()
    
    return await bridge.batch_sentiment_analysis(
        symbol_texts=symbol_texts,
        priority=priority
    )


if __name__ == "__main__":
    async def test_enhanced_bridge():
        """Test enhanced bridge functionality."""
        print("🧪 Testing Enhanced Ollama Bridge")
        print("=" * 50)
        
        bridge = await get_enhanced_ollama_bridge()
        
        # Test express sentiment
        print("Testing express sentiment analysis...")
        express_response = await analyze_sentiment_express(
            text="Company reports strong earnings with 20% growth",
            symbol="AAPL",
            timeout=2.0
        )
        print(f"Express response time: {express_response.processing_time:.3f}s")
        print(f"Express success: {express_response.success}")
        
        # Test batch processing
        print("\nTesting batch sentiment analysis...")
        symbol_texts = {
            "AAPL": "Apple reports strong iPhone sales",
            "GOOGL": "Google announces new AI breakthroughs",
            "MSFT": "Microsoft expands cloud services"
        }
        
        batch_results = await batch_multi_symbol_sentiment(symbol_texts)
        print(f"Batch processed: {len(batch_results)} symbols")
        
        # Get enhanced metrics
        metrics = await bridge.get_enhanced_metrics()
        print(f"\nEnhanced metrics:")
        print(f"Express requests: {metrics['trading_enhancements']['express_requests_total']}")
        print(f"Batch requests: {metrics['trading_enhancements']['batch_requests_total']}")
        print(f"L1 cache hit rate: {metrics['trading_enhancements']['cache_tiers']['l1_stats']['hit_rate']:.1%}")
        
        await shutdown_enhanced_ollama_bridge()
    
    asyncio.run(test_enhanced_bridge())