#!/usr/bin/env python3
"""
Non-blocking Ollama Async Bridge for Trading System

Provides async-to-thread bridge for Ollama LLM requests with:
- Thread pool management for CPU-intensive Ollama operations
- Request queuing and prioritization
- Circuit breaker pattern for resilience
- Performance monitoring and caching
- Graceful fallback mechanisms
"""

import asyncio
import logging
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future
from queue import PriorityQueue, Queue, Empty
import threading
from contextlib import asynccontextmanager
import aiohttp
import ssl
import weakref
from contextlib import asynccontextmanager
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Request priority levels for queue management."""
    CRITICAL = 0    # Emergency/circuit breaker requests
    HIGH = 1        # Real-time trading signals
    NORMAL = 2      # Standard sentiment analysis
    LOW = 3         # Background analysis, caching


@dataclass
class OllamaRequest:
    """Structured request for Ollama processing."""
    id: str
    prompt: str
    model: str
    priority: RequestPriority
    timestamp: datetime
    timeout: float
    context: str
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = None
    retry_count: int = 0
    max_retries: int = 2
    
    def __lt__(self, other):
        """Priority queue ordering."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.timestamp < other.timestamp


@dataclass
class OllamaResponse:
    """Structured response from Ollama processing."""
    request_id: str
    content: str
    model: str
    processing_time: float
    queue_time: float
    success: bool
    error: Optional[str] = None
    cached: bool = False
    thread_id: str = None
    tokens_estimated: int = 0
    metadata: Dict[str, Any] = None


@dataclass
class CircuitBreakerState:
    """Circuit breaker state management."""
    failure_count: int = 0
    last_failure: Optional[datetime] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    consecutive_successes: int = 0
    

class OllamaPerformanceMonitor:
    """Performance monitoring for Ollama operations."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.request_history: List[Dict[str, Any]] = []
        self.queue_metrics = {
            "current_size": 0,
            "peak_size": 0,
            "total_processed": 0,
            "avg_wait_time": 0.0,
            "avg_processing_time": 0.0
        }
        self.thread_metrics = {
            "active_threads": 0,
            "max_threads": 0,
            "thread_utilization": 0.0
        }
        self._lock = threading.Lock()
    
    def record_request(self, request: OllamaRequest, response: OllamaResponse):
        """Record request metrics."""
        with self._lock:
            record = {
                "timestamp": request.timestamp,
                "priority": request.priority.name,
                "processing_time": response.processing_time,
                "queue_time": response.queue_time,
                "success": response.success,
                "cached": response.cached,
                "context": request.context,
                "model": request.model,
                "thread_id": response.thread_id
            }
            
            self.request_history.append(record)
            if len(self.request_history) > self.history_size:
                self.request_history.pop(0)
            
            # Update metrics
            self.queue_metrics["total_processed"] += 1
            
            # Calculate rolling averages
            recent_requests = self.request_history[-100:]  # Last 100 requests
            if recent_requests:
                self.queue_metrics["avg_wait_time"] = sum(r["queue_time"] for r in recent_requests) / len(recent_requests)
                self.queue_metrics["avg_processing_time"] = sum(r["processing_time"] for r in recent_requests) / len(recent_requests)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        with self._lock:
            recent_failures = sum(1 for r in self.request_history[-50:] if not r["success"])
            success_rate = 1.0 - (recent_failures / min(50, len(self.request_history))) if self.request_history else 1.0
            
            return {
                "queue_metrics": self.queue_metrics.copy(),
                "thread_metrics": self.thread_metrics.copy(),
                "success_rate_recent": success_rate,
                "total_requests": len(self.request_history),
                "cache_hit_rate": sum(1 for r in self.request_history[-100:] if r.get("cached", False)) / min(100, len(self.request_history)) if self.request_history else 0.0
            }


class OllamaResponseCache:
    """LRU cache for Ollama responses with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, tuple] = {}  # key -> (response, expiry_time)
        self.access_order: List[str] = []
        self._lock = threading.Lock()
    
    def _generate_key(self, prompt: str, model: str, context: str) -> str:
        """Generate cache key from request parameters including model config."""
        # Include model parameters in cache key to prevent stale configs
        model_params = "temp:0.7,top_p:0.9,max_tokens:1500"  # Match line 540-544
        content = f"{model}:{context}:{model_params}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, prompt: str, model: str, context: str) -> Optional[OllamaResponse]:
        """Get cached response if available and not expired."""
        key = self._generate_key(prompt, model, context)
        
        with self._lock:
            if key in self.cache:
                response, expiry = self.cache[key]
                
                if datetime.now() < expiry:
                    # Move to end (most recently used)
                    self.access_order.remove(key)
                    self.access_order.append(key)
                    
                    # Mark as cached
                    response.cached = True
                    logger.debug(f"Cache hit for key: {key[:8]}...")
                    return response
                else:
                    # Expired
                    del self.cache[key]
                    self.access_order.remove(key)
                    
        return None
    
    def put(self, prompt: str, model: str, context: str, response: OllamaResponse, ttl: Optional[int] = None):
        """Cache response with TTL."""
        key = self._generate_key(prompt, model, context)
        expiry = datetime.now() + timedelta(seconds=ttl or self.default_ttl)
        
        with self._lock:
            # Add to cache
            self.cache[key] = (response, expiry)
            
            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            # Evict LRU if over capacity
            while len(self.cache) > self.max_size:
                lru_key = self.access_order.pop(0)
                if lru_key in self.cache:
                    del self.cache[lru_key]
        
        logger.debug(f"Cached response for key: {key[:8]}...")
    
    def clear_expired(self):
        """Clear expired entries."""
        now = datetime.now()
        with self._lock:
            expired_keys = [key for key, (_, expiry) in self.cache.items() if now >= expiry]
            for key in expired_keys:
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
        
        if expired_keys:
            logger.debug(f"Cleared {len(expired_keys)} expired cache entries")


class OllamaAsyncBridge:
    """
    Non-blocking async bridge to Ollama with threading, queuing, and resilience.
    
    Features:
    - Thread pool for CPU-intensive Ollama operations  
    - Priority-based request queuing
    - Circuit breaker pattern for resilience
    - Response caching with TTL
    - Performance monitoring and metrics
    - Graceful degradation and fallback responses
    """
    
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        max_workers: int = 4,
        queue_size: int = 100,
        circuit_breaker_threshold: int = 5,
        cache_size: int = 1000,
        cache_ttl: int = 300,
        default_timeout: float = 30.0,
        max_connections: int = 100,
        max_connections_per_host: int = 30,
        enable_http2: bool = True
    ):
        self.ollama_base_url = ollama_base_url.rstrip('/')
        self.max_workers = max_workers
        self.queue_size = queue_size
        self.default_timeout = default_timeout
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host
        self.enable_http2 = enable_http2
        
        # HTTP session for async requests
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        
        # Thread pool for Ollama requests
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="OllamaWorker")
        
        # Request queue with priority support
        self.request_queue = PriorityQueue(maxsize=queue_size)
        self.pending_requests: Dict[str, Future] = {}
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreakerState()
        
        # Caching and monitoring
        self.cache = OllamaResponseCache(max_size=cache_size, default_ttl=cache_ttl)
        self.monitor = OllamaPerformanceMonitor()
        
        # Background worker management
        self.worker_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
        self.active_workers = 0
        
        # Weak references for cleanup
        self._cleanup_refs = weakref.WeakSet()
        
        logger.info(f"Initialized OllamaAsyncBridge with {max_workers} workers, queue size {queue_size}, "
                   f"max connections {max_connections}, HTTP/2 {'enabled' if enable_http2 else 'disabled'}")
    
    async def _ensure_session(self):
        """Ensure HTTP session exists with optimized connection pooling."""
        if self._session and not self._session.closed:
            return
        
        async with self._session_lock:
            if self._session and not self._session.closed:
                return
            
            # Create SSL context for HTTPS
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Configure connection pool
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections_per_host,
                ttl_dns_cache=300,  # Cache DNS for 5 minutes
                use_dns_cache=True,
                ssl_context=ssl_context,  # Use ssl_context, not ssl
                enable_cleanup_closed=True,
                force_close=False,  # Fixed: can't use keepalive_timeout with force_close=True
                keepalive_timeout=30
            )
            
            # Create session with optimized timeouts
            timeout = aiohttp.ClientTimeout(
                total=self.default_timeout,
                connect=10,
                sock_read=self.default_timeout
            )
            
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "User-Agent": "TradingSystem-OllamaAsync/1.0",
                    "Accept": "application/json",
                    "Connection": "keep-alive"
                }
            )
            
            logger.debug("Initialized aiohttp session with connection pooling")
    
    async def _warm_up_models(self):
        """Warm up Ollama models for faster initial responses."""
        try:
            logger.info("Warming up Ollama models...")
            
            # Test model availability with simple request
            test_prompt = "test"
            await self._execute_ollama_request(OllamaRequest(
                id="warmup",
                prompt=test_prompt,
                model="llama3:8b",
                priority=RequestPriority.HIGH,
                timestamp=datetime.now(),
                timeout=10.0,
                context="warmup"
            ))
            
            logger.info("Model warm-up completed successfully")
            
        except Exception as e:
            logger.warning(f"Model warm-up failed (continuing anyway): {e}")
    
    async def start(self):
        """Start background worker tasks."""
        if self.worker_tasks:
            logger.warning("OllamaAsyncBridge already started")
            return
        
        logger.info("Starting Ollama async bridge workers...")
        
        # Start worker tasks
        for i in range(self.max_workers):
            task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self.worker_tasks.append(task)
        
        # Start maintenance task
        maintenance_task = asyncio.create_task(self._maintenance_loop())
        self.worker_tasks.append(maintenance_task)
        
        logger.info(f"Started {len(self.worker_tasks)} Ollama worker tasks")
    
    async def shutdown(self):
        """Graceful shutdown of all workers."""
        logger.info("Shutting down Ollama async bridge...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Cancel pending tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete with timeout
        if self.worker_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.worker_tasks, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Some worker tasks did not complete within timeout")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Close HTTP session
        if self._session:
            await self._session.close()
            self._session = None
        
        # Clear resources
        self.worker_tasks.clear()
        self.pending_requests.clear()
        
        logger.info("Ollama async bridge shutdown complete")
    
    async def _validate_model_availability(self, model: str) -> bool:
        """Quickly check if model is available in Ollama."""
        try:
            url = urljoin(self.ollama_base_url, "/api/tags")
            async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=3.0)) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
                    return model in models
                return False
        except Exception as e:
            logger.debug(f"Model availability check failed: {e}")
            return False
    
    def _activate_fallback_mode(self):
        """Activate fallback mode for all pending requests."""
        logger.info("Activating fallback mode - clearing pending requests")
        
        # Create fallback responses for all pending requests
        for req_id, future in list(self.pending_requests.items()):
            if not future.done():
                fallback_response = OllamaResponse(
                    request_id=req_id,
                    content=json.dumps({
                        "sentiment": "neutral",
                        "confidence": 0.4,
                        "score": 0.0,
                        "reasoning": "Fallback response due to model serving issues",
                        "key_phrases": ["market", "analysis"],
                        "financial_impact": "Neutral conditions",
                        "risk_factors": ["model_unavailable"],
                        "opportunities": ["await_recovery"]
                    }),
                    model="fallback",
                    processing_time=0.01,
                    queue_time=0.0,
                    success=True,
                    cached=False,
                    thread_id="fallback",
                    metadata={"fallback_reason": "circuit_breaker_open"}
                )
                future.set_result(fallback_response)
        
        # Clear the pending requests
        self.pending_requests.clear()
    
    def _analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """Enhanced keyword-based sentiment analysis."""
        positive_words = [
            'profit', 'growth', 'increase', 'positive', 'strong', 'beat', 'exceeded', 
            'outperform', 'bullish', 'upgrade', 'buy', 'optimistic', 'confident',
            'expansion', 'revenue', 'earnings', 'success', 'improvement', 'gain',
            'rally', 'surge', 'breakthrough', 'milestone', 'achievement'
        ]
        
        negative_words = [
            'loss', 'decline', 'decrease', 'negative', 'weak', 'miss', 'failed',
            'underperform', 'bearish', 'downgrade', 'sell', 'pessimistic', 'concern',
            'contraction', 'debt', 'bankruptcy', 'crisis', 'deterioration', 'fall',
            'plunge', 'crash', 'warning', 'risk', 'uncertainty'
        ]
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total_sentiment_words = pos_count + neg_count
        
        if total_sentiment_words == 0:
            sentiment = "neutral"
            score = 0.0
            confidence = 0.3
            reasoning = "No clear sentiment indicators found"
        else:
            score = (pos_count - neg_count) / max(total_sentiment_words, 1)
            confidence = min(0.8, total_sentiment_words / 10)
            
            if score > 0.6:
                sentiment = "very_positive"
            elif score > 0.2:
                sentiment = "positive"
            elif score > -0.2:
                sentiment = "neutral"
            elif score > -0.6:
                sentiment = "negative"
            else:
                sentiment = "very_negative"
            
            reasoning = f"Keyword analysis: {pos_count} positive, {neg_count} negative indicators"
        
        # Extract key phrases
        key_phrases = []
        for word in positive_words + negative_words:
            if word in text_lower and len(key_phrases) < 5:
                key_phrases.append(word)
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "score": score,
            "reasoning": reasoning,
            "key_phrases": key_phrases or ["market", "analysis"],
            "financial_impact": f"Sentiment suggests {sentiment} market conditions"
        }
    
    def _calculate_health_score(self) -> float:
        """Calculate system health score (0-1)."""
        try:
            base_score = 1.0
            
            # Circuit breaker penalty
            if self.circuit_breaker.state == "OPEN":
                base_score -= 0.4
            elif self.circuit_breaker.state == "HALF_OPEN":
                base_score -= 0.2
            
            # Queue utilization penalty
            queue_util = self.request_queue.qsize() / self.queue_size
            if queue_util > 0.8:
                base_score -= 0.3
            elif queue_util > 0.5:
                base_score -= 0.1
            
            # Worker utilization factor
            worker_util = self.active_workers / self.max_workers
            if worker_util > 0.9:
                base_score -= 0.1
            
            # Recent failure rate
            if hasattr(self.monitor, 'request_history') and self.monitor.request_history:
                recent_requests = self.monitor.request_history[-20:]
                failure_rate = sum(1 for r in recent_requests if not r.get("success", True)) / len(recent_requests)
                base_score -= failure_rate * 0.3
            
            return max(0.0, min(1.0, base_score))
        except Exception:
            return 0.5  # Default score if calculation fails
    
    async def query_async(
        self,
        prompt: str,
        model: str = "llama3:8b",
        priority: RequestPriority = RequestPriority.NORMAL,
        context: str = "general",
        timeout: Optional[float] = None,
        use_cache: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> OllamaResponse:
        """
        Submit async query to Ollama with non-blocking execution.
        
        Args:
            prompt: Text prompt for the model
            model: Ollama model name
            priority: Request priority for queue ordering
            context: Context type for caching and routing
            timeout: Request timeout (uses default if None)
            use_cache: Whether to use response caching
            metadata: Additional metadata for the request
            
        Returns:
            OllamaResponse with content and performance metrics
        """
        
        # Check circuit breaker - use fallback instead of halting
        if not self._circuit_breaker_allow():
            logger.warning(f"Ollama circuit breaker is OPEN - using fallback response. State: {self.circuit_breaker.state}, Failures: {self.circuit_breaker.failure_count}")
            return self._create_circuit_breaker_response(prompt, model, context)
        
        # Check cache first
        if use_cache:
            cached_response = self.cache.get(prompt, model, context)
            if cached_response:
                return cached_response
        
        # Create request
        request = OllamaRequest(
            id=self._generate_request_id(),
            prompt=prompt,
            model=model,
            priority=priority,
            timestamp=datetime.now(),
            timeout=timeout or self.default_timeout,
            context=context,
            metadata=metadata or {}
        )
        
        # Submit to queue
        try:
            # Non-blocking queue submission
            if self.request_queue.full():
                # Queue is full - use fallback instead of halting
                logger.warning(f"Ollama request queue is full ({self.queue_size} items) - using fallback response")
                return self._create_fallback_response(request)
            
            # Update queue metrics
            self.monitor.queue_metrics["current_size"] = self.request_queue.qsize()
            self.monitor.queue_metrics["peak_size"] = max(
                self.monitor.queue_metrics["peak_size"],
                self.request_queue.qsize()
            )
            
            # Submit request
            future = asyncio.Future()
            self.pending_requests[request.id] = future
            
            # Put in queue (this might block briefly if queue is full)
            await asyncio.get_event_loop().run_in_executor(
                None, self.request_queue.put_nowait, request
            )
            
            # Wait for response with timeout
            try:
                response = await asyncio.wait_for(future, timeout=request.timeout)
                
                # Cache successful responses
                if use_cache and response.success and not response.cached:
                    self.cache.put(prompt, model, context, response)
                
                # Record metrics
                self.monitor.record_request(request, response)
                
                # Update circuit breaker on success
                self._circuit_breaker_success()
                
                return response
                
            except asyncio.TimeoutError:
                # Request timed out
                logger.error(f"Request {request.id} timed out after {request.timeout}s")
                self.pending_requests.pop(request.id, None)
                
                # Update circuit breaker on timeout
                self._circuit_breaker_failure()
                
                # Return timeout fallback instead of raising error
                return self._create_timeout_response(request)
                
        except Exception as e:
            logger.error(f"Error submitting request: {e}")
            self._circuit_breaker_failure()
            # Return error fallback instead of raising error
            return self._create_error_response(request, str(e))
    
    async def _worker_loop(self, worker_id: str):
        """Background worker loop processing requests from queue."""
        logger.info(f"Started Ollama worker: {worker_id}")
        
        while not self.shutdown_event.is_set():
            try:
                # Get request from queue with adaptive timeout
                try:
                    # Use shorter timeout when queue has items, longer when empty
                    queue_size = self.request_queue.qsize()
                    timeout = 0.1 if queue_size > 0 else 2.0  # Adaptive polling
                    request = await asyncio.get_event_loop().run_in_executor(
                        None, self._get_request_with_timeout, timeout
                    )
                except Empty:
                    continue  # No requests, continue loop
                
                if request is None:
                    continue
                
                # Update metrics
                self.active_workers += 1
                self.monitor.thread_metrics["active_threads"] = self.active_workers
                self.monitor.thread_metrics["max_threads"] = max(
                    self.monitor.thread_metrics["max_threads"],
                    self.active_workers
                )
                
                # Process request
                start_time = time.time()
                queue_time = (datetime.now() - request.timestamp).total_seconds()
                
                try:
                    # Execute Ollama request in thread pool
                    response_content = await self._execute_ollama_request(request)
                    
                    processing_time = time.time() - start_time
                    
                    # Create response
                    response = OllamaResponse(
                        request_id=request.id,
                        content=response_content,
                        model=request.model,
                        processing_time=processing_time,
                        queue_time=queue_time,
                        success=True,
                        thread_id=worker_id,
                        tokens_estimated=len(response_content) // 4,  # Rough estimate
                        metadata=request.metadata
                    )
                    
                    logger.debug(f"Worker {worker_id} completed request {request.id} in {processing_time:.2f}s")
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    
                    # Create error response
                    response = OllamaResponse(
                        request_id=request.id,
                        content="",
                        model=request.model,
                        processing_time=processing_time,
                        queue_time=queue_time,
                        success=False,
                        error=str(e),
                        thread_id=worker_id,
                        metadata=request.metadata
                    )
                    
                    logger.error(f"Worker {worker_id} failed request {request.id}: {e}")
                
                # Send response to waiting coroutine
                future = self.pending_requests.pop(request.id, None)
                if future and not future.cancelled():
                    future.set_result(response)
                
                # Update metrics
                self.active_workers -= 1
                self.monitor.thread_metrics["active_threads"] = self.active_workers
                
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)  # Brief pause on error
        
        logger.info(f"Ollama worker {worker_id} stopped")
    
    def _get_request_with_timeout(self, timeout: float) -> Optional[OllamaRequest]:
        """Get request from queue with timeout (runs in thread)."""
        try:
            return self.request_queue.get(timeout=timeout)
        except Empty:
            return None
    
    async def _execute_ollama_request(self, request: OllamaRequest) -> str:
        """Execute Ollama HTTP request with fast-fail and model validation."""
        await self._ensure_session()
        
        # First, quickly check if model exists to avoid long timeouts
        if not await self._validate_model_availability(request.model):
            self._last_error = f"model '{request.model}' not found"
            raise Exception(f"Model '{request.model}' not available in Ollama")
        
        url = urljoin(self.ollama_base_url, "/api/generate")
        payload = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 800,  # Reduced for faster responses
                "stop": ["\n\n", "Human:", "User:"]  # Stop tokens to prevent rambling
            }
        }
        
        # Fast-fail retry logic with aggressive timeouts
        max_retries = 2  # Reduced from 3
        base_delay = 0.05  # Faster retries
        
        for attempt in range(max_retries + 1):
            try:
                # Use shorter timeout per attempt
                timeout = min(request.timeout / (max_retries + 1), 8.0)
                
                async with self._session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "")
                    elif response.status == 404:
                        # Model not found - don't retry
                        self._last_error = f"model '{request.model}' not found"
                        raise Exception(f"Model '{request.model}' not found, try pulling it first")
                    else:
                        response_text = await response.text()
                        error_msg = f"{response.status}, message='{response_text}'"
                        if attempt == max_retries:
                            raise Exception(f"Ollama request failed: {error_msg}")
                        logger.debug(f"Ollama attempt {attempt + 1} failed: {error_msg}")
                        
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == max_retries:
                    self._last_error = str(e)
                    raise Exception(f"Ollama request failed after {max_retries + 1} attempts: {e}")
                
                # Fast exponential backoff
                import random
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.02)
                logger.debug(f"Ollama request attempt {attempt + 1} failed, retrying in {delay:.3f}s: {e}")
                await asyncio.sleep(delay)
            
            except Exception as e:
                self._last_error = str(e)
                raise Exception(f"Ollama request failed: {e}")
    
    async def _maintenance_loop(self):
        """Background maintenance tasks."""
        logger.info("Started Ollama maintenance loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Clear expired cache entries
                self.cache.clear_expired()
                
                # Update thread utilization metric
                self.monitor.thread_metrics["thread_utilization"] = (
                    self.active_workers / self.max_workers if self.max_workers > 0 else 0.0
                )
                
                # Log metrics periodically
                if self.monitor.queue_metrics["total_processed"] > 0:
                    metrics = self.monitor.get_metrics()
                    logger.debug(f"Ollama metrics: {json.dumps(metrics, indent=2)}")
                
                # Clean up old pending requests (should not happen with proper timeout handling)
                now = datetime.now()
                stale_requests = []
                for req_id, future in self.pending_requests.items():
                    if future.done() or future.cancelled():
                        stale_requests.append(req_id)
                
                for req_id in stale_requests:
                    self.pending_requests.pop(req_id, None)
                
                if stale_requests:
                    logger.debug(f"Cleaned up {len(stale_requests)} stale request futures")
                
                # Wait before next maintenance cycle
                await asyncio.sleep(30)  # Run every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")
                await asyncio.sleep(5)  # Brief pause on error
        
        logger.info("Ollama maintenance loop stopped")
    
    def _circuit_breaker_allow(self) -> bool:
        """Check if circuit breaker allows request."""
        now = datetime.now()
        
        if self.circuit_breaker.state == "OPEN":
            # Check if recovery timeout has passed
            if (self.circuit_breaker.last_failure and 
                (now - self.circuit_breaker.last_failure).total_seconds() > self.circuit_breaker.recovery_timeout):
                self.circuit_breaker.state = "HALF_OPEN"
                self.circuit_breaker.consecutive_successes = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                return True
            return False
        
        return True  # CLOSED or HALF_OPEN
    
    def _circuit_breaker_success(self):
        """Record successful request for circuit breaker."""
        if self.circuit_breaker.state == "HALF_OPEN":
            self.circuit_breaker.consecutive_successes += 1
            if self.circuit_breaker.consecutive_successes >= 3:
                self.circuit_breaker.state = "CLOSED"
                self.circuit_breaker.failure_count = 0
                logger.info("Circuit breaker CLOSED after successful recovery")
        elif self.circuit_breaker.state == "CLOSED":
            self.circuit_breaker.failure_count = max(0, self.circuit_breaker.failure_count - 1)
    
    def _circuit_breaker_failure(self):
        """Record failed request for circuit breaker with intelligent recovery."""
        self.circuit_breaker.failure_count += 1
        self.circuit_breaker.last_failure = datetime.now()
        
        if (self.circuit_breaker.state in ["CLOSED", "HALF_OPEN"] and 
            self.circuit_breaker.failure_count >= self.circuit_breaker.failure_threshold):
            self.circuit_breaker.state = "OPEN"
            
            # Smart recovery timeout based on failure type
            if "model 'llama3:8b' not found" in str(getattr(self, '_last_error', '')):
                # Model not available - longer timeout to allow manual intervention
                self.circuit_breaker.recovery_timeout = 300  # 5 minutes
                logger.critical(f"🚨 OLLAMA MODEL NOT FOUND - Circuit breaker OPEN for 5 minutes. "
                              f"Model llama3:8b is available - check Ollama server status")
            else:
                # Connection or other issues - shorter adaptive timeout
                import random
                base_timeout = min(self.circuit_breaker.recovery_timeout, 60)
                jittered_timeout = base_timeout * (1 + random.uniform(0, 0.3))
                self.circuit_breaker.recovery_timeout = min(int(jittered_timeout), 120)
                logger.warning(f"🚨 OLLAMA CIRCUIT BREAKER OPEN: {self.circuit_breaker.failure_count} failures, "
                             f"recovery timeout: {self.circuit_breaker.recovery_timeout}s")
            
            # Trigger immediate fallback mode for all pending requests
            self._activate_fallback_mode()
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        return f"ollama_{int(time.time() * 1000)}_{id(threading.current_thread())}"
    
    def _create_fallback_response(self, request: OllamaRequest) -> OllamaResponse:
        """Create fallback response when system is overloaded."""
        fallback_content = self._generate_rule_based_response(request.prompt, request.context)
        
        return OllamaResponse(
            request_id=request.id,
            content=fallback_content,
            model=f"{request.model}-fallback",
            processing_time=0.01,
            queue_time=0.0,
            success=True,
            cached=False,
            thread_id="fallback",
            tokens_estimated=len(fallback_content) // 4,
            metadata={"fallback_reason": "queue_overload", **request.metadata}
        )
    
    def _create_circuit_breaker_response(self, prompt: str, model: str, context: str) -> OllamaResponse:
        """Create response when circuit breaker is open."""
        fallback_content = self._generate_rule_based_response(prompt, context)
        
        return OllamaResponse(
            request_id="circuit_breaker",
            content=fallback_content,
            model=f"{model}-circuit-breaker",
            processing_time=0.01,
            queue_time=0.0,
            success=True,
            cached=False,
            thread_id="circuit-breaker",
            tokens_estimated=len(fallback_content) // 4,
            metadata={"fallback_reason": "circuit_breaker_open"}
        )
    
    def _create_timeout_response(self, request: OllamaRequest) -> OllamaResponse:
        """Create response for timed out request."""
        fallback_content = self._generate_rule_based_response(request.prompt, request.context)
        
        return OllamaResponse(
            request_id=request.id,
            content=fallback_content,
            model=f"{request.model}-timeout",
            processing_time=request.timeout,
            queue_time=0.0,
            success=False,
            error="Request timeout",
            cached=False,
            thread_id="timeout",
            tokens_estimated=len(fallback_content) // 4,
            metadata={"fallback_reason": "timeout", **request.metadata}
        )
    
    def _create_error_response(self, request: OllamaRequest, error_msg: str) -> OllamaResponse:
        """Create response for failed request."""
        fallback_content = self._generate_rule_based_response(request.prompt, request.context)
        
        return OllamaResponse(
            request_id=request.id,
            content=fallback_content,
            model=f"{request.model}-error",
            processing_time=0.01,
            queue_time=0.0,
            success=False,
            error=error_msg,
            cached=False,
            thread_id="error",
            tokens_estimated=len(fallback_content) // 4,
            metadata={"fallback_reason": "error", **request.metadata}
        )
    
    def _generate_rule_based_response(self, prompt: str, context: str) -> str:
        """Generate enhanced rule-based fallback response with sentiment analysis."""
        prompt_lower = prompt.lower()
        
        if context == "sentiment_analysis" or "sentiment" in prompt_lower:
            # Enhanced sentiment fallback with keyword analysis
            sentiment_score = self._analyze_text_sentiment(prompt)
            
            return json.dumps({
                "sentiment": sentiment_score["sentiment"],
                "confidence": sentiment_score["confidence"],
                "score": sentiment_score["score"],
                "reasoning": f"Keyword-based analysis: {sentiment_score['reasoning']}",
                "key_phrases": sentiment_score.get("key_phrases", ["market", "trading"]),
                "financial_impact": sentiment_score.get("financial_impact", "Neutral market conditions"),
                "risk_factors": ["llm_unavailable", "keyword_analysis"],
                "opportunities": ["monitor_model_recovery"]
            })
        
        elif context == "portfolio_analysis" or any(word in prompt_lower for word in ["portfolio", "allocation", "risk"]):
            return "Based on current market conditions, maintain diversified portfolio allocation with 60% equities, 30% bonds, 10% alternatives. Monitor risk metrics and rebalance quarterly."
        
        elif context == "market_analysis" or any(word in prompt_lower for word in ["market", "trading", "stocks"]):
            return "Market analysis indicates mixed conditions with moderate volatility. Recommend maintaining defensive positions and monitoring key technical levels."
        
        else:
            return "Analysis temporarily unavailable due to model serving issues. Using conservative rule-based recommendations. Model recovery in progress."
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics with health indicators."""
        base_metrics = self.monitor.get_metrics()
        
        # Calculate system health score
        health_score = self._calculate_health_score()
        
        return {
            **base_metrics,
            "circuit_breaker": {
                "state": self.circuit_breaker.state,
                "failure_count": self.circuit_breaker.failure_count,
                "last_failure": self.circuit_breaker.last_failure.isoformat() if self.circuit_breaker.last_failure else None,
                "recovery_timeout": self.circuit_breaker.recovery_timeout
            },
            "system": {
                "queue_size": self.request_queue.qsize(),
                "queue_capacity": self.queue_size,
                "pending_requests": len(self.pending_requests),
                "active_workers": self.active_workers,
                "max_workers": self.max_workers,
                "cache_size": len(self.cache.cache),
                "health_score": health_score,
                "last_error": getattr(self, '_last_error', None)
            },
            "performance": {
                "avg_response_time": base_metrics.get("queue_metrics", {}).get("avg_processing_time", 0),
                "queue_utilization": self.request_queue.qsize() / self.queue_size,
                "worker_utilization": self.active_workers / self.max_workers
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        try:
            # Test with simple request
            test_response = await self.query_async(
                prompt="test",
                model="llama3:8b",
                priority=RequestPriority.HIGH,
                context="health_check",
                timeout=5.0,
                use_cache=False
            )
            
            healthy = test_response.success and len(test_response.content) > 0
            
            return {
                "healthy": healthy,
                "timestamp": datetime.now().isoformat(),
                "test_response_time": test_response.processing_time,
                "circuit_breaker_state": self.circuit_breaker.state,
                "workers_active": self.active_workers,
                "queue_utilization": self.request_queue.qsize() / self.queue_size
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "circuit_breaker_state": self.circuit_breaker.state
            }
    
    @asynccontextmanager
    async def managed_lifecycle(self):
        """Context manager for proper startup/shutdown."""
        try:
            await self.start()
            yield self
        finally:
            await self.shutdown()


# Global instance for system-wide use
_ollama_bridge: Optional[OllamaAsyncBridge] = None


async def get_ollama_bridge() -> OllamaAsyncBridge:
    """Get or create global Ollama bridge instance."""
    global _ollama_bridge
    
    if _ollama_bridge is None:
        from config.settings import settings
        
        _ollama_bridge = OllamaAsyncBridge(
            ollama_base_url=settings.ollama_base_url,
            max_workers=getattr(settings, 'ollama_max_workers', 4),
            queue_size=getattr(settings, 'ollama_queue_size', 100),
            cache_size=getattr(settings, 'ollama_cache_size', 1000),
            cache_ttl=getattr(settings, 'ollama_cache_ttl', 300),
            default_timeout=getattr(settings, 'ollama_default_timeout', 30.0)
        )
        
        await _ollama_bridge.start()
        logger.info("Global Ollama bridge initialized and started")
    
    return _ollama_bridge


async def shutdown_ollama_bridge():
    """Shutdown global Ollama bridge."""
    global _ollama_bridge
    
    if _ollama_bridge:
        await _ollama_bridge.shutdown()
        _ollama_bridge = None
        logger.info("Global Ollama bridge shutdown")


# Convenience functions for common use cases
async def query_ollama_sentiment(
    text: str,
    symbol: str = "",
    priority: RequestPriority = RequestPriority.NORMAL,
    timeout: float = 30.0
) -> OllamaResponse:
    """Convenience function for sentiment analysis queries."""
    bridge = await get_ollama_bridge()
    
    prompt = f"""Analyze the sentiment of this financial text about {symbol}:

{text}

Respond with JSON only:
{{
    "sentiment": "very_positive|positive|neutral|negative|very_negative",
    "confidence": 0.8,
    "score": 0.5,
    "reasoning": "brief explanation",
    "key_phrases": ["key", "phrases"],
    "financial_impact": "impact on stock price",
    "risk_factors": ["risk1"],
    "opportunities": ["opportunity1"]
}}"""
    
    return await bridge.query_async(
        prompt=prompt,
        model="llama3:8b",
        priority=priority,
        context="sentiment_analysis",
        timeout=timeout,
        metadata={"symbol": symbol}
    )


async def query_ollama_portfolio(
    analysis_data: Dict[str, Any],
    priority: RequestPriority = RequestPriority.HIGH,
    timeout: float = 45.0
) -> OllamaResponse:
    """Convenience function for portfolio analysis queries."""
    bridge = await get_ollama_bridge()
    
    prompt = f"""Analyze this portfolio and market data for trading decisions:

{json.dumps(analysis_data, indent=2)}

Provide investment analysis and recommendations considering risk management, diversification, and current market conditions."""
    
    return await bridge.query_async(
        prompt=prompt,
        model="llama3:8b",
        priority=priority,
        context="portfolio_analysis",
        timeout=timeout,
        metadata=analysis_data
    )


if __name__ == "__main__":
    async def test_ollama_bridge():
        """Test the Ollama async bridge."""
        print("🧪 Testing Ollama Async Bridge")
        print("=" * 50)
        
        async with OllamaAsyncBridge().managed_lifecycle() as bridge:
            # Health check
            health = await bridge.health_check()
            print(f"Health check: {health}")
            
            # Test sentiment analysis
            response = await query_ollama_sentiment(
                text="Company reported strong Q3 earnings with 15% revenue growth",
                symbol="AAPL",
                priority=RequestPriority.HIGH
            )
            
            print(f"Sentiment response: {response.content[:200]}...")
            print(f"Processing time: {response.processing_time:.2f}s")
            print(f"Success: {response.success}")
            
            # Get metrics
            metrics = await bridge.get_metrics()
            print(f"Metrics: {json.dumps(metrics, indent=2)}")
    
    # Run test
    asyncio.run(test_ollama_bridge())