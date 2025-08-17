#!/usr/bin/env python3
"""
Strict Rate Limiter for External APIs
Prevents 429 errors by enforcing conservative rate limits that are well below API limits.

Rate Limits (Conservative):
- News API: 2 requests/minute (vs 1000/day limit)
- Alpha Vantage: 1 request/minute (vs 5/minute limit)  
- Any API: Configurable with exponential backoff

Design Philosophy:
- NEVER exceed rate limits - system should halt if limits hit
- Conservative limits to prevent abuse
- Global rate limiting across all components
- Fail fast when rate limits would be exceeded
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class APIProvider(Enum):
    """Supported API providers with their rate limits."""
    NEWS_API = "news_api"
    ALPHA_VANTAGE = "alpha_vantage" 
    FINNHUB = "finnhub"
    CUSTOM = "custom"


@dataclass
class RateLimit:
    """Rate limit configuration for an API provider."""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_size: int = 1  # Max requests in burst
    cooldown_seconds: int = 60  # Cooldown after hitting limit


class StrictRateLimiter:
    """
    Strict rate limiter that prevents API abuse and 429 errors.
    
    Key Features:
    - Global rate limiting across all system components
    - Conservative limits well below API maximums
    - Immediate rejection when limits would be exceeded
    - No queuing or retry logic - fail fast design
    """
    
    # Conservative rate limits (well below API limits)
    RATE_LIMITS = {
        APIProvider.NEWS_API: RateLimit(
            requests_per_minute=2,   # Very conservative (API allows 1000/day)
            requests_per_hour=30,
            requests_per_day=500,
            burst_size=1,
            cooldown_seconds=30
        ),
        APIProvider.ALPHA_VANTAGE: RateLimit(
            requests_per_minute=1,   # Conservative (API allows 5/minute)
            requests_per_hour=50,
            requests_per_day=400,
            burst_size=1,
            cooldown_seconds=60
        ),
        APIProvider.FINNHUB: RateLimit(
            requests_per_minute=30,  # Conservative (API allows 60/minute)
            requests_per_hour=1000,
            requests_per_day=10000,
            burst_size=5,
            cooldown_seconds=10
        )
    }
    
    def __init__(self):
        """Initialize the strict rate limiter."""
        self._request_history: Dict[APIProvider, list] = {}
        self._last_request: Dict[APIProvider, float] = {}
        self._lock = threading.Lock()
        self._cooldown_until: Dict[APIProvider, float] = {}
        
        logger.info("🚦 Strict rate limiter initialized")
        for provider, limit in self.RATE_LIMITS.items():
            logger.info(f"   {provider.value}: {limit.requests_per_minute}/min, {limit.requests_per_hour}/hour")
    
    def can_make_request(self, provider: APIProvider, endpoint: str = "") -> bool:
        """
        Check if a request can be made without exceeding rate limits.
        
        Args:
            provider: API provider to check
            endpoint: Specific endpoint (for logging)
            
        Returns:
            True if request is allowed, False if it would exceed limits
        """
        with self._lock:
            current_time = time.time()
            
            # Check if in cooldown period
            if provider in self._cooldown_until:
                if current_time < self._cooldown_until[provider]:
                    cooldown_remaining = self._cooldown_until[provider] - current_time
                    logger.warning(f"🚦 Rate limit cooldown active for {provider.value}: {cooldown_remaining:.1f}s remaining")
                    return False
                else:
                    # Cooldown expired
                    del self._cooldown_until[provider]
            
            if provider not in self.RATE_LIMITS:
                logger.warning(f"Unknown provider {provider}, allowing request")
                return True
            
            limit = self.RATE_LIMITS[provider]
            
            # Initialize history if needed
            if provider not in self._request_history:
                self._request_history[provider] = []
            
            # Clean old requests from history
            cutoff_minute = current_time - 60
            cutoff_hour = current_time - 3600
            cutoff_day = current_time - 86400
            
            self._request_history[provider] = [
                req_time for req_time in self._request_history[provider]
                if req_time > cutoff_day
            ]
            
            # Count recent requests
            requests_last_minute = sum(1 for req_time in self._request_history[provider] if req_time > cutoff_minute)
            requests_last_hour = sum(1 for req_time in self._request_history[provider] if req_time > cutoff_hour)
            requests_last_day = len(self._request_history[provider])
            
            # Check all rate limits
            if requests_last_minute >= limit.requests_per_minute:
                logger.error(f"🚦 RATE LIMIT EXCEEDED - {provider.value}: {requests_last_minute}/{limit.requests_per_minute} per minute")
                self._trigger_cooldown(provider)
                return False
            
            if requests_last_hour >= limit.requests_per_hour:
                logger.error(f"🚦 RATE LIMIT EXCEEDED - {provider.value}: {requests_last_hour}/{limit.requests_per_hour} per hour")
                self._trigger_cooldown(provider)
                return False
            
            if requests_last_day >= limit.requests_per_day:
                logger.error(f"🚦 RATE LIMIT EXCEEDED - {provider.value}: {requests_last_day}/{limit.requests_per_day} per day")
                self._trigger_cooldown(provider)
                return False
            
            # Check burst protection
            if provider in self._last_request:
                time_since_last = current_time - self._last_request[provider]
                min_interval = 60.0 / limit.requests_per_minute
                
                if time_since_last < min_interval:
                    logger.warning(f"🚦 Burst protection for {provider.value}: {time_since_last:.1f}s < {min_interval:.1f}s")
                    return False
            
            return True
    
    def record_request(self, provider: APIProvider, endpoint: str = "") -> None:
        """
        Record that a request was made to update rate limiting counters.
        
        Args:
            provider: API provider that was called
            endpoint: Specific endpoint (for logging)
        """
        with self._lock:
            current_time = time.time()
            
            if provider not in self._request_history:
                self._request_history[provider] = []
            
            self._request_history[provider].append(current_time)
            self._last_request[provider] = current_time
            
            logger.debug(f"🚦 Recorded request to {provider.value} {endpoint}")
    
    def _trigger_cooldown(self, provider: APIProvider) -> None:
        """Trigger cooldown period after rate limit exceeded."""
        cooldown_seconds = self.RATE_LIMITS[provider].cooldown_seconds
        self._cooldown_until[provider] = time.time() + cooldown_seconds
        
        logger.error(f"🚦 COOLDOWN TRIGGERED for {provider.value}: {cooldown_seconds}s")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate limiting status for all providers."""
        status = {}
        current_time = time.time()
        
        with self._lock:
            for provider, limit in self.RATE_LIMITS.items():
                if provider not in self._request_history:
                    self._request_history[provider] = []
                
                # Count recent requests
                cutoff_minute = current_time - 60
                cutoff_hour = current_time - 3600
                cutoff_day = current_time - 86400
                
                requests_last_minute = sum(1 for req_time in self._request_history[provider] if req_time > cutoff_minute)
                requests_last_hour = sum(1 for req_time in self._request_history[provider] if req_time > cutoff_hour)
                requests_last_day = sum(1 for req_time in self._request_history[provider] if req_time > cutoff_day)
                
                cooldown_remaining = 0
                if provider in self._cooldown_until:
                    cooldown_remaining = max(0, self._cooldown_until[provider] - current_time)
                
                status[provider.value] = {
                    "requests_last_minute": f"{requests_last_minute}/{limit.requests_per_minute}",
                    "requests_last_hour": f"{requests_last_hour}/{limit.requests_per_hour}",
                    "requests_last_day": f"{requests_last_day}/{limit.requests_per_day}",
                    "cooldown_remaining": f"{cooldown_remaining:.1f}s" if cooldown_remaining > 0 else "None",
                    "can_make_request": self.can_make_request(provider)
                }
        
        return status
    
    def enforce_rate_limit(self, provider: APIProvider, endpoint: str = "") -> None:
        """
        Enforce rate limit - raise exception if request not allowed.
        
        Args:
            provider: API provider to check
            endpoint: Specific endpoint (for error message)
            
        Raises:
            RuntimeError: If rate limit would be exceeded
        """
        if not self.can_make_request(provider, endpoint):
            raise RuntimeError(f"Rate limit exceeded for {provider.value} {endpoint} - system halting to prevent 429 errors")
        
        self.record_request(provider, endpoint)


# Global rate limiter instance
_global_rate_limiter: Optional[StrictRateLimiter] = None


def get_rate_limiter() -> StrictRateLimiter:
    """Get global rate limiter instance."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = StrictRateLimiter()
    return _global_rate_limiter


def rate_limited_request(provider: APIProvider, endpoint: str = ""):
    """
    Decorator to enforce rate limiting on API calls.
    
    Usage:
        @rate_limited_request(APIProvider.NEWS_API, "/everything")
        async def fetch_news():
            # API call here
    """
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            rate_limiter = get_rate_limiter()
            rate_limiter.enforce_rate_limit(provider, endpoint)
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Convenience functions for specific APIs
def enforce_news_api_limit(endpoint: str = ""):
    """Enforce News API rate limit."""
    get_rate_limiter().enforce_rate_limit(APIProvider.NEWS_API, endpoint)


def enforce_alpha_vantage_limit(endpoint: str = ""):
    """Enforce Alpha Vantage rate limit.""" 
    get_rate_limiter().enforce_rate_limit(APIProvider.ALPHA_VANTAGE, endpoint)


def enforce_finnhub_limit(endpoint: str = ""):
    """Enforce Finnhub rate limit."""
    get_rate_limiter().enforce_rate_limit(APIProvider.FINNHUB, endpoint)


if __name__ == "__main__":
    # Test the rate limiter
    limiter = StrictRateLimiter()
    
    print("Testing rate limiter...")
    
    # Test normal requests
    for i in range(3):
        if limiter.can_make_request(APIProvider.NEWS_API):
            limiter.record_request(APIProvider.NEWS_API, f"/test{i}")
            print(f"Request {i}: Allowed")
        else:
            print(f"Request {i}: BLOCKED")
        time.sleep(0.5)
    
    print("\nCurrent status:")
    status = limiter.get_status()
    for provider, info in status.items():
        print(f"{provider}: {info}")