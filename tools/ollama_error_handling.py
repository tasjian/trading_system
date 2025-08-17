#!/usr/bin/env python3
"""
Advanced Error Handling and Resilience for Ollama Trading System

Implements sophisticated error handling patterns including:
- Jittered exponential backoff
- Circuit breaker with half-open state
- Request classification and priority-based retry policies
- Error aggregation and trend analysis
- Graceful degradation strategies
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Error categories for classification."""
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error" 
    RATE_LIMIT_ERROR = "rate_limit_error"
    MODEL_ERROR = "model_error"
    SERVER_ERROR = "server_error"
    CLIENT_ERROR = "client_error"
    CIRCUIT_BREAKER_ERROR = "circuit_breaker_error"
    QUEUE_FULL_ERROR = "queue_full_error"
    AUTHENTICATION_ERROR = "authentication_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ErrorEvent:
    """Individual error event record."""
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    context: Dict[str, Any]
    request_id: Optional[str] = None
    retry_attempt: int = 0
    recovery_action: Optional[str] = None


@dataclass
class RetryPolicy:
    """Retry policy configuration."""
    max_attempts: int = 3
    base_delay: float = 0.1
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter_range: float = 0.1
    backoff_multiplier: float = 1.0
    retry_categories: List[ErrorCategory] = field(default_factory=list)


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3
    success_threshold: int = 2
    monitoring_window: int = 300


class AdvancedCircuitBreaker:
    """Advanced circuit breaker with exponential backoff and jitter."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        self.failure_history = deque(maxlen=100)
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if not self._allow_request():
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
            
        except Exception as e:
            await self._on_failure(e)
            raise
    
    def _allow_request(self) -> bool:
        """Check if request should be allowed through circuit breaker."""
        now = datetime.now()
        
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            # Check if recovery timeout has passed
            if (self.last_failure_time and 
                (now - self.last_failure_time).total_seconds() >= self.config.recovery_timeout):
                self.state = "HALF_OPEN"
                self.half_open_calls = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                return True
            return False
        elif self.state == "HALF_OPEN":
            # Allow limited requests in half-open state
            return self.half_open_calls < self.config.half_open_max_calls
        
        return False
    
    async def _on_success(self):
        """Handle successful request."""
        async with self._lock:
            if self.state == "HALF_OPEN":
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = "CLOSED"
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info("Circuit breaker CLOSED after successful recovery")
            elif self.state == "CLOSED":
                # Reset failure count on successful request
                self.failure_count = max(0, self.failure_count - 1)
    
    async def _on_failure(self, error: Exception):
        """Handle failed request."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            self.failure_history.append((self.last_failure_time, str(error)))
            
            if self.state == "HALF_OPEN":
                self.half_open_calls += 1
                # If we fail in half-open, go back to open
                self.state = "OPEN"
                self.success_count = 0
                # Apply exponential backoff with jitter
                self._apply_exponential_backoff()
                logger.warning("Circuit breaker back to OPEN state after half-open failure")
            
            elif self.state == "CLOSED" and self.failure_count >= self.config.failure_threshold:
                self.state = "OPEN"
                self._apply_exponential_backoff()
                logger.critical(f"Circuit breaker OPEN: {self.failure_count} failures")
    
    def _apply_exponential_backoff(self):
        """Apply exponential backoff with jitter to recovery timeout."""
        base_timeout = self.config.recovery_timeout
        
        # Exponential backoff based on failure count
        backoff_multiplier = min(2 ** (self.failure_count - self.config.failure_threshold), 8)
        
        # Add jitter (±25%)
        jitter = random.uniform(0.75, 1.25)
        
        self.config.recovery_timeout = min(int(base_timeout * backoff_multiplier * jitter), 300)
        
        logger.debug(f"Applied exponential backoff: recovery timeout = {self.config.recovery_timeout}s")
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current circuit breaker state information."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "recovery_timeout": self.config.recovery_timeout,
            "half_open_calls": self.half_open_calls,
            "recent_failures": len(self.failure_history)
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class ErrorClassifier:
    """Classify errors into categories and severities."""
    
    def __init__(self):
        self.error_patterns = {
            # Network errors
            r"Connection.*refused|Network.*unreachable|DNS.*resolution": (ErrorCategory.NETWORK_ERROR, ErrorSeverity.HIGH),
            r"timeout|timed out": (ErrorCategory.TIMEOUT_ERROR, ErrorSeverity.MEDIUM),
            
            # Rate limiting
            r"rate limit|too many requests|429": (ErrorCategory.RATE_LIMIT_ERROR, ErrorSeverity.MEDIUM),
            
            # Server errors
            r"500|502|503|504|Internal.*Error": (ErrorCategory.SERVER_ERROR, ErrorSeverity.HIGH),
            r"model.*not.*found|model.*loading": (ErrorCategory.MODEL_ERROR, ErrorSeverity.HIGH),
            
            # Client errors
            r"400|401|403|invalid.*request": (ErrorCategory.CLIENT_ERROR, ErrorSeverity.LOW),
            r"authentication|unauthorized": (ErrorCategory.AUTHENTICATION_ERROR, ErrorSeverity.CRITICAL),
            
            # System errors
            r"queue.*full|capacity.*exceeded": (ErrorCategory.QUEUE_FULL_ERROR, ErrorSeverity.MEDIUM),
            r"circuit.*breaker": (ErrorCategory.CIRCUIT_BREAKER_ERROR, ErrorSeverity.HIGH),
        }
    
    def classify_error(self, error: Exception, context: Dict[str, Any] = None) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Classify an error into category and severity."""
        error_str = str(error).lower()
        
        import re
        for pattern, (category, severity) in self.error_patterns.items():
            if re.search(pattern, error_str):
                return category, severity
        
        # Check context for additional clues
        if context:
            if context.get("status_code", 0) >= 500:
                return ErrorCategory.SERVER_ERROR, ErrorSeverity.HIGH
            elif context.get("status_code", 0) >= 400:
                return ErrorCategory.CLIENT_ERROR, ErrorSeverity.LOW
        
        return ErrorCategory.UNKNOWN_ERROR, ErrorSeverity.MEDIUM


class AdvancedRetryHandler:
    """Advanced retry handler with intelligent backoff and error classification."""
    
    def __init__(self):
        self.error_classifier = ErrorClassifier()
        self.retry_policies = self._create_default_policies()
        self.error_history = deque(maxlen=1000)
        self.circuit_breakers: Dict[str, AdvancedCircuitBreaker] = {}
    
    def _create_default_policies(self) -> Dict[ErrorCategory, RetryPolicy]:
        """Create default retry policies for different error categories."""
        return {
            ErrorCategory.NETWORK_ERROR: RetryPolicy(
                max_attempts=5,
                base_delay=0.5,
                max_delay=60.0,
                exponential_base=2.0,
                jitter_range=0.2
            ),
            ErrorCategory.TIMEOUT_ERROR: RetryPolicy(
                max_attempts=3,
                base_delay=1.0,
                max_delay=30.0,
                exponential_base=1.5,
                jitter_range=0.1
            ),
            ErrorCategory.RATE_LIMIT_ERROR: RetryPolicy(
                max_attempts=10,
                base_delay=2.0,
                max_delay=120.0,
                exponential_base=2.0,
                jitter_range=0.3
            ),
            ErrorCategory.SERVER_ERROR: RetryPolicy(
                max_attempts=4,
                base_delay=1.0,
                max_delay=45.0,
                exponential_base=2.0,
                jitter_range=0.2
            ),
            ErrorCategory.MODEL_ERROR: RetryPolicy(
                max_attempts=2,
                base_delay=5.0,
                max_delay=60.0,
                exponential_base=1.5,
                jitter_range=0.1
            ),
            # Don't retry client errors by default
            ErrorCategory.CLIENT_ERROR: RetryPolicy(max_attempts=1),
            ErrorCategory.AUTHENTICATION_ERROR: RetryPolicy(max_attempts=1),
            ErrorCategory.CIRCUIT_BREAKER_ERROR: RetryPolicy(max_attempts=1),
        }
    
    def get_circuit_breaker(self, service_name: str) -> AdvancedCircuitBreaker:
        """Get or create circuit breaker for service."""
        if service_name not in self.circuit_breakers:
            config = CircuitBreakerConfig()
            self.circuit_breakers[service_name] = AdvancedCircuitBreaker(config)
        
        return self.circuit_breakers[service_name]
    
    async def execute_with_retry(
        self,
        func: Callable,
        service_name: str,
        request_id: str,
        context: Dict[str, Any] = None,
        custom_policy: Optional[RetryPolicy] = None,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with retry logic and circuit breaker protection."""
        circuit_breaker = self.get_circuit_breaker(service_name)
        context = context or {}
        
        async def protected_func():
            return await func(*args, **kwargs)
        
        last_error = None
        
        for attempt in range((custom_policy or RetryPolicy()).max_attempts):
            try:
                # Use circuit breaker protection
                result = await circuit_breaker.call(protected_func)
                
                # Log successful retry if this wasn't the first attempt
                if attempt > 0:
                    logger.info(f"Request {request_id} succeeded on attempt {attempt + 1}")
                
                return result
                
            except CircuitBreakerOpenError as e:
                # Circuit breaker is open, don't retry
                self._record_error(request_id, e, context, attempt, "circuit_breaker_open")
                raise e
                
            except Exception as e:
                last_error = e
                category, severity = self.error_classifier.classify_error(e, context)
                
                # Record error
                self._record_error(request_id, e, context, attempt, "retry_attempt")
                
                # Get retry policy for this error category
                policy = custom_policy or self.retry_policies.get(category, RetryPolicy(max_attempts=1))
                
                # Check if we should retry
                if attempt >= policy.max_attempts - 1:
                    logger.error(f"Request {request_id} failed after {attempt + 1} attempts: {e}")
                    break
                
                # Calculate delay with exponential backoff and jitter
                delay = self._calculate_retry_delay(policy, attempt)
                
                logger.warning(f"Request {request_id} attempt {attempt + 1} failed ({category.value}), "
                             f"retrying in {delay:.2f}s: {e}")
                
                await asyncio.sleep(delay)
        
        # All retries exhausted
        raise last_error
    
    def _calculate_retry_delay(self, policy: RetryPolicy, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter."""
        # Exponential backoff
        delay = policy.base_delay * (policy.exponential_base ** attempt)
        
        # Apply backoff multiplier
        delay *= policy.backoff_multiplier
        
        # Add jitter to avoid thundering herd
        jitter = random.uniform(-policy.jitter_range, policy.jitter_range)
        delay *= (1 + jitter)
        
        # Clamp to max delay
        return min(delay, policy.max_delay)
    
    def _record_error(
        self,
        request_id: str,
        error: Exception,
        context: Dict[str, Any],
        attempt: int,
        recovery_action: str
    ):
        """Record error event for analysis."""
        category, severity = self.error_classifier.classify_error(error, context)
        
        error_event = ErrorEvent(
            timestamp=datetime.now(),
            category=category,
            severity=severity,
            message=str(error),
            context=context,
            request_id=request_id,
            retry_attempt=attempt,
            recovery_action=recovery_action
        )
        
        self.error_history.append(error_event)
    
    def get_error_statistics(self, lookback_minutes: int = 60) -> Dict[str, Any]:
        """Get error statistics for the specified time period."""
        cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
        recent_errors = [e for e in self.error_history if e.timestamp >= cutoff_time]
        
        if not recent_errors:
            return {
                "total_errors": 0,
                "error_rate": 0.0,
                "lookback_minutes": lookback_minutes
            }
        
        # Group by category
        category_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for error in recent_errors:
            category_counts[error.category.value] += 1
            severity_counts[error.severity.name] += 1
        
        # Calculate error rate trend
        first_half = recent_errors[:len(recent_errors)//2]
        second_half = recent_errors[len(recent_errors)//2:]
        
        trend = "stable"
        if len(second_half) > len(first_half) * 1.5:
            trend = "increasing"
        elif len(second_half) < len(first_half) * 0.5:
            trend = "decreasing"
        
        return {
            "total_errors": len(recent_errors),
            "lookback_minutes": lookback_minutes,
            "category_breakdown": dict(category_counts),
            "severity_breakdown": dict(severity_counts),
            "error_trend": trend,
            "most_common_category": max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None,
            "circuit_breaker_states": {
                name: cb.get_state_info() 
                for name, cb in self.circuit_breakers.items()
            }
        }
    
    def get_recommendations(self) -> List[str]:
        """Get recommendations based on error patterns."""
        stats = self.get_error_statistics()
        recommendations = []
        
        if stats["total_errors"] == 0:
            return ["System is operating normally with no recent errors."]
        
        # High error rate
        if stats["total_errors"] > 50:
            recommendations.append("High error rate detected. Consider reducing request load or scaling resources.")
        
        # Network issues
        if stats["category_breakdown"].get("network_error", 0) > 10:
            recommendations.append("Frequent network errors. Check network connectivity and DNS resolution.")
        
        # Timeout issues
        if stats["category_breakdown"].get("timeout_error", 0) > 15:
            recommendations.append("High timeout rate. Consider increasing timeout values or optimizing model performance.")
        
        # Rate limiting
        if stats["category_breakdown"].get("rate_limit_error", 0) > 5:
            recommendations.append("Rate limiting detected. Implement request throttling or increase API limits.")
        
        # Server errors
        if stats["category_breakdown"].get("server_error", 0) > 8:
            recommendations.append("Server errors occurring. Check Ollama server status and resource utilization.")
        
        # Circuit breaker issues
        open_breakers = [name for name, info in stats.get("circuit_breaker_states", {}).items() 
                        if info.get("state") == "OPEN"]
        if open_breakers:
            recommendations.append(f"Circuit breakers open for: {', '.join(open_breakers)}. "
                                 "Service degradation in progress.")
        
        return recommendations or ["No specific recommendations at this time."]


# Global retry handler instance
_retry_handler: Optional[AdvancedRetryHandler] = None


def get_retry_handler() -> AdvancedRetryHandler:
    """Get or create global retry handler."""
    global _retry_handler
    
    if _retry_handler is None:
        _retry_handler = AdvancedRetryHandler()
        logger.info("Initialized advanced retry handler")
    
    return _retry_handler


# Convenience functions for common retry patterns
async def retry_ollama_request(func: Callable, request_id: str, context: Dict[str, Any] = None, *args, **kwargs):
    """Retry Ollama request with appropriate policy."""
    handler = get_retry_handler()
    return await handler.execute_with_retry(
        func, "ollama", request_id, context, *args, **kwargs
    )


async def retry_sentiment_analysis(func: Callable, request_id: str, symbol: str, *args, **kwargs):
    """Retry sentiment analysis with trading-specific policy."""
    handler = get_retry_handler()
    
    # Custom policy for sentiment analysis
    policy = RetryPolicy(
        max_attempts=3,
        base_delay=0.2,
        max_delay=10.0,
        exponential_base=1.5,
        jitter_range=0.1
    )
    
    context = {"operation": "sentiment_analysis", "symbol": symbol}
    return await handler.execute_with_retry(
        func, "sentiment", request_id, context, policy, *args, **kwargs
    )


if __name__ == "__main__":
    # Test the error handling system
    async def test_error_handling():
        print("🔧 Testing Advanced Error Handling")
        print("=" * 50)
        
        handler = get_retry_handler()
        
        # Test function that fails a few times then succeeds
        call_count = 0
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary network error")
            return "Success!"
        
        try:
            result = await handler.execute_with_retry(
                flaky_function, "test_service", "test_001"
            )
            print(f"Result: {result}")
            print(f"Required {call_count} attempts")
        except Exception as e:
            print(f"Final failure: {e}")
        
        # Get error statistics
        stats = handler.get_error_statistics()
        print(f"\nError Statistics: {json.dumps(stats, indent=2, default=str)}")
        
        # Get recommendations
        recommendations = handler.get_recommendations()
        print(f"\nRecommendations:")
        for rec in recommendations:
            print(f"  • {rec}")
    
    asyncio.run(test_error_handling())