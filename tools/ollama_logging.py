#!/usr/bin/env python3
"""
Advanced Logging and Debugging System for Ollama Trading System

Implements:
- Structured JSON logging with context propagation
- Performance tracing and profiling
- Trading-specific log enrichment
- Correlation ID tracking across components
- Log aggregation and analysis
- Debug mode with detailed request/response logging
"""

import asyncio
import json
import logging
import time
import traceback
import sys
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager, asynccontextmanager
from functools import wraps
import inspect

# Third-party imports for enhanced logging
try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False


class LogLevel(Enum):
    """Log levels with numeric values for filtering."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class LogCategory(Enum):
    """Log categories for filtering and analysis."""
    SYSTEM = "system"
    PERFORMANCE = "performance"
    TRADING = "trading"
    SENTIMENT = "sentiment"
    HEALTH = "health"
    ERROR = "error"
    AUDIT = "audit"
    SECURITY = "security"


@dataclass
class LogContext:
    """Log context for request tracing."""
    correlation_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    symbol: Optional[str] = None
    operation: Optional[str] = None
    model: Optional[str] = None
    instance: Optional[str] = None
    extra_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceTrace:
    """Performance trace for detailed timing analysis."""
    trace_id: str
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    sub_traces: List['PerformanceTrace'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StructuredLogger:
    """Enhanced structured logger with trading system integration."""
    
    def __init__(self, name: str, log_level: LogLevel = LogLevel.INFO):
        self.name = name
        self.log_level = log_level
        self.context_stack = threading.local()
        self.performance_traces = {}
        
        # Set up Python logger
        self._setup_python_logger()
        
        # Set up structured logger if available
        if HAS_STRUCTLOG:
            self._setup_structured_logger()
    
    def _setup_python_logger(self):
        """Set up standard Python logger with custom formatting."""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.log_level.value)
        
        # Create custom formatter
        formatter = StructuredJSONFormatter()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for production
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"{self.name}.log"),
            mode='a',
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def _setup_structured_logger(self):
        """Set up structlog if available."""
        if not HAS_STRUCTLOG:
            return
        
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.struct_logger = structlog.get_logger(self.name)
    
    def get_context(self) -> Optional[LogContext]:
        """Get current log context from thread-local storage."""
        return getattr(self.context_stack, 'context', None)
    
    def set_context(self, context: LogContext):
        """Set log context for current thread."""
        self.context_stack.context = context
    
    def clear_context(self):
        """Clear log context for current thread."""
        self.context_stack.context = None
    
    @contextmanager
    def context_manager(self, context: LogContext):
        """Context manager for scoped logging context."""
        old_context = self.get_context()
        self.set_context(context)
        try:
            yield context
        finally:
            if old_context:
                self.set_context(old_context)
            else:
                self.clear_context()
    
    def _enrich_log_data(self, level: LogLevel, message: str, **kwargs) -> Dict[str, Any]:
        """Enrich log data with context and metadata."""
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level.name,
            "logger": self.name,
            "message": message,
            "thread_id": threading.current_thread().ident,
            "process_id": os.getpid(),
        }
        
        # Add context if available
        context = self.get_context()
        if context:
            data.update({
                "correlation_id": context.correlation_id,
                "session_id": context.session_id,
                "user_id": context.user_id,
                "request_id": context.request_id,
                "symbol": context.symbol,
                "operation": context.operation,
                "model": context.model,
                "instance": context.instance,
            })
            
            # Add extra fields
            if context.extra_fields:
                data.update(context.extra_fields)
        
        # Add additional kwargs
        data.update(kwargs)
        
        # Remove None values
        return {k: v for k, v in data.items() if v is not None}
    
    def debug(self, message: str, **kwargs):
        """Debug level logging."""
        if self.log_level.value <= LogLevel.DEBUG.value:
            data = self._enrich_log_data(LogLevel.DEBUG, message, **kwargs)
            self.logger.debug(json.dumps(data))
    
    def info(self, message: str, **kwargs):
        """Info level logging."""
        if self.log_level.value <= LogLevel.INFO.value:
            data = self._enrich_log_data(LogLevel.INFO, message, **kwargs)
            self.logger.info(json.dumps(data))
    
    def warning(self, message: str, **kwargs):
        """Warning level logging."""
        if self.log_level.value <= LogLevel.WARNING.value:
            data = self._enrich_log_data(LogLevel.WARNING, message, **kwargs)
            self.logger.warning(json.dumps(data))
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Error level logging with optional exception details."""
        if exception:
            kwargs.update({
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "traceback": traceback.format_exc()
            })
        
        data = self._enrich_log_data(LogLevel.ERROR, message, **kwargs)
        self.logger.error(json.dumps(data))
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Critical level logging with optional exception details."""
        if exception:
            kwargs.update({
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "traceback": traceback.format_exc()
            })
        
        data = self._enrich_log_data(LogLevel.CRITICAL, message, **kwargs)
        self.logger.critical(json.dumps(data))
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics."""
        self.info(
            f"Performance: {operation}",
            category=LogCategory.PERFORMANCE.value,
            operation=operation,
            duration_ms=duration * 1000,
            **kwargs
        )
    
    def log_trading_signal(self, signal_type: str, symbol: str, confidence: float, **kwargs):
        """Log trading signal generation."""
        self.info(
            f"Trading signal generated: {signal_type} for {symbol}",
            category=LogCategory.TRADING.value,
            signal_type=signal_type,
            symbol=symbol,
            confidence=confidence,
            **kwargs
        )
    
    def log_sentiment_analysis(self, symbol: str, sentiment: str, confidence: float, source: str, **kwargs):
        """Log sentiment analysis results."""
        self.info(
            f"Sentiment analysis: {sentiment} for {symbol} from {source}",
            category=LogCategory.SENTIMENT.value,
            symbol=symbol,
            sentiment=sentiment,
            confidence=confidence,
            source=source,
            **kwargs
        )
    
    def log_health_check(self, instance: str, status: str, response_time: float, **kwargs):
        """Log health check results."""
        self.info(
            f"Health check: {instance} is {status}",
            category=LogCategory.HEALTH.value,
            instance=instance,
            health_status=status,
            response_time_ms=response_time * 1000,
            **kwargs
        )
    
    def log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log audit events for compliance."""
        self.info(
            f"Audit event: {event_type}",
            category=LogCategory.AUDIT.value,
            event_type=event_type,
            **details
        )
    
    def start_trace(self, operation: str, **metadata) -> str:
        """Start a performance trace."""
        trace_id = str(uuid.uuid4())
        trace = PerformanceTrace(
            trace_id=trace_id,
            operation=operation,
            start_time=time.time(),
            metadata=metadata
        )
        
        self.performance_traces[trace_id] = trace
        
        self.debug(
            f"Started trace: {operation}",
            category=LogCategory.PERFORMANCE.value,
            trace_id=trace_id,
            operation=operation,
            **metadata
        )
        
        return trace_id
    
    def end_trace(self, trace_id: str, **additional_metadata):
        """End a performance trace."""
        if trace_id not in self.performance_traces:
            self.warning(f"Trace {trace_id} not found")
            return
        
        trace = self.performance_traces[trace_id]
        trace.end_time = time.time()
        trace.duration = trace.end_time - trace.start_time
        trace.metadata.update(additional_metadata)
        
        self.log_performance(
            operation=trace.operation,
            duration=trace.duration,
            trace_id=trace_id,
            **trace.metadata
        )
        
        # Clean up
        del self.performance_traces[trace_id]
    
    @contextmanager
    def trace_operation(self, operation: str, **metadata):
        """Context manager for tracing operations."""
        trace_id = self.start_trace(operation, **metadata)
        try:
            yield trace_id
        finally:
            self.end_trace(trace_id)


class StructuredJSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        # Check if message is already JSON
        try:
            log_data = json.loads(record.getMessage())
            return json.dumps(log_data)
        except (json.JSONDecodeError, ValueError):
            # Fall back to creating JSON structure
            log_data = {
                "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }
            
            if record.exc_info:
                log_data["exception"] = self.formatException(record.exc_info)
            
            return json.dumps(log_data)


class RequestTracker:
    """Track requests across async operations with correlation IDs."""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.active_requests = {}
    
    def start_request(self, operation: str, **metadata) -> LogContext:
        """Start tracking a new request."""
        correlation_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())
        
        context = LogContext(
            correlation_id=correlation_id,
            request_id=request_id,
            operation=operation,
            extra_fields=metadata
        )
        
        self.active_requests[request_id] = {
            "context": context,
            "start_time": time.time(),
            "operation": operation,
            "metadata": metadata
        }
        
        self.logger.set_context(context)
        self.logger.info(
            f"Request started: {operation}",
            category=LogCategory.SYSTEM.value,
            **metadata
        )
        
        return context
    
    def end_request(self, request_id: str, success: bool = True, **result_metadata):
        """End request tracking."""
        if request_id not in self.active_requests:
            self.logger.warning(f"Request {request_id} not found in active requests")
            return
        
        request_data = self.active_requests[request_id]
        duration = time.time() - request_data["start_time"]
        
        # Set context for this log
        self.logger.set_context(request_data["context"])
        
        self.logger.info(
            f"Request completed: {request_data['operation']}",
            category=LogCategory.SYSTEM.value,
            success=success,
            duration_ms=duration * 1000,
            **result_metadata
        )
        
        # Clean up
        del self.active_requests[request_id]
        self.logger.clear_context()
    
    @asynccontextmanager
    async def track_async_request(self, operation: str, **metadata):
        """Async context manager for request tracking."""
        context = self.start_request(operation, **metadata)
        try:
            yield context
            self.end_request(context.request_id, success=True)
        except Exception as e:
            self.end_request(context.request_id, success=False, error=str(e))
            raise


def performance_log(logger: StructuredLogger):
    """Decorator for automatic performance logging."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            operation = f"{func.__module__}.{func.__name__}"
            
            with logger.trace_operation(operation, function=func.__name__, module=func.__module__):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            operation = f"{func.__module__}.{func.__name__}"
            
            with logger.trace_operation(operation, function=func.__name__, module=func.__module__):
                return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class OllamaDebugLogger:
    """Specialized debug logger for Ollama interactions."""
    
    def __init__(self, base_logger: StructuredLogger, debug_enabled: bool = False):
        self.logger = base_logger
        self.debug_enabled = debug_enabled
        self.request_response_log = []
    
    def log_request(self, url: str, payload: Dict[str, Any], headers: Dict[str, str] = None):
        """Log outgoing request details."""
        if not self.debug_enabled:
            return
        
        log_data = {
            "direction": "outbound",
            "url": url,
            "payload": payload,
            "headers": headers or {}
        }
        
        self.logger.debug(
            f"Ollama request to {url}",
            category=LogCategory.SYSTEM.value,
            **log_data
        )
        
        # Store for debugging
        self.request_response_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": "request",
            **log_data
        })
    
    def log_response(self, url: str, status: int, response_data: Any, response_time: float):
        """Log incoming response details."""
        if not self.debug_enabled:
            return
        
        log_data = {
            "direction": "inbound",
            "url": url,
            "status_code": status,
            "response_time_ms": response_time * 1000,
            "response_size": len(str(response_data)) if response_data else 0
        }
        
        # Include response data only in debug mode and if reasonable size
        if isinstance(response_data, dict) and len(str(response_data)) < 1000:
            log_data["response_data"] = response_data
        
        self.logger.debug(
            f"Ollama response from {url} ({status})",
            category=LogCategory.SYSTEM.value,
            **log_data
        )
        
        # Store for debugging
        self.request_response_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": "response",
            **log_data
        })
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """Get summary of debug information."""
        if not self.request_response_log:
            return {"message": "No debug data available"}
        
        requests = [log for log in self.request_response_log if log["type"] == "request"]
        responses = [log for log in self.request_response_log if log["type"] == "response"]
        
        return {
            "total_requests": len(requests),
            "total_responses": len(responses),
            "avg_response_time": sum(r.get("response_time_ms", 0) for r in responses) / len(responses) if responses else 0,
            "status_codes": [r.get("status_code") for r in responses],
            "recent_interactions": self.request_response_log[-10:]  # Last 10 interactions
        }


# Global logger instances
_loggers: Dict[str, StructuredLogger] = {}


def get_logger(name: str, level: LogLevel = LogLevel.INFO) -> StructuredLogger:
    """Get or create a structured logger instance."""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, level)
    
    return _loggers[name]


def create_correlation_context(operation: str, symbol: str = None, model: str = None) -> LogContext:
    """Create a new correlation context for request tracking."""
    return LogContext(
        correlation_id=str(uuid.uuid4()),
        request_id=str(uuid.uuid4()),
        operation=operation,
        symbol=symbol,
        model=model
    )


# Convenience functions for common logging scenarios
def log_ollama_request(logger: StructuredLogger, url: str, model: str, prompt: str, response_time: float, success: bool):
    """Log Ollama request with standardized format."""
    logger.info(
        f"Ollama request: {model}",
        category=LogCategory.SYSTEM.value,
        url=url,
        model=model,
        prompt_length=len(prompt),
        response_time_ms=response_time * 1000,
        success=success
    )


def log_trading_decision(logger: StructuredLogger, symbol: str, action: str, confidence: float, reasoning: str):
    """Log trading decision with detailed context."""
    logger.info(
        f"Trading decision: {action} {symbol}",
        category=LogCategory.TRADING.value,
        symbol=symbol,
        action=action,
        confidence=confidence,
        reasoning=reasoning
    )


if __name__ == "__main__":
    # Test the logging system
    async def test_logging():
        print("📝 Testing Advanced Logging System")
        print("=" * 50)
        
        # Create logger
        logger = get_logger("test_logger", LogLevel.DEBUG)
        
        # Test context management
        context = create_correlation_context("test_operation", symbol="AAPL", model="llama3.1")
        
        with logger.context_manager(context):
            logger.info("Testing structured logging with context")
            
            # Test performance tracing
            with logger.trace_operation("sample_operation", symbol="AAPL"):
                await asyncio.sleep(0.1)  # Simulate work
                
                logger.log_sentiment_analysis("AAPL", "positive", 0.85, "social_media")
                logger.log_trading_signal("BUY", "AAPL", 0.9)
        
        # Test request tracking
        tracker = RequestTracker(logger)
        
        async with tracker.track_async_request("sentiment_analysis", symbol="GOOGL"):
            await asyncio.sleep(0.05)  # Simulate work
            logger.info("Performing sentiment analysis")
        
        # Test debug logger
        debug_logger = OllamaDebugLogger(logger, debug_enabled=True)
        debug_logger.log_request("http://localhost:11434/api/generate", {"model": "llama3.1", "prompt": "test"})
        debug_logger.log_response("http://localhost:11434/api/generate", 200, {"response": "test response"}, 0.5)
        
        summary = debug_logger.get_debug_summary()
        print(f"Debug summary: {json.dumps(summary, indent=2)}")
    
    asyncio.run(test_logging())