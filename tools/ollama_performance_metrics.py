#!/usr/bin/env python3
"""
Advanced Performance Metrics and Observability for Ollama Trading System

Provides comprehensive metrics collection, analysis, and alerting 
specifically designed for high-frequency trading requirements.
"""

import asyncio
import json
import logging
import time
import psutil
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import weakref

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str]
    metric_type: MetricType


@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time."""
    timestamp: datetime
    response_times: List[float]
    success_rate: float
    queue_depth: int
    active_connections: int
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hit_rate: float
    circuit_breaker_state: str
    throughput_rps: float
    error_rate: float
    p50_latency: float
    p95_latency: float
    p99_latency: float


class AdvancedMetricsCollector:
    """Advanced metrics collection with statistical analysis."""
    
    def __init__(self, history_size: int = 10000, sampling_interval: float = 1.0):
        self.history_size = history_size
        self.sampling_interval = sampling_interval
        
        # Metric storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Performance tracking
        self.response_times = deque(maxlen=1000)
        self.request_timestamps = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.success_counts = defaultdict(int)
        
        # System monitoring
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Alerting thresholds
        self.alert_thresholds = {
            "p95_latency_ms": 5000,
            "error_rate_percent": 5.0,
            "memory_growth_mb": 500,
            "cpu_usage_percent": 80.0,
            "queue_depth": 50,
            "throughput_drop_percent": 50.0
        }
    
    def record_request_start(self, request_id: str, priority: str, context: str):
        """Record request start for latency tracking."""
        with self._lock:
            timestamp = time.time()
            self.request_timestamps.append((request_id, timestamp, priority, context))
    
    def record_request_complete(self, request_id: str, success: bool, response_time: float):
        """Record request completion with success/failure tracking."""
        with self._lock:
            # Record response time
            self.response_times.append(response_time)
            
            # Track success/failure by context
            if success:
                self.success_counts['total'] += 1
            else:
                self.error_counts['total'] += 1
            
            # Update counters
            self.increment_counter("requests_total", {"success": str(success)})
            self.record_histogram("response_time_ms", response_time * 1000)
    
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            key = self._make_key(name, labels or {})
            self.counters[key] += 1
            
            # Store metric point
            self.metrics[key].append(MetricPoint(
                timestamp=datetime.now(),
                value=self.counters[key],
                labels=labels or {},
                metric_type=MetricType.COUNTER
            ))
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        with self._lock:
            key = self._make_key(name, labels or {})
            self.gauges[key] = value
            
            # Store metric point
            self.metrics[key].append(MetricPoint(
                timestamp=datetime.now(),
                value=value,
                labels=labels or {},
                metric_type=MetricType.GAUGE
            ))
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram value."""
        with self._lock:
            key = self._make_key(name, labels or {})
            self.histograms[key].append(value)
            
            # Store metric point
            self.metrics[key].append(MetricPoint(
                timestamp=datetime.now(),
                value=value,
                labels=labels or {},
                metric_type=MetricType.HISTOGRAM
            ))
    
    def _make_key(self, name: str, labels: Dict[str, str]) -> str:
        """Create a unique key for a metric with labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile from values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_current_snapshot(self) -> PerformanceSnapshot:
        """Get current performance snapshot."""
        with self._lock:
            now = datetime.now()
            
            # Calculate response time statistics
            recent_times = list(self.response_times)
            if recent_times:
                p50 = self.get_percentile(recent_times, 50)
                p95 = self.get_percentile(recent_times, 95)
                p99 = self.get_percentile(recent_times, 99)
            else:
                p50 = p95 = p99 = 0.0
            
            # Calculate success rate
            total_success = self.success_counts.get('total', 0)
            total_errors = self.error_counts.get('total', 0)
            total_requests = total_success + total_errors
            success_rate = total_success / total_requests if total_requests > 0 else 1.0
            
            # Calculate throughput (requests per second)
            recent_requests = [ts for _, ts, _, _ in self.request_timestamps 
                             if now.timestamp() - ts < 60]  # Last minute
            throughput_rps = len(recent_requests) / 60.0
            
            # System metrics
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            cpu_percent = self.process.cpu_percent()
            
            # Get latest gauge values
            queue_depth = self.gauges.get("queue_depth", 0)
            active_connections = self.gauges.get("active_connections", 0)
            cache_hit_rate = self.gauges.get("cache_hit_rate", 0)
            circuit_breaker_state = self.gauges.get("circuit_breaker_state", "CLOSED")
            
            return PerformanceSnapshot(
                timestamp=now,
                response_times=recent_times[-100:],  # Last 100 response times
                success_rate=success_rate,
                queue_depth=int(queue_depth),
                active_connections=int(active_connections),
                memory_usage_mb=memory_mb,
                cpu_usage_percent=cpu_percent,
                cache_hit_rate=cache_hit_rate,
                circuit_breaker_state=str(circuit_breaker_state),
                throughput_rps=throughput_rps,
                error_rate=(1.0 - success_rate) * 100,
                p50_latency=p50,
                p95_latency=p95,
                p99_latency=p99
            )
    
    def detect_anomalies(self, snapshot: PerformanceSnapshot) -> List[Dict[str, Any]]:
        """Detect performance anomalies based on thresholds."""
        anomalies = []
        
        # High latency
        if snapshot.p95_latency > self.alert_thresholds["p95_latency_ms"] / 1000:
            anomalies.append({
                "type": "high_latency",
                "severity": "warning",
                "message": f"P95 latency {snapshot.p95_latency*1000:.0f}ms exceeds threshold",
                "threshold": self.alert_thresholds["p95_latency_ms"],
                "current_value": snapshot.p95_latency * 1000
            })
        
        # High error rate
        if snapshot.error_rate > self.alert_thresholds["error_rate_percent"]:
            anomalies.append({
                "type": "high_error_rate",
                "severity": "critical",
                "message": f"Error rate {snapshot.error_rate:.1f}% exceeds threshold",
                "threshold": self.alert_thresholds["error_rate_percent"],
                "current_value": snapshot.error_rate
            })
        
        # Memory growth
        memory_growth = snapshot.memory_usage_mb - self.baseline_memory
        if memory_growth > self.alert_thresholds["memory_growth_mb"]:
            anomalies.append({
                "type": "memory_leak",
                "severity": "warning",
                "message": f"Memory usage grew by {memory_growth:.0f}MB",
                "threshold": self.alert_thresholds["memory_growth_mb"],
                "current_value": memory_growth
            })
        
        # High CPU usage
        if snapshot.cpu_usage_percent > self.alert_thresholds["cpu_usage_percent"]:
            anomalies.append({
                "type": "high_cpu",
                "severity": "warning",
                "message": f"CPU usage {snapshot.cpu_usage_percent:.1f}% is high",
                "threshold": self.alert_thresholds["cpu_usage_percent"],
                "current_value": snapshot.cpu_usage_percent
            })
        
        # Queue backup
        if snapshot.queue_depth > self.alert_thresholds["queue_depth"]:
            anomalies.append({
                "type": "queue_backup",
                "severity": "critical",
                "message": f"Queue depth {snapshot.queue_depth} is too high",
                "threshold": self.alert_thresholds["queue_depth"],
                "current_value": snapshot.queue_depth
            })
        
        return anomalies
    
    def get_trending_analysis(self, lookback_minutes: int = 30) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
            
            # Get recent response times
            recent_times = [t for t in self.response_times 
                          if datetime.now().timestamp() - t < lookback_minutes * 60]
            
            if len(recent_times) < 10:
                return {
                    "insufficient_data": True,
                    "sample_size": len(recent_times)
                }
            
            # Calculate trends
            first_half = recent_times[:len(recent_times)//2]
            second_half = recent_times[len(recent_times)//2:]
            
            avg_first = statistics.mean(first_half) if first_half else 0
            avg_second = statistics.mean(second_half) if second_half else 0
            
            trend_direction = "improving" if avg_second < avg_first else "degrading"
            trend_magnitude = abs(avg_second - avg_first) / avg_first if avg_first > 0 else 0
            
            return {
                "lookback_minutes": lookback_minutes,
                "sample_size": len(recent_times),
                "trend_direction": trend_direction,
                "trend_magnitude_percent": trend_magnitude * 100,
                "average_latency_first_half": avg_first,
                "average_latency_second_half": avg_second,
                "volatility": statistics.stdev(recent_times) if len(recent_times) > 1 else 0
            }
    
    def export_metrics(self, format_type: str = "prometheus") -> str:
        """Export metrics in various formats."""
        with self._lock:
            if format_type == "prometheus":
                return self._export_prometheus()
            elif format_type == "json":
                return self._export_json()
            else:
                raise ValueError(f"Unsupported format: {format_type}")
    
    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Counters
        for key, value in self.counters.items():
            lines.append(f"ollama_{key} {value}")
        
        # Gauges  
        for key, value in self.gauges.items():
            lines.append(f"ollama_{key} {value}")
        
        # Histograms (export as summary statistics)
        for key, values in self.histograms.items():
            if values:
                lines.append(f"ollama_{key}_sum {sum(values)}")
                lines.append(f"ollama_{key}_count {len(values)}")
                lines.append(f"ollama_{key}_p50 {self.get_percentile(list(values), 50)}")
                lines.append(f"ollama_{key}_p95 {self.get_percentile(list(values), 95)}")
                lines.append(f"ollama_{key}_p99 {self.get_percentile(list(values), 99)}")
        
        return "\n".join(lines)
    
    def _export_json(self) -> str:
        """Export metrics in JSON format."""
        snapshot = self.get_current_snapshot()
        trending = self.get_trending_analysis()
        anomalies = self.detect_anomalies(snapshot)
        
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "current_snapshot": asdict(snapshot),
            "trending_analysis": trending,
            "anomalies": anomalies,
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "alert_thresholds": self.alert_thresholds
        }
        
        return json.dumps(export_data, indent=2, default=str)


class TradingSystemObserver:
    """Trading system specific performance observer."""
    
    def __init__(self, metrics_collector: AdvancedMetricsCollector):
        self.metrics = metrics_collector
        self.trading_session_start = datetime.now()
        self.market_hours_cache = {}
        
    def record_sentiment_analysis(self, symbol: str, latency: float, success: bool, priority: str):
        """Record sentiment analysis performance."""
        self.metrics.record_request_complete(f"sentiment_{symbol}", success, latency)
        self.metrics.increment_counter("sentiment_analyses_total", {
            "symbol": symbol,
            "priority": priority,
            "success": str(success)
        })
        self.metrics.record_histogram("sentiment_latency_ms", latency * 1000, {"priority": priority})
    
    def record_trading_signal(self, signal_type: str, confidence: float, processing_time: float):
        """Record trading signal generation metrics."""
        self.metrics.increment_counter("trading_signals_total", {"type": signal_type})
        self.metrics.record_histogram("signal_confidence", confidence)
        self.metrics.record_histogram("signal_processing_time_ms", processing_time * 1000)
    
    def record_model_performance(self, model: str, tokens_processed: int, inference_time: float):
        """Record LLM model performance metrics."""
        self.metrics.increment_counter("model_inferences_total", {"model": model})
        self.metrics.record_histogram("model_inference_time_ms", inference_time * 1000, {"model": model})
        self.metrics.record_histogram("tokens_processed", tokens_processed, {"model": model})
        
        # Calculate tokens per second
        tokens_per_sec = tokens_processed / inference_time if inference_time > 0 else 0
        self.metrics.record_histogram("tokens_per_second", tokens_per_sec, {"model": model})
    
    def get_trading_session_summary(self) -> Dict[str, Any]:
        """Get summary of current trading session performance."""
        session_duration = (datetime.now() - self.trading_session_start).total_seconds()
        snapshot = self.metrics.get_current_snapshot()
        
        return {
            "session_duration_hours": session_duration / 3600,
            "session_start": self.trading_session_start.isoformat(),
            "current_performance": asdict(snapshot),
            "total_requests": self.metrics.counters.get("requests_total", 0),
            "avg_throughput_rps": snapshot.throughput_rps,
            "session_reliability": snapshot.success_rate,
            "performance_grade": self._calculate_performance_grade(snapshot)
        }
    
    def _calculate_performance_grade(self, snapshot: PerformanceSnapshot) -> str:
        """Calculate overall performance grade A-F."""
        score = 100
        
        # Deduct points for various issues
        if snapshot.p95_latency > 5.0:  # > 5 seconds
            score -= 30
        elif snapshot.p95_latency > 2.0:  # > 2 seconds
            score -= 15
        
        if snapshot.error_rate > 5.0:  # > 5% errors
            score -= 40
        elif snapshot.error_rate > 1.0:  # > 1% errors
            score -= 20
        
        if snapshot.queue_depth > 50:
            score -= 20
        elif snapshot.queue_depth > 20:
            score -= 10
        
        if snapshot.cpu_usage_percent > 80:
            score -= 15
        
        if snapshot.cache_hit_rate < 30:
            score -= 10
        
        # Assign letter grade
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"


# Global metrics collector instance
_metrics_collector: Optional[AdvancedMetricsCollector] = None
_trading_observer: Optional[TradingSystemObserver] = None


def get_metrics_collector() -> AdvancedMetricsCollector:
    """Get or create global metrics collector."""
    global _metrics_collector
    
    if _metrics_collector is None:
        _metrics_collector = AdvancedMetricsCollector()
        logger.info("Initialized advanced metrics collector")
    
    return _metrics_collector


def get_trading_observer() -> TradingSystemObserver:
    """Get or create trading system observer."""
    global _trading_observer
    
    if _trading_observer is None:
        _trading_observer = TradingSystemObserver(get_metrics_collector())
        logger.info("Initialized trading system observer")
    
    return _trading_observer


async def start_metrics_background_collection(interval: int = 10):
    """Start background metrics collection."""
    collector = get_metrics_collector()
    
    while True:
        try:
            # Update system metrics
            snapshot = collector.get_current_snapshot()
            
            collector.set_gauge("memory_usage_mb", snapshot.memory_usage_mb)
            collector.set_gauge("cpu_usage_percent", snapshot.cpu_usage_percent)
            collector.set_gauge("throughput_rps", snapshot.throughput_rps)
            
            # Check for anomalies
            anomalies = collector.detect_anomalies(snapshot)
            for anomaly in anomalies:
                if anomaly["severity"] == "critical":
                    logger.critical(f"Performance anomaly detected: {anomaly['message']}")
                else:
                    logger.warning(f"Performance warning: {anomaly['message']}")
            
            await asyncio.sleep(interval)
            
        except Exception as e:
            logger.error(f"Metrics collection error: {e}")
            await asyncio.sleep(5)


if __name__ == "__main__":
    # Test the metrics system
    async def test_metrics():
        print("🔬 Testing Advanced Metrics Collection")
        print("=" * 50)
        
        collector = get_metrics_collector()
        observer = get_trading_observer()
        
        # Simulate some metrics
        for i in range(100):
            collector.record_request_complete(f"test_{i}", True, 0.1 + i * 0.01)
            observer.record_sentiment_analysis("AAPL", 0.2, True, "HIGH")
            await asyncio.sleep(0.01)
        
        # Get snapshot
        snapshot = collector.get_current_snapshot()
        print(f"Performance Snapshot:")
        print(f"  P95 Latency: {snapshot.p95_latency*1000:.1f}ms")
        print(f"  Success Rate: {snapshot.success_rate:.1%}")
        print(f"  Throughput: {snapshot.throughput_rps:.1f} RPS")
        
        # Export metrics
        json_export = collector.export_metrics("json")
        print(f"\nExported {len(json_export)} bytes of metrics data")
        
        # Trading session summary
        summary = observer.get_trading_session_summary()
        print(f"Performance Grade: {summary['performance_grade']}")
    
    asyncio.run(test_metrics())