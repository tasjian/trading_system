#!/usr/bin/env python3
"""
LLM Pipeline MLOps Monitoring System

Comprehensive monitoring for LLM sentiment analysis pipeline including:
- Model serving health and performance metrics
- Pipeline latency and throughput monitoring  
- Error rate tracking and alerting
- Resource utilization monitoring
- Circuit breaker state tracking
- Fallback usage analytics
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Core pipeline performance metrics."""
    timestamp: datetime
    
    # Latency metrics (milliseconds)
    avg_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    
    # Throughput metrics
    requests_per_minute: float
    successful_requests: int
    failed_requests: int
    
    # Error metrics
    error_rate: float
    timeout_rate: float
    circuit_breaker_trips: int
    
    # Resource metrics
    queue_utilization: float
    worker_utilization: float
    cache_hit_rate: float
    
    # Model serving metrics
    ollama_availability: bool
    model_load_time: float
    fallback_usage_rate: float


@dataclass
class AlertThresholds:
    """Alerting thresholds for pipeline monitoring."""
    max_avg_response_time: float = 10000  # 10 seconds
    max_error_rate: float = 0.15  # 15%
    max_timeout_rate: float = 0.10  # 10%
    min_cache_hit_rate: float = 0.30  # 30%
    max_queue_utilization: float = 0.80  # 80%
    max_fallback_usage: float = 0.50  # 50%


class LLMPipelineMonitor:
    """Comprehensive MLOps monitoring for LLM pipeline."""
    
    def __init__(self, 
                 metrics_window_minutes: int = 15,
                 alert_cooldown_minutes: int = 5,
                 storage_path: str = "logs/llm_metrics.jsonl"):
        
        self.metrics_window = timedelta(minutes=metrics_window_minutes)
        self.alert_cooldown = timedelta(minutes=alert_cooldown_minutes)
        self.storage_path = Path(storage_path)
        
        # Metrics storage
        self.response_times = deque(maxlen=1000)
        self.request_outcomes = deque(maxlen=1000)  # True=success, False=failure
        self.timeout_events = deque(maxlen=1000)
        self.circuit_breaker_events = deque(maxlen=100)
        self.fallback_usage = deque(maxlen=1000)
        
        # Real-time counters
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.timeout_count = 0
        self.circuit_breaker_trips = 0
        
        # Alert management
        self.alert_thresholds = AlertThresholds()
        self.last_alerts = {}  # alert_type -> timestamp
        
        # Background monitoring
        self.monitoring_active = False
        self.monitor_task = None
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info("LLM Pipeline Monitor initialized")
    
    async def start_monitoring(self):
        """Start background monitoring task."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started LLM pipeline monitoring")
    
    async def stop_monitoring(self):
        """Stop background monitoring task."""
        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped LLM pipeline monitoring")
    
    def record_request(self, 
                      response_time_ms: float,
                      success: bool,
                      timeout: bool = False,
                      used_fallback: bool = False,
                      circuit_breaker_open: bool = False):
        """Record a single request outcome."""
        
        with self._lock:
            timestamp = datetime.now()
            
            # Record metrics
            self.response_times.append((timestamp, response_time_ms))
            self.request_outcomes.append((timestamp, success))
            
            if timeout:
                self.timeout_events.append(timestamp)
                self.timeout_count += 1
            
            if used_fallback:
                self.fallback_usage.append(timestamp)
            
            if circuit_breaker_open:
                self.circuit_breaker_events.append(timestamp)
                self.circuit_breaker_trips += 1
            
            # Update counters
            self.total_requests += 1
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
    
    def get_current_metrics(self) -> PipelineMetrics:
        """Get current pipeline metrics."""
        
        with self._lock:
            now = datetime.now()
            window_start = now - self.metrics_window
            
            # Filter recent data
            recent_response_times = [
                rt for ts, rt in self.response_times
                if ts >= window_start
            ]
            
            recent_outcomes = [
                outcome for ts, outcome in self.request_outcomes
                if ts >= window_start
            ]
            
            recent_timeouts = [
                ts for ts in self.timeout_events
                if ts >= window_start
            ]
            
            recent_fallbacks = [
                ts for ts in self.fallback_usage
                if ts >= window_start
            ]
            
            recent_cb_events = [
                ts for ts in self.circuit_breaker_events
                if ts >= window_start
            ]
            
            # Calculate metrics
            if recent_response_times:
                sorted_times = sorted(recent_response_times)
                avg_response_time = sum(sorted_times) / len(sorted_times)
                p50_response_time = sorted_times[len(sorted_times) // 2]
                p95_response_time = sorted_times[int(len(sorted_times) * 0.95)]
                p99_response_time = sorted_times[int(len(sorted_times) * 0.99)]
            else:
                avg_response_time = p50_response_time = p95_response_time = p99_response_time = 0.0
            
            total_recent = len(recent_outcomes)
            successful_recent = sum(recent_outcomes) if recent_outcomes else 0
            
            error_rate = (total_recent - successful_recent) / max(total_recent, 1)
            timeout_rate = len(recent_timeouts) / max(total_recent, 1)
            fallback_usage_rate = len(recent_fallbacks) / max(total_recent, 1)
            
            # Requests per minute
            requests_per_minute = total_recent / max(self.metrics_window.total_seconds() / 60, 1)
            
            return PipelineMetrics(
                timestamp=now,
                avg_response_time=avg_response_time,
                p50_response_time=p50_response_time,
                p95_response_time=p95_response_time,
                p99_response_time=p99_response_time,
                requests_per_minute=requests_per_minute,
                successful_requests=successful_recent,
                failed_requests=total_recent - successful_recent,
                error_rate=error_rate,
                timeout_rate=timeout_rate,
                circuit_breaker_trips=len(recent_cb_events),
                queue_utilization=0.0,  # Will be updated by external sources
                worker_utilization=0.0,  # Will be updated by external sources
                cache_hit_rate=0.0,  # Will be updated by external sources
                ollama_availability=True,  # Will be updated by external sources
                model_load_time=0.0,  # Will be updated by external sources
                fallback_usage_rate=fallback_usage_rate
            )
    
    async def check_alerts(self, metrics: PipelineMetrics) -> List[Dict[str, Any]]:
        """Check for alert conditions and return alerts."""
        
        alerts = []
        now = datetime.now()
        
        # Average response time alert
        if metrics.avg_response_time > self.alert_thresholds.max_avg_response_time:
            if self._should_send_alert("high_latency", now):
                alerts.append({
                    "type": "high_latency",
                    "severity": "warning",
                    "message": f"High average response time: {metrics.avg_response_time:.1f}ms",
                    "timestamp": now.isoformat(),
                    "metric_value": metrics.avg_response_time,
                    "threshold": self.alert_thresholds.max_avg_response_time
                })
        
        # Error rate alert
        if metrics.error_rate > self.alert_thresholds.max_error_rate:
            severity = "critical" if metrics.error_rate > 0.25 else "warning"
            if self._should_send_alert("high_error_rate", now):
                alerts.append({
                    "type": "high_error_rate", 
                    "severity": severity,
                    "message": f"High error rate: {metrics.error_rate:.1%}",
                    "timestamp": now.isoformat(),
                    "metric_value": metrics.error_rate,
                    "threshold": self.alert_thresholds.max_error_rate
                })
        
        # Timeout rate alert
        if metrics.timeout_rate > self.alert_thresholds.max_timeout_rate:
            if self._should_send_alert("high_timeout_rate", now):
                alerts.append({
                    "type": "high_timeout_rate",
                    "severity": "warning",
                    "message": f"High timeout rate: {metrics.timeout_rate:.1%}",
                    "timestamp": now.isoformat(),
                    "metric_value": metrics.timeout_rate,
                    "threshold": self.alert_thresholds.max_timeout_rate
                })
        
        # Fallback usage alert
        if metrics.fallback_usage_rate > self.alert_thresholds.max_fallback_usage:
            severity = "critical" if metrics.fallback_usage_rate > 0.75 else "warning"
            if self._should_send_alert("high_fallback_usage", now):
                alerts.append({
                    "type": "high_fallback_usage",
                    "severity": severity,
                    "message": f"High fallback usage: {metrics.fallback_usage_rate:.1%}",
                    "timestamp": now.isoformat(),
                    "metric_value": metrics.fallback_usage_rate,
                    "threshold": self.alert_thresholds.max_fallback_usage
                })
        
        # Model availability alert
        if not metrics.ollama_availability:
            if self._should_send_alert("model_unavailable", now):
                alerts.append({
                    "type": "model_unavailable",
                    "severity": "critical",
                    "message": "Ollama model server unavailable",
                    "timestamp": now.isoformat(),
                    "metric_value": False,
                    "threshold": True
                })
        
        return alerts
    
    def _should_send_alert(self, alert_type: str, timestamp: datetime) -> bool:
        """Check if alert should be sent (respects cooldown)."""
        last_alert = self.last_alerts.get(alert_type)
        if not last_alert or timestamp - last_alert >= self.alert_cooldown:
            self.last_alerts[alert_type] = timestamp
            return True
        return False
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        logger.info("Starting monitoring loop")
        
        while self.monitoring_active:
            try:
                # Get current metrics
                metrics = self.get_current_metrics()
                
                # Check for alerts
                alerts = await self.check_alerts(metrics)
                
                # Process alerts
                for alert in alerts:
                    await self._handle_alert(alert)
                
                # Store metrics
                await self._store_metrics(metrics)
                
                # Log summary every 5 minutes
                if metrics.timestamp.minute % 5 == 0:
                    await self._log_metrics_summary(metrics)
                
                # Wait before next iteration
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)  # Brief pause on error
        
        logger.info("Monitoring loop stopped")
    
    async def _handle_alert(self, alert: Dict[str, Any]):
        """Handle a triggered alert."""
        
        # Log the alert
        log_level = logging.CRITICAL if alert["severity"] == "critical" else logging.WARNING
        logger.log(log_level, f"🚨 LLM Pipeline Alert: {alert['message']}")
        
        # Store alert for later analysis
        await self._store_alert(alert)
        
        # TODO: Integrate with external alerting systems
        # - Send to Slack/Discord
        # - Send email notifications
        # - Trigger PagerDuty
        # - Update monitoring dashboard
    
    async def _store_metrics(self, metrics: PipelineMetrics):
        """Store metrics to persistent storage."""
        try:
            # Ensure directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Append metrics to JSONL file
            with open(self.storage_path, 'a') as f:
                f.write(json.dumps(asdict(metrics), default=str) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")
    
    async def _store_alert(self, alert: Dict[str, Any]):
        """Store alert to persistent storage."""
        try:
            alert_path = self.storage_path.parent / "llm_alerts.jsonl"
            
            with open(alert_path, 'a') as f:
                f.write(json.dumps(alert, default=str) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
    
    async def _log_metrics_summary(self, metrics: PipelineMetrics):
        """Log a summary of current metrics."""
        
        summary = (
            f"📊 LLM Pipeline Metrics Summary:\n"
            f"  • Requests/min: {metrics.requests_per_minute:.1f}\n"
            f"  • Avg response time: {metrics.avg_response_time:.1f}ms\n"
            f"  • Error rate: {metrics.error_rate:.1%}\n"
            f"  • Timeout rate: {metrics.timeout_rate:.1%}\n"
            f"  • Fallback usage: {metrics.fallback_usage_rate:.1%}\n"
            f"  • Circuit breaker trips: {metrics.circuit_breaker_trips}\n"
            f"  • Ollama available: {'✅' if metrics.ollama_availability else '❌'}"
        )
        
        logger.info(summary)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall pipeline health status."""
        
        metrics = self.get_current_metrics()
        
        # Calculate health score (0-100)
        health_score = 100.0
        
        # Deduct points for issues
        if metrics.error_rate > 0.05:  # >5% errors
            health_score -= min(30, metrics.error_rate * 100)
        
        if metrics.avg_response_time > 5000:  # >5s avg response
            health_score -= min(20, (metrics.avg_response_time - 5000) / 1000 * 5)
        
        if metrics.fallback_usage_rate > 0.20:  # >20% fallback usage
            health_score -= min(25, metrics.fallback_usage_rate * 50)
        
        if not metrics.ollama_availability:
            health_score -= 40
        
        health_score = max(0, health_score)
        
        # Determine status
        if health_score >= 90:
            status = "healthy"
        elif health_score >= 70:
            status = "degraded"
        elif health_score >= 40:
            status = "unhealthy"
        else:
            status = "critical"
        
        return {
            "status": status,
            "health_score": health_score,
            "metrics": asdict(metrics),
            "last_updated": datetime.now().isoformat()
        }
    
    def update_external_metrics(self, 
                               queue_utilization: float = None,
                               worker_utilization: float = None,
                               cache_hit_rate: float = None,
                               ollama_availability: bool = None,
                               model_load_time: float = None):
        """Update metrics from external sources."""
        
        # This method allows external components to update metrics
        # that the monitor cannot directly observe
        pass


# Global monitor instance
_pipeline_monitor: Optional[LLMPipelineMonitor] = None


async def get_pipeline_monitor() -> LLMPipelineMonitor:
    """Get or create global pipeline monitor instance."""
    global _pipeline_monitor
    
    if _pipeline_monitor is None:
        _pipeline_monitor = LLMPipelineMonitor()
        await _pipeline_monitor.start_monitoring()
        logger.info("Global LLM pipeline monitor initialized")
    
    return _pipeline_monitor


async def shutdown_pipeline_monitor():
    """Shutdown global pipeline monitor."""
    global _pipeline_monitor
    
    if _pipeline_monitor:
        await _pipeline_monitor.stop_monitoring()
        _pipeline_monitor = None
        logger.info("Global LLM pipeline monitor shutdown")


# Convenience functions for easy integration
async def record_llm_request(response_time_ms: float,
                           success: bool,
                           timeout: bool = False,
                           used_fallback: bool = False,
                           circuit_breaker_open: bool = False):
    """Record LLM request metrics."""
    monitor = await get_pipeline_monitor()
    monitor.record_request(response_time_ms, success, timeout, used_fallback, circuit_breaker_open)


async def get_llm_health_status() -> Dict[str, Any]:
    """Get LLM pipeline health status."""
    monitor = await get_pipeline_monitor()
    return await monitor.get_health_status()


if __name__ == "__main__":
    async def test_monitor():
        """Test the monitoring system."""
        print("🧪 Testing LLM Pipeline Monitor")
        print("=" * 50)
        
        monitor = await get_pipeline_monitor()
        
        # Simulate some requests
        import random
        for i in range(20):
            await record_llm_request(
                response_time_ms=random.uniform(1000, 8000),
                success=random.random() > 0.1,  # 90% success rate
                timeout=random.random() < 0.05,  # 5% timeout rate
                used_fallback=random.random() < 0.15,  # 15% fallback usage
                circuit_breaker_open=random.random() < 0.02  # 2% circuit breaker
            )
        
        # Get metrics
        metrics = monitor.get_current_metrics()
        print(f"Avg response time: {metrics.avg_response_time:.1f}ms")
        print(f"Error rate: {metrics.error_rate:.1%}")
        print(f"Fallback usage: {metrics.fallback_usage_rate:.1%}")
        
        # Get health status
        health = await monitor.get_health_status()
        print(f"Health status: {health['status']} ({health['health_score']:.1f}/100)")
        
        await shutdown_pipeline_monitor()
    
    asyncio.run(test_monitor())