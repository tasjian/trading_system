#!/usr/bin/env python3
"""
Production Monitoring for Ollama Async Bridge

Provides comprehensive monitoring, alerting, and health checks
specifically designed for trading system requirements.
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"  
    CRITICAL = "critical"


@dataclass
class PerformanceAlert:
    """Performance alert structure."""
    timestamp: datetime
    severity: AlertSeverity
    metric: str
    current_value: float
    threshold: float
    message: str
    context: Dict[str, Any]


class OllamaProductionMonitor:
    """Production-grade monitoring for Ollama bridge in trading environment."""
    
    def __init__(self, alert_callback: Optional[callable] = None):
        self.alert_callback = alert_callback or self._default_alert_handler
        self.alerts_history: List[PerformanceAlert] = []
        self.monitoring_active = False
        
        # Performance thresholds for trading system
        self.thresholds = {
            "response_time_p95": 5.0,      # 95th percentile < 5s
            "success_rate": 0.95,          # > 95% success rate
            "queue_utilization": 0.8,      # < 80% queue utilization
            "cache_hit_rate": 0.3,         # > 30% cache hit rate
            "circuit_breaker_failures": 3, # < 3 circuit breaker trips per hour
            "memory_usage_mb": 2000,       # < 2GB memory usage
            "thread_utilization": 0.9      # < 90% thread utilization
        }
        
        # Tracking variables
        self.last_metrics = {}
        self.monitoring_start_time = None
        
    def _default_alert_handler(self, alert: PerformanceAlert):
        """Default alert handler - logs alerts."""
        level_map = {
            AlertSeverity.INFO: logging.info,
            AlertSeverity.WARNING: logging.warning,
            AlertSeverity.CRITICAL: logging.critical
        }
        
        log_func = level_map.get(alert.severity, logging.info)
        log_func(f"🚨 OLLAMA ALERT [{alert.severity.value.upper()}] {alert.metric}: "
                f"{alert.current_value} (threshold: {alert.threshold}) - {alert.message}")
    
    async def start_monitoring(self, check_interval: int = 30):
        """Start continuous monitoring loop."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_start_time = datetime.now()
        logger.info("🔍 Starting Ollama production monitoring...")
        
        while self.monitoring_active:
            try:
                await self.perform_health_check()
                await asyncio.sleep(check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)  # Brief pause on error
        
        logger.info("Ollama production monitoring stopped")
    
    def stop_monitoring(self):
        """Stop monitoring loop."""
        self.monitoring_active = False
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Comprehensive health check with threshold validation."""
        try:
            from tools.ollama_async_bridge import get_ollama_bridge
            
            bridge = await get_ollama_bridge()
            metrics = await bridge.get_metrics()
            health = await bridge.health_check()
            
            # Current timestamp
            check_time = datetime.now()
            
            # Validate against thresholds
            alerts = self._check_thresholds(metrics, health, check_time)
            
            # Process any new alerts
            for alert in alerts:
                self.alerts_history.append(alert)
                self.alert_callback(alert)
            
            # Calculate uptime
            uptime = (check_time - self.monitoring_start_time).total_seconds() / 3600 if self.monitoring_start_time else 0
            
            health_report = {
                "timestamp": check_time.isoformat(),
                "uptime_hours": uptime,
                "system_healthy": health.get("healthy", False),
                "alerts_count": len(alerts),
                "recent_alerts": len([a for a in self.alerts_history if (check_time - a.timestamp).seconds < 3600]),
                "metrics": metrics,
                "raw_health": health,
                "thresholds": self.thresholds
            }
            
            self.last_metrics = metrics
            return health_report
            
        except Exception as e:
            error_alert = PerformanceAlert(
                timestamp=datetime.now(),
                severity=AlertSeverity.CRITICAL,
                metric="monitoring_error",
                current_value=1.0,
                threshold=0.0,
                message=f"Health check failed: {str(e)}",
                context={"error": str(e)}
            )
            self.alerts_history.append(error_alert)
            self.alert_callback(error_alert)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "system_healthy": False,
                "error": str(e)
            }
    
    def _check_thresholds(self, metrics: Dict[str, Any], health: Dict[str, Any], check_time: datetime) -> List[PerformanceAlert]:
        """Check metrics against defined thresholds."""
        alerts = []
        
        # Check response time (from health check)
        response_time = health.get("test_response_time", 0)
        if response_time > self.thresholds["response_time_p95"]:
            alerts.append(PerformanceAlert(
                timestamp=check_time,
                severity=AlertSeverity.WARNING,
                metric="response_time_p95",
                current_value=response_time,
                threshold=self.thresholds["response_time_p95"],
                message=f"Response time ({response_time:.2f}s) exceeds threshold",
                context={"health": health}
            ))
        
        # Check success rate
        success_rate = metrics.get("success_rate_recent", 1.0)
        if success_rate < self.thresholds["success_rate"]:
            severity = AlertSeverity.CRITICAL if success_rate < 0.8 else AlertSeverity.WARNING
            alerts.append(PerformanceAlert(
                timestamp=check_time,
                severity=severity,
                metric="success_rate",
                current_value=success_rate,
                threshold=self.thresholds["success_rate"],
                message=f"Success rate ({success_rate:.1%}) below threshold",
                context={"metrics": metrics}
            ))
        
        # Check queue utilization
        system_metrics = metrics.get("system", {})
        queue_size = system_metrics.get("queue_size", 0)
        queue_capacity = system_metrics.get("queue_capacity", 1)
        queue_util = queue_size / queue_capacity if queue_capacity > 0 else 0
        
        if queue_util > self.thresholds["queue_utilization"]:
            alerts.append(PerformanceAlert(
                timestamp=check_time,
                severity=AlertSeverity.WARNING,
                metric="queue_utilization",
                current_value=queue_util,
                threshold=self.thresholds["queue_utilization"],
                message=f"Queue utilization ({queue_util:.1%}) too high",
                context={"queue_size": queue_size, "capacity": queue_capacity}
            ))
        
        # Check cache hit rate
        cache_hit_rate = metrics.get("cache_hit_rate", 0)
        if cache_hit_rate < self.thresholds["cache_hit_rate"]:
            alerts.append(PerformanceAlert(
                timestamp=check_time,
                severity=AlertSeverity.INFO,
                metric="cache_hit_rate",
                current_value=cache_hit_rate,
                threshold=self.thresholds["cache_hit_rate"],
                message=f"Cache hit rate ({cache_hit_rate:.1%}) below optimal",
                context={"cache_metrics": metrics}
            ))
        
        # Check circuit breaker state
        circuit_breaker = metrics.get("circuit_breaker", {})
        cb_state = circuit_breaker.get("state", "CLOSED")
        if cb_state == "OPEN":
            alerts.append(PerformanceAlert(
                timestamp=check_time,
                severity=AlertSeverity.CRITICAL,
                metric="circuit_breaker_open",
                current_value=1.0,
                threshold=0.0,
                message="Circuit breaker is OPEN - Ollama service degraded",
                context={"circuit_breaker": circuit_breaker}
            ))
        elif cb_state == "HALF_OPEN":
            alerts.append(PerformanceAlert(
                timestamp=check_time,
                severity=AlertSeverity.WARNING,
                metric="circuit_breaker_recovery",
                current_value=0.5,
                threshold=0.0,
                message="Circuit breaker in recovery mode",
                context={"circuit_breaker": circuit_breaker}
            ))
        
        # Check thread utilization
        thread_util = metrics.get("thread_metrics", {}).get("thread_utilization", 0)
        if thread_util > self.thresholds["thread_utilization"]:
            alerts.append(PerformanceAlert(
                timestamp=check_time,
                severity=AlertSeverity.WARNING,
                metric="thread_utilization",
                current_value=thread_util,
                threshold=self.thresholds["thread_utilization"],
                message=f"Thread utilization ({thread_util:.1%}) very high",
                context={"thread_metrics": metrics.get("thread_metrics", {})}
            ))
        
        return alerts
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        now = datetime.now()
        
        # Recent alerts (last hour)
        recent_alerts = [a for a in self.alerts_history 
                        if (now - a.timestamp).seconds < 3600]
        
        # Alert counts by severity
        alert_counts = {
            "critical": sum(1 for a in recent_alerts if a.severity == AlertSeverity.CRITICAL),
            "warning": sum(1 for a in recent_alerts if a.severity == AlertSeverity.WARNING),
            "info": sum(1 for a in recent_alerts if a.severity == AlertSeverity.INFO)
        }
        
        return {
            "monitoring_active": self.monitoring_active,
            "uptime_hours": (now - self.monitoring_start_time).total_seconds() / 3600 if self.monitoring_start_time else 0,
            "total_alerts": len(self.alerts_history),
            "recent_alerts": len(recent_alerts),
            "alert_counts": alert_counts,
            "last_metrics": self.last_metrics,
            "thresholds": self.thresholds
        }
    
    def export_monitoring_data(self, filename: Optional[str] = None) -> str:
        """Export monitoring data to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ollama_monitoring_export_{timestamp}.json"
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "monitoring_summary": self.get_monitoring_summary(),
            "alerts_history": [
                {
                    "timestamp": alert.timestamp.isoformat(),
                    "severity": alert.severity.value,
                    "metric": alert.metric,
                    "current_value": alert.current_value,
                    "threshold": alert.threshold,
                    "message": alert.message,
                    "context": alert.context
                }
                for alert in self.alerts_history
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Monitoring data exported to {filename}")
        return filename


# Global monitor instance
_production_monitor: Optional[OllamaProductionMonitor] = None


async def get_production_monitor() -> OllamaProductionMonitor:
    """Get or create global production monitor."""
    global _production_monitor
    
    if _production_monitor is None:
        _production_monitor = OllamaProductionMonitor()
        logger.info("Ollama production monitor initialized")
    
    return _production_monitor


async def start_production_monitoring(check_interval: int = 30):
    """Start production monitoring in background."""
    monitor = await get_production_monitor()
    
    # Start monitoring in background task
    monitoring_task = asyncio.create_task(monitor.start_monitoring(check_interval))
    logger.info("Ollama production monitoring started in background")
    
    return monitoring_task


if __name__ == "__main__":
    async def test_monitoring():
        """Test the monitoring system."""
        print("🔍 Testing Ollama Production Monitoring")
        print("=" * 50)
        
        monitor = OllamaProductionMonitor()
        
        # Perform health check
        health_report = await monitor.perform_health_check()
        print(f"Health Report: {json.dumps(health_report, indent=2, default=str)}")
        
        # Get summary
        summary = monitor.get_monitoring_summary()
        print(f"Summary: {json.dumps(summary, indent=2, default=str)}")
        
        # Export data
        filename = monitor.export_monitoring_data()
        print(f"Exported monitoring data to: {filename}")
    
    asyncio.run(test_monitoring())